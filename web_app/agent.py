import os
import sys
import json
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Set, Dict, Any, List, AsyncGenerator
from datetime import datetime


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pydantic import Field
from typing import Optional, Set

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models import LiteLlm
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents.callback_context import CallbackContext

from google.genai import types

from config.settings import DATA_DIR, VECTOR_STORE_DIR, OPENAI_API_KEY, OPENAI_MODEL, DEFAULT_USER

# Define the active user for the lifetime of this server process
ACTIVE_USER = os.environ.get("CURRENT_USER", DEFAULT_USER)

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store_utils import VectorStoreUtilsMixin
from src.retrieval.hierarchical_retriever import HierarchicalRetriever

# Import the new Intent Router
from web_app.tools import intent_router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMonitorPlugin(BasePlugin):
    """
    Enhanced ADK Plugin to monitor session events and track LLM usage metadata.
    Logs:
    - Token usage (Prompt, Completion, Total) for different agents.
    - Conversation context sent to the Planner agent (system instructions filtered).
    - Session-level compaction diagnostics.
    """
    def __init__(self):
        super().__init__(name="memory_monitor")

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        """
        Intercept the model request to log the context being seen by specific agents.
        """
        # Fix: CallbackContext does not expose .agent directly, use .agent_name
        agent_name = getattr(callback_context, "agent_name", "Internal (System/Compactor)")
        
        # We only care about printing dynamic context for PlannerAgent or relevant agents
        if agent_name in ["PlannerAgent", "AnswerSynthesisAgent", "Internal (System/Compactor)"]:
            print(f"\n" + "🔹" * 10)
            print(f"📡 [LLM CALL] Starting request for: {agent_name}")
            
            # Print non-system contents (conversation history)
            for i, content in enumerate(llm_request.contents):
                if content.role != "system":
                    # FIX: Iterate and join ALL text parts instead of just Parts[0]
                    parts_texts = []
                    if content.parts:
                        for p in content.parts:
                            if p.text:
                                parts_texts.append(p.text)
                    
                    full_text = " ".join(parts_texts)
                    
                    # For long history, show preview
                    preview = full_text[:500] + "..." if len(full_text) > 500 else full_text
                    print(f"  📜 Context Item [{i}] ({content.role}): {preview}")
                else:
                    # Just mention system instruction exists without printing fully
                    print(f"  📝 [System Instruction] (Filtered)")
            print("🔹" * 10)
        return None

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        """
        Intercept the model response to extract and log usage metadata (tokens).
        """
        # Fix: CallbackContext does not expose .agent directly, use .agent_name
        agent_name = getattr(callback_context, "agent_name", "Internal (System/Compactor)")
        
        usage = llm_response.usage_metadata
        if usage:
            print("\n" + "💰" * 10)
            print(f"📊 [Token Usage] Agent: {agent_name}")
            print(f"   📥 Prompt Tokens: {usage.prompt_token_count}")
            print(f"   📤 Output Tokens: {usage.candidates_token_count}")
            print(f"   ⚙️  Total Tokens: {usage.total_token_count}")
            print("💰" * 10 + "\n")
        return None

    async def after_run_callback(self, *, invocation_context: "InvocationContext") -> None:
        """
        ADK Plugin to monitor session events and detect compaction behavior.
        """
        session = invocation_context.session
        events = session.events
        
        # Gather all unique invocation IDs (ignoring metadata/compaction events)
        invocation_ids = []
        seen_inv_ids = set()
        
        for e in events:
            if e.invocation_id and e.invocation_id not in seen_inv_ids:
                if not (e.actions and e.actions.compaction):
                    invocation_ids.append(e.invocation_id)
                    seen_inv_ids.add(e.invocation_id)
        
        turn_count = len(invocation_ids)
        
        print("\n" + "🚀" * 15)
        print(f"🔍 [MemoryMonitor] Session Diagnostics")
        print(f"📂 Session ID: {session.id[:8]}...")
        print(f"📈 Total Events: {len(events)}")
        print(f"🔄 Completed Turns (Invocations): {turn_count}")
        
        # Look for compaction summary events using actions.compaction (Reliable)
        summaries = [e for e in events if e.actions and e.actions.compaction]
        
        if summaries:
            print(f"✅ COMPACTION DETECTED! Found {len(summaries)} summary event(s).")
            latest_summary = summaries[-1]
            comp_data = latest_summary.actions.compaction
            
            content = comp_data.compacted_content
            if content and content.parts:
                # JOIN ALL PARTS and print FULL text (no truncation)
                full_summary = " ".join([p.text for p in content.parts if p.text])
                print(f"📝 Full Summary Content:\n{full_summary}")
                
                # --- NEW: Mem0 Integration (Background Task) ---
                user_id = invocation_context.session.state.get("CURRENT_USER", "default_user")
                from src.memory.mem0_service import mem0_service
                
                # Fire-and-forget: run mem0.add in the background without blocking the response
                logger.info(f"[Mem0] Dispatching background task to extract memory for {user_id}...")
                asyncio.create_task(
                    asyncio.to_thread(
                        mem0_service.add_session_memories,
                        user_id=user_id,
                        summary_text=full_summary
                    )
                )
                
            print(f"⏰ Compacted Range: {comp_data.start_timestamp} to {comp_data.end_timestamp}")
        else:
            threshold = 4 
            if turn_count >= threshold:
                print(f"⚠️  WARNING: Turn count ({turn_count}) >= threshold ({threshold}), but NO compaction found!")
            else:
                print(f"⏳ Waiting for threshold... (Current: {turn_count}/{threshold})")
        
        print("🚀" * 15 + "\n")


# Clean useless environment variable SSL_CERT_FILE
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]
        print(f"Warning: Removed invalid SSL_CERT_FILE: {cert_path}")


def format_context_for_answer(chunks: List[Dict[str, Any]], max_chunks: int = 15) -> str:
    """
    Helper function to format chunks for answer generation.
    Format:
    [data012_summary_1] (Date: 2025-11-29)
    <chunk text>

    [data013_summary_3] (Date: 2025-11-30T20:21:00+08:00)
    <chunk text>
    """
    if not chunks:
        return "No relevant context found."

    lines: List[str] = []

    for chunk in chunks[:max_chunks]:
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "")
        meta = chunk.get("metadata") or {}
        dt = meta.get("datetime", "N/A")

        lines.append(f"[{chunk_id}] (Date: {dt})")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


class FullRAGSystemAgent(BaseAgent):
    """
    Custom RAG System Agent - orchestrates planner, query rewrite, and retrieval.
    
    Flow:
    1. Initialize RAG components in __init__ (not as tool)
    2. Run planner agent to decide: need_rewrite, need_rag
    3. If need_rewrite: call rewrite_query_async
    4. If need_rag: call retriever (with rewritten query if available)
    5. Store results in state for AnswerAgent
    """
    
    # Pydantic field declarations
    initialized: bool = Field(default=False)
    retriever: Optional[HierarchicalRetriever] = Field(default=None)
    total_meetings: int = Field(default=0)
    all_meeting_ids: Set[str] = Field(default_factory=set)
    intent_router: Any = Field(default=None)
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "FullRAGSystemAgent"):
        """
        Initialize the FullRAGSystemAgent.
        - Initializes RAG components immediately (not as tool)
        - Sets up IntentRouter reference
        """
        # Register IntentRouter as sub-agent so ADK manages its lifecycle
        super().__init__(name=name, sub_agents=[intent_router])
        
        self.intent_router = intent_router
        
        # Lock for thread-safe incremental sync (hot reload)
        self._sync_lock = threading.Lock()
        
        # MeetingWatcher instance (started later via start_watcher())
        self._watcher = None
        
        # THEN initialize RAG components
        logger.info("=" * 80)
        logger.info("FullRAGSystemAgent: Starting initialization...")
        self._initialize_components()
        logger.info("FullRAGSystemAgent: Initialization complete!")
        logger.info("=" * 80)
    
    def _match_user_folder(self, user_id: str) -> str:
        """
        Match a given user_id to a physical folder name in DATA_DIR.
        Supports case-insensitive matching. Falls back to ACTIVE_USER if no match.
        """
        # "user" is the default ID in ADK web UI if not specified
        if not user_id or user_id.lower() == "user":
            return ACTIVE_USER
            
        data_path = Path(DATA_DIR)
        if not data_path.exists():
            return ACTIVE_USER
            
        # Case-insensitive search for the user folder
        try:
            for folder in data_path.iterdir():
                if folder.is_dir() and folder.name.lower() == user_id.lower():
                    return folder.name
        except Exception as e:
            logger.warning(f"[{self.name}] Error scanning DATA_DIR for user match: {e}")
            
        return user_id # Fallback to raw ID if folder not found

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Main execution logic for the RAG agent.
        1. Extract query
        2. Resolve user dynamically (ADK Native)
        3. Delegate entirely to the ReAct IntentRouter
        """
        # Step 0: Extract user query and resolve user identity
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = ctx.user_content.parts[0].text or ""
            
        raw_user_id = ctx.user_id
        current_user = self._match_user_folder(raw_user_id)
        
        # Store in session state for other components
        ctx.session.state["CURRENT_USER"] = current_user
        ctx.session.state["user_query"] = user_query
        
        logger.info(f"[{self.name}] User ID: '{raw_user_id}' -> Matched Folder: '{current_user}'")
        logger.info(f"[{self.name}] Delegating query to IntentRouter: {user_query[:100]}...")
        
        if not user_query:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text="Error: No user query provided.")]
                )
            )
            return

        # Delegate the entire turn to the new IntentRouter
        async for event in self.intent_router.run_async(ctx):
            yield event
    







    def _initialize_components(self):
        """
        Initialize all RAG components: data loader, vector store, retriever.
        This is called once in __init__ (not lazily).
        """
        if self.initialized:
            return
        
        logger.info("Initializing RAG components...")
        
        data_dir = DATA_DIR
        if data_dir is None:
            raise ValueError('data_dir is none.')
        
        include_chunk_level = False  # ADK UI not include chunk level
        
        # Step 1: Load meeting data
        logger.info("\n[Step 1] Loading meeting data...")
        loader = DataLoader(data_dir)
        all_meetings = loader.load_all_meetings()
        logger.info(f"✓ Loaded {len(all_meetings)} meetings from data directory")

        if not all_meetings:
            raise ValueError("No meetings found. Please check your data directory.")

        # Step 2: Check for existing vector store
        logger.info("\n[Step 2] Checking for existing vector store...")
        vector_store = None
        processed_meeting_ids: Set[str] = set()

        if VECTOR_STORE_DIR.exists():
            try:
                logger.info(f"✓ Found existing vector store at {VECTOR_STORE_DIR}")
                logger.info("   Loading existing vector store...")
                vector_store = VectorStoreUtilsMixin.load(VECTOR_STORE_DIR)
                processed_meeting_ids = vector_store.get_processed_meeting_ids()
                logger.info(f"✓ Loaded existing vector store with {len(processed_meeting_ids)} processed meetings")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                logger.info("   Will create new vector store...")
                vector_store = None
                processed_meeting_ids = set()
        else:
            logger.info("No existing vector store found. Will create new one.")

        # Step 3: Filter to new meetings only
        new_meetings = []
        for meeting in all_meetings:
            if meeting.meeting_id not in processed_meeting_ids:
                new_meetings.append(meeting)

        logger.info(f"\n[Step 3] Found {len(new_meetings)} new meetings to process")

        # Step 4: Process new meetings
        if new_meetings:
            logger.info("\n[Step 4] Processing new meetings...")

            # Chunk new meetings
            logger.info("   Chunking new meetings...")
            chunker = HierarchicalChunker()
            new_chunks = chunker.chunk_all_levels(new_meetings, include_chunk_level=include_chunk_level)

            for level, chunks in new_chunks.items():
                logger.info(f"     {level:10s}: {len(chunks):4d} chunks")

            # Initialize vector store if needed
            if vector_store is None:
                logger.info("   Creating new vector store...")
                vector_store = VectorStoreUtilsMixin()
            else:
                logger.info("   Using existing vector store...")

            # Add new chunks to vector store
            logger.info("   Adding new chunks to vector store...")
            for level in ['metadata', 'summary', 'meeting']:
                chunks = new_chunks.get(level, [])
                if chunks:
                    vector_store.add_chunks(
                        chunks=chunks,
                        level=level,
                        generate_embedding=True
                    )
                    logger.info(f"     ✓ Added {len(chunks)} chunks to {level} level")

            # Save updated vector store
            logger.info("   Saving updated vector store...")
            vector_store.save(VECTOR_STORE_DIR)
            logger.info(f"✓ Vector store saved to {VECTOR_STORE_DIR}")

        else:
            logger.info("\n[Step 4] No new meetings to process ✓")

        # Step 5: Registry Check
        # We no longer pin a single user's IDs at startup to support multi-user.
        # But we can verify if the default user exists as a sanity check.
        logger.info("\n[Step 5] Verifying user metadata...")
        from sub_agents.metadata_manager import get_meetings_metadata
        mdf = get_meetings_metadata(ACTIVE_USER)
        visible_ids_count = len(mdf.records) if mdf.records else 0
        logger.info(f"✓ Default user '{ACTIVE_USER}' has {visible_ids_count} meetings ready.")
        
        # Step 6: Create retriever
        logger.info("\n[Step 6] Creating hierarchical retriever...")
        if vector_store is None:
            raise RuntimeError("vector initialization failed")
        self.retriever = HierarchicalRetriever(vector_store)
        logger.info("✓ Hierarchical retriever created")

        # Mark as initialized
        self.initialized = True
        
        logger.info("\n" + "=" * 80)
        logger.info("RAG System Initialized Successfully!")
        logger.info(f" Total meetings indexed across all users: {len(all_meetings)}")
        logger.info(f" Default user context: {ACTIVE_USER}")
        logger.info(f" Processed meetings in VectorStore: {len(processed_meeting_ids)}")
        logger.info(f"   New meetings processed: {len(new_meetings)}")
        logger.info("=" * 80)

    # ================================================================
    #  Hot Reload: Incremental Sync & Watcher
    # ================================================================

    def _incremental_sync(self, new_con_dirs: List[Path]) -> None:
        """
        Incrementally index new meetings detected by MeetingWatcher.

        This method is called from the watcher's background thread.
        It is protected by _sync_lock to prevent concurrent writes
        to the vector store.  Read operations (user queries) are NOT
        blocked — they simply see the old data until the sync completes
        (eventual consistency).

        Steps:
            1. Load each new meeting via DataLoader
            2. Chunk them with HierarchicalChunker
            3. Add chunks to the existing vector store
            4. Persist vector store to disk
            5. Reload the pandas MeetingsDataFrame singleton
            6. Update self.all_meeting_ids

        Args:
            new_con_dirs: List of Paths to newly detected con* directories.
        """
        with self._sync_lock:
            logger.info("\n" + "🔄" * 20)
            logger.info(
                f"[IncrementalSync] Processing {len(new_con_dirs)} new meeting dir(s): "
                f"{[d.name for d in new_con_dirs]}"
            )

            if self.retriever is None or self.retriever.vector_store is None:
                logger.error("[IncrementalSync] Retriever / VectorStore not initialized, aborting.")
                return

            vector_store = self.retriever.vector_store
            include_chunk_level = False  # Consistent with _initialize_components

            loader = DataLoader(DATA_DIR)
            new_meetings = []
            affected_users = set()

            # Step 1: Load meetings from the new directories
            for con_dir in new_con_dirs:
                try:
                    # Get user name from parent directory
                    user_name = con_dir.parent.name
                    affected_users.add(user_name)
                    
                    meeting = loader._load_meeting(con_dir)
                    new_meetings.append(meeting)
                    logger.info(
                        f"[IncrementalSync]   ✓ Loaded: {meeting.meeting_id} (User: {user_name})"
                    )
                except Exception as e:
                    logger.error(
                        f"[IncrementalSync]   ✗ Failed to load {con_dir.name}: {e}",
                        exc_info=True,
                    )

            if not new_meetings:
                logger.warning("[IncrementalSync] No meetings loaded successfully, aborting.")
                return

            # Step 2: Chunk new meetings
            logger.info("[IncrementalSync] Chunking new meetings...")
            chunker = HierarchicalChunker()
            new_chunks = chunker.chunk_all_levels(new_meetings, include_chunk_level=include_chunk_level)

            for level, chunks in new_chunks.items():
                logger.info(f"[IncrementalSync]   {level:10s}: {len(chunks):4d} chunks")

            # Step 3: Add chunks to vector store
            logger.info("[IncrementalSync] Adding chunks to vector store...")
            for level in ['metadata', 'summary', 'meeting']:
                chunks = new_chunks.get(level, [])
                if chunks:
                    vector_store.add_chunks(
                        chunks=chunks,
                        level=level,
                        generate_embedding=True,
                    )
                    logger.info(
                        f"[IncrementalSync]   ✓ Added {len(chunks)} chunks to {level} level"
                    )

            # Step 4: Persist vector store to disk
            logger.info("[IncrementalSync] Saving vector store...")
            vector_store.save(VECTOR_STORE_DIR)

            # Step 5: Reload MetadataManager for affected users
            # This ensures the new meeting is visible in the participant list and filters
            from sub_agents.metadata_manager import reload_meetings_metadata
            for user_name in affected_users:
                logger.info(f"[IncrementalSync] Reloading metadata for user: {user_name}")
                reload_meetings_metadata(user_name)

            # Step 6: Update meeting ID registry for the active user
            # (In a real multi-user environment, this might need more logic)
            from sub_agents.metadata_manager import get_meetings_metadata
            mdf = get_meetings_metadata(ACTIVE_USER)
            visible_ids = set(r["meeting_id"] for r in mdf.records)
            self.all_meeting_ids = visible_ids
            self.total_meetings = len(self.all_meeting_ids)

            logger.info("\n" + "=" * 60)
            logger.info("[IncrementalSync] ✅ Incremental sync complete!")
            logger.info(f"  New meetings indexed: {[m.meeting_id for m in new_meetings]}")
            logger.info(f"  Total meetings now for {ACTIVE_USER}: {self.total_meetings}")
            logger.info("=" * 60 + "\n")

    def _on_meeting_deleted(self, rel_path: Optional[str], person_name: Optional[str]) -> None:
        """
        Handle meeting deletion by reloading metadata.
        
        Args:
            rel_path: Relative path of the deleted con* directory (e.g. "Elon/con1")
            person_name: Name of the person if a whole person directory was deleted
        """
        with self._sync_lock:
            logger.info("\n" + "🗑️" * 20)
            from sub_agents.metadata_manager import reload_meetings_metadata
            
            if person_name:
                logger.info(f"[DeletionSync] Person directory deleted: {person_name}")
                reload_meetings_metadata(person_name)
            elif rel_path:
                logger.info(f"[DeletionSync] Meeting directory deleted: {rel_path}")
                # Extract user name from the first part of the relative path
                user_name = rel_path.split("/")[0]
                reload_meetings_metadata(user_name)
            
            # Refresh active user registry
            from sub_agents.metadata_manager import get_meetings_metadata
            mdf = get_meetings_metadata(ACTIVE_USER)
            self.all_meeting_ids = set(r["meeting_id"] for r in mdf.records)
            self.total_meetings = len(self.all_meeting_ids)
            
            logger.info("[DeletionSync] ✅ Metadata reloaded after deletion")
            logger.info("🗑️" * 20 + "\n")

    def start_watcher(self) -> None:
        """
        Start the MeetingWatcher to monitor data directory for new meetings.

        Should be called AFTER _initialize_components() has completed.
        The watcher runs in a daemon thread and will not block shutdown.
        """
        from src.watcher.meeting_watcher import MeetingWatcher

        # Derive processed directory paths.
        # Use recursive glob to find all con* directories in user folders.
        data_dir = Path(DATA_DIR)
        processed_rel_paths = set()
        for con_dir in data_dir.rglob("con*"):
            if con_dir.is_dir():
                try:
                    rel_path = str(con_dir.relative_to(data_dir)).replace("\\", "/")
                    processed_rel_paths.add(rel_path)
                except ValueError:
                    continue

        self._watcher = MeetingWatcher(
            data_dir=data_dir,
            on_new_meetings_callback=self._incremental_sync,
            on_deleted_meetings_callback=self._on_meeting_deleted,
            processed_rel_paths=processed_rel_paths,
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        """Stop the MeetingWatcher if running."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None


# Create the root agent for Google ADK
root_agent = FullRAGSystemAgent(name="FullRAGSystemAgent")

# Pre-load all lazy-initialized NLP components (spaCy, Presidio, pandas, embeddings)
# so the first user query does not incur loading overhead.
from sub_agents.query_rewriter_agent import warmup_all as _warmup_nlp
_warmup_nlp()

# Wrap into an App with Context Compaction for multi-turn history
compaction_llm = LiteLlm(
    model=OPENAI_MODEL or "gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

compaction_config = EventsCompactionConfig(
    compaction_interval=3,
    overlap_size=1,
    summarizer=LlmEventSummarizer(llm=compaction_llm)
)

app = App(
    name="web_app",
    root_agent=root_agent,
    plugins=[MemoryMonitorPlugin()],
    events_compaction_config=compaction_config
)