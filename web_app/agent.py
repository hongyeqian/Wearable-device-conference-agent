import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Set, Dict, Any, List, AsyncGenerator
from datetime import datetime


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pydantic import Field
from typing import Optional, Set

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models import LiteLlm
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents.callback_context import CallbackContext

from google.genai import types

from config.settings import DATA_DIR, VECTOR_STORE_DIR, CURRENT_USER, OPENAI_API_KEY, OPENAI_MODEL

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store_utils import VectorStoreUtilsMixin
from src.retrieval.hierarchical_retriever import HierarchicalRetriever

from sub_agents.query_rewriter_agent import rewrite_query_async, get_last_pandas_query_result
from sub_agents.planner_agent import planner_agent, Plan


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
    answer_synthesis_agent: Any = Field(default=None)  # For integrated answer generation
    
    # Model config for Pydantic
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "FullRAGSystemAgent"):
        """
        Initialize the FullRAGSystemAgent.
        - Initializes RAG components immediately (not as tool)
        - Sets up planner agent reference
        """
        # Call parent __init__ FIRST to initialize Pydantic fields
        super().__init__(name=name, sub_agents=[planner_agent])
        
        # Initialize answer synthesis agent (for integrated answer generation)
        from sub_agents.answer_agent import answer_synthesis_agent as answer_agent
        self.answer_synthesis_agent = answer_agent
        
        # THEN initialize RAG components
        logger.info("=" * 80)
        logger.info("FullRAGSystemAgent: Starting initialization...")
        self._initialize_components()
        logger.info("FullRAGSystemAgent: Initialization complete!")
        logger.info("=" * 80)
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Custom orchestration logic:
        1. Get user query from state
        2. Run planner agent to decide need_rewrite / need_rag
        3. If need_rewrite: run query rewrite pipeline
        4. If need_rag: run retrieval
        5. Yield final event with results in state
        """
        logger.info(f"[{self.name}] Starting RAG system orchestration...")
        
        # Step 0: Extract user query from user_content and store in session state
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = ctx.user_content.parts[0].text or ""
        
        # Store in session state for planner_agent to access via {user_query}
        ctx.session.state["user_query"] = user_query
        logger.info(f"[{self.name}] User query extracted and stored: {user_query[:100]}...")
        
        # Step 1: Get user query from state (set by the runner)
        if not user_query:
            logger.warning(f"[{self.name}] No user_query found in state")
            yield Event(
                author=self.name,
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text="Error: No user query provided.")]
                )
            )
            return
        
        logger.info(f"[{self.name}] Processing query: {user_query}")
        
        # Step 2: Run planner agent to decide need_rewrite / need_rag
        logger.info(f"[{self.name}] Running planner agent...")
        plan = await self._run_planner(ctx, user_query)
        
        if not plan:
            logger.error(f"[{self.name}] Planner failed to produce a plan")
            yield Event(
                author=self.name,
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text="Error: Failed to analyze query.")]
                )
            )
            return
        
        logger.info(f"[{self.name}] Plan: need_rewrite={plan.need_rewrite}, need_rag={plan.need_rag}, reason={plan.reason}")
        
        # Step 3: Process rewrite if needed
        rewrite_result = None
        if plan.need_rewrite:
            logger.info(f"[{self.name}] Running query rewrite...")
            query_to_rewrite = plan.resolved_query if plan.resolved_query else user_query
            rewrite_result = await self._run_query_rewrite(query_to_rewrite)
            if rewrite_result:
                ctx.session.state["rewrite_result"] = {
                    "rewritten_query": rewrite_result.rewritten_query,
                    "relevant_meeting_ids": rewrite_result.relevant_meeting_ids
                }
                logger.info(f"[{self.name}] Rewrite result: {rewrite_result.rewritten_query}")
            else:
                logger.warning(f"[{self.name}] Query rewrite failed, using original query")
        
        # Step 4: Run RAG retrieval if needed
        if plan.need_rag:
            logger.info(f"[{self.name}] Running RAG retrieval...")
            # Determine which query to use for retrieval
            if rewrite_result:
                retrieval_query = rewrite_result.rewritten_query
                relevant_meeting_ids = rewrite_result.relevant_meeting_ids
            else:
                retrieval_query = plan.resolved_query if plan.resolved_query else user_query
                relevant_meeting_ids = []
            
            retrieval_chunks = await self._run_retrieval(
                ctx=ctx,
                query=retrieval_query,
                relevant_meeting_ids=relevant_meeting_ids
            )
            
            ctx.session.state["retrieval_chunks"] = retrieval_chunks
            logger.info(f"[{self.name}] Retrieved {len(retrieval_chunks)} chunks")
        else:
            # When RAG is not needed, still set empty chunks for answer agent
            ctx.session.state["retrieval_chunks"] = []
            logger.info(f"[{self.name}] No RAG needed, setting empty chunks")
        
        # Step 5: Run answer synthesis agent to generate final answer
        logger.info(f"[{self.name}] Running answer synthesis agent...")
        async for event in self.answer_synthesis_agent.run_async(ctx):
            # Pass through events from answer agent
            yield event
        
        # Final event is yielded by answer_agent, no need for additional yield
    
    async def _run_planner(self, ctx: InvocationContext, user_query: str) -> Plan:
        """
        Run the planner agent to decide need_rewrite and need_rag.
        """
        # Store user_query in state for planner to access
        ctx.session.state["user_query"] = user_query
        
        # Run the planner agent and get events
        plan = None
        async for event in planner_agent.run_async(ctx):
            # Check if this is the final response
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text
                logger.info(f"[{self.name}] Planner final response: {response_text}")
                
                # Parse JSON from the response text
                try:
                    if not response_text:
                        logger.error(f"[{self.name}] Planner response text is empty")
                        continue
                    plan_dict = json.loads(response_text)
                    plan = Plan(
                        need_rewrite=plan_dict.get("need_rewrite", False),
                        need_rag=plan_dict.get("need_rag", True),
                        resolved_query=plan_dict.get("resolved_query", user_query),
                        reason=plan_dict.get("reason", "")
                    )
                    logger.info(f"[{self.name}] Parsed plan from response: {plan}")
                    return plan
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] Failed to parse plan JSON: {response_text}, error: {e}")
        
        # Fallback: default plan
        logger.warning(f"[{self.name}] Planner didn't return valid plan, using default")
        return Plan(need_rewrite=False, need_rag=True, reason="Default: using RAG")
    
    async def _run_query_rewrite(self, query: str):
        """
        Run the three-stage query rewrite pipeline.
        """
        try:
            result = await rewrite_query_async(query)
            
            # Fallback: extract meeting_ids from pandas_query result if missing
            if result.relevant_meeting_ids:
                pass  # Already has meeting_ids
            elif result.rewritten_query != query:
                # Try to extract from pandas_query result
                last_query_result = get_last_pandas_query_result()
                if last_query_result and "meeting_ids" in last_query_result:
                    extracted_ids = last_query_result.get("meeting_ids", [])
                    if extracted_ids:
                        logger.info(f"[{self.name}] Extracted meeting_ids from pandas_query: {extracted_ids}")
                        result.relevant_meeting_ids = extracted_ids
            
            logger.info(f"[{self.name}] Query rewrite complete: {query} -> {result.rewritten_query}")
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Query rewrite error: {e}")
            return None
    
    async def _run_retrieval(
        self,
        ctx: InvocationContext,
        query: str,
        relevant_meeting_ids: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run the retrieval process.
        """
        try:
            # Ensure retriever is initialized
            if self.retriever is None:
                logger.error(f"[{self.name}] Retriever not initialized!")
                return []
            
            # Use meeting_ids filter if available
            meeting_ids_filter = set(relevant_meeting_ids) if relevant_meeting_ids else None
            
            # Perform search
            if meeting_ids_filter:
                results = self.retriever.vector_store.search_with_meeting_ids_filter(
                    query_text=query,
                    level='summary',
                    meeting_ids=meeting_ids_filter,
                    top_k=top_k * 2
                )
            else:
                results = self.retriever.vector_store.search_with_meeting_ids_filter(
                    query_text=query,
                    level='summary',
                    meeting_ids=None,
                    top_k=top_k * 2
                )
            
            # Extract hits from results
            if isinstance(results, dict):
                hits = results.get("hits")
                if isinstance(hits, list):
                    results = hits
            
            # Sort and limit results
            chunks = sorted(results, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k]
            
            # Format chunks
            clean_chunks = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text"),
                    "metadata": {"datetime": c.get("metadata", {}).get("datetime", "N/A")},
                }
                for c in chunks
            ]
            
            return clean_chunks
            
        except Exception as e:
            logger.error(f"[{self.name}] Retrieval error: {e}")
            return []
    
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

        # Step 5: Register meeting IDs
        logger.info("\n[Step 5] Registering meeting IDs...")
        self.all_meeting_ids = {m.meeting_id for m in all_meetings}
        logger.info(f"✓ Registered {len(self.all_meeting_ids)} meeting IDs")
        
        # Step 6: Create retriever
        logger.info("\n[Step 6] Creating hierarchical retriever...")
        if vector_store is None:
            raise RuntimeError("vector initialization failed")
        self.retriever = HierarchicalRetriever(vector_store)
        logger.info("✓ Hierarchical retriever created")

        # Mark as initialized
        self.initialized = True
        self.total_meetings = len(all_meetings)
        
        logger.info("\n" + "=" * 80)
        logger.info("RAG System Initialized Successfully!")
        logger.info(f" Total meetings available: {len(all_meetings)}")
        logger.info(f" Processed meetings: {len(processed_meeting_ids)}")
        logger.info(f"   New meetings processed: {len(new_meetings)}")
        logger.info("=" * 80)


# Create the root agent for Google ADK
# FullRAGSystemAgent now includes answer_synthesis_agent internally
root_agent = FullRAGSystemAgent(name="FullRAGSystemAgent")

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