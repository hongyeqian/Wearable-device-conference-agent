import os
import sys
from pathlib import Path
from typing import Optional, Set, Dict, Any, List
from datetime import datetime


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pydantic import Field
from typing import Optional, Set

from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext

from config.settings import DATA_DIR, VECTOR_STORE_DIR, CURRENT_USER

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store_utils import VectorStoreUtilsMixin
from src.retrieval.hierarchical_retriever import HierarchicalRetriever

from sub_agents.planner_agent import planner_agent
from sub_agents.query_rewriter_agent import rewrite_query_async, get_last_pandas_query_result
from sub_agents.answer_agent import answer_synthesis_agent


def format_context_for_answer(chunks: List[Dict[str, Any]], max_chunks: int = 15) -> str:
    """
    This is a helper function for our agents,change the format of the time
    structure is as follpwing：
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




# clean useless environment variable SSL_CERT_FILE
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]
        print(f"Warning: Removed invalid SSL_CERT_FILE: {cert_path}")

# return models for tool function

class FullRAGSystemAgent(LlmAgent):
    
    initialized: bool = Field(default=False)
    retriever: Optional[HierarchicalRetriever] = Field(default=None)
    total_meetings: int = Field(default= 0)
    all_meeting_ids: Set[str] = Field(default_factory=set)
    
    

    async def rag_retrieve_func(self, query: str, tool_context: ToolContext, top_k: int = 10) -> Dict[str, Any]:
        try:
            tool_context.state["retrieval_chunks"] = []
            plan = tool_context.state.get("plan", {})
            if not plan.get("need_rag", False):
                tool_context.state["retrieval_chunks"] = []
                return {"skipped": True, "chunks": [], "total_chunks": 0}

            # for protection
            if not tool_context.state.get("initialized", False):
                print("Warning: System not initialized in setup_state_func, performing emergency initialization...")
                self._initialize_components()
                tool_context.state["initialized"] = True

            if not self.initialized:
                raise RuntimeError("System initialization failed")

            # Get original query from state
            user_query = tool_context.state.get("user_query", query)

            # Get processed query and meeting IDs from Query Rewriter
            rewrite_result = tool_context.state.get("rewrite_result", {})
            processed_query = rewrite_result.get("rewritten_query", user_query)
            relevant_meeting_ids = rewrite_result.get("relevant_meeting_ids", [])
            
            # Fallback: If rewrite_result has no meeting_ids but query was rewritten,
            # try to extract from pandas_query result (in case LLM forgot to include them)
            if not relevant_meeting_ids and processed_query != user_query:
                last_query_result = get_last_pandas_query_result()
                if last_query_result and "meeting_ids" in last_query_result:
                    extracted_ids = last_query_result.get("meeting_ids", [])
                    if extracted_ids:
                        print(f"Extracted meeting_ids from pandas_query result: {extracted_ids}")
                        relevant_meeting_ids = extracted_ids
            
            # If no filter result or empty list, search all meetings
            if not relevant_meeting_ids:
                relevant_meeting_ids = list(self.all_meeting_ids) if hasattr(self, 'all_meeting_ids') else []
            
            print(f"Filter result: {relevant_meeting_ids}")
            
            # Ensure retriever is initialized
            if self.retriever is None:
                raise RuntimeError("Retriever not initialized")
            
            # Perform search with optional meeting_ids filter
            # Note: We now use the processed_query directly without rewriter output
            if relevant_meeting_ids:
                # Use search_with_meeting_ids_filter for filtered search
                results = self.retriever.vector_store.search_with_meeting_ids_filter(
                    query_text=processed_query,
                    level='summary',
                    meeting_ids=set(relevant_meeting_ids),
                    top_k=top_k * 2  # Get more results since we're filtering
                )
                
                # Merge results - keep unique chunks
                results = results
            else:
                # No relevant meeting IDs - search all documents
                results = self.retriever.vector_store.search_with_meeting_ids_filter(
                    query_text=processed_query,
                    level='summary',
                    meeting_ids=None,  # Search all documents
                    top_k=top_k * 2
                )

            # check if the form suitable for returning
            if isinstance(results, dict):
                hits = results.get("hits")
                if isinstance(hits, list):
                    results = hits
                    
                

            chunks = sorted(results, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k]
            clean_chunks = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text"),
                    "metadata": {"datetime": c.get("metadata", {}).get("datetime", "N/A")},
                }
                for c in chunks
            ]
            
            tool_context.state["retrieval_chunks"]= clean_chunks
            return { 
                "skipped": False,
                "chunks": clean_chunks,
                "total_chunks": len(chunks),
            }

        except Exception as e:
            return {"skipped": True, "chunks": [], "total_chunks": 0, "error": str(e)}

    
    def setup_state_func(self, tool_context: ToolContext) -> Dict[str, Any]:
        """Setup required state variables for all tools - now includes full system initialization"""
        
        # check for initialize
        if tool_context.state.get("initialized", False):
            return {
                "status": "already_initialized",
                "message": "System already initialized"
            }

        # Initial setup for RAG - this will be refactored later
        # as we cannot mix initialization logic with QnA logic
        self._initialize_components()

        # set our initial memory
        tool_context.state["initialized"] = True
        tool_context.state["index_status"] = {
            "total_meetings": self.total_meetings,
            "vector_store_ready": self.retriever is not None
        }
        tool_context.state["now_str"] = datetime.now().date().isoformat()
        tool_context.state["current_user_name"] = CURRENT_USER
        tool_context.state["system_message"] = "✅ System initialized: meetings loaded, index ready."
        tool_context.state["retrieval_chunks"] = []
        return {
            "status": "state_initialized",
            "message": tool_context.state["system_message"]
        }

    async def rewrite_query_func(self, query: str, tool_context: ToolContext) -> Dict[str, Any]:
        """
        Query Rewrite function - calls the three-stage pipeline asynchronously.
        
        This function integrates with the new rewrite_query_async() which:
        - Stage 1: check_ambiguity (synchronous)
        - Stage 2: resolve_entities (synchronous)
        - Stage 3: LLM ReAct via await (async)
        
        Returns:
            Dict with rewritten_query and relevant_meeting_ids
        """
        try:
            # Call the async three-stage pipeline
            result = await rewrite_query_async(query)
            
            # Extract results
            rewritten_query = result.rewritten_query
            relevant_meeting_ids = result.relevant_meeting_ids
            
            # Fallback: If rewrite_result has no meeting_ids but query was rewritten,
            # try to extract from pandas_query result (in case LLM forgot to include them)
            if not relevant_meeting_ids and rewritten_query != query:
                last_query_result = get_last_pandas_query_result()
                if last_query_result and "meeting_ids" in last_query_result:
                    extracted_ids = last_query_result.get("meeting_ids", [])
                    if extracted_ids:
                        print(f"Extracted meeting_ids from pandas_query result: {extracted_ids}")
                        relevant_meeting_ids = extracted_ids
            
            print(f"Query rewrite result:")
            print(f"  Original: {query}")
            print(f"  Rewritten: {rewritten_query}")
            print(f"  Meeting IDs: {relevant_meeting_ids}")
            
            # Store in state for retrieve_func to use
            tool_context.state["rewrite_result"] = {
                "rewritten_query": rewritten_query,
                "relevant_meeting_ids": relevant_meeting_ids
            }
            
            return {
                "success": True,
                "rewritten_query": rewritten_query,
                "relevant_meeting_ids": relevant_meeting_ids
            }
            
        except Exception as e:
            print(f"Error in rewrite_query_func: {e}")
            # On error, return original query
            return {
                "success": False,
                "rewritten_query": query,
                "relevant_meeting_ids": [],
                "error": str(e)
            }

    def __init__(self):       
        # create tools
        setup_tool = FunctionTool(func=self.setup_state_func)
        retrieve_tool = FunctionTool(func=self.rag_retrieve_func)
        plan_tool = AgentTool(agent=planner_agent)
        rewrite_tool = FunctionTool(func=self.rewrite_query_func) 
        answer_tool = AgentTool(agent=answer_synthesis_agent)

        
        super().__init__(
        name="FullRAGSystemAgent",
        model=LiteLlm(model="gpt-4o-mini"),
        instruction="""You are a RAG system orchestrator. Follow this dynamic sequence:
0. First, always call setup_tool to initialize state variables, if user has an input sentence
1. Second, always call plan_tool to analyze the query and create an execution plan
2. If plan.need_rewrite is True, call rewrite_tool, you must send the original query to rewrite_tool
3. If plan.need_rag is True, call retrieve_tool with the query (MUST use output from rewrite_tool if need_rewrite is True, otherwise use the original user query)
   If plan.need_rag is False, do not call retrieve_tool
4. Always call answer_tool with the original user_query and retrieved chunks (or empty chunks if retrieval was skipped)

Always use the user's latest message as the query.
State is managed automatically by the sub-agents via their output_keys.
""",
        tools=[setup_tool, plan_tool,rewrite_tool, retrieve_tool, answer_tool],
        include_contents= "none"
    )
        

        
    def _initialize_components(self):
        """Delay initialization of all RAG components"""
        if self.initialized:
            return
            
        print("Initializing RAG system for first query...")
        self.retriever = self._initialize_complete_rag_system()
                
        self.initialized = True
        
    
        
    def _initialize_complete_rag_system(self) -> HierarchicalRetriever:
        """
        rag pipeline
        """
        print("=" * 80)
        print("Initializing Complete RAG System for ADK UI")
        print("=" * 80)
        
        data_dir = DATA_DIR
        if data_dir is None:
            raise ValueError('data_dir is none.')
        
        include_chunk_level = False  # ADK UI not include chunk level
        
        # Load all meetings from data directory
        print("\n[Step 1] Loading meeting data...")
        loader = DataLoader(data_dir)
        all_meetings = loader.load_all_meetings()
        print(f"✓ Loaded {len(all_meetings)} meetings from data directory")

        if not all_meetings:
            raise ValueError("No meetings found. Please check your data directory.")

        # Check for existing vector store
        print("\n[Step 2] Checking for existing vector store...")
        vector_store = None
        processed_meeting_ids: Set[str] = set()

        if VECTOR_STORE_DIR.exists():
            try:
                print(f"✓ Found existing vector store at {VECTOR_STORE_DIR}")
                print("   Loading existing vector store...")
                vector_store = VectorStoreUtilsMixin.load(VECTOR_STORE_DIR)
                processed_meeting_ids = vector_store.get_processed_meeting_ids()
                print(f"✓ Loaded existing vector store with {len(processed_meeting_ids)} processed meetings")
            except Exception as e:
                print(f"Warning: Failed to load existing vector store: {e}")
                print("   Will create new vector store...")
                vector_store = None
                processed_meeting_ids = set()
        else:
            print("No existing vector store found. Will create new one.")

        # Filter to new meetings only
        new_meetings = []
        for meeting in all_meetings:
            if meeting.meeting_id not in processed_meeting_ids:
                new_meetings.append(meeting)

        print(f"\n[Step 3] Found {len(new_meetings)} new meetings to process")

        # Process new meetings
        if new_meetings:
            print("\n[Step 4] Processing new meetings...")

            # Chunk new meetings
            print("   Chunking new meetings...")
            chunker = HierarchicalChunker()
            new_chunks = chunker.chunk_all_levels(new_meetings, include_chunk_level=include_chunk_level)

            for level, chunks in new_chunks.items():
                print(f"     {level:10s}: {len(chunks):4d} chunks")

            # Initialize vector store if needed
            if vector_store is None:
                print("   Creating new vector store...")
                vector_store = VectorStoreUtilsMixin()
            else:
                print("   Using existing vector store...")

            # Add new chunks to vector store
            print("   Adding new chunks to vector store...")
            for level in ['metadata', 'summary', 'meeting']:
                chunks = new_chunks.get(level, [])
                if chunks:
                    vector_store.add_chunks(
                        chunks=chunks,
                        Level=level,
                        generate_embedding=True
                    )
                    print(f"     ✓ Added {len(chunks)} chunks to {level} level")

            # Save updated vector store
            print("   Saving updated vector store...")
            vector_store.save(VECTOR_STORE_DIR)
            print(f"✓ Vector store saved to {VECTOR_STORE_DIR}")

        else:
            print("\n[Step 4] No new meetings to process ✓")

        # Load meeting IDs for Filter Agent
        print("\n[Step 5b] Registering meeting IDs...")
        self.all_meeting_ids = {m.meeting_id for m in all_meetings}
        print(f"✓ Registered {len(self.all_meeting_ids)} meeting IDs")
        
        # Create retriever
        print("\n[Step 6] Creating hierarchical retriever...")
        if vector_store is None:
            raise RuntimeError("vector initailazation failed")
        retriever = HierarchicalRetriever(vector_store)
        print("✓ Hierarchical retriever created")


        
        print("\n" + "=" * 80)
        print("RAG System Initialized Successfully!")
        print(f" Total meetings available: {len(all_meetings)}")
        print(f" Processed meetings: {len(processed_meeting_ids)}")
        print(f"   New meetings processed: {len(new_meetings)}")
        print("=" * 80)
        
        self.total_meetings = len(all_meetings)
        return retriever


# golbal agent for google adk web surface UI
root_agent = FullRAGSystemAgent()