import os
import sys
from pathlib import Path
from typing import Optional, Set, Dict, Any, List


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store import HybridSearchVectorStore
from src.retrieval.vector_store_utils import VectorStoreUtilsMixin
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.retrieval.rag_pipeline import RAGPipeline
from src.retrieval.orchestrator import handle_user_query
from config.settings import DATA_DIR, VECTOR_STORE_DIR, shared_session_service, APP_NAME, USER_ID
from src.retrieval import query_rewriter_muti_agent


from google.adk.agents import BaseAgent
from google.genai import types
from google.adk.events import Event

from pydantic import Field



# clean useless environment variable SSL_CERT_FILE
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]
        print(f"Warning: Removed invalid SSL_CERT_FILE: {cert_path}")


class FullRAGSystemAgent(BaseAgent):
    
    rag_pipeline: Optional[RAGPipeline] = Field(default=None)
    initialized: bool = Field(default=False)
    meeting_catalog: str = Field(default="")
    def __init__(self):
        super().__init__(name="FullRAGSystemAgent")
        self.rag_pipeline = None
        self.initialized = False
        self.meeting_catalog = ""
        
    def _initialize_complete_rag_system(self) -> RAGPipeline:
        """
        rag pipeline
        """
        print("=" * 80)
        print("🔄 Initializing Complete RAG System for ADK UI")
        print("=" * 80)
        
        data_dir = DATA_DIR
        if data_dir is None:
            raise ValueError('data_dir is none.')
        
        include_chunk_level = False  # ADK UI not include chunk level
        
        # Step 1: Load all meetings from data directory
        print("\n[Step 1] Loading meeting data...")
        loader = DataLoader(data_dir)
        all_meetings = loader.load_all_meetings()
        print(f"✓ Loaded {len(all_meetings)} meetings from data directory")

        if not all_meetings:
            raise ValueError("No meetings found. Please check your data directory.")

        # Step 2: Check for existing vector store
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

        # Step 3: Filter to new meetings only
        new_meetings = []
        for meeting in all_meetings:
            if meeting.meeting_id not in processed_meeting_ids:
                new_meetings.append(meeting)

        print(f"\n[Step 3] Found {len(new_meetings)} new meetings to process")

        # Step 4: Process new meetings
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

        # Step 5: Register all meetings with query rewriter (for MEETING_CATALOG)
        print("\n[Step 5] Registering meetings with query rewriter...")
        query_rewriter_muti_agent.set_meetings(all_meetings)
        self.meeting_catalog = query_rewriter_muti_agent.MEETING_CATALOG
        
        print("DEBUG: loaded meetings count:", len(all_meetings))
        print(f"DEBUG: MEETING_CATALOG preview:\n{self.meeting_catalog[:500]}...")

        # Step 6: Create retriever
        print("\n[Step 6] Creating hierarchical retriever...")
        if vector_store is None:
            raise RuntimeError("vector initailazation failed")
        retriever = HierarchicalRetriever(vector_store)
        print("✓ Hierarchical retriever created")

        # Step 7: Create RAG pipeline  
        print("\n[Step 7] Creating RAG pipeline...")
        from src.retrieval.answer_generator_muti_agent import AnswerGeneratorMultiAgent
        from src.retrieval.query_rewriter_muti_agent import QueryRewriter

        # Construct agents using unified session keys from config.settings
        answer_generator = AnswerGeneratorMultiAgent(
            session_service=shared_session_service, 
            user_id=USER_ID, 
            app_name=APP_NAME
        )
        query_rewriter = QueryRewriter(
            session_service=shared_session_service, 
            user_id=USER_ID, 
            app_name=APP_NAME
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            answer_generator=answer_generator,
            query_rewriter=query_rewriter,
            use_query_rewriter=True
        )
        
        print("✓ RAG pipeline created")
        
        print("\n" + "=" * 80)
        print("🎉 RAG System Initialized Successfully!")
        print(f"   Total meetings available: {len(all_meetings)}")
        print(f"   Processed meetings: {len(processed_meeting_ids)}")
        print(f"   New meetings processed: {len(new_meetings)}")
        print("=" * 80)
        
        return pipeline
    
    async def _run_async_impl(self, invocation_context, **kwargs):
        """
        user question and retrieve logic
        """
        try:
            #delay initialization for rag pipeline
            if not self.initialized:
                print("🔄 Initializing RAG system for first query...")
                self.rag_pipeline = self._initialize_complete_rag_system()
                self.initialized = True
            
            # extract user query
            user_message = invocation_context.user_content
            if isinstance(user_message, types.Content):
                user_query = user_message.parts[0].text if user_message.parts else ""
            else:
                user_query = str(user_message)
            
            user_query = user_query.strip()
# empty query situation
            if not user_query:
                yield Event(
                    author= "assistant",
                    content=types.Content(
                        role="assistant", 
                        parts=[types.Part(text="You do not ask a question")]
                    )
                )
                return
            
            print(f"🔍 Processing query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
            
            # call complete rag logic
            result = await handle_user_query(
                user_query=user_query,
                meeting_catalog=self.meeting_catalog,
                rag_pipeline=self.rag_pipeline
            )
            
            # extract answers
            answer = result.get('answer', 'Sorry, I cannot generate answers.')
            
            # for debug
            if result.get('rewritten_query'):
                print(f"   → Rewritten: {result['rewritten_query']}")
            if result.get('num_chunks'):
                print(f"   → Used {result['num_chunks']} chunks")
            
            yield Event(
                author= "assistant",
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text=answer)]
                )
            )
            
        except Exception as e:
            error_msg = f"There is an error when dealing with query: {str(e)}"
            print(f"❌ RAGSystemAgent error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            yield Event(
                author = "assistant",
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text=f"There is some errors when dealing with your query：{str(e)}")]
                )
            )


# golbal agent for google adk web surface UI
root_agent = FullRAGSystemAgent()