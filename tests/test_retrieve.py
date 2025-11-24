import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store import HybridSearchVectorStore
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from config.settings import DATA_DIR

def print_metadata_results(results, query):
    print("\n metadata results are following:")
    print(f"Query: {query}")
    print(f"Found: {len(results)} results")
    
    if not results:
        print("We cannot find results")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n Metadata results #{i}")
        print(f"Meeting id:{result.get('meeting_id', 'N/A')}")
        print(f"Chunk Id:{result.get('chunk_id', 'N/A')}")
        print(f"Hybrid Score: {result.get('hybrid_score', 0)}")
        
        metadata = result.get('metadata', {})
        if metadata:
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Date: {metadata.get('datetime', 'N/A')}")
            if metadata.get('summary_brief'):
                brief = metadata.get('summary_brief', '')
                display_brief = brief[:200] + "..." if len(brief) > 200 else brief
                print(f"  Brief Summary: {display_brief}")
                
                
def print_summary_results(results, query):
    print("\n summary results are following:")
    print(f"Query: {query}")
    print(f"Found: {len(results)} results")
    
    if not results:
        print("We cannot find results")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n Summary results #{i}")
        print(f"Meeting id:{result.get('meeting_id', 'N/A')}")
        print(f"Chunk Id:{result.get('chunk_id', 'N/A')}")
        print(f"Hybrid Score: {result.get('hybrid_score', 0)}")
        
        metadata = result.get('metadata', {})
        references = metadata.get('reference', [])
        
        if references:
            if isinstance(references, list):
                print(f" Entry IDs ({len(references)}): {', '.join(references[:5])}{'...' if len(references) > 5 else ''}")
            else:
                print(f" Entry ID: {references}")
        else:
            print(f"  Entry IDs: None")
        
        # show the text
        text = result.get('text', '')
        if text:
            display_text = text[:300] + "..." if len(text) > 300 else text
            print(f" Text Preview: {display_text}")
            
            
            
def print_meeting_results(results, query):
    print("\n meeting results are following:")
    print(f"Query: {query}")
    print(f"Found: {len(results)} results")
    
    if not results:
        print("We cannot find results")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n Meeting results #{i}")
        print(f"Meeting id:{result.get('meeting_id', 'N/A')}")
        print(f"Chunk Id:{result.get('chunk_id', 'N/A')}")
        print(f"  Entry ID: {result.get('metadata', {}).get('entry_id', 'N/A')}")
        print(f"Hybrid Score: {result.get('hybrid_score', 0)}")
        
        text = result.get('text', '')
        if text:
            display_text = text[:300] + "..." if len(text) > 300 else text
            print(f"  Text Preview: {display_text}")



       
            
def test_hierarchical_retriever(query:str, top_k_metadata: int=5, top_k_summary: int=10, top_k_meeting: int=10):
    print("HIERARCHICAL RETRIEVER TEST")
    
    # print load data
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    print(f"Loaded {len(meetings)} meetings")
    
    if not meetings:
        print("No meetings found, there maybe an error")
        return
    
    # chunk meetings
    print("\n Chunking meetings...")
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings)
    
    for level, chunk in all_chunks.items():
        print(f"  {level:10s}: {len(chunk):4d} chunks")
        
        
    # Initialize vector store
    print("\n Initializing VectorStore...")
    vector_store = HybridSearchVectorStore()
    
    # add chunks to each level
    print("\n Adding chunks to vector store")
    for level in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level, [])
        if chunks:
            vector_store.add_chunks(
                chunks=chunks,
                Level=level,
                generate_embedding=True
            )
            print(f"Added {len(chunks)} chunks to {level} level")
            
            
            
    print("\n Creating HierarchicalRetriever...")
    retriever = HierarchicalRetriever(vector_store)
    print("HierarchicalRetriever created")
    
    
    # 6. Execute hierarchical search
    print(f"\n[Step 6] Executing hierarchical search...")
    print(f"Query: '{query}'")
    print(f"Top K - Metadata: {top_k_metadata}, Summary: {top_k_summary}, Meeting: {top_k_meeting}")
    
    results = retriever.search(
        query=query,
        top_k_metadata=top_k_metadata,
        top_k_summary=top_k_summary,
        top_k_meeting=top_k_meeting
    )
    
    
    print_metadata_results(results.metadata_results, query)
    print_summary_results(results.summary_results, query)
    print_meeting_results(results.meeting_results, query)
    
    print("\n" + "="*80)
    print("✅ Test completed!")
    print("="*80)
    
    return results


def interactive_test():
    print("\n" + "="*80)
    print("HIERARCHICAL RETRIEVER INTERACTIVE TEST")
    print("="*80)
    

    print("\n[Initialization] Setting up vector store...")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    print(f"✅ Loaded {len(meetings)} meetings")
    
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings, include_chunk_level=False)
    
    vector_store = HybridSearchVectorStore()
    for level in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level, [])
        if chunks:
            vector_store.add_chunks(
                chunks=chunks,
                Level=level,
                generate_embedding=True
            )
    
    retriever = HierarchicalRetriever(vector_store)
    print("✅ Setup complete!")
    
    # 交互式循环
    print("\n" + "="*80)
    print("Enter queries to test hierarchical retrieval")
    print("Commands: 'quit' or 'exit' to exit")
    print("="*80)
    
    while True:
        try:
            query = input("\n> Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # 执行搜索
            results = retriever.search(
                query=query,
                top_k_metadata=1,
                top_k_summary=1,
                top_k_meeting=10
            )
            
            print_metadata_results(results.metadata_results, query)
            print_summary_results(results.summary_results, query)
            print_meeting_results(results.meeting_results, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    
    # test_hierarchical_retriever(
    #     query="",
    #     top_k_metadata=5,
    #     top_k_summary=10,
    #     top_k_meeting=10
    # )
    

    interactive_test()
            