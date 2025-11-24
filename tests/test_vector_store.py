import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store_utils import VectorStoreUtilsMixin
from config.settings import DATA_DIR


def print_result(result: dict, idx: int):
    """print single search result"""
    print(f"\n{'='*80}")
    print(f"[Result #{idx}]")
    print(f"{'='*80}")
    print(f"Chunk ID: {result.get('chunk_id', 'N/A')}")
    print(f"Meeting ID: {result.get('meeting_id', 'N/A')}")
    print(f"Index: {result.get('index', 'N/A')}")
    print(f"\nScores:")
    print(f"  Hybrid Score: {result.get('hybrid_score', 0):.4f}")
    print(f"  Vector Score: {result.get('vector_score', 0):.4f} (normalized: {result.get('vector_score_norm', 0):.4f})")
    print(f"  BM25 Score: {result.get('bm25_score', 0):.4f} (normalized: {result.get('bm25_score_norm', 0):.4f})")
    
    # metadata
    # metadata = result.get('metadata', {})
    # if metadata:
    #     print(f"\nMetadata:")
    #     for key, value in metadata.items():
    #         if value:  
    #             print(f"  {key}: {value}")
    
    # show text
    text = result.get('text', '')
    if text:
        print(f"\nText Content:")
        print(f"{'-'*80}")
        # if text is too long, truncate it
        display_text = text if len(text) <= 1000 else text[:1000] + "\n... (truncated)"
        print(display_text)
        print(f"{'-'*80}")


def print_results_summary(level: str, query: str, results: list, top_k: int):
    """print search results summary"""
    print("\n" + "="*80)
    print(f"SEARCH RESULTS - {level.upper()} Level")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Top K: {top_k}")
    print(f"Found: {len(results)} results")
    print("="*80)


def interactive_search(vector_store: VectorStoreUtilsMixin):
    """interactive search loop"""
    current_level = 'metadata'
    default_top_k = 5
    
    print("\n" + "="*80)
    print("INTERACTIVE VECTOR STORE TEST")
    print("="*80)
    print("\nAvailable commands:")
    print("  - Enter a query to search")
    print("  - 'level <name>' to switch level (metadata/summary/meeting)")
    print("  - 'topk <number>' to set top_k")
    print("  - 'stats' to show vector store statistics")
    print("  - 'help' to show this help")
    print("  - 'quit' or 'exit' to exit")
    print("\n" + "="*80)
    
    while True:
        try:
            # show current settings
            print(f"\n[Current Level: {current_level.upper()}, Top K: {default_top_k}]")
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # process commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Enter a query to search")
                print("  - 'level <name>' to switch level (metadata/summary/meeting)")
                print("  - 'topk <number>' to set top_k")
                print("  - 'stats' to show vector store statistics")
                print("  - 'help' to show this help")
                print("  - 'quit' or 'exit' to exit")
                continue
            
            elif user_input.lower() == 'stats':
                stats = vector_store.get_stats()
                print("\n" + "="*80)
                print("VECTOR STORE STATISTICS")
                print("="*80)
                for level_name, stat in stats.items():
                    print(f"\n{level_name.upper()} Level:")
                    print(f"  Document Count: {stat['doc_count']}")
                    print(f"  FAISS Index Size: {stat['faiss_index_size']}")
                    print(f"  Has BM25: {stat['has_bm25']}")
                print("="*80)
                continue
            
            elif user_input.lower().startswith('level '):
                new_level = user_input[6:].strip().lower()
                if new_level in ['metadata', 'summary', 'meeting']:
                    current_level = new_level
                    print(f"✅ Switched to {current_level} level")
                else:
                    print(f"❌ Invalid level: {new_level}. Must be one of: metadata, summary, meeting")
                continue
            
            elif user_input.lower().startswith('topk '):
                try:
                    new_topk = int(user_input[5:].strip())
                    if new_topk > 0:
                        default_top_k = new_topk
                        print(f"✅ Top K set to {default_top_k}")
                    else:
                        print("❌ Top K must be a positive integer")
                except ValueError:
                    print("❌ Invalid number for top_k")
                continue
            
            # otherwise, process as query
            query = user_input
            print(f"\n🔍 Searching {current_level} level with query: '{query}'")
            print("Please wait...")
            
            # execute search
            results = vector_store.search(
                query_text=query,
                Level=current_level,
                top_k=default_top_k
            )
            
            # show results
            print_results_summary(current_level, query, results, default_top_k)
            
            if not results:
                print("\n⚠️  No results found")
                continue
            
            # show each result
            for i, result in enumerate(results, 1):
                print_result(result, i)
            
            print(f"\n✅ Displayed {len(results)} results")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """main function"""
    print("="*80)
    print("VECTOR STORE INTERACTIVE TEST")
    print("="*80)
    
    # 1. load data
    print("\n[Step 1] Loading meeting data...")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    print(f"✅ Loaded {len(meetings)} meetings")
    
    if not meetings:
        print("❌ No meetings found! Cannot proceed.")
        return
    
    # 2. chunk meetings
    print("\n[Step 2] Chunking meetings...")
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings, include_chunk_level=False)
    
    for level, chunks in all_chunks.items():
        print(f"  {level:10s}: {len(chunks):4d} chunks")
    
    # 3. initialize vector store
    print("\n[Step 3] Initializing VectorStore...")
    vector_store = VectorStoreUtilsMixin()
    
    # 4. add chunks to each level
    print("\n[Step 4] Adding chunks to vector store...")
    for level in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level, [])
        if chunks:
            vector_store.add_chunks(
                chunks=chunks,
                Level=level,
                generate_embedding=True
            )
    
    # 5. show statistics
    print("\n[Step 5] Vector Store Statistics:")
    stats = vector_store.get_stats()
    for level, stat in stats.items():
        print(f"  {level:10s}: {stat['doc_count']:4d} docs, "
              f"FAISS: {stat['faiss_index_size']:4d}, "
              f"BM25: {'Yes' if stat['has_bm25'] else 'No'}")
    
    # 6. enter interactive mode
    interactive_search(vector_store)


if __name__ == "__main__":
    main()