"""
Test script for ElasticsearchVectorStore
Tests all major functionality including indexing, searching, and filtering
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.vector_store_es import ElasticsearchVectorStore
from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from config.settings import DATA_DIR


def test_health_check(vector_store: ElasticsearchVectorStore):
    """Test 1: Health check"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    is_healthy = vector_store.health_check()
    if is_healthy:
        print("✅ Elasticsearch connection successful")
        
        # Get cluster info
        cluster_info = vector_store.get_cluster_info()
        print(f"   Cluster: {cluster_info.get('cluster_name', 'unknown')}")
        print(f"   Version: {cluster_info.get('version', 'unknown')}")
        print(f"   Status: {cluster_info.get('status', 'unknown')}")
    else:
        print("❌ Elasticsearch connection failed")
        print("   Please make sure Elasticsearch is running and configured correctly")
        return False
    
    return True


def test_create_indices(vector_store: ElasticsearchVectorStore):
    """Test 2: Create indices"""
    print("\n" + "="*80)
    print("TEST 2: Create Indices")
    print("="*80)
    
    try:
        vector_store.create_indices(force_recreate=True)
        print("✅ All indices created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create indices: {e}")
        return False


def test_add_chunks(vector_store: ElasticsearchVectorStore, all_chunks: dict):
    """Test 3: Add chunks to indices"""
    print("\n" + "="*80)
    print("TEST 3: Add Chunks to Indices")
    print("="*80)
    
    success = True
    for level in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level, [])
        if not chunks:
            print(f"⚠️  No {level} chunks to index")
            continue
        
        try:
            count = vector_store.add_chunks(
                chunks=chunks,
                level=level,
                generate_embeddings=True,
                batch_size=50,
                show_progress=True
            )
            print(f"✅ Indexed {count}/{len(chunks)} {level} chunks")
        except Exception as e:
            print(f"❌ Failed to index {level} chunks: {e}")
            success = False
    
    return success


def test_index_stats(vector_store: ElasticsearchVectorStore):
    """Test 4: Get index statistics"""
    print("\n" + "="*80)
    print("TEST 4: Index Statistics")
    print("="*80)
    
    try:
        stats = vector_store.get_index_stats()
        print("Index document counts:")
        for level, count in stats.items():
            print(f"  {level:10s}: {count:4d} documents")
        print("✅ Statistics retrieved successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return False


def test_search_basic(vector_store: ElasticsearchVectorStore):
    """Test 5: Basic search without filters"""
    print("\n" + "="*80)
    print("TEST 5: Basic Search (No Filters)")
    print("="*80)
    
    test_queries = [
        ("What topics were discussed in data001?", "metadata"),
        ("What is the participants in data001?", "summary"),
        ("How is the project going for data001?", "meeting")
    ]
    
    success = True
    for query, level in test_queries:
        try:
            results = vector_store.search(
                query_text=query,
                level=level,
                top_k=3
            )
            
            print(f"\nQuery: '{query}' (level: {level})")
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  [{i}] Score: {result['score']:.4f} | Meeting: {result['meeting_id']}")
                print(f"      Text preview: {result['text'][:100]}...")
            
            if results:
                print("✅ Search successful")
            else:
                print("⚠️  No results found (this might be normal if index is empty)")
        except Exception as e:
            print(f"❌ Search failed: {e}")
            success = False
    
    return success


def test_search_with_filters(vector_store: ElasticsearchVectorStore):
    """Test 6: Search with metadata filters"""
    print("\n" + "="*80)
    print("TEST 6: Search with Filters")
    print("="*80)
    
    # Test regular metadata filters
    test_cases = [
        {
            "name": "Filter by meeting_type",
            "query": "What is the meeting content of podcast interview?",
            "level": "metadata",
            "filters": {"meeting_type": "Podcast Interview"}
        },
        {
            "name": "Filter by topics (list)",
            "query": "Which task talks about Persistence and Career Rebuilding?",
            "level": "metadata",
            "filters": {"topics": ["Persistence and Career Rebuilding"]}
        },
        {
            "name": "Filter by meeting_ids",
            "query": "what is the first meeting about?",
            "level": "metadata",
            "meeting_ids": ["data001"]
        }
    ]
    
    success = True
    for test_case in test_cases:
        try:
            print(f"\n{test_case['name']}:")
            print(f"  Query: '{test_case['query']}'")
            print(f"  Filters: {test_case.get('filters', {})}")
            if 'meeting_ids' in test_case:
                print(f"  Meeting IDs: {test_case['meeting_ids']}")
            
            results = vector_store.search(
                query_text=test_case['query'],
                level=test_case['level'],
                top_k=3,
                filters=test_case.get('filters'),
                meeting_ids=test_case.get('meeting_ids')
            )
            
            print(f"  Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):  # Show first 2
                print(f"    [{i}] Meeting: {result['meeting_id']} | Score: {result['score']:.4f}")

                # print text 
                text = result.get('text', '')
                if text and text.strip():
                    text_preview = text[:150].replace('\n', '')
                    if len(text)>150:
                        text_preview += "..."
                    print(f"Text preview: {text_preview}")
                
                    
            print("  ✅ Filter search successful")
        except Exception as e:
            print(f"  ❌ Filter search failed: {e}")
            success = False
    
    return success


def test_search_with_actions_filters(vector_store: ElasticsearchVectorStore):
    """Test 7: Search with actions filters (nested)"""
    print("\n" + "="*80)
    print("TEST 7: Search with Actions Filters (Nested)")
    print("="*80)
    
    test_cases = [
        {
            "name": "Filter by action status",
            "query": "action status in meeting001",
            "level": "metadata",
            "filters": {"actions.status": "open"},
            "meeting_ids": ["data001"]
        },
        {
            "name": "Filter by action priority",
            "query": "priority in meeting001",
            "level": "metadata",
            "filters": {"actions.priority": ["high", "medium"]},
            "meeting_ids": ["data001"]
        },
        {
            "name": "Filter by assignee name",
            "query": "assignee in meeting001",
            "level": "metadata",
            "filters": {"actions.assignee.name": "Nimi Mehta"},
            "meeting_ids": ["data001"]
        }
    ]
    
    success = True
    for test_case in test_cases:
        try:
            print(f"\n{test_case['name']}:")
            print(f"  Query: '{test_case['query']}'")
            print(f"  Filters: {test_case['filters']}")
            
            results = vector_store.search(
                query_text=test_case['query'],
                level=test_case['level'],
                top_k=3,
                filters=test_case['filters'],
                meeting_ids=test_case.get('meeting_ids')
            )
            
            print(f"  Found {len(results)} results")
            for i, result in enumerate(results[:3], 1):
                print(f"    [{i}] Meeting: {result['meeting_id']} | Score: {result['score']:.4f}")
                # Show actions in metadata if available
                actions = result.get('metadata', {}).get('actions', [])
                if actions:
                    print(f"        Actions: {len(actions)} action(s)")
                    
                    
                    
                # print text 
                text = result.get('text', '')
                if text and text.strip():
                    text_preview = text[:150].replace('\n', '')
                    if len(text)>150:
                        text_preview += "..."
                    print(f"Text preview: {text_preview}")
            
            print("  ✅ Actions filter search successful")
        except Exception as e:
            print(f"  ❌ Actions filter search failed: {e}")
            import traceback
            traceback.print_exc() 
            success = False
    
    return success


def test_delete_indices(vector_store: ElasticsearchVectorStore, cleanup: bool = False):
    """Test 8: Delete indices (optional cleanup)"""
    if not cleanup:
        print("\n" + "="*80)
        print("TEST 8: Delete Indices (SKIPPED - set cleanup=True to enable)")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("TEST 8: Delete Indices (Cleanup)")
    print("="*80)
    
    try:
        vector_store.delete_all_indices()
        print("✅ All indices deleted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to delete indices: {e}")
        return False


def main():
    """Main test function"""
    print("="*80)
    print("ELASTICSEARCH VECTOR STORE TEST SUITE")
    print("="*80)
    
    # Initialize vector store
    print("\n[Initialization] Creating ElasticsearchVectorStore...")
    vector_store = ElasticsearchVectorStore()
    
    # Test 1: Health check
    if not test_health_check(vector_store):
        print("\n❌ Health check failed. Please check your Elasticsearch configuration.")
        return
    
    # Load and chunk data
    print("\n" + "="*80)
    print("PREPARATION: Loading and Chunking Data")
    print("="*80)
    
    print("\n[Step 1] Loading meetings...")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    print(f"✅ Loaded {len(meetings)} meetings")
    
    if not meetings:
        print("❌ No meetings found! Cannot proceed with tests.")
        return
    
    print("\n[Step 2] Chunking meetings...")
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings, include_chunk_level=False)
    
    for level, chunks in all_chunks.items():
        print(f"  {level:10s}: {len(chunks):4d} chunks")

    # Run tests
    test_results = {}
    
    # test_results['create_indices'] = test_create_indices(vector_store)
    # test_results['add_chunks'] = test_add_chunks(vector_store, all_chunks)
    # test_results['index_stats'] = test_index_stats(vector_store)
    #test_results['search_basic'] = test_search_basic(vector_store)
    #test_results['search_filters'] = test_search_with_filters(vector_store)
    test_results['search_actions'] = test_search_with_actions_filters(vector_store)
    
    # Optional cleanup (set to True to delete indices after testing)
    test_results['delete_indices'] = test_delete_indices(vector_store, cleanup=False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)


if __name__ == "__main__":
    main()