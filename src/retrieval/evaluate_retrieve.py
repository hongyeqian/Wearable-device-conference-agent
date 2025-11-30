import sys
import json
from pathlib import Path
from typing import List, Set, Dict, Any
from datetime import datetime
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store import HybridSearchVectorStore
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.retrieval.query_rewriter import QueryRewriter
from config.settings import DATA_DIR


def extract_meeting_level_ids(results: List[Dict[str, Any]]) -> Set[str]:
    """Extract entry_ids from meeting level results"""
    entry_ids = set()
    for result in results:
        metadata = result.get('metadata', {})
        entry_id = metadata.get('entry_id')
        if entry_id:
            entry_ids.add(entry_id)
    return entry_ids


def extract_summary_level_ids(results: List[Dict[str, Any]]) -> Set[str]:
    """Extract summary_ids from summary level results"""
    summary_ids = set()
    for result in results:
        metadata = result.get('metadata', {})
        summary_ids_list = metadata.get('summary_ids', [])
        if isinstance(summary_ids_list, list):
            summary_ids.update(summary_ids_list)
        elif isinstance(summary_ids_list, str):
            summary_ids.add(summary_ids_list)
    return summary_ids


def normalize_expected_ids(expected: Any) -> Set[str]:
    """Convert expected IDs to a set, handling both string and list formats"""
    if isinstance(expected, str):
        return {expected}
    elif isinstance(expected, list):
        return set(expected)
    else:
        return set()


def calculate_recall(retrieved: Set[str], expected: Set[str]) -> float:
    """Calculate recall: |retrieved ∩ expected| / |expected|"""
    if not expected:
        return 0.0
    intersection = retrieved & expected
    return len(intersection) / len(expected)


def format_metadata_level_results(results: List[Dict[str, Any]], output: StringIO):
    """Format metadata level results with chunk IDs and meeting IDs"""
    output.write(f"\n  Metadata Level Results ({len(results)} chunks):\n")
    if not results:
        output.write("    No results found\n")
        return set()
    
    meeting_ids = set()
    for i, result in enumerate(results, 1):
        chunk_id = result.get('chunk_id', 'N/A')
        meeting_id = result.get('meeting_id', 'N/A')
        score = result.get('hybrid_score', result.get('vector_score', 0))
        
        if meeting_id and meeting_id != 'N/A':
            meeting_ids.add(meeting_id)
        
        output.write(f"    [{i}] Chunk ID: {chunk_id}\n")
        output.write(f"        Meeting ID: {meeting_id}\n")
        output.write(f"        Score: {score:.4f}\n")
    
    if meeting_ids:
        output.write(f"\n    Extracted Meeting IDs: {sorted(meeting_ids)}\n")
    
    return meeting_ids


def format_summary_level_results(results: List[Dict[str, Any]], output: StringIO):
    """Format summary level results with chunk IDs, summary_ids, and references"""
    output.write(f"\n  Summary Level Results ({len(results)} chunks):\n")
    if not results:
        output.write("    No results found\n")
        return set(), set()
    
    all_summary_ids = set()
    all_references = set()
    
    for i, result in enumerate(results, 1):
        chunk_id = result.get('chunk_id', 'N/A')
        meeting_id = result.get('meeting_id', 'N/A')
        score = result.get('hybrid_score', result.get('vector_score', 0))
        metadata = result.get('metadata', {})
        
        summary_ids = metadata.get('summary_ids', [])
        if isinstance(summary_ids, str):
            summary_ids = [summary_ids] if summary_ids else []
        elif not isinstance(summary_ids, list):
            summary_ids = []
        
        references = metadata.get('references', [])
        if isinstance(references, str):
            references = [references] if references else []
        elif not isinstance(references, list):
            references = []
        
        all_summary_ids.update(summary_ids)
        all_references.update(references)
        
        output.write(f"    [{i}] Chunk ID: {chunk_id}\n")
        output.write(f"        Meeting ID: {meeting_id}\n")
        output.write(f"        Score: {score:.4f}\n")
        if summary_ids:
            output.write(f"        Summary IDs: {sorted(summary_ids)}\n")
        if references:
            output.write(f"        References (Entry IDs): {sorted(references)}\n")
    
    if all_summary_ids:
        output.write(f"\n    Extracted Summary IDs: {sorted(all_summary_ids)}\n")
    if all_references:
        output.write(f"\n    Extracted Entry IDs (from references): {sorted(all_references)}\n")
    
    return all_summary_ids, all_references


def format_meeting_level_results(results: List[Dict[str, Any]], output: StringIO):
    """Format meeting level results with chunk IDs and entry_ids"""
    output.write(f"\n  Meeting Level Results ({len(results)} chunks):\n")
    if not results:
        output.write("    No results found\n")
        return set()
    
    all_entry_ids = set()
    
    for i, result in enumerate(results, 1):
        chunk_id = result.get('chunk_id', 'N/A')
        meeting_id = result.get('meeting_id', 'N/A')
        score = result.get('hybrid_score', result.get('vector_score', 0))
        metadata = result.get('metadata', {})
        
        entry_id = metadata.get('entry_id', '')
        if entry_id:
            all_entry_ids.add(entry_id)
        
        section = metadata.get('section', 'N/A')
        
        output.write(f"    [{i}] Chunk ID: {chunk_id}\n")
        output.write(f"        Meeting ID: {meeting_id}\n")
        output.write(f"        Section: {section}\n")
        output.write(f"        Entry ID: {entry_id if entry_id else 'N/A'}\n")
        output.write(f"        Score: {score:.4f}\n")
    
    if all_entry_ids:
        output.write(f"\n    Extracted Entry IDs: {sorted(all_entry_ids)}\n")
    
    return all_entry_ids


def run_evaluation(
    query_file: str = "query_test.json",
    top_k_metadata: int = 5,
    top_k_summary: int = 10,
    top_k_meeting: int = 10,
    suppress_output: bool = True
):
    """
    Run evaluation on queries from query_test.json
    
    Args:
        query_file: Path to JSON file with test queries
        top_k_metadata: Top K for metadata level search
        top_k_summary: Top K for summary level search
        top_k_meeting: Top K for meeting level search
        suppress_output: If True, suppress HierarchicalRetriever's print statements
    """
    # Create output directory
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_results_{timestamp}.txt"
    
    # Use StringIO to collect all output
    output = StringIO()
    
    # Load test queries
    query_path = project_root / query_file
    with open(query_path, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    
    output.write("=" * 80 + "\n")
    output.write("HIERARCHICAL RETRIEVER EVALUATION\n")
    output.write("=" * 80 + "\n")
    
    # Initialize retriever
    output.write("\n[Initialization] Setting up vector store...\n")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    output.write(f"Loaded {len(meetings)} meetings\n")
    
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
            output.write(f"Added {len(chunks)} chunks to {level} level\n")
    
    retriever = HierarchicalRetriever(vector_store)
    output.write("Setup complete!\n")
    
    # Suppress output if needed
    import builtins
    original_print = builtins.print
    if suppress_output:
        def silent_print(*args, **kwargs):
            # Only suppress prints from hierarchical_retriever
            if 'Searching' in str(args) or 'Found' in str(args) or 'Extract' in str(args) or 'Entry IDs' in str(args):
                return
            original_print(*args, **kwargs)
        builtins.print = silent_print
    
    # Run evaluation
    output.write("\n" + "=" * 80 + "\n")
    output.write("RUNNING EVALUATION\n")
    output.write("=" * 80 + "\n")
    
    all_meeting_recalls = []
    all_summary_recalls = []
    
    for i, test_case in enumerate(test_queries, 1):
        query_id = test_case.get('id', i)
        query = test_case['query']
        expected_meeting_ids = normalize_expected_ids(test_case.get('meetLevel_index', []))
        expected_summary_ids = normalize_expected_ids(test_case.get('summary_index', []))
        
        output.write(f"\n{'='*80}\n")
        output.write(f"[Query {query_id}] {query}\n")
        output.write(f"{'='*80}\n")
        
        # Perform search
        try:
            results = retriever.search(
                query=query,
                top_k_metadata=top_k_metadata,
                top_k_summary=top_k_summary,
                top_k_meeting=top_k_meeting
            )
            
            # Format results in hierarchical order
            output.write("\n" + "-" * 80 + "\n")
            output.write("RETRIEVAL RESULTS (by level)\n")
            output.write("-" * 80 + "\n")
            
            # 1. Metadata Level
            format_metadata_level_results(results.metadata_results, output)
            
            # 2. Summary Level
            retrieved_summary_ids, _ = format_summary_level_results(results.summary_results, output)
            
            # 3. Meeting Level
            retrieved_meeting_ids = format_meeting_level_results(results.meeting_results, output)
            
            # Calculate recall
            meeting_recall = calculate_recall(retrieved_meeting_ids, expected_meeting_ids)
            summary_recall = calculate_recall(retrieved_summary_ids, expected_summary_ids)
            
            all_meeting_recalls.append(meeting_recall)
            all_summary_recalls.append(summary_recall)
            
            # Format recall evaluation
            output.write("\n" + "-" * 80 + "\n")
            output.write("RECALL EVALUATION\n")
            output.write("-" * 80 + "\n")
            
            
            output.write(f"\n  Summary Level Recall:\n")
            output.write(f"    Expected Summary IDs: {sorted(expected_summary_ids) if expected_summary_ids else 'None'}\n")
            output.write(f"    Retrieved Summary IDs: {sorted(retrieved_summary_ids) if retrieved_summary_ids else 'None'}\n")
            intersection_summary = retrieved_summary_ids & expected_summary_ids
            output.write(f"    Intersection: {sorted(intersection_summary) if intersection_summary else 'None'}\n")
            output.write(f"    Recall: {summary_recall:.3f} ({len(intersection_summary)}/{len(expected_summary_ids)})\n")
            
            output.write(f"\n  Meeting Level Recall:\n")
            output.write(f"    Expected Entry IDs: {sorted(expected_meeting_ids) if expected_meeting_ids else 'None'}\n")
            output.write(f"    Retrieved Entry IDs: {sorted(retrieved_meeting_ids) if retrieved_meeting_ids else 'None'}\n")
            intersection_meeting = retrieved_meeting_ids & expected_meeting_ids
            output.write(f"    Intersection: {sorted(intersection_meeting) if intersection_meeting else 'None'}\n")
            output.write(f"    Recall: {meeting_recall:.3f} ({len(intersection_meeting)}/{len(expected_meeting_ids)})\n")
            

            
        except Exception as e:
            output.write(f"  Error: {e}\n")
            import traceback
            output.write(traceback.format_exc())
            all_meeting_recalls.append(0.0)
            all_summary_recalls.append(0.0)
    
    # Restore print
    if suppress_output:
        builtins.print = original_print
    
    # Format summary statistics
    output.write("\n" + "=" * 80 + "\n")
    output.write("EVALUATION SUMMARY\n")
    output.write("=" * 80 + "\n")
    
    avg_meeting_recall = sum(all_meeting_recalls) / len(all_meeting_recalls) if all_meeting_recalls else 0.0
    avg_summary_recall = sum(all_summary_recalls) / len(all_summary_recalls) if all_summary_recalls else 0.0
    
    output.write(f"\nMeeting Level Recall:\n")
    output.write(f"  Average: {avg_meeting_recall:.3f}\n")
    output.write(f"  Min: {min(all_meeting_recalls):.3f}\n")
    output.write(f"  Max: {max(all_meeting_recalls):.3f}\n")
    
    output.write(f"\nSummary Level Recall:\n")
    output.write(f"  Average: {avg_summary_recall:.3f}\n")
    output.write(f"  Min: {min(all_summary_recalls):.3f}\n")
    output.write(f"  Max: {max(all_summary_recalls):.3f}\n")
    
    output.write("\n" + "=" * 80 + "\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    # Print only summary to console
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Meeting Level Recall - Average: {avg_meeting_recall:.3f}")
    print(f"  Summary Level Recall - Average: {avg_summary_recall:.3f}")
    print("=" * 80 + "\n")
    
    return {
        'meeting_recalls': all_meeting_recalls,
        'summary_recalls': all_summary_recalls,
        'avg_meeting_recall': avg_meeting_recall,
        'avg_summary_recall': avg_summary_recall,
        'output_file': str(output_file)
    }


def run_evaluation_summary_only(
    query_file: str = "query_test.json",
    top_k_summary: int = 10,
    suppress_output: bool = True
):
    """
    Run evaluation using only summary level retrieval (query -> summary level -> return results).
    This is a simplified evaluation path that skips metadata and meeting levels.
    
    Args:
        query_file: Path to JSON file with test queries
        top_k_summary: Top K for summary level search
        suppress_output: If True, suppress HierarchicalRetriever's print statements
    """
    # Create output directory
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_summary_only_{timestamp}.txt"
    
    # Use StringIO to collect all output
    output = StringIO()
    
    # Load test queries
    query_path = project_root / query_file
    with open(query_path, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    
    output.write("=" * 80 + "\n")
    output.write("SUMMARY-ONLY RETRIEVER EVALUATION\n")
    output.write("=" * 80 + "\n")
    
    # Initialize retriever
    output.write("\n[Initialization] Setting up vector store...\n")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    output.write(f"Loaded {len(meetings)} meetings\n")
    
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
            output.write(f"Added {len(chunks)} chunks to {level} level\n")
    
    retriever = HierarchicalRetriever(vector_store)
    output.write("Setup complete!\n")
    
    # Suppress output if needed
    import builtins
    original_print = builtins.print
    if suppress_output:
        def silent_print(*args, **kwargs):
            # Only suppress prints from hierarchical_retriever
            if 'Searching' in str(args) or 'Found' in str(args) or 'Extract' in str(args) or 'Entry IDs' in str(args):
                return
            original_print(*args, **kwargs)
        builtins.print = silent_print
    
    # Run evaluation
    output.write("\n" + "=" * 80 + "\n")
    output.write("RUNNING EVALUATION (SUMMARY LEVEL ONLY)\n")
    output.write("=" * 80 + "\n")
    
    all_summary_recalls = []
    
    for i, test_case in enumerate(test_queries, 1):
        query_id = test_case.get('id', i)
        query = test_case['query']
        expected_summary_ids = normalize_expected_ids(test_case.get('summary_index', []))
        
        output.write(f"\n{'='*80}\n")
        output.write(f"[Query {query_id}] {query}\n")
        output.write(f"{'='*80}\n")
        
        # Perform search using summary-only method
        try:
            results = retriever.search_summary_only(
                query=query,
                top_k_summary=top_k_summary
            )
            
            # Format results
            output.write("\n" + "-" * 80 + "\n")
            output.write("RETRIEVAL RESULTS (SUMMARY LEVEL ONLY)\n")
            output.write("-" * 80 + "\n")
            
            # Summary Level
            retrieved_summary_ids, _ = format_summary_level_results(results.summary_results, output)
            
            # Calculate recall (only for summary level)
            summary_recall = calculate_recall(retrieved_summary_ids, expected_summary_ids)
            all_summary_recalls.append(summary_recall)
            
            # Format recall evaluation
            output.write("\n" + "-" * 80 + "\n")
            output.write("RECALL EVALUATION\n")
            output.write("-" * 80 + "\n")
            
            output.write(f"\n  Summary Level Recall:\n")
            output.write(f"    Expected Summary IDs: {sorted(expected_summary_ids) if expected_summary_ids else 'None'}\n")
            output.write(f"    Retrieved Summary IDs: {sorted(retrieved_summary_ids) if retrieved_summary_ids else 'None'}\n")
            intersection_summary = retrieved_summary_ids & expected_summary_ids
            output.write(f"    Intersection: {sorted(intersection_summary) if intersection_summary else 'None'}\n")
            output.write(f"    Recall: {summary_recall:.3f} ({len(intersection_summary)}/{len(expected_summary_ids)})\n")
            
        except Exception as e:
            output.write(f"  Error: {e}\n")
            import traceback
            output.write(traceback.format_exc())
            all_summary_recalls.append(0.0)
    
    # Restore print
    if suppress_output:
        builtins.print = original_print
    
    # Format summary statistics
    output.write("\n" + "=" * 80 + "\n")
    output.write("EVALUATION SUMMARY\n")
    output.write("=" * 80 + "\n")
    
    avg_summary_recall = sum(all_summary_recalls) / len(all_summary_recalls) if all_summary_recalls else 0.0
    
    output.write(f"\nSummary Level Recall:\n")
    output.write(f"  Average: {avg_summary_recall:.3f}\n")
    output.write(f"  Min: {min(all_summary_recalls):.3f}\n")
    output.write(f"  Max: {max(all_summary_recalls):.3f}\n")
    
    output.write("\n" + "=" * 80 + "\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    # Print only summary to console
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE (SUMMARY-ONLY)")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Summary Level Recall - Average: {avg_summary_recall:.3f}")
    print("=" * 80 + "\n")
    
    return {
        'summary_recalls': all_summary_recalls,
        'avg_summary_recall': avg_summary_recall,
        'output_file': str(output_file)
    }


def run_evaluation_with_rewriter(
    query_file: str = "query_test.json",
    top_k_vector: int = 5,
    top_k_bm25: int = 20,
    num_paraphrases: int = 2,
    level: str = 'summary',
    suppress_output: bool = True
):
    """
    Run evaluation using query rewriter with multi-query strategy.
    
    Args:
        query_file: Path to JSON file with test queries
        top_k_vector: Top k for each vector query (default: 5)
        top_k_bm25: Top k for BM25 keyword query (default: 20)
        num_paraphrases: Number of paraphrases to use (default: 2)
        level: Level to search ('metadata', 'summary', 'meeting')
        suppress_output: If True, suppress print statements
    """
    # Create output directory
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_with_rewriter_{timestamp}.txt"
    
    # Use StringIO to collect all output
    output = StringIO()
    
    # Load test queries
    query_path = project_root / query_file
    with open(query_path, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    
    output.write("=" * 80 + "\n")
    output.write("MULTI-QUERY RETRIEVER EVALUATION (WITH QUERY REWRITER)\n")
    output.write("=" * 80 + "\n")
    
    # Initialize retriever
    output.write("\n[Initialization] Setting up vector store...\n")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    output.write(f"Loaded {len(meetings)} meetings\n")
    
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings, include_chunk_level=False)
    
    vector_store = HybridSearchVectorStore()
    for level_name in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level_name, [])
        if chunks:
            vector_store.add_chunks(
                chunks=chunks,
                Level=level_name,
                generate_embedding=True
            )
            output.write(f"Added {len(chunks)} chunks to {level_name} level\n")
    
    retriever = HierarchicalRetriever(vector_store)
    
    # Initialize query rewriter
    output.write("\n[Initialization] Setting up query rewriter...\n")
    query_rewriter = QueryRewriter()
    output.write("Setup complete!\n")
    
    # Suppress output if needed
    import builtins
    original_print = builtins.print
    if suppress_output:
        def silent_print(*args, **kwargs):
            # Only suppress prints from hierarchical_retriever and vector_store
            msg = str(args)
            if any(keyword in msg for keyword in ['Searching', 'Found', 'Extract', 'Entry IDs', 'Vector multi-query', 'BM25 keyword query', 'Vector search', 'BM25 search', 'Merged results']):
                return
            original_print(*args, **kwargs)
        builtins.print = silent_print
    
    # Run evaluation
    output.write("\n" + "=" * 80 + "\n")
    output.write(f"RUNNING EVALUATION (MULTI-QUERY, LEVEL: {level.upper()})\n")
    output.write("=" * 80 + "\n")
    
    all_summary_recalls = []
    
    for i, test_case in enumerate(test_queries, 1):
        query_id = test_case.get('id', i)
        original_query = test_case['query']
        expected_summary_ids = normalize_expected_ids(test_case.get('summary_index', []))
        
        output.write(f"\n{'='*80}\n")
        output.write(f"[Query {query_id}] {original_query}\n")
        output.write(f"{'='*80}\n")
        
        # Perform search using query rewriter + multi-query
        try:
            # Step 1: Rewrite query
            output.write("\n[Query Rewriting]\n")
            query_rewrite = query_rewriter.rewrite(original_query)
            
            output.write(f"  Normalized Query: {query_rewrite.get('normalized_query', 'N/A')}\n")
            output.write(f"  Main Clause: {query_rewrite.get('main_clause', 'N/A')}\n")
            
            entities = query_rewrite.get('entities', {})
            people = entities.get('people', [])
            keywords = entities.get('keywords', [])
            output.write(f"  Entities - People: {people}\n")
            output.write(f"  Entities - Keywords: {keywords}\n")
            
            paraphrases = query_rewrite.get('paraphrases', [])
            if paraphrases:
                output.write(f"  Paraphrases ({len(paraphrases)}):\n")
                for j, p in enumerate(paraphrases, 1):
                    output.write(f"    {j}. {p}\n")
            
            # Step 2: Multi-query search
            output.write(f"\n[Multi-Query Retrieval] (level={level}, top_k_vector={top_k_vector}, top_k_bm25={top_k_bm25})\n")
            results = retriever.search_with_rewriter(
                original_query=original_query,
                query_rewrite=query_rewrite,
                level=level,
                top_k_vector=top_k_vector,
                top_k_bm25=top_k_bm25,
                num_paraphrases=num_paraphrases
            )
            
            # Format results
            output.write("\n" + "-" * 80 + "\n")
            output.write(f"RETRIEVAL RESULTS ({level.upper()} LEVEL)\n")
            output.write("-" * 80 + "\n")
            output.write(f"Total chunks retrieved: {len(results)}\n")
            
            if not results:
                output.write("  No results found\n")
                all_summary_recalls.append(0.0)
                continue
            
            # Extract summary_ids from results
            retrieved_summary_ids = set()
            for j, result in enumerate(results, 1):
                chunk_id = result.get('chunk_id', 'N/A')
                score_vector_norm = result.get('score_vector_norm', 0.0)  # 使用归一化分数
                score_bm25_norm = result.get('score_bm25_norm', 0.0)  # 使用归一化分数
                hybrid_score = result.get('hybrid_score', 0.0)
                
                metadata = result.get('metadata', {})
                summary_ids_list = metadata.get('summary_ids', [])
                if isinstance(summary_ids_list, str):
                    summary_ids_list = [summary_ids_list] if summary_ids_list else []
                elif not isinstance(summary_ids_list, list):
                    summary_ids_list = []
                
                retrieved_summary_ids.update(summary_ids_list)
                
                output.write(f"\n  [{j}] Chunk ID: {chunk_id}\n")
                output.write(f"      Score Vector (norm): {score_vector_norm:.4f}\n")
                output.write(f"      Score BM25 (norm): {score_bm25_norm:.4f}\n")
                output.write(f"      Hybrid Score: {hybrid_score:.4f}\n")
                if summary_ids_list:
                    output.write(f"      Summary IDs: {sorted(summary_ids_list)}\n")
            
            # Calculate recall (only for summary level)
            summary_recall = calculate_recall(retrieved_summary_ids, expected_summary_ids)
            all_summary_recalls.append(summary_recall)
            
            # Format recall evaluation
            output.write("\n" + "-" * 80 + "\n")
            output.write("RECALL EVALUATION\n")
            output.write("-" * 80 + "\n")
            
            output.write(f"\n  Summary Level Recall:\n")
            output.write(f"    Expected Summary IDs: {sorted(expected_summary_ids) if expected_summary_ids else 'None'}\n")
            output.write(f"    Retrieved Summary IDs: {sorted(retrieved_summary_ids) if retrieved_summary_ids else 'None'}\n")
            intersection_summary = retrieved_summary_ids & expected_summary_ids
            output.write(f"    Intersection: {sorted(intersection_summary) if intersection_summary else 'None'}\n")
            output.write(f"    Recall: {summary_recall:.3f} ({len(intersection_summary)}/{len(expected_summary_ids)})\n")
            
        except Exception as e:
            output.write(f"  Error: {e}\n")
            import traceback
            output.write(traceback.format_exc())
            all_summary_recalls.append(0.0)
    
    # Restore print
    if suppress_output:
        builtins.print = original_print
    
    # Format summary statistics
    output.write("\n" + "=" * 80 + "\n")
    output.write("EVALUATION SUMMARY\n")
    output.write("=" * 80 + "\n")
    
    avg_summary_recall = sum(all_summary_recalls) / len(all_summary_recalls) if all_summary_recalls else 0.0
    
    output.write(f"\nSummary Level Recall:\n")
    output.write(f"  Average: {avg_summary_recall:.3f}\n")
    output.write(f"  Min: {min(all_summary_recalls):.3f}\n")
    output.write(f"  Max: {max(all_summary_recalls):.3f}\n")
    
    output.write("\n" + "=" * 80 + "\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    # Print only summary to console
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE (MULTI-QUERY WITH REWRITER)")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Summary Level Recall - Average: {avg_summary_recall:.3f}")
    print(f"  Configuration: level={level}, top_k_vector={top_k_vector}, top_k_bm25={top_k_bm25}")
    print("=" * 80 + "\n")
    
    return {
        'summary_recalls': all_summary_recalls,
        'avg_summary_recall': avg_summary_recall,
        'output_file': str(output_file)
    }

if __name__ == "__main__":
    # Option 1: Original summary-only evaluation
    # results = run_evaluation_summary_only(
    #     query_file="query_test.json",
    #     top_k_summary=10,
    #     suppress_output=True
    # )
    
    # Option 2: Multi-query evaluation with query rewriter
    results = run_evaluation_with_rewriter(
        query_file="query_test.json",
        top_k_vector=5,
        top_k_bm25=20,
        num_paraphrases=2,
        level='summary',
        suppress_output=True
    )