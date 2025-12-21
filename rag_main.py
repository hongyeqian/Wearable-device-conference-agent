"""
RAG System Main Entry Point
Complete RAG pipeline with query rewriting, hierarchical retrieval, and answer generation.
"""


import os
# clean useless environment variable SSL_CERT_FILE
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]
        print(f"Warning: Removed invalid SSL_CERT_FILE: {cert_path}")


import sys
import argparse
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import DataLoader
from src.chunking.chunker import HierarchicalChunker
from src.retrieval.vector_store import HybridSearchVectorStore
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.retrieval.rag_pipeline import RAGPipeline
from config.settings import DATA_DIR
from src.retrieval import query_rewriter_muti_agent

# 全局会议目录字符串，由 rag_main 初始化时填充
MEETING_CATALOG: str = ""


def set_meetings(meetings: List[Any]) -> None:
    """
    由 rag_main 在系统初始化时调用，
    把 loader.load_all_meetings() 的结果转换成一个简短的“会议目录”文本，
    用于提供给 LLM 作为上下文。
    """
    global MEETING_CATALOG
    lines: List[str] = []

    for m in meetings or []:
        # datetime
        dt_raw = getattr(m, "datetime", None)
        date_str = ""
        if isinstance(dt_raw, datetime):
            date_str = dt_raw.date().isoformat()
        elif isinstance(dt_raw, str):
            try:
                # 支持 "2025-11-30T20:21:00+08:00" 或 "2025-11-30"
                text = dt_raw.strip().replace("+08:00", "")
                if "T" in text:
                    d = datetime.fromisoformat(text).date()
                else:
                    d = datetime.strptime(text, "%Y-%m-%d").date()
                date_str = d.isoformat()
            except Exception:
                pass

        # participants（只拿 name）
        names: List[str] = []
        for p in getattr(m, "participants", []):
            if isinstance(p, dict):
                name = (p.get("name") or "").strip()
            else:
                name = str(p).strip()
            if name:
                names.append(name)

        if date_str and names:
            lines.append(f"- {date_str}: " + ", ".join(names))

    MEETING_CATALOG = "\n".join(lines)


def save_chunks_to_file(
    query: str,
    chunks_used: List[Dict[str, Any]],
    output_dir: Path = None,
    save_original_rank: bool = False
) -> List[Path]:
    """
    Save the chunks used for answer generation to a text file.
    Uses citation_number from chunks if available to match LLM citations.
    
    Args:
        query: The user query
        chunks_used: List of chunks that were used for answer generation
        output_dir: Output directory (defaults to project_root/outputs/circle_result)
        save_original_rank: If True, also saves a version with chunks in original relevance order
        
    Returns:
        List of paths to saved files
    """
    if output_dir is None:
        project_root = Path(__file__).parent
        output_dir = project_root / "outputs" / "circle_result"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename: timestamp + query hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')
    
    saved_files = []
    
    # Always save grouped version (current format)
    filename = f"chunks_{timestamp}_{query_hash}_{safe_query}.txt"
    output_file = output_dir / filename
    _write_chunks_file(output_file, query, chunks_used, "GROUPED BY MEETING")
    saved_files.append(output_file)
    
    # Optionally save original rank version
    if save_original_rank:
        filename_original = f"chunks_{timestamp}_{query_hash}_{safe_query}_original_rank.txt"
        output_file_original = output_dir / filename_original
        _write_chunks_file(output_file_original, query, chunks_used, "ORIGINAL RANK (BY RELEVANCE)", use_original_order=True)
        saved_files.append(output_file_original)
    
    return saved_files

def _write_chunks_file(
    output_file: Path,
    query: str,
    chunks_used: List[Dict[str, Any]],
    format_type: str,
    use_original_order: bool = False
):
    """Helper function to write chunks to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHUNKS USED FOR ANSWER GENERATION\n")
        f.write(f"Format: {format_type}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Query: {query}\n")
        f.write(f"Number of chunks: {len(chunks_used)}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Sort chunks based on format type
        if use_original_order:
            # Use original order (chunks should already be sorted by relevance)
            sorted_chunks = chunks_used
        else:
            # Sort by citation_number if available (grouped format)
            sorted_chunks = sorted(
                chunks_used,
                key=lambda x: x.get('citation_number', 999)
            )
        
        for chunk in sorted_chunks:
            # Use citation_number if available, otherwise use index
            citation_num = chunk.get('citation_number')
            if citation_num is None:
                citation_num = sorted_chunks.index(chunk) + 1
            
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK #[{citation_num}]\n")
            f.write(f"{'='*80}\n\n")
            
            # Basic info
            f.write(f"Citation Number: [{citation_num}]\n")
            f.write(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}\n")
            f.write(f"Meeting ID: {chunk.get('meeting_id', 'N/A')}\n")
            f.write(f"Level: {chunk.get('level', 'N/A')}\n")
            
            # Scores
            if 'hybrid_score' in chunk:
                f.write(f"Hybrid Score: {chunk.get('hybrid_score', 0):.4f}\n")
            if 'score_vector' in chunk:
                f.write(f"Vector Score: {chunk.get('score_vector', 0):.4f}\n")
            if 'score_bm25' in chunk:
                f.write(f"BM25 Score: {chunk.get('score_bm25', 0):.4f}\n")
            
            # Metadata
            metadata = chunk.get('metadata', {})
            if metadata:
                f.write(f"\nMetadata:\n")
                if 'title' in metadata:
                    f.write(f"  Title: {metadata.get('title', 'N/A')}\n")
                if 'datetime' in metadata:
                    f.write(f"  Date: {metadata.get('datetime', 'N/A')}\n")
                if 'summary_ids' in metadata:
                    summary_ids = metadata.get('summary_ids', [])
                    if summary_ids:
                        f.write(f"  Summary IDs: {', '.join(summary_ids) if isinstance(summary_ids, list) else summary_ids}\n")
                if 'references' in metadata:
                    references = metadata.get('references', [])
                    if references:
                        f.write(f"  References: {', '.join(references) if isinstance(references, list) else references}\n")
                if 'entry_id' in metadata:
                    f.write(f"  Entry ID: {metadata.get('entry_id', 'N/A')}\n")
                if 'reference' in metadata:
                    f.write(f"  Reference: {metadata.get('reference', 'N/A')}\n")
            
            # Text content
            f.write(f"\nText Content:\n")
            f.write("-" * 80 + "\n")
            text = chunk.get('text', '')
            f.write(text)
            f.write("\n" + "-" * 80 + "\n")
            f.write("\n")


def initialize_rag_system(
    data_dir: Path = None,
    include_chunk_level: bool = False
) -> RAGPipeline:
    """
    Initialize the complete RAG system.
    
    Args:
        data_dir: Path to data directory (defaults to DATA_DIR from config)
        include_chunk_level: Whether to include chunk level (default: False)
        
    Returns:
        Initialized RAGPipeline instance
    """
    
    if data_dir is None:
        data_dir = DATA_DIR
        
    if data_dir is None:
        raise ValueError('data_dir is none.')
    
    
        
    print("=" * 80)
    print("Initializing RAG System")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[Step 1] Loading meeting data...")
    loader = DataLoader(data_dir)
    meetings = loader.load_all_meetings()
    print(f"Loaded {len(meetings)} meetings")

    if not meetings:
        raise ValueError("No meetings found. Please check your data directory.")

    # ★ 把 meetings 列表注册给 rewriter，用于 prompt 里的 MEETING CATALOG
    #query_rewriter.set_meetings(meetings)
    query_rewriter_muti_agent.set_meetings(meetings)

    # Step 2: Chunk meetings
    print("\n[Step 2] Chunking meetings...")
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings, include_chunk_level=include_chunk_level)
    
    for level, chunks in all_chunks.items():
        print(f"  {level:10s}: {len(chunks):4d} chunks")
    
    # Step 3: Initialize vector store
    print("\n[Step 3] Initializing vector store...")
    vector_store = HybridSearchVectorStore()
    
    # Step 4: Add chunks to vector store
    print("\n[Step 4] Adding chunks to vector store...")
    for level in ['metadata', 'summary', 'meeting']:
        chunks = all_chunks.get(level, [])
        if chunks:
            vector_store.add_chunks(
                chunks=chunks,
                Level=level,
                generate_embedding=True
            )
            print(f"Added {len(chunks)} chunks to {level} level")
    
    # Step 5: Create retriever
    print("\n[Step 5] Creating hierarchical retriever...")
    retriever = HierarchicalRetriever(vector_store)
    
    # Step 6: Create RAG pipeline
    print("\n[Step 6] Creating RAG pipeline...")
    from src.retrieval.answer_generator_muti_agent import AnswerGeneratorMultiAgent
    answer_generator = AnswerGeneratorMultiAgent()
    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
        use_query_rewriter=True
    )
    
    print("\n" + "=" * 80)
    print("RAG System Initialized Successfully!")
    print("=" * 80)
    
    return pipeline


def interactive_mode(pipeline: RAGPipeline):
    """Run RAG system in interactive mode."""
    # Flag to control whether to save original rank version
    save_original_rank = False
    
    print("\n" + "=" * 80)
    print("Interactive RAG Query Mode")
    print("=" * 80)
    print("Commands:")
    print("  - Enter a question to get an answer")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'config' to show current configuration")
    print("  - 'save_original_rank' to toggle saving original rank version (currently: OFF)")
    print("=" * 80)
    
    while True:
        try:
            query = input("\n> Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'config':
                print("\nCurrent Configuration:")
                print(f"  Query Rewriter: {'Enabled' if pipeline.use_query_rewriter else 'Disabled'}")
                print(f"  Save Original Rank: {'ON' if save_original_rank else 'OFF'}")
                continue
            
            if query.lower() == 'save_original_rank':
                save_original_rank = not save_original_rank
                status = "ON" if save_original_rank else "OFF"
                print(f"\n✅ Save Original Rank: {status}")
                print("   (When ON, both grouped and original rank versions will be saved)")
                continue
            
            # Process query
            print("\n" + "-" * 80)
            print(f"Processing query: {query}")
            print("-" * 80)
            
            result = pipeline.query(
                user_query=query,
                top_k_vector=5,
                top_k_bm25=20,
                num_paraphrases=2,
                max_chunks_for_answer=15
            )
            
            # Save chunks to file
            try:
                saved_files = save_chunks_to_file(
                    query, 
                    result['chunks_used'],
                    save_original_rank=save_original_rank
                )
                for file_path in saved_files:
                    print(f"\n💾 Saved chunks to: {file_path}")
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to save chunks: {e}")
            
            # Display results
            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(result['answer'])
            
            if result['rewritten_query']:
                print(f"\n(Query rewritten from: '{result['original_query']}' to: '{result['rewritten_query']}')")
            
            print(f"\n(Used {result['num_chunks']} chunks from retrieval)")
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


def single_query_mode(
    pipeline: RAGPipeline,
    query: str,
    top_k_vector: int = 5,
    top_k_bm25: int = 20,
    num_paraphrases: int = 2,
    max_chunks: int = 15,
    verbose: bool = False
):
    """Run RAG system for a single query."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    result = pipeline.query(
        user_query=query,
        top_k_vector=top_k_vector,
        top_k_bm25=top_k_bm25,
        num_paraphrases=num_paraphrases,
        max_chunks_for_answer=max_chunks
    )
    
    # Save chunks to file
    try:
        output_files = save_chunks_to_file(query, result['chunks_used'])
        print(f"\nSaved chunks to: {', '.join([str(f) for f in output_files])}")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save chunks: {e}")
    
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result['answer'])
    print("=" * 80)
    
    if verbose:
        if result['rewritten_query']:
            print(f"\nRewritten Query: {result['rewritten_query']}")
        print(f"\nChunks Used: {result['num_chunks']}")
        print(f"Retrieval Results: {len(result['retrieval_results'])} chunks retrieved")
        
        # Show query rewrite details if available
        if result['query_rewrite_data']:
            rewrite_data = result['query_rewrite_data']
            print(f"\nQuery Rewrite Details:")
            print(f"  Normalized Query: {rewrite_data.get('normalized_query', 'N/A')}")
            print(f"  Main Clause: {rewrite_data.get('main_clause', 'N/A')}")
            if rewrite_data.get('paraphrases'):
                print(f"  Paraphrases: {', '.join(rewrite_data['paraphrases'][:3])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for Meeting Transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rag_main.py

  # Single query
  python rag_main.py --query "What are the action items?"

  # Single query with custom parameters
  python rag_main.py --query "Who mentioned RAG?" --top-k-vector 10 --top-k-bm25 30

  # Verbose output
  python rag_main.py --query "What was discussed?" --verbose
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Path to data directory (defaults to config DATA_DIR)'
    )
    
    parser.add_argument(
        '--top-k-vector',
        type=int,
        default=5,
        help='Top K for each vector query (default: 5)'
    )
    
    parser.add_argument(
        '--top-k-bm25',
        type=int,
        default=20,
        help='Top K for BM25 keyword query (default: 20)'
    )
    
    parser.add_argument(
        '--num-paraphrases',
        type=int,
        default=2,
        help='Number of paraphrases to use (default: 2)'
    )
    
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=15,
        help='Maximum chunks to use for answer generation (default: 15)'
    )
    
    parser.add_argument(
        '--no-rewriter',
        action='store_true',
        help='Disable query rewriter'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--include-chunk-level',
        action='store_true',
        help='Include chunk level (not recommended)'
    )
    
    parser.add_argument(
        '--save-original-rank',
        action='store_true',
        help='Save a version of chunks in original relevance order'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        data_dir = Path(args.data_dir) if args.data_dir else None
        pipeline = initialize_rag_system(
            data_dir=data_dir,
            include_chunk_level=args.include_chunk_level
        )
        
        if args.no_rewriter:
            pipeline.use_query_rewriter = False
        
        # Run in appropriate mode
        if args.query:
            single_query_mode(
                pipeline=pipeline,
                query=args.query,
                top_k_vector=args.top_k_vector,
                top_k_bm25=args.top_k_bm25,
                num_paraphrases=args.num_paraphrases,
                max_chunks=args.max_chunks,
                verbose=args.verbose,
                save_original_rank=args.save_original_rank
            )
        else:
            interactive_mode(pipeline)
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
