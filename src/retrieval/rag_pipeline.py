"""
RAG Pipeline
Unified pipeline for query rewriting, hierarchical retrieval, and answer generation.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hierarchical_retriever import HierarchicalRetriever, HierarchicalResults
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.answer_generator import AnswerGenerator


class RAGPipeline:
    """Complete RAG pipeline with query rewriting, retrieval, and answer generation"""
    
    def __init__(
        self,
        retriever: HierarchicalRetriever,
        answer_generator: Optional[AnswerGenerator] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        use_query_rewriter: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: HierarchicalRetriever instance
            answer_generator: AnswerGenerator instance (creates default if None)
            query_rewriter: QueryRewriter instance (creates default if None and use_query_rewriter=True)
            use_query_rewriter: Whether to use query rewriting (default: True)
        """
        self.retriever = retriever
        self.use_query_rewriter = use_query_rewriter
        
        if answer_generator is None:
            self.answer_generator = AnswerGenerator()
        else:
            self.answer_generator = answer_generator
        
        if use_query_rewriter:
            if query_rewriter is None:
                self.query_rewriter = QueryRewriter()
            else:
                self.query_rewriter = query_rewriter
        else:
            self.query_rewriter = None
    

    
    def query(
        self,
        user_query: str,
        top_k_vector: int =5,
        top_k_bm25: int =20,
        num_paraphrases: int=2,
        max_chunks_for_answer: int = 15,
        use_rewritten_query: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: rewrite query, retrieve, and generate answer.
        
        Args:
            user_query: Original user question
            top_k_vector: Top k for each vector query (default: 5)
            top_k_bm25: Top k for BM25 keyword query (default: 20)
            num_paraphrases: Number of paraphrases to use (default: 2)
            max_chunks_for_answer: Maximum chunks to use for answer generation (default: 15)
            use_rewritten_query: Whether to use rewritten query for retrieval (default: True)
            
        Returns:
            Dictionary containing:
            - answer: Generated answer text
            - original_query: Original user query
            - rewritten_query: Rewritten query (if used)
            - query_rewrite_data: Full query rewrite data (if used)
            - retrieval_results: HierarchicalResults object
            - chunks_used: List of chunks used for answer
            - num_chunks: Number of chunks used
        """
        # Query rewriting
        query_rewrite_data = None

        if self.use_query_rewriter and self.query_rewriter:
            try:
                query_rewrite_data = self.query_rewriter.rewrite(user_query)
            except Exception as e:
                print(f"Warning: Query rewriting failed: {e}")
                # If rewriting fails, create a minimal rewrite structure
                query_rewrite_data = {
                    'normalized_query': user_query,
                    'paraphrases': [],
                    'entities': {}
                }
        else:
            # If rewriter is disabled, create minimal structure
            query_rewrite_data = {
                'normalized_query': user_query,
                'paraphrases': [],
                'entities': {}
            }
        
        # Hierarchical retrieval
        retrieval_results = self.retriever.search_with_rewriter(
            original_query=user_query,
            query_rewrite=query_rewrite_data,
            level='summary',
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            num_paraphrases=num_paraphrases
        )
        
        chunks_with_level = []
        for chunk in retrieval_results:
            chunk['level'] = 'summary'
            chunks_with_level.append(chunk)
            
        chunks_with_level.sort(
            key= lambda x: x.get('hybrid_score', 0),
            reverse= True
        )
        selected_chunks = chunks_with_level[:max_chunks_for_answer]
        
        # Generate answer
        # Use normalized_query if it contains date conversion, otherwise use original query
        query_for_llm = user_query
        if query_rewrite_data:
            normalized = query_rewrite_data.get('normalized_query', '')
            # Check if normalized query has date conversion (contains YYYY-MM-DD format)
            import re
            if re.search(r'\d{4}-\d{2}-\d{2}', normalized):
                # Normalized query has specific date, use it instead of original
                query_for_llm = normalized

        answer_result = self.answer_generator.generate_answer(
            query=query_for_llm,  # Use normalized query with date conversion
            chunks=selected_chunks,
            max_chunks=max_chunks_for_answer,
            include_citations=True
        )
        
        return {
            'answer': answer_result['answer'],
            'original_query': user_query,
            'rewritten_query': query_rewrite_data.get('normalized_query') if query_rewrite_data and query_rewrite_data.get('normalized_query') != user_query else None,
            'query_rewrite_data': query_rewrite_data,
            'retrieval_results': retrieval_results,
            'chunks_used': answer_result['chunks_used'],
            'num_chunks': answer_result['num_chunks']
        }
