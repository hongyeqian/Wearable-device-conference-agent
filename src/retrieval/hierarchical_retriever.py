from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

import sys
from pathlib import Path
project_root =  Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from src.retrieval.vector_store import HybridSearchVectorStore

@dataclass
class HierarchicalResults:
    """
    this is the container for the hierarchical retrieval results
    """
    
    metadata_results: List[Dict[str, Any]] = field(default_factory=list)
    summary_results: List[Dict[str, Any]] = field(default_factory=list)
    meeting_results: List[Dict[str, Any]] = field(default_factory=list)
    
    candidate_meeting_ids: Set[str] = field(default_factory= set)
    candidate_entry_ids: Set[str] = field(default_factory= set)
    
                            
class HierarchicalRetriever:
    
    def __init__(
        self, 
        vector_store: HybridSearchVectorStore
    ):
        self.vector_store = vector_store
    
        
    def _extract_ids_from_summaries(self, summary_results: List[Dict[str, Any]]):
        entry_ids = set()
        
        for result in summary_results:
            metadata = result.get('metadata', {})
            references = metadata.get('references', [])
            
            if isinstance(references, list):
                for ref in references:
                    if isinstance(ref, str) and ref.strip():
                        entry_ids.add(ref.strip())
                        
            elif isinstance(references, str) and references.strip():
                entry_ids.add(references.strip())
                        
                        
        return entry_ids
    
    
    
    def search(self, query,top_k_metadata:int=5, top_k_summary: int =10, top_k_meeting:int =10) -> HierarchicalResults:
        results = HierarchicalResults()
        
        # firstly, we search the metadata level
        print(f"\n Searching metadata level (top_k={top_k_metadata})..")
        
        results.metadata_results = self.vector_store.search(
            query_text= query,
            Level= 'metadata',
            top_k= top_k_metadata
        )
        
        for result in results.metadata_results:
            meeting_id = result.get('meeting_id')
            
            if meeting_id:
                results.candidate_meeting_ids.add(meeting_id)   #data001
                
        print(f"Found {len(results.metadata_results)} metadata results")
        print(f"Extract {len(results.candidate_meeting_ids)} candidate meeting ids")
        
        if results.candidate_meeting_ids:
            sorted_meeting_ids = sorted(list(results.candidate_meeting_ids))
            print(f"  Meeting IDs: {sorted_meeting_ids[:10]}{'...' if len(sorted_meeting_ids) > 10 else ''}")
        else:
            print("No meeting IDs found in metadata results!")
            return results
        
        results.summary_results = self.vector_store.search_with_meeting_ids_filter(
            query_text= query,
            level= 'summary',
            meeting_ids=results.candidate_meeting_ids,
            top_k=top_k_summary
        )

        
        print(f"Found {len(results.summary_results)} summary results")
        
        results.candidate_entry_ids = self._extract_ids_from_summaries(results.summary_results)
        print(f"Extracted {len(results.candidate_entry_ids)} candidates entries")
        
        
        if results.candidate_entry_ids:
            sorted_entry_ids = sorted(list(results.candidate_entry_ids))
            print(f"Entry IDs: {sorted_entry_ids[:10]}{'...' if len(sorted_entry_ids) > 10 else ''}")
        else:
            print("No entry IDs found in summary references!")
            print("  This might indicate missing references in summary chunks.")
            return results 
        
        # use entry_ids to filter meeting level
        print(f"\nSearching meeting level filtered by entry IDs (top_k={top_k_meeting})...")
        results.meeting_results = self.vector_store.search_with_entry_ids_filter(
            query_text=query,
            level='meeting',
            entry_ids=results.candidate_entry_ids,
            top_k=top_k_meeting
        )
        
        print(f"Found {len(results.meeting_results)} meeting results")
                   
        return results
    
    
    def search_summary_only(self, query: str, top_k_summary: int = 10) -> HierarchicalResults:
        results = HierarchicalResults()
        
        # Directly search summary level without any filtering
        print(f"\nSearching summary level directly (top_k={top_k_summary})...")
        
        results.summary_results = self.vector_store.search(
            query_text=query,
            Level='summary',
            top_k=top_k_summary
        )
        
        print(f"Found {len(results.summary_results)} summary results")
        
        return results
        
        
    def search_with_rewriter(
        self,
        original_query: str,
        query_rewrite: Dict[str, Any],
        level: str = 'summary',
        top_k_vector: int = 5,
        top_k_bm25: int = 20,
        num_paraphrases: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search using query rewriter with multi-query strategy.
        
        Args:
            original_query: Original user query
            query_rewrite: Query rewrite result from QueryRewriter
            level: Level to search ('metadata', 'summary', 'meeting')
            top_k_vector: Top k for each vector query
            top_k_bm25: Top k for BM25 keyword query
            num_paraphrases: Number of paraphrases to use
            
        Returns:
            List of search results with score_vector and score_bm25
        """
        # Perform multi-query search (no time filtering)
        return self.vector_store.search_multi_query(
            original_query=original_query,
            query_rewrite=query_rewrite,
            level=level,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            num_paraphrases=num_paraphrases
        )