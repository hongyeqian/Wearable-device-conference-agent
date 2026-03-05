"""
Hierarchical Retriever for the RAG system
Retrieves documents using a three-level hierarchical approach: metadata -> summary -> meeting
"""
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from src.retrieval.vector_store import HybridSearchVectorStore

@dataclass
class HierarchicalResults:
    """
    Container for hierarchical retrieval results.
    Holds results from each level of the retrieval hierarchy.
    """
    
    metadata_results: List[Dict[str, Any]] = field(default_factory=list)
    summary_results: List[Dict[str, Any]] = field(default_factory=list)
    meeting_results: List[Dict[str, Any]] = field(default_factory=list)
    
    candidate_meeting_ids: Set[str] = field(default_factory=set)
    candidate_entry_ids: Set[str] = field(default_factory=set)
    
                            
class HierarchicalRetriever:
    
    def __init__(
        self, 
        vector_store: HybridSearchVectorStore
    ):
        self.vector_store = vector_store
    
        
    def _extract_ids_from_summaries(self, summary_results: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract entry IDs from summary results.
        
        Args:
            summary_results: List of summary search results
            
        Returns:
            Set of entry IDs extracted from summary metadata references
        """
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
    
    
    
    
    def search(self, query, top_k_metadata: int = 5, top_k_summary: int = 10, top_k_meeting: int = 10) -> HierarchicalResults:
        """
        Perform hierarchical search across three levels.
        
        Args:
            query: Query text
            top_k_metadata: Number of results to retrieve from metadata level
            top_k_summary: Number of results to retrieve from summary level
            top_k_meeting: Number of results to retrieve from meeting level
            
        Returns:
            HierarchicalResults containing results from all levels
        """
        results = HierarchicalResults()
        
        # First, search the metadata level
        print(f"\n Searching metadata level (top_k={top_k_metadata})..")
        
        results.metadata_results = self.vector_store.search(
            query_text= query,
            level= 'metadata',
            top_k= top_k_metadata
        )
        
        for result in results.metadata_results:
            meeting_id = result.get('meeting_id')
            
            if meeting_id:
                # Collect candidate meeting IDs for next level filtering
                results.candidate_meeting_ids.add(meeting_id)
                
        print(f"Found {len(results.metadata_results)} metadata results")
        print(f"Extracted {len(results.candidate_meeting_ids)} candidate meeting IDs")
        
        if results.candidate_meeting_ids:
            sorted_meeting_ids = sorted(list(results.candidate_meeting_ids))
            print(f"  Meeting IDs: {sorted_meeting_ids[:10]}{'...' if len(sorted_meeting_ids) > 10 else ''}")
        else:
            print("No meeting IDs found in metadata results!")
            return results
        
        # Search summary level with meeting IDs filter
        results.summary_results = self.vector_store.search_with_meeting_ids_filter(
            query_text= query,
            level= 'summary',
            meeting_ids=results.candidate_meeting_ids,
            top_k=top_k_summary
        )

        
        print(f"Found {len(results.summary_results)} summary results")
        
        # Extract entry IDs from summary results for meeting level filtering
        results.candidate_entry_ids = self._extract_ids_from_summaries(results.summary_results)
        print(f"Extracted {len(results.candidate_entry_ids)} candidate entry IDs")
        
        
        if results.candidate_entry_ids:
            sorted_entry_ids = sorted(list(results.candidate_entry_ids))
            print(f"Entry IDs: {sorted_entry_ids[:10]}{'...' if len(sorted_entry_ids) > 10 else ''}")
        else:
            print("No entry IDs found in summary references!")
            print("  This might indicate missing references in summary chunks.")
            return results 
        
        # Using entry IDs to filter meeting level search
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
        """
        Search only the summary level without hierarchical filtering.
        
        Args:
            query: Query text
            top_k_summary: Number of results to retrieve from summary level
            
        Returns:
            HierarchicalResults containing summary results
        """
        results = HierarchicalResults()
        
        # Directly search summary level without any filtering
        print(f"\nSearching summary level directly (top_k={top_k_summary})...")
        
        results.summary_results = self.vector_store.search(
            query_text=query,
            level='summary',
            top_k=top_k_summary
        )
        
        print(f"Found {len(results.summary_results)} summary results")
        
        return results
