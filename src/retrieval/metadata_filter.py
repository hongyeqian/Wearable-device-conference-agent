"""
Metadata Filter Module
Filters metadata-level results based on time constraints from query rewrite.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import HybridSearchVectorStore


class MetadataFilter:
    """
    Filter metadata by time constraints extracted from query rewrite.
    """
    
    def __init__(self, vector_store: HybridSearchVectorStore):
        """
        Initialize metadata filter.
        
        Args:
            vector_store: HybridSearchVectorStore instance for metadata search
        """
        self.vector_store = vector_store
    
    def filter_by_time(
        self,
        query_rewrite: Dict[str, Any],
        top_k_metadata: int = 20
    ) -> Set[str]:
        """
        Filter metadata by time constraints from query_rewrite.
        
        Args:
            query_rewrite: Query rewrite result containing time entities
            top_k_metadata: Top k metadata results to retrieve before filtering
            
        Returns:
            Set of meeting_ids that match the time constraints.
            Returns empty set if no time constraints found or no matching meetings.
        """
        # Extract time information from query_rewrite
        entities = query_rewrite.get('entities', {})
        time_expressions = entities.get('time', [])
        
        if not time_expressions:
            # No time constraint, return empty set (will search all meetings)
            return set()
        
        # Parse time expressions - they should already be in format like "2025-11-29"
        target_dates = []
        date_range = None
        
        for time_expr in time_expressions:
            time_expr = time_expr.strip()
            
            # Check if it's a date range (e.g., "2025-11-08 to 2025-11-14")
            if ' to ' in time_expr:
                try:
                    start_str, end_str = time_expr.split(' to ')
                    start_date = datetime.strptime(start_str.strip(), '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_str.strip(), '%Y-%m-%d').date()
                    date_range = (start_date, end_date)
                    break  # Use the first valid date range
                except ValueError:
                    continue
            else:
                # Single date
                try:
                    date_obj = datetime.strptime(time_expr, '%Y-%m-%d')
                    target_dates.append(date_obj.date())
                except ValueError:
                    continue
        
        if not target_dates and not date_range:
            # Could not parse any time, return empty set
            print("Metadata filter: Could not parse time expressions, returning empty set")
            return set()
        
        # Search metadata level
        # Use normalized_query or fallback to main_clause/keywords
        search_query = query_rewrite.get('normalized_query', '')
        if not search_query:
            search_query = query_rewrite.get('main_clause', '')
            if not search_query:
                keywords = entities.get('keywords', [])
                search_query = ' '.join(keywords)
        
        metadata_results = self.vector_store.search(
            query_text=search_query,
            Level='metadata',
            top_k=top_k_metadata
        )
        
        if not metadata_results:
            print("Metadata filter: No metadata results found")
            return set()
        
        print(f"Metadata filter: Found {len(metadata_results)} metadata results")
        
        # Filter metadata results by time
        matching_meeting_ids = set()
        
        for result in metadata_results:
            metadata = result.get('metadata', {})
            meeting_id = result.get('meeting_id')
            chunk_datetime_str = metadata.get('datetime')
            
            if not meeting_id:
                continue
            
            if not chunk_datetime_str or chunk_datetime_str == 'None':
                # If metadata has no datetime, skip it
                continue
            
            try:
                # Parse chunk datetime (format: "2025-11-30T20:21:00+08:00" or "2025-11-30")
                if 'T' in chunk_datetime_str:
                    chunk_date = datetime.fromisoformat(chunk_datetime_str.replace('+08:00', '')).date()
                else:
                    chunk_date = datetime.strptime(chunk_datetime_str, '%Y-%m-%d').date()
                
                # Check if chunk date matches filter
                if date_range:
                    start_date, end_date = date_range
                    if start_date <= chunk_date <= end_date:
                        matching_meeting_ids.add(meeting_id)
                elif target_dates:
                    if chunk_date in target_dates:
                        matching_meeting_ids.add(meeting_id)
            except (ValueError, AttributeError):
                # If parsing fails, skip this metadata
                continue
        
        if matching_meeting_ids:
            sorted_ids = sorted(list(matching_meeting_ids))
            print(f"Metadata filter: Found {len(matching_meeting_ids)} meetings matching time constraints")
            print(f"  Meeting IDs: {sorted_ids[:10]}{'...' if len(sorted_ids) > 10 else ''}")
        else:
            print("Metadata filter: No meetings match the time constraints")
        
        return matching_meeting_ids
