"""
Native Python Metadata Manager
Replaces the old pandas_utils.py, providing in-memory native Python dictionary filtering
for meeting metadata, ensuring fast, dependency-free, and strict user-isolation.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import DATA_DIR


class MeetingsMetadataManager:
    """Native Python Dictionary manager for meeting metadata, scoped to a specific user"""
    
    def __init__(self, user_name: str):
        self.user_name = user_name
        self.records: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load summary_metadata.json only from the user's specific data directory"""
        data_dir = Path(DATA_DIR)
        user_dir = data_dir / self.user_name
        
        self.records = []
        
        if not user_dir.exists() or not user_dir.is_dir():
            print(f"Warning: User directory {user_dir} does not exist.")
            return

        # Scan con* directories within the user's folder
        for con_dir in user_dir.glob("con*"):
            if not con_dir.is_dir(): continue
            
            summary_meta_file = con_dir / "summary_metadata.json"
            if summary_meta_file.exists():
                try:
                    with open(summary_meta_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Flatten metadata
                    record = {
                        "meeting_id": metadata.get("meeting_id", ""),
                        "owner": self.user_name,
                        "datetime": metadata.get("datetime", ""),
                        "date": self._extract_date(metadata.get("datetime", "")),
                        "year": self._extract_year(metadata.get("datetime", "")),
                        "month": self._extract_month(metadata.get("datetime", "")),
                        "day": self._extract_day(metadata.get("datetime", "")),
                        "participants": metadata.get("participants", []),
                        "participants_str": ", ".join(metadata.get("participants", [])),
                        "topics": metadata.get("topics", []),
                        "topics_str": "; ".join(metadata.get("topics", [])),
                    }
                    
                    # Process actions
                    actions = metadata.get("actions", [])
                    action_tasks = [a.get("task", "") for a in actions if a.get("task")]
                    record["action_tasks"] = action_tasks
                    record["action_tasks_str"] = "; ".join(action_tasks)
                    
                    self.records.append(record)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {summary_meta_file}: {e}")
        
        # Sort by date (newest first)
        self.records.sort(key=lambda x: x["date"], reverse=True)
    
    def _extract_date(self, datetime_str: str) -> str:
        """Extract date from ISO datetime string"""
        if not datetime_str:
            return ""
        try:
            dt = datetime_str.replace("+08:00", "").replace("Z", "")
            if "T" in dt:
                return dt.split("T")[0]
            return dt[:10]
        except:
            return ""
    
    def _extract_year(self, datetime_str: str) -> Optional[int]:
        """Extract year"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[0])
            except:
                pass
        return None
    
    def _extract_month(self, datetime_str: str) -> Optional[int]:
        """Extract month"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[1])
            except:
                pass
        return None
    
    def _extract_day(self, datetime_str: str) -> Optional[int]:
        """Extract day"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[2])
            except:
                pass
        return None

    def get_all_participants(self) -> List[str]:
        """Get all unique participants for the loaded user"""
        all_parts = []
        for r in self.records:
            parts = r.get("participants", [])
            if isinstance(parts, list):
                all_parts.extend(parts)
        return list(set(all_parts))
    
    def find_person(self, fuzzy_name: str, threshold: float = 0.6) -> List[str]:
        """
        Fuzzy match person name within the user's participant list
        """
        if not self.records:
            return []
        
        from thefuzz import fuzz
        from thefuzz import process
        
        all_participants = self.get_all_participants()
        if not all_participants:
            return []
            
        matches = process.extract(
            fuzzy_name, 
            all_participants, 
            scorer=fuzz.token_sort_ratio,
            limit=3
        )
        
        result = []
        for match_name, score in matches:
            if score / 100 >= threshold:
                result.append(match_name)
        
        return result
    
    def filter_by_person(self, person_name: str) -> List[Dict[str, Any]]:
        """Filter meetings by person name for the current user"""
        return [r for r in self.records if person_name in r.get("participants", [])]
    
    def get_last_n_meetings(self, n: int = 3, person_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the last N meetings for the current user
        """
        records_to_use = self.records
        
        # If person_name is provided, filter by person first
        if person_name:
            records_to_use = self.filter_by_person(person_name)
            
        return records_to_use[:n]
    
    def filter_by_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter meetings by date range for the current user"""
        result = []
        for r in self.records:
            date_val = r.get("date", "")
            if not date_val:
                continue
            
            valid = True
            if start_date and date_val < start_date:
                valid = False
            if end_date and date_val > end_date:
                valid = False
                
            if valid:
                result.append(r)
        return result
    
    def filter_by_year_month(self, year: Optional[int] = None, month: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter meetings by year and/or month for the current user"""
        result = []
        for r in self.records:
            valid = True
            if year is not None and r.get("year") != year:
                valid = False
            if month is not None and r.get("month") != month:
                valid = False
                
            if valid:
                result.append(r)
        return result


# Global manager cache: {user_name: MeetingsMetadataManager}
_manager_cache: Dict[str, MeetingsMetadataManager] = {}

def get_meetings_metadata(user_name: str) -> MeetingsMetadataManager:
    """Get or create a MeetingsMetadataManager instance for the specific user"""
    global _manager_cache
    if user_name not in _manager_cache:
        _manager_cache[user_name] = MeetingsMetadataManager(user_name)
    return _manager_cache[user_name]

def reload_meetings_metadata(user_name: str) -> MeetingsMetadataManager:
    """Force-rebuild the metadata manager for the specific user."""
    global _manager_cache
    _manager_cache[user_name] = MeetingsMetadataManager(user_name)
    return _manager_cache[user_name]

