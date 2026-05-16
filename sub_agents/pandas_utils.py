"""
Flat DataFrame utility
Converts meeting metadata into a flat pandas DataFrame for easy querying
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import DATA_DIR


class MeetingsDataFrame:
    """Flat DataFrame for meeting metadata"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load all summary_metadata.json from DATA_DIR and convert to flat DataFrame"""
        data_dir = Path(DATA_DIR)
        records = []
        
        for user_dir in data_dir.glob("*"):
            if not user_dir.is_dir(): continue
            
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
                            "owner": user_dir.name,  # The username from directory structure
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
                        
                        records.append(record)
                        
                    except Exception as e:
                        print(f"Warning: Failed to load {summary_meta_file}: {e}")
        
        self.df = pd.DataFrame(records)
        # Sort by date (newest first)
        if not self.df.empty:
            self.df = self.df.sort_values("date", ascending=False).reset_index(drop=True)
    
    def _extract_date(self, datetime_str: str) -> str:
        """Extract date from ISO datetime string"""
        if not datetime_str:
            return ""
        try:
            # Handle timezone
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
    
    def _get_df_for_user(self, current_user: Optional[str] = None) -> pd.DataFrame:
        """Get DataFrame filtered by user owner"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        if not current_user:
            return self.df
        return self.df[self.df["owner"] == current_user]

    def get_all_participants(self, current_user: Optional[str] = None) -> List[str]:
        """Get all unique participants for the given user"""
        df = self._get_df_for_user(current_user)
        if df.empty:
            return []
        all_parts = []
        for parts in df["participants"]:
            if isinstance(parts, list):
                all_parts.extend(parts)
        return list(set(all_parts))
    
    def find_person(self, fuzzy_name: str, current_user: Optional[str] = None, threshold: float = 0.6) -> List[str]:
        """
        Fuzzy match person name
        
        Args:
            fuzzy_name: Fuzzy input (e.g., "Hongye", "hq", "he")
            current_user: The authenticated user to filter matches by
            threshold: Similarity threshold
            
        Returns:
            List of matched participants
        """
        df = self._get_df_for_user(current_user)
        if df.empty:
            return []
        
        from thefuzz import fuzz
        from thefuzz import process
        
        all_participants = self.get_all_participants(current_user)
        
        # Use thefuzz for fuzzy matching
        matches = process.extract(
            fuzzy_name, 
            all_participants, 
            scorer=fuzz.token_sort_ratio,
            limit=3
        )
        
        # Filter matches below threshold
        result = []
        for match_name, score in matches:
            if score / 100 >= threshold:
                result.append(match_name)
        
        return result
    
    def get_last_n_meetings(self, n: int = 3, person_name: Optional[str] = None, current_user: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the last N meetings
        
        Args:
            n: Number of meetings to return
            person_name: Optional. If provided, returns only meetings attended by this person
            current_user: Optional. If provided, filters by meeting owner
        
        Returns:
            List of meetings (sorted by date, newest first)
        """
        df_to_use = self._get_df_for_user(current_user)
        if df_to_use.empty:
            return []
        
        # If person_name is provided, filter by person first
        if person_name:
            df_to_use = self.filter_by_person(person_name, current_user=current_user)
            if df_to_use.empty:
                return []
        
        # Get the last N meetings
        n = min(n, len(df_to_use))
        result = df_to_use.head(n).to_dict("records")
        return result
    
    def filter_by_person(self, person_name: str, current_user: Optional[str] = None) -> pd.DataFrame:
        """Filter meetings by person name"""
        df = self._get_df_for_user(current_user)
        if df.empty:
            return pd.DataFrame()
        
        # Check if person name is in participants list
        mask = df["participants"].apply(
            lambda x: person_name in x if isinstance(x, list) else False
        )
        return df[mask]
    
    def filter_by_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None, current_user: Optional[str] = None) -> pd.DataFrame:
        """Filter by date range"""
        df = self._get_df_for_user(current_user)
        if df.empty:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(df))
        
        if start_date:
            mask &= df["date"] >= start_date
        if end_date:
            mask &= df["date"] <= end_date
        
        return df[mask]
    
    def filter_by_year_month(self, year: Optional[int] = None, month: Optional[int] = None, current_user: Optional[str] = None) -> pd.DataFrame:
        """Filter by year and/or month"""
        df = self._get_df_for_user(current_user)
        if df.empty:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(df))
        
        if year:
            mask &= df["year"] == year
        if month:
            mask &= df["month"] == month
        
        return df[mask]
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts"""
        if self.df is None:
            return []
        return self.df.to_dict("records")


# Global instance
_meetings_df: Optional[MeetingsDataFrame] = None


def get_meetings_df() -> MeetingsDataFrame:
    """Get global MeetingsDataFrame instance"""
    global _meetings_df
    if _meetings_df is None:
        _meetings_df = MeetingsDataFrame()
    return _meetings_df


def reload_meetings_df() -> MeetingsDataFrame:
    """Force-rebuild the global MeetingsDataFrame singleton.

    Call this after new meetings have been indexed so that the pandas
    metadata cache (used by query rewriter, person matcher, etc.)
    reflects the latest data on disk (summary_metadata.json files).

    Returns:
        The newly created MeetingsDataFrame instance.
    """
    global _meetings_df
    _meetings_df = MeetingsDataFrame()
    return _meetings_df


# ============ Test Code ============
if __name__ == "__main__":
    mdf = get_meetings_df()
    print(f"Loaded {len(mdf.df)} meetings")
    print("\nAll participants:")
    print(mdf.get_all_participants())
    print("\nLast 3 meetings:")
    for m in mdf.get_last_n_meetings(3):
        print(f"  {m['date']}: {m['meeting_id']}")
    
    print("\nFind 'Hongye':")
    print(mdf.find_person("Hongye"))
    
    print("\nFilter by person 'Hongye Qian':")
    print(mdf.filter_by_person("Hongye Qian")[["meeting_id", "date", "participants_str"]])
