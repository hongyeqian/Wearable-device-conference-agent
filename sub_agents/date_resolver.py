"""
Date parsing utility
Uses dateparser for robust date parsing
"""
from typing import Optional
from datetime import datetime
import dateparser


class DateResolver:
    """Date parser/resolver"""
    
    def __init__(self, reference_now: datetime = None):
        """
        Args:
            reference_now: Reference time (defaults to current time)
        """
        self.reference_now = reference_now or datetime.now()
    
    def resolve(self, date_str: str) -> Optional[str]:
        """
        Parse date string to ISO format date
        
        Args:
            date_str: Date string, e.g., "yesterday", "last week", "November", "2025-11-30"
            
        Returns:
            ISO format date string (YYYY-MM-DD) or None
        """
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Use dateparser for parsing
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        
        return None


# ============ Test Code ============
if __name__ == "__main__":
    resolver = DateResolver()
    
    test_dates = [
        "yesterday",
        "3 days ago",
        "last week",
        "last month",
        "November",
        "November 2025",
        "2025-11-30",
        "last 3 meetings",
    ]
    
    print("Reference now:", resolver.reference_now)
    print()
    
    for date_str in test_dates:
        result = resolver.resolve(date_str)
        print(f"'{date_str}' -> {result}")
