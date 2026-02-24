"""
时间解析工具
使用 dateparser 进行鲁棒的时间解析
"""
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import dateparser
import re


class DateResolver:
    """时间解析器"""
    
    def __init__(self, reference_now: datetime = None):
        """
        Args:
            reference_now: 参考时间（默认当前时间）
        """
        self.reference_now = reference_now or datetime.now()
    
    def resolve(self, date_str: str) -> Optional[str]:
        """
        解析时间字符串为 ISO 日期
        
        Args:
            date_str: 时间字符串，如 "yesterday", "last week", "November", "2025-11-30"
            
        Returns:
            ISO 格式日期字符串 (YYYY-MM-DD) 或 None
        """
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # 1. 尝试直接解析 ISO 格式
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        
        # 2. 尝试相对时间表达
        result = self._resolve_relative(date_str)
        if result:
            return result
        
        # 3. 尝试月份名
        result = self._resolve_month(date_str)
        if result:
            return result
        
        return None
    
    def _resolve_relative(self, date_str: str) -> Optional[str]:
        """解析相对时间"""
        date_lower = date_str.lower()
        
        # "last N days/weeks/months/years"
        match = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", date_lower)
        if match:
            n = int(match.group(1))
            unit = match.group(2)
            
            if unit in ["day", "days"]:
                delta = timedelta(days=n)
            elif unit in ["week", "weeks"]:
                delta = timedelta(weeks=n)
            elif unit in ["month", "months"]:
                delta = timedelta(days=n * 30)
            elif unit in ["year", "years"]:
                delta = timedelta(days=n * 365)
            else:
                return None
            
            result = self.reference_now - delta
            return result.strftime("%Y-%m-%d")
        
        # "the last N meetings" - 特殊处理，由 pandas 查询处理
        if "last" in date_lower and "meeting" in date_lower:
            # 返回特殊标记，让调用方用 pandas 查询
            return f"[last_n:{re.search(r'(\d+)', date_lower).group(1)}]"
        
        return None
    
    def _resolve_month(self, date_str: str) -> Optional[str]:
        """解析月份名"""
        month_names = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        
        date_lower = date_str.lower().strip()
        
        for month_name, month_num in month_names.items():
            if month_name in date_lower:
                # 提取年份
                year_match = re.search(r"(20\d{2})", date_str)
                year = int(year_match.group(1)) if year_match else self.reference_now.year
                
                # 返回月份第一天
                return f"{year}-{month_num:02d}-01"
        
        return None
    
    def is_relative(self, date_str: str) -> bool:
        """判断是否为相对时间"""
        relative_patterns = [
            r"yesterday",
            r"today",
            r"tomorrow",
            r"\d+\s+days?\s+ago",
            r"last\s+\d+\s+(day|week|month|year)",
            r"last\s+(day|week|month|year)",
            r"previous",
            r"recent",
        ]
        
        date_lower = date_str.lower()
        for pattern in relative_patterns:
            if re.search(pattern, date_lower):
                return True
        
        return False
    
    def needs_year(self, date_str: str) -> bool:
        """判断是否需要补充年份"""
        # 月份名需要年份
        month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
        
        date_lower = date_str.lower().strip()
        
        # 只有月份名，没有年份
        for month in month_names:
            if month in date_lower and "20" not in date_str:
                return True
        
        return False


def get_meeting_dates_from_query(query: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    从查询中提取日期信息
    
    Returns:
        (year, month, day) - 任何值为 None 表示未指定
    """
    # 尝试解析完整日期
    parsed = dateparser.parse(query)
    if parsed:
        return (parsed.year, parsed.month, parsed.day)
    
    # 尝试解析年月
    resolver = DateResolver()
    date_str = resolver._resolve_month(query)
    if date_str:
        parts = date_str.split("-")
        return (int(parts[0]), int(parts[1]), None)
    
    return (None, None, None)


# ============ 测试代码 ============
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
        is_rel = resolver.is_relative(date_str)
        needs_yr = resolver.needs_year(date_str)
        print(f"'{date_str}' -> {result}")
        print(f"  is_relative: {is_rel}, needs_year: {needs_yr}")
