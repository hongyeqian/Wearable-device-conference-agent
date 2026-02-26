"""
时间解析工具
使用 dateparser 进行鲁棒的时间解析
"""
from typing import Optional, Tuple
from datetime import datetime
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
        
        # 使用 dateparser 解析
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        
        return None
    
    # def is_relative(self, date_str: str) -> bool:
    #     """判断是否为相对时间"""
    #     relative_patterns = [
    #         r"yesterday",
    #         r"today",
    #         r"tomorrow",
    #         r"\d+\s+days?\s+ago",
    #         r"last\s+\d+\s+(day|week|month|year)",
    #         r"last\s+(day|week|month|year)",
    #         r"previous",
    #         r"recent",
    #     ]
        
    #     date_lower = date_str.lower()
    #     for pattern in relative_patterns:
    #         if re.search(pattern, date_lower):
    #             return True
        
    #     return False
    
    # def needs_year(self, date_str: str) -> bool:
    #     """判断是否需要补充年份"""
    #     # 月份名需要年份
    #     month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
        
    #     date_lower = date_str.lower().strip()
        
    #     # 只有月份名，没有年份
    #     for month in month_names:
    #         if month in date_lower and "20" not in date_str:
    #             return True
        
    #     return False


# def get_meeting_dates_from_query(query: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
#     """
#     从查询中提取日期信息
    
#     Returns:
#         (year, month, day) - 任何值为 None 表示未指定
#     """
#     # 使用 dateparser 解析
#     parsed = dateparser.parse(query)
#     if parsed:
#         return (parsed.year, parsed.month, parsed.day)
    
#     return (None, None, None)


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
        # is_rel = resolver.is_relative(date_str)
        # needs_yr = resolver.needs_year(date_str)
        print(f"'{date_str}' -> {result}")
        print(f"  is_relative: {is_rel}, needs_year: {needs_yr}")
