"""
占位符解析和替换工具
用于识别和替换查询中的占位符
"""
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import dateparser


# 占位符正则表达式
PLACEHOLDER_PATTERNS = {
    "current_user": r"\[current_user\]",
    "person": r"\[person:([^\]]+)\]",
    "date": r"\[date:([^\]]+)\]",
    "last_n": r"\[last_n:(\d+)\]",
}


class PlaceholderResolver:
    """占位符解析器"""
    
    def __init__(self, current_user: str = "Hongye Qian"):
        self.current_user = current_user
        self._resolved: Dict[str, str] = {}
    
    def extract_placeholders(self, query: str) -> List[Tuple[str, str]]:
        """
        提取查询中的所有占位符
        
        Returns:
            [(placeholder_type, placeholder_value), ...]
        """
        results = []
        
        # current_user
        for match in re.finditer(PLACEHOLDER_PATTERNS["current_user"], query, re.IGNORECASE):
            results.append(("current_user", ""))
        
        # person
        for match in re.finditer(PLACEHOLDER_PATTERNS["person"], query, re.IGNORECASE):
            results.append(("person", match.group(1)))
        
        # date
        for match in re.finditer(PLACEHOLDER_PATTERNS["date"], query, re.IGNORECASE):
            results.append(("date", match.group(1)))
        
        # last_n
        for match in re.finditer(PLACEHOLDER_PATTERNS["last_n"], query, re.IGNORECASE):
            results.append(("last_n", match.group(1)))
        
        return results
    
    def has_placeholders(self, query: str) -> bool:
        """检查查询是否包含占位符"""
        return len(self.extract_placeholders(query)) > 0
    
    def resolve_current_user(self) -> str:
        """解析 current_user 占位符"""
        return self.current_user
    
    def resolve_person(self, fuzzy_name: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """
        解析 person 占位符
        
        Args:
            fuzzy_name: 模糊输入
            candidates: 候选列表（从 pandas 查询得到）
            threshold: 相似度阈值
            
        Returns:
            匹配的人名或 None
        """
        if not candidates:
            return None
        
        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0]
        
        # 模糊匹配
        from thefuzz import fuzz
        from thefuzz import process
        
        matches = process.extract(
            fuzzy_name,
            candidates,
            scorer=fuzz.token_sort_ratio,
            limit=1
        )
        
        if matches and matches[0][1] / 100 >= threshold:
            return matches[0][0]
        
        return None
    
    def resolve_date(self, date_str: str) -> Optional[str]:
        """
        解析 date 占位符
        
        Args:
            date_str: 如 "yesterday", "last week", "November", "2025-11-30"
            
        Returns:
            ISO 格式日期字符串或 None
        """
        # 先尝试直接解析
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        
        # 尝试解析 "last N days/weeks/months"
        if "last" in date_str.lower():
            match = re.search(r"last\s+(\d+)\s+(day|week|month|year)", date_str.lower())
            if match:
                n = int(match.group(1))
                unit = match.group(2)
                now = datetime.now()
                
                if unit == "day":
                    delta = timedelta(days=n)
                elif unit == "week":
                    delta = timedelta(weeks=n)
                elif unit == "month":
                    delta = timedelta(days=n * 30)
                elif unit == "year":
                    delta = timedelta(days=n * 365)
                else:
                    return None
                
                result = now - delta
                return result.strftime("%Y-%m-%d")
        
        return None
    
    def resolve_last_n(self, n: int, meeting_dates: List[str]) -> List[str]:
        """
        解析 last_n 占位符
        
        Args:
            n: 数量
            meeting_dates: 会议日期列表（已排序，从新到旧）
            
        Returns:
            最近的 N 个日期
        """
        return meeting_dates[:n]
    
    def replace_placeholder(self, query: str, placeholder: str, value: str) -> str:
        """替换单个占位符"""
        # 转义特殊字符用于正则
        escaped_placeholder = re.escape(placeholder)
        return re.sub(escaped_placeholder, value, query, flags=re.IGNORECASE)
    
    def resolve_all(self, query: str, context: Dict) -> str:
        """
        解析所有占位符
        
        Args:
            query: 包含占位符的查询
            context: 上下文，包含人名候选、日期等
            
        Returns:
            解析后的查询
        """
        result = query
        
        # 1. 替换 current_user
        if self.has_placeholders(result):
            result = self.replace_placeholder(
                result, 
                "[current_user]", 
                self.current_user
            )
        
        # 2. 替换 person
        if self.has_placeholders(result):
            for placeholder, value in self.extract_placeholders(result):
                if placeholder == "person":
                    candidates = context.get("person_candidates", [])
                    resolved = self.resolve_person(value, candidates)
                    if resolved:
                        result = self.replace_placeholder(result, f"[person:{value}]", resolved)
        
        # 3. 替换 date
        if self.has_placeholders(result):
            for placeholder, value in self.extract_placeholders(result):
                if placeholder == "date":
                    resolved = self.resolve_date(value)
                    if resolved:
                        result = self.replace_placeholder(result, f"[date:{value}]", resolved)
        
        # 4. 替换 last_n
        if self.has_placeholders(result):
            meeting_dates = context.get("meeting_dates", [])
            for placeholder, value in self.extract_placeholders(result):
                if placeholder == "last_n":
                    n = int(value)
                    resolved_dates = self.resolve_last_n(n, meeting_dates)
                    if resolved_dates:
                        dates_str = ", ".join(resolved_dates)
                        result = self.replace_placeholder(result, f"[last_n:{n}]", dates_str)
        
        return result


# ============ 时间解析辅助函数 ============

def parse_relative_date(date_str: str, reference_now: datetime = None) -> Optional[datetime]:
    """
    解析相对时间表达
    
    Args:
        date_str: 如 "yesterday", "3 days ago", "last week"
        reference_now: 参考时间（默认当前时间）
        
    Returns:
        datetime 对象或 None
    """
    if reference_now is None:
        reference_now = datetime.now()
    
    # 使用 dateparser
    parsed = dateparser.parse(
        date_str,
        settings={
            'RELATIVE_BASE': reference_now,
            'PREFER_DAY_OF_MONTH': 'first',
        }
    )
    
    return parsed


def parse_meeting_date(date_str: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    解析会议日期表达
    
    Returns:
        (year, month, None) 或 (year, month, day)
    """
    # 尝试完整日期
    parsed = dateparser.parse(date_str)
    if parsed:
        return (parsed.year, parsed.month, parsed.day)
    
    # 尝试年月
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
    
    date_str_lower = date_str.lower().strip()
    
    # 年月
    for month_name, month_num in month_names.items():
        if month_name in date_str_lower:
            # 提取年份
            year_match = re.search(r"(20\d{2})", date_str)
            if year_match:
                return (int(year_match.group(1)), month_num, None)
            else:
                # 使用当前年份
                return (datetime.now().year, month_num, None)
    
    return (None, None, None)


# ============ 测试代码 ============
if __name__ == "__main__":
    resolver = PlaceholderResolver()
    
    # 测试提取
    query = "What did [current_user] discuss in [last_n:3]?"
    print("Query:", query)
    print("Placeholders:", resolver.extract_placeholders(query))
    
    # 测试解析
    print("\nResolve 'yesterday':", resolver.resolve_date("yesterday"))
    print("Resolve 'last week':", resolver.resolve_date("last week"))
    print("Resolve 'November':", resolver.resolve_date("November"))
    
    # 测试完整替换
    context = {
        "person_candidates": ["Hongye Qian", "Ankit Kumar"],
        "meeting_dates": ["2025-11-30", "2025-11-29", "2025-11-28", "2025-11-27"]
    }
    
    query2 = "What did [person:hongye] discuss in [last_n:3]?"
    result = resolver.resolve_all(query2, context)
    print("\nOriginal:", query2)
    print("Resolved:", result)
