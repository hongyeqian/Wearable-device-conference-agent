"""
人名模糊匹配工具
使用 Presidio + TheFuzz 进行鲁棒的人名匹配
"""
from typing import List, Optional, Dict, Any
from thefuzz import fuzz, process


# 名字别名表 - 用于处理同一人的不同写法
NAME_ALIASES: Dict[str, List[str]] = {
    "Hongye Qian": ["Hongye", "Qian", "hq", "hqian", "H Qian", "HY Qian"],
    "Ankit Kumar": ["Ankit", "Kumar", "ak", "akumar", "A Kumar"],
}


class PersonMatcher:
    """人名模糊匹配器"""
    
    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold: 相似度阈值，低于此值认为不匹配
        """
        self.threshold = threshold
    
    def normalize_name(self, name: str) -> str:
        """
        标准化名字
        
        Args:
            name: 原始名字
            
        Returns:
            标准化后的名字
        """
        # 去除首尾空格
        name = name.strip()
        
        # 检查别名表
        for canonical, aliases in NAME_ALIASES.items():
            if name.lower() in [a.lower() for a in aliases]:
                return canonical
        
        return name
    
    def is_ambiguous(self, name: str, candidates: List[str]) -> bool:
        """
        判断名字是否模糊（需要进一步解析）
        
        Args:
            name: 待检查的名字
            candidates: 候选名字列表
            
        Returns:
            True 如果名字模糊
        """
        # 常见模糊词
        ambiguous_words = {"he", "she", "they", "him", "her", "them", "i", "me", "we", "us"}
        
        name_lower = name.lower().strip()
        
        # 是模糊词
        if name_lower in ambiguous_words:
            return True
        
        # 在候选列表中找不到
        if name not in candidates:
            # 尝试标准化后查找
            normalized = self.normalize_name(name)
            if normalized not in candidates:
                return True
        
        return False
    
    def find_match(self, fuzzy_name: str, candidates: List[str]) -> Optional[str]:
        """
        找到最佳匹配
        
        Args:
            fuzzy_name: 模糊输入
            candidates: 候选列表
            
        Returns:
            匹配的名字或 None
        """
        if not candidates:
            return None
        
        # 先标准化输入
        normalized = self.normalize_name(fuzzy_name)
        
        # 如果标准化后在候选中，直接返回
        if normalized in candidates:
            return normalized
        
        # 使用 thefuzz 进行模糊匹配
        matches = process.extract(
            normalized,
            candidates,
            scorer=fuzz.token_sort_ratio,
            limit=3
        )
        
        # 检查阈值
        for match_name, score in matches:
            if score / 100 >= self.threshold:
                return match_name
        
        return None
    
    def find_all_matches(self, fuzzy_name: str, candidates: List[str]) -> List[str]:
        """
        找到所有匹配（多个可能）
        
        Args:
            fuzzy_name: 模糊输入
            candidates: 候选列表
            
        Returns:
            所有匹配的名字列表
        """
        if not candidates:
            return []
        
        normalized = self.normalize_name(fuzzy_name)
        
        matches = process.extract(
            normalized,
            candidates,
            scorer=fuzz.token_sort_ratio,
            limit=5
        )
        
        result = []
        for match_name, score in matches:
            if score / 100 >= self.threshold:
                result.append(match_name)
        
        return result
    
    def resolve_pronoun(self, pronoun: str, context_participants: List[str]) -> Optional[str]:
        """
        解析代词
        
        Args:
            pronoun: 代词 (he, she, they 等)
            context_participants: 上下文中的参与者列表
            
        Returns:
            解析后的人名或 None
        """
        pronoun_lower = pronoun.lower().strip()
        
        # 如果只有一个参与者，直接返回
        if len(context_participants) == 1:
            return context_participants[0]
        
        # 多参与者时，代词需要根据上下文判断
        # 这里暂时返回 None，让 LLM 判断
        return None


# ============ 测试代码 ============
if __name__ == "__main__":
    matcher = PersonMatcher()
    
    # 候选列表
    candidates = ["Hongye Qian", "Ankit Kumar", "Satya Nadella", "Elon Musk", "Nimi Mehta"]
    
    print("Candidates:", candidates)
    
    # 测试模糊匹配
    test_names = ["Hongye", "Qian", "Ankit", "he", "Elon", "Unknown"]
    
    for name in test_names:
        match = matcher.find_match(name, candidates)
        is_amb = matcher.is_ambiguous(name, candidates)
        print(f"\n'{name}' -> match: {match}, ambiguous: {is_amb}")
