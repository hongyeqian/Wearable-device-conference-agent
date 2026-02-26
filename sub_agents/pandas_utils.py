"""
扁平化 DataFrame 工具
用于将会议元数据转换为扁平化的 pandas DataFrame，方便查询
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import DATA_DIR


class MeetingsDataFrame:
    """会议元数据的扁平化 DataFrame"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """从 DATA_DIR 加载所有 summary_metadata.json 并转换为扁平化 DataFrame"""
        data_dir = Path(DATA_DIR)
        records = []
        
        for con_dir in sorted(data_dir.glob("con*")):
            if not con_dir.is_dir():
                continue
            
            summary_meta_file = con_dir / "summary_metadata.json"
            if summary_meta_file.exists():
                try:
                    with open(summary_meta_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 扁平化
                    record = {
                        "meeting_id": metadata.get("meeting_id", ""),
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
                    
                    # 处理 actions
                    actions = metadata.get("actions", [])
                    action_tasks = [a.get("task", "") for a in actions if a.get("task")]
                    record["action_tasks"] = action_tasks
                    record["action_tasks_str"] = "; ".join(action_tasks)
                    
                    records.append(record)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {summary_meta_file}: {e}")
        
        self.df = pd.DataFrame(records)
        # 按日期排序（从新到旧）
        if not self.df.empty:
            self.df = self.df.sort_values("date", ascending=False).reset_index(drop=True)
    
    def _extract_date(self, datetime_str: str) -> str:
        """从 ISO datetime 字符串提取日期"""
        if not datetime_str:
            return ""
        try:
            # 处理时区
            dt = datetime_str.replace("+08:00", "").replace("Z", "")
            if "T" in dt:
                return dt.split("T")[0]
            return dt[:10]
        except:
            return ""
    
    def _extract_year(self, datetime_str: str) -> Optional[int]:
        """提取年份"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[0])
            except:
                pass
        return None
    
    def _extract_month(self, datetime_str: str) -> Optional[int]:
        """提取月份"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[1])
            except:
                pass
        return None
    
    def _extract_day(self, datetime_str: str) -> Optional[int]:
        """提取日"""
        date = self._extract_date(datetime_str)
        if date:
            try:
                return int(date.split("-")[2])
            except:
                pass
        return None
    
    def get_all_participants(self) -> List[str]:
        """获取所有参与者（去重）"""
        if self.df is None or self.df.empty:
            return []
        all_parts = []
        for parts in self.df["participants"]:
            if isinstance(parts, list):
                all_parts.extend(parts)
        return list(set(all_parts))
    
    def find_person(self, fuzzy_name: str, threshold: float = 0.6) -> List[str]:
        """
        模糊匹配人名
        
        Args:
            fuzzy_name: 模糊输入（如 "Hongye", "hq", "he"）
            threshold: 相似度阈值
            
        Returns:
            匹配的参与者列表
        """
        if self.df is None or self.df.empty:
            return []
        
        from thefuzz import fuzz
        from thefuzz import process
        
        all_participants = self.get_all_participants()
        
        # 使用 thefuzz 进行模糊匹配
        matches = process.extract(
            fuzzy_name, 
            all_participants, 
            scorer=fuzz.token_sort_ratio,
            limit=3
        )
        
        # 过滤低于阈值的匹配
        result = []
        for match_name, score in matches:
            if score / 100 >= threshold:
                result.append(match_name)
        
        return result
    
    def get_last_n_meetings(self, n: int = 3) -> List[Dict[str, Any]]:
        """获取最近的 N 个会议"""
        if self.df is None or self.df.empty:
            return []
        
        n = min(n, len(self.df))
        result = self.df.head(n).to_dict("records")
        return result
    
    def filter_by_person(self, person_name: str) -> pd.DataFrame:
        """按人名筛选会议"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # 检查 participants 列表中是否包含该人名
        mask = self.df["participants"].apply(
            lambda x: person_name in x if isinstance(x, list) else False
        )
        return self.df[mask]
    
    def filter_by_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """按日期范围筛选"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(self.df))
        
        if start_date:
            mask &= self.df["date"] >= start_date
        if end_date:
            mask &= self.df["date"] <= end_date
        
        return self.df[mask]
    
    def filter_by_year_month(self, year: Optional[int] = None, month: Optional[int] = None) -> pd.DataFrame:
        """按年月筛选"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(self.df))
        
        if year:
            mask &= self.df["year"] == year
        if month:
            mask &= self.df["month"] == month
        
        return self.df[mask]
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        if self.df is None:
            return []
        return self.df.to_dict("records")


# 全局实例
_meetings_df: Optional[MeetingsDataFrame] = None


def get_meetings_df() -> MeetingsDataFrame:
    """获取全局 MeetingsDataFrame 实例"""
    global _meetings_df
    if _meetings_df is None:
        _meetings_df = MeetingsDataFrame()
    return _meetings_df


# ============ 测试代码 ============
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
