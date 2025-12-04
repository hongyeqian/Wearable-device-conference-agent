"""
Query Rewriter Module
Rewrites user queries into structured format for better retrieval performance.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from pathlib import Path
import sys


import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, DATA_DIR
from src.data_loader.loader import DataLoader

# 全局会议目录字符串，由 rag_main 初始化时填充
MEETING_CATALOG: str = ""


def set_meetings(meetings: List[Any]) -> None:
    """
    由 rag_main 在系统初始化时调用，
    把 loader.load_all_meetings() 的结果转换成一个简短的“会议目录”文本，
    用于提供给 LLM 作为上下文。
    """
    global MEETING_CATALOG
    lines: List[str] = []

    for m in meetings or []:
        # datetime
        dt_raw = getattr(m, "datetime", None)
        date_str = ""
        if isinstance(dt_raw, datetime):
            date_str = dt_raw.date().isoformat()
        elif isinstance(dt_raw, str):
            try:
                # 支持 "2025-11-30T20:21:00+08:00" 或 "2025-11-30"
                text = dt_raw.strip().replace("+08:00", "")
                if "T" in text:
                    d = datetime.fromisoformat(text).date()
                else:
                    d = datetime.strptime(text, "%Y-%m-%d").date()
                date_str = d.isoformat()
            except Exception:
                pass

        # participants（只拿 name）
        names: List[str] = []
        for p in getattr(m, "participants", []):
            if isinstance(p, dict):
                name = (p.get("name") or "").strip()
            else:
                name = str(p).strip()
            if name:
                names.append(name)

        if date_str and names:
            lines.append(f"- {date_str}: " + ", ".join(names))

    MEETING_CATALOG = "\n".join(lines)


# Pydantic models for structured output
class Entities(BaseModel):
    """Entity extraction structure"""
    people: List[str] = Field(default_factory=list, description="List of people mentioned")
    organizations: List[str] = Field(default_factory=list, description="List of organizations mentioned")
    products_or_projects: List[str] = Field(default_factory=list, description="List of products or projects mentioned")
    events_or_meetings: List[str] = Field(default_factory=list, description="List of events or meetings mentioned")
    time: List[str] = Field(default_factory=list, description="Time-related expressions, with relative times converted to specific dates (e.g., 'yesterday' -> '2025-01-15', 'last week' -> '2025-01-08 to 2025-01-14')")
    keywords: List[str] = Field(default_factory=list, description="Key topic phrases or keywords")


class QueryRewrite(BaseModel):
    """Complete query rewrite structure"""
    normalized_query: str = Field(description="Normalized question keeping core meaning but removing surface details")
    main_clause: str = Field(description="Short phrase describing what information is being requested")
    details: str = Field(description="Time range, speakers, events, locations, or other constraints")
    entities: Entities = Field(description="Extracted entities grouped by category")
    paraphrases: List[str] = Field(description="2-3 paraphrased questions with same meaning but different wording")


# System prompt for query rewriting
QUERY_REWRITE_PROMPT_TEMPLATE = """You are a query rewriting assistant for a meeting-transcript RAG system.

CURRENT DATE AND TIME CONTEXT:
Current Date: {current_date}
Current Time: {current_time}
Day of Week: {day_of_week}

MEETING CATALOG (for resolving time / meeting references):
{meeting_catalog}

Given a user question about a meeting or tech topic, you must:

1. Normalize the question: keep the core meaning but remove surface details
   like explicit instructions ("show", "display"), overly specific dates/IDs,
   and formatting. 
    
   IMPORTANT: If the question contains time constraints (yesterday, last week, etc.),
   include the converted specific date in the normalized query to help retrieval.
   Example: "What did Ankit discuss yesterday?" → 
            "What did Ankit discuss on 2025-11-29?"

2. Extract main clause, details, and structured entities (people, organizations,
   products, events, time expressions, keywords).

3. For MEETING-RELATED time references:
   - When the user mentions or implies a specific meeting 
     (e.g., "last meeting", "previous sync", "最近一次会议", 
      or similar semantics like "their last discussion"),
     you MUST use the MEETING CATALOG above to infer which concrete date it refers to.
   - Choose the MOST RECENT date in the catalog that matches the participants in the question.
   - Put that concrete date into:
       • entities.time (as 'YYYY-MM-DD')
       • normalized_query (replace the relative phrase with 'on YYYY-MM-DD')
       • all paraphrases (make sure the date appears explicitly).

4. If the MEETING CATALOG does not contain any meeting that matches the participants,
   fall back to using the CURRENT DATE AND TIME CONTEXT to interpret relative expressions
   like "yesterday", "last week", etc.

5. Generate 2–3 paraphrased questions that keep the same meaning as the
   original question but use different wording.
   
   IMPORTANT: Include time information in paraphrases when relevant.
   Example: "What topics did Ankit and Hongye cover in their discussion on 2025-11-29?"
"""


class QueryRewriter:
    """Query rewriter using OpenAI structured outputs"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the query rewriter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
            model: OpenAI model name (defaults to OPENAI_MODEL from config, or "gpt-4o-2024-08-06")
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL or "gpt-4o-2024-08-06"
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # for last meeting qustions
        self._meeting_index = None
    
    def _get_current_time_context(self) -> Dict[str, str]:
        """
        Get current date and time context for time resolution.
        
        Returns:
            Dictionary with current_date, current_time, day_of_week
        """
        now = datetime.now()
        return {
            'current_date': now.strftime('%Y-%m-%d'),
            'current_time': now.strftime('%H:%M:%S'),
            'day_of_week': now.strftime('%A')
        }
    
    def _build_prompt_with_time_context(self) -> str:
        """
        Build the prompt with current time context.
        """
        time_context = self._get_current_time_context()
        catalog = MEETING_CATALOG or "No meetings available."
        return QUERY_REWRITE_PROMPT_TEMPLATE.format(
            meeting_catalog=catalog,
            **time_context,
        )
    
    
    
    
    def rewrite(self, user_query: str) -> Dict[str, Any]:
        """
        Rewrite a user query into structured format.
        
        Args:
            user_query: The original user query string
            
        Returns:
            Dictionary containing:
            - normalized_query: str
            - main_clause: str
            - details: str
            - entities: dict with keys (people, organizations, products_or_projects, 
                       events_or_meetings, time, keywords)
            - paraphrases: list of str
        """
        try:
            # Build prompt with current time context
            prompt = self._build_prompt_with_time_context()
            
            # Use OpenAI structured outputs
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_query}
                ],
                text_format=QueryRewrite,
                temperature=self.temperature
            )
            
            # Convert Pydantic model to dict
            rewrite_result = response.output_parsed.model_dump()
            
            return rewrite_result
            
        except Exception as e:
            # Fallback: try with JSON mode if structured outputs not available
            print(f"Warning: Structured outputs failed ({e}), trying JSON mode...")
            return self._rewrite_with_json_mode(user_query)
    
    def _rewrite_with_json_mode(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback method using JSON mode for older models.
        """
        import json
        
        # Build prompt with current time context
        prompt = self._build_prompt_with_time_context()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    
    def rewrite_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Rewrite multiple queries in batch.
        
        Args:
            queries: List of user query strings
            
        Returns:
            List of rewrite dictionaries
        """
        results = []
        for query in queries:
            try:
                result = self.rewrite(query)
                results.append(result)
            except Exception as e:
                print(f"Error rewriting query '{query}': {e}")
                results.append({
                    "normalized_query": query,
                    "main_clause": "",
                    "details": "",
                    "entities": {
                        "people": [],
                        "organizations": [],
                        "products_or_projects": [],
                        "events_or_meetings": [],
                        "time": [],
                        "keywords": []
                    },
                    "paraphrases": []
                })
        return results


def create_rewriter(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3
) -> QueryRewriter:
    """
    Factory function to create a QueryRewriter instance.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model name
        temperature: Temperature for generation
        
    Returns:
        QueryRewriter instance
    """
    return QueryRewriter(api_key=api_key, model=model, temperature=temperature)


if __name__ == "__main__":
    import json

    sample_query = "What did Ankit and Hongye Qian discuss in their last meeting?"
    rewriter = QueryRewriter()

    try:
        result = rewriter.rewrite(sample_query)
        print("Original query:", sample_query)
        print("\nStructured output:\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Query rewrite failed: {exc}")
