"""
Query Rewrite Agent - Single Agent Architecture
Uses ReAct/COT loop for query rewriting to resolve person/time ambiguities
"""
import sys
import json
import re
from pathlib import Path
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.models import LiteLlm
from config.settings import OPENAI_API_KEY, CURRENT_USER, DATA_DIR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
import dateparser
from sub_agents.pandas_utils import get_meetings_df
# from sub_agents.placeholder_utils import PlaceholderResolver
from sub_agents.date_resolver import DateResolver
from sub_agents.person_matcher import PersonMatcher


# ============ Pydantic Models ============

class RewriteOutput(BaseModel):
    """Rewrite output structure - simplified format"""
    model_config = {"extra": "forbid"}
    
    rewritten_query: str = Field(description="Rewritten query with ambiguities resolved")
    relevant_meeting_ids: List[str] = Field(description="Filtered meeting ID list based on resolved ambiguities")


# ============ Tool Functions ============

# Global counter for pandas_query calls (enforced limit)
_pandas_query_call_count = 0
_PANDAS_QUERY_MAX_CALLS = 3

def reset_pandas_query_counter():
    """Reset the pandas_query call counter - call this before each new query"""
    global _pandas_query_call_count
    _pandas_query_call_count = 0

def reset_counter() -> str:
    """
    Reset the pandas_query call counter.
    Call this at the start of each new query processing to ensure fresh counting.
    
    Returns:
        JSON formatted reset confirmation
    """
    reset_pandas_query_counter()
    return json.dumps({
        "status": "reset",
        "message": "pandas_query call counter has been reset to 0",
        "max_calls_per_query": _PANDAS_QUERY_MAX_CALLS
    })

def pandas_query(
    query_type: str,
    fuzzy_name: Optional[str] = None,
    n: Optional[int] = None,
    date_str: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Pandas query tool for resolving ambiguities
    
    Args:
        query_type: Query type
            - "find_person": Fuzzy name matching
            - "last_n_meetings": Get last N meetings
            - "date_from_relative": Relative to absolute date
            - "filter_by_person": Filter by person name
            - "filter_by_date": Filter by date (year/month)
            - "filter_by_date_range": Filter by date range
            - "get_all_participants": Get all participants
        
        fuzzy_name: Fuzzy name (for find_person)
        n: Count (for last_n_meetings)
        date_str: Date string (for date_from_relative)
        year: Year (for filter_by_date)
        month: Month (for filter_by_date)
        start_date: Start date (for filter_by_date_range), format: YYYY-MM-DD
        end_date: End date (for filter_by_date_range), format: YYYY-MM-DD
    
    Returns:
        JSON formatted query result
    """
    global _pandas_query_call_count
    
    # Check call count limit (code enforced)
    if _pandas_query_call_count >= _PANDAS_QUERY_MAX_CALLS:
        return json.dumps({
            "error": "MAX_CALLS_REACHED",
            "message": f"pandas_query has been called {_pandas_query_call_count} times. Maximum {_PANDAS_QUERY_MAX_CALLS} calls allowed per query. Please proceed with the information you have."
        })
    
    # Increment counter
    _pandas_query_call_count += 1
    
    mdf = get_meetings_df()
    
    # Include call count in response for transparency
    call_info = {
        "call_number": _pandas_query_call_count,
        "max_calls": _PANDAS_QUERY_MAX_CALLS,
        "remaining": _PANDAS_QUERY_MAX_CALLS - _pandas_query_call_count
    }
    
    if query_type == "find_person":
        if not fuzzy_name:
            return json.dumps({"error": "fuzzy_name is required"})
        
        matches = mdf.find_person(fuzzy_name)
        return json.dumps({
            "query_type": "find_person",
            "fuzzy_name": fuzzy_name,
            "matches": matches,
            "count": len(matches)
        })
    
    elif query_type == "last_n_meetings":
        if not n:
            n = 3
        
        meetings = mdf.get_last_n_meetings(n)
        dates = [m["date"] for m in meetings]
        return json.dumps({
            "query_type": "last_n_meetings",
            "n": n,
            "meetings": meetings,
            "dates": dates
        })
    
    elif query_type == "date_from_relative":
        if not date_str:
            return json.dumps({"error": "date_str is required"})
        
        resolver = DateResolver()
        resolved = resolver.resolve(date_str)
        return json.dumps({
            "query_type": "date_from_relative",
            "input": date_str,
            "resolved": resolved,
            # "is_relative": resolver.is_relative(date_str),
            # "needs_year": resolver.needs_year(date_str) if resolved is None else False
        })
    
    elif query_type == "filter_by_person":
        if not fuzzy_name:
            return json.dumps({"error": "fuzzy_name is required"})
        
        df = mdf.filter_by_person(fuzzy_name)
        meeting_ids = df["meeting_id"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_person",
            "person": fuzzy_name,
            "meeting_ids": meeting_ids,
            "count": len(meeting_ids)
        })
    
    elif query_type == "filter_by_date":
        df_filtered = mdf.filter_by_year_month(
            year=year if year is not None else None,
            month=month if month is not None else None
        )
        meeting_ids = df_filtered["meeting_id"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_date",
            "year": year,
            "month": month,
            "meeting_ids": meeting_ids,
            "count": len(meeting_ids)
        })
    
    elif query_type == "filter_by_date_range":
        if not start_date and not end_date:
            return json.dumps({"error": "At least one of start_date or end_date is required"})
        
        # pandas_utils.filter_by_date_range accepts Optional[str]
        df_filtered = mdf.filter_by_date_range(
            start_date=start_date, 
            end_date=end_date
        )
        meeting_ids = df_filtered["meeting_id"].tolist()
        dates = df_filtered["date"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_date_range",
            "start_date": start_date,
            "end_date": end_date,
            "meeting_ids": meeting_ids,
            "dates": dates,
            "count": len(meeting_ids)
        })
    
    elif query_type == "get_all_participants":
        participants = mdf.get_all_participants()
        return json.dumps({
            "query_type": "get_all_participants",
            "participants": participants,
            "count": len(participants)
        })
    
    else:
        return json.dumps({"error": f"Unknown query_type: {query_type}"})


# def resolve_placeholders(
#     query: str,
#     person_candidates: Optional[List[str]] = None,
#     meeting_dates: Optional[List[str]] = None,
# ) -> str:
#     """
#     Placeholder resolution tool
    
#     Args:
#         query: Query containing placeholders
#         person_candidates: Person name candidates list
#         meeting_dates: Meeting dates list
    
#     Returns:
#         JSON formatted resolution result
#     """
#     if person_candidates is None:
#         person_candidates = []
#     if meeting_dates is None:
#         meeting_dates = []
    
#     resolver = PlaceholderResolver(current_user=CURRENT_USER)
#     context = {
#         "person_candidates": person_candidates,
#         "meeting_dates": meeting_dates
#     }
    
#     resolved_query = resolver.resolve_all(query, context)
#     has_placeholders = resolver.has_placeholders(resolved_query)
    
#     return json.dumps({
#         "original_query": query,
#         "resolved_query": resolved_query,
#         "has_placeholders": has_placeholders,
#         "remaining_placeholders": resolver.extract_placeholders(resolved_query) if has_placeholders else []
#     })


def check_ambiguity(query: str) -> str:
    """
    Check ambiguities in query using NLP libraries
    
    Uses spaCy for time entity detection and PersonMatcher (with pandas participants) for person entity detection.
    Person detection uses ListRecognizer with actual meeting participants for 100% accurate matching.
    
    Args:
        query: User query
    
    Returns:
        JSON formatted ambiguity check result
    """
    import spacy
    import dateparser
    from sub_agents.person_matcher import PersonMatcher
    
    # Use module-level cache for spaCy model
    global _spacy_nlp
    
    # Lazy load spaCy model
    if '_spacy_nlp' not in globals():
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_nlp = spacy.load("en_core_web_sm")
    
    # Use spaCy to detect DATE/TIME entities
    # Return ALL detected time entities (Stage 2 will handle resolution)
    doc = _spacy_nlp(query)
    time_entities = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    
    # Return all spaCy-detected time entities
    # Stage 2 will use dateparser to resolve them
    # Note: "last meeting", "recent meetings" etc. won't be detected by spaCy
    # They will be handled by LLM in Stage 3
    found_time = time_entities
    
    # NOTE: We don't filter by dateparser here - we return all detected entities
    # Stage 2 is responsible for resolving absolute times using dateparser
    
    # Person detection using PersonMatcher with pandas participants
    # This uses ListRecognizer for 100% accurate matching against actual meeting participants
    matcher = PersonMatcher(threshold=0.6)
    person_detection_result = matcher.detect_and_resolve(query)
    
    # Get resolved person names
    resolved_persons = person_detection_result.get("resolved", [])  # Resolved full names
    
    # Get participants for fuzzy matching
    participants = person_detection_result.get("participants", [])
    
    # Create mapping: original text in query -> resolved name
    # Since PersonMatcher doesn't return original text, we need to find it manually
    # by matching resolved names back against the query using TheFuzz
    from thefuzz import fuzz
    
    person_mapping = {}
    query_lower = query.lower()
    
    for resolved in resolved_persons:
        # Try to find what text in the query matched this resolved name
        # Check if the resolved name or any part of it appears in the query
        parts = resolved.split()
        for part in parts:
            if len(part) < 3:  # Skip short parts
                continue
            # Check if this part appears in the query
            if part.lower() in query_lower:
                # Found a match - use the original text from query
                # Find the actual case from query
                import re
                match = re.search(re.escape(part), query, re.IGNORECASE)
                if match:
                    original_text = match.group()
                    person_mapping[original_text] = resolved
                    break
    
    # Get resolved person names
    found_person = resolved_persons
    
    # Also detect pronouns
    # Todo: Becasue we do not realize the memory logic for agent, so we do not consider he, she at the current stage
    person_ambiguous = {"i", "me", "my", "mine", "we", "us", "our", "ours", 
                        "he", "she", "they", "him", "her", "them"}
    query_lower = query.lower()
    found_person_pronouns = []
    for word in person_ambiguous:
        if f" {word} " in query_lower or f" {word}?" in query_lower or f" {word}," in query_lower:
            found_person_pronouns.append(word)
    
    # Combine: resolved persons + pronouns
    found_person = found_person + found_person_pronouns
    
    return json.dumps({
        "has_person_ambiguity": len(found_person) > 0,
        "person_terms": found_person,
        "person_mapping": person_mapping,  # Mapping from original name to resolved name
        "has_time_ambiguity": len(found_time) > 0,
        "time_terms": found_time,
    })


def resolve_entities(original_query: str, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    阶段二：Resolution - 直接替换（方案A）
    
    1. 第一人称 → 替换为 CURRENT_USER
    2. 绝对时间 → 用 dateparser 解析为具体日期
    3. 人名 → 使用 person_mapping 进行替换
    4. "last meeting" 等模糊时间 → 原样传给阶段三，不处理
    
    Args:
        original_query: 原始查询
        stage1_result: 阶段一检测结果
    
    Returns:
        包含 resolved_query 和 replacement_log 的字典
    """
    query = original_query
    person_terms = stage1_result.get('person_terms', [])
    person_mapping = stage1_result.get('person_mapping', {})  # {original_name: resolved_name}
    time_terms = stage1_result.get('time_terms', [])  # Only spaCy-detected absolute times
    
    replacement_log = []
    
    # 1. Handle pronouns -> CURRENT_USER (use word boundary regex)
    first_person = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    for term in person_terms:
        term_lower = term.lower()
        if term_lower in first_person:
            replacement_log.append((term, CURRENT_USER))
            # Use word boundary to avoid replacing substrings
            query = re.sub(r'\b' + re.escape(term) + r'\b', CURRENT_USER, query, flags=re.IGNORECASE)
    
    # 2. Handle time terms -> dateparser解析
    for term in time_terms:
        parsed = dateparser.parse(term)
        if parsed:
            date_str = parsed.strftime("%Y-%m-%d")
            replacement_log.append((term, date_str))
            query = query.replace(term, date_str)
    
    # 3. Person names -> 使用 person_mapping 替换原始人名为解析后人名
    # person_mapping: {original_name: resolved_name}, e.g., {"Hongye": "Hongye Qian"}
    for original_name, resolved_name in person_mapping.items():
        if original_name != resolved_name:  # Only replace if different
            replacement_log.append((original_name, resolved_name))
            query = re.sub(r'\b' + re.escape(original_name) + r'\b', resolved_name, query, flags=re.IGNORECASE)
    
    # 4. "last meeting" is NOT handled here - passes through to Stage 3
    # (spaCy doesn't detect it, so it's not in time_terms)
    
    return {
        'resolved_query': query,
        'replacement_log': replacement_log
    }


# ============ Create Tools ============

pandas_query_tool = FunctionTool(func=pandas_query)

# resolve_placeholders_tool = FunctionTool(func=resolve_placeholders)

check_ambiguity_tool = FunctionTool(func=check_ambiguity)

reset_counter_tool = FunctionTool(func=reset_counter)


# ============ Agent Instruction ============

AGENT_INSTRUCTION = f"""
You are a Query Rewrite Agent - Stage 3 (LLM ReAct)

## CRITICAL: Only handle if "last N meetings" pattern EXISTS in the query!
- If the query does NOT contain "last meeting", "last N meetings", "recent meetings", "past meetings", or "previous meetings", return the query AS IS with empty relevant_meeting_ids
- Do NOT add meeting dates if they're not mentioned in the query
- Do NOT call pandas_query unless the query explicitly mentions meeting count/time

## IMPORTANT: This is Stage 3 of a 3-Stage Architecture
- Stage 1 (Detection): Already executed by check_ambiguity function
- Stage 2 (Resolution): Already executed by resolve_entities function
- Stage 3 (This Agent): Handle ONLY "last N meetings" type ambiguities

## Input to This Stage
The query you receive has ALREADY been processed by Stage 1 and Stage 2:
- First-person pronouns (I, me, my) → already replaced with "{CURRENT_USER}"
- Absolute time expressions (yesterday, last week) → already resolved to dates
- Person names → already resolved to full names

Your job is to ONLY handle remaining ambiguities related to MEETING COUNT:
- "last meeting" → need to find the actual date via pandas_query
- "last N meetings" → need to find dates via pandas_query
- "recent meetings" → need to find dates via pandas_query

## Current User
- current_user_name: {CURRENT_USER}

## Your Tools
1. reset_counter_tool: Reset the pandas_query call counter (MUST call this FIRST)
2. pandas_query: Execute Pandas queries to get meeting information (MAX 3 CALLS - CODE ENFORCED)

## IMPORTANT: Max 3 pandas_query Calls (Code Enforced)
- You MUST NOT call pandas_query more than 3 times per query
- If you reach 3 calls and still need more info, proceed with what you have
- This limit is CODE-ENFORCED and cannot be bypassed

## What You Should Do
1. Call reset_counter_tool FIRST
2. Analyze the query for "last N meetings" type patterns
3. ONLY call pandas_query if the query explicitly mentions meeting count/time
4. Rewrite the query by replacing "last N meetings" with actual dates
5. Return the rewritten query and relevant_meeting_ids

## Meeting Count Patterns to Handle (ONLY if present in query)
- "last N meetings" (e.g., "last 3 meetings") → Call pandas_query(query_type="last_n_meetings", n=3)
- "recent meetings" → Call pandas_query(query_type="last_n_meetings", n=3)
- "recent N meetings" (e.g., "recent 5 meetings") → Call pandas_query(query_type="last_n_meetings", n=5)
- "past meetings" → Call pandas_query(query_type="last_n_meetings", n=3)
- "previous meetings" → Call pandas_query(query_type="last_n_meetings", n=3)
- "last meeting" (singular) → Call pandas_query(query_type="last_n_meetings", n=1)

## What NOT to Do
- Do NOT call check_ambiguity (already done in Stage 1)
- Do NOT re-resolve pronouns or dates (already done in Stage 2)
- Do NOT handle explicit person names or explicit dates

## Output Format (must return JSON - simplified)
{{
  "rewritten_query": "What did Hongye Qian discuss in 2025-11-30?",
  "relevant_meeting_ids": ["data001", "data002"]
}}

## Important Reminders
- Think in English, output in English JSON
- Only handle "last N meetings" type ambiguities
- meeting_ids are obtained through pandas_query
"""


# ============ Agent Class ============

class QueryRewriterAgent(LlmAgent):
    """Single Agent architecture for Query Rewrite"""
    
    def __init__(self):
        llm_model = LiteLlm(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )
        
        super().__init__(
            name="QueryRewriteAgent",
            model=llm_model,
            instruction=AGENT_INSTRUCTION,
            description="Rewrite ambiguous user queries into explicit ones",
            tools=[
                reset_counter_tool,
                check_ambiguity_tool,
                pandas_query_tool,
                # resolve_placeholders_tool,
            ],
            output_schema=RewriteOutput,
            output_key="rewrite_result",
        )


# Create global instance
query_rewriter_agent = QueryRewriterAgent()


# ============ Test Code ============

if __name__ == "__main__":
    # Test tools
    print("=== Test check_ambiguity ===")
    result = check_ambiguity("What did I discuss in the last 3 meetings?")
    print(result)
    
    # print("\n=== Test pandas_query ===")
    # result = pandas_query(query_type="get_all_participants")
    # print(result)
    
    # result = pandas_query(query_type="last_n_meetings", n=3)
    # print(result)
    
    # print("\n=== Test resolve_placeholders ===")
    # query = "What did [person:Hongye] discuss in [last_n:3]?"
    # result = resolve_placeholders(
    #     query=query,
    #     person_candidates=["Hongye Qian", "Ankit Kumar"],
    #     meeting_dates=["2025-11-30", "2025-11-29", "2025-11-28"]
    # )
    # print(result)
