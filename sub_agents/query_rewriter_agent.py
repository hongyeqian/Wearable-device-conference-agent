"""
Query Rewrite Agent - Single Agent Architecture
Uses ReAct/COT loop for query rewriting to resolve person/time ambiguities
"""
import sys
import json
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
from sub_agents.pandas_utils import get_meetings_df
from sub_agents.placeholder_utils import PlaceholderResolver
from sub_agents.date_resolver import DateResolver
from sub_agents.person_matcher import PersonMatcher


# ============ Pydantic Models ============

class RewriteOutput(BaseModel):
    """Rewrite output structure"""
    model_config = {"extra": "forbid"}
    
    rewritten_query: str = Field(description="Rewritten query")
    placeholders_resolved: bool = Field(description="Whether all placeholders are resolved")
    meeting_ids: List[str] = Field(description="Filtered meeting ID list")
    reasoning: str = Field(description="Rewriting reasoning process")
    ambiguous_terms: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Unresolved ambiguous terms and candidates for answer agent"
    )


# ============ Tool Functions ============

def pandas_query(
    query_type: str,
    fuzzy_name: Optional[str] = None,
    n: Optional[int] = None,
    date_str: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> str:
    """
    Pandas query tool for resolving ambiguities
    
    Args:
        query_type: Query type
            - "find_person": Fuzzy name matching
            - "last_n_meetings": Get last N meetings
            - "date_from_relative": Relative to absolute date
            - "filter_by_person": Filter by person name
            - "filter_by_date": Filter by date
            - "get_all_participants": Get all participants
        
        fuzzy_name: Fuzzy name (for find_person)
        n: Count (for last_n_meetings)
        date_str: Date string (for date_from_relative)
        year: Year (for filter_by_date)
        month: Month (for filter_by_date)
    
    Returns:
        JSON formatted query result
    """
    mdf = get_meetings_df()
    
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
            "is_relative": resolver.is_relative(date_str),
            "needs_year": resolver.needs_year(date_str) if resolved is None else False
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
    
    elif query_type == "get_all_participants":
        participants = mdf.get_all_participants()
        return json.dumps({
            "query_type": "get_all_participants",
            "participants": participants,
            "count": len(participants)
        })
    
    else:
        return json.dumps({"error": f"Unknown query_type: {query_type}"})


def resolve_placeholders(
    query: str,
    person_candidates: Optional[List[str]] = None,
    meeting_dates: Optional[List[str]] = None,
) -> str:
    """
    Placeholder resolution tool
    
    Args:
        query: Query containing placeholders
        person_candidates: Person name candidates list
        meeting_dates: Meeting dates list
    
    Returns:
        JSON formatted resolution result
    """
    if person_candidates is None:
        person_candidates = []
    if meeting_dates is None:
        meeting_dates = []
    
    resolver = PlaceholderResolver(current_user=CURRENT_USER)
    context = {
        "person_candidates": person_candidates,
        "meeting_dates": meeting_dates
    }
    
    resolved_query = resolver.resolve_all(query, context)
    has_placeholders = resolver.has_placeholders(resolved_query)
    
    return json.dumps({
        "original_query": query,
        "resolved_query": resolved_query,
        "has_placeholders": has_placeholders,
        "remaining_placeholders": resolver.extract_placeholders(resolved_query) if has_placeholders else []
    })


def check_ambiguity(query: str) -> str:
    """
    Check ambiguities in query
    
    Args:
        query: User query
    
    Returns:
        JSON formatted ambiguity check result
    """
    # Person ambiguous words
    person_ambiguous = {"i", "me", "my", "mine", "we", "us", "our", "ours", "he", "she", "they", "him", "her", "them"}
    
    # Time ambiguous words
    time_ambiguous = {
        "yesterday", "today", "tomorrow", "last week", "last month", "last year",
        "recently", "recent", "previous", "last", "ago", "days ago", "weeks ago"
    }
    
    query_lower = query.lower()
    
    # Check person
    found_person = []
    for word in person_ambiguous:
        if f" {word} " in query_lower or f" {word}?" in query_lower or f" {word}," in query_lower:
            found_person.append(word)
    
    # Check time
    found_time = []
    for phrase in time_ambiguous:
        if phrase in query_lower:
            found_time.append(phrase)
    
    # Check "last N meetings" or "last meeting"
    import re
    last_n_match = re.search(r"last\s+(\d+)\s+meetings?", query_lower)
    has_last_meeting = "last meeting" in query_lower or "last meeting." in query_lower
    
    return json.dumps({
        "has_person_ambiguity": len(found_person) > 0,
        "person_terms": found_person,
        "has_time_ambiguity": len(found_time) > 0 or has_last_meeting,
        "time_terms": found_time + (["last meeting"] if has_last_meeting else []),
        "has_last_n": last_n_match is not None or has_last_meeting,
        "last_n_value": int(last_n_match.group(1)) if last_n_match else (1 if has_last_meeting else None),
    })


# ============ Create Tools ============

pandas_query_tool = FunctionTool(func=pandas_query)

resolve_placeholders_tool = FunctionTool(func=resolve_placeholders)

check_ambiguity_tool = FunctionTool(func=check_ambiguity)


# ============ Agent Instruction ============

AGENT_INSTRUCTION = f"""
You are a Query Rewrite Agent responsible for rewriting ambiguous user queries into explicit ones.

## Current User
- current_user_name: {CURRENT_USER}
- All "I", "me", "my", "we", "us", "our" must be mapped to "{CURRENT_USER}"

## Your Tools
1. check_ambiguity: Check for ambiguities in the query
2. pandas_query: Execute Pandas queries to get specific information
3. resolve_placeholders: Resolve placeholders

## Placeholder Specification (use when rewriting)
- [current_user] → Replace directly with "{CURRENT_USER}"
- [person:xxx] → Fuzzy person name, need to determine via pandas query
- [date:xxx] → Relative time, need to determine via pandas query
- [last_n:N] → Last N meetings, need to get dates via pandas query

## Rewriting Rules (must follow)

### 1. Person Handling
- "I", "me", "my", "we", "us", "our" → [current_user]
- Other pronouns (he, she, they) → [person:he], etc.
- Fuzzy names (e.g., "Hongye" might be "Hongye Qian") → [person:Hongye]

### 2. Time Handling
- "yesterday", "last week" → [date:yesterday], [date:last week]
- "last 3 meetings" → [last_n:3]
- Month names (e.g., "November") → [date:November]
- If year is missing → [date:November 2025]

### 3. Determine if Rewriting is Needed
- If person is explicit (e.g., "What did Hongye Qian discuss") → No need to rewrite person
- If time is explicit (e.g., "What did he discuss on November 11th, 2025") → No need to rewrite time
- If query is not related to meetings (e.g., "What's the weather") → No rewriting needed, return as is

## ReAct Loop Process

Step 1: Call check_ambiguity to check for ambiguities
Step 2: Based on results, decide next step
  - If there are ambiguities → Call pandas_query to get specific information
  - If there are placeholders → Call resolve_placeholders to resolve
Step 3: Check if there are still unresolved ambiguities
Step 4: If there are → Loop to Step 2
Step 5: Output final result

## Output Format (must return JSON)
{{
  "rewritten_query": "What did Hongye Qian discuss in 2025-11-30?",
  "placeholders_resolved": true,
  "meeting_ids": ["data001", "data002"],
  "ambiguous_terms": {{}},
  "reasoning": "Step 1: Found 'I' needs to map to current_user... Step 2: Found 'last 3 meetings' needs query..."
}}

## Handling Unresolvable Ambiguous Terms
If a fuzzy person name cannot be matched in meeting participants (or match is not unique):
- Record the term and all candidates in ambiguous_terms
- Set placeholders_resolved: false
- Answer agent will inform user the query is unclear based on this info

Example:
{{
  "rewritten_query": "What did [person:UnknownPerson] discuss?",
  "placeholders_resolved": false,
  "meeting_ids": [],
  "ambiguous_terms": {{
    "person:UnknownPerson": {{
      "input": "UnknownPerson",
      "candidates": ["Hongye Qian", "Ankit Kumar"],
      "reason": "No match found in meeting participants"
    }}
  }},
  "reasoning": "..."
}}

## Important Reminders
- Think in English, output in English JSON
- Must call tools at each step, not just guess
- If a placeholder cannot be resolved (e.g., fuzzy name not found), still output and explain why
- meeting_ids are obtained through pandas_query filtering
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
                check_ambiguity_tool,
                pandas_query_tool,
                resolve_placeholders_tool,
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
    
    print("\n=== Test pandas_query ===")
    result = pandas_query(query_type="get_all_participants")
    print(result)
    
    result = pandas_query(query_type="last_n_meetings", n=3)
    print(result)
    
    print("\n=== Test resolve_placeholders ===")
    query = "What did [person:Hongye] discuss in [last_n:3]?"
    result = resolve_placeholders(
        query=query,
        person_candidates=["Hongye Qian", "Ankit Kumar"],
        meeting_dates=["2025-11-30", "2025-11-29", "2025-11-28"]
    )
    print(result)
