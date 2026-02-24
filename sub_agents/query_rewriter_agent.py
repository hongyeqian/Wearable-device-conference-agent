import sys
import json
from pathlib import Path
from typing import List, Any, Dict

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from config.settings import OPENAI_API_KEY, CURRENT_USER, DATA_DIR
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MEETING_CATALOG: str = ""

# Global variable to store summary metadata for filtering
SUMMARY_METADATA_INDEX: List[Dict[str, Any]] = []


def set_meetings(meetings: List[Any]) -> None:
    """
    Called by rag_main during system initialization, 
    it converts the result of loader.load_all_meetings() into a brief "meeting directory" text for providing to the LLM as context. 
    The meetings are sorted from earliest to latest to avoid time recognition errors.
    """
    global MEETING_CATALOG
    entries: List[tuple] = []  # tuples of (date_obj, list_of_names)

    for m in meetings or []:
        # datetime -> date
        dt_raw = getattr(m, "datetime", None)
        date_obj = None
        if isinstance(dt_raw, datetime):
            date_obj = dt_raw.date()
        elif isinstance(dt_raw, str):
            try:
                text = dt_raw.strip().replace("+08:00", "")
                if "T" in text:
                    date_obj = datetime.fromisoformat(text).date()
                else:
                    date_obj = datetime.strptime(text, "%Y-%m-%d").date()
            except Exception:
                date_obj = None

        # participants（only pick name）
        names: List[str] = []
        for p in getattr(m, "participants", []):
            if isinstance(p, dict):
                name = (p.get("name") or "").strip()
            else:
                name = str(p).strip()
            if name:
                names.append(name)

        # add in list
        if date_obj and names:
            entries.append((date_obj, names))

    # sort them
    entries.sort(key=lambda x: x[0])

    # final list
    lines: List[str] = [f"- {d.isoformat()}: " + ", ".join(names) for d, names in entries]

    MEETING_CATALOG = "\n".join(lines)


def load_summary_metadata() -> List[Dict[str, Any]]:
    """
    Load all summary_metadata.json files from data directory.
    This provides structured data for the Filter Agent to use for meeting selection.
    """
    global SUMMARY_METADATA_INDEX
    
    if SUMMARY_METADATA_INDEX:
        return SUMMARY_METADATA_INDEX
    
    data_dir = Path(DATA_DIR)
    metadata_list = []
    
    for con_dir in sorted(data_dir.glob("con*")):
        if not con_dir.is_dir():
            continue
            
        summary_meta_file = con_dir / "summary_metadata.json"
        if summary_meta_file.exists():
            try:
                with open(summary_meta_file, 'r', encoding='utf-8') as f:
                    metadata_list.append(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load {summary_meta_file}: {e}")
    
    SUMMARY_METADATA_INDEX = metadata_list
    return metadata_list


# Pydantic models for structured output
class Entities(BaseModel):
    """Entity extraction structure"""
    model_config = {"extra": "forbid"}
    
    people: List[str] = Field(description="List of people mentioned")
    organizations: List[str] = Field(description="List of organizations mentioned")
    time: List[str] = Field(description="Time-related expressions, with relative times converted to specific dates")
    keywords: List[str] = Field(description="Key topic phrases or keywords")

class QueryRewriteOutput(BaseModel):
    """Complete query rewrite structure"""
    model_config = {"extra": "forbid"}
    
    normalized_query: str = Field(description="Normalized question keeping core meaning but removing surface details")
    entities: Entities = Field()
    paraphrases: List[str] = Field(description="2-3 paraphrased questions with same meaning but different wording")
    speaker_perspective: str = Field(
        description=(
            "Who is the main speaker perspective of this question, for example: "
            "'current_user', 'Ankit', 'Hongye Qian', or 'third_person_observer'. "
            "This MUST be inferred from the question text and current user context."
        )
    )


class FilterOutput(BaseModel):
    """Filter agent output - contains filtered meeting IDs"""
    model_config = {"extra": "forbid"}
    
    relevant_meeting_ids: List[str] = Field(description="List of meeting IDs that match the query criteria")
    reasoning: str = Field(description="Explanation of how the filtering was done")


class QueryRewriterAgent(SequentialAgent):
    def __init__(self):
        # LLM models
        llm_model = LiteLlm(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        llm_model_structured = LiteLlm(model="gpt-4o", api_key=OPENAI_API_KEY)

        # ===== Pronoun Agent =====
        # When user ask: What did I discuss on November 11th.?
        # This will be used as a tool by Router Agent
        pronoun_agent = LlmAgent(
            name="PronounRewriteAgent",
            model=llm_model,
            instruction=f"""
You rewrite the ORIGINAL user_query by replacing first-person references with the current user name.

CRITICAL: You MUST read current_user_name: {CURRENT_USER}. This is the name that "I/we/me/my/our/us" should map to.

Inputs in session.state:
- user_query: original text (REQUIRED)
- current_user_name: the name that "I/we/me/my/our/us" should map to (REQUIRED - {CURRENT_USER})

Rules:
1) FIRST, read current_user_name: {CURRENT_USER} to get the actual user name.
2) Replace first-person pronouns ("I", "me", "my", "mine", "we", "us", "our", "ours") with the current_user_name you read from state.
3) Do NOT change other names or content.
4) Return ONLY the rewritten query as plain text (no JSON, no extra commentary).
5) If no first-person pronoun is present, return the original query unchanged.
6) If current_user_name is not found in state, use "the current user" as fallback, but this should not happen in normal operation.

Example: If user_query is "What did Ankit and I discuss?" and current_user_name is "Hongye Qian", output should be "What did Ankit and Hongye Qian discuss?"
""",
            output_key="pronoun_rewrite_result",
            include_contents="none",
        )

        # ===== Time Agent =====
        # What did hongye discuss yesterday?
        # This will be used as a tool by Router Agent
        time_agent = LlmAgent(
            name="TimeRewriteAgent",
            model=llm_model,
            instruction=f"""
You are a TimeRewrite agent. Use the meeting catalog and reference time to convert relative time expressions into explicit ISO dates (YYYY-MM-DD).

Meeting catalog (most recent first):
{{{{meeting_catalog}}}}

Reference "now": {{{{now_str}}}}

Rules (CRITICAL — follow exactly):
1) For "last N meetings": find the N most recent meeting dates from the meeting catalog (already sorted most recent first). Return EXACT dates in ISO format, separated by commas, in descending order. If fewer than N meetings exist, return the available dates and explicitly state none for missing dates (e.g. "No chunks found for 2025-11-28").
2) For relative expressions like "yesterday", "3 days ago", compute dates relative to the provided reference "now".
3) ONLY output the rewritten query text in plain text with explicit dates. DO NOT output explanations, JSON, markdown or any extra text.
4) If you cannot determine any dates, output the original query unchanged.

Examples:
- Input: "What did we discuss in the last two meetings?"
- Meeting catalog: "- 2025-11-30: ...\\n- 2025-11-29: ..."
- Output: "What did we discuss on 2025-11-30 and 2025-11-29?"
""",
            output_key="time_rewrite_result",
            include_contents="none",
        )

        # Convert agents to tools for Router to use
        pronoun_tool = AgentTool(agent=pronoun_agent)
        time_tool = AgentTool(agent=time_agent)

        # ===== Router Agent =====
        # Decides whether to call pronoun and time tools
        # Uses "agent as tool" pattern - calls pronoun/time agent as needed
        router_agent = LlmAgent(
            name="RouterAgent",
            model=llm_model,
            instruction=f"""
You are a router that determines what preprocessing is needed for the user query.

You have access to two tools:
1. pronoun_tool: Rewrites first-person pronouns (I, me, my, we, us, our) to the current user name
2. time_tool: Converts relative time expressions to explicit dates

Analyze the user query and decide:
1. need_pronoun_rewrite: Does the query contain first-person pronouns? If yes, CALL pronoun_tool
2. need_time_rewrite: Does the query contain relative time expressions (yesterday, last week, recent, etc.)? If yes, CALL time_tool

IMPORTANT - Agent as Tool pattern:
- Read session.state['user_query'] to get the original query
- If need_pronoun_rewrite is true, YOU MUST CALL pronoun_tool with the user_query
- If need_time_rewrite is true, YOU MUST CALL time_tool (use the result from pronoun if both are needed)
- After calling tools, the results will be stored in session.state['pronoun_rewrite_result'] and session.state['time_rewrite_result']

Workflow:
1. First, determine if you need to call each tool
2. If need_pronoun_rewrite=True, call pronoun_tool
3. If need_time_rewrite=True, call time_tool (use output from pronoun if both called)
4. Finally, output your decision as JSON

Output a JSON with exactly these fields:
- need_pronoun_rewrite: true/false
- need_time_rewrite: true/false
- pronoun_called: true/false (did you call the tool?)
- time_called: true/false (did you call the tool?)
- reason: brief explanation

Return ONLY valid JSON, no extra text.
""",
            output_key="router_decision",
            tools=[pronoun_tool, time_tool],
            include_contents="none",
        )

        # ===== Filter Agent =====
        # Extracts entities from normalized_query and filters meetings
        # Uses Chain-of-Thought (CoT) approach: naturally integrates pronoun/time results
        # Includes FALLBACK LOGIC: if pipeline_query doesn't exist, use user_query
        filter_agent = LlmAgent(
            name="FilterAgent",
            model=llm_model_structured,
            instruction="""
You are a Filter Agent that selects relevant meetings based on the query.

You have access to the following in session.state:
1. session.state['user_query']: The original user query
2. session.state['pronoun_rewrite_result']: Result from pronoun rewriting (if called)
3. session.state['time_rewrite_result']: Result from time rewriting (if called)
4. session.state['summary_metadata_index']: List of all meeting metadata

INTEGRATION LOGIC (IMPORTANT):
- Use the MOST PROCESSED query available:
  * If time_rewrite_result exists → use it (most processed)
  * Else if pronoun_rewrite_result exists → use it
  * Else use user_query (original)
- This handles all cases naturally:
  * Both called: time_rewrite_result has both transformations
  * Only pronoun: pronoun_rewrite_result
  * Neither: user_query

FALLBACK LOGIC:
- If none of the above exist, use user_query as fallback
- This ensures graceful degradation

Your task (use Chain-of-Thought approach):

STEP 1 - Query Selection (with integration + fallback):
- Read time_rewrite_result first (most processed)
- If empty/not exist, read pronoun_rewrite_result
- If still empty, use user_query

STEP 2 - Entity Extraction:
Analyze the selected query to extract:
- People: Who is mentioned? (e.g., "Ankit", "Hongye Qian", "the current user")
- Time: What time/date is mentioned? (e.g., "last meeting", "Nov 30", "yesterday")
- Topics/Keywords: What topics are being asked about? (e.g., "architecture", "summary", "actions")
- Actions: Is the user asking about tasks/actions? (e.g., "what did Ankit ask me to do")

STEP 3 - Filtering:
Match extracted entities against each meeting's metadata:
- Participants: Check if any person is in the meeting's participant list
- Datetime: Check if the date matches (for relative dates like "last week", calculate from now)
- Topics: Check if any topic keyword appears in the meeting's topics list
- Actions: Check if action tasks or assignees match

IMPORTANT: 
- If NO specific filter criteria found (query is too generic), return ALL meeting IDs
- Use case-insensitive matching for names and keywords
- The summary_metadata_index contains topics extracted from meeting summaries

Output JSON with exactly these fields:
- relevant_meeting_ids: array of meeting IDs (e.g., ["data012", "data013"]) - return ALL if no clear criteria
- reasoning: brief explanation of filtering logic

Return ONLY valid JSON, no extra text.
""",
            output_schema=FilterOutput,
            output_key="filter",
            include_contents="none",
        )

        # Initialize SequentialAgent with sub_agents
        # Flow: Router → (calls pronoun/time as tools) → Filter
        super().__init__(
            name="QueryRewritePipeline",
            sub_agents=[router_agent, filter_agent],
        )


# Create global instance
query_rewriter_agent = QueryRewriterAgent()
