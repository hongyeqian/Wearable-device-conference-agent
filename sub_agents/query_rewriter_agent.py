import sys
from pathlib import Path
from typing import List, Any

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models import LiteLlm
from config.settings import OPENAI_API_KEY, CURRENT_USER
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MEETING_CATALOG: str = ""


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

# Pydantic models for structured output
class Entities(BaseModel):
    """Entity extraction structure"""
    model_config = {"extra": "forbid"}
    
    people: List[str] = Field(description="List of people mentioned")
    organizations: List[str] = Field(description="List of organizations mentioned")
    #products_or_projects: List[str] = Field(description="List of products or projects mentioned")
    #events_or_meetings: List[str] = Field(description="List of events or meetings mentioned")
    time: List[str] = Field(description="Time-related expressions, with relative times converted to specific dates")
    keywords: List[str] = Field(description="Key topic phrases or keywords")

class QueryRewriteOutput(BaseModel):
    """Complete query rewrite structure"""
    model_config = {"extra": "forbid"}
    
    normalized_query: str = Field(description="Normalized question keeping core meaning but removing surface details")
    #main_clause: str = Field(description="Short phrase describing what information is being requested")
    #details: str = Field(description="Time range, speakers, events, locations, or other constraints")
    entities: Entities = Field()  # 去掉 description
    paraphrases: List[str] = Field(description="2-3 paraphrased questions with same meaning but different wording")
    speaker_perspective: str = Field(
        description=(
            "Who is the main speaker perspective of this question, for example: "
            "'current_user', 'Ankit', 'Hongye Qian', or 'third_person_observer'. "
            "This MUST be inferred from the question text and current user context."
        )
    )

class QueryRewriterAgent(SequentialAgent):
    def __init__(self):
        # LLM models
        llm_model = LiteLlm(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        llm_model_structured = LiteLlm(model="gpt-4o", api_key=OPENAI_API_KEY)

        # Pronoun agent
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

        # Time agent
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

        # Structured agent
        structured_agent = LlmAgent(
            name="StructuredRewriteAgent",
            model=llm_model_structured,
            instruction=r"""
You take the processed user query from session.state['pipeline_query'] (or fallback to session.state['user_query']) and produce structured JSON with EXACT fields:
- normalized_query: string
- entities: object with {people: [], organizations: [], time: [], keywords: []}
- keywords: array of strings
- paraphrases: array of 2-4 strings
- speaker_perspective: string (current_user / a named person / third_person_observer)

Return ONLY valid JSON matching this structure. No extra text, no markdown, no explanation.
""",
            output_schema=QueryRewriteOutput,
            output_key="rewrite",
            include_contents="none",
        )

        # Initialize SequentialAgent with sub_agents
        super().__init__(
            name="QueryRewritePipeline",
            sub_agents=[pronoun_agent, time_agent, structured_agent],
        )

# Create global instance
query_rewriter_agent = QueryRewriterAgent()