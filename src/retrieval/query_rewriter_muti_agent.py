import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Any, Optional

from pydantic import BaseModel
from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent
from google.adk.models import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from config.settings import OPENAI_API_KEY
from config.settings import shared_session_service

from pydantic import BaseModel, Field
from typing import List

from config.settings import CURRENT_USER

import json
from datetime import datetime

MEETING_CATALOG: str = ""



# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

now_str = datetime.now().astimezone().isoformat()
    
    

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
    
    
llm_model = LiteLlm(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
llm_model_structured = LiteLlm(model="gpt-4o", api_key=OPENAI_API_KEY)


# Import unified session keys from config
from config.settings import APP_NAME, USER_ID


class QueryRewriter:
    """
    Multi-agent query rewriter
    """
    def __init__(
        self,
        app_name: str = APP_NAME,
        user_id: str = USER_ID,
        session_id_prefix: str = "rewrite-",
        session_service: InMemorySessionService = shared_session_service,
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id_prefix = session_id_prefix

        # Each instance maintains its own InMemorySessionService.
        self.session_service = session_service

        # initial all agents
        self._init_agents()

    def _init_agents(self):
        """initial ADK Agents"""
        # Time agent
        self.time_agent = LlmAgent(
            name="TimeRewriteAgent",
            model=llm_model,
            instruction="""
You are a TimeRewrite agent. Use the meeting catalog below and the reference time to convert relative time expressions into explicit ISO dates (YYYY-MM-DD).

Meeting catalog (most recent first):
{meeting_catalog}

Reference "now": {now_str}  

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
            #include_contents="none",
        )

        # Pronoun agent
        self.pronoun_agent = LlmAgent(
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

        # Structured agent
        self.structured_agent = LlmAgent(
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
            output_key="structured_rewrite_result",
            include_contents="none",
        )

        # Pipeline Agent (expose this for adk web surface)
        self.pipeline_agent = SequentialAgent(
            name="QueryRewritePipeline",
            sub_agents=[self.pronoun_agent, self.time_agent, self.structured_agent],
        )

    async def _rewrite_async(self, user_query: str, turn_session_id: str ) -> dict:
        if not turn_session_id:
            raise ValueError("turn_session_id is required. Orchestrator must create/manage the session lifecycle.")

        session_id = turn_session_id

        # Try to read existing session; if missing, raise so orchestrator can decide what to do
        try:
            existing = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=session_id
            )
        except Exception as e:
            # Fail fast: rewriter must not create session
            raise RuntimeError(f"Session retrieval failed for session_id={session_id}. Orchestrator must create it before calling rewriter.") from e
        if existing is None:
            raise RuntimeError(f"Session not found for session_id={session_id}. Orchestrator must create it before calling rewriter.")

        # Build memory_context_text from existing turn_memory (if any)
        tm = existing.state.get("turn_memory", []) or []
        memory_context_text = "\n".join([f"- ({m.get('type','fact')}) {m.get('content','')}" for m in tm])
        
        # right after memory_context_text computed (around where tm set)
        print(f"DEBUG[_rewrite_async]: turn_session_id={session_id}, user_id={self.user_id}")
        try:
            print("DEBUG[_rewrite_async]: session.state keys:", list(existing.state.keys()))
            print("DEBUG[_rewrite_async]: session.state['meeting_catalog'] (len):", len(existing.state.get('meeting_catalog','')) if existing.state.get('meeting_catalog') else None)
        except Exception:
            print("DEBUG[_rewrite_async]: no existing.state or cannot read it")
        print("DEBUG[_rewrite_async]: memory_context_text:", memory_context_text[:500])
        # also show module-level MEETING_CATALOG and now_str
        print("DEBUG[_rewrite_async]: module MEETING_CATALOG preview:", MEETING_CATALOG[:500])
        print("DEBUG[_rewrite_async]: now_str in module:", now_str)

        # Construct user content and run pipeline as before (use same runner and session_id)
        content = types.Content(role="user", parts=[types.Part(text=user_query)])

        # use pipeline_agent
        runner = Runner(agent=self.pipeline_agent, app_name=self.app_name, session_service=self.session_service)
        events = runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=content,
        )
        async for _ in events:
            pass

        final_session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=session_id,
        )
        if final_session is None:
            raise RuntimeError(f"Session not found after pipeline for session_id={session_id}.")
        structure_result = final_session.state.get("structured_rewrite_result") or {}
        

        print("DEBUG: final_session.state keys:", list(final_session.state.keys()))


        time_result = final_session.state.get("time_rewrite_result")
        print("DEBUG: time_rewrite_result:", time_result)
        # 如果 ADK 返回的是 Pydantic 对象，转成 dict
        if isinstance(structure_result, BaseModel):
            structure_result = structure_result.model_dump()
            
            
        structure_result["turn_session_id"] = session_id

        entities = structure_result.get("entities") or {}
        entities.setdefault("people", [])
        entities.setdefault("keywords", [])
        structure_result["entities"] = entities
        structure_result.setdefault("normalized_query", user_query)
        structure_result.setdefault("paraphrases", [])


        return structure_result

    def rewrite(self, user_query: str, turn_session_id: str) -> dict:
        """
        Provide synchronous API externally
        """
        return asyncio.run(self._rewrite_async(user_query, turn_session_id))