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
    由 rag_main 在系统初始化时调用，
    把 loader.load_all_meetings() 的结果转换成一个简短的"会议目录"文本，
    用于提供给 LLM 作为上下文。按日期从早到晚排序，避免时间识别错误。
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

        # participants（只拿 name）
        names: List[str] = []
        for p in getattr(m, "participants", []):
            if isinstance(p, dict):
                name = (p.get("name") or "").strip()
            else:
                name = str(p).strip()
            if name:
                names.append(name)

        # 仅在能解析到日期且有 participant 名称时加入列表
        if date_obj and names:
            entries.append((date_obj, names))

    # 按日期从早到晚排序
    entries.sort(key=lambda x: x[0])

    # 生成最终目录行
    lines: List[str] = [f"- {d.isoformat()}: " + ", ".join(names) for d, names in entries]

    MEETING_CATALOG = "\n".join(lines)
            
            
    

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


APP_NAME = "agents"
USER_ID = "u2"
SESSION_ID = "s2"







class QueryRewriter:
    """
    Multi-agent query rewriter, API 兼容原来的 QueryRewriter.rewrite(user_query) -> dict
    """
    def __init__(
        self,
        app_name: str = "agents",
        user_id: str = "u2",
        session_id_prefix: str = "rewrite-",
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id_prefix = session_id_prefix

        # 每个实例自己维护一个 InMemorySessionService
        self.session_service = InMemorySessionService()

    async def _rewrite_async(self, user_query: str) -> dict:
        # 为每次查询生成一个 session_id，避免状态串扰
        import uuid
        session_id = f"{self.session_id_prefix}{uuid.uuid4().hex[:8]}"

        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=session_id,
            state={"user_query": user_query, "CURRENT_USER": CURRENT_USER, "meeting_catalog": MEETING_CATALOG, "now_str": now_str}
        )

        try:
            print(f"meeting_catalog: {MEETING_CATALOG}")
        except Exception as e:
            print(f"Error: can not print meeting_catalog!!!!!!!!!!!")

        content = types.Content(role="user", parts=[types.Part(text=user_query)])
        
        
        # Time agent: 读取 pronoun_rewrite_result（如果存在）或 user_query，输出到 time_rewrite_result
        time_agent = LlmAgent(
            name="TimeRewriteAgent",
            model=llm_model,
            instruction="""
        Convert relative time to explicit dates using the meeting catalog.

        Read the meeting catalog from {meeting_catalog}.
        Read the current date and time from {now_str}. Use this as the reference "now" when interpreting relative phrases like "last N meetings", "yesterday", "3 days ago", etc.
        Read the query from {pronoun_rewrite_result} (if exists) or {user_query}.

        For "last N meetings", find N matching dates from the meeting catalog, sort descending, take the most recent N.
        Output rewritten query with explicit dates.

        Example:
        - Input: "What did we discuss in the last two meetings?"
        - Meeting catalog: "- 2025-11-30: Ankit, Hongye Qian\n- 2025-11-29: Ankit, Hongye Qian\n- 2025-11-28: Ankit, Hongye Qian"
        - Output: "What did we discuss on 2025-11-30 and 2025-11-29?"
        """,
            output_key="time_rewrite_result",
            #include_contents="none",
        )




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
            output_key="structured_rewrite_result",
            include_contents="none",
        )

        pipeline_agent = SequentialAgent(
            name="QueryRewritePipeline",
            sub_agents=[pronoun_agent, time_agent, structured_agent],
        )


        runner = Runner(agent=pipeline_agent, app_name=self.app_name, session_service=self.session_service)
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

        structure_result = final_session.state.get("structured_rewrite_result") or {}

        # 如果 ADK 返回的是 Pydantic 对象，转成 dict
        if isinstance(structure_result, BaseModel):
            structure_result = structure_result.model_dump()

        # 做一点兜底，确保字段存在（避免 downstream KeyError）
        entities = structure_result.get("entities") or {}
        entities.setdefault("people", [])
        entities.setdefault("keywords", [])
        structure_result["entities"] = entities
        structure_result.setdefault("normalized_query", user_query)
        structure_result.setdefault("paraphrases", [])

        return structure_result

    def rewrite(self, user_query: str) -> dict:
        """
        对外提供同步 API，兼容原来的 QueryRewriter.rewrite
        """
        return asyncio.run(self._rewrite_async(user_query))
    
    
if __name__ == "__main__":
    rewriter = QueryRewriter()
    result = rewriter.rewrite("What did Ankit and I discuss in the last three meetings?")
    print(json.dumps(result, indent=2, ensure_ascii=False))

