import asyncio
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, date

from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.errors.already_exists_error import AlreadyExistsError

from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from config.settings import shared_session_service
import uuid


# for time parsing and filtering

def _parse_date_safe(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip().replace("+08:00", "")
    try:
        if "T" in s:
            return datetime.fromisoformat(s).date()
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_time_expressions(time_exprs: List[str]):
    dates = set()
    date_range = None

    for expr in time_exprs or []:
        expr = (expr or "").strip()
        if not expr:
            continue

        if " to " in expr:
            try:
                start_str, end_str = expr.split(" to ")
                start = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
                end = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
                date_range = (start, end)
                break
            except Exception:
                continue
        else:
            d = _parse_date_safe(expr)
            if d:
                dates.add(d)

    return dates, date_range


def filter_chunks_by_time(
    chunks: List[Dict[str, Any]],
    query_rewrite: Dict[str, Any],
    max_chunks: int = 15,
) -> List[Dict[str, Any]]:
    entities = (query_rewrite or {}).get("entities") or {}
    time_exprs: List[str] = entities.get("time") or []

    if not time_exprs:
        return chunks[:max_chunks]

    dates, date_range = _parse_time_expressions(time_exprs)
    filtered: List[Dict[str, Any]] = []

    for chunk in chunks:
        dt_str = (chunk.get("metadata") or {}).get("datetime")
        d = _parse_date_safe(dt_str)
        if not d:
            continue

        keep = False
        if date_range:
            start, end = date_range
            if start <= d <= end:
                keep = True
        elif dates:
            if d in dates:
                keep = True

        if keep:
            filtered.append(chunk)

    if not filtered:
        filtered = chunks

    return filtered[:max_chunks]




def format_context_for_answer(chunks: List[Dict[str, Any]], max_chunks: int = 15) -> str:
    """
    form like this：
    [data012_summary_1] (Date: 2025-11-29)
    <chunk text>

    [data013_summary_3] (Date: 2025-11-30T20:21:00+08:00)
    <chunk text>
    """
    if not chunks:
        return "No relevant context found."

    lines: List[str] = []

    for chunk in chunks[:max_chunks]:
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "")
        meta = chunk.get("metadata") or {}
        dt = meta.get("datetime", "N/A")

        lines.append(f"[{chunk_id}] (Date: {dt})")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)







class AnswerGeneratorMultiAgent:
    """
    Multi-agent answer generator， API：
        generate_answer(user_query, chunks, query_rewrite, max_chunks=15) -> dict
    return：
        {
            "answer": str,
            "chunks_used": List[Dict],
            "num_chunks": int,
        }
    """

    def __init__(
        self,
        app_name: str = "answer_agents",
        user_id: str = "u-answer",
        session_id_prefix: str = "answer-",
        session_service = shared_session_service,
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id_prefix = session_id_prefix
        self.session_service = session_service

        # initial Answer Agent
        self._init_agent()

    def _init_agent(self):
        """初始化 Answer Agent"""
        llm_model_answer = LiteLlm(
            model=OPENAI_MODEL or "gpt-4o",
            api_key=OPENAI_API_KEY,
        )

        self.answer_agent = LlmAgent(
            name="AnswerSynthesisAgent",
            model=llm_model_answer,
            instruction=r"""
You are the answer generation agent of a meeting-based RAG system.

You will receive a single user question and a context made of multiple chunks.
Each chunk is formatted like:

[chunk_id] (Date: YYYY-MM-DD or full datetime)
<chunk text>

Your tasks:
Your tasks:
1. Use ONLY information from the given context. Do NOT invent facts.
2. Answer the user's question clearly and concisely.
3. Whenever you use information from a chunk, add its citation using the exact chunk_id in square brackets, e.g. [data012_summary_1] or [data012_summary_1][data013_meeting_3].
4. CRITICAL: If the question mentions specific dates or time ranges:
   - You MUST cover ALL dates mentioned in the question.
   - For EACH date mentioned, check if there are chunks in the context with that date.
   - If chunks exist for a date: include information from those chunks in your answer.
   - If NO chunks exist for a date: explicitly state "No information is available for [date]" or "No chunks found for [date]".
   - Structure the answer by date, like:
     "On 2025-11-28, ... [chunk_id]
      On 2025-11-29, ... [chunk_id]
      On 2025-11-30, ... [chunk_id]"
   - DO NOT skip any date that was mentioned in the question, even if no chunks are available for that date.
5. If the context does not contain any relevant information, say that you cannot find relevant information in the provided chunks.

Output:
- Only the final answer text with inline citations using [chunk_id].
- No JSON. No markdown formatting like bullet lists unless it helps clarity.
""",
            output_key="answer_text",
            include_contents="none",
        )

    async def _generate_answer_async(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        query_rewrite: Dict[str, Any],
        max_chunks: int = 15,
    ) -> Dict[str, Any]:
        # Require orchestrator-managed turn_session_id
        session_id = None
        if isinstance(query_rewrite, dict):
            session_id = query_rewrite.get("turn_session_id")
        if not session_id:
            raise ValueError("turn_session_id is required. Orchestrator must create/manage the session lifecycle.")

        # Read existing session; rewriter/orchestrator must have created it
        try:
            session = await self.session_service.get_session(app_name=self.app_name, user_id=self.user_id, session_id=session_id)
        except Exception as e:
            raise RuntimeError(f"Session retrieval failed for session_id={session_id}. Orchestrator must create it before calling answer generator.") from e
        if session is None:
            raise RuntimeError(f"Session not found for session_id={session_id}. Orchestrator must create it before calling answer generator.")

        # ensure turn_memory field exists
        session.state.setdefault("turn_memory", [])
        session.state["user_query"] = user_query

        # 1) filter time
        filtered_chunks = filter_chunks_by_time(chunks, query_rewrite, max_chunks=max_chunks)
        # 2) construct memory
        context_str = format_context_for_answer(filtered_chunks, max_chunks=max_chunks)
        chunks_used = filtered_chunks[:max_chunks]

        # 4) construct user content to LlmAgent
        user_content_text = f"""Question:
{user_query}

Context:
{context_str}

Please answer the question using only the context above. 
Remember to cite chunks by their chunk_id in square brackets, like [data012_summary_1].
"""
        content = types.Content(
            role="user",
            parts=[types.Part(text=user_content_text)],
        )

        runner = Runner(
            agent=self.answer_agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )
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
            raise RuntimeError(f"Session not found after runner for session_id={session_id}.")
        answer_text = final_session.state.get("answer_text") or "I couldn't generate an answer."

        # append QA pair into the same turn session's turn_memory
        entries = session.state.get("turn_memory", []) or []

        entry = {
            "id": f"m-{uuid.uuid4().hex[:8]}",
            "type": "dialog_fact",
            "content": f"Q: {user_query}\nA: {answer_text}",
            "participants": (query_rewrite or {}).get("entities", {}).get("people", []),
            "dates": (query_rewrite or {}).get("entities", {}).get("time", []),
            "source_chunk_ids": [c.get("chunk_id") for c in chunks_used or []],
            "created_at": datetime.utcnow().isoformat(),
            "importance": 0.6,
        }
        entries.append(entry)
        session.state["turn_memory"] = entries

        # Do NOT delete/create session here. Return updated session.state for orchestrator to persist if needed.
        return {
            "answer": answer_text,
            "chunks_used": chunks_used,
            "num_chunks": len(chunks_used),
            "turn_session_id": session_id,
            "session_state": session.state,
        }

    def generate_answer(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        query_rewrite: Dict[str, Any],
        max_chunks: int = 15,
    ) -> Dict[str, Any]:
        if not isinstance(query_rewrite, dict) or not query_rewrite.get("turn_session_id"):
            raise ValueError("query_rewrite must contain 'turn_session_id'")
        return asyncio.run(
            self._generate_answer_async(
                user_query=user_query,
                chunks=chunks,
                query_rewrite=query_rewrite,
                max_chunks=max_chunks,
            )
        )