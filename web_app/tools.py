import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from google.adk.tools import ToolContext, FunctionTool
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LiteLlm

# Pyrefly ignore for configuration
# pyrefly: ignore [missing-import]
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

_project_root = Path(__file__).parent.parent
_LOG_DIR = _project_root / "logs"
_EMAIL_LOG = _LOG_DIR / "mock_email_history.jsonl"
_CAL_LOG = _LOG_DIR / "mock_calendar_history.jsonl"

def _log_to_file(filepath: Path, record: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Physical Isolation Tools for Human-in-the-loop (Option B)
# ---------------------------------------------------------------------------

async def draft_email(recipient: str, subject: str, content: str, tool_context: ToolContext) -> dict:
    """Draft an email to a recipient. Use this to prepare an email before executing it.
    
    Args:
        recipient: Name or email address of the recipient.
        subject: Concise subject line.
        content: The body of the email.
    """
    return {
        "status": "drafted",
        "recipient": recipient,
        "subject": subject,
        "content": content,
        "message": "Draft created successfully. Ask the user if they are satisfied with this draft before calling execute_email."
    }

async def execute_email(recipient: str, subject: str, content: str, tool_context: ToolContext) -> dict:
    """Execute sending an email. ONLY call this AFTER the user has explicitly confirmed the draft from draft_email.
    
    Args:
        recipient: Name or email address of the recipient.
        subject: Concise subject line.
        content: The body of the email.
    """
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    email_id = f"mock_{ts}_{uuid.uuid4().hex[:8]}"
    
    record = {
        "email_id": email_id,
        "timestamp": now.isoformat(),
        "recipient": recipient,
        "subject": subject,
        "content": content,
        "mock": True
    }
    
    try:
        _log_to_file(_EMAIL_LOG, record)
    except Exception as e:
        return {"status": "error", "message": f"Failed to log mock email: {e}"}
        
    return {"status": "sent", "email_id": email_id, "message": "Email has been sent successfully."}

async def draft_meeting(participants: list[str], title: str, start_datetime: str, tool_context: ToolContext) -> dict:
    """Draft a calendar meeting. Use this to prepare a meeting before executing it.
    
    Args:
        participants: List of participant names or email addresses.
        title: Title of the meeting.
        start_datetime: ISO formatted date and time for the meeting (YYYY-MM-DD HH:MM).
    """
    return {
        "status": "drafted",
        "participants": participants,
        "title": title,
        "start_datetime": start_datetime,
        "message": "Meeting draft created successfully. Ask the user for confirmation before calling execute_meeting."
    }

async def execute_meeting(participants: list[str], title: str, start_datetime: str, tool_context: ToolContext) -> dict:
    """Execute scheduling a calendar meeting. ONLY call this AFTER the user has explicitly confirmed the draft from draft_meeting.
    
    Args:
        participants: List of participant names or email addresses.
        title: Title of the meeting.
        start_datetime: ISO formatted date and time for the meeting (YYYY-MM-DD HH:MM).
    """
    now = datetime.now(timezone.utc)
    event_id = f"mock_cal_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    record = {
        "event_id": event_id,
        "timestamp": now.isoformat(),
        "participants": participants,
        "title": title,
        "suggested_datetime": start_datetime,
        "mock": True
    }
    
    try:
        _log_to_file(_CAL_LOG, record)
    except Exception as e:
        return {"status": "error", "message": f"Failed to log mock calendar event: {e}"}
        
    return {"status": "scheduled", "event_id": event_id, "message": "Meeting has been scheduled successfully."}

async def query_user_memory(query: str, tool_context: ToolContext) -> str:
    """Search the user's long-term memory for preferences, background, and personal relationships.
    Use this when the user asks personal questions like 'Who are my teammates?' or 'What do I like?'.
    
    Args:
        query: The search query string for long term memory.
    """
    from src.memory.mem0_service import mem0_service
    user = tool_context.state.get("CURRENT_USER", "")
    if not user:
        return "No active user found to query memory."
    return mem0_service.search_memories(user, query, limit=20)


# ---------------------------------------------------------------------------
# QA Specialist Agent (Answer Agent with RAG hidden inside)
# ---------------------------------------------------------------------------

async def prepare_qa_context(*, callback_context: CallbackContext, **kwargs):
    """
    Callback that runs before the QA specialist answers.
    It runs the Query Rewriter and Retrieval, packing the chunks into state for the QA prompt.
    """
    user_query = callback_context.state.get("user_query", "")
    current_user = callback_context.state.get("CURRENT_USER", "")
    
    from sub_agents.query_rewriter_agent import rewrite_query_async
    from web_app.agent import format_context_for_answer
    
    print(f"\n[QA Specialist] Rewriting query: '{user_query}' for user '{current_user}'...")
    rewrite_result = await rewrite_query_async(user_query, current_user=current_user)
    
    retrieval_query = user_query
    relevant_meeting_ids = []
    if rewrite_result:
        retrieval_query = rewrite_result.rewritten_query
        relevant_meeting_ids = rewrite_result.relevant_meeting_ids
        print(f"[QA Specialist] Rewritten query: '{retrieval_query}' | IDs: {relevant_meeting_ids}")
        
    # Get retriever from root agent which was passed via state or singleton
    from web_app.agent import root_agent
    retriever = root_agent.retriever
    
    if not retriever:
        callback_context.state["retrieval_chunks"] = "Error: Retriever not initialized."
        return
        
    # Base security filter
    from sub_agents.metadata_manager import get_meetings_metadata
    mdf = get_meetings_metadata(current_user)
    all_visible_ids = set(r["meeting_id"] for r in mdf.records) if mdf.records else set()
    
    meeting_ids_filter = set(relevant_meeting_ids).intersection(all_visible_ids) if relevant_meeting_ids else all_visible_ids
    
    if not meeting_ids_filter:
        callback_context.state["retrieval_chunks"] = "No visible meetings match the query filters."
        return
        
    print(f"[QA Specialist] Retrieving from VectorDB...")
    results = retriever.vector_store.search_with_meeting_ids_filter(
        query_text=retrieval_query,
        level='summary',
        meeting_ids=meeting_ids_filter,
        top_k=20
    )
    
    hits = results.get("hits", []) if isinstance(results, dict) else results
    chunks = sorted(hits, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:10]
    
    clean_chunks = [
        {
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text"),
            "metadata": {"datetime": c.get("metadata", {}).get("datetime", "N/A")},
        }
        for c in chunks
    ]
    
    formatted_chunks = format_context_for_answer(clean_chunks)
    callback_context.state["retrieval_chunks"] = formatted_chunks
    
    # Also inject long term memory for personal context
    from src.memory.mem0_service import mem0_service
    long_term_context = mem0_service.search_memories(user_id=current_user, query=user_query, limit=10)
    callback_context.state["long_term_memory_context"] = long_term_context


llm_model = LiteLlm(
    model=OPENAI_MODEL or "gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

qa_specialist = LlmAgent(
    name="QA_Specialist",
    model=llm_model,
    instruction="""
You are the QA specialist of a meeting-based RAG system.
NEVER MAKE UP INFORMATION. If no relevant data is found, say so explicitly.

<user_background>
{long_term_memory_context}
</user_background>

<retrieved_meeting_documents>
{retrieval_chunks}
Each item has keys: "chunk_id", "text", "metadata": {{"datetime": "YYYY-MM-DD"}}
</retrieved_meeting_documents>

Rules:
1) Use ONLY information from <retrieved_meeting_documents> and <user_background>. Do NOT invent facts.
2) Cite chunk_id inline when using info from meeting documents: e.g. [data012_summary_1].
3) For date-specific queries: check metadata.datetime per chunk; if no chunk matches, explicitly say "No information available for [date]".
4) Keep answers concise with inline citations only.
""",
    description="Specialist for answering questions about past meetings, discussions, action items, or meeting summaries. Delegate to this agent when the user asks a factual question about meetings.",
    before_agent_callback=prepare_qa_context,
)

# ---------------------------------------------------------------------------
# Intent Router (Main ReAct Agent)
# ---------------------------------------------------------------------------

intent_router = LlmAgent(
    name="IntentRouter",
    model=llm_model,
    instruction="""
You are a highly intelligent ReAct (Reasoning & Acting) assistant.
Your job is to help users manage meetings, answer questions, and send emails.

You have access to the following capabilities:
1. QA Specialist Sub-Agent: Use this for ANY question about meeting content (summaries, decisions, action items).
2. query_user_memory Tool: Use this for questions about the user's personal preferences or relationships.
3. draft_email & execute_email Tools: Use to draft and send emails.
4. draft_meeting & execute_meeting Tools: Use to draft and schedule meetings.

CRITICAL RULES for execution (Human-in-the-loop):
- NEVER call `execute_email` without FIRST calling `draft_email` and showing the draft to the user for explicit confirmation.
- NEVER call `execute_meeting` without FIRST calling `draft_meeting` and showing the details to the user for explicit confirmation.
- If the user asks to modify a draft, DO NOT execute it. Create a new draft or just output the new drafted text and ask for confirmation again.

CRITICAL RULES for QA:
- Do not try to answer meeting questions from your own knowledge. ALWAYS delegate to the QA_Specialist.
""",
    tools=[
        FunctionTool(draft_email), 
        FunctionTool(execute_email), 
        FunctionTool(draft_meeting), 
        FunctionTool(execute_meeting), 
        FunctionTool(query_user_memory)
    ],
    sub_agents=[qa_specialist],
)
