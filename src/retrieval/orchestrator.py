import uuid, datetime
from config.settings import shared_session_service, APP_NAME, TURN_SESSION_INITIAL_STATE, USER_ID
from src.retrieval import query_rewriter_muti_agent

async def handle_user_query(user_query: str, rag_pipeline):
    """
    Unified orchestrator for RAG queries using consistent session keys.
    All components use the same APP_NAME and USER_ID from config.settings.
    """
    turn_session_id = f"turn-{uuid.uuid4().hex[:8]}"
    state = TURN_SESSION_INITIAL_STATE.copy()
    state["user_query"] = user_query
    state["now_str"] = datetime.datetime.utcnow().isoformat()
    state["meeting_catalog"] = query_rewriter_muti_agent.MEETING_CATALOG

    # create session using unified keys
    await shared_session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=turn_session_id, state=state)

    # call async pipeline and pass turn_session_id
    result = await rag_pipeline.query_async(user_query, turn_session_id=turn_session_id)

    # if answer_result includes updated session_state, persist it (update or delete+create)
    answer_result = result.get("answer_result") or {}
    session_state = answer_result.get("session_state")
    if session_state is not None:
        try:
            await shared_session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=turn_session_id)
        except Exception:
            pass
        await shared_session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=turn_session_id, state=session_state)

    return result