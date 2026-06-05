"""
Short Memory
------------
Context provider and pending-state builder.

This is a plain Python helper module — NOT an LLM Agent and NOT a class.

Role boundaries (after 2026-05-08 refactor)
-------------------------------------------
short_memory IS responsible for:
  - Reading pending_confirmation out of ADK session.state and exposing it via
    `get_short_context()`
  - Building a new pending_confirmation dict from a skill's structured_output
    (`build_pending_confirmation`)
  - Providing pure classification utilities for pending replies (rule + LLM)
    that the Intent Router can call

short_memory is NOT responsible for:
  - Deciding routing (skill / planner / pending_action) — that is intent_router
  - Generating final user-facing response text — that is web_app/agent.py
  - Mutating session state directly — caller (agent.py) attaches state_delta
    to the yielded Event

State persistence — WHY EventActions.state_delta is required
-------------------------------------------------------------
ADK's InMemorySessionService.get_session() returns copy.deepcopy(session) every
turn.  Direct mutation of ctx.session.state is therefore lost when the turn
ends.  The orchestrator (web_app/agent.py) attaches state_delta to yielded
Events; append_event() then applies it to storage_session so the next turn's
deep-copy includes the change.

Public API
----------
get_short_context(ctx) -> dict
    Snapshot of session-level context the Intent Router needs to make a
    routing decision.  Currently only carries `pending_confirmation`.

build_pending_confirmation(skill_name, user_query, skill_result) -> dict | None
    Build a pending_confirmation dict from a skill result, or return None.
    Caller persists it via EventActions.state_delta.

interpret_confirmation_reply(user_query, pending) -> dict
    Rule fast-path classifier for very short, unambiguous replies
    (yes/no/maybe/etc).  Return {reply_type, confidence, reason}.

classify_pending_context_reply(user_query, pending) -> dict
    LLM-based general classifier of a user message in pending context.
    Distinguishes direct pending answer / pending modification / new
    standalone request / unclear.  Return:
        {reply_relation, action, recipient_override,
         pending_to_clear, confidence, reason}.

_match_participants(names, pending_participants) -> list
    Whitelist-validate LLM-extracted recipient names against pending_participants.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based word sets for reply classification
# ---------------------------------------------------------------------------

_CONFIRM_EXACT: frozenset = frozenset({
    "yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "do it",
    "go ahead", "send it", "please do", "sounds good",
    "yes please", "please send", "send them", "send the invite",
})
_REJECT_EXACT: frozenset = frozenset({
    "no", "n", "nope", "cancel", "stop", "skip",
    "not now", "no need", "never mind", "nevermind",
    "don't send", "dont send", "do not send",
})
_CONFIRM_SINGLE: frozenset = frozenset({
    "yes", "yeah", "yep", "sure", "ok", "okay", "confirm",
})
_REJECT_SINGLE: frozenset = frozenset({
    "no", "nope", "cancel", "stop", "skip",
})
_NEW_REQUEST_STARTERS: frozenset = frozenset({
    "what", "who", "when", "where", "how", "why", "which",
    "schedule", "send", "search", "find", "show", "list",
    "tell", "get", "give", "create", "book", "arrange",
    "what's", "whats",
})


# ---------------------------------------------------------------------------
# Public API — context snapshot
# ---------------------------------------------------------------------------

def get_short_context(ctx: Any) -> dict:
    """
    Return a snapshot of session-level context the Intent Router needs to
    make a routing decision.

    Currently only carries `pending_confirmation`; future short-term context
    (e.g. `last_meeting_id`, recent topic, etc.) should be added here.

    The Intent Router treats this dict as read-only.
    """
    return {
        "pending_confirmation": ctx.session.state.get("pending_confirmation"),
    }


# ---------------------------------------------------------------------------
# Public API — pending classification utilities (pure functions)
# ---------------------------------------------------------------------------

def interpret_confirmation_reply(user_query: str, pending_confirmation: dict) -> dict:
    """
    Rule-based classifier for a user reply against an open pending_confirmation.

    Returns
    -------
    {
        "reply_type": "confirm" | "reject" | "unclear" | "new_request",
        "confidence": float,
        "reason":     str
    }

    Classification levels
    ---------------------
    1. Exact phrase exact-match sets  → confirm / reject (confidence 0.95)
    2. Short reply (≤ 3 words)
       a. Single word match           → confirm / reject (confidence 0.85)
       b. No match                    → unclear (0.5)
    3. Longer reply
       a. Starts with new-request word → new_request (0.85)
       b. Otherwise                    → new_request (0.70)
    """
    q = user_query.strip().lower().rstrip("!.,?")
    words = q.split()

    if q in _CONFIRM_EXACT:
        return {"reply_type": "confirm", "confidence": 0.95,
                "reason": f"exact affirmative match: '{q}'"}
    if q in _REJECT_EXACT:
        return {"reply_type": "reject", "confidence": 0.95,
                "reason": f"exact negative match: '{q}'"}

    if len(words) <= 3:
        if any(w in _CONFIRM_SINGLE for w in words):
            return {"reply_type": "confirm", "confidence": 0.85,
                    "reason": f"affirmative word in short reply: '{q}'"}
        if any(w in _REJECT_SINGLE for w in words):
            return {"reply_type": "reject", "confidence": 0.85,
                    "reason": f"negative word in short reply: '{q}'"}
        return {"reply_type": "unclear", "confidence": 0.50,
                "reason": f"short reply, no confirm/reject match: '{q}'"}

    first_word = words[0] if words else ""
    if first_word in _NEW_REQUEST_STARTERS:
        return {"reply_type": "new_request", "confidence": 0.85,
                "reason": f"starts with new-request word: '{first_word}'"}

    return {"reply_type": "new_request", "confidence": 0.70,
            "reason": f"longer query, treating as new request: '{q[:60]}'"}


def classify_pending_context_reply(user_query: str, pending: dict) -> dict:
    """
    Generic LLM classifier of a user message when a pending_confirmation is open.

    The classifier distinguishes THREE structural relations between the user
    message and the pending action — not just yes/no:

      1. direct_pending_answer
         The message is an answer to the pending question.
         e.g. "yes", "no", "maybe", "sure send it", "don't send".

      2. pending_modification
         The message modifies the pending action (typically narrowing the
         recipient list) but does NOT introduce a new independent action.
         e.g. "only send to Ankit", "yes but only Hongye".

      3. new_standalone_request
         The message expresses a complete new user intent (its own verb +
         object, or a stand-alone question), even if it mentions names that
         appear in the pending action.
         e.g. "Schedule a follow-up meeting with Hongye",
              "Send the meeting summary to Ankit",
              "What did Hongye say last meeting?".

      4. unclear
         The message cannot be confidently placed in any of the above.

    Output schema
    -------------
    {
      "reply_relation": "direct_pending_answer" | "pending_modification"
                        | "new_standalone_request" | "unclear",
      "action": "confirm" | "reject" | "unclear"
                | "confirm_with_recipient_override" | None,
      "recipient_override": list[str],
      "pending_to_clear": bool,
      "confidence": float,
      "reason": str,
    }

    Rules
    -----
    - direct_pending_answer  → action ∈ {confirm, reject, unclear},
                               pending_to_clear is False
    - pending_modification   → action = "confirm_with_recipient_override",
                               recipient_override is the names the user listed,
                               pending_to_clear is False
    - new_standalone_request → action = None,
                               pending_to_clear is True
    - unclear                → action = "unclear",
                               pending_to_clear is False (keep pending)

    Falls back to a safe unclear/keep-pending result on any error.
    """
    safe_fallback = {
        "reply_relation": "unclear",
        "action": "unclear",
        "recipient_override": [],
        "pending_to_clear": False,
        "confidence": 0.5,
        "reason": "classifier fallback",
    }

    try:
        import litellm
    except ImportError:
        return {**safe_fallback, "reason": "litellm not available"}

    participants = pending.get("participants", [])
    title        = pending.get("title", "")
    suggested_dt = pending.get("suggested_datetime") or ""
    pending_kind = pending.get("type", "send_meeting_notifications")

    prompt = (
        "You are a classifier for a meeting-intelligence assistant.\n"
        "There is an OPEN pending action awaiting the user's confirmation.\n"
        "Decide the STRUCTURAL relation between the user's NEW message and that pending action.\n\n"
        "Pending action context:\n"
        f"  Pending action type: {pending_kind}\n"
        f"  Meeting title:       {title or '(none)'}\n"
        f"  When:                {suggested_dt or '(not set)'}\n"
        f"  Pending participants:{', '.join(participants) if participants else '(none)'}\n\n"
        f"User's new message: \"{user_query}\"\n\n"
        "Pick exactly ONE reply_relation:\n\n"
        '  "direct_pending_answer"\n'
        "      The message is a direct answer to the pending question.\n"
        "      It does NOT introduce its own verb+object action.\n"
        '      Examples: "yes", "no", "maybe", "sure send it", "don\'t send", "ok go ahead".\n\n'
        '  "pending_modification"\n'
        "      The message modifies the pending action (typically narrows recipients)\n"
        "      but does NOT introduce a NEW independent action.\n"
        '      Examples: "only send to Ankit", "yes but only Hongye",\n'
        '                "send it just to Hongye Qian instead".\n'
        "      Only use this when the message clearly refers to the SAME pending notification\n"
        "      (e.g. via words like \"only\", \"just\", \"instead\", \"but to ...\")\n"
        "      AND the names listed appear in the pending participants above.\n\n"
        '  "new_standalone_request"\n'
        "      The message expresses a COMPLETE new intent — it has its own verb and object,\n"
        "      or it is a stand-alone question, EVEN IF it mentions names from the pending list.\n"
        "      Heuristics:\n"
        '        - Imperative starting with a new action verb ("Schedule ...", "Book ...",\n'
        '          "Find ...", "Send the meeting summary ...", "Search ...", "List ...").\n'
        "        - Question form (\"What did ...\", \"Who is ...\", \"How many ...\").\n"
        "        - The object/topic is DIFFERENT from the pending notification\n"
        '          (e.g. "send meeting summary to Ankit" is about email CONTENT, not this invite).\n\n'
        '  "unclear"\n'
        "      None of the above can be confidently determined.\n\n"
        "Then fill the output JSON.\n"
        "Return ONLY valid JSON (no markdown, no commentary):\n"
        "{\n"
        '  "reply_relation": "direct_pending_answer" | "pending_modification" | "new_standalone_request" | "unclear",\n'
        '  "action":         "confirm" | "reject" | "unclear" | "confirm_with_recipient_override" | null,\n'
        '  "recipient_override": [<names mentioned by the user>],\n'
        '  "pending_to_clear":  true | false,\n'
        '  "confidence":        <float 0.0-1.0>,\n'
        '  "reason":            "<one short sentence>"\n'
        "}\n\n"
        "Field rules:\n"
        "  - direct_pending_answer  → action ∈ {confirm, reject, unclear}, pending_to_clear=false\n"
        "  - pending_modification   → action='confirm_with_recipient_override',\n"
        "                             recipient_override = exact names the user listed,\n"
        "                             pending_to_clear=false\n"
        "  - new_standalone_request → action=null, pending_to_clear=true\n"
        "  - unclear                → action='unclear', pending_to_clear=false"
    )

    try:
        resp = litellm.completion(
            model=OPENAI_MODEL or "gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
    except Exception as exc:
        print(f"[ShortMemory] classify_pending_context_reply LLM error: {exc}")
        return {**safe_fallback, "reason": f"LLM call failed: {exc}"}

    relation = parsed.get("reply_relation")
    action   = parsed.get("action")
    if relation not in {
        "direct_pending_answer", "pending_modification",
        "new_standalone_request", "unclear",
    }:
        relation = "unclear"

    # Coerce action to be consistent with the relation (defence in depth)
    if relation == "new_standalone_request":
        action = None
    elif relation == "pending_modification":
        action = "confirm_with_recipient_override"
    elif relation == "direct_pending_answer":
        if action not in {"confirm", "reject", "unclear"}:
            action = "unclear"
    else:  # unclear
        action = "unclear"

    return {
        "reply_relation":     relation,
        "action":             action,
        "recipient_override": parsed.get("recipient_override") or [],
        "pending_to_clear":   bool(parsed.get("pending_to_clear", relation == "new_standalone_request")),
        "confidence":         float(parsed.get("confidence", 0.7)),
        "reason":             str(parsed.get("reason", "")),
    }


def _match_participants(names: list, pending_participants: list) -> list:
    """
    Whitelist-validate LLM-extracted recipient names against pending_participants.

    Matching is case-insensitive with word-boundary substring check so that
    "Ankit" matches "Ankit", "Hongye" matches "Hongye Qian", but "Lydia"
    matches nothing if Lydia is not in the list.

    Returns matched full participant names from pending_participants.
    """
    matched = []
    for name in names:
        nl = name.strip().lower()
        if not nl:
            continue
        for p in pending_participants:
            pl = p.lower()
            if nl == pl or re.search(r'\b' + re.escape(nl) + r'\b', pl):
                if p not in matched:
                    matched.append(p)
    return matched


# ---------------------------------------------------------------------------
# Public API — pending state builder
# ---------------------------------------------------------------------------

def build_pending_confirmation(
    skill_name: str,
    user_query: str,
    skill_result: dict,
) -> Optional[dict]:
    """
    Build a pending_confirmation dict from a skill result, or return None.

    The returned dict should be stored via EventActions.state_delta (not by
    direct ctx.session.state mutation) so that it survives to the next turn.
    """
    print(f"[ShortMemoryDebug] build_pending called: skill={skill_name!r}, status={skill_result.get('status')!r}")
    if skill_result.get("status") == "error":
        print("[ShortMemoryDebug] build_pending: returning None — status is error")
        return None

    structured = skill_result.get("structured_output") or {}
    follow_up = structured.get("follow_up_action")
    ftype = follow_up.get("type") if follow_up else None
    print(f"[ShortMemoryDebug] build_pending: structured keys={list(structured.keys())}, follow_up type={ftype!r}")

    if not follow_up or ftype != "send_email":
        print("[ShortMemoryDebug] build_pending: returning None — no send_email follow_up")
        return None

    pending = {
        "type": "send_meeting_notifications",
        "source_skill": skill_name,
        "original_query": user_query,
        "participants": follow_up.get("participants", []),
        "title": follow_up.get("title", ""),
        "suggested_datetime": follow_up.get("suggested_datetime"),
        "calendar_event_id": structured.get("event_id", ""),
        "suggested_query": follow_up.get("suggested_query", ""),
        "status": "awaiting_confirmation",
    }
    print(
        f"[ShortMemory] Built pending confirmation: type={pending['type']!r}, "
        f"skill={skill_name!r}, participants={pending['participants']}, title={pending['title']!r}"
    )
    logger.info(
        f"[ShortMemory] Built pending confirmation: {pending['type']} "
        f"(skill={skill_name}, participants={pending['participants']}, title={pending['title']!r})"
    )
    return pending
