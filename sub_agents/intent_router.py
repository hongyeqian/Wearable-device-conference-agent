"""
Intent Router
-------------
Unified routing decision maker.

Inputs
------
- user_query (str)
- short_context (dict | None) — see sub_agents.short_memory.get_short_context()

Outputs (one of three route_types)
----------------------------------
1. pending_action — there is an awaiting pending_confirmation in short_context
                    and the user reply is interpreted as confirm / reject /
                    unclear / confirm_with_recipient_override.

2. skill          — query maps to a registered skill (rule or LLM scored).

3. planner        — fall back to planner / RAG.

Pipeline order
--------------
0. If short_context.pending_confirmation is present (status='awaiting_confirmation'):
     a. rule fast path (sub_agents.short_memory.interpret_confirmation_reply)
        — accepted only for unambiguous yes/no (reply_type in {confirm, reject}
        AND confidence >= 0.85). Returns route_type='pending_action'.
     b. otherwise the general pending-context classifier
        (sub_agents.short_memory.classify_pending_context_reply) is invoked.
        It returns a structural reply_relation:
          - direct_pending_answer  → route_type='pending_action'
          - pending_modification   → route_type='pending_action'
                                     (action=confirm_with_recipient_override)
          - new_standalone_request → fall through to skill/planner routing
                                     AND set pending_to_clear=True
          - unclear                → route_type='pending_action', action='unclear'
                                     (keep pending)

1. Rule scoring on registered skills (trigger_keywords + example overlap):
     - best_score >= HIGH_CONFIDENCE_THRESHOLD → return rule result immediately
     - best_score >= THRESHOLD                 → tentative skill result
     - best_score <  THRESHOLD                  → tentative planner result

2. If LLM_FALLBACK_ENABLED and best_score < HIGH_CONFIDENCE_THRESHOLD:
     - LLM router fallback. Receives compact skill metadata only.
     - skill_name is whitelist-validated.
     - 'clarify' is normalised to 'planner' with needs_clarification=True.

3. If LLM call fails → return rule result tagged router_source='fallback'.

Design notes
------------
- Fully data-driven for skill routing: reads metadata from the registry at
  call time. Adding a new skill requires only registry registration.
- short_memory provides the classification utilities; intent_router decides.
- agent.py executes the decision (no decision logic in agent.py).

Rule scoring
------------
Each candidate skill receives a float score in [0, 1]:
  +0.4 per trigger_keyword found verbatim in the lowercased query
  +0.2 per example whose word set overlaps the query by >= 2 words
  capped at 1.0

Thresholds
----------
THRESHOLD                 = 0.3  minimum rule score to consider a skill match
HIGH_CONFIDENCE_THRESHOLD = 0.7  rule score >= this → skip LLM fallback
LLM_FALLBACK_ENABLED      = True global toggle; set False to disable LLM calls
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7
LLM_FALLBACK_ENABLED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_intent(user_query: str, short_context: Optional[dict] = None) -> dict:
    """
    Make a unified routing decision.

    Parameters
    ----------
    user_query    : the raw user input for this turn
    short_context : optional snapshot from short_memory.get_short_context();
                    currently consulted for `pending_confirmation`

    Returns
    -------
    A dict whose `route_type` is one of:

    "pending_action":
        {
          "route_type": "pending_action",
          "action": "confirm" | "reject" | "unclear" | "confirm_with_recipient_override",
          "pending": <pending_confirmation dict>,
          "recipient_override": list[str],
          "confidence": float,
          "reason": str,
          "router_source": "rule" | "llm",
        }

    "skill":
        {
          "route_type": "skill",
          "skill_name": str,
          "confidence": "high" | "medium" | "low",
          "score": float,
          "reason": str,
          "router_source": "rule" | "llm" | "fallback",
          "needs_clarification": bool,
          "clarification_question": str | None,
          "pending_to_clear": bool,    # True when a pending was just resolved as new_request
        }

    "planner":
        {
          "route_type": "planner",
          "skill_name": None,
          "confidence": "high" | "medium" | "low",
          "score": float,
          "reason": str,
          "router_source": "rule" | "llm" | "fallback",
          "needs_clarification": bool,
          "clarification_question": str | None,
          "pending_to_clear": bool,
        }
    """
    # ----------------------------------------------------------------- #
    # Stage 0 — pending_action branch
    # ----------------------------------------------------------------- #
    pending_to_clear = False
    if short_context:
        pending = short_context.get("pending_confirmation")
        if pending and pending.get("status") == "awaiting_confirmation":
            pending_decision = _decide_pending_action(user_query, pending)
            if pending_decision is not None:
                return pending_decision
            # reply_type was 'new_request' — clear the pending and continue
            # routing the same query through skill/planner stages.
            pending_to_clear = True
            logger.info("[IntentRouter] Pending classified as new_request — clearing pending and routing as new query")
            print("[IntentRouter] Pending classified as new_request — clearing pending, routing as new query")

    # ----------------------------------------------------------------- #
    # Stage 1 — rule scoring
    # ----------------------------------------------------------------- #
    from skills.registry import get_enabled_skills

    skills = get_enabled_skills()
    if not skills:
        return _planner_result(0.0, "no skills registered", pending_to_clear=pending_to_clear)

    q = user_query.lower()
    scores = {name: _score(q, skill) for name, skill in skills.items()}
    best_name = max(scores, key=scores.__getitem__)
    best_score = scores[best_name]

    if best_score >= THRESHOLD:
        rule_result = {
            "route_type": "skill",
            "skill_name": best_name,
            "confidence": _confidence(best_score),
            "score": best_score,
            "reason": f"matched skill '{best_name}' (score={best_score:.2f})",
            "router_source": "rule",
            "needs_clarification": False,
            "clarification_question": None,
            "pending_to_clear": pending_to_clear,
        }
    else:
        rule_result = {
            "route_type": "planner",
            "skill_name": None,
            "confidence": "low",
            "score": best_score,
            "reason": "no skill exceeded threshold",
            "router_source": "rule",
            "needs_clarification": False,
            "clarification_question": None,
            "pending_to_clear": pending_to_clear,
        }

    # High-confidence rule match → skip LLM entirely
    if not LLM_FALLBACK_ENABLED or best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return rule_result

    # ----------------------------------------------------------------- #
    # Stage 2 — LLM router fallback
    # ----------------------------------------------------------------- #
    logger.info(
        f"[IntentRouter] Rule score={best_score:.2f} < {HIGH_CONFIDENCE_THRESHOLD} "
        f"— invoking LLM router fallback"
    )
    print(f"[IntentRouter] Rule score={best_score:.2f} — invoking LLM fallback")
    llm_result = _route_intent_llm(user_query, skills)
    if llm_result is not None:
        llm_result["pending_to_clear"] = pending_to_clear
        return llm_result

    print("[IntentRouter] LLM fallback failed — using rule result")
    return {**rule_result, "router_source": "fallback"}


# ---------------------------------------------------------------------------
# Stage 0 helper — pending_action decision
# ---------------------------------------------------------------------------

def _decide_pending_action(user_query: str, pending: dict) -> Optional[dict]:
    """
    Decide what a user message means in the context of an open pending action.

    Strategy
    --------
    1. Rule fast path (sub_agents.short_memory.interpret_confirmation_reply):
       only used for very short, unambiguous direct replies (yes/no/etc).
       Accepted only when reply_type is confirm/reject AND confidence >= 0.85.

    2. Otherwise the general LLM classifier
       (sub_agents.short_memory.classify_pending_context_reply) is consulted.
       It returns a structural reply_relation:
         - direct_pending_answer  → pending_action with action ∈ confirm/reject/unclear
         - pending_modification   → pending_action with action=confirm_with_recipient_override
         - new_standalone_request → caller falls through, pending_to_clear=True
         - unclear                → pending_action with action=unclear (keep pending)

    Returns
    -------
    A route_type='pending_action' dict, OR None when the message is a new
    standalone request and the caller should clear pending and re-route.
    """
    from sub_agents.short_memory import (
        interpret_confirmation_reply,
        classify_pending_context_reply,
    )

    # ---- Rule fast path: only for unambiguous yes/no ----------------------
    rule = interpret_confirmation_reply(user_query, pending)
    if rule["reply_type"] in {"confirm", "reject"} and rule["confidence"] >= 0.85:
        print(
            f"[IntentRouter] Pending rule fast path: action={rule['reply_type']!r} "
            f"(conf={rule['confidence']:.2f})"
        )
        logger.info(
            f"[IntentRouter] pending_action: action={rule['reply_type']!r} "
            f"(source=rule, conf={rule['confidence']:.2f})"
        )
        return {
            "route_type": "pending_action",
            "action": rule["reply_type"],
            "pending": pending,
            "recipient_override": [],
            "confidence": rule["confidence"],
            "reason": rule.get("reason", ""),
            "router_source": "rule",
        }

    # ---- General LLM classifier ------------------------------------------
    print(
        f"[IntentRouter] Pending rule did not match high-confidence yes/no "
        f"(reply_type={rule['reply_type']!r}, conf={rule['confidence']:.2f}) "
        f"— invoking general pending-context classifier"
    )
    classification = classify_pending_context_reply(user_query, pending)
    relation = classification["reply_relation"]
    action   = classification["action"]
    print(
        f"[IntentRouter] Pending LLM: relation={relation!r}, action={action!r}, "
        f"conf={classification['confidence']:.2f}, reason={classification.get('reason', '')!r}"
    )

    # 1. New standalone request → fall through to skill/planner routing.
    if relation == "new_standalone_request":
        return None

    # 2. All other relations resolve to a pending_action.
    logger.info(
        f"[IntentRouter] pending_action: relation={relation!r}, action={action!r} "
        f"(source=llm, conf={classification['confidence']:.2f})"
    )
    return {
        "route_type": "pending_action",
        "action": action,
        "pending": pending,
        "recipient_override": classification.get("recipient_override") or [],
        "confidence": classification["confidence"],
        "reason": classification.get("reason", ""),
        "router_source": "llm",
        "reply_relation": relation,
    }


# ---------------------------------------------------------------------------
# Stage 2 — LLM router fallback (skill / planner)
# ---------------------------------------------------------------------------

def _route_intent_llm(query: str, skills: dict) -> Optional[dict]:
    """
    LLM-based router fallback for skill/planner classification.

    Sends only the minimum necessary context: user query + compact skill metadata.
    No conversation history, no RAG context, no skill source code.

    Returns a result dict compatible with route_intent's schema, or None on error.
    """
    try:
        import litellm
        from config.settings import OPENAI_API_KEY, OPENAI_MODEL
    except ImportError:
        return None

    enabled_names = list(skills.keys())

    # Compact skill descriptions: name / description / keywords / up to 3 examples
    skill_lines = []
    for name, skill in skills.items():
        kws = ", ".join(getattr(skill, "trigger_keywords", ())) or "(none)"
        exs = "; ".join(list(getattr(skill, "examples", ()))[:3]) or "(none)"
        desc = getattr(skill, "description", "") or ""
        skill_lines.append(
            f"- name: {name}\n"
            f"  description: {desc}\n"
            f"  trigger_keywords: {kws}\n"
            f"  examples: {exs}"
        )
    skills_text = "\n".join(skill_lines)

    prompt = (
        "You are a routing classifier for a meeting-intelligence assistant.\n"
        "Classify the user query into one of the routing options below.\n\n"
        f"User query: \"{query}\"\n\n"
        "Available action skills:\n"
        f"{skills_text}\n\n"
        "Routing options:\n"
        '  "skill"   — user clearly wants to execute one of the listed action skills\n'
        '  "planner" — user is asking a question that needs knowledge retrieval/RAG\n'
        '               (e.g. questions about meeting content, summaries, history, tasks)\n'
        '  "clarify" — query is too ambiguous to route; ask the user to clarify\n\n'
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        "{\n"
        '  "route_type": "skill" | "planner" | "clarify",\n'
        '  "skill_name": "<exact name from available skills, or null>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reason": "<brief reason>",\n'
        '  "clarification_question": "<question to ask the user, or null>"\n'
        "}"
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

        rtype = parsed.get("route_type", "planner")
        if rtype not in {"skill", "planner", "clarify"}:
            rtype = "planner"

        skill_name = parsed.get("skill_name") or None
        confidence_val = float(parsed.get("confidence", 0.6))
        reason = str(parsed.get("reason", ""))
        clarification_q = parsed.get("clarification_question") or None

        # Whitelist validation — never trust LLM-returned skill name blindly
        if rtype == "skill":
            if not skill_name or skill_name not in enabled_names:
                logger.warning(
                    f"[IntentRouter] LLM returned unknown skill '{skill_name}' "
                    f"(enabled: {enabled_names}) — downgrading to planner"
                )
                print(
                    f"[IntentRouter] LLM skill '{skill_name}' not in whitelist "
                    f"{enabled_names} — downgrading to planner"
                )
                rtype = "planner"
                skill_name = None
        else:
            skill_name = None

        # Normalise 'clarify' → 'planner' for downstream compatibility
        needs_clarification = (rtype == "clarify")
        if rtype == "clarify":
            rtype = "planner"

        print(
            f"[IntentRouter] LLM: route_type={rtype!r}, skill={skill_name!r}, "
            f"conf={confidence_val:.2f}, reason={reason!r}"
        )
        logger.info(
            f"[IntentRouter] LLM result: route_type={rtype!r}, skill_name={skill_name!r}, "
            f"confidence={confidence_val:.2f}"
        )

        return {
            "route_type": rtype,
            "skill_name": skill_name,
            "confidence": _confidence(confidence_val),
            "score": confidence_val,
            "reason": f"[LLM] {reason}",
            "router_source": "llm",
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_q,
        }

    except Exception as exc:
        logger.warning(f"[IntentRouter] LLM fallback error: {exc}")
        print(f"[IntentRouter] LLM fallback error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(q: str, skill) -> float:
    score = 0.0
    for kw in getattr(skill, "trigger_keywords", ()):
        if kw.lower() in q:
            score += 0.4
    for ex in getattr(skill, "examples", ()):
        overlap = len(set(q.split()) & set(ex.lower().split()))
        if overlap >= 2:
            score += 0.2
    return min(score, 1.0)


def _confidence(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _planner_result(score: float, reason: str, pending_to_clear: bool = False) -> dict:
    return {
        "route_type": "planner",
        "skill_name": None,
        "confidence": "low",
        "score": score,
        "reason": reason,
        "router_source": "rule",
        "needs_clarification": False,
        "clarification_question": None,
        "pending_to_clear": pending_to_clear,
    }
