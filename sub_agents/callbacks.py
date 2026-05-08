"""
Skill Output Format Preservation Callback
------------------------------------------
before_agent_callback for answer_synthesis_agent.

When the Intent Router dispatches a skill (schedule_meeting, send_email, etc.)
the orchestrator sets three in-turn flags in ctx.session.state:

    ctx.session.state["preserve_skill_output_format"] = True
    ctx.session.state["protected_skill_output"]       = skill_result_dict
    ctx.session.state["protected_skill_name"]         = "schedule_meeting"

These are plain dict mutations (NOT via EventActions.state_delta), so they live
only for the duration of the current invocation.  They are NOT persisted to the
next turn.

This callback fires before answer_synthesis_agent executes.  If the flag is set
it returns the skill's text_output directly as a model Content object, bypassing
the LLM entirely.  If the flag is absent (normal RAG query) it returns None and
the agent runs as usual.

Clearing: writing via callback_context.state persists the cleared values through
state_delta, preventing stale flags from leaking if the flow ever changes.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

logger = logging.getLogger(__name__)


def preserve_skill_output_callback(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """
    before_agent_callback for answer_synthesis_agent.

    Returns Content (bypasses LLM) when a protected skill output is present.
    Returns None to let the agent run normally for ordinary RAG queries.
    """
    state = callback_context.state

    if not state.get("preserve_skill_output_format"):
        return None

    output: dict = state.get("protected_skill_output") or {}
    skill_name: str = state.get("protected_skill_name") or "unknown"

    # Clear flags via callback_context.state so they are written to state_delta
    # and won't survive past this event even if something unexpectedly persists.
    state["preserve_skill_output_format"] = False
    state["protected_skill_output"] = None
    state["protected_skill_name"] = None

    text: str = output.get("text_output") or json.dumps(
        output.get("structured_output", output), ensure_ascii=False, indent=2
    )

    print(
        f"[SkillOutputGuard] Bypassing LLM — returning protected output "
        f"for skill={skill_name!r} ({len(text)} chars)"
    )
    logger.info(
        "[SkillOutputGuard] before_agent_callback fired: protected output returned "
        f"for skill={skill_name!r}, LLM skipped"
    )

    return types.Content(role="model", parts=[types.Part(text=text)])
