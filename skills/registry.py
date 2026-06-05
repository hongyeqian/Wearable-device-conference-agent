"""
Skill Registry & Dispatcher
----------------------------
Central lookup table for all registered skills.

Usage
-----
    from skills.registry import registry, dispatch

    # Dict-style access
    result = registry["search_action_items"].run(query, context)

    # Dispatch helper (name + query + optional context)
    result = dispatch("send_email", query="Send summary to Hongye")

Adding a new skill
------------------
1. Create skills/<your_skill>.py with a module-level `skill` singleton.
2. Import it here and add to _SKILL_MODULES.

ADK / planner integration notes
---------------------------------
The registry dict is intentionally flat and keyed by string — it can be wrapped
as a Google ADK FunctionTool or passed to any planner that resolves skill names
at runtime.  Each skill's run() signature is:

    run(query: str, context: dict | None = None) -> dict

which maps directly to a single-argument ADK tool call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skills.base import BaseSkill

# ---------------------------------------------------------------------------
# Register skills here — lazy import so the module loads even when a skill's
# optional heavy dependency (e.g. litellm) is unavailable.
# ---------------------------------------------------------------------------

def _load_skills() -> dict[str, BaseSkill]:
    from skills.search_action_items import skill as search_skill
    from skills.send_email import skill as email_skill
    from skills.schedule_meeting import skill as calendar_skill
    return {
        search_skill.name: search_skill,
        email_skill.name: email_skill,
        calendar_skill.name: calendar_skill,
    }


# Singleton registry dict — populated on first access
_registry: dict[str, BaseSkill] | None = None


def _get_registry() -> dict[str, BaseSkill]:
    global _registry
    if _registry is None:
        _registry = _load_skills()
    return _registry


class _RegistryProxy:
    """
    Thin proxy so callers can do `registry["skill_name"]` or iterate skills.
    Initialisation is deferred until first access.
    """

    def __getitem__(self, name: str) -> BaseSkill:
        r = _get_registry()
        if name not in r:
            raise KeyError(
                f"Skill '{name}' not found. Available: {list(r.keys())}"
            )
        return r[name]

    def __contains__(self, name: str) -> bool:
        return name in _get_registry()

    def keys(self):
        return _get_registry().keys()

    def values(self):
        return _get_registry().values()

    def items(self):
        return _get_registry().items()

    def list_skills(self) -> list[str]:
        return list(_get_registry().keys())


registry = _RegistryProxy()


def get_enabled_skills() -> dict[str, BaseSkill]:
    """Return only registered skills where enabled=True."""
    return {k: v for k, v in _get_registry().items() if getattr(v, "enabled", True)}


# ---------------------------------------------------------------------------
# Convenience dispatch function
# ---------------------------------------------------------------------------

def dispatch(
    skill_name: str,
    query: str,
    context: dict | None = None,
    **kwargs: Any,
) -> dict:
    """
    Look up a skill by name and call run().

    Parameters
    ----------
    skill_name : registered skill identifier
    query      : natural-language query / instruction
    context    : optional structured context dict
    **kwargs   : merged into context (convenience for callers)

    Returns
    -------
    Canonical result dict (see skills/base.py for schema).
    """
    if kwargs:
        context = {**(context or {}), **kwargs}
    try:
        skill = registry[skill_name]
    except KeyError:
        return {
            "skill_name": skill_name,
            "status": "error",
            "rewritten_query": None,
            "meeting_ids": None,
            "structured_output": None,
            "text_output": None,
            "error": {
                "code": "SKILL_NOT_FOUND",
                "message": (
                    f"'{skill_name}' is not registered. "
                    f"Available skills: {registry.list_skills()}"
                ),
            },
        }
    return skill.run(query, context)
