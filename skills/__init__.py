"""
Skills package — integration-ready callable modules.

Each skill exposes:
    run(query: str, context: dict | None = None) -> dict

Use the registry to dispatch by name:
    from skills.registry import registry
    result = registry["search_action_items"].run(query, context)
"""
from skills.registry import registry, dispatch

__all__ = ["registry", "dispatch"]
