"""
Base skill interface.

All skills must implement:
    run(query: str, context: dict | None = None) -> dict

Return contract
---------------
Success:
    {
        "skill_name": str,
        "status": "success",
        "rewritten_query": str | None,
        "meeting_ids": list[str] | None,
        "structured_output": dict,
        "text_output": str,
        "error": None,
    }

Error:
    {
        "skill_name": str,
        "status": "error",
        "rewritten_query": None,
        "meeting_ids": None,
        "structured_output": None,
        "text_output": None,
        "error": {"code": str, "message": str},
    }
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Abstract base for all skills."""

    name: str = ""
    description: str = ""
    trigger_keywords: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    priority: int = 0
    enabled: bool = True

    @abstractmethod
    def run(self, query: str, context: dict | None = None) -> dict:
        """Execute the skill and return a structured result dict."""
        ...

    # ------------------------------------------------------------------ #
    # Helpers for building canonical response dicts                        #
    # ------------------------------------------------------------------ #

    def _success(
        self,
        structured_output: dict,
        text_output: str,
        rewritten_query: str | None = None,
        meeting_ids: list | None = None,
        **extra: Any,
    ) -> dict:
        return {
            "skill_name": self.name,
            "status": "success",
            "rewritten_query": rewritten_query,
            "meeting_ids": meeting_ids,
            "structured_output": structured_output,
            "text_output": text_output,
            "error": None,
            **extra,
        }

    def _error(self, code: str, message: str) -> dict:
        return {
            "skill_name": self.name,
            "status": "error",
            "rewritten_query": None,
            "meeting_ids": None,
            "structured_output": None,
            "text_output": None,
            "error": {"code": code, "message": message},
        }
