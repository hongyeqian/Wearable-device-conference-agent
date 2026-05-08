"""
base.py — Abstract base class for calendar providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import CalendarEventRequest, CalendarEventResult


class CalendarProvider(ABC):
    """Abstract base for all calendar provider implementations."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the canonical provider name (e.g. 'google', 'outlook')."""
        ...

    @abstractmethod
    def check_auth(self, user_id: str) -> dict:
        """
        Check whether the user has valid credentials for this provider.

        Returns a dict:
          {
            "connected":  bool,
            "user_email": str | None,
            "auth_url":   str | None,   # populated when not connected
          }
        """
        ...

    @abstractmethod
    def create_event(self, request: CalendarEventRequest) -> CalendarEventResult:
        """Create a calendar event and return a CalendarEventResult."""
        ...
