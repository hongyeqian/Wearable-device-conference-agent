"""
schemas.py — Dataclass models for Calendar MCP integration.
Uses dataclasses (not pydantic) to avoid adding heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CalendarEventRequest:
    title: str
    start_time: str                         # ISO datetime string
    user_id: str = "local_user"
    provider: Optional[str] = None
    end_time: Optional[str] = None
    timezone: str = "Asia/Singapore"
    participants: List[str] = field(default_factory=list)   # display names
    attendees: List[str] = field(default_factory=list)      # valid email addresses only
    description: str = ""
    location: Optional[str] = None
    create_online_meeting: bool = False
    based_on_meeting: Optional[str] = None


@dataclass
class CalendarEventResult:
    provider: str
    status: str                             # "success" | "error" | "mock" | "auth_required"
    event_id: Optional[str] = None
    html_link: Optional[str] = None
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timezone: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    auth_url: Optional[str] = None


@dataclass
class CalendarProviderStatus:
    name: str
    connected: bool
    user_email: Optional[str] = None
    auth_url: Optional[str] = None


class CalendarProviderName:
    GOOGLE = "google"
    OUTLOOK = "outlook"
