"""
calendar_service.py — Orchestrator that routes calendar operations to the right provider.
"""
from __future__ import annotations

from .schemas import CalendarEventRequest, CalendarEventResult, CalendarProviderName, CalendarProviderStatus
from .auth.token_store import get_preferred_provider, init_db
from .providers.google_calendar import GoogleCalendarProvider
from .providers.outlook_calendar import OutlookCalendarProvider


def get_provider_status(user_id: str) -> list[CalendarProviderStatus]:
    """Return a list of CalendarProviderStatus for every supported provider."""
    init_db()
    statuses: list[CalendarProviderStatus] = []

    # Google
    google_auth = GoogleCalendarProvider().check_auth(user_id)
    statuses.append(CalendarProviderStatus(
        name=CalendarProviderName.GOOGLE,
        connected=google_auth["connected"],
        user_email=google_auth.get("user_email"),
        auth_url=google_auth.get("auth_url"),
    ))

    # Outlook (scaffold)
    outlook_auth = OutlookCalendarProvider().check_auth(user_id)
    statuses.append(CalendarProviderStatus(
        name=CalendarProviderName.OUTLOOK,
        connected=outlook_auth["connected"],
        user_email=outlook_auth.get("user_email"),
        auth_url=outlook_auth.get("auth_url"),
    ))

    return statuses


def create_calendar_event(request: CalendarEventRequest) -> CalendarEventResult:
    """Route the event creation request to the appropriate provider."""
    init_db()

    provider_name = (
        request.provider
        or get_preferred_provider(request.user_id)
        or CalendarProviderName.GOOGLE
    )

    if provider_name == CalendarProviderName.GOOGLE:
        return GoogleCalendarProvider().create_event(request)
    elif provider_name == CalendarProviderName.OUTLOOK:
        return OutlookCalendarProvider().create_event(request)
    else:
        return CalendarEventResult(
            provider=provider_name,
            status="error",
            error_code="UNKNOWN_PROVIDER",
            error_message=f"Unknown calendar provider: '{provider_name}'. Supported: google, outlook.",
        )
