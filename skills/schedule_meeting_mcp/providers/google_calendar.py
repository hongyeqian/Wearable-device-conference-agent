"""
google_calendar.py — Google Calendar provider using the Google Calendar API v3.

Requires: google-auth, google-api-python-client
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

from ..schemas import CalendarEventRequest, CalendarEventResult, CalendarProviderName
from ..auth.token_store import get_token, save_token, delete_token
from ..auth.google_oauth import refresh_access_token, get_auth_url
from .base import CalendarProvider


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_email(s: str) -> bool:
    return bool(_EMAIL_RE.match(s.strip()))


def _build_credentials(token_dict: dict):
    """Build a google.oauth2.credentials.Credentials object from a stored token dict."""
    from google.oauth2.credentials import Credentials

    creds = Credentials(
        token=token_dict.get("access_token"),
        refresh_token=token_dict.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
        scopes=[os.getenv("GOOGLE_CALENDAR_SCOPES",
                          "https://www.googleapis.com/auth/calendar.events")],
    )
    return creds


class GoogleCalendarProvider(CalendarProvider):
    """Creates real Google Calendar events via the Google Calendar API v3."""

    def get_provider_name(self) -> str:
        return CalendarProviderName.GOOGLE

    def check_auth(self, user_id: str) -> dict:
        """
        Returns {"connected": bool, "user_email": str|None, "auth_url": str|None}.
        If not connected, generates a fresh auth URL (without persisting a state here).
        """
        token = get_token(user_id, CalendarProviderName.GOOGLE)
        if not token:
            import secrets as _secrets
            state = _secrets.token_urlsafe(16)
            auth_url = get_auth_url(user_id, state)
            return {"connected": False, "user_email": None, "auth_url": auth_url}

        # Check expiry
        expires_at = token.get("expires_at")
        if expires_at and time.time() > expires_at - 60:
            # Try to refresh
            refresh_tok = token.get("refresh_token")
            if refresh_tok:
                try:
                    new_tok = refresh_access_token(refresh_tok)
                    new_tok["expires_at"] = time.time() + new_tok.get("expires_in", 3600)
                    new_tok["provider_user_email"] = token.get("provider_user_email")
                    new_tok.setdefault("refresh_token", refresh_tok)
                    save_token(user_id, CalendarProviderName.GOOGLE, new_tok)
                    token = get_token(user_id, CalendarProviderName.GOOGLE)
                except Exception:
                    # Refresh failed — treat as disconnected
                    import secrets as _secrets
                    state = _secrets.token_urlsafe(16)
                    auth_url = get_auth_url(user_id, state)
                    return {"connected": False, "user_email": None, "auth_url": auth_url}
            else:
                import secrets as _secrets
                state = _secrets.token_urlsafe(16)
                auth_url = get_auth_url(user_id, state)
                return {"connected": False, "user_email": None, "auth_url": auth_url}

        return {
            "connected": True,
            "user_email": token.get("provider_user_email"),
            "auth_url": None,
        }

    def create_event(self, request: CalendarEventRequest) -> CalendarEventResult:
        """Create a real Google Calendar event."""
        user_id = request.user_id

        # 1. Get stored token
        token = get_token(user_id, CalendarProviderName.GOOGLE)
        if not token:
            import secrets as _secrets
            state = _secrets.token_urlsafe(16)
            auth_url = get_auth_url(user_id, state)
            return CalendarEventResult(
                provider=CalendarProviderName.GOOGLE,
                status="auth_required",
                error_code="NO_TOKEN",
                error_message="Google Calendar is not connected. Please authorize first.",
                auth_url=auth_url,
            )

        # 2. Refresh if token is expired (with 60-second buffer)
        expires_at = token.get("expires_at")
        if expires_at and time.time() > expires_at - 60:
            refresh_tok = token.get("refresh_token")
            if not refresh_tok:
                import secrets as _secrets
                state = _secrets.token_urlsafe(16)
                auth_url = get_auth_url(user_id, state)
                return CalendarEventResult(
                    provider=CalendarProviderName.GOOGLE,
                    status="auth_required",
                    error_code="TOKEN_EXPIRED",
                    error_message="Google Calendar token expired and no refresh token available.",
                    auth_url=auth_url,
                )
            try:
                new_tok = refresh_access_token(refresh_tok)
                new_tok["expires_at"] = time.time() + new_tok.get("expires_in", 3600)
                new_tok["provider_user_email"] = token.get("provider_user_email")
                new_tok.setdefault("refresh_token", refresh_tok)
                save_token(user_id, CalendarProviderName.GOOGLE, new_tok)
                token = get_token(user_id, CalendarProviderName.GOOGLE)
            except Exception as exc:
                import secrets as _secrets
                state = _secrets.token_urlsafe(16)
                auth_url = get_auth_url(user_id, state)
                return CalendarEventResult(
                    provider=CalendarProviderName.GOOGLE,
                    status="auth_required",
                    error_code="TOKEN_REFRESH_FAILED",
                    error_message=f"Failed to refresh Google token: {exc}",
                    auth_url=auth_url,
                )

        # 3. Build attendees list — only include valid email addresses
        valid_attendees: list[str] = []
        non_email_participants: list[str] = []

        for item in (request.attendees or []):
            if _is_valid_email(item):
                valid_attendees.append(item)
            else:
                non_email_participants.append(item)

        # Also check participants list for any embedded emails
        for item in (request.participants or []):
            if _is_valid_email(item):
                if item not in valid_attendees:
                    valid_attendees.append(item)
            else:
                non_email_participants.append(item)

        warnings: list[str] = []
        if non_email_participants:
            warnings.append(
                f"The following participants were not added as attendees (not valid email addresses): "
                f"{', '.join(non_email_participants)}"
            )

        # 4. Build event description
        description = request.description or ""
        if non_email_participants:
            extra = "Participants (no email): " + ", ".join(non_email_participants)
            description = (description + "\n\n" + extra).strip() if description else extra

        # 5. Normalise start_time and determine end_time.
        # Google Calendar API requires RFC 3339 / full ISO format, e.g. "2026-05-08T15:00:00".
        # Inputs like "2026-05-08 15:00" (space separator, no seconds) are rejected with 400.
        from datetime import datetime, timedelta

        def _to_google_dt(s: str) -> str:
            """Normalise any ISO-ish datetime string to 'YYYY-MM-DDTHH:MM:SS'."""
            s = s.strip()
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                # Date-only "YYYY-MM-DD" → default to 09:00
                try:
                    from datetime import date as _date
                    dt = datetime.combine(_date.fromisoformat(s), datetime.min.time().replace(hour=9))
                except Exception:
                    return s  # give up, pass through
            return dt.strftime("%Y-%m-%dT%H:%M:%S")

        start_time = _to_google_dt(request.start_time)

        end_time = request.end_time
        if not end_time:
            try:
                duration = int(os.getenv("DEFAULT_EVENT_DURATION_MINUTES", "30"))
                end_time = (datetime.fromisoformat(start_time) + timedelta(minutes=duration)).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                end_time = start_time
        else:
            end_time = _to_google_dt(end_time)

        # 6. Build the event body for Google Calendar API
        event_body: dict = {
            "summary": request.title,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": request.timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": request.timezone,
            },
        }

        if request.location:
            event_body["location"] = request.location

        if valid_attendees:
            event_body["attendees"] = [{"email": e} for e in valid_attendees]

        # 7. Call the Google Calendar API
        try:
            from googleapiclient.discovery import build as _gapi_build
            creds = _build_credentials(token)

            insert_kwargs: dict = {
                "calendarId": "primary",
                "body": event_body,
                "sendUpdates": "all" if valid_attendees else "none",
            }

            if request.create_online_meeting:
                import uuid as _uuid
                event_body["conferenceData"] = {
                    "createRequest": {
                        "requestId": _uuid.uuid4().hex,
                        "conferenceSolutionKey": {"type": "hangoutsMeet"},
                    }
                }
                insert_kwargs["conferenceDataVersion"] = 1

            service = _gapi_build("calendar", "v3", credentials=creds)
            created = service.events().insert(**insert_kwargs).execute()

            return CalendarEventResult(
                provider=CalendarProviderName.GOOGLE,
                status="success",
                event_id=created.get("id"),
                html_link=created.get("htmlLink"),
                title=created.get("summary"),
                start_time=created.get("start", {}).get("dateTime"),
                end_time=created.get("end", {}).get("dateTime"),
                timezone=request.timezone,
                attendees=valid_attendees,
                warnings=warnings,
            )

        except Exception as exc:
            return CalendarEventResult(
                provider=CalendarProviderName.GOOGLE,
                status="error",
                error_code="API_ERROR",
                error_message=str(exc),
                warnings=warnings,
            )
