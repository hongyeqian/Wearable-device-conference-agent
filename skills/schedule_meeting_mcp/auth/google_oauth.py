"""
google_oauth.py — Google OAuth 2.0 flow helpers.

Uses `requests` directly (not google-auth-oauthlib) for simplicity.
Credentials are read from environment / config/.env:
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_REDIRECT_URI       (default: http://127.0.0.1:8001/auth/google/callback)
  GOOGLE_CALENDAR_SCOPES    (default: https://www.googleapis.com/auth/calendar.events)
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlencode

import requests

from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / "config" / ".env", override=False)

# ---------------------------------------------------------------------------
# Constant endpoints
# ---------------------------------------------------------------------------

_GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _client_id() -> str:
    v = os.getenv("GOOGLE_CLIENT_ID", "")
    if not v:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID is not set. Add it to config/.env."
        )
    return v


def _client_secret() -> str:
    v = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not v:
        raise RuntimeError(
            "GOOGLE_CLIENT_SECRET is not set. Add it to config/.env."
        )
    return v


def _redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8001/auth/google/callback")


def _scopes() -> str:
    return os.getenv(
        "GOOGLE_CALENDAR_SCOPES",
        "https://www.googleapis.com/auth/calendar.events",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_auth_url(user_id: str, state: str) -> str:
    """
    Build a Google OAuth 2.0 authorization URL.

    The returned URL should be visited by the end-user in their browser.
    `state` must be validated on the callback to prevent CSRF.
    """
    params = {
        "client_id":     _client_id(),
        "redirect_uri":  _redirect_uri(),
        "response_type": "code",
        "scope":         _scopes(),
        "access_type":   "offline",   # request a refresh_token
        "prompt":        "consent",   # force consent so refresh_token is always returned
        "state":         state,
    }
    return f"{_GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    """
    Exchange an authorization code for access + refresh tokens.

    Returns a dict with keys:
      access_token, refresh_token, expires_in, scope, token_type
    Raises RuntimeError on failure.
    """
    resp = requests.post(
        _GOOGLE_TOKEN_ENDPOINT,
        data={
            "code":          code,
            "client_id":     _client_id(),
            "client_secret": _client_secret(),
            "redirect_uri":  _redirect_uri(),
            "grant_type":    "authorization_code",
        },
        timeout=15,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(
            f"Google token exchange failed: {data.get('error')} — {data.get('error_description', '')}"
        )
    return data


def refresh_access_token(refresh_token: str) -> dict:
    """
    Use a refresh token to obtain a new access token.

    Returns a dict with at least: access_token, expires_in, token_type.
    Raises RuntimeError on failure.
    """
    resp = requests.post(
        _GOOGLE_TOKEN_ENDPOINT,
        data={
            "refresh_token": refresh_token,
            "client_id":     _client_id(),
            "client_secret": _client_secret(),
            "grant_type":    "refresh_token",
        },
        timeout=15,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(
            f"Google token refresh failed: {data.get('error')} — {data.get('error_description', '')}"
        )
    return data


def get_user_email(access_token: str) -> Optional[str]:
    """
    Fetch the authenticated user's email address from Google's userinfo endpoint.

    Returns the email string, or None if the request fails.
    """
    try:
        resp = requests.get(
            _GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        data = resp.json()
        return data.get("email")
    except Exception:
        return None
