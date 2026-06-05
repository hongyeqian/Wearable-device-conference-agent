"""
microsoft_oauth.py — Microsoft / Outlook OAuth 2.0 scaffold.

This module is a stub. Microsoft Calendar integration requires:
  AZURE_CLIENT_ID
  AZURE_CLIENT_SECRET
  AZURE_TENANT_ID         (or "common" for multi-tenant)
  AZURE_REDIRECT_URI      (default: http://127.0.0.1:8001/auth/microsoft/callback)

Add these to config/.env and implement the functions below when ready.
"""
from __future__ import annotations

import os
from typing import Optional


_NOT_CONFIGURED_MSG = (
    "Microsoft Calendar OAuth is not yet configured. "
    "Set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID in config/.env, "
    "then implement this module."
)


def get_auth_url(user_id: str, state: str) -> str:
    """Build a Microsoft OAuth 2.0 authorization URL (stub)."""
    raise NotImplementedError(_NOT_CONFIGURED_MSG)


def exchange_code_for_token(code: str) -> dict:
    """Exchange an authorization code for tokens (stub)."""
    raise NotImplementedError(_NOT_CONFIGURED_MSG)


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh a Microsoft access token (stub)."""
    raise NotImplementedError(_NOT_CONFIGURED_MSG)


def get_user_email(access_token: str) -> Optional[str]:
    """Fetch the Microsoft account email (stub)."""
    raise NotImplementedError(_NOT_CONFIGURED_MSG)
