"""
token_store.py — SQLite-backed token store for OAuth credentials.

Tables:
  calendar_tokens          — per-user, per-provider OAuth tokens
  user_calendar_preferences — preferred provider per user
  oauth_states             — short-lived PKCE/state values for OAuth flows
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

def _db_path() -> str:
    from config.settings import PROJECT_ROOT
    default = str(PROJECT_ROOT / "skills/schedule_meeting_mcp/auth/calendar_tokens.db")
    return os.getenv("CALENDAR_TOKEN_DB_PATH", default)


def _get_conn() -> sqlite3.Connection:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they do not already exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calendar_tokens (
                user_id             TEXT NOT NULL,
                provider            TEXT NOT NULL,
                provider_user_email TEXT,
                access_token        TEXT NOT NULL,
                refresh_token       TEXT,
                expires_at          REAL,
                scope               TEXT,
                token_type          TEXT DEFAULT 'Bearer',
                created_at          REAL NOT NULL,
                updated_at          REAL NOT NULL,
                PRIMARY KEY (user_id, provider)
            );

            CREATE TABLE IF NOT EXISTS user_calendar_preferences (
                user_id            TEXT PRIMARY KEY,
                preferred_provider TEXT NOT NULL,
                updated_at         REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS oauth_states (
                state      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                provider   TEXT NOT NULL,
                created_at REAL NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Token CRUD
# ---------------------------------------------------------------------------

def save_token(user_id: str, provider: str, token_data: dict) -> None:
    """Persist (or replace) a token for the given user+provider."""
    now = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO calendar_tokens
                (user_id, provider, provider_user_email, access_token, refresh_token,
                 expires_at, scope, token_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                provider_user_email = excluded.provider_user_email,
                access_token        = excluded.access_token,
                refresh_token       = COALESCE(excluded.refresh_token, refresh_token),
                expires_at          = excluded.expires_at,
                scope               = excluded.scope,
                token_type          = excluded.token_type,
                updated_at          = excluded.updated_at
            """,
            (
                user_id,
                provider,
                token_data.get("provider_user_email"),
                token_data.get("access_token", ""),
                token_data.get("refresh_token"),
                token_data.get("expires_at"),
                token_data.get("scope"),
                token_data.get("token_type", "Bearer"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_token(user_id: str, provider: str) -> Optional[dict]:
    """Return the stored token dict or None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM calendar_tokens WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def delete_token(user_id: str, provider: str) -> None:
    """Remove a stored token (disconnect / revoke locally)."""
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM calendar_tokens WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Provider preferences
# ---------------------------------------------------------------------------

def set_preferred_provider(user_id: str, provider: str) -> None:
    now = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO user_calendar_preferences (user_id, preferred_provider, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                preferred_provider = excluded.preferred_provider,
                updated_at         = excluded.updated_at
            """,
            (user_id, provider, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_preferred_provider(user_id: str) -> Optional[str]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT preferred_provider FROM user_calendar_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["preferred_provider"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# OAuth state management
# ---------------------------------------------------------------------------

_STATE_TTL_SECONDS = 600   # 10 minutes


def save_oauth_state(state: str, user_id: str, provider: str) -> None:
    """Persist a short-lived OAuth state token."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_states (state, user_id, provider, created_at, used) VALUES (?, ?, ?, ?, 0)",
            (state, user_id, provider, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def validate_oauth_state(state: str) -> Optional[dict]:
    """
    Validate an OAuth state token.

    Returns {"user_id": str, "provider": str} if valid and unused, else None.
    Marks the state as used on success to prevent replay.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM oauth_states WHERE state = ? AND used = 0",
            (state,),
        ).fetchone()
        if row is None:
            return None
        # Check expiry
        if time.time() - row["created_at"] > _STATE_TTL_SECONDS:
            return None
        # Mark used
        conn.execute(
            "UPDATE oauth_states SET used = 1 WHERE state = ?",
            (state,),
        )
        conn.commit()
        return {"user_id": row["user_id"], "provider": row["provider"]}
    finally:
        conn.close()
