"""
auth_web_server.py — FastAPI OAuth callback server.

Runs on port 8001 to handle Google (and future Microsoft) OAuth callbacks.

Start with:
    uvicorn skills.schedule_meeting_mcp.auth_web_server:app --port 8001 --reload

Or:
    python -m skills.schedule_meeting_mcp.auth_web_server
"""
from __future__ import annotations

import os
import secrets
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .auth.token_store import (
    init_db,
    save_token,
    save_oauth_state,
    validate_oauth_state,
    set_preferred_provider,
)
from .auth.google_oauth import get_auth_url, exchange_code_for_token, get_user_email


app = FastAPI(title="Calendar OAuth Server", version="1.0.0")


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "running", "service": "Calendar OAuth Server", "port": 8001}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

@app.get("/auth/google/start")
async def google_auth_start(request: Request, user_id: str = "local_user"):
    """
    Initiate the Google OAuth flow.
    Generates a state token, saves it, then redirects to Google's consent screen.
    """
    init_db()
    state = secrets.token_urlsafe(32)
    save_oauth_state(state, user_id, "google")
    auth_url = get_auth_url(user_id, state)
    return RedirectResponse(url=auth_url)


@app.get("/auth/google/callback")
async def google_auth_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
):
    """
    Handle the Google OAuth callback.
    Exchanges the authorization code for tokens and stores them.
    """
    if error:
        return HTMLResponse(
            content=_html_page(
                "Authorization Failed",
                f"<p>Google OAuth error: <strong>{error}</strong></p>"
                "<p>Please close this window and try again.</p>",
                success=False,
            ),
            status_code=400,
        )

    if not code or not state:
        return HTMLResponse(
            content=_html_page(
                "Bad Request",
                "<p>Missing <code>code</code> or <code>state</code> parameter.</p>",
                success=False,
            ),
            status_code=400,
        )

    # Validate state
    init_db()
    state_data = validate_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            content=_html_page(
                "Invalid State",
                "<p>The OAuth state token is invalid or has expired. "
                "Please start the authorization process again.</p>",
                success=False,
            ),
            status_code=400,
        )

    user_id = state_data["user_id"]

    # Exchange code for tokens
    try:
        token_data = exchange_code_for_token(code)
    except Exception as exc:
        return HTMLResponse(
            content=_html_page(
                "Token Exchange Failed",
                f"<p>Could not exchange authorization code: <strong>{exc}</strong></p>",
                success=False,
            ),
            status_code=500,
        )

    # Get user email
    access_token = token_data.get("access_token", "")
    user_email = get_user_email(access_token)

    # Compute absolute expiry timestamp
    expires_in = token_data.get("expires_in", 3600)
    token_data["expires_at"] = time.time() + expires_in
    token_data["provider_user_email"] = user_email

    # Persist token
    save_token(user_id, "google", token_data)
    set_preferred_provider(user_id, "google")

    email_display = user_email or "(unknown)"
    return HTMLResponse(
        content=_html_page(
            "Google Calendar Connected",
            f"<p>Successfully connected Google Calendar for <strong>{email_display}</strong>.</p>"
            "<p>You can now close this window and return to the application.</p>",
            success=True,
        )
    )


# ---------------------------------------------------------------------------
# Microsoft OAuth (scaffold)
# ---------------------------------------------------------------------------

@app.get("/auth/microsoft/start")
async def microsoft_auth_start():
    return HTMLResponse(
        content=_html_page(
            "Microsoft Calendar — Not Configured",
            "<p>Microsoft Calendar integration is not yet configured.</p>"
            "<p>To enable it, add <code>AZURE_CLIENT_ID</code>, <code>AZURE_CLIENT_SECRET</code>, "
            "and <code>AZURE_TENANT_ID</code> to <code>config/.env</code>.</p>",
            success=False,
        )
    )


@app.get("/auth/microsoft/callback")
async def microsoft_auth_callback():
    return HTMLResponse(
        content=_html_page(
            "Microsoft OAuth — Not Configured",
            "<p>Microsoft OAuth callback is not yet configured.</p>",
            success=False,
        )
    )


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _html_page(title: str, body: str, success: bool = True) -> str:
    color = "#2ecc71" if success else "#e74c3c"
    icon = "✓" if success else "✗"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           display: flex; align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; background: #f5f5f5; }}
    .card {{ background: white; border-radius: 12px; padding: 40px 48px;
             box-shadow: 0 4px 24px rgba(0,0,0,0.1); max-width: 480px; text-align: center; }}
    .icon {{ font-size: 48px; color: {color}; margin-bottom: 16px; }}
    h1 {{ color: #333; margin: 0 0 16px; font-size: 24px; }}
    p {{ color: #666; line-height: 1.6; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">{icon}</div>
    <h1>{title}</h1>
    {body}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="127.0.0.1", port=8001)
