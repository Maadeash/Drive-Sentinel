"""
Sign-in for the technician app.

A signed session cookie (Starlette's SessionMiddleware: itsdangerous-signed,
HttpOnly, SameSite=Lax), and one gate every request passes through. The gate
lets through, unauthenticated, exactly three things:

  /signin              the page and its form POST
  /static/*, favicon   stylesheet, scripts and icon -- no data in any of them
  /api/alerts/ingest   the DEVICE channel. A drive's outbox POSTs 21-byte alert
                       packets here; it has no browser session and must not
                       need one, or alerts would pile up in the outbox. It
                       accepts only a packet of exactly the wire size.

Everything else needs a session: a page without one is redirected to /signin
(with `next` so the user lands where they were going), an API call gets 401.

Pages live in webapp/pages/, NOT under /static, so no page HTML is reachable
around the gate.

Credentials come from the environment -- DRIVESENTINEL_USER and
DRIVESENTINEL_PASSWORD -- and default to technician / demo, documented in
webapp/README.md. They are read on every check, so a test (or an operator) can
set them without re-importing this module. The session
secret is DS_SESSION_SECRET when set, otherwise random per process -- so a
restart signs everyone out, which is the safe default for a laptop demo.
"""

from __future__ import annotations

import hmac
import os
import secrets
from typing import Dict, Optional
from urllib.parse import quote

from starlette.responses import JSONResponse, RedirectResponse

DEFAULT_USER = "technician"
DEFAULT_PASSWORD = "demo"


def users() -> Dict[str, Dict[str, str]]:
    """The one account, from DRIVESENTINEL_USER / DRIVESENTINEL_PASSWORD."""
    name = (os.environ.get("DRIVESENTINEL_USER") or DEFAULT_USER).strip().lower()
    password = os.environ.get("DRIVESENTINEL_PASSWORD") or DEFAULT_PASSWORD
    return {name: {"password": password, "name": name.capitalize()}}

SESSION_MAX_AGE_S = 8 * 3600
PUBLIC_PATHS = {"/signin", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)
DEVICE_PATHS = {"/api/alerts/ingest"}

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
]


def session_secret() -> str:
    return os.environ.get("DS_SESSION_SECRET") or secrets.token_hex(32)


def check(username: str, password: str) -> Optional[str]:
    """The canonical username on success, None otherwise. Constant-time compare."""
    u = users().get((username or "").strip().lower())
    ok = u is not None and hmac.compare_digest(
        (password or "").encode("utf-8"), u["password"].encode("utf-8"))
    return (username or "").strip().lower() if ok else None


def safe_next(target: Optional[str]) -> str:
    """Only a local path; never another origin, never back to sign-in."""
    t = target or ""
    if (not t.startswith("/") or t.startswith("//") or "\\" in t
            or t.startswith("/signin") or t.startswith("/signout")):
        return "/fleet"
    return t


def is_public(path: str) -> bool:
    return (path in PUBLIC_PATHS or path in DEVICE_PATHS
            or any(path.startswith(p) for p in PUBLIC_PREFIXES))


class AuthGate:
    """ASGI middleware. Must sit INSIDE SessionMiddleware (it reads the session)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        user = (scope.get("session") or {}).get("user")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += SECURITY_HEADERS
                # Pages and API: never stored. Static assets: stored, but revalidated
                # on every use (a 304 when unchanged), so a changed script -- a new
                # nav entry, say -- reaches a browser that already had the old one.
                headers.append((b"cache-control",
                                b"no-cache" if path.startswith("/static/") else b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        if is_public(path) or user in users():
            return await self.app(scope, receive, send_with_headers)
        if path.startswith("/api/"):
            resp = JSONResponse({"error": "sign-in required"}, status_code=401)
        else:
            qs = scope.get("query_string", b"").decode("latin-1")
            target = path + ("?" + qs if qs else "")
            resp = RedirectResponse("/signin?next=" + quote(target, safe=""),
                                    status_code=303)
        await resp(scope, receive, send_with_headers)
