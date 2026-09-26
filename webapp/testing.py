"""
Run the app on a real port inside a test, and talk to it with urllib.

Starlette's in-process TestClient needs httpx, which is not installed; a real
server is the better test anyway, because the alert pipeline's whole point is
the HTTP POST and the resend when the endpoint is down.

`LiveServer` holds a cookie jar, so a test signs in once with `login()` and
every later request carries the session. `raw()` never follows redirects, so a
test can see the 303 to /signin itself.
"""

from __future__ import annotations

import http.client
import http.cookiejar
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class LiveServer:
    def __init__(self, app, port: Optional[int] = None):
        import uvicorn
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1",
                                                     port=self.port,
                                                     log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.jar = http.cookiejar.CookieJar()
        cookies = urllib.request.HTTPCookieProcessor(self.jar)
        self._open = urllib.request.build_opener(cookies)
        self._open_raw = urllib.request.build_opener(cookies, _NoRedirect)

    def __enter__(self):
        self._thread.start()
        t0 = time.time()
        while not self._server.started:
            if time.time() - t0 > 20:
                raise RuntimeError("server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)

    # -- session ---------------------------------------------------------------
    def login(self, username: str = "technician", password: str = "demo",
              next_path: str = "/fleet") -> Tuple[int, str]:
        """POST the sign-in form. Returns (status, Location) without following."""
        data = urllib.parse.urlencode({"username": username, "password": password,
                                       "next": next_path}).encode()
        status, headers, _ = self.raw("/signin", method="POST", data=data,
                                      ctype="application/x-www-form-urlencoded")
        return status, headers.get("Location", "")

    def logout(self) -> Tuple[int, str]:
        status, headers, _ = self.raw("/signout", method="POST")
        return status, headers.get("Location", "")

    # -- requests --------------------------------------------------------------
    def raw(self, path: str, method: str = "GET", data: Optional[bytes] = None,
            ctype: Optional[str] = None) -> Tuple[int, "http.client.HTTPMessage", bytes]:
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": ctype} if ctype else {})
        try:
            with self._open_raw.open(req, timeout=30) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:                  # 3xx/4xx land here
            return e.code, e.headers, e.read()

    def get(self, path: str):
        with self._open.open(self.url + path, timeout=30) as r:
            body = r.read()
            ctype = r.headers.get("Content-Type", "")
        return json.loads(body) if "json" in ctype else body.decode("utf-8")

    def post(self, path: str, data: bytes = b"", ctype="application/octet-stream"):
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": ctype})
        with self._open.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
