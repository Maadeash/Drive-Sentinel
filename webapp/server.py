"""
DriveSentinel technician web app -- HTTP layer.

    .venv/Scripts/python.exe -m webapp            (http://127.0.0.1:8000)

ASGI on Starlette + uvicorn, both already in the project environment. FastAPI is
Starlette underneath; it is not installed here, so the app is written directly
against Starlette. Porting is a change of import, not of design.

Every page has its own URL, so back/forward and bookmarks work without a
client-side router. Pages are plain HTML in webapp/pages/ (behind the sign-in
gate); CSS and JS are in webapp/static/ -- no build step, no CDN, no external
request of any kind.

The user-facing API returns only what `fleet.Fleet` lets through: no file paths,
no internal IDs, no fault codes. The engineering view's API is the exception and
is reached only from the footer link.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Optional
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from drivesentinel import alerts as A
from drivesentinel.engines import ENGINE_SOFTWARE, EngineRouter
from drivesentinel.sources import BoardSource, RecordedScenarioSource
from . import auth, content
from .boardlive import BoardFeed
from .fleet import LIFTS, Fleet, SLUG_STAGE, lift_label
from .service import VAR, Service, jsonable

ENGINE_FILE = os.path.join(VAR, "engine.json")


def load_board_address() -> str:
    """DS_BOARD_ADDRESS if set, else the address last saved on Settings, else none."""
    env = os.environ.get("DS_BOARD_ADDRESS")
    if env is not None:
        return env.strip()
    try:
        with open(ENGINE_FILE) as fh:
            return str(json.load(fh).get("address") or "")
    except (OSError, ValueError):
        return ""


def save_board_address(address: str) -> None:
    os.makedirs(VAR, exist_ok=True)
    with open(ENGINE_FILE, "w") as fh:
        json.dump({"address": address}, fh)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
PAGES_DIR = os.path.join(HERE, "pages")

BRAND = {
    "name": "DriveSentinel",
    "footer": "Prepared for KONE Elevate'26 · Problem Statement 06",
}

# path -> page file. Parameterised pages are routed separately below.
PAGES = {
    "/fleet": "fleet.html", "/alerts": "alerts.html", "/monitor": "monitor.html",
    "/hardware": "hardware.html", "/about": "about.html",
    "/settings": "settings.html", "/board": "board.html",
    "/engineering": "engineering.html",
}
USER_PAGES = [p for p in PAGES if p != "/engineering"] + ["/lifts/{slug}", "/alerts/{ref}"]

# Addresses from the previous version of the app keep working as bookmarks.
LEGACY = {"/replay": "/monitor", "/lift": "/fleet", "/alert": "/alerts",
          "/sensors": "/hardware"}          # the Sensors page was removed on 2026-09-24


def create_app(service: Optional[Service] = None, run_fleet: bool = True,
               session_secret: Optional[str] = None) -> Starlette:
    state = {"svc": service, "fleet": Fleet(service) if service else None}

    def svc() -> Service:
        return state["svc"]

    def fleet() -> Fleet:
        return state["fleet"]

    def J(obj, status=200):
        return JSONResponse(jsonable(obj), status_code=status)

    def not_found():
        return J({"error": "not found"}, 404)

    # -- pages ---------------------------------------------------------------
    # Every script and stylesheet link carries its file's modification time, so a
    # changed asset is a new URL: a browser holding an old copy (heuristically
    # cached before the no-cache header existed) cannot keep running it.
    asset_ref = re.compile(r'"/static/([\w./-]+\.(?:js|css))"')

    def versioned(m):
        path = os.path.join(STATIC, *m.group(1).split("/"))
        v = int(os.path.getmtime(path)) if os.path.exists(path) else 0
        return f'"/static/{m.group(1)}?v={v}"'

    def page_file(name):
        with open(os.path.join(PAGES_DIR, name), encoding="utf-8") as fh:
            return HTMLResponse(asset_ref.sub(versioned, fh.read()))

    def page(request: Request):
        return page_file(PAGES[request.url.path])

    def lift_page(request: Request):
        if request.path_params["slug"] not in fleet().lifts:
            return RedirectResponse("/fleet", status_code=303)
        return page_file("lift.html")

    def alert_page(request: Request):
        return page_file("alert.html")

    def home(request: Request):
        return RedirectResponse("/fleet", status_code=303)

    def legacy(request: Request):
        return RedirectResponse(LEGACY[request.url.path], status_code=301)

    # -- sign-in ---------------------------------------------------------------
    async def signin(request: Request):
        nxt = auth.safe_next(request.query_params.get("next"))
        if request.method == "GET":
            if request.session.get("user") in auth.users():
                return RedirectResponse(nxt, status_code=303)
            return page_file("signin.html")
        form = await request.form()
        nxt = auth.safe_next(form.get("next") or request.query_params.get("next"))
        user = auth.check(str(form.get("username", "")), str(form.get("password", "")))
        if user is None:
            return RedirectResponse(f"/signin?error=1&next={quote(nxt, safe='')}",
                                    status_code=303)
        request.session.clear()
        request.session["user"] = user
        return RedirectResponse(nxt, status_code=303)

    async def signout(request: Request):
        request.session.clear()
        return RedirectResponse("/signin?signed_out=1", status_code=303)

    async def session_info(request: Request):
        u = request.session.get("user")
        return J({"user": u, "name": auth.users()[u]["name"]})

    # -- user-facing API -------------------------------------------------------
    async def meta(request):
        s = svc()
        return J({"brand": BRAND, "stages": fleet().stages_meta(),
                  "status_order": ["ALARM", "ADVISORY", "NORMAL", "NO DATA"],
                  "alert_types": A.ALERT_TYPE,
                  "streaming": not s.fleet_done,
                  "user": auth.users()[request.session["user"]]["name"]})

    async def fleet_overview(request):
        return J(fleet().overview())

    async def lift(request):
        d = fleet().lift(request.path_params["slug"])
        return J(d) if d else not_found()

    async def alerts(request):
        q = request.query_params
        return J(fleet().alerts(q.get("status") or None, q.get("stage") or None,
                                q.get("lift") or None))

    async def alert_detail(request):
        d = fleet().alert_detail(request.path_params["ref"])
        return J(d) if d else not_found()

    async def monitor_options(request):
        return J(fleet().monitor_options())

    async def monitor(request):
        p = request.path_params
        if p["stage"] not in SLUG_STAGE:
            return not_found()
        try:
            since = max(0, int(request.query_params.get("since") or 0))
        except ValueError:
            since = 0
        d = fleet().monitor(p["lift"], p["stage"], since)
        return J(d) if d else not_found()

    async def hardware(request):
        return J(content.hardware())

    async def about(request):
        return J(content.about(svc().metrics))

    async def settings(request):
        return J(content.settings(svc()))

    async def restart(request):
        svc().restart_fleet()
        return J({"restarted": True})

    # -- inference engine for the bearing stage (Settings) ------------------------
    def router():
        return getattr(svc().source, "router", None)

    async def engine_get(request):
        r = router()
        if r is None:          # a service built without the board path
            return J({"active": ENGINE_SOFTWARE, "address": "", "board": "not set",
                      "build_id": None, "windows": {}, "last": None})
        return J(r.state())

    async def engine_set(request):
        r = router()
        if r is None:
            return J({"error": "no engine router"}, 409)
        try:
            body = await request.json()
        except Exception:
            return J({"error": "expected JSON {\"address\": ...}"}, 400)
        address = str(body.get("address") or "").strip()
        if len(address) > 200 or any(c.isspace() for c in address):
            return J({"error": "invalid address"}, 400)
        r.set_address(address)
        save_board_address(address)
        return J(r.state())

    # -- board live: a view of the bearing feed the fleet stream runs on ------------
    async def live_get(request):
        feed = getattr(svc().source, "feed", None)
        s = feed.state() if feed is not None else {
            "status": "off", "reason": "This app was started without the board feed.",
            "detail": None, "board": None, "now": None, "lifts": {}, "events": [],
            "totals": {}, "timing": {}, "raw_view": None}
        f = fleet()
        s["lifts"] = [dict(v, status=(f._current(int(k))[1] or {}).get("status", "NO DATA"))
                      for k, v in sorted((s.get("lifts") or {}).items(), key=lambda kv: kv[1]["label"])]
        s["alerts"] = f.alerts(stage_slug="bearing")[:10]
        n = next((n for (l, sl), n in f.unit_of.items() if sl == "bearing"), None)
        s["meta"] = (f._signal_meta("bearing", svc().timelines[n]["meta"])
                     if n is not None else None)
        s["streaming"] = not svc().fleet_done
        return J(s)

    # -- engineering view (footer link) -------------------------------------------
    async def engineering(request):
        return J(content.engineering(svc().metrics))

    # -- device channel: alert packets from a drive's outbox -----------------------
    async def ingest(request):
        body = await request.body()
        if len(body) != A.WIRE_BYTES:
            return J({"error": f"expected a {A.WIRE_BYTES}-byte alert packet"}, 400)
        try:
            return J(svc().ingest(body))
        except (ValueError, KeyError, IndexError) as e:
            return J({"error": str(e)}, 400)

    def favicon(request):                    # browsers ask for it regardless of <link>
        return FileResponse(os.path.join(STATIC, "favicon.svg"), media_type="image/svg+xml")

    routes = (
        [Route("/", home), Route("/signin", signin, methods=["GET", "POST"]),
         Route("/signout", signout, methods=["GET", "POST"])]
        + [Route(p, page) for p in PAGES]
        + [Route("/lifts/{slug}", lift_page), Route("/alerts/{ref}", alert_page)]
        + [Route(p, legacy) for p in LEGACY]
        + [Route("/favicon.ico", favicon),
           Route("/api/session", session_info),
           Route("/api/meta", meta),
           Route("/api/fleet", fleet_overview),
           Route("/api/lifts/{slug}", lift),
           Route("/api/alerts", alerts),
           Route("/api/alerts/ingest", ingest, methods=["POST"]),
           Route("/api/alerts/{ref}", alert_detail),
           Route("/api/monitor", monitor_options),
           Route("/api/monitor/{lift}/{stage}", monitor),
           Route("/api/hardware", hardware),
           Route("/api/about", about),
           Route("/api/settings", settings),
           Route("/api/stream/restart", restart, methods=["POST"]),
           Route("/api/engine", engine_get, methods=["GET"]),
           Route("/api/engine", engine_set, methods=["POST"]),
           Route("/api/board/live", live_get, methods=["GET"]),
           Route("/api/engineering", engineering),
           Mount("/static", StaticFiles(directory=STATIC), name="static")])

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if state["svc"] is None:
            port = int(os.environ.get("DS_PORT", "8000"))
            # verdict="engine": the deployed network drives the bearing stage --
            # the owner's decision of 2026-09-24; in-sample on this data
            # (drivesentinel/sources.py BoardSource, claims_audit 1.28).
            base = RecordedScenarioSource()
            router = EngineRouter(address=load_board_address())
            # The bearing stage from raw data on the board, when it can (the owner's
            # decision of 2026-09-24); the other stages stay on this computer.
            lift_of = {l["units"].get("bearing"): lift_label(l) for l in LIFTS}
            feed = BoardFeed(router, {u.unit_no: {"scenario": u.scenario,
                                                  "label": lift_of.get(u.scenario, "")}
                                      for u in base.units() if u.branch == "bearing"})
            source = BoardSource(base, router, verdict="engine", feed=feed)
            state["svc"] = Service(source=source,
                                   endpoint=f"http://127.0.0.1:{port}/api/alerts/ingest",
                                   pace_s=float(os.environ.get("DS_PACE", "0.25")))
            state["fleet"] = Fleet(state["svc"])
        if run_fleet:
            # May start before uvicorn accepts connections; any alert that
            # meets a closed port waits in the outbox and is resent.
            state["svc"].start_fleet()
        yield
        feed = getattr(state["svc"].source, "feed", None) if state["svc"] else None
        if feed is not None:
            feed.end()

    middleware = [
        Middleware(SessionMiddleware, secret_key=session_secret or auth.session_secret(),
                   session_cookie="ds_session", max_age=auth.SESSION_MAX_AGE_S,
                   same_site="lax", https_only=False),
        Middleware(auth.AuthGate),
    ]
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
