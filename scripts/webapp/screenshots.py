"""
Browser verification and screenshots of the running web app, laptop and phone width.

Drives the Microsoft Edge that ships with Windows over the DevTools protocol
(the `websockets` package already in the venv) -- nothing to download. The app
must already be running (`python -m webapp`, port 8000) with its drive stream
finished, so every alert exists.

    .venv/Scripts/python.exe scripts/webapp/screenshots.py [--base http://127.0.0.1:8000]
    -> docs/screenshots/<name>_<laptop|phone>.png
       docs/screenshots/console.json   per shot: console errors, horizontal
                                       overflow, banned words in the rendered text;
                                       plus the navigation and sign-in checks

SIGN-IN. The browser is given a session cookie obtained by POSTing the demo
credentials to /signin from Python -- the same request the form makes. No
password is typed into a page. The unauthenticated shot of /signin, and the
check that a protected page redirects there, run without the cookie.

Why DevTools and not `msedge --screenshot`: Chromium on Windows will not make a
window narrower than about 500 px, so a 390 px "phone" shot was a 500 px layout
cropped to 390. Device-metrics emulation gives a true 390 px viewport.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import base64
import http.cookiejar
import importlib.util
import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

import sys
import urllib.error

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "screenshots")
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
WIDTHS = {"laptop": 1280, "phone": 390}


@functools.lru_cache(maxsize=1)
def _vocab():
    """The banned lists, from the test file, so the browser and pytest agree."""
    spec = importlib.util.spec_from_file_location(
        "tw", os.path.join(ROOT, "tests", "test_webapp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.BANNED_PHRASES, m.BANNED_PATTERNS, m.FORBIDDEN_CLAIMS


def banned_in(text: str):
    phrases, patterns, claims = _vocab()
    low = text.lower()
    hits = [p for p in phrases if p in low]
    hits += [f"{why}: {m.group(0)}" for pat, why in patterns for m in [re.search(pat, text)] if m]
    hits += [f"claim: {m.group(0)}" for pat in claims for m in [re.search(pat, text)] if m]
    return hits


def session_cookie(base: str) -> str:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), NoRedirect)
    data = urllib.parse.urlencode({"username": "technician", "password": "demo"}).encode()
    try:
        op.open(urllib.request.Request(base + "/signin", data=data, method="POST"), timeout=30)
    except urllib.error.HTTPError as e:
        if e.code != 303:
            raise
    value = next((c.value for c in jar if c.name == "ds_session"), None)
    if not value:
        raise RuntimeError("sign-in did not return a session cookie")
    return value


def get_json(base: str, cookie: str, path: str):
    req = urllib.request.Request(base + path, headers={"Cookie": f"ds_session={cookie}"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def shots(base: str, cookie: str):
    alerts = get_json(base, cookie, "/api/alerts")
    first = {}
    for a in sorted(alerts, key=lambda a: a["id"]):
        first.setdefault((a["lift_slug"], a["stage"]), a["id"])
    why = lambda lift, stage: f"/alerts/{first[(lift, stage)]}"
    return [
        ("00_signin", "/signin", False, 0),
        ("01_fleet", "/fleet", True, 0),
        ("02_lift_07", "/lifts/lift-07", True, 0),
        ("02_lift_01", "/lifts/lift-01", True, 0),
        ("03_alerts", "/alerts", True, 0),
        ("03_alerts_filter_alarm", "/alerts?status=ALARM", True, 0),
        ("04_why_bearing_alarm_outer_race", why("lift-07", "bearing"), True, 0),
        ("04_why_bearing_alarm_inner_race", why("lift-03", "bearing"), True, 0),
        ("04_why_supply_phase_lost_running", why("lift-07", "supply"), True, 0),
        ("04_why_supply_started_phase_missing", why("lift-03", "supply"), True, 0),
        ("04_why_inverter_open_circuit", why("lift-07", "inverter"), True, 0),
        ("04_why_inverter_short_circuit", why("lift-12", "inverter"), True, 0),
        ("04_why_inverter_overheating", why("lift-01", "inverter"), True, 0),
        ("04_why_winding_lift_07", why("lift-07", "winding"), True, 0),
        ("04_why_winding_lift_03", why("lift-03", "winding"), True, 0),
        # Live monitor runs: each waits past its first alert (bearing t=1 s,
        # inverter t=2 s, winding t=1 s at the data's own pace)
        ("05_monitor_bearing_alarm", "/monitor?lift=lift-07&stage=bearing&start=1", True, 6),
        ("05_monitor_inverter_advisory", "/monitor?lift=lift-07&stage=inverter&start=1", True, 6),
        ("05_monitor_winding_advisory", "/monitor?lift=lift-07&stage=winding&start=1", True, 6),
        ("06_sensors", "/sensors", True, 0),
        ("07_hardware", "/hardware", True, 0),
        ("08_about", "/about", True, 0),
        ("09_settings", "/settings", True, 0),
        ("10_engineering", "/engineering", True, 0),
    ]


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.events = ws, 0, []

    async def call(self, method, **params):
        self.n += 1
        my = self.n
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            self.events.append(m)

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                            awaitPromise=True)
        return r["result"].get("value")

    def errors(self):
        return [e["params"] for e in self.events
                if e.get("method") == "Runtime.exceptionThrown"
                or (e.get("method") == "Runtime.consoleAPICalled" and e["params"]["type"] == "error")
                or (e.get("method") == "Log.entryAdded" and e["params"]["entry"]["level"] == "error")]


READY = ("document.readyState === 'complete' && (location.pathname === '/signin'"
         " ? !!document.querySelector('form.auth-card')"
         " : (!!document.querySelector('footer.site-footer') && !document.querySelector('.skeleton')))")


async def setup(c: CDP, base: str, cookie, w: int, phone: bool):
    await c.call("Page.enable")
    await c.call("Runtime.enable")
    await c.call("Log.enable")
    await c.call("Network.enable")
    await c.call("Emulation.setFocusEmulationEnabled", enabled=True)
    await c.call("Page.bringToFront")
    await c.call("Emulation.setDeviceMetricsOverride", width=w, height=844 if phone else 900,
                 deviceScaleFactor=1, mobile=phone)
    if cookie:
        host = urllib.parse.urlparse(base).hostname
        await c.call("Network.setCookie", name="ds_session", value=cookie, domain=host,
                     path="/", httpOnly=True, sameSite="Lax")


async def wait_ready(c: CDP, what: str, timeout=45):
    t0 = time.time()
    while not await c.js(READY):
        if time.time() - t0 > timeout:
            text = await c.js("document.body ? document.body.innerText.slice(0, 300) : ''")
            raise RuntimeError(f"{what} did not render in {timeout} s: {text!r}")
        await asyncio.sleep(0.25)


async def capture(ws_url, base, cookie, name, path, dev, w, extra_wait):
    phone = dev == "phone"
    async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
        c = CDP(ws)
        await setup(c, base, cookie, w, phone)
        c.events.clear()
        await c.call("Page.navigate", url=base + path)
        await wait_ready(c, f"{path} ({dev})")
        await asyncio.sleep(1.5 + extra_wait)
        text = await c.js("document.body.innerText")
        h = await c.js("document.documentElement.scrollHeight")
        sw = await c.js("document.documentElement.scrollWidth")
        extra = {}
        if name.startswith("05_monitor"):
            extra["alerts_arrived"] = await c.js("document.querySelectorAll('#alerts .alert-card').length")
            extra["status"] = await c.js("(document.querySelector('#status .badge')||{}).textContent||''")
            extra["headline"] = await c.js("(document.querySelector('#alerts .alert-headline')||{}).textContent||''")
        await c.call("Emulation.setDeviceMetricsOverride", width=w, height=int(h),
                     deviceScaleFactor=1, mobile=phone)
        await asyncio.sleep(0.4)
        shot = await c.call("Page.captureScreenshot", format="png")
        png = os.path.join(OUT, f"{name}_{dev}.png")
        with open(png, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        errs = c.errors()
        banned = [] if name == "10_engineering" else banned_in(text)
        rec = {"errors": len(errs), "scroll_width": sw, "viewport": w,
               "banned_in_rendered_text": banned, **extra,
               "detail": [json.dumps(x)[:300] for x in errs]}
        print(f"{os.path.relpath(png, ROOT)}  {w}x{h}  sw {sw}  errors {len(errs)}"
              f"  banned {banned or '-'}  {extra or ''}")
        return rec


async def ready_or_reload(c: CDP, what: str, out: dict):
    """wait_ready, with one reload if the page stalls. In one long headless
    session a later page intermittently never runs its script (the reason every
    screenshot gets a fresh browser); the navigation check must stay one session
    to test history, so it gets one reload per page instead. A retry is recorded,
    and a page that stalls twice still fails."""
    try:
        await wait_ready(c, what, timeout=20)
    except RuntimeError:
        out.setdefault("reloads", []).append(what)
        await c.call("Page.reload")
        await wait_ready(c, what)


async def navigation_checks(ws_url, base, cookie, dev, w):
    """Back/forward across real URLs, the active-nav highlight, the phone menu,
    and the redirect to sign-in when there is no session."""
    phone = dev == "phone"
    out = {}
    async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
        c = CDP(ws)
        await setup(c, base, None, w, phone)
        await c.call("Page.navigate", url=base + "/alerts")
        await ready_or_reload(c, "signed-out /alerts", out)
        out["signed_out_redirect"] = await c.js("location.pathname + location.search")
        host = urllib.parse.urlparse(base).hostname
        await c.call("Network.setCookie", name="ds_session", value=cookie, domain=host,
                     path="/", httpOnly=True, sameSite="Lax")
        c.events.clear()
        visited = []
        for p in ("/fleet", "/alerts", "/monitor", "/hardware"):
            await c.call("Page.navigate", url=base + p)
            await ready_or_reload(c, p, out)
            visited.append({"path": await c.js("location.pathname"),
                            "active_nav": await c.js("(document.querySelector('nav.main a.active')||{}).textContent||''"),
                            "h1": await c.js("(document.querySelector('h1')||{}).textContent||''")})
        out["visited"] = visited
        hist = []
        for step in ("back", "back", "forward"):
            await c.js(f"history.{step}()")
            await asyncio.sleep(1.0)
            await ready_or_reload(c, f"after {step}", out)
            hist.append({"step": step, "path": await c.js("location.pathname"),
                         "h1": await c.js("(document.querySelector('h1')||{}).textContent||''")})
        out["history"] = hist
        out["history_ok"] = [h["path"] for h in hist] == ["/monitor", "/alerts", "/monitor"]
        # click a nav link rather than navigating by URL
        await c.js("document.querySelector('nav.main a[href=\"/about\"]').click()")
        await asyncio.sleep(1.2)
        await ready_or_reload(c, "click to About", out)
        out["click_nav"] = {"path": await c.js("location.pathname"),
                            "active_nav": await c.js("(document.querySelector('nav.main a.active')||{}).textContent||''")}
        if phone:
            out["menu_hidden_before"] = await c.js("getComputedStyle(document.querySelector('nav.main')).display")
            await c.js("document.querySelector('.menu-toggle').click()")
            await asyncio.sleep(0.4)
            out["menu_shown_after"] = await c.js("getComputedStyle(document.querySelector('nav.main')).display")
            shot = await c.call("Page.captureScreenshot", format="png")
            with open(os.path.join(OUT, "11_phone_menu_open_phone.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
        # sign out from Settings (the header has only the name and navigation)
        await c.call("Page.navigate", url=base + "/settings")
        await ready_or_reload(c, "settings", out)
        await c.js("document.querySelector('form[action=\"/signout\"] button').click()")
        await asyncio.sleep(1.2)
        await ready_or_reload(c, "after sign-out", out)
        out["after_signout"] = await c.js("location.pathname + location.search")
        await c.call("Page.navigate", url=base + "/fleet")
        await ready_or_reload(c, "signed-out /fleet", out)
        out["fleet_after_signout"] = await c.js("location.pathname")
        out["errors"] = len(c.errors())
        out["detail"] = [json.dumps(x)[:300] for x in c.errors()]
    print(f"navigation ({dev}): {json.dumps(out)[:600]}")
    return out


def _edge_page(edge: str, prof: str):
    """Start one headless Edge; return (process, page websocket URL)."""
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    proc = subprocess.Popen([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--user-data-dir={prof}", f"--remote-debugging-port={port}",
                             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while True:
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
            return proc, next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"]
        except Exception:
            if time.time() - t0 > 30:
                proc.kill()
                raise
            time.sleep(0.3)


def with_edge(edge, fn):
    prof = tempfile.mkdtemp(prefix="ds_edge_")
    proc, ws = _edge_page(edge, prof)
    try:
        return asyncio.run(fn(ws))
    finally:
        proc.kill()
        try:                             # a slow exit is not a failed shot
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default="", help="substring filter on shot names")
    ap.add_argument("--navigation-only", action="store_true",
                    help="re-run only the navigation checks, updating the record")
    args = ap.parse_args()
    edge = next(p for p in EDGE if os.path.exists(p))
    os.makedirs(OUT, exist_ok=True)
    cookie = session_cookie(args.base)
    record = {}
    rec_path = os.path.join(OUT, "console.json")
    if (args.only or args.navigation_only) and os.path.exists(rec_path):      # a partial re-run updates the record
        with open(rec_path) as fh:
            record = json.load(fh)
    # One fresh Edge per shot: in one long-lived headless session a later page
    # intermittently never ran its script. The navigation check below is the
    # deliberate exception -- it has to be one session to test history.
    for name, path, auth, wait in ([] if args.navigation_only else shots(args.base, cookie)):
        if args.only and args.only not in name:
            continue
        for dev, w in WIDTHS.items():
            try:
                record[f"{name}_{dev}"] = with_edge(edge, lambda ws: capture(
                    ws, args.base, cookie if auth else None, name, path, dev, w, wait))
            except Exception as e:                     # record it and carry on
                record[f"{name}_{dev}"] = {"failed": str(e)[:400]}
                print(f"FAILED {name}_{dev}: {e}")
    if not args.only:
        for dev, w in WIDTHS.items():
            try:
                record[f"_navigation_{dev}"] = with_edge(edge, lambda ws: navigation_checks(
                    ws, args.base, cookie, dev, w))
            except Exception as e:
                record[f"_navigation_{dev}"] = {"failed": str(e)[:400]}
                print(f"FAILED navigation {dev}: {e}")
    with open(rec_path, "w") as fh:
        json.dump(record, fh, indent=2)
    bad = {k: v for k, v in record.items()
           if v.get("failed") or v.get("errors") or v.get("banned_in_rendered_text")
           or (v.get("scroll_width") or 0) > (v.get("viewport") or 10 ** 9)}
    print("failures / console errors / overflow / banned words:", bad or "none")


if __name__ == "__main__":
    main()
