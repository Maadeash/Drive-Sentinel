"""
The technician web app: the alert pipeline, the data-source seam, sign-in, and
what the pages may say.

Every property the brief names has a test here:

  * an alert is built correctly from fusion state;
  * an ADVISORY stage never produces an ALARM (two guards, both tested);
  * alerts are buffered on disk while the endpoint is down and resent after;
  * the packet is under its stated bound;
  * the data source is swappable (a fake source drives the whole service);
  * sign-in is required for every page except /signin, and sign-out works;
  * no user-facing page or API response carries the banned vocabulary
    (engineering view exempt), none claims a real lift is connected, and a
    page says the accelerator runs on a board only when a genuine board run
    (board/board_run.json) is committed;
  * winding alerts never name a fault type or a phase;
  * the fleet shows every unit exactly once;
  * every number the app shows matches the results JSON it was read from.

The server tests run uvicorn on a real port (webapp/testing.py): starlette's
TestClient needs httpx, which is not installed, and the HTTP POST is the point.
"""

from __future__ import annotations

import json
import os
import re
import time
from html.parser import HTMLParser
from typing import Dict, Iterator, List

import numpy as np
import pytest

from drivesentinel import alerts as A
from drivesentinel import config as C
from drivesentinel import fusion as FU
from drivesentinel.sources import DataSource, Observation, RecordedScenarioSource, UnitInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "webapp", "static")
PAGES = os.path.join(ROOT, "webapp", "pages")

# Every user-facing page. The engineering view is the one exemption.
USER_PAGE_FILES = ["signin.html", "fleet.html", "lift.html", "alerts.html", "alert.html",
                   "monitor.html", "hardware.html", "about.html",
                   "settings.html", "board.html"]
USER_JS = ["common.js", "charts.js", "signals.js"]

# The brief's list, as literal phrases (case-insensitive) ...
BANNED_PHRASES = ["artifacts", ".npz", "scenario", "replay", "recording", "simulated",
                  "demo mode", "cannot", "not estimated", "insufficient", "only tested",
                  "lab-tested", "missed detection",
                  # ... and the previous version's own caveat wording
                  "prototype", "not connected", "suspected location", "not resolved",
                  "not identified", "severity not", "documented miss"]
# ... dataset names, dataset file names and internal IDs, as patterns
BANNED_PATTERNS = [
    (r"(?i)paderborn|kaist|bacha|thomas et al|mendeley", "dataset name"),
    (r"\bKAT\b", "dataset name"),
    (r"\bFILE ?\d", "dataset file name"),
    (r"\bK[AI]?\d{2,3}\b", "bearing specimen ID"),
    (r"(?<![\w-])F[0-8]\b", "inverter run ID"),
    (r"DEMO-\d", "internal unit ID"),
    (r"\bS[1-5]\b", "internal stage code"),
    (r"(?i)inverter_telemetry|top_class|unit_no|t_data|recorded_condition|heldout|held_out",
     "internal field name"),
    (r"(?i)inter[-_ ]turn|inter[-_ ]coil", "winding fault type"),
    (r"(?<![A-Za-z])[A-Za-z]:[\\/]|\b(?:artifacts|webapp|drivesentinel|scripts|docs|data_ext)/",
     "file path"),
    (r"\.(?:py|npz|mem|rpt|tcl|xdc|csv|mat|tdms)\b|\.json\b(?!\s*\()", "file name"),
]
# Claims no page may make.
FORBIDDEN_CLAIMS = [
    r"(?i)\bconnected to (a|the|your|real)\b", r"(?i)\breal (elevator|lift)s?\b",
    r"(?i)\bin the field\b",
]
# Claims no page may make UNLESS a genuine board run is committed
# (board/board_run.json, ran_on_hardware, not the mock) -- since 2026-09-24 one is.
BOARD_CLAIMS = [
    r"(?i)\b(ran|runs|running|run) on (a|the)? ?(board|pynq)",
    r"(?i)\bdeployed (on|to) (a|the) (board|pynq)", r"(?i)\bon[- ]board run",
    r"(?i)\btested on (a|the) board",
]


def _board_ran() -> bool:
    p = os.path.join(ROOT, "board", "board_run.json")
    if not os.path.exists(p):
        return False
    r = json.load(open(p))
    return r.get("ran_on_hardware") is True and not str(r.get("pynq_version", "")).startswith("MOCK")


INTERNAL_TIER_WORDS = ["Fault-capable", "INDICATIVE"]


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _strip_asset_refs(html: str) -> str:
    """Remove <link href> / <script src> to this app's own assets: page code, not text."""
    return re.sub(r'(href|src)="/static/[^"]*"', "", html)


# ===========================================================================
# fixtures: one real service, streamed to the end without pacing
# ===========================================================================

@pytest.fixture(scope="module")
def source():
    return RecordedScenarioSource()


@pytest.fixture(scope="module")
def service(source, tmp_path_factory):
    from webapp.service import Service
    s = Service(source=source, endpoint=None, pace_s=0.0)      # direct ingest
    s.start_fleet()
    t0 = time.time()
    while not s.fleet_done:
        assert time.time() - t0 < 180, "fleet stream did not finish"
        time.sleep(0.05)
    return s


@pytest.fixture(scope="module")
def fleet(service):
    from webapp.fleet import Fleet
    return Fleet(service)


@pytest.fixture(scope="module")
def server(service):
    """Signed in."""
    from webapp.server import create_app
    from webapp.testing import LiveServer
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        status, loc = srv.login()
        assert status == 303 and loc == "/fleet"
        yield srv


@pytest.fixture(scope="module")
def anon(service):
    """Never signed in."""
    from webapp.server import create_app
    from webapp.testing import LiveServer
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        yield srv


def _by_scenario(service, scenario) -> List[Dict]:
    n = next(n for n, u in service.units.items() if u.scenario == scenario)
    return service.feed(unit=n)


def _view(branch, status, top_class="outer_race", authority=False, conf=0.9, n=10):
    return {"branch": branch, "status": status, "top_class": top_class,
            "confidence": conf, "n_obs": n, "fault_authority": authority,
            "p_fault": conf}


# ===========================================================================
# 1. an alert is built correctly from fusion state
# ===========================================================================

def test_alert_from_fusion_state_round_trips():
    a = A.build_alert(2, _view("bearing", "Fault", "outer_race", authority=True, conf=0.98),
                      seq=1, t_data_s=1.25, now=1_700_000_000)
    assert (a.stage, a.status, a.fault, a.phase) == ("S5", "ALARM", "outer_race", -1)
    assert a.t_data_ms == 1250 and a.n_obs == 10
    b = A.Alert.unpack(a.pack())
    assert b == A.Alert(**{**a.__dict__, "confidence": 0.98})
    r = b.render()
    assert r["unit"] == "DEMO-02"
    assert r["alert_type"] == "ALARM — schedule inspection"
    assert r["what"] == A.WHAT["outer_race"] and r["action"] == A.ACTION["outer_race"]
    assert r["how_bad"] == "Confidence 98 %"


def test_supply_alert_carries_measured_phase_and_nothing_else_does():
    s = A.build_alert(5, _view("supply", "Warning", "phase_loss_running"), 1, 0.0,
                      context={"phase": 1, "rotating": 1})
    assert s.phase == 1 and "Phase L2" in s.render()["where"]
    w = A.build_alert(1, _view("winding", "Warning", "inter_turn"), 1, 0.0,
                      context={"phase": 1})            # a stray phase is dropped
    assert w.phase == -1


def test_emitter_fires_only_on_status_change():
    em = A.AlertEmitter(3)
    fired = [em.observe(_view("inverter_telemetry", st, "open_circuit"), float(i))
             for i, st in enumerate(["Normal", "Normal", "Warning", "Warning",
                                     "Warning", "Normal", "Normal"])]
    got = [(a.status, a.seq) for a in fired if a]
    assert got == [("ADVISORY", 1), ("NORMAL", 2)]


def test_replayed_alerts_match_a_fresh_fusion_replay(service, source):
    """The feed is exactly what BranchState + AlertEmitter produce, nothing added."""
    for u in source.units():
        st, em, expect = None, A.AlertEmitter(u.unit_no), []
        for ob in source.observations(u.unit_no):
            if ob.probs is None:
                continue
            st = st or FU.BranchState(metric=service.metrics[u.branch], labels=ob.labels)
            a = em.observe(st.update(ob.probs), ob.t_s, ob.context)
            if a:
                expect.append((a.seq, a.status, a.fault, a.phase))
        got = sorted((r["seq"], r["status"], r["fault"],
                      A.Alert.unpack(bytes.fromhex(r["packet_hex"])).phase)
                     for r in service.feed(unit=u.unit_no))
        assert got == expect, u.unit_id


# ===========================================================================
# 2. an ADVISORY stage never produces an ALARM
# ===========================================================================

def test_build_alert_refuses_alarm_without_authority():
    for branch in ("supply", "inverter_telemetry", "winding"):
        with pytest.raises(AssertionError):
            A.build_alert(1, _view(branch, "Fault", authority=False), 1, 0.0)


def test_fusion_caps_every_advisory_stage_at_warning(service):
    """First guard: BranchState itself, fed pure fault evidence for 50 updates."""
    for branch, m in service.metrics.items():
        if m.fault_authority:
            continue
        labels = ["healthy", "fault"]
        st = FU.BranchState(metric=m, labels=labels)
        statuses = {st.update(np.array([0.0, 1.0]))["status"] for _ in range(50)}
        assert "Fault" not in statuses, branch


def test_only_the_bearing_stage_is_alarm_ready(service):
    rows = {r["stage"]: r for r in service.stage_rows()}
    assert rows["S5"]["tier_label"] == "ALARM-READY"
    for st in ("S1", "S3", "S4"):
        assert rows[st]["tier_label"] == "ADVISORY"
        assert not rows[st]["alarm_capable"]


def test_no_alarm_from_an_advisory_stage_in_the_whole_replay(service):
    alarm_ready = {r["stage"] for r in service.stage_rows() if r["alarm_capable"]}
    for rec in service.feed():
        if rec["status"] == "ALARM":
            assert rec["stage"] in alarm_ready, rec


# ===========================================================================
# 3. buffering and resend when the endpoint is down
# ===========================================================================

def test_outbox_buffers_while_down_and_resends_once_up(tmp_path, source):
    from webapp.server import create_app
    from webapp.service import Service
    from webapp.testing import LiveServer, free_port

    port = free_port()                                  # nothing listening yet
    ob = A.Outbox(str(tmp_path / "spool.jsonl"), f"http://127.0.0.1:{port}/api/alerts/ingest",
                  timeout_s=0.5)
    alerts = [A.build_alert(2, _view("bearing", "Fault", authority=True), seq=i, t_data_s=i)
              for i in (1, 2, 3)]
    for a in alerts:
        assert ob.send(a) is False                      # endpoint down: kept
    assert ob.pending() == 3 and ob.failed_attempts >= 3

    # the spool survives a restart of the sender
    ob2 = A.Outbox(ob.path, ob.endpoint, timeout_s=2.0)
    assert ob2.pending() == 3

    receiver = Service(source=source, endpoint=None, pace_s=0.0)
    with LiveServer(create_app(service=receiver, run_fleet=False), port=port):
        assert ob2.flush() is True
        assert ob2.pending() == 0 and ob2.delivered == 3
        assert sorted(r["seq"] for r in receiver.feed()) == [1, 2, 3]
        # a resend after a lost acknowledgement is not counted twice
        ob2.send(alerts[0])
        assert len(receiver.feed()) == 3


def test_ingest_rejects_a_packet_of_the_wrong_size(server):
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as e:
        server.post("/api/alerts/ingest", b"\x01" * (A.WIRE_BYTES - 1))
    assert e.value.code == 400


# ===========================================================================
# 4. the packet is small, and bounded
# ===========================================================================

PACKET_BOUND_BYTES = 32


def test_packet_size_under_stated_bound():
    a = A.build_alert(65535, _view("bearing", "Fault", authority=True), seq=2**32 - 1,
                      t_data_s=4e6, now=2**32 - 1)
    assert len(a.pack()) == A.packet_bytes() == 21 <= PACKET_BOUND_BYTES
    # what it replaces: one bearing model input window, float32
    assert A.raw_window_bytes() == C.N_INPUT_CHANNELS * C.N_ORDER_BINS * 4 == 10240


# ===========================================================================
# 5. the data source is swappable
# ===========================================================================

class FakeSource:
    """A stand-in for a live sensor feed: one inverter unit, one fault episode."""

    def units(self) -> List[UnitInfo]:
        return [UnitInfo(unit_no=1, unit_id="DEMO-01", scenario="fake_inverter",
                         branch="inverter_telemetry", stage="S3", title="Fake unit",
                         dataset="none", n_steps=30, has_predictions=True, meta={})]

    def observations(self, unit_no: int) -> Iterator[Observation]:
        labels = ["normal", "open_circuit", "short_circuit", "over_temp"]
        for i in range(30):
            p = np.array([0.0, 1.0, 0.0, 0.0]) if 5 <= i < 20 else np.array([1.0, 0, 0, 0])
            yield Observation(step=i, t_s=float(i), labels=labels, probs=p,
                              frame={}, context={})


def test_fake_source_satisfies_the_protocol_and_drives_the_service():
    from webapp.service import Service
    src: DataSource = FakeSource()
    s = Service(source=src, endpoint=None, pace_s=0.0)
    s.start_fleet()
    t0 = time.time()
    while not s.fleet_done:
        assert time.time() - t0 < 10
        time.sleep(0.01)
    got = [(r["status"], r["fault"]) for r in sorted(s.feed(), key=lambda r: r["seq"])]
    assert got == [("ADVISORY", "open_circuit"), ("NORMAL", "")]


def test_live_source_refuses_to_pretend():
    from drivesentinel.sources import LiveSensorSource
    with pytest.raises(NotImplementedError):
        LiveSensorSource().units()


# ===========================================================================
# 6. what each stage may and may not say
# ===========================================================================

def test_winding_headline_never_names_a_fault_type(service, fleet):
    """User decision 2: 'Winding anomaly detected', no type -- the type call is
    crossed on both demo units, so naming it would be wrong half the time."""
    recs = [r for r in service.feed() if r["branch"] == "winding" and r["status"] != "NORMAL"]
    assert recs, "no winding alert in the stream"
    for r in recs:
        assert r["what"] == A.WINDING_HEADLINE == "Winding anomaly detected"
        v = fleet.alert_view(r)
        assert v["headline"] == "Winding anomaly detected"
        text = json.dumps(v).lower()
        for t in ("inter-turn", "inter-coil", "inter_turn", "inter_coil", "inter turn", "inter coil"):
            assert t not in text, (t, r["id"])
    for f in ("inter_turn", "inter_coil", "undetermined"):
        assert A.what_text("winding", f) == "Winding anomaly detected"


def test_winding_fault_code_still_travels_in_the_packet(service):
    """Only the WORDS drop the type. The packet and the engineering side keep it."""
    codes = {A.Alert.unpack(bytes.fromhex(r["packet_hex"])).fault
             for r in service.feed() if r["branch"] == "winding" and r["status"] != "NORMAL"}
    assert codes and codes <= {"inter_turn", "inter_coil"}


def test_winding_alerts_never_name_a_phase_or_a_severity(service, fleet):
    recs = [r for r in service.feed() if r["branch"] == "winding"]
    for r in recs:
        v = fleet.alert_view(r)
        assert not re.search(r"\bL[123]\b", json.dumps(v)), r["id"]
        assert v["how_bad"] is None, "winding severity must be omitted (decision 1)"
        assert A.Alert.unpack(bytes.fromhex(r["packet_hex"])).phase == -1
    for f in ("inter_turn", "inter_coil"):
        assert not re.search(r"\bL[123]\b", A.where_text("winding", f, phase=1))
        assert A.how_bad_text("winding", f, 0.9) == ""


def test_inverter_location_wording(service, fleet):
    """User decision 3: 'Location: inverter power stage'. No severity field."""
    recs = [r for r in service.feed() if r["branch"] == "inverter_telemetry"
            and r["status"] != "NORMAL"]
    assert {r["fault"] for r in recs} >= {"open_circuit", "short_circuit", "over_temp"}
    for r in recs:
        v = fleet.alert_view(r)
        assert v["where"] == "Location: inverter power stage", v
        assert v["how_bad"] is None
    for f in ("open_circuit", "short_circuit", "over_temp", "undetermined"):
        assert A.where_text("inverter_telemetry", f) == "Location: inverter power stage"


def test_supply_alerts_name_the_collapsed_phase(service, source):
    for scn in ("supply_phase_loss_FILE2", "supply_phase_loss_FILE5"):
        recs = [r for r in _by_scenario(service, scn) if r["status"] != "NORMAL"]
        assert recs, scn
        for r in recs:
            assert re.search(r"Phase L[123]", r["where"]), r
            assert r["status"] == "ADVISORY"


def test_supply_phase_named_is_the_phase_lost_in_the_raw_current(service):
    """Independent of the rule's path: scripts/webapp/check_supply_phase.py."""
    chk = _read("artifacts", "webapp", "supply_phase_check.json")["files"]
    for scn, f in (("supply_phase_loss_FILE2", "FILE 2"), ("supply_phase_loss_FILE5", "FILE 5")):
        for r in _by_scenario(service, scn):
            if r["status"] != "NORMAL":
                assert r["where"].startswith(f"Phase {chk[f]['lost_phase']} "), (scn, r["where"])


def test_units_give_the_documented_outcomes(service):
    first = lambda scn: next(iter(sorted(_by_scenario(service, scn),
                                         key=lambda r: r["seq"])), None)
    ka04, ki18 = first("bearing_KA04"), first("bearing_KI18")
    assert (ka04["status"], ka04["fault"]) == ("ALARM", "outer_race")
    assert (ki18["status"], ki18["fault"]) == ("ALARM", "inner_race")
    assert _by_scenario(service, "bearing_KI05") == []          # the documented miss
    assert _by_scenario(service, "bearing_K001") == []          # healthy specimen


def test_empty_fields_are_omitted_not_explained(fleet):
    for a in fleet.alerts():
        for k in ("where", "how_bad"):
            assert a[k] is None or (a[k].strip() and a[k] != "—"), (k, a)


# ===========================================================================
# 7. sign-in
# ===========================================================================

def _all_page_paths(fleet):
    from webapp.server import PAGES as SP
    ref = fleet.alerts()[0]["id"]
    return list(SP) + ["/lifts/lift-07", f"/alerts/{ref}", "/"]


def _all_api_paths(fleet):
    ref = fleet.alerts()[0]["id"]
    return ["/api/session", "/api/meta", "/api/fleet", "/api/lifts/lift-07", "/api/alerts",
            f"/api/alerts/{ref}", "/api/monitor", "/api/monitor/lift-07/bearing",
            "/api/hardware", "/api/about", "/api/settings",
            "/api/engine", "/api/board/live", "/api/engineering"]


def test_every_page_requires_sign_in(anon, fleet):
    for p in _all_page_paths(fleet):
        status, h, _ = anon.raw(p)
        assert status == 303, p
        assert h.get("Location", "").startswith("/signin"), (p, h.get("Location"))
    status, h, body = anon.raw("/signin")
    assert status == 200 and b'action="/signin"' in body


def test_redirect_to_sign_in_remembers_where_you_were_going(anon):
    status, h, _ = anon.raw("/alerts?status=ALARM")
    from urllib.parse import unquote
    assert status == 303
    assert unquote(h["Location"]) == "/signin?next=/alerts?status=ALARM"


def test_every_api_requires_sign_in(anon, fleet):
    for p in _all_api_paths(fleet):
        status, _, body = anon.raw(p)
        assert status == 401, p
    status, _, _ = anon.raw("/api/stream/restart", method="POST")
    assert status == 401


def test_device_channel_needs_no_session_but_only_takes_packets(anon):
    """A drive's outbox has no browser session. It gets exactly one door."""
    status, _, _ = anon.raw("/api/alerts/ingest", method="POST", data=b"\x01" * 5,
                            ctype="application/octet-stream")
    assert status == 400                        # reached the handler, refused the size


def test_page_html_is_not_reachable_around_the_gate(anon):
    for f in USER_PAGE_FILES + ["engineering.html"]:
        assert anon.raw(f"/static/{f}")[0] == 404, f
        assert not os.path.exists(os.path.join(STATIC, f)), f
    assert anon.raw("/static/css/app.css")[0] == 200            # assets carry no data


def test_wrong_password_is_refused(service):
    from webapp.server import create_app
    from webapp.testing import LiveServer
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        status, loc = srv.login("technician", "wrong")
        assert status == 303 and "error=1" in loc
        assert srv.raw("/api/fleet")[0] == 401
        status, loc = srv.login("nobody", "demo")
        assert "error=1" in loc and srv.raw("/api/fleet")[0] == 401


def test_sign_in_sign_out_round_trip(service, fleet):
    from webapp.server import create_app
    from webapp.testing import LiveServer
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        assert srv.login(next_path="/hardware") == (303, "/hardware")
        for p in _all_page_paths(fleet):
            assert srv.raw(p)[0] in (200, 303), p
        for p in _all_api_paths(fleet):
            assert srv.raw(p)[0] == 200, p
        assert srv.get("/api/session")["name"] == "Technician"
        # signed in: /signin sends you on rather than showing the form again
        assert srv.raw("/signin")[0] == 303
        assert srv.logout() == (303, "/signin?signed_out=1")
        assert srv.raw("/fleet")[0] == 303
        assert srv.raw("/api/fleet")[0] == 401


def test_next_cannot_leave_the_site(service):
    from webapp import auth
    for bad in ("//evil.example/x", "https://evil.example", "\\\\evil", "/signin", "", None):
        assert auth.safe_next(bad) == "/fleet", bad
    assert auth.safe_next("/alerts?status=ALARM") == "/alerts?status=ALARM"
    from webapp.server import create_app
    from webapp.testing import LiveServer
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        assert srv.login(next_path="//evil.example/x") == (303, "/fleet")


def test_session_cookie_is_httponly_and_samesite(service):
    from webapp.server import create_app
    from webapp.testing import LiveServer
    import urllib.parse
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        data = urllib.parse.urlencode({"username": "technician", "password": "demo"}).encode()
        _, h, _ = srv.raw("/signin", method="POST", data=data,
                          ctype="application/x-www-form-urlencoded")
        cookie = h.get("Set-Cookie", "").lower()
        assert cookie.startswith("ds_session=") and "httponly" in cookie and "samesite=lax" in cookie


def test_sign_in_form_posts_the_fields_the_server_reads():
    """The browser form and the handler must agree on names, or sign-in breaks
    in the browser while every HTTP test passes."""
    class F(HTMLParser):
        def __init__(self):
            super().__init__(); self.form = None; self.inputs = []
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "form":
                self.form = a
            if tag == "input":
                self.inputs.append(a)
    p = F(); p.feed(_src("webapp", "pages", "signin.html"))
    assert p.form["method"] == "post" and p.form["action"] == "/signin"
    names = {i.get("name"): i.get("type") for i in p.inputs}
    assert names == {"next": "hidden", "username": "text", "password": "password"}


def test_demo_credentials_are_documented(monkeypatch):
    from webapp import auth
    monkeypatch.delenv("DRIVESENTINEL_USER", raising=False)
    monkeypatch.delenv("DRIVESENTINEL_PASSWORD", raising=False)
    readme = _src("webapp", "README.md")
    for user, u in auth.users().items():
        assert f"`{user}`" in readme and f"`{u['password']}`" in readme
    assert "DRIVESENTINEL_USER" in readme and "DRIVESENTINEL_PASSWORD" in readme


def test_credentials_come_from_the_environment(monkeypatch):
    from webapp import auth
    monkeypatch.delenv("DRIVESENTINEL_USER", raising=False)
    monkeypatch.delenv("DRIVESENTINEL_PASSWORD", raising=False)
    assert auth.check("technician", "demo") == "technician"
    monkeypatch.setenv("DRIVESENTINEL_USER", "Operator")
    monkeypatch.setenv("DRIVESENTINEL_PASSWORD", "s3cret")
    assert auth.check("operator", "s3cret") == "operator"
    assert auth.check("technician", "demo") is None           # the default is replaced
    assert auth.check("operator", "demo") is None


def test_env_credentials_work_end_to_end(monkeypatch, service):
    from webapp.server import create_app
    from webapp.testing import LiveServer
    monkeypatch.setenv("DRIVESENTINEL_USER", "operator")
    monkeypatch.setenv("DRIVESENTINEL_PASSWORD", "s3cret")
    with LiveServer(create_app(service=service, run_fleet=False)) as srv:
        assert srv.login("technician", "demo")[1].startswith("/signin?error=1")
        status, loc = srv.login("operator", "s3cret")
        assert status == 303 and loc == "/fleet"


def test_signin_page_is_bare():
    html = _src("webapp", "pages", "signin.html")
    brand = html.split('class="auth-brand"')[1].split("</section>")[0]
    assert "DriveSentinel" in brand
    for gone in ("<h1", "<ul", "<li", "class=\"wave\"", "claims"):
        assert gone not in brand, gone
    form = html.split('class="auth-form"')[1]
    assert "<h2>Sign in</h2>" in form
    assert "Use your technician account" not in html and 'class="lead"' not in html
    assert "Prepared for KONE Elevate'26 · Problem Statement 06" in form


def test_header_is_name_and_navigation_only():
    js = _src("webapp", "static", "js", "common.js")
    header = js.split('header.innerHTML = `')[1].split("`;")[0]
    assert "b.name" in header and "nav" in header
    for gone in ("tagline", "user-chip", "avatar", "signout", "ICON.mark"):
        assert gone not in header, gone
    assert "tagline" not in js
    # sign-out stays reachable, on Settings
    assert 'action="/signout"' in _src("webapp", "pages", "settings.html")


def test_no_intro_paragraphs_under_page_titles():
    for f in os.listdir(os.path.join(ROOT, "webapp", "pages")):
        html = _src("webapp", "pages", f)
        assert '<p class="lead">' not in html, f


def test_protected_responses_are_not_cached(server):
    for p in ("/fleet", "/api/fleet"):
        _, h, _ = server.raw(p)
        assert h.get("Cache-Control") == "no-store", p
        assert h.get("X-Frame-Options") == "DENY"
    # Assets may be kept, but are revalidated on every use: a changed script (a new
    # nav entry) must reach a browser that has the old one, without a hard refresh.
    _, h, _ = server.raw("/static/js/common.js")
    assert h.get("Cache-Control") == "no-cache"
    # ... and every page links its assets by version, so an old copy is never reused
    _, _, body = server.raw("/board")
    refs = re.findall(r'"/static/[^"]+\.(?:js|css)(\?v=\d+)?"', body.decode("utf-8"))
    assert refs and all(refs), refs


# ===========================================================================
# 8. vocabulary and claims on every user-facing page
# ===========================================================================

def _user_api_text(server, fleet) -> Dict[str, str]:
    out = {p: json.dumps(server.get(p), ensure_ascii=False)
           for p in _all_api_paths(fleet) if p != "/api/engineering"}
    for a in fleet.alerts():
        out[f"/api/alerts/{a['id']}"] = json.dumps(server.get(f"/api/alerts/{a['id']}"),
                                                   ensure_ascii=False)
    for o in server.get("/api/monitor"):
        out[f"/api/lifts/{o['slug']}"] = json.dumps(server.get(f"/api/lifts/{o['slug']}"),
                                                    ensure_ascii=False)
        for s in o["stages"]:
            p = f"/api/monitor/{o['slug']}/{s['slug']}"
            out[p] = json.dumps(server.get(p), ensure_ascii=False)
    return out


def _user_texts(server, fleet) -> Dict[str, str]:
    texts = {f: _strip_asset_refs(_src("webapp", "pages", f)) for f in USER_PAGE_FILES}
    texts.update({js: _src("webapp", "static", "js", js) for js in USER_JS})
    texts.update(_user_api_text(server, fleet))
    return texts


def test_no_banned_vocabulary_on_any_user_facing_page_or_api(server, fleet):
    texts = _user_texts(server, fleet)
    hits = []
    for where, text in texts.items():
        low = text.lower()
        hits += [(where, ph) for ph in BANNED_PHRASES if ph in low]
        hits += [(where, why, m.group(0)) for pat, why in BANNED_PATTERNS
                 for m in [re.search(pat, text)] if m]
    assert not hits, hits


def test_no_page_claims_a_real_lift_and_board_claims_follow_the_evidence(server, fleet):
    """No page may claim a real lift, ever. A page may say the accelerator runs on a
    board only when a genuine board run is committed -- and then the Hardware page
    must say so, from that record."""
    texts = _user_texts(server, fleet)
    ran = _board_ran()
    for where, text in texts.items():
        for pat in FORBIDDEN_CLAIMS + ([] if ran else BOARD_CLAIMS):
            assert not re.search(pat, text), (where, pat)
    hw = server.get("/api/hardware")
    assert hw["ran_on_hardware"] is ran
    page = _src("webapp", "pages", "hardware.html")
    assert "Verified bit-exact in simulation" in page and "Synthesised for Zynq-7020" in page
    assert "Running on ${DS.esc(h.board.board)}" in page          # when the record exists
    assert "Board deployment: next step" in page                  # when it does not
    if ran:
        run = _read("board", "board_run.json")
        b = hw["board"]
        assert b["build_id"] == run["build_id"] and b["windows"] == run["windows"]
        assert b["logits_exact"] == run["logit_registers_exact"]
        assert b["median_ms"] == run["timing"]["dma_plus_inference_wall"]["median_ms"]
    else:
        assert hw["board"] is None


def test_user_pages_do_not_leak_internal_tier_words(server, fleet):
    for where, text in _user_texts(server, fleet).items():
        for w in INTERNAL_TIER_WORDS:
            assert w not in text, (w, where)


def test_engineering_view_is_exempt_and_still_complete(server):
    """Unchanged content, for the user's own reference -- and reachable only via
    the footer link, never the top navigation."""
    eng = server.get("/api/engineering")
    for k in ("stage_rows", "metric_cards", "authority", "floor", "fusion", "trip",
              "resolution", "ki05", "lobo_pooled"):
        assert k in eng, k
    assert any(r["source"] == "EXTRAPOLATION" for r in eng["resolution"])
    js = _src("webapp", "static", "js", "common.js")
    nav = js.split("const NAV = [")[1].split("];")[0]
    assert "/engineering" not in nav
    assert '<a href="/engineering">Engineering view</a>' in js


def test_chrome_has_footer_and_no_banner(server):
    meta = server.get("/api/meta")
    assert meta["brand"]["footer"] == "Prepared for KONE Elevate'26 · Problem Statement 06"
    assert "banner" not in meta["brand"]
    js = _src("webapp", "static", "js", "common.js")
    assert "banner" not in js.lower()
    for f in USER_PAGE_FILES:
        html = _src("webapp", "pages", f)
        assert "DS.chrome(" in html or f == "signin.html", f
        assert not re.search(r"(src|href)=\"(https?:)?//", html), f
    assert "Prepared for KONE Elevate'26 · Problem Statement 06" in _src("webapp", "pages",
                                                                          "signin.html")
    for js in USER_JS:
        src = _src("webapp", "static", "js", js)
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg\")", src), js


def test_every_page_has_its_own_url_and_old_bookmarks_redirect(server):
    from webapp.server import PAGES as SP, LEGACY
    js = _src("webapp", "static", "js", "common.js")
    for path in ("/fleet", "/alerts", "/monitor", "/board", "/hardware", "/about",
                 "/settings"):
        assert path in SP and f'["{path}",' in js, path
    for old, new in LEGACY.items():
        status, h, _ = server.raw(old)
        assert status == 301 and h["Location"] == new, old


def test_about_carries_the_roadmap_verbatim_and_no_caveats(server):
    d = server.get("/api/about")
    assert d["roadmap"] == ("Advisory stages are promoted to alarm-ready through calibration "
                            "on fleet data, with no change to the models.")
    assert [s["name"] for s in d["stages"]] == ["Power supply", "Inverter", "Motor winding",
                                                "Bearing"]
    assert [l["status"] for l in d["alert_levels"]] == ["ALARM", "ADVISORY", "NORMAL"]
    for k in ("ki05", "datasets", "validation_story"):
        assert k not in d


def test_settings_shows_the_board_connected_only_while_it_is_the_engine():
    from drivesentinel.engines import ENGINE_FPGA, ENGINE_SOFTWARE
    from webapp import content

    class FakeService:
        fleet_done, fleet_started_unix = True, None

        def __init__(self, active):
            self.source = type("S", (), {"router": type("R", (), {"active": active})()})()

        def outbox_state(self):
            return {"mode": "direct"}

    for active, state in ((ENGINE_FPGA, "Connected · bearing engine"),
                          (ENGINE_SOFTWARE, "Ready for connection")):
        row = content.settings(FakeService(active))["sources"][1]
        assert (row["state"], row["active"], row["live"]) == (state, False, active == ENGINE_FPGA)


def test_settings_lists_both_data_sources(server):
    s = server.get("/api/settings")["sources"]
    assert [(x["name"], x["state"], x["active"]) for x in s] == [
        ("Drive stream", "Active", True),
        ("Drive interface — PYNQ-Z2", "Ready for connection", False)]


# ===========================================================================
# 9. the fleet shows every unit exactly once
# ===========================================================================

def test_every_unit_is_in_exactly_one_lift_slot_of_its_own_stage(service, fleet):
    from webapp.fleet import STAGE_SLUG
    assert fleet.unplaced_units() == []
    assert sorted(fleet.place_of) == sorted(service.units)
    for (lift, slug), n in fleet.unit_of.items():
        assert STAGE_SLUG[service.units[n].stage] == slug


def test_fleet_alerts_are_the_service_alerts_one_to_one(service, fleet):
    assert len(fleet.alerts()) == len(service.feed())
    assert len({a["id"] for a in fleet.alerts()}) == len(service.feed())


def test_fleet_summary_counts_its_own_stages(server):
    d = server.get("/api/fleet")
    slots = [s for l in d["lifts"] for s in l["stages"]]
    S = d["summary"]
    assert S["lifts"] == len(d["lifts"])
    assert S["stages_monitored"] == sum(s["monitored"] for s in slots)
    for k, st in (("alarm", "ALARM"), ("advisory", "ADVISORY"), ("normal", "NORMAL")):
        assert S[k] == sum(s["status"] == st for s in slots)
    for s in slots:
        if not s["monitored"]:
            assert s["status"] == "NO DATA"


def test_lift_status_is_the_worst_of_its_stages_and_never_invents_an_alarm(server, service):
    from webapp.fleet import STATUS_RANK
    alarm_ready = {r["name"] for r in service.stage_rows() if r["alarm_capable"]}
    for l in server.get("/api/fleet")["lifts"]:
        assert l["status"] == max((s["status"] for s in l["stages"]), key=STATUS_RANK.get)
        for s in l["stages"]:
            if s["status"] == "ALARM":
                assert s["name"] in alarm_ready, (l["label"], s)


def test_monitor_shows_what_the_stream_computed_and_the_feed_holds(server, service, fleet):
    """Live monitor shows the stream's own results (service.live_records), not a
    precomputed copy. Those equal the precomputed timeline step for step -- status
    and fault evidence -- and fire exactly the alerts the feed holds."""
    for o in server.get("/api/monitor"):
        for s in o["stages"]:
            m = server.get(f"/api/monitor/{o['slug']}/{s['slug']}")
            n = fleet.unit_of[(o["slug"], s["slug"])]
            tl = service.timelines[n]["steps"]
            assert m["steps"] and m["n_steps"] == len(service.live_records[n]) == len(tl)
            assert [x["status"] for x in m["steps"]] == [x["status"] for x in tl]
            assert [x["p_fault"] for x in m["steps"]] == [x.get("p_fault") for x in tl]
            tail = server.get(f"/api/monitor/{o['slug']}/{s['slug']}?since={len(tl) - 3}")
            assert tail["steps"] == m["steps"][-3:]
            fired = [x["alert"]["headline"] for x in m["steps"] if x["alert"]]
            feed = [r["what"] for r in sorted(service.feed(unit=n), key=lambda r: r["seq"])]
            assert fired == feed, (o["slug"], s["slug"])
            ts = [x["t"] for x in m["steps"]]
            assert ts[0] == 0 and ts == sorted(ts)
            if s["slug"] != "bearing":                    # ADVISORY stages never alarm
                assert all(x["alert"] is None or x["alert"]["status"] != "ALARM"
                           for x in m["steps"])


# ===========================================================================
# 10. every number shown matches the results JSON
# ===========================================================================

def test_technical_details_match_results_json(server, fleet):
    lobo = _read("artifacts", "runs", "lobo_summary.json")
    m = FU.load_branch_metrics()
    fa = C.FUSION_CONFIG["fault_authority"]
    from webapp.fleet import SLUG_BRANCH
    seen = set()
    for a in fleet.alerts():
        t = server.get(f"/api/alerts/{a['id']}")["technical"]
        b = SLUG_BRANCH[a["stage"]]
        seen.add(b)
        assert t["score"] == m[b].metric_value, b
        assert t["validation_groups"] == m[b].n_validation_groups, b
        assert t["alarm_ready_groups"] == fa["min_validation_groups"]
        assert t["alarm_ready_score"] == fa["min_macro_f1"]
        assert t["packet_bytes"] == A.packet_bytes()
    assert seen == set(SLUG_BRANCH.values())
    assert m["bearing"].metric_value == lobo["pooled"]["macro_f1"]


def test_validation_methods_quote_their_numbers_from_json(fleet):
    from webapp.fleet import validation_method
    m = FU.load_branch_metrics()
    n = m["bearing"].n_validation_groups
    assert f"each of the {n} bearings" in validation_method("bearing", m["bearing"])
    assert f"other {n - 1}" in validation_method("bearing", m["bearing"])
    sup = _read("artifacts", "multistage", "supply", "supply_results.json")
    runs = sup["R1_rule_per_recording"]["scores"]["n"]
    assert f"on {runs} test runs from {m['supply'].n_validation_groups} motors" in \
        validation_method("supply", m["supply"])
    inv = _read("artifacts", "multistage", "inverter_telemetry", "inverter_telemetry_results.json")
    a, b = re.search(r"(\d+)/(\d+)", inv["V1_block_split"]["protocol"]).groups()
    assert f"first {a} %" in validation_method("inverter_telemetry", m["inverter_telemetry"])
    assert f"final {b} %" in validation_method("inverter_telemetry", m["inverter_telemetry"])
    assert f"across {m['winding'].n_validation_groups} sessions" in \
        validation_method("winding", m["winding"])


def test_engineering_view_is_the_dashboards_own_panel_data(server):
    from dashboard import panels as P
    eng = server.get("/api/engineering")
    metrics = FU.load_branch_metrics()
    assert eng["metric_cards"] == json.loads(json.dumps(P.metric_cards(metrics)))
    assert eng["stage_rows"] == json.loads(json.dumps(P.stage_rows(metrics)))
    assert eng["floor"] == C.FUSION_CONFIG["fault_authority"]
    lobo = _read("artifacts", "runs", "lobo_summary.json")
    assert eng["lobo_pooled"] == lobo["pooled"]
    ki05 = next(f for f in lobo["folds"] if f["test_bearing"] == "KI05")
    # NaN (no validation split in that fold) travels as JSON null
    assert eng["ki05"] == {k: (None if isinstance(v, float) and v != v else v)
                           for k, v in ki05.items()}


def test_hardware_numbers_match_rtl_json(server):
    hw = server.get("/api/hardware")
    synth = _read("artifacts", "rtl", "synth", "synth.json")
    system = _read("artifacts", "rtl", "system", "system.json")
    verify = _read("artifacts", "rtl", "verify.json")
    params = _read("artifacts", "rtl", "rtl_params.json")
    assert hw["ran_on_hardware"] is _board_ran()
    assert hw["part"] == synth["part"] and hw["achieved_mhz_core"] == synth["achieved_mhz"]
    assert hw["core_only"]["wns_ns"] == synth["wns_ns"]
    assert hw["full_system"]["wns_ns"] == system["wns_ns"]
    assert hw["full_system"]["timing_met"] == system["timing_met"]
    assert hw["bitstream_written"] == system.get("bitstream_written")
    assert hw["weights"] == params["w_depth"] and hw["macs"] == params["total_macs"]
    cyc = {verify["runs"][k].get("cycles_max") for k in ("v1", "v3", "v5")
           if verify["runs"].get(k, {}).get("cycles_max")}
    assert cyc == {hw["cycles"]}                                  # fixed, one value
    v1 = verify["runs"]["v1"]
    assert hw["v1_committed"]["windows_run"] == v1["windows_run"]
    assert hw["v1_committed"]["class_mismatches"] == v1["class_mismatches"]


def test_about_edge_numbers_match_rtl_json(server):
    e = server.get("/api/about")["edge"]
    verify = _read("artifacts", "rtl", "verify.json")
    params = _read("artifacts", "rtl", "rtl_params.json")
    cyc = next(verify["runs"][k]["cycles_max"] for k in ("v1", "v3", "v5")
               if verify["runs"].get(k, {}).get("cycles_max"))
    assert e["cycles"] == cyc and e["latency_ms_100mhz"] == pytest.approx(cyc / 1e5)
    assert e["weights"] == params["w_depth"]
    assert e["packet_bytes"] == A.packet_bytes() == 21
    assert e["window_bytes"] == A.raw_window_bytes() == 10240
    ev = server.get("/api/about")["evidence"]
    assert ev == {"rolling_window": C.FUSION_CONFIG["rolling_window"],
                  "k_consecutive": C.FUSION_CONFIG["k_consecutive"]}


def test_the_sensors_page_is_gone(server):
    """Removed on 2026-09-24 (owner's decision); old bookmarks land on Hardware."""
    from webapp.server import PAGES as SP
    assert "/sensors" not in SP and "sensors" not in _src("webapp", "static", "js", "common.js")
    assert not os.path.exists(os.path.join(PAGES, "sensors.html"))
    assert server.raw("/api/sensors")[0] == 404


def test_confidence_shown_is_the_fusion_confidence(service):
    """The alert's confidence is the fusion view's at the step it fired."""
    for r in service.feed():
        tl = service.timelines[r["unit_no"]]
        step = next(s for s in tl["steps"] if s.get("alert_seq") == r["seq"])
        assert r["confidence"] == round(round(step["confidence"] * 100) / 100, 2), r["id"]
