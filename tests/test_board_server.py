"""
The board-to-web connection, tested with no board attached.

board/server.py runs here against the software stand-in for `pynq`
(board/dryrun/pynq), whose "core" is the bit-accurate RTL model with the RTL's
real register semantics -- including the sticky DONE bit. So these tests check
the SERVER's code and the web app's ENGINE ROUTING, not the FPGA. Agreement with
the golden reference is true by construction behind the mock; what is not, and
is tested:

  * the server reads the right registers and refuses a wrong build ID;
  * the busy-bit protocol returns fresh results on every window, not only the first;
  * the web app falls back to the software engine when the board is down, and
    returns to the board when it answers again;
  * a verdict is labelled "FPGA (PYNQ-Z2)" only when the board produced it;
  * whose verdict drives the bearing stage (the web app: the deployed network),
    and that the engine computing it never changes an alert.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")
DRY = os.path.join(BOARD, "dryrun")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_pynq():
    sys.path.insert(0, DRY)
    try:
        pynq = importlib.import_module("pynq")
        importlib.reload(pynq)
        assert pynq.__version__.startswith("MOCK")
        yield pynq
    finally:
        sys.path.remove(DRY)
        sys.modules.pop("pynq", None)


@pytest.fixture(scope="module")
def bsrv():
    spec = importlib.util.spec_from_file_location("ds_board_server",
                                                  os.path.join(BOARD, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def windows():
    return np.load(os.path.join(BOARD, "test_windows.npz"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    """board/server.py on a real port, in a thread."""

    def __init__(self, bsrv, acc, port=None):
        self.port = port or _free_port()
        self.httpd = bsrv.serve(acc, "127.0.0.1", self.port)
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(timeout=5)

    @property
    def address(self):
        return f"127.0.0.1:{self.port}"


def _acc(bsrv, mock_pynq, build_id=None):
    def factory(bit):
        ol = mock_pynq.Overlay(bit)
        if build_id is not None:
            ol.ds_0.s_axi.build_id = build_id     # ol.ds_0 is a bare hierarchy proxy
        return ol
    return bsrv.Accelerator(os.path.join(BOARD, "drivesentinel.bit"),
                            overlay_factory=factory, allocate=mock_pynq.allocate)


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

def test_server_imports_and_serves_on_python_36s_http_server(bsrv, mock_pynq, windows,
                                                             monkeypatch):
    """
    The board runs Python 3.6 (PYNQ 2.5), which has no http.server
    .ThreadingHTTPServer (added in 3.7). Remove it, re-import server.py, and
    check the fallback both imports and actually answers a request.
    """
    import http.server
    import urllib.request
    monkeypatch.delattr(http.server, "ThreadingHTTPServer")
    spec = importlib.util.spec_from_file_location("ds_board_server_py36",
                                                  os.path.join(BOARD, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.ThreadingHTTPServer.__module__ == m.__name__     # the fallback, not stdlib's
    acc = _acc(m, mock_pynq)
    with Running(m, acc) as srv:
        h = json.loads(urllib.request.urlopen(f"http://{srv.address}/health", timeout=5).read())
        assert h["build_id"] == "0x23eb56bf"


def test_board_side_files_use_no_python_37_features():
    """The board's Python is 3.6. Static guard for the features that bit or would
    have bitten on it: postponed annotations (3.7), and stdlib names added in 3.7
    imported without a fallback."""
    for f in ("server.py", "mmio_resolve.py", "frontend.py"):
        src = open(os.path.join(BOARD, f), encoding="utf-8").read()
        assert "from __future__ import annotations" not in src, f
        assert ":=" not in src, f                                  # 3.8
        for name in ("ThreadingHTTPServer",):
            for line in src.splitlines():
                if f"import" in line and name in line and "from http.server import" in line:
                    idx = src.index(line)
                    assert "try:" in src[max(0, idx - 80):idx], (f, line)


def test_server_register_map_equals_ds_defs(bsrv):
    txt = open(os.path.join(ROOT, "rtl", "ds_defs.vh"), encoding="utf-8").read()
    regs = {m.group(1): int(m.group(2), 16) for m in
            re.finditer(r"`define DS_REG_(\w+)\s+8'h([0-9A-Fa-f]+)", txt)}
    bits = {m.group(1): int(m.group(2)) for m in
            re.finditer(r"`define DS_ST_(\w+)\s+(\d+)", txt)}
    magic = int(re.search(r"`define DS_MAGIC\s+32'h([0-9A-Fa-f]+)", txt).group(1), 16)
    assert bsrv.REG == regs
    assert bsrv.ST == bits
    assert bsrv.MAGIC == magic
    params = json.load(open(os.path.join(ROOT, "artifacts", "rtl", "rtl_params.json")))
    assert bsrv.EXPECTED_BUILD_ID == params["expected_build_id"] == 0x23EB56BF
    assert bsrv.EXPECTED_CONFIG == (params["frac_width"], 1, params["logit_shift"])


def test_server_refuses_a_build_id_mismatch(bsrv, mock_pynq):
    with pytest.raises(bsrv.BuildIdMismatch):
        _acc(bsrv, mock_pynq, build_id=0xDEADBEEF)
    acc = _acc(bsrv, mock_pynq)
    assert acc.identity["build_id"] == "0x23eb56bf"


def test_board_results_match_golden_reference_on_real_windows(bsrv, mock_pynq, windows):
    """All 120 packaged windows over HTTP, float path: class vs golden_reference,
    logit registers vs the RTL model's predicted registers, cycles."""
    import urllib.request
    sys.path.insert(0, os.path.join(BOARD, "golden"))
    try:
        import golden_reference as golden
    finally:
        sys.path.remove(os.path.join(BOARD, "golden"))
    golden_pred = golden.predict(windows["x_float"].astype(np.float64))
    with Running(bsrv, _acc(bsrv, mock_pynq)) as srv:
        health = json.loads(urllib.request.urlopen(
            f"http://{srv.address}/health", timeout=5).read())
        assert health["build_id"] == "0x23eb56bf" and health["ok"]
        got = []
        for w in windows["x_float"]:
            req = urllib.request.Request(f"http://{srv.address}/infer",
                                         data=np.ascontiguousarray(w, "<f4").tobytes(),
                                         method="POST")
            got.append(json.loads(urllib.request.urlopen(req, timeout=5).read()))
    assert [g["class"] for g in got] == golden_pred.tolist()
    assert np.array_equal(np.array([g["logits"] for g in got]), windows["logit_reg"])
    assert {g["cycles"] for g in got} == {int(windows["expected_cycles"])}
    assert {g["engine"] for g in got} == {"FPGA (PYNQ-Z2)"}


def test_server_int8_path_equals_float_path(bsrv, mock_pynq, windows):
    acc = _acc(bsrv, mock_pynq)
    for i in range(3):
        a = acc.infer_bytes(windows["x_int8"][i].tobytes())
        b = acc.infer_bytes(np.ascontiguousarray(windows["x_float"][i], "<f4").tobytes())
        assert a == b
    with pytest.raises(ValueError):
        acc.infer_bytes(b"\x00" * 100)


def test_server_uses_the_busy_protocol_not_the_sticky_done_bit(bsrv, mock_pynq, windows):
    """Consecutive windows with different classes: stale registers would repeat."""
    acc = _acc(bsrv, mock_pynq)
    idx = [int(np.flatnonzero(windows["pred_rtl"] == c)[0]) for c in (0, 1, 2, 0, 2, 1)]
    for i in idx:
        r = acc.infer(windows["x_int8"][i])
        assert r["class"] == int(windows["pred_rtl"][i])
        assert r["logits"] == [int(v) for v in windows["logit_reg"][i]]


# ---------------------------------------------------------------------------
# the web app's engine routing
# ---------------------------------------------------------------------------

def test_software_engine_equals_the_board_register_for_register(windows):
    from drivesentinel.engines import SoftwareEngine
    sw = SoftwareEngine()
    for i in range(0, 120, 7):
        r = sw.infer(windows["x_float"][i])
        assert r["class"] == int(windows["pred_rtl"][i])
        assert r["logits"] == [int(v) for v in windows["logit_reg"][i]]
        assert r["engine"] == "Software" and r["cycles"] is None


def test_no_address_means_software(windows):
    from drivesentinel.engines import EngineRouter
    r = EngineRouter(address=None)
    assert r.infer(windows["x_float"][0])["engine"] == "Software"
    st = r.state()
    assert st["active"] == "Software" and st["board"] == "not set"


def test_fallback_engages_when_the_board_is_down_and_recovers(bsrv, mock_pynq, windows):
    from drivesentinel.engines import EngineRouter
    port = _free_port()
    r = EngineRouter(address=f"127.0.0.1:{port}", retry_s=0.3, timeout_s=1.0)
    # board down from the start
    res = r.infer(windows["x_float"][0])
    assert res["engine"] == "Software"
    assert r.state()["active"] == "Software" and r.state()["board"] == "unreachable"
    # board comes up on that port: after the retry interval the router goes back to it
    with Running(bsrv, _acc(bsrv, mock_pynq), port=port):
        time.sleep(0.35)
        res = r.infer(windows["x_float"][1])
        assert res["engine"] == "FPGA (PYNQ-Z2)" and res["cycles"] == 1786724
        st = r.state()
        assert st["active"] == "FPGA (PYNQ-Z2)" and st["build_id"] == "0x23eb56bf"
    # board goes away again: the very next window falls back, nothing is lost
    res = r.infer(windows["x_float"][2])
    assert res["engine"] == "Software"
    assert res["class"] == int(windows["pred_rtl"][2])
    assert r.state()["active"] == "Software"
    assert r.state()["windows"] == {"Software": 2, "FPGA (PYNQ-Z2)": 1}


class _Impostor(BaseHTTPRequestHandler):
    """Claims to be the FPGA, with the wrong build ID."""
    build_id = "0xdeadbeef"

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True, "build_id": self.build_id, "engine": "FPGA (PYNQ-Z2)"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send({"class": 2, "logits": [0, 0, 1], "cycles": 1786724,
                    "engine": "FPGA (PYNQ-Z2)", "build_id": self.build_id})

    def log_message(self, *a):
        pass


def test_a_board_with_the_wrong_build_id_is_refused(windows):
    from drivesentinel.engines import EngineRouter
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Impostor)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        r = EngineRouter(address=f"127.0.0.1:{httpd.server_address[1]}", retry_s=60)
        res = r.infer(windows["x_float"][0])
        assert res["engine"] == "Software"
        assert res["class"] == int(windows["pred_rtl"][0])   # the software verdict, not the impostor's
        assert r.state()["board"] == "build ID mismatch" and r.state()["active"] == "Software"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_fpga_label_only_ever_comes_from_the_board_client():
    """Static check of the rule in engines.py: the FPGA label is written in exactly
    one place, BoardClient.infer, and state() derives `active` from board_ok."""
    src = open(os.path.join(ROOT, "drivesentinel", "engines.py"), encoding="utf-8").read()
    writes = [m.start() for m in re.finditer(r'"engine":\s*ENGINE_FPGA', src)]
    assert len(writes) == 1
    board_infer = src.index("class BoardClient")
    assert board_infer < writes[0] < src.index("class EngineStats")
    assert "ENGINE_FPGA if (self.board is not None and self.board_ok)" in src


def test_held_out_policy_keeps_the_fold_models_verdict():
    """verdict="held_out": every recorded bearing window still goes through the
    engine, but the stage's probs stay the held-out fold model's."""
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, RecordedScenarioSource
    base = RecordedScenarioSource()
    src = BoardSource(base, EngineRouter(address=None), verdict="held_out")
    bearing = [u for u in base.units() if u.branch == "bearing"]
    assert bearing
    for u in bearing[:2]:
        a = list(base.observations(u.unit_no))
        b = list(src.observations(u.unit_no))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.deployment_in_sample and y.engine["engine"] == "Software"
            assert np.array_equal(x.probs, y.probs)
    other = next(u for u in base.units() if u.branch != "bearing")
    for ob in src.observations(other.unit_no):
        assert ob.engine is None and ob.window is None


def test_engine_policy_uses_the_deployed_networks_verdict():
    """verdict="engine" (the web app's setting): the bearing stage's probs are the
    softmax of the engine's logit registers, whose argmax is the engine's class."""
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, RecordedScenarioSource
    base = RecordedScenarioSource()
    src = BoardSource(base, EngineRouter(address=None))          # default: "engine"
    assert src.verdict == "engine"
    u = next(u for u in base.units() if u.branch == "bearing")
    for ob in list(src.observations(u.unit_no))[:10]:
        assert int(np.argmax(ob.probs)) == ob.engine["class"]
        assert abs(float(ob.probs.sum()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        BoardSource(base, EngineRouter(address=None), verdict="whatever")


def _run_fleet(src):
    from webapp.service import Service
    svc = Service(source=src, endpoint=None, pace_s=0.0)
    svc.start_fleet()
    t0 = time.time()
    while not svc.fleet_done:
        assert time.time() - t0 < 180
        time.sleep(0.05)
    return svc


def _bearing_alerts(svc):
    return {u.scenario: [(r["seq"], r["status"], r["fault"])
                         for r in sorted(svc.feed(unit=n), key=lambda r: r["seq"])]
            for n, u in svc.units.items() if u.branch == "bearing"}


def test_engine_policy_alerts_are_the_same_on_the_fpga_and_in_software(bsrv, mock_pynq):
    """The owner's decision of 2026-09-24, as the web app runs it. Which engine
    computes the deployed network's verdict must never change an alert: stream
    once with no board (Software) and once through board/server.py (FPGA,
    behind the mock), and compare every bearing alert. Also: KI05 -- the
    documented miss under the held-out model -- now raises an ALARM, and every
    alert finds its trace on the Why? page."""
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, RecordedScenarioSource
    base = RecordedScenarioSource()
    sw = _run_fleet(BoardSource(base, EngineRouter(address=None)))
    with Running(bsrv, _acc(bsrv, mock_pynq)) as srv:
        fpga_src = BoardSource(base, EngineRouter(address=srv.address, timeout_s=5.0))
        fp = _run_fleet(fpga_src)
        counts = fpga_src.router.state()["windows"]
    assert counts["Software"] == 0 and counts["FPGA (PYNQ-Z2)"] == 480
    assert _bearing_alerts(sw) == _bearing_alerts(fp)
    alerts = _bearing_alerts(fp)
    assert alerts["bearing_KI05"][-1][1:] == ("ALARM", "inner_race")
    assert alerts["bearing_K001"] == []
    for svc in (sw, fp):
        for r in svc.feed():
            assert svc.alert_detail(r["id"])["trigger_index"] is not None, r["id"]


def test_a_new_machine_window_is_driven_by_the_engine(windows):
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, Observation

    class Live:
        def units(self):
            return []

        def observations(self, n):
            for i in range(3):
                yield Observation(step=i, t_s=float(i),
                                  labels=["healthy", "inner_race", "outer_race"],
                                  probs=None, frame={}, context={},
                                  window=windows["x_float"][i], deployment_in_sample=False)
    got = list(BoardSource(Live(), EngineRouter(address=None)).observations(1))
    for i, ob in enumerate(got):
        assert int(np.argmax(ob.probs)) == int(windows["pred_rtl"][i])
        assert abs(float(ob.probs.sum()) - 1.0) < 1e-9


def test_web_app_reports_software_with_no_board_and_a_bad_address(tmp_path, monkeypatch):
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, RecordedScenarioSource
    from webapp import server as S
    from webapp.service import Service
    from webapp.testing import LiveServer
    monkeypatch.setattr(S, "ENGINE_FILE", str(tmp_path / "engine.json"))
    src = BoardSource(RecordedScenarioSource(), EngineRouter(address=None, retry_s=60))
    svc = Service(source=src, endpoint=None, pace_s=0.0)
    with LiveServer(S.create_app(service=svc, run_fleet=False)) as srv:
        assert srv.login()[0] == 303
        e = srv.get("/api/engine")
        assert e["active"] == "Software" and e["board"] == "not set"
        status, e = srv.post("/api/engine",
                             json.dumps({"address": f"127.0.0.1:{_free_port()}"}).encode(),
                             "application/json")
        assert status == 200 and e["active"] == "Software"
        # stream the fleet: every bearing window goes to the (down) board, falls back
        svc.start_fleet()
        t0 = time.time()
        while not svc.fleet_done:
            assert time.time() - t0 < 120
            time.sleep(0.05)
        e = srv.get("/api/engine")
        assert e["active"] == "Software" and e["board"] == "unreachable"
        assert e["windows"]["FPGA (PYNQ-Z2)"] == 0 and e["windows"]["Software"] > 0
        assert json.load(open(tmp_path / "engine.json"))["address"].startswith("127.0.0.1:")
        page = srv.get("/settings")
        assert "Bearing engine" in page and "Board address" in page
