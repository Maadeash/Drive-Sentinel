"""
The DSP front end on the board, and the live view of it.

board/frontend.py is drivesentinel/dsp.py ported to run on the PYNQ-Z2's ARM
(Python 3.6, older scipy): same maths, filters shipped precomputed, only
scipy.signal.sosfilt needed. These tests hold it to the frozen original --
exactly, not approximately, because the FPGA's input is 8-bit and one flipped
quantisation step is a different network input:

  * the params file is what its generator writes from config.py today;
  * on a synthetic signal and (when the data is on disk) on real test-rig data,
    the port's float output equals dsp.process_recording's and its int8 input to
    the network is identical, with scipy.fft AND with the numpy.fft fallback the
    board uses when its scipy predates scipy.fft;
  * board/server.py's POST /process, on the mock pynq, returns int8 inputs equal
    to the laptop's and logits equal to the bit-accurate model's;
  * webapp/boardlive.py streams through it, checks every window, raises alerts,
    and never puts a file name, a specimen ID or an internal field on the page.

What these tests cannot show is how long the board's ARM takes, or that the
board's own scipy agrees: that is measured on the board, window by window, by
the live view's integrity check (claims_audit 1.29).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import dsp as D
from drivesentinel.engines import SoftwareEngine, encode_raw, process_raw
from tests.test_board_server import Running, _acc, bsrv, mock_pynq  # noqa: F401  (fixtures)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")
REAL = os.path.join(C.DATA_ROOT, "KA04", "N09_M07_F10_KA04_1.mat")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fe():
    return _load("ds_board_frontend", os.path.join(BOARD, "frontend.py"))


@pytest.fixture(scope="module")
def sw():
    return SoftwareEngine()


def synthetic(seed=0, seconds=4.0, f_elec=60.0):
    """Two phase currents with fault sidebands, and vibration with impulses at the
    outer-race rate ringing a 4 kHz resonance -- enough structure that every one
    of the five channels is non-trivial."""
    rng = np.random.default_rng(seed)
    n = int(seconds * C.FS_FAST) + 1
    t = np.arange(n) / C.FS_FAST
    fr = f_elec / C.POLE_PAIRS
    side = C.FAULT_ORDERS_NOMINAL["bpfo"] * fr
    c1 = (np.sin(2 * np.pi * f_elec * t) + 0.03 * np.sin(2 * np.pi * (f_elec + side) * t)
          + 0.01 * rng.standard_normal(n))
    c2 = (np.sin(2 * np.pi * f_elec * t - 2 * np.pi / 3)
          + 0.03 * np.sin(2 * np.pi * (f_elec - side) * t) + 0.01 * rng.standard_normal(n))
    ring = np.exp(-t[:640] * 900) * np.sin(2 * np.pi * 4000 * t[:640])
    pulses = np.zeros(n)
    pulses[(np.arange(0, seconds, 1 / side) * C.FS_FAST).astype(int)] = 1.0
    vib = np.convolve(pulses, ring)[:n] + 0.2 * rng.standard_normal(n)
    return c1, c2, vib


def _int8(sw, specs):
    return np.stack([sw.quantise(w) for w in specs])


# ---------------------------------------------------------------------------
# the port
# ---------------------------------------------------------------------------

def test_params_file_is_what_the_generator_writes_today():
    gen = _load("ds_make_frontend_params", os.path.join(BOARD, "make_frontend_params.py"))
    with open(os.path.join(BOARD, "frontend_params.json"), encoding="utf-8") as fh:
        committed = json.load(fh)
    assert committed == json.loads(json.dumps(gen.params())), \
        "board/frontend_params.json is stale: run board/make_frontend_params.py"


def test_the_board_needs_only_sosfilt_from_scipy():
    src = open(os.path.join(BOARD, "frontend.py"), encoding="utf-8").read()
    scipy_imports = [l.strip() for l in src.splitlines() if re.match(r"\s*(from|import) scipy", l)]
    assert scipy_imports == ["from scipy.fft import fft as _fft, ifft as _ifft, rfft as _rfft",
                             "from scipy.signal import sosfilt           # scipy >= 0.16; "
                             "the only compiled routine used"]
    # the scipy.fft import sits in a try with a numpy.fft (+ Bluestein) fallback
    assert "except ImportError:" in src and 'use_fft("numpy.fft")' in src


def test_raw_spectrum_edges_are_the_laptops_not_recomputed_on_the_board(fe):
    """numpy 1.13's geomspace leaves its first edge at 200.00000000000003 Hz and drops
    the exact 200 Hz bin: on the PYNQ-Z2 that was 1 differing input byte per window,
    up to 60 when it moved the channel's median (2026-09-24)."""
    code = "\n".join(l.split("#")[0] for l in
                     open(os.path.join(BOARD, "frontend.py"), encoding="utf-8").read().splitlines())
    assert "geomspace(" not in code
    fs_vib, edges = fe._RAW_EDGES
    assert fs_vib == C.FS_FAST / C.DECIMATE_VIBRATION and edges[0] == C.RAW_SPECTRUM_HZ[0]
    assert np.array_equal(edges, np.geomspace(C.RAW_SPECTRUM_HZ[0],
                                              min(C.RAW_SPECTRUM_HZ[1], 0.98 * fs_vib / 2),
                                              C.N_ORDER_BINS + 1))


@pytest.fixture
def numpy_fft(fe):
    """The board's path (scipy 0.19.1 has no scipy.fft): numpy.fft + Bluestein."""
    fe.use_fft("numpy.fft")
    try:
        yield fe
    finally:
        fe.use_fft("scipy.fft")


@pytest.mark.parametrize("backend", ["scipy.fft", "numpy.fft"])
def test_port_equals_frozen_dsp_on_a_synthetic_signal(fe, sw, backend):
    fe.use_fft(backend)
    try:
        c1, c2, vib = synthetic()
        ref, _ = D.process_recording(c1, c2, vib)
        got, diags, timings = fe.process_recording(c1, c2, vib)
    finally:
        fe.use_fft("scipy.fft")
    assert got.shape == ref.shape == (7, 5, C.N_ORDER_BINS) and got.dtype == np.float32
    assert np.allclose(got, ref, rtol=0, atol=1e-5)
    assert np.array_equal(_int8(sw, got), _int8(sw, ref))
    assert all(abs(d["rpm"] - 60.0 * 60.0 / C.POLE_PAIRS) < 1.0 for d in diags)
    assert set(timings) == {"decimate_ms", "envelope_ms", "spectra_ms"}


def test_port_is_bitwise_equal_with_scipy_fft(fe):
    c1, c2, vib = synthetic(seed=3)
    assert np.array_equal(fe.process_recording(c1, c2, vib)[0], D.process_recording(c1, c2, vib)[0])


def test_bluestein_is_the_dft_for_the_board_s_prime_lengths(fe):
    """128,001 = 3 x 42,667 is the Hilbert length for 4 s of vibration; numpy 1.13's
    fftpack is O(n x p) on it (a data file did not finish in 280 s on the board)."""
    assert fe._largest_prime_factor(128001) == 42667 > fe.SMOOTH
    assert fe._largest_prime_factor(32000) == 5 and fe._largest_prime_factor(4000) == 5
    rng = np.random.default_rng(7)
    for n in (4099, 16001, 128001):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ref = np.fft.fft(x)
        assert np.abs(fe.bluestein_fft(x) - ref).max() <= 1e-12 * np.abs(ref).max(), n
        assert np.allclose(fe._np_ifft(ref), x, rtol=0, atol=1e-12)
        assert np.allclose(fe._np_rfft(x.real), np.fft.rfft(x.real), rtol=0, atol=1e-9)


def test_the_board_s_fft_path_uses_bluestein_on_the_envelope(numpy_fft, monkeypatch):
    sizes = []
    real = numpy_fft.bluestein_fft
    monkeypatch.setattr(numpy_fft, "bluestein_fft", lambda x: (sizes.append(x.size), real(x))[1])
    numpy_fft.process_recording(*synthetic(seed=4))
    assert sizes == [128001] * 4           # fft + ifft, two vibration bands; nothing else


@pytest.mark.skipif(not os.path.exists(REAL), reason="raw test-rig data not on this machine")
def test_board_fft_path_equals_frozen_dsp_on_real_test_rig_data(numpy_fft, sw):
    from drivesentinel import dataset as DS
    c1, c2, vib, _ = DS.load_analysis_channels(REAL)
    ref, _ = D.process_recording(c1, c2, vib)
    got, _, _ = numpy_fft.process_recording(c1, c2, vib)
    assert np.allclose(got, ref, rtol=0, atol=1e-5)
    assert np.array_equal(_int8(sw, got), _int8(sw, ref))


def test_two_workers_give_the_single_process_output_exactly():
    """The server splits the envelopes and the windows across the board's two cores
    (start_workers(2)); the output must not change by a bit. Imported as the server
    imports it, so worker processes can find the module."""
    import sys
    sys.path.insert(0, BOARD)
    try:
        import frontend as F
        c1, c2, vib = synthetic(seed=5)
        one, d1, _ = F.process_recording(c1, c2, vib)
        assert F.start_workers(2) == 2
        try:
            two, d2, t = F.process_recording(c1, c2, vib)
        finally:
            F.stop_workers()
        assert F.WORKERS == 1
        assert np.array_equal(one, two) and d1 == d2
        assert min(t.values()) >= 0.0
    finally:
        sys.path.remove(BOARD)
        sys.modules.pop("frontend", None)


@pytest.mark.skipif(not os.path.exists(REAL), reason="raw test-rig data not on this machine")
def test_port_equals_frozen_dsp_on_real_test_rig_data(fe, sw):
    from drivesentinel import dataset as DS
    c1, c2, vib, _ = DS.load_analysis_channels(REAL)
    ref, _ = D.process_recording(c1, c2, vib)
    got, diags, _ = fe.process_recording(c1, c2, vib)
    assert np.array_equal(got, ref)
    assert np.array_equal(_int8(sw, got), _int8(sw, ref))


# ---------------------------------------------------------------------------
# board/server.py POST /process, on the mock pynq
# ---------------------------------------------------------------------------

def test_raw_body_round_trips_through_the_servers_decoder(bsrv):
    c1, c2, vib = synthetic(seconds=0.1)
    fs, (a, b, v) = bsrv.decode_raw(encode_raw(c1, c2, vib, C.FS_FAST))
    assert fs == C.FS_FAST
    assert np.array_equal(a, c1) and np.array_equal(b, c2) and np.array_equal(v, vib)
    body = encode_raw(c1, c2, vib, C.FS_FAST)
    for bad in (body[:3], body[:-8], body + b"\0" * 8):
        with pytest.raises(ValueError):
            bsrv.decode_raw(bad)


def test_process_on_the_board_equals_the_laptop(bsrv, mock_pynq, sw):
    acc = _acc(bsrv, mock_pynq)
    c1, c2, vib = synthetic(seed=1)
    with Running(bsrv, acc) as srv:
        h = json.loads(urllib.request.urlopen(f"http://{srv.address}/health", timeout=5).read())
        assert h["frontend"]["available"] and h["frontend"]["fft"] == "scipy.fft"
        out = process_raw(srv.address, c1, c2, vib, C.FS_FAST)
    ref = _int8(sw, D.process_recording(c1, c2, vib)[0])
    assert len(out["windows"]) == 7 and out["bytes_sent"] > 6_000_000
    import base64
    for k, w in enumerate(out["windows"]):
        q = np.frombuffer(base64.b64decode(w["int8"]), dtype=np.int8).reshape(5, 512)
        assert np.array_equal(q, ref[k]), k
        assert w["logits"] == sw.infer_int8(q)["logits"], k
        assert len(w["view"]) == C.N_ORDER_BINS // 2
    assert set(out["timings"]) >= {"decimate_ms", "envelope_ms", "spectra_ms", "quantise_ms",
                                   "fpga_ms", "total_ms"}


def test_process_says_why_when_the_front_end_is_missing(bsrv, mock_pynq, monkeypatch):
    monkeypatch.setattr(bsrv, "FRONTEND", None)
    monkeypatch.setattr(bsrv, "FRONTEND_ERROR", "ImportError: cannot import name 'sosfilt'")
    acc = _acc(bsrv, mock_pynq)
    c1, c2, vib = synthetic(seconds=1.2)
    with Running(bsrv, acc) as srv:
        with pytest.raises(urllib.error.HTTPError) as e:
            process_raw(srv.address, c1, c2, vib, C.FS_FAST)
        assert e.value.code == 503 and "sosfilt" in json.loads(e.value.read())["error"]
        h = json.loads(urllib.request.urlopen(f"http://{srv.address}/health", timeout=5).read())
        assert h["frontend"]["available"] is False and "sosfilt" in h["frontend"]["error"]


# ---------------------------------------------------------------------------
# webapp/boardlive.py
# ---------------------------------------------------------------------------

def _feed_stream(bsrv, mock_pynq, tmp_path, monkeypatch, address, files_per_unit=2,
                 frontend=True):
    """The web app's real stream -- Service over BoardSource(RecordedScenarioSource)
    -- with the bearing feed pointed at board/server.py on the mock pynq, and each
    bearing unit's raw files replaced by synthetic 4 s signals."""
    from drivesentinel import dataset as DS
    from drivesentinel.engines import EngineRouter
    from drivesentinel.sources import BoardSource, RecordedScenarioSource
    from webapp import boardlive as BL
    from webapp.service import Service
    base = RecordedScenarioSource()
    bearing = {u.unit_no: {"scenario": u.scenario, "label": "Lift %02d" % u.unit_no}
               for u in base.units() if u.branch == "bearing"}
    sig = {}
    def paths(scenario, data_root=None):
        out = []
        for i in range(files_per_unit):
            p = tmp_path / ("%s_%d.mat" % (scenario, i))
            p.write_bytes(b"")
            sig[str(p)] = synthetic(seed=len(sig) + i)
            out.append(str(p))
        return out
    monkeypatch.setattr(BL, "segment_paths", paths)
    monkeypatch.setattr(DS, "load_analysis_channels", lambda p: sig[p] + (None,))
    router = EngineRouter(address=address)
    feed = BL.BoardFeed(router, bearing)
    svc = Service(source=BoardSource(base, router, verdict="engine", feed=feed),
                  endpoint=None, pace_s=0.0)
    return svc, feed, bearing, router


def _run(svc, timeout=240):
    svc.start_fleet()
    t0 = time.time()
    while not svc.fleet_done:
        assert time.time() - t0 < timeout, "stream did not finish"
        time.sleep(0.05)


def test_the_fleet_stream_takes_its_bearing_windows_from_the_board(bsrv, mock_pynq, tmp_path,
                                                                    monkeypatch):
    with Running(bsrv, _acc(bsrv, mock_pynq)) as srv:
        svc, feed, bearing, router = _feed_stream(bsrv, mock_pynq, tmp_path, monkeypatch,
                                                  srv.address)
        _run(svc)
    s = feed.state()
    assert s["status"] == "done", (s["reason"], s["detail"])
    t = s["totals"]
    n_win = 7 * 2 * len(bearing)
    assert t["on_board_windows"] == t["windows"] == n_win and t["on_laptop"] == 0
    assert t["int8_identical"] == t["verdict_same"] == t["fpga_exact"] == n_win
    assert router.state()["windows"]["FPGA (PYNQ-Z2)"] == n_win
    for n in bearing:                       # every bearing step the pages show came from the board
        recs = svc.live_records[n]
        assert len(recs) == 14
        assert all(r["engine"] == {"engine": "FPGA (PYNQ-Z2)", "cycles": 1786724, "dsp": "board"}
                   for r in recs)
        assert all(len(r["frame"]["spectrum"]) == C.N_ORDER_BINS // 2 for r in recs)
        assert [r["t"] for r in recs] == [0.5 * k for k in range(14)]
    others = [n for n in svc.units if n not in bearing]     # the laptop stages ran as before
    assert all(len(svc.live_records[n]) == len(svc.timelines[n]["steps"]) for n in others)
    text = json.dumps(s)
    for leak in (r"KA04|KI18|KI05|K001", r"\.mat\b", r"held_out", r"top_class",
                 r"(?i)recording", r"(?i)scenario", r"DEMO-\d", r"\bS5\b"):
        assert not re.search(leak, text), leak


def test_without_the_board_the_bearing_stays_on_its_stored_path(bsrv, mock_pynq, tmp_path,
                                                                  monkeypatch):
    svc, feed, bearing, router = _feed_stream(bsrv, mock_pynq, tmp_path, monkeypatch, "")
    assert feed.begin() is False and "address" in feed.state()["reason"]
    monkeypatch.setattr(bsrv, "FRONTEND", None)
    monkeypatch.setattr(bsrv, "FRONTEND_ERROR", "ModuleNotFoundError: No module named 'scipy'")
    with Running(bsrv, _acc(bsrv, mock_pynq)) as srv:
        router.set_address(srv.address)
        assert feed.begin() is False
        s = feed.state()
        assert "not installed" in s["reason"] and "scipy" in s["detail"]
        assert s["board"]["frontend"] is False
        _run(svc)                           # the stream still runs, on the stored windows
    for n in bearing:
        assert len(svc.live_records[n]) == len(svc.timelines[n]["steps"])
        assert all((r["engine"] or {}).get("dsp") == "laptop" for r in svc.live_records[n])
