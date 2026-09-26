"""
Is board/server.py, running on the PYNQ-Z2, giving the web app the right answers?

The web app's Settings page shows how many bearing windows the FPGA served, but
not whether those answers were right. This asks the running board server the
same question the web app does -- one quantised window per HTTP request,
through drivesentinel.engines.BoardClient, exactly the web app's path -- for
every bearing window the app streams (artifacts/demo/bearing_*.npz, 4 x 120)
plus the 120 packaged board test windows, and compares class and all three
logit registers with the bit-accurate software model (SoftwareEngine), which
is what the FPGA must reproduce. It also records the HTTP round trip per
window, and, if the web app is running, its engine counters.

    .venv/Scripts/python.exe scripts/webapp/check_board_server.py [--board 192.168.2.99:8765]
    -> artifacts/webapp/board_server_check.json

Needs the board server running (board/SERVER.md). Read-only on both ends.

WHAT THIS IS NOT. An accuracy figure: every one of these windows comes from a
bearing the deployed network was trained on. It checks the hardware and the
HTTP path against the software model, nothing more.
"""

from __future__ import annotations

import argparse
import glob
import http.cookiejar
import json
import os
import platform
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivesentinel.engines import BoardClient, SoftwareEngine  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "webapp", "board_server_check.json")


def windows():
    """(source, index, float window) for every window the check sends."""
    for p in sorted(glob.glob(os.path.join(ROOT, "artifacts", "demo", "bearing_*.npz"))):
        z = np.load(p, allow_pickle=False)
        for i, w in enumerate(z["spectra"]):
            yield f"demo {z['bearing']}", i, w
    T = np.load(os.path.join(ROOT, "board", "test_windows.npz"))
    for i, w in enumerate(T["x_float"]):
        yield "board/test_windows.npz", i, w


def webapp_engine_state(base: str):
    """The running web app's /api/engine, via the documented demo sign-in."""
    from webapp import auth
    user, u = next(iter(auth.users().items()))
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try:
        op.open(urllib.request.Request(
            base + "/signin", method="POST",
            data=urllib.parse.urlencode({"username": user, "password": u["password"]}).encode()),
            timeout=5)
        return json.load(op.open(base + "/api/engine", timeout=5))
    except Exception as e:                      # the app need not be running
        return {"unavailable": str(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="192.168.2.99:8765")
    ap.add_argument("--webapp", default="http://127.0.0.1:8000")
    a = ap.parse_args()

    before = webapp_engine_state(a.webapp)
    sw = SoftwareEngine()
    board = BoardClient(a.board, timeout_s=5.0)
    health = board.check()

    rows, rtt = [], []
    for src, i, w in windows():
        want = sw.infer(w)
        t0 = time.perf_counter()
        got = board.infer(sw.quantise(w))
        rtt.append(time.perf_counter() - t0)
        rows.append({"source": src, "index": i,
                     "class_ok": got["class"] == want["class"],
                     "logits_ok": got["logits"] == want["logits"],
                     "cycles": got["cycles"], "engine": got["engine"]})

    by_src = {}
    for r in rows:
        s = by_src.setdefault(r["source"], {"windows": 0, "class_agree": 0, "logits_exact": 0})
        s["windows"] += 1
        s["class_agree"] += r["class_ok"]
        s["logits_exact"] += r["logits_ok"]
    ms = np.asarray(rtt) * 1e3
    out = {
        "what": ("board/server.py on the PYNQ-Z2, asked over HTTP through "
                 "drivesentinel.engines.BoardClient (the web app's path); every answer "
                 "compared with SoftwareEngine, the bit-accurate model. In-sample "
                 "windows: a hardware/path check, not an accuracy figure."),
        "date_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "laptop": platform.platform(),
        "board_health": health,
        "windows": len(rows),
        "class_agree": sum(r["class_ok"] for r in rows),
        "logits_exact": sum(r["logits_ok"] for r in rows),
        "cycles_values": sorted({r["cycles"] for r in rows}),
        "engines_reported": sorted({r["engine"] for r in rows}),
        "by_source": by_src,
        "mismatches": [r for r in rows if not (r["class_ok"] and r["logits_ok"])],
        "http_round_trip_ms": {"median": float(np.median(ms)),
                               "p95": float(np.percentile(ms, 95)),
                               "min": float(ms.min()), "max": float(ms.max()),
                               "note": "laptop -> board -> laptop over the direct cable: "
                                       "upload of the int8 window (quantised on the "
                                       "laptop), the FPGA inference, the JSON reply"},
        "webapp_engine_before": before,
        "webapp_engine_after": webapp_engine_state(a.webapp),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_source",)}, indent=2))
    return 0 if not out["mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
