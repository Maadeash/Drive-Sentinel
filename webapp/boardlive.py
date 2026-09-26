"""
The bearing stage, computed on the board: raw sensor data in, verdicts out.

When the PYNQ-Z2 answers with its DSP front end available, this feed is where
every lift's bearing observations come from -- for the Fleet, the Alerts, Live
monitor and the Why? pages alike (the owner's decision of 2026-09-24: the bearing
is processed on the board, the other three stages on the laptop). One worker
thread takes the raw data files behind the four bearing lifts in turn -- 64 kHz
motor current (two phases) and vibration, 4 s per file, straight from
data/<bearing>/*.mat -- and sends each to board/server.py's POST /process. The
board filters, demodulates and computes the spectra on its ARM cores, quantises,
and runs each 1 s window through the FPGA. Its 7 verdicts per file become 7
observations of that lift's bearing unit, with the board's own spectrum as the
frame. The fleet stream (webapp/service.py) takes them as they arrive: the board
is the bottleneck, so a unit whose next window is not ready is skipped for that
tick (sources.NOT_READY) rather than stalling the other stages.

The Board live page is a view of this same feed -- what the board is doing now,
for which lift, how long each stage took -- not a second stream.

Two checks run on every file, on the laptop, while the board works:
  * INTEGRITY. The laptop computes the same file with drivesentinel/dsp.py and
    quantises it; the board's int8 input is compared byte for byte, and the
    verdict on the laptop's input with the board's.
  * FPGA. The bit-accurate model is run on the board's own int8 input; the
    FPGA's logit registers must match it exactly.

If a file cannot be processed on the board (the board stopped answering), that
file's windows are computed on the laptop and sent through the engine router
(FPGA /infer, or software) instead, labelled so; the stream never stalls.

WHAT IS AND IS NOT LIVE. Everything after the raw samples is computed live. The
samples come from stored test-rig data; no sensor is attached to the board. And
the deployed network's verdicts on this data are in-sample (claims_audit 1.28).

Nothing in state() names a data file, a dataset or a specimen.
"""

from __future__ import annotations

import base64
import collections
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Optional

import numpy as np

from drivesentinel import config as C
from drivesentinel.engines import (CLASSES, ENGINE_FPGA, BoardError, SoftwareEngine,
                                   probs_from_logits, process_raw)
from drivesentinel.sources import NOT_READY, Observation

DEMO_DIR = os.path.join(C.ARTIFACT_DIR, "demo")
KEEP_EVENTS = 12
AHEAD = 14                  # windows the board may compute ahead of the stream, per unit
VIEW_CHANNEL = 2            # the vibration envelope, 500-2500 Hz: the pages' bearing view
RAW_VIEW = {"vibration_ms": 20.0, "current_ms": 100.0, "points": 640}


def _preview(x: np.ndarray, fs: float, ms: float, points: int) -> List[float]:
    n = int(round(fs * ms / 1000.0))
    seg = np.asarray(x[:n], dtype=np.float64)
    step = max(1, seg.size // points)
    return [round(float(v), 5) for v in seg[::step][:points]]


def _view(spec: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in np.asarray(spec[VIEW_CHANNEL]).reshape(-1, 2).mean(axis=1)]


def segment_paths(scenario: str, data_root: str = C.DATA_ROOT) -> List[str]:
    """The raw data files behind a bearing unit, in the order its stored windows
    were drawn from them (artifacts/demo/<scenario>.npz `filenames`)."""
    z = np.load(os.path.join(DEMO_DIR, scenario + ".npz"), allow_pickle=False)
    names = list(dict.fromkeys(str(f) for f in z["filenames"]))
    return [os.path.join(data_root, str(z["bearing"]), n) for n in names]


class BoardFeed:
    def __init__(self, router, units: Dict[int, Dict], data_root: str = C.DATA_ROOT):
        """`units`: bearing unit number -> {"scenario", "label"} (label: the lift,
        for the Board live page)."""
        self.router = router
        self.units = dict(units)
        self.data_root = data_root
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sw: Optional[SoftwareEngine] = None
        self.active = False
        self._reset()

    # -- the stream's side ----------------------------------------------------
    def begin(self) -> bool:
        """(Re)start from each unit's first file. True if the board will compute the
        bearing stage; False leaves the stream on its stored path."""
        self.end()
        self._reset()
        address = self.router.address if self.router is not None else ""
        if not address:
            self._set(status="off", reason="No board address is set.")
            return False
        try:
            with urllib.request.urlopen(address + "/health", timeout=5) as r:
                health = json.loads(r.read())
        except Exception:
            self._set(status="off", reason="The board is not answering at %s." % address)
            return False
        fe = health.get("frontend") or {}
        self._set(board={"address": address, "build_id": health.get("build_id"),
                         "python": health.get("python"), "numpy": health.get("numpy"),
                         "scipy": health.get("scipy"), "fft": fe.get("fft"),
                         "workers": fe.get("workers"), "frontend": bool(fe.get("available"))})
        if not fe.get("available"):
            self._set(status="off", reason="The board's signal-processing module is not "
                                           "installed.",
                      detail=fe.get("error") or "the board server predates signal processing")
            return False
        paths = {}
        for n, u in self.units.items():
            try:
                paths[n] = [p for p in segment_paths(u["scenario"], self.data_root)
                            if os.path.exists(p)]
            except Exception:
                paths[n] = []
        if not any(paths.values()):
            self._set(status="off", reason="The raw sensor data is not on this computer.")
            return False
        self._paths = paths
        with self._lock:
            for n in self.units:
                self._s["lifts"][str(n)]["segments"] = len(paths.get(n, []))
                self._done[n] = not paths.get(n)
        self.active = True
        self._stop.clear()
        self._set(status="running")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def end(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=150)
        self._thread = None
        self.active = False

    def claims(self, unit_no: int) -> bool:
        return self.active and unit_no in self.units

    def observations(self, unit_no: int) -> Iterator:
        """This unit's windows as the board returns them; NOT_READY while it works."""
        while True:
            with self._lock:
                q = self._queues[unit_no]
                ob = q.popleft() if q else None
                done = ob is None and self._done[unit_no]
            if ob is not None:
                yield ob
            elif done:
                return
            else:
                yield NOT_READY

    # -- the page's side ------------------------------------------------------
    def state(self) -> Dict:
        with self._lock:
            s = json.loads(json.dumps(self._s))
        for i, e in enumerate(s["events"]):
            if i > 0:                              # only the newest carries signals
                e.pop("raw", None)
                e.pop("views", None)
        return s

    # -- internals ---------------------------------------------------------------
    def _reset(self):
        self._queues = {n: collections.deque() for n in self.units}
        self._done = {n: False for n in self.units}
        self._t = {n: 0.0 for n in self.units}
        self._step = {n: 0 for n in self.units}
        self._paths: Dict[int, List[str]] = {}
        with self._lock:
            self._s = {
                "status": "idle", "reason": None, "detail": None, "board": None,
                "now": None,
                "lifts": {str(n): {"label": u["label"], "segment": 0, "segments": 0}
                          for n, u in self.units.items()},
                "totals": {"segments": 0, "windows": 0, "bytes_sent": 0, "signal_s": 0.0,
                           "board_ms": 0.0, "int8_identical": 0, "bytes_differing": 0,
                           "verdict_same": 0, "fpga_exact": 0, "on_laptop": 0,
                           "on_board_windows": 0},
                "timing": {"realtime_factor": None},
                "events": [], "raw_view": RAW_VIEW,
            }

    def _set(self, **kw):
        with self._lock:
            self._s.update(kw)

    def _run(self):
        try:
            self._loop()
        except Exception as e:                      # never leave the page saying "running"
            self._set(status="error", reason="The bearing feed stopped unexpectedly.",
                      detail="%s: %s" % (type(e).__name__, e))
        finally:
            with self._lock:
                for n in self.units:
                    self._done[n] = True
                if self._s["status"] == "running":
                    self._s["status"] = "done"
                self._s["now"] = None

    def _loop(self):
        from drivesentinel import dataset as DS
        from drivesentinel import dsp as D
        self._sw = self._sw or SoftwareEngine()
        pos = {n: 0 for n in self.units}
        while not self._stop.is_set():
            pending = [n for n in sorted(self.units) if pos[n] < len(self._paths.get(n, []))]
            if not pending:
                return
            worked = False
            for n in pending:
                if self._stop.is_set():
                    return
                with self._lock:
                    if len(self._queues[n]) >= AHEAD:       # the stream has not caught up
                        continue
                path = self._paths[n][pos[n]]
                pos[n] += 1
                worked = True
                label = self.units[n]["label"]
                self._set(now={"lift": label, "segment": pos[n],
                               "segments": len(self._paths[n]), "since_unix": time.time()})
                c1, c2, vib, _ = DS.load_analysis_channels(path)
                self._one(n, label, pos[n], c1, c2, vib, D)
                with self._lock:
                    if pos[n] >= len(self._paths[n]):
                        self._done[n] = True
            if not worked:
                time.sleep(0.1)

    def _one(self, n, label, seg, c1, c2, vib, D):
        sw = self._sw
        ref, ref_diag = D.process_recording(c1, c2, vib)       # the laptop's own reference
        ref_q = [sw.quantise(w) for w in ref]
        address = self.router.address
        error, res, rtt_ms = None, None, None
        try:
            t0 = time.perf_counter()
            res = process_raw(address, c1, c2, vib, C.FS_FAST, timeout_s=120.0)
            rtt_ms = (time.perf_counter() - t0) * 1e3
        except BoardError as e:
            error = "the board answered, but not as the expected core (%s)" % e
        except urllib.error.HTTPError as e:
            try:
                error = json.loads(e.read()).get("error")
            except Exception:
                error = "HTTP %s" % e.code
        except Exception as e:
            error = "the board stopped answering (%s)" % type(e).__name__

        obs, wins, views = [], [], []
        identical = fpga_exact = same = bytes_diff = 0
        if res is not None:                                    # computed on the board
            for k, w in enumerate(res["windows"]):
                q = np.frombuffer(base64.b64decode(w["int8"]), dtype=np.int8).reshape(5, 512)
                if k < len(ref_q):
                    nd = int(np.count_nonzero(q != ref_q[k]))
                    identical += nd == 0
                    bytes_diff += nd
                    same += sw.infer_int8(ref_q[k])["class"] == w["class"]
                fpga_exact += sw.infer_int8(q)["logits"] == w["logits"]
                engine = {"engine": ENGINE_FPGA, "cycles": w["cycles"], "logits": w["logits"],
                          "label": w["label"], "dsp": "board"}
                self.router.record(engine)
                obs.append(self._obs(n, w["logits"], {"spectrum": w["view"]}, engine))
                wins.append({"label": w["label"], "cycles": w["cycles"],
                             "fpga_ms": round(w["fpga_ms"], 2), "rpm": round(w["rpm"], 1)})
                views.append(w["view"])
        else:                                                  # the laptop, and the router
            for k, spec in enumerate(ref):
                r = self.router.infer(spec)
                engine = {"engine": r["engine"], "cycles": r.get("cycles"),
                          "logits": r["logits"], "label": r["label"], "dsp": "laptop"}
                view = _view(spec)
                obs.append(self._obs(n, r["logits"], {"spectrum": view}, engine))
                wins.append({"label": r["label"], "cycles": r.get("cycles"), "fpga_ms": None,
                             "rpm": round(ref_diag[k]["rpm"], 1)})
                views.append(view)

        signal_s = len(c1) / C.FS_FAST
        tm = {k: round(v, 1) for k, v in res["timings"].items()} if res else None
        event = {"lift": label, "segment": seg, "time_unix": time.time(),
                 "where": "board" if res else "laptop", "error": error,
                 "windows": wins, "rpm": wins[0]["rpm"] if wins else None, "timings": tm,
                 "rtt_ms": round(rtt_ms, 1) if rtt_ms else None,
                 "network_ms": (round(max(0.0, rtt_ms - res["timings"]["total_ms"]), 1)
                                if res else None),
                 "bytes_sent": res.get("bytes_sent") if res else None,
                 "identical": identical, "fpga_exact": fpga_exact, "verdict_same": same,
                 "bytes_differing": bytes_diff, "compared": len(wins) if res else 0,
                 "views": views,
                 "raw": {"vibration": _preview(vib, C.FS_FAST, RAW_VIEW["vibration_ms"],
                                               RAW_VIEW["points"]),
                         "current_1": _preview(c1, C.FS_FAST, RAW_VIEW["current_ms"],
                                               RAW_VIEW["points"]),
                         "current_2": _preview(c2, C.FS_FAST, RAW_VIEW["current_ms"],
                                               RAW_VIEW["points"])}}
        with self._lock:
            self._queues[n].extend(obs)
            s = self._s
            s["events"].insert(0, event)
            del s["events"][KEEP_EVENTS:]
            s["lifts"][str(n)]["segment"] = seg
            t = s["totals"]
            t["segments"] += 1
            t["windows"] += len(obs)
            if res:
                t["on_board_windows"] += len(obs)
                t["signal_s"] = round(t["signal_s"] + signal_s, 3)
                t["bytes_sent"] += int(res.get("bytes_sent") or 0)
                t["board_ms"] = round(t["board_ms"] + res["timings"]["total_ms"], 1)
                t["int8_identical"] += identical
                t["bytes_differing"] += bytes_diff
                t["verdict_same"] += same
                t["fpga_exact"] += fpga_exact
                s["timing"]["realtime_factor"] = (round(t["signal_s"] * 1e3 / t["board_ms"], 2)
                                                  if t["board_ms"] else None)
            else:
                t["on_laptop"] += 1

    def _obs(self, n, logits, frame, engine) -> Observation:
        ob = Observation(step=self._step[n], t_s=round(self._t[n], 3), labels=list(CLASSES),
                         probs=probs_from_logits(logits, self._sw.lsb), frame=frame,
                         context={}, deployment_in_sample=True, engine=engine)
        self._step[n] += 1
        self._t[n] += C.HOP_SECONDS
        return ob
