"""
The technician app's state: units, their fusion timelines, the alert feed.

Everything here is computed by code the engineering dashboard also uses:

  fusion      drivesentinel.fusion.BranchState / load_branch_metrics
  tiers       fusion.BranchMetric.tier -> alerts.TIER_LABEL
  alerts      drivesentinel.alerts (AlertEmitter, Outbox)
  data        drivesentinel.sources.RecordedScenarioSource (panels.load_scenario,
              panels.assert_out_of_sample underneath)
  metrics     the results JSON under artifacts/, read at start-up

so the two apps cannot show different numbers: they read them through the same
functions. tests/test_webapp.py checks it.

HOW ALERTS REACH THE FEED
-------------------------
Not by a function call. A background "fleet" replay walks every unit, runs each
observation through the fusion state and the AlertEmitter, and hands each alert
to an `alerts.Outbox`, which POSTs the 21-byte packet to this app's own
/api/alerts/ingest endpoint -- the same HTTP path a drive in the field would
use. If the endpoint is not answering (the replay can start before the server
has bound its port), the packet waits in the on-disk outbox and is resent;
tests/test_webapp.py drives that path explicitly with the endpoint down.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from drivesentinel import alerts as A
from drivesentinel import config as C
from drivesentinel import fusion as FU
from drivesentinel.sources import NOT_READY, DataSource, RecordedScenarioSource

ROOT = C.PROJECT_ROOT
VAR = os.path.join(ROOT, "webapp", "var")

# Stages the app shows, in power-flow order. S2 (DC link) and S3 (inverter)
# share one branch, so they are one card, as in dashboard/panels.py STAGES.
APP_STAGES = [("S1", "supply"), ("S3", "inverter_telemetry"), ("S4", "winding"),
              ("S5", "bearing")]

# One unit per stage makes the composite demo lift. Composite, and labelled so:
# no dataset shares a machine across stages (dashboard: COMPOSITE REPLAY).
COMPOSITE = {"S1": "supply_phase_loss_FILE2", "S3": "inverter_F2_open_circuit",
             "S4": "winding_severity_ramp_1000W", "S5": "bearing_KA04"}


def engine_view(res: Optional[Dict]) -> Optional[Dict]:
    """Which engine produced a bearing verdict, for the pages: its label, the FPGA's
    cycle count, and where the signal processing ran."""
    if not res:
        return None
    return {"engine": res.get("engine"), "cycles": res.get("cycles"),
            "dsp": res.get("dsp", "laptop")}


def jsonable(o):
    """numpy -> plain Python, recursively, for JSON responses."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ===========================================================================
# plain-language test descriptions, built from the results JSON
# ===========================================================================

def _json(*parts):
    p = os.path.join(C.ARTIFACT_DIR, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def how_tested(branch: str, m: FU.BranchMetric) -> str:
    if not m.measured:
        return "No test result is available for this stage."
    f1 = f"macro-F1 {m.metric_value:.4f}"
    if branch == "bearing":
        n = m.n_validation_groups
        return (f"Tested on bearings it had never seen: the model was trained on "
                f"{n - 1} bearings and tested on the one left out, and this was "
                f"repeated for each of the {n} bearings. Result: {f1}.")
    if branch == "supply":
        r = _json("multistage", "supply", "supply_results.json") or {}
        n = r.get("R1_rule_per_recording", {}).get("scores", {}).get("n")
        return (f"A fixed rule with no settings learned from data, checked on {n} "
                f"lab recordings from {m.n_validation_groups} motors, one motor at "
                f"a time. Result: {f1}.")
    if branch == "inverter_telemetry":
        return ("Trained on the first 70 % of each lab recording and tested on the "
                "last 30 %. There is one recording per fault condition, so no "
                f"separate unit exists to test on. Result: {f1}.")
    if branch == "winding":
        return (f"Trained on lab recordings from some measurement sessions and "
                f"tested on a session it had not seen, for each of "
                f"{m.n_validation_groups} sessions. Result: {f1}.")
    return f1


# ===========================================================================
# the service
# ===========================================================================

class Service:
    def __init__(self, source: Optional[DataSource] = None,
                 endpoint: Optional[str] = None, pace_s: float = 0.25,
                 spool: Optional[str] = None):
        self.source = source or RecordedScenarioSource()
        self.metrics = FU.load_branch_metrics()
        self.units = {u.unit_no: u for u in self.source.units()}
        self.pace_s = pace_s
        self.timelines = {n: self._timeline(u) for n, u in self.units.items()}
        self.store: Dict[str, Dict] = {}
        self._store_lock = threading.Lock()
        self.live_step: Dict[int, int] = {n: -1 for n in self.units}
        # What the stream actually computed, step by step: the pages show these, so a
        # bearing verdict on screen is the one the engine (the FPGA, when the board is
        # connected) returned -- not the precomputed timeline, which stays behind for
        # stage metadata and for any alert that arrived from outside the stream.
        self.live_records: Dict[int, List[Dict]] = {n: [] for n in self.units}
        # wall-clock time of each unit's latest observation; display only
        self.live_unix: Dict[int, Optional[float]] = {n: None for n in self.units}
        self.fleet_done = False
        self.fleet_started_unix: Optional[float] = None
        os.makedirs(VAR, exist_ok=True)
        self.outbox = (A.Outbox(spool or os.path.join(VAR, "outbox.jsonl"), endpoint)
                       if endpoint else None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- stage tiers ------------------------------------------------------
    def stage_rows(self) -> List[Dict]:
        rows = []
        for st, branch in APP_STAGES:
            m = self.metrics.get(branch)
            tier = m.tier if m else "NOT MEASURED"
            rows.append({"stage": st, "branch": branch,
                         "name": A.STAGE_NAME[st],
                         "tier_label": A.TIER_LABEL[tier],
                         "alarm_capable": bool(m and m.fault_authority)})
        return rows

    def technical(self, branch: str) -> Dict:
        m = self.metrics[branch]
        return {"tier_label": A.TIER_LABEL[m.tier],
                "metric_name": m.metric_name,
                "metric_value": m.metric_value,
                "protocol": m.protocol,
                "validation_groups": m.n_validation_groups,
                "groups_needed": C.FUSION_CONFIG["fault_authority"]["min_validation_groups"],
                "metric_needed": C.FUSION_CONFIG["fault_authority"]["min_macro_f1"],
                "how_tested": how_tested(branch, m),
                "source": (os.path.relpath(m.source, ROOT).replace(os.sep, "/")
                           if m.source else None)}

    # -- timelines: the fusion replay of one unit, precomputed ------------
    def _timeline(self, u) -> Dict:
        m = self.metrics[u.branch]
        st = None
        em = A.AlertEmitter(u.unit_no)
        steps = []
        # The precomputed timeline backs the Why? pages. A source with an engine
        # (BoardSource) supplies the same verdicts from its software engine --
        # bit-identical to the FPGA -- so an alert and its explanation always
        # agree, and start-up makes no board calls. The live stream below goes
        # through self.source, engine and all.
        obs = getattr(self.source, "timeline_observations", self.source.observations)
        for ob in obs(u.unit_no):
            if ob.probs is None:
                steps.append({"t": ob.t_s, "status": "NO DATA", "frame": ob.frame})
                continue
            if st is None:
                st = FU.BranchState(metric=m, labels=ob.labels)
            v = st.update(ob.probs)
            a = em.observe(v, ob.t_s, ob.context)
            steps.append({
                "t": round(ob.t_s, 3), "status": A.STATUS_OF[v["status"]],
                "p_fault": round(v["p_fault"], 4), "top_class": v["top_class"],
                "confidence": round(v["confidence"], 4), "n_obs": v["n_obs"],
                "frame": ob.frame, "context": ob.context,
                "alert": a.render() if a else None,
                "alert_seq": a.seq if a else None})
        return {"unit": u.unit_id, "unit_no": u.unit_no, "scenario": u.scenario,
                "branch": u.branch, "stage": u.stage, "title": u.title,
                "dataset": u.dataset, "meta": u.meta, "steps": steps}

    def unit_summary(self, n: int) -> Dict:
        u = self.units[n]
        tl = self.timelines[n]
        i = self.live_step[n]
        cur = tl["steps"][i] if 0 <= i < len(tl["steps"]) else None
        return {"unit_no": n, "unit": u.unit_id, "title": u.title,
                "stage": u.stage, "stage_name": A.STAGE_NAME[u.stage],
                "branch": u.branch, "dataset": u.dataset, "scenario": u.scenario,
                "n_steps": len(tl["steps"]), "live_step": i,
                "live_status": (cur or {}).get("status", "NORMAL" if u.has_predictions
                                                  else "NO DATA")}

    # -- the fleet replay -------------------------------------------------
    def start_fleet(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.fleet_done = False
        self.live_step = {n: -1 for n in self.units}
        self.live_unix = {n: None for n in self.units}
        self.live_records = {n: [] for n in self.units}
        self._thread = threading.Thread(target=self._run_fleet, daemon=True)
        self._thread.start()

    def restart_fleet(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._store_lock:
            self.store.clear()
        self.start_fleet()

    def _run_fleet(self) -> None:
        self.fleet_started_unix = time.time()
        begin = getattr(self.source, "begin_stream", None)
        if begin is not None:
            begin()                     # e.g. the board takes the bearing stage from raw data
        emitters = {n: A.AlertEmitter(n, sink=self._deliver) for n in self.units}
        states: Dict[int, Optional[FU.BranchState]] = {n: None for n in self.units}
        # lazy, so an engine behind the source is called as each step is reached
        iters = {n: iter(self.source.observations(n)) for n in self.units}
        active = set(self.units)
        while active and not self._stop.is_set():
            for n in sorted(active):
                ob = next(iters[n], None)
                if ob is None:
                    active.discard(n)
                    continue
                if ob is NOT_READY:           # the board is still computing this unit's next window
                    continue
                rec = {"t": round(ob.t_s, 3), "status": "NO DATA", "frame": ob.frame,
                       "context": ob.context, "engine": engine_view(ob.engine)}
                if ob.probs is not None:
                    u = self.units[n]
                    if states[n] is None:
                        states[n] = FU.BranchState(metric=self.metrics[u.branch],
                                                   labels=ob.labels)
                    v = states[n].update(ob.probs)
                    a = emitters[n].observe(v, ob.t_s, ob.context)
                    rec.update(status=A.STATUS_OF[v["status"]],
                               p_fault=round(v["p_fault"], 4), top_class=v["top_class"],
                               confidence=round(v["confidence"], 4), n_obs=v["n_obs"],
                               alert=a.render() if a else None,
                               alert_seq=a.seq if a else None)
                self.live_records[n].append(rec)
                self.live_step[n] += 1
                self.live_unix[n] = time.time()
            if self.outbox and self.outbox.pending():
                self.outbox.flush()
            time.sleep(self.pace_s)
        # the endpoint may have been down for the last few: keep trying
        t_end = time.time() + 30
        while self.outbox and self.outbox.pending() and time.time() < t_end:
            self.outbox.flush()
            time.sleep(1.0)
        self.fleet_done = True

    def _deliver(self, alert: A.Alert) -> None:
        if self.outbox:
            self.outbox.send(alert)
        else:
            self.ingest(alert.pack())

    # -- the receiving end ---------------------------------------------------
    def ingest(self, packet: bytes) -> Dict:
        a = A.Alert.unpack(packet)
        key = f"{a.unit}-{a.seq}"
        with self._store_lock:
            if key in self.store:                    # a resend: already have it
                return {"id": key, "duplicate": True}
            rec = a.render()
            rec.update({"id": key, "unit_no": a.unit, "fault": a.fault,
                        "branch": a.branch, "packet_hex": packet.hex(),
                        "packet_bytes": len(packet),
                        "received_unix": time.time()})
            self.store[key] = rec
        return {"id": key, "duplicate": False}

    def feed(self, status: Optional[str] = None, stage: Optional[str] = None,
             unit: Optional[int] = None) -> List[Dict]:
        with self._store_lock:
            rows = list(self.store.values())
        if status:
            rows = [r for r in rows if r["status"] == status]
        if stage:
            rows = [r for r in rows if r["stage"] == stage]
        if unit:
            rows = [r for r in rows if r["unit_no"] == unit]
        return sorted(rows, key=lambda r: (r["received_unix"], r["seq"]), reverse=True)

    def alert_detail(self, key: str) -> Optional[Dict]:
        with self._store_lock:
            rec = self.store.get(key)
        if rec is None:
            return None
        n = rec["unit_no"]
        tl = self.timelines[n]
        # The evidence the stream itself computed; the precomputed timeline only for
        # an alert the stream did not produce (a packet ingested from elsewhere).
        steps = self.live_records.get(n) or []
        idx = next((i for i, s in enumerate(steps) if s.get("alert_seq") == rec["seq"]), None)
        if idx is None:
            steps = tl["steps"]
            idx = next((i for i, s in enumerate(steps)
                        if s.get("alert_seq") == rec["seq"]), None)
        lo = max(0, (idx or 0) - 40)
        hi = min(len(steps), (idx or 0) + 41)
        window = [{"t": s["t"], "status": s["status"],
                   "p_fault": s.get("p_fault"), "confidence": s.get("confidence")}
                  for s in steps[lo:hi]]
        trig = steps[idx] if idx is not None else None
        return {"alert": rec, "unit": self.unit_summary(n),
                "notes": self._notes(rec, tl),
                "meta": tl["meta"], "trigger": trig, "trigger_index": idx,
                "trace": window, "technical": self.technical(rec["branch"]),
                "fusion": {"rolling_window": C.FUSION_CONFIG["rolling_window"],
                           "k_consecutive": C.FUSION_CONFIG["k_consecutive"],
                           "tau_fault": C.FUSION_CONFIG["tau_fault"],
                           "tau_warning": C.FUSION_CONFIG["tau_warning"]}}

    def _notes(self, rec: Dict, tl: Dict) -> List[Dict]:
        """Plain-language facts behind this alert, each read from a file."""
        out = []
        b = rec["branch"]
        rc = tl["meta"].get("recorded_condition")
        if rc:
            called = rec["fault"] or "normal"
            out.append({"title": "Recorded condition",
                        "text": (f"The dataset records this recording as "
                                 f"'{rc.replace('_', ' ')}'. The stage called it "
                                 f"'{called.replace('_', ' ')}'"
                                 + ("." if called == rc or rec["status"] == "NORMAL"
                                    else " — a wrong call, shown as it happened."))})
        if b == "winding":
            chk = _json("webapp", "winding_severity_check.json")
            if chk:
                out.append({"title": "Why there is no severity band",
                            "text": (f"The signal this stage measures does not track "
                                     f"how severe the short is: the largest rank "
                                     f"correlation with recorded severity, within any "
                                     f"one motor, is {chk['max_abs_rho']:.2f}. A band "
                                     f"would be a guess, so none is shown."),
                            "source": "artifacts/webapp/winding_severity_check.json"})
            out.append({"title": "Why no phase is named",
                        "text": ("Every winding fault in the dataset is on the same "
                                 "phase, so the stage could never learn to tell "
                                 "phases apart.")})
        if b == "inverter_telemetry":
            out.append({"title": "Why the location is only suspected",
                        "text": ("The stage recognises the kind of fault from two "
                                 "output phase currents and the module temperature. "
                                 "It does not resolve which switch leg failed.")})
        if b == "supply":
            out.append({"title": "How the phase is found",
                        "text": ("The lost phase is the one whose current collapsed "
                                 "below the rule's threshold — measured directly, "
                                 "not inferred. Running or stalled is measured from "
                                 "vibration.")})
        return out

    def outbox_state(self) -> Dict:
        if not self.outbox:
            return {"mode": "direct", "pending": 0}
        return {"mode": "http", "endpoint": self.outbox.endpoint,
                "pending": self.outbox.pending(),
                "delivered": self.outbox.delivered,
                "failed_attempts": self.outbox.failed_attempts,
                "packet_bytes": A.packet_bytes(),
                "raw_window_bytes": A.raw_window_bytes()}
