"""
Alerts: what a drive tells a technician, built from fusion state.

WHAT AN ALERT IS
----------------
The fusion layer (`fusion.BranchState`) already decides each stage's status --
Normal, Warning or Fault -- with its hysteresis and its pre-registered authority
floor. This module does not decide anything. It:

  1. maps that status to the technician's words, per stage tier;
  2. fires ONLY when the status changes, so the feed is not a stream;
  3. packs the alert into a fixed 21-byte packet -- codes, not text -- so
     "a few bytes instead of the raw data" is a measured fact, not a slogan;
  4. delivers it by HTTP POST and, when the endpoint is unreachable, keeps it
     on disk and resends it when the endpoint returns.

THE RULE THAT MUST NEVER BREAK
------------------------------
An ADVISORY stage never produces an ALARM. `fusion.BranchState` already caps an
INDICATIVE branch at Warning; `build_alert` checks it again and raises rather
than emit an ALARM from a branch without Fault authority. Two independent
guards on the one property the whole tier system exists for.

WHAT THE TEXT MAY SAY
---------------------
Only what the stage actually determines:

  power supply   which phase collapsed (L1/L2/L3) -- measured directly from the
                 per-phase current -- and whether the motor was turning,
                 measured by the rule from vibration.
  inverter       the fault family, and "Location: inverter power stage". The
                 switch leg is not named: the branch classifies the family from
                 two phase currents and does not resolve the leg.
  motor winding  "Winding anomaly detected" -- no fault TYPE, no phase, no
                 severity. The type was dropped from the words (user decision,
                 2026-09-21) because the held-out type call is crossed on both
                 demo units: the inter-turn ramp is called inter-coil and the
                 inter-coil ramp inter-turn. The fault CODE is still carried in
                 the packet and the engineering view; only the text drops it.
                 No phase: every fault in the dataset is on the same phase. No
                 severity: the signal does not track it
                 (artifacts/webapp/winding_severity_check.json).
  bearing        inner or outer race, with the fusion confidence.

A field a stage does not determine is left EMPTY rather than filled with a
sentence saying so; the pages omit empty fields. That is the rule for inverter
and winding "how bad".
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

# ===========================================================================
# vocabulary -- user-facing, and the only place it is defined
# ===========================================================================

STATUS_OF = {"Fault": "ALARM", "Warning": "ADVISORY", "Normal": "NORMAL"}
STATUS_ORDER = ("NORMAL", "ADVISORY", "ALARM")         # NO DATA is not a status change

TIER_LABEL = {"Fault-capable": "ALARM-READY", "INDICATIVE": "ADVISORY",
              "NOT MEASURED": "NO DATA"}

ALERT_TYPE = {"ALARM": "ALARM — schedule inspection",
              "ADVISORY": "ADVISORY — early sign, check at next visit",
              "NORMAL": "Back to normal"}

STAGE_NAME = {"S1": "Power supply", "S3": "Inverter", "S4": "Motor winding",
              "S5": "Bearing"}

BRANCH_STAGE = {"supply": "S1", "inverter_telemetry": "S3", "winding": "S4",
                "bearing": "S5"}

PHASE_NAME = {0: "L1", 1: "L2", 2: "L3"}

# Fault codes are one byte on the wire. 0 means "no fault" (a NORMAL alert).
FAULT_CODES = ["", "phase_loss_running", "single_phasing_start",
               "open_circuit", "short_circuit", "over_temp",
               "inter_turn", "inter_coil", "inner_race", "outer_race",
               "undetermined"]
FAULT_CODE = {f: i for i, f in enumerate(FAULT_CODES)}

WHAT = {
    "phase_loss_running": "Phase lost while running",
    "single_phasing_start": "Started with a phase missing",
    "open_circuit": "Open circuit in the inverter",
    "short_circuit": "Short circuit in the inverter",
    "over_temp": "Inverter overheating",
    "inter_turn": "Winding anomaly detected",           # type deliberately not named
    "inter_coil": "Winding anomaly detected",
    "inner_race": "Inner-race bearing damage",
    "outer_race": "Outer-race bearing damage",
    "undetermined": "Anomaly detected",
}

WINDING_HEADLINE = "Winding anomaly detected"


def what_text(branch: str, fault: str) -> str:
    """The alert's headline. A winding alert never names a fault type."""
    if branch == "winding":
        return WINDING_HEADLINE
    return WHAT.get(fault, WHAT["undetermined"])


ACTION = {
    "phase_loss_running": ("Check the supply fuses, contactor and terminals on the "
                           "lost phase. Do not restart until all three phases are "
                           "present."),
    "single_phasing_start": ("Check the incoming supply and the contactor on the "
                             "missing phase before any further start attempt."),
    "open_circuit": ("Inspect the inverter switches and gate drive for a device "
                     "that has failed open."),
    "short_circuit": ("Isolate the drive and inspect the inverter switches for a "
                      "short before re-energising."),
    "over_temp": ("Check inverter cooling: fan, heatsink, airflow and cabinet "
                  "temperature."),
    "inter_turn": ("Plan a winding resistance-balance and insulation test at the "
                   "next visit."),
    "inter_coil": ("Plan a winding resistance-balance and insulation test at the "
                   "next visit."),
    "inner_race": ("Schedule a bearing inspection and confirm with a vibration "
                   "measurement."),
    "outer_race": ("Schedule a bearing inspection and confirm with a vibration "
                   "measurement."),
    "undetermined": "Review the stage's signals at the next visit.",
}


def where_text(branch: str, fault: str, phase: int = -1) -> str:
    if branch == "supply":
        return (f"Phase {PHASE_NAME[phase]} (collapsed current, measured directly)"
                if phase in PHASE_NAME else "Power supply input")
    if branch == "inverter_telemetry":
        return "Location: inverter power stage"
    if branch == "winding":
        return "Motor stator winding"
    if branch == "bearing":
        return "Motor bearing"
    return "—"


def how_bad_text(branch: str, fault: str, confidence: float, rotating: int = -1) -> str:
    if branch == "supply":
        if fault == "single_phasing_start":
            return "Motor stalled — it could not start on the remaining phases"
        if fault == "phase_loss_running":
            return "Motor still running on the remaining phases"
        return "—"
    if branch in ("winding", "inverter_telemetry"):
        return ""                        # not determined by the stage: omitted
    if branch == "bearing":
        return f"Confidence {confidence * 100:.0f} %"
    return "—"


# ===========================================================================
# the alert
# ===========================================================================

WIRE_FORMAT = "<BHBBBBBbIII"
WIRE_VERSION = 1
WIRE_BYTES = struct.calcsize(WIRE_FORMAT)      # 21


@dataclass
class Alert:
    """One status change of one stage on one unit. Everything else is derived."""
    unit: int                 # unit number; the app shows it as DEMO-<nn>
    stage: str                # S1 / S3 / S4 / S5
    status: str               # ALARM / ADVISORY / NORMAL
    fault: str                # a FAULT_CODES entry; "" for NORMAL
    confidence: float         # fusion rolling-mean confidence, 0..1
    n_obs: int                # observations pooled in the rolling window
    phase: int = -1           # supply only: 0/1/2 = L1/L2/L3, -1 = n/a
    seq: int = 0              # per-unit sequence number, for idempotent resend
    t_data_ms: int = 0        # milliseconds into the recording
    t_unix: int = 0           # wall clock when the alert was built

    # -- the few bytes -----------------------------------------------------
    def pack(self) -> bytes:
        return struct.pack(
            WIRE_FORMAT, WIRE_VERSION, self.unit,
            int(self.stage[1]), STATUS_ORDER.index(self.status),
            FAULT_CODE[self.fault], min(100, int(round(self.confidence * 100))),
            min(255, self.n_obs), self.phase, self.seq, self.t_data_ms, self.t_unix)

    @staticmethod
    def unpack(b: bytes) -> "Alert":
        v, unit, st, status, fault, conf, n, phase, seq, tms, tunix = \
            struct.unpack(WIRE_FORMAT, b)
        if v != WIRE_VERSION:
            raise ValueError(f"alert wire version {v}, expected {WIRE_VERSION}")
        return Alert(unit=unit, stage=f"S{st}", status=STATUS_ORDER[status],
                     fault=FAULT_CODES[fault], confidence=conf / 100.0, n_obs=n,
                     phase=phase, seq=seq, t_data_ms=tms, t_unix=tunix)

    # -- the words ---------------------------------------------------------
    @property
    def branch(self) -> str:
        return next(b for b, s in BRANCH_STAGE.items() if s == self.stage)

    def render(self) -> Dict:
        """The technician-facing text. Deterministic from the packed fields."""
        f = self.fault
        return {
            "unit": f"DEMO-{self.unit:02d}",
            "stage": self.stage,
            "stage_name": STAGE_NAME[self.stage],
            "status": self.status,
            "alert_type": ALERT_TYPE[self.status],
            "what": what_text(self.branch, f) if f else "Back to normal",
            "where": where_text(self.branch, f, self.phase) if f else "—",
            "how_bad": how_bad_text(self.branch, f, self.confidence) if f else "—",
            "action": ACTION.get(f, "No action needed.") if f else "No action needed.",
            "confidence": round(self.confidence, 2),
            "n_obs": self.n_obs,
            "seq": self.seq,
            "t_data_s": self.t_data_ms / 1000.0,
            "t_unix": self.t_unix,
        }


def build_alert(unit: int, view: Dict, seq: int, t_data_s: float,
                context: Optional[Dict] = None, now: Optional[float] = None) -> Alert:
    """
    One alert from one fusion view (`fusion.BranchState.update()` output).

    Raises if an INDICATIVE branch would produce an ALARM. `BranchState` already
    prevents it; this is the second, independent guard.
    """
    context = context or {}
    status = STATUS_OF[view["status"]]
    if status == "ALARM" and not view.get("fault_authority", False):
        raise AssertionError(
            f"{view['branch']}: a branch without Fault authority produced ALARM. "
            f"An ADVISORY stage must never alarm.")
    fault = ""
    if status != "NORMAL":
        fault = str(view["top_class"])
        if context.get("undetermined") or fault not in FAULT_CODE:
            fault = "undetermined"
        if fault in ("healthy", "normal"):
            fault = "undetermined"
    phase = int(context.get("phase", -1)) if view["branch"] == "supply" and fault else -1
    return Alert(unit=unit, stage=BRANCH_STAGE[view["branch"]], status=status,
                 fault=fault, confidence=float(view["confidence"]),
                 n_obs=int(view["n_obs"]), phase=phase, seq=seq,
                 t_data_ms=int(round(t_data_s * 1000)),
                 t_unix=int(now if now is not None else time.time()))


class AlertEmitter:
    """
    Emits an alert only when a stage's user-facing status CHANGES.

    The fusion hysteresis (k consecutive updates to raise or clear, a clearing
    threshold below the raising one) already stops single noisy windows from
    flipping the status. This class adds nothing to that logic; it only refuses
    to repeat a status that has not changed, so the feed shows transitions.
    """

    def __init__(self, unit: int, sink: Optional[Callable[[Alert], None]] = None):
        self.unit = unit
        self.sink = sink
        self.last: Dict[str, str] = {}
        self.seq = 0

    def observe(self, view: Dict, t_data_s: float,
                context: Optional[Dict] = None) -> Optional[Alert]:
        status = STATUS_OF[view["status"]]
        prev = self.last.get(view["branch"], "NORMAL")
        if status == prev:
            return None
        self.last[view["branch"]] = status
        self.seq += 1
        a = build_alert(self.unit, view, self.seq, t_data_s, context)
        if self.sink:
            self.sink(a)
        return a


# ===========================================================================
# delivery: POST, and a disk outbox that survives the endpoint being down
# ===========================================================================

class Outbox:
    """
    Store-and-forward for alert packets.

    `send()` appends the packet to an on-disk spool FIRST, then tries to flush.
    `flush()` POSTs queued packets oldest-first and removes each only after the
    endpoint answers 2xx. If the endpoint is down the packets stay on disk --
    across a process restart, too -- and the next `flush()` resends them.

    The receiver deduplicates on (unit, seq), so a packet resent after a lost
    acknowledgement is not counted twice. That is what makes "network drops,
    nothing is lost" true without also making it "network drops, some alerts
    arrive twice".
    """

    def __init__(self, spool_path: str, endpoint: str, timeout_s: float = 2.0):
        self.path = spool_path
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(spool_path)), exist_ok=True)
        if not os.path.exists(spool_path):
            open(spool_path, "w").close()
        self.delivered = 0
        self.failed_attempts = 0

    def _read(self) -> List[str]:
        with open(self.path) as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    def _write(self, lines: List[str]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("".join(l + "\n" for l in lines))
        os.replace(tmp, self.path)                   # atomic on the same volume

    def pending(self) -> int:
        with self._lock:
            return len(self._read())

    def send(self, alert: Alert) -> bool:
        with self._lock:
            with open(self.path, "a") as fh:
                fh.write(alert.pack().hex() + "\n")
        return self.flush()

    def flush(self) -> bool:
        """True when the spool is empty afterwards."""
        with self._lock:
            lines = self._read()
            sent = 0
            for hx in lines:
                req = urllib.request.Request(
                    self.endpoint, data=bytes.fromhex(hx), method="POST",
                    headers={"Content-Type": "application/octet-stream"})
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                        if not 200 <= r.status < 300:
                            break
                except (urllib.error.URLError, ConnectionError, OSError):
                    self.failed_attempts += 1
                    break
                sent += 1
            if sent:
                self._write(lines[sent:])
                self.delivered += sent
            return sent == len(lines)


def packet_bytes() -> int:
    """The alert's size on the wire. Documented, and bounded by a test."""
    return WIRE_BYTES


def raw_window_bytes() -> int:
    """
    What the alert replaces, for the bearing stage: one conditioned model input
    window, N_INPUT_CHANNELS x N_ORDER_BINS float32 (config.py). The raw
    waveform behind it is larger still.
    """
    from . import config as C
    return int(C.N_INPUT_CHANNELS * C.N_ORDER_BINS * 4)
