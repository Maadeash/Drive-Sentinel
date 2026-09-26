"""
The fleet view: what the technician pages are allowed to see.

`service.Service` holds the units (one monitored drive stage each), their
precomputed fusion timelines and the alert store. This module is the only code
that turns those into page data, and it does two jobs.

1. UNITS BECOME LIFTS
   Each lift is a named drive with up to four monitored stages. `LIFTS` assigns
   every unit to exactly one stage slot of one lift, and `Fleet.__init__` asserts
   it: each unit used once, each in a slot of its own stage. So the fleet shows
   every pipeline output exactly once -- nothing duplicated to look busier,
   nothing dropped to look cleaner. A slot with no unit shows NO DATA; it is
   never filled with a guess.

   The lift names, sites and drive tags are fictional. The stages of one lift
   come from different public test rigs, because no public dataset covers a
   whole drive (the engineering dashboard's COMPOSITE REPLAY). That, and the
   data's recorded origin, are stated in webapp/README.md and
   docs/claims_audit.md 1.23; the engineering view keeps the internal names.

2. INTERNAL FIELDS STAY INTERNAL
   Nothing below returns a file path, unit or scenario ID, fault code, top class,
   data-time offset, dataset name, recorded condition or protocol code. Numbers
   pass through unchanged from the service, which reads them from the results
   JSON. tests/test_webapp.py scans every response from here for the banned
   vocabulary and checks every number against its source.

Statuses and alerts are NOT decided here. They come from `fusion.BranchState`
and `alerts.AlertEmitter` via the service, so the ADVISORY-never-ALARM guards
apply unchanged.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from drivesentinel import alerts as A
from drivesentinel import config as C

# Stage slugs are what URLs and page code use; stage codes stay internal.
STAGE_SLUG = {"S1": "supply", "S3": "inverter", "S4": "winding", "S5": "bearing"}
SLUG_STAGE = {v: k for k, v in STAGE_SLUG.items()}
SLUG_BRANCH = {"supply": "supply", "inverter": "inverter_telemetry",
               "winding": "winding", "bearing": "bearing"}
STAGE_ORDER = ["supply", "inverter", "winding", "bearing"]      # power-flow order

STATUS_RANK = {"ALARM": 3, "ADVISORY": 2, "NORMAL": 1, "NO DATA": 0}

# The fleet. Unit keys are the manifest ids (internal; never sent to a page).
LIFTS: List[Dict] = [
    {"slug": "lift-07", "name": "Lift 07", "location": "Tower B",
     "site": "Northbank Plaza", "drive": "Drive D-0712",
     "units": {"supply": "supply_phase_loss_FILE2",
               "inverter": "inverter_F2_open_circuit",
               "winding": "winding_severity_ramp_1000W",
               "bearing": "bearing_KA04"}},
    {"slug": "lift-03", "name": "Lift 03", "location": "Tower A",
     "site": "Northbank Plaza", "drive": "Drive D-0305",
     "units": {"supply": "supply_phase_loss_FILE5",
               "winding": "winding_inter_coil_ramp_1000W",
               "bearing": "bearing_KI18"}},
    {"slug": "lift-12", "name": "Lift 12", "location": "Tower B",
     "site": "Northbank Plaza", "drive": "Drive D-1204",
     "units": {"inverter": "inverter_F3_short_circuit",
               "bearing": "bearing_K001"}},
    {"slug": "lift-01", "name": "Lift 01", "location": "Core 1",
     "site": "Riverside House", "drive": "Drive D-0109",
     "units": {"inverter": "inverter_F6_over_temp",
               "bearing": "bearing_KI05"}},
]

# The signal each stage shows, and which frame fields a page needs for it.
SIGNAL_FRAME_KEYS = {"bearing": ("spectrum",), "supply": ("rms",),
                     "inverter": ("imbalance",), "winding": ("neg_seq_ratio",)}
SIGNAL_META_KEYS = {
    "bearing": ("x_max_order", "n_bins", "markers", "view", "x_label"),
    "supply": ("series", "series_t", "view", "y_label", "x_label"),
    "inverter": ("series", "series_t", "normal_band", "view", "y_label", "x_label"),
    "winding": ("series", "view", "y_label", "x_label"),
}

RECENT_STEPS = 60


def lift_label(lift: Dict) -> str:
    return f"{lift['name']} · {lift['location']}"


def worst(statuses) -> str:
    s = [x for x in statuses if x]
    return max(s, key=lambda x: STATUS_RANK[x]) if s else "NO DATA"


class Fleet:
    def __init__(self, svc):
        self.svc = svc
        by_scenario = {u.scenario: n for n, u in svc.units.items()}
        self.unit_of: Dict[Tuple[str, str], int] = {}
        self.place_of: Dict[int, Tuple[Dict, str]] = {}
        for lift in LIFTS:
            for slug, scn in lift["units"].items():
                n = by_scenario.get(scn)
                if n is None:                       # a unit this source does not have
                    continue
                u = svc.units[n]
                assert STAGE_SLUG[u.stage] == slug, f"{scn} placed in the {slug} slot"
                assert n not in self.place_of, f"{scn} assigned to two lifts"
                self.unit_of[(lift["slug"], slug)] = n
                self.place_of[n] = (lift, slug)
        self.lifts = {l["slug"]: l for l in LIFTS}

    def unplaced_units(self) -> List[int]:
        """Units the fleet does not show. Asserted empty for the real source."""
        return sorted(set(self.svc.units) - set(self.place_of))

    # -- ratings -------------------------------------------------------------
    def rating(self, slug: str) -> str:
        m = self.svc.metrics.get(SLUG_BRANCH[slug])
        return A.TIER_LABEL[m.tier if m else "NOT MEASURED"]

    def stages_meta(self) -> List[Dict]:
        return [{"slug": s, "name": A.STAGE_NAME[SLUG_STAGE[s]], "rating": self.rating(s)}
                for s in STAGE_ORDER]

    # -- one stage slot, now -------------------------------------------------
    def _live(self, n: int) -> List[Dict]:
        """What the stream has computed for unit n so far (service.live_records)."""
        return self.svc.live_records.get(n) or []

    def _current(self, n: int) -> Tuple[int, Optional[Dict]]:
        steps = self._live(n)
        return len(steps) - 1, (steps[-1] if steps else None)

    def _slot(self, lift: Dict, slug: str, with_signal: bool) -> Dict:
        row = {"slug": slug, "name": A.STAGE_NAME[SLUG_STAGE[slug]],
               "rating": self.rating(slug), "status": "NO DATA",
               "monitored": False, "last_unix": None}
        n = self.unit_of.get((lift["slug"], slug))
        if n is None:
            return row
        i, cur = self._current(n)
        row.update(monitored=True, last_unix=self.svc.live_unix.get(n),
                   status=(cur or {}).get("status", "NO DATA"))
        if with_signal:
            steps = self._live(n)
            lo = max(0, i - RECENT_STEPS)
            t0 = steps[0]["t"] if steps else 0.0
            row["signal"] = self._signal_meta(slug, self.svc.timelines[n]["meta"])
            row["recent"] = [{"t": round(s["t"] - t0, 3), "p_fault": s.get("p_fault"),
                              "frame": self._frame(slug, s.get("frame"))}
                             for s in steps[lo:i + 1]] if i >= 0 else []
        return row

    @staticmethod
    def _frame(slug: str, frame: Optional[Dict]) -> Optional[Dict]:
        if not frame:
            return None
        return {k: frame[k] for k in SIGNAL_FRAME_KEYS[slug] if k in frame}

    @staticmethod
    def _signal_meta(slug: str, meta: Dict) -> Dict:
        return {k: meta[k] for k in SIGNAL_META_KEYS[slug] if k in meta}

    # -- the fleet page ------------------------------------------------------
    def overview(self) -> Dict:
        lifts = []
        for lift in LIFTS:
            slots = [self._slot(lift, s, with_signal=False) for s in STAGE_ORDER]
            status = worst([s["status"] for s in slots])
            events = self.alerts(lift_slug=lift["slug"])
            last = [s["last_unix"] for s in slots if s["last_unix"]]
            lifts.append({"slug": lift["slug"], "name": lift["name"],
                          "label": lift_label(lift), "location": lift["location"],
                          "site": lift["site"], "drive": lift["drive"],
                          "status": status, "stages": slots,
                          "latest_alert": events[0] if events else None,
                          "last_unix": max(last) if last else None})
        lifts.sort(key=lambda l: (-STATUS_RANK[l["status"]], l["name"]))
        active = [s for l in lifts for s in l["stages"]]
        return {"lifts": lifts,
                "summary": {"lifts": len(lifts),
                            "stages_monitored": sum(s["monitored"] for s in active),
                            "alarm": sum(s["status"] == "ALARM" for s in active),
                            "advisory": sum(s["status"] == "ADVISORY" for s in active),
                            "normal": sum(s["status"] == "NORMAL" for s in active)},
                "streaming": not self.svc.fleet_done}

    def lift(self, slug: str) -> Optional[Dict]:
        lift = self.lifts.get(slug)
        if lift is None:
            return None
        slots = [self._slot(lift, s, with_signal=True) for s in STAGE_ORDER]
        return {"slug": slug, "name": lift["name"], "label": lift_label(lift),
                "location": lift["location"], "site": lift["site"],
                "drive": lift["drive"], "status": worst([s["status"] for s in slots]),
                "stages": slots, "events": self.alerts(lift_slug=slug)[:20],
                "streaming": not self.svc.fleet_done}

    # -- alerts ----------------------------------------------------------------
    def _ref(self, rec: Dict) -> Optional[str]:
        place = self.place_of.get(rec["unit_no"])
        if place is None:
            return None
        lift, slug = place
        return f"{lift['slug']}-{slug}-{rec['seq']}"

    def _key(self, ref: str) -> Optional[str]:
        m = re.fullmatch(r"(lift-\d+)-([a-z]+)-(\d+)", ref or "")
        if not m:
            return None
        n = self.unit_of.get((m.group(1), m.group(2)))
        return None if n is None else f"{n}-{int(m.group(3))}"

    def alert_view(self, rec: Dict) -> Optional[Dict]:
        ref = self._ref(rec)
        if ref is None:
            return None
        lift, slug = self.place_of[rec["unit_no"]]
        blank = lambda v: None if v in (None, "", "—") else v
        return {"id": ref, "lift": lift_label(lift), "lift_slug": lift["slug"],
                "site": lift["site"], "drive": lift["drive"],
                "stage": slug, "stage_name": rec["stage_name"],
                "status": rec["status"], "alert_type": rec["alert_type"],
                "headline": rec["what"], "where": blank(rec["where"]),
                "how_bad": blank(rec["how_bad"]), "action": rec["action"],
                "confidence": rec["confidence"], "n_obs": rec["n_obs"],
                "time_unix": rec["received_unix"]}

    def alerts(self, status: Optional[str] = None, stage_slug: Optional[str] = None,
               lift_slug: Optional[str] = None) -> List[Dict]:
        stage = SLUG_STAGE.get(stage_slug) if stage_slug else None
        out = []
        for rec in self.svc.feed(status or None, stage):
            v = self.alert_view(rec)
            if v and (not lift_slug or v["lift_slug"] == lift_slug):
                out.append(v)
        return out

    def alert_detail(self, ref: str) -> Optional[Dict]:
        key = self._key(ref)
        d = self.svc.alert_detail(key) if key else None
        if d is None:
            return None
        rec = d["alert"]
        lift, slug = self.place_of[rec["unit_no"]]
        tl = self.svc.timelines[rec["unit_no"]]
        trig = d["trigger"] or {}
        return {"alert": self.alert_view(rec),
                "signal": {"kind": slug, **self._signal_meta(slug, tl["meta"])},
                "trigger": {"t": trig.get("t"), "frame": self._frame(slug, trig.get("frame"))},
                "trace": [{"t": s["t"], "p_fault": s["p_fault"]} for s in d["trace"]],
                "fusion": {"rolling_window": d["fusion"]["rolling_window"],
                           "k_consecutive": d["fusion"]["k_consecutive"]},
                "technical": self.technical(slug, rec)}

    def technical(self, slug: str, rec: Optional[Dict] = None) -> Dict:
        """Collapsed on the Why? page. Neutral wording; numbers from the metrics."""
        branch = SLUG_BRANCH[slug]
        m = self.svc.metrics[branch]
        fa = C.FUSION_CONFIG["fault_authority"]
        out = {"rating": self.rating(slug),
               "score_name": m.metric_name, "score": m.metric_value,
               "validation_groups": m.n_validation_groups,
               "method": validation_method(branch, m),
               "alarm_ready_groups": fa["min_validation_groups"],
               "alarm_ready_score": fa["min_macro_f1"]}
        if rec is not None:
            out.update(packet_bytes=rec["packet_bytes"], packet_hex=rec["packet_hex"])
        return out

    # -- live monitor ----------------------------------------------------------
    def monitor_options(self) -> List[Dict]:
        return [{"slug": l["slug"], "label": lift_label(l), "site": l["site"],
                 "drive": l["drive"],
                 "stages": [{"slug": s, "name": A.STAGE_NAME[SLUG_STAGE[s]],
                             "rating": self.rating(s)}
                            for s in STAGE_ORDER if (l["slug"], s) in self.unit_of]}
                for l in LIFTS]

    def monitor(self, lift_slug: str, slug: str, since: int = 0) -> Optional[Dict]:
        """The steps the stream has computed for one lift's stage, from `since` on --
        so the page polls for what is new instead of replaying a precomputed copy."""
        n = self.unit_of.get((lift_slug, slug))
        if n is None:
            return None
        lift = self.lifts[lift_slug]
        tl = self.svc.timelines[n]
        live = self._live(n)
        t0 = live[0]["t"] if live else 0.0
        steps = []
        for s in live[since:]:
            a = None
            if s.get("alert"):
                stored = self.svc.store.get(f"{n}-{s['alert_seq']}")
                rec = dict(s["alert"], unit_no=n,
                           received_unix=(stored or {}).get("received_unix"),
                           packet_bytes=None, packet_hex=None)
                a = self.alert_view(rec)
            steps.append({"t": round(s["t"] - t0, 3), "status": s["status"],
                          "p_fault": s.get("p_fault"),
                          "frame": self._frame(slug, s.get("frame")), "alert": a,
                          "engine": s.get("engine")})
        return {"lift": lift_label(lift), "lift_slug": lift_slug, "site": lift["site"],
                "drive": lift["drive"], "stage": slug,
                "stage_name": A.STAGE_NAME[SLUG_STAGE[slug]], "rating": self.rating(slug),
                "signal": {"kind": slug, **self._signal_meta(slug, tl["meta"])},
                "since": since, "n_steps": len(live),
                "streaming": not self.svc.fleet_done,
                "last_unix": self.svc.live_unix.get(n), "steps": steps}


def validation_method(branch: str, m) -> str:
    """How a stage was validated, in plain words. Every number is read, not typed."""
    if not m.measured:
        return "—"
    n = m.n_validation_groups
    if branch == "bearing":
        return (f"Evaluated on bearings withheld from training: each of the {n} "
                f"bearings was held out in turn while the model trained on the "
                f"other {n - 1}.")
    if branch == "supply":
        from .content import _json
        r = _json("artifacts", "multistage", "supply", "supply_results.json") or {}
        runs = r.get("R1_rule_per_recording", {}).get("scores", {}).get("n")
        return (f"A fixed threshold rule with no learned parameters, evaluated motor "
                f"by motor on {runs} test runs from {n} motors.")
    if branch == "inverter_telemetry":
        from .content import _json
        r = _json("artifacts", "multistage", "inverter_telemetry",
                  "inverter_telemetry_results.json") or {}
        split = re.search(r"(\d+)/(\d+)", r.get("V1_block_split", {}).get("protocol", ""))
        if split:
            return (f"Trained on the first {split.group(1)} % of each test run and "
                    f"evaluated on the final {split.group(2)} %, with a gap between "
                    f"the two.")
        return "Trained and evaluated on separate, time-ordered blocks of each test run."
    if branch == "winding":
        return (f"Evaluated on measurement sessions withheld from training, one "
                f"session at a time across {n} sessions.")
    return "—"
