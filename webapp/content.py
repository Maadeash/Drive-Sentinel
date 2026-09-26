"""
Page content for Hardware, About, Settings and the Engineering view.

Every figure comes from a file under artifacts/ or a constant in the code that
produced it, read through the helpers the dashboard and the results renderer
already use. Nothing numeric is typed here. Where a sentence is fixed text, it
contains no number.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Optional

from drivesentinel import config as C
from drivesentinel import fusion as FU

ROOT = C.PROJECT_ROOT


def _json(*parts) -> Optional[Dict]:
    p = os.path.join(ROOT, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _script(rel: str):
    p = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Hardware
# ===========================================================================

def fixed_cycles() -> Optional[int]:
    """Cycles per verdict, from the committed verification JSON. The same in
    every run that reports it (a test asserts one value)."""
    runs = (_json("artifacts", "rtl", "verify.json") or {}).get("runs", {})
    for k in ("v1", "v3", "v5"):
        if runs.get(k, {}).get("cycles_max"):
            return runs[k]["cycles_max"]
    return None


def board_evidence() -> Dict:
    """
    What has run on the board, from the board's own record -- empty unless
    board/board_run.json exists, says ran_on_hardware, and did not come from the
    dry-run mock (the rule tests/test_board_package.py applies). `served` adds the
    board server's check against the software model, when that record exists
    and found no mismatch (artifacts/webapp/board_server_check.json).
    """
    run = _json("board", "board_run.json")
    if not (run and run.get("ran_on_hardware") is True
            and not str(run.get("pynq_version", "")).startswith("MOCK")):
        return {}
    wall = run.get("timing", {}).get("dma_plus_inference_wall", {})
    ev = {"board": str(run.get("board", "")).split(" (")[0],
          "build_id": run["build_id"], "build_id_match": run["build_id_match"],
          "windows": run["windows"], "class_agree": run["class_agree_with_golden"],
          "logits_exact": run["logit_registers_exact"],
          "cycles_equal": run["cycles_equal_simulation"],
          "median_ms": wall.get("median_ms"), "p95_ms": wall.get("p95_ms")}
    chk = _json("artifacts", "webapp", "board_server_check.json")
    if chk and not chk.get("mismatches"):
        ev["served"] = {"windows": chk["windows"], "class_agree": chk["class_agree"],
                        "logits_exact": chk["logits_exact"],
                        "round_trip_median_ms": chk["http_round_trip_ms"]["median"]}
    return ev


def hardware() -> Dict:
    rr = _script("scripts/rtl/render_results.py")        # the results renderer's parsers
    art = os.path.join(ROOT, "artifacts", "rtl")
    core_j = _json("artifacts", "rtl", "synth", "synth.json") or {}
    sys_j = _json("artifacts", "rtl", "system", "system.json") or {}
    core_u = rr.parse_util(os.path.join(art, "synth", "post_route_util.rpt"))
    sys_u = rr.parse_util(os.path.join(art, "system", "post_route_util.rpt"))
    verify = _json("artifacts", "rtl", "verify.json") or {}
    params = _json("artifacts", "rtl", "rtl_params.json") or {}
    runs = verify.get("runs", {})
    cycles = fixed_cycles()

    v1 = runs.get("v1")
    v1_live = None
    try:                                  # the simulator's own per-window reports,
        rv = _script("scripts/rtl/run_verify.py")    # read-only, only present locally
        v1_live = rv.snapshot_v1()
    except Exception:
        v1_live = None

    def resources(u, j):
        if not u:
            return None
        return {"lut": u.get("slice_luts"), "lut_available": u.get("slice_luts_available"),
                "ff": u.get("slice_registers"), "ff_available": u.get("slice_registers_available"),
                "bram": u.get("bram_tiles"), "bram_available": u.get("bram_tiles_available"),
                "dsp": u.get("dsps"), "dsp_available": u.get("dsps_available"),
                "wns_ns": j.get("wns_ns"), "timing_met": j.get("timing_met")}

    def v(k):
        r = runs.get(k) or {}
        return r

    return {
        "part": core_j.get("part"),
        "vivado": core_j.get("vivado"),
        "weights": params.get("w_depth"), "biases": params.get("b_depth"),
        "macs": params.get("total_macs"),
        "cycles": cycles,
        "latency_us_100mhz": (cycles * 10.0 / 1000.0) if cycles else None,
        "achieved_mhz_core": core_j.get("achieved_mhz"),
        "core_only": resources(core_u, core_j),
        "full_system": resources(sys_u, sys_j),
        "bitstream_written": sys_j.get("bitstream_written"),
        "ran_on_hardware": bool(board_evidence()),
        "board": board_evidence() or None,
        "verification": {
            "v2_logit_lsb": (v1 or v("v3")).get("max_abs_logit_diff_lsb"),
            "v3": {k: v("v3").get(k) for k in ("acc_compared", "acc_mismatches",
                                                "windows_run", "complete")},
            "v4": {k: v("v4").get(k) for k in ("vectors", "exact_ties",
                                                "q_mismatches", "pass")},
            "v5": {k: v("v5").get(k) for k in ("windows_run", "class_mismatches",
                                                "logit_mismatches", "complete")},
            "v6": {"build_id": params.get("expected_build_id_hex"),
                   "match": v("v3").get("build_id_match")},
        },
        "v1_committed": ({k: v1.get(k) for k in ("windows_run", "windows_expected",
                                                   "class_mismatches", "logit_mismatches",
                                                   "chunks_complete", "shards",
                                                   "snapshot", "complete")}
                         if v1 else None),
        "v1_live": ({k: v1_live.get(k) for k in ("windows_run", "windows_expected",
                                                  "class_mismatches", "logit_mismatches",
                                                  "chunks_complete", "shards", "complete")}
                    if v1_live else None),
    }


# ===========================================================================
# About
# ===========================================================================

ROADMAP = ("Advisory stages are promoted to alarm-ready through calibration on fleet "
           "data, with no change to the models.")

STAGE_DESCRIPTION = {
    "supply": ("Watches the three supply phases for a lost or missing phase, and "
               "tells a motor still running from one that has stalled."),
    "inverter": ("Recognises open-circuit, short-circuit and overheating faults in the "
                 "inverter power stage from its output currents and module "
                 "temperature."),
    "winding": ("Detects early winding anomalies from the balance of the three motor "
                "currents."),
    "bearing": ("Detects outer-race and inner-race bearing damage from motor vibration "
                "and current."),
}


def about(metrics: Dict[str, FU.BranchMetric]) -> Dict:
    from drivesentinel import alerts as A
    cycles = fixed_cycles()
    params = _json("artifacts", "rtl", "rtl_params.json") or {}
    stages = []
    for slug, branch, code in (("supply", "supply", "S1"),
                               ("inverter", "inverter_telemetry", "S3"),
                               ("winding", "winding", "S4"),
                               ("bearing", "bearing", "S5")):
        m = metrics.get(branch)
        stages.append({"slug": slug, "name": A.STAGE_NAME[code],
                       "rating": A.TIER_LABEL[m.tier if m else "NOT MEASURED"],
                       "description": STAGE_DESCRIPTION[slug]})
    return {
        "overview": [
            "DriveSentinel monitors four stages of an elevator drive: power supply, "
            "inverter, motor winding and bearing. It raises an alert when a stage's "
            "status changes.",
            "Each alert states the finding, the location, the severity where the "
            "stage measures it, and one action.",
        ],
        "stages": stages,
        "alert_levels": [
            {"status": "ALARM", "label": A.ALERT_TYPE["ALARM"],
             "meaning": "A confirmed fault sign. Schedule an inspection now.",
             "raised_by": "Alarm-ready stages"},
            {"status": "ADVISORY", "label": A.ALERT_TYPE["ADVISORY"],
             "meaning": "An early sign. Check it at the next planned visit.",
             "raised_by": "Advisory stages"},
            {"status": "NORMAL", "label": A.ALERT_TYPE["NORMAL"],
             "meaning": "The stage has returned to normal.",
             "raised_by": "Every stage"},
        ],
        "evidence": {"rolling_window": C.FUSION_CONFIG["rolling_window"],
                     "k_consecutive": C.FUSION_CONFIG["k_consecutive"]},
        "edge": {"part_family": "Zynq-7020", "cycles": cycles,
                 "latency_ms_100mhz": cycles * 10.0 / 1e6 if cycles else None,
                 "weights": params.get("w_depth"),
                 "packet_bytes": A.packet_bytes(),
                 "window_bytes": A.raw_window_bytes()},
        "roadmap": ROADMAP,
    }


# ===========================================================================
# Settings
# ===========================================================================

def settings(svc) -> Dict:
    from drivesentinel import alerts as A
    from drivesentinel.engines import ENGINE_FPGA
    ob = svc.outbox_state()
    # The PYNQ-Z2 is "Connected" when it is serving as the bearing engine right now.
    # It is not the data source -- the drive stream still is -- so `active` stays False.
    router = getattr(svc.source, "router", None)
    board_live = bool(router is not None and router.active == ENGINE_FPGA)
    return {
        "sources": [
            {"key": "stream", "name": "Drive stream", "state": "Active", "active": True,
             "description": ("Streams drive signals through the detection, fusion and "
                             "alert pipeline.")},
            {"key": "pynq", "name": "Drive interface — PYNQ-Z2",
             "state": "Connected · bearing engine" if board_live else "Ready for connection",
             "active": False, "live": board_live,
             "description": ("Connects the on-drive FPGA board through the same "
                             "data-source interface. Every page, rule and alert "
                             "stays the same.")},
        ],
        "stream": {"streaming": not svc.fleet_done,
                   "started_unix": svc.fleet_started_unix},
        "delivery": {"packet_bytes": A.packet_bytes(),
                     "window_bytes": A.raw_window_bytes(),
                     "delivered": ob.get("delivered"),
                     "pending": ob.get("pending"),
                     "store_and_forward": ob.get("mode") == "http"},
    }


# ===========================================================================
# Engineering view -- the existing dashboard's panels, unchanged
# ===========================================================================

def engineering(metrics: Dict[str, FU.BranchMetric]) -> Dict:
    from dashboard import panels as P
    lobo = _json("artifacts", "runs", "lobo_summary.json") or {}
    ki05 = next((f for f in lobo.get("folds", []) if f["test_bearing"] == "KI05"), None)
    tp = P.trip_panel()
    return {
        "stage_rows": P.stage_rows(metrics),
        "metric_cards": P.metric_cards(metrics),
        "authority": FU.authority_table(metrics),
        "floor": C.FUSION_CONFIG["fault_authority"],
        "fusion": {k: C.FUSION_CONFIG[k] for k in ("rolling_window", "tau_fault",
                                                   "tau_warning", "hysteresis",
                                                   "k_consecutive")},
        "trip": {k: tp[k] for k in ("enabled", "caption", "bpfo", "bpfi", "gap",
                                    "at_900", "at_20", "window_s",
                                    "needed_window_at_20")},
        "resolution": [
            {"shaft_rpm": r["shaft_rpm"], "revs": r["shaft_revs_in_window"],
             "order_resolution": r["order_resolution"],
             "source": ("Paderborn, measured" if r["shaft_rpm"] in (900.0, 1500.0)
                        else "EXTRAPOLATION")}
            for r in tp["resolution"]],
        "ki05": ki05,
        "lobo_pooled": lobo.get("pooled"),
    }
