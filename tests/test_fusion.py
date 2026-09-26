"""
Rule-based fusion.

The tests that matter are the ones about AUTHORITY: an indicative branch must be
unable to raise Fault no matter how confident it is. That cap is the reason the
pre-registered floor exists, and it is the thing most likely to be quietly
"fixed" by a future edit.
"""

import json
import os

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import fusion as F

LABELS = ("healthy", "inner_race", "outer_race")


def metric(name="test", stage="S5", f1=0.90, groups=29, measured=True):
    return F.BranchMetric(name=name, stage=stage, metric_value=f1,
                          protocol="leave-one-group-out",
                          n_validation_groups=groups, measured=measured)


def state(f1=0.90, groups=29, measured=True, **cfg):
    c = dict(C.FUSION_CONFIG)
    c.update(cfg)
    return F.BranchState(metric=metric(f1=f1, groups=groups, measured=measured),
                         labels=LABELS, cfg=c)


def feed(st, p_fault, n):
    """n updates at a given fault probability."""
    out = None
    for _ in range(n):
        out = st.update([1.0 - p_fault, p_fault, 0.0])
    return out


# ---------------------------------------------------------------------------
# authority
# ---------------------------------------------------------------------------

def test_floor_is_the_pre_registered_one():
    fa = C.FUSION_CONFIG["fault_authority"]
    assert fa["min_validation_groups"] == 3
    assert fa["min_macro_f1"] == 0.75
    assert fa["fixed_at"] == "2026-09-18T07:19:23Z"


def test_branch_meeting_both_tests_has_authority():
    assert metric(f1=0.80, groups=29).fault_authority is True


def test_too_few_groups_fails_however_good_the_metric():
    assert metric(f1=0.99, groups=2).fault_authority is False


def test_too_low_a_metric_fails_however_many_groups():
    assert metric(f1=0.74, groups=100).fault_authority is False


def test_unmeasured_branch_has_no_authority():
    assert metric(measured=False).fault_authority is False
    assert metric(measured=False).tier == "NOT MEASURED"


def test_tier_names():
    assert metric(f1=0.80, groups=29).tier == "Fault-capable"
    assert metric(f1=0.62, groups=2).tier == "INDICATIVE"


# ---------------------------------------------------------------------------
# threshold crossing and the K-consecutive rule
# ---------------------------------------------------------------------------

def test_one_confident_window_does_not_raise_anything():
    st = state()
    v = feed(st, 0.99, 1)
    assert v["status"] == "Normal"


def test_k_consecutive_updates_raise_fault():
    K = C.FUSION_CONFIG["k_consecutive"]
    st = state()
    assert feed(st, 0.99, K - 1)["status"] != "Fault"
    assert feed(st, 0.99, 1)["status"] == "Fault"


def test_a_contrary_sample_restarts_the_run():
    K = C.FUSION_CONFIG["k_consecutive"]
    st = state()
    feed(st, 0.99, K - 1)
    feed(st, 0.0, 1)                      # breaks the run
    assert st.status != "Fault"


def test_warning_needs_k_updates_too():
    K = C.FUSION_CONFIG["k_consecutive"]
    st = state()
    assert feed(st, 0.5, K - 1)["status"] == "Normal"
    assert feed(st, 0.5, 1)["status"] == "Warning"


def test_below_warning_threshold_stays_normal():
    assert feed(state(), 0.1, 20)["status"] == "Normal"


# ---------------------------------------------------------------------------
# hysteresis, both directions
# ---------------------------------------------------------------------------

def test_clearing_requires_dropping_below_tau_w_minus_h():
    cfg = C.FUSION_CONFIG
    K = cfg["k_consecutive"]
    st = state()
    feed(st, 0.99, K)
    assert st.status == "Fault"
    # Just under tau_W but inside the hysteresis band: must NOT clear.
    feed(st, cfg["tau_warning"] - cfg["hysteresis"] / 2, 20)
    assert st.status == "Fault"


def test_clearing_happens_below_the_hysteresis_band():
    cfg = C.FUSION_CONFIG
    N, K = cfg["rolling_window"], cfg["k_consecutive"]
    st = state()
    feed(st, 0.99, K)
    assert st.status == "Fault"
    # N updates to flush the rolling window, then K below the band to clear.
    feed(st, cfg["tau_warning"] - cfg["hysteresis"] - 0.05, N + K)
    assert st.status == "Normal"


def test_clearing_is_slower_than_raising_and_that_is_deliberate():
    """
    p_fault is a ROLLING MEAN, so a branch that has just raised Fault carries
    several high samples in its window. Clearing therefore needs enough quiet
    updates to flush those out, and only then K consecutive below the band.
    Raising takes K updates; clearing takes roughly N + K.

    Asymmetric on purpose: a drive that briefly looks healthy mid-fault should
    not clear the alarm.
    """
    cfg = C.FUSION_CONFIG
    N, K = cfg["rolling_window"], cfg["k_consecutive"]
    st = state()
    feed(st, 0.99, K)
    assert st.status == "Fault"

    quiet = cfg["tau_warning"] - cfg["hysteresis"] - 0.05
    feed(st, quiet, K + 1)
    assert st.status == "Fault", "cleared before the window could flush"

    feed(st, quiet, N)
    assert st.status == "Normal"


def test_hysteresis_prevents_flapping():
    """Oscillating either side of tau_W must not toggle the status."""
    cfg = C.FUSION_CONFIG
    st = state()
    feed(st, 0.99, cfg["k_consecutive"])
    for _ in range(10):
        st.update([1 - (cfg["tau_warning"] + 0.01), cfg["tau_warning"] + 0.01, 0.0])
        st.update([1 - (cfg["tau_warning"] - 0.01), cfg["tau_warning"] - 0.01, 0.0])
    assert st.status == "Fault"


# ---------------------------------------------------------------------------
# the low-confidence cap -- the most important test in this file
# ---------------------------------------------------------------------------

def test_indicative_branch_cannot_raise_fault_at_p_099():
    """
    An INDICATIVE branch pinned at p_fault = 0.99 for many updates must reach
    Warning and stop. If this ever passes as Fault, the floor has been defeated.
    """
    st = state(f1=0.62, groups=2)                 # the measured B-S4 winding case
    v = feed(st, 0.99, 50)
    assert st.metric.tier == "INDICATIVE"
    assert v["status"] == "Warning"
    assert v["status"] != "Fault"


def test_unmeasured_branch_cannot_raise_fault_either():
    st = state(measured=False)
    v = feed(st, 1.0, 50)
    assert v["status"] == "Warning"


def test_indicative_branch_still_reports_its_evidence():
    """Capped, not silenced -- the panel stays live."""
    st = state(f1=0.62, groups=2)
    feed(st, 0.99, 10)
    ev = st.evidence()
    assert "WARNING" in ev
    assert "LOW CONFIDENCE BRANCH" in ev


def test_capable_branch_evidence_has_no_low_confidence_tag():
    st = state(f1=0.80, groups=29)
    feed(st, 0.99, 10)
    assert "LOW CONFIDENCE BRANCH" not in st.evidence()


# ---------------------------------------------------------------------------
# rolling window and outputs
# ---------------------------------------------------------------------------

def test_rolling_window_is_bounded():
    N = C.FUSION_CONFIG["rolling_window"]
    st = state()
    v = feed(st, 0.9, N + 25)
    assert v["n_obs"] == N
    assert v["n_obs_total"] == N + 25


def test_rolling_mean_smooths_a_single_spike():
    st = state()
    feed(st, 0.0, 9)
    v = st.update([0.0, 1.0, 0.0])
    assert v["p_fault"] == pytest.approx(0.1, abs=1e-9)


def test_top_class_and_confidence_come_from_the_rolling_mean():
    st = state()
    for _ in range(5):
        st.update([0.1, 0.2, 0.7])
    v = st.update([0.1, 0.2, 0.7])
    assert v["top_class"] == "outer_race"
    assert v["confidence"] == pytest.approx(0.7, abs=1e-6)


def test_wrong_probability_length_raises():
    with pytest.raises(ValueError, match="expected 3 probabilities"):
        state().update([0.5, 0.5])


def test_reset_clears_state():
    st = state()
    feed(st, 0.99, 10)
    st.reset()
    assert st.status == "Normal" and len(st.window) == 0


# ---------------------------------------------------------------------------
# worst-stage aggregation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("statuses,expect", [
    (["Normal", "Normal"], "Normal"),
    (["Normal", "Warning"], "Warning"),
    (["Warning", "Fault"], "Fault"),
    (["Fault", "Normal", "Warning"], "Fault"),
    ([], "Normal"),
])
def test_worst_status_wins(statuses, expect):
    assert F.worst(statuses) == expect


def test_system_status_names_the_driving_branch():
    views = [
        {"branch": "bearing", "stage": "S5", "status": "Fault"},
        {"branch": "winding", "stage": "S4", "status": "Warning"},
    ]
    s = F.system_status(views)
    assert s["status"] == "Fault"
    assert s["driver"] == "bearing" and s["driver_stage"] == "S5"


def test_system_status_on_no_branches():
    assert F.system_status([])["status"] == "Normal"


def test_an_indicative_branch_cannot_drive_the_system_to_fault():
    """End-to-end: the cap must survive aggregation."""
    ind = state(f1=0.62, groups=2)
    cap = state(f1=0.80, groups=29)
    v_ind = feed(ind, 0.99, 20)
    v_cap = feed(cap, 0.0, 20)
    assert F.system_status([v_ind, v_cap])["status"] == "Warning"


# ---------------------------------------------------------------------------
# reading authority from disk
# ---------------------------------------------------------------------------

def test_metrics_load_from_the_real_artifacts():
    m = F.load_branch_metrics()
    assert set(m) == {"bearing", "winding", "supply", "inverter_telemetry"}


def test_bearing_has_authority_as_measured():
    b = F.load_branch_metrics()["bearing"]
    if not b.measured:
        pytest.skip("lobo_summary.json absent")
    assert b.n_validation_groups == 29
    assert b.fault_authority is True


def test_winding_is_indicative_as_measured():
    w = F.load_branch_metrics()["winding"]
    if not w.measured:
        pytest.skip("winding_results.json absent")
    assert w.n_validation_groups == 2
    assert w.fault_authority is False
    assert w.tier == "INDICATIVE"


def test_missing_json_degrades_to_not_measured(tmp_path):
    """A branch with no results file must not crash the dashboard."""
    m = F._multistage_metric("supply", "S1", art_dir=str(tmp_path))
    assert m.measured is False
    assert m.tier == "NOT MEASURED"
    assert m.fault_authority is False
    assert "absent" in m.note


def test_missing_bearing_json_degrades_too(tmp_path):
    m = F._bearing_metric(run_dir=str(tmp_path))
    assert m.measured is False and m.tier == "NOT MEASURED"


def test_authority_table_is_generated_not_typed():
    rows = {r["branch"]: r for r in F.authority_table(F.load_branch_metrics())}
    assert set(rows) == {"bearing", "winding", "supply", "inverter_telemetry"}
    if rows["winding"]["measured"]:
        assert rows["winding"]["groups_test"] is False
        assert rows["winding"]["metric_test"] is False


def test_evidence_string_has_the_spec_fields():
    st = state(f1=0.80, groups=29)
    feed(st, 0.9, 10)
    ev = st.evidence("negative-sequence up")
    for token in ("S5", "pooled obs", "leave-one-group-out", "negative-sequence up"):
        assert token in ev


def test_evidence_on_an_unmeasured_branch_says_so():
    st = state(measured=False)
    feed(st, 0.5, 5)
    assert "NOT MEASURED" in st.evidence()
