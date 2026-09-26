"""
B-S1 supply threshold rule.

The rule is the deliverable, so the tests are about the rule's behaviour, not a
model's accuracy. The two that matter most:

  * the threshold's real sensitivity, measured rather than asserted: the
    RECORDING-LEVEL verdict holds across 0.03-0.40, while the window-level safe
    band is only (0.048, 0.053). An earlier draft claimed the wider figure for
    both, and that was wrong;
  * a de-energised machine is not a faulted one -- the ratio test is blind to
    "all three phases off" and would otherwise score FILE 1's first 8 seconds as
    normal running.
"""

import os

import numpy as np
import pytest

from drivesentinel.adapters import thomas_motor as T
from drivesentinel.branches import supply as S
from drivesentinel.common.schema import Recording

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_ext", "thomas_motor")
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "FILE 1.mat")),
    reason="D4 not present (data_ext/ is gitignored)")

FS = 5_000.0          # decimated for the synthetic tests; the rule is rate-agnostic


def synth(amps=(0.5, 0.5, 0.5), vib=0.03, dur=4.0, fs=FS, drop_at=None,
          drop_phase=1, drop_to=0.012):
    """Three phase currents at 50 Hz, optionally dropping one phase part-way."""
    t = np.arange(int(fs * dur)) / fs
    sig = {}
    for k, (name, a) in enumerate(zip(T.CURRENTS, amps)):
        env = np.full_like(t, a)
        if drop_at is not None and k == drop_phase:
            env[t >= drop_at] = drop_to
        sig[name] = (env * np.sqrt(2) * np.cos(2 * np.pi * 50 * t - k * 2 * np.pi / 3)
                     ).astype(np.float32)
    sig["vib_x"] = (vib * np.sqrt(2) *
                    np.random.default_rng(0).standard_normal(t.size)).astype(np.float32)
    return Recording(source="thomas_motor", stage="S1", signals=sig,
                     fs={k: fs for k in sig}, label="normal", group="motor_A_healthy",
                     condition={"motor": "healthy", "scenario": "synthetic"},
                     provenance={"source_file": "synthetic"})


# ---------------------------------------------------------------------------
# the rule's basic behaviour
# ---------------------------------------------------------------------------

def test_balanced_running_machine_is_normal():
    v = S.classify(synth())
    assert {w.state for w in v} == {"normal"}


def test_a_dropped_phase_is_detected():
    v = S.classify(synth(drop_at=2.0))
    assert any(w.state == "phase_loss_running" for w in v)


def test_the_right_phase_is_named():
    v = S.classify(synth(drop_at=2.0, drop_phase=2))
    lost = {p for w in v for p in w.lost_phases}
    assert lost == {2}


def test_detection_happens_after_the_drop_not_before():
    v = S.classify(synth(drop_at=2.0))
    first = min(w.t for w in v if w.state != "normal")
    assert first >= 2.0 - S.RULE["window_s"]


def test_a_de_energised_machine_is_off_not_normal():
    """
    All three phases at the dead level. The ratio test cannot fire -- each phase
    is compared against a median that is equally dead -- so without the absolute
    floor this would be scored as a healthy running machine.
    """
    v = S.classify(synth(amps=(0.012, 0.012, 0.012)))
    assert {w.state for w in v} == {"off"}


def test_off_is_checked_before_the_ratio_test():
    v = S.classify(synth(amps=(0.012, 0.012, 0.012), drop_at=1.0, drop_to=0.0001))
    assert all(w.state == "off" for w in v)


# ---------------------------------------------------------------------------
# threshold sensitivity -- measured, and narrower than first claimed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.05, 0.10, 0.20, 0.30])
def test_window_verdicts_hold_across_the_measured_safe_range(frac):
    """
    The synthetic drop is 0.012 against 0.5, i.e. a ratio of 0.024, so a
    threshold below that correctly does NOT fire. Only thresholds above the
    real worst-case lost-phase ratio (0.0476, measured on FILE 10) are tested
    here -- see test_recording_verdict_is_robust_across_thresholds for the
    property that actually matters.
    """
    v = S.classify(synth(drop_at=2.0), cfg={"lost_fraction": frac})
    assert any(w.state == "phase_loss_running" for w in v)
    assert any(w.state == "normal" for w in v)


@pytest.mark.parametrize("frac", [0.01, 0.02])
def test_a_threshold_below_the_real_ratio_misses_the_event(frac):
    """
    Documents the lower bound rather than hiding it. A dead phase is not
    infinitely far from a live one: measured ratios during real events run
    0.008-0.048, so a 1 % threshold misses events. The shipped 0.05 sits above
    all of them.
    """
    v = S.classify(synth(drop_at=2.0), cfg={"lost_fraction": frac})
    assert not any(w.state == "phase_loss_running" for w in v)


def test_a_mild_imbalance_is_not_a_phase_loss():
    """20 % low is unbalance, not loss. The rule must not cry wolf."""
    v = S.classify(synth(amps=(0.5, 0.4, 0.5)))
    assert {w.state for w in v} == {"normal"}


# ---------------------------------------------------------------------------
# rotation separates the two fault classes
# ---------------------------------------------------------------------------

def test_rotating_phase_loss_is_phase_loss_running():
    v = S.classify(synth(drop_at=2.0, vib=0.03))
    assert any(w.state == "phase_loss_running" for w in v)
    assert not any(w.state == "single_phasing_start" for w in v)


def test_stalled_phase_loss_is_single_phasing_start():
    """Measured: stalled vib_x RMS 0.0008-0.0012, rotating 0.017-0.041."""
    v = S.classify(synth(drop_at=2.0, vib=0.001))
    assert any(w.state == "single_phasing_start" for w in v)
    assert not any(w.state == "phase_loss_running" for w in v)


def test_the_two_classes_differ_only_by_rotation():
    """Same currents, different vibration -> different class. That is the design."""
    a = S.recording_verdict(S.classify(synth(drop_at=2.0, vib=0.03)))
    b = S.recording_verdict(S.classify(synth(drop_at=2.0, vib=0.001)))
    assert a == "phase_loss_running"
    assert b == "single_phasing_start"


# ---------------------------------------------------------------------------
# events and latency
# ---------------------------------------------------------------------------

def test_events_are_contiguous_runs():
    ev = S.events(S.classify(synth(drop_at=2.0)))
    assert len(ev) == 1
    assert ev[0]["t0"] >= 1.8
    assert ev[0]["duration_s"] > 1.0


def test_a_single_glitch_is_not_an_event():
    v = S.classify(synth())
    v = list(v)
    v[3] = S.WindowVerdict(v[3].t, "phase_loss_running", (1,), v[3].rms, True, 0.03)
    assert S.events(v, min_duration_s=0.5) == []


def test_latency_is_reported_and_bounded_by_the_hop():
    lat = S.detection_latency(S.classify(synth(drop_at=2.0)))
    assert lat is not None
    assert lat["latency_s"] >= 0.0
    assert lat["latency_s"] < 1.0


def test_no_latency_without_an_event():
    assert S.detection_latency(S.classify(synth())) is None


def test_recording_verdict_ignores_off_windows():
    """A machine switched off for part of a capture is not 'off' as a diagnosis."""
    assert S.recording_verdict(S.classify(synth(amps=(0.012, 0.012, 0.012)))) == "normal"


# ---------------------------------------------------------------------------
# integration -- the real ten files
# ---------------------------------------------------------------------------

@needs_data
def test_adapter_loads_ten_recordings_in_two_motor_groups():
    recs = T.load()
    assert len(recs) == 10
    assert {r.group for r in recs} == {"motor_A_healthy", "motor_B_faulty"}
    assert sum(r.group == "motor_A_healthy" for r in recs) == 5


@needs_data
def test_adapter_refuses_a_file_of_the_wrong_size(tmp_path):
    import shutil
    for i in range(1, 11):
        shutil.copy(os.path.join(ROOT, f"FILE {i}.mat"), tmp_path / f"FILE {i}.mat")
    (tmp_path / "FILE 3.mat").write_bytes(b"x" * 1000)
    with pytest.raises(ValueError, match="expected"):
        T.index(str(tmp_path))


@needs_data
def test_adapter_does_not_use_the_label_vector():
    p = T.load(limit=1)[0].provenance
    assert p["label_vector_used"] is False
    assert p["label_reconstruction"]["validated"] is False


@needs_data
@pytest.mark.slow
def test_rule_gets_every_recording_right():
    """10/10 recording-level. The rule is the deliverable; this is its score."""
    for r in T.load():
        assert S.recording_verdict(S.classify(r)) == r.label, r.provenance["source_file"]


@needs_data
@pytest.mark.slow
def test_the_four_real_events_are_found_with_the_right_phase():
    """Measured: I2 is the lost phase in FILE 2, 5, 7 and 10."""
    expect = {2: "phase_loss_running", 5: "single_phasing_start",
              7: "phase_loss_running", 10: "single_phasing_start"}
    for r in T.load():
        n = r.condition["file_no"]
        ev = S.events(S.classify(r))
        if n in expect:
            assert ev, f"FILE {n}: no event found"
            assert ev[0]["state"] == expect[n]
            assert 1 in ev[0]["phases"], f"FILE {n}: expected phase I2"
        else:
            assert not ev, f"FILE {n}: false alarm"


@needs_data
@pytest.mark.slow
def test_latency_is_within_a_couple_of_seconds_on_real_events():
    for r in T.load():
        lat = S.detection_latency(S.classify(r))
        if lat:
            assert 0.0 <= lat["latency_s"] <= 2.0, r.provenance["source_file"]


@needs_data
def test_bearing_confound_is_stated_not_claimed():
    note = T.bearing_confound_note()
    assert "ONE motor per bearing condition" in note
    assert "negative result" in note


@needs_data
@pytest.mark.slow
@pytest.mark.parametrize("frac", [0.03, 0.05, 0.10, 0.20, 0.30, 0.40])
def test_recording_verdict_is_robust_across_thresholds(frac):
    """
    The property that matters: the RECORDING-LEVEL verdict, which is what the
    branch reports, is 10/10 across a 13x range of thresholds. The window-level
    safe band is much narrower (0.048-0.053) because of transition windows, and
    that distinction is stated in the module docstring rather than glossed.
    """
    for r in T.load():
        v = S.recording_verdict(S.classify(r, cfg={"lost_fraction": frac}))
        assert v == r.label, f"{r.provenance['source_file']} at frac={frac}"
