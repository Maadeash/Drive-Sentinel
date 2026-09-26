"""
The common Recording contract.

Most of these assert that the schema REFUSES something. That is deliberate: the
value of a shared schema is not that it holds data, it is that it stops an adapter
shipping a recording whose phases are misaligned, whose sample rate is missing, or
whose group is unset -- each of which produces a plausible, wrong number rather
than an error.
"""

import numpy as np
import pytest

from drivesentinel.common.schema import Recording, STAGES, windows


def make(**overrides):
    base = dict(
        source="kaist_pmsm",
        stage="S4",
        signals={"ia": np.zeros(1000, dtype=np.float32),
                 "ib": np.zeros(1000, dtype=np.float32)},
        fs={"ia": 10_000.0, "ib": 10_000.0},
        label="healthy",
        group="1000W",
    )
    base.update(overrides)
    return Recording(**base)


# ---------------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------------

def test_a_valid_recording_constructs():
    r = make()
    assert r.label == "healthy"
    assert r.simulated is False
    assert r.condition == {}


def test_recording_is_frozen():
    r = make()
    with pytest.raises(Exception):
        r.label = "inter_turn"


def test_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        make(stage="S9")


def test_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="source"):
        make(source="some_kaggle_thing")


def test_rejects_an_empty_group():
    """group is what makes an honest split possible; it is not optional."""
    with pytest.raises(ValueError, match="group is required"):
        make(group="")


def test_rejects_a_channel_with_no_sample_rate():
    with pytest.raises(ValueError, match="no sample rate"):
        make(signals={"ia": np.zeros(10), "ic": np.zeros(10)},
             fs={"ia": 1000.0})


def test_rejects_a_two_dimensional_channel():
    with pytest.raises(ValueError, match="1-D"):
        make(signals={"ia": np.zeros((10, 3))}, fs={"ia": 1000.0})


def test_rejects_an_empty_channel():
    with pytest.raises(ValueError, match="empty"):
        make(signals={"ia": np.zeros(0)}, fs={"ia": 1000.0})


def test_rejects_a_nonpositive_sample_rate():
    with pytest.raises(ValueError, match="fs="):
        make(signals={"ia": np.zeros(10)}, fs={"ia": 0.0})


def test_rejects_a_recording_with_no_signals():
    with pytest.raises(ValueError, match="no signals"):
        make(signals={}, fs={})


def test_every_declared_stage_is_constructible():
    for s in STAGES:
        assert make(stage=s).stage == s


# ---------------------------------------------------------------------------
# duration: shortest channel, not longest
# ---------------------------------------------------------------------------

def test_duration_uses_the_shortest_channel():
    """
    Indexing against the longest channel walks off the end of a shorter one.
    (D2 itself turns out to have equal-length channels -- the earlier claim of a
    1 s spread was a chunked-segment miscount in a hand-written parser. The rule
    still guards adapters that produce ragged channels.)
    """
    r = make(signals={"ia": np.zeros(1000), "ib": np.zeros(800)},
             fs={"ia": 1000.0, "ib": 1000.0})
    assert r.duration() == pytest.approx(0.8)


def test_duration_of_a_named_channel():
    r = make(signals={"ia": np.zeros(1000), "ib": np.zeros(800)},
             fs={"ia": 1000.0, "ib": 1000.0})
    assert r.duration("ia") == pytest.approx(1.0)


def test_duration_accounts_for_differing_rates():
    """D2 pairs 100 kHz current with 25.6 kHz vibration in one recording."""
    r = make(signals={"ia": np.zeros(100_000), "vib": np.zeros(25_600)},
             fs={"ia": 100_000.0, "vib": 25_600.0})
    assert r.duration() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# channel access
# ---------------------------------------------------------------------------

def test_channels_returns_them_in_the_order_asked():
    r = make(signals={"ia": np.full(5, 1.0), "ib": np.full(5, 2.0)},
             fs={"ia": 10.0, "ib": 10.0})
    a, b = r.channels("ib", "ia")
    assert a[0] == 2.0 and b[0] == 1.0


def test_channels_names_what_is_missing_and_what_exists():
    r = make()
    with pytest.raises(KeyError) as e:
        r.channels("ia", "ic")
    assert "ic" in str(e.value)
    assert "ia" in str(e.value)     # the "have" list


def test_aligned_truncates_to_the_common_length():
    r = make(signals={"ia": np.zeros(1000), "ib": np.zeros(800)},
             fs={"ia": 1000.0, "ib": 1000.0})
    a, b = r.aligned("ia", "ib")
    assert len(a) == len(b) == 800


def test_aligned_refuses_to_mix_sample_rates():
    """
    Silently resampling one of three phase currents produces a plausible and
    wrong negative-sequence number. Refuse instead.
    """
    r = make(signals={"ia": np.zeros(1000), "vib": np.zeros(256)},
             fs={"ia": 1000.0, "vib": 256.0})
    with pytest.raises(ValueError, match="resample explicitly"):
        r.aligned("ia", "vib")


# ---------------------------------------------------------------------------
# with_signals: how an adapter applies decimation or a polarity fix
# ---------------------------------------------------------------------------

def test_with_signals_preserves_identity_fields():
    r = make(condition={"severity": 0.68}, group="1000W")
    r2 = r.with_signals({"ia": np.ones(10)}, fs={"ia": 100.0})
    assert r2.group == "1000W"
    assert r2.label == r.label
    assert r2.condition == {"severity": 0.68}
    assert r2.fs == {"ia": 100.0}


def test_with_signals_does_not_mutate_the_original():
    r = make()
    r.with_signals({"ia": np.ones(10)}, fs={"ia": 100.0})
    assert set(r.signals) == {"ia", "ib"}
    assert r.fs["ia"] == 10_000.0


def test_with_signals_can_override_other_fields():
    r = make()
    r2 = r.with_signals(r.signals, condition={"polarity": [1, -1]})
    assert r2.condition == {"polarity": [1, -1]}


def test_with_signals_still_validates():
    r = make()
    with pytest.raises(ValueError, match="no sample rate"):
        r.with_signals({"iq": np.ones(10)}, fs={"ia": 100.0})


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------

def test_windows_shape_and_count():
    """1000 samples at 1 kHz, 0.2 s window, 0.1 s hop -> 9 windows of 200."""
    r = make(signals={"ia": np.arange(1000, dtype=float),
                      "ib": np.arange(1000, dtype=float)},
             fs={"ia": 1000.0, "ib": 1000.0})
    w = windows(r, ("ia", "ib"), window_s=0.2, hop_s=0.1)
    assert w.shape == (9, 2, 200)
    assert w.dtype == np.float32


def test_windows_are_contiguous_and_correctly_hopped():
    r = make(signals={"ia": np.arange(100, dtype=float)}, fs={"ia": 100.0})
    w = windows(r, ("ia",), window_s=0.2, hop_s=0.1)
    assert w[0, 0, 0] == 0
    assert w[1, 0, 0] == 10      # hop of 0.1 s at 100 Hz
    assert w[0, 0, -1] == 19


def test_windows_returns_empty_when_the_recording_is_too_short():
    r = make(signals={"ia": np.zeros(50)}, fs={"ia": 1000.0})
    w = windows(r, ("ia",), window_s=1.0, hop_s=0.5)
    assert w.shape == (0, 1, 1000)


def test_windows_respects_the_shorter_channel():
    r = make(signals={"ia": np.zeros(1000), "ib": np.zeros(500)},
             fs={"ia": 1000.0, "ib": 1000.0})
    w = windows(r, ("ia", "ib"), window_s=0.1, hop_s=0.1)
    assert w.shape[0] == 5       # bounded by ib, not ia


def test_windows_rejects_a_degenerate_hop():
    r = make(signals={"ia": np.zeros(1000)}, fs={"ia": 1000.0})
    with pytest.raises(ValueError, match="must be positive"):
        windows(r, ("ia",), window_s=0.1, hop_s=0.0)


def test_d2_planned_windowing_gives_the_expected_count():
    """
    The B-S4 spec: 120 s at 10 kHz after decimation, 1.0 s window, 0.5 s hop.
    Audit section 4.1 predicts ~239 windows per recording; pin it.
    """
    r = make(signals={"ia": np.zeros(1_200_000, dtype=np.float32)},
             fs={"ia": 10_000.0})
    w = windows(r, ("ia",), window_s=1.0, hop_s=0.5)
    assert w.shape == (239, 1, 10_000)


def test_d4_planned_windowing_gives_the_expected_count():
    """B-S1 spec: 20 s at 50 kHz, 0.2 s window, 0.1 s hop -> 199 windows."""
    r = make(source="thomas_motor", stage="S1",
             signals={"v1": np.zeros(1_000_000, dtype=np.float32)},
             fs={"v1": 50_000.0})
    w = windows(r, ("v1",), window_s=0.2, hop_s=0.1)
    assert w.shape[0] == 199


# ---------------------------------------------------------------------------
# provenance and the simulated flag
# ---------------------------------------------------------------------------

def test_simulated_defaults_false_and_is_settable():
    assert make().simulated is False
    assert make(source="sim_gem", simulated=True).simulated is True


def test_provenance_carries_adapter_decisions():
    """
    The D2 adapter must record the polarity it DETECTED, because the correction
    is inferred rather than read. Audit section 3.2, finding 4.
    """
    r = make(provenance={"polarity_detected": [1, 1, -1],
                         "channels": ["ai0", "ai2", "ai3"],
                         "dedup_of": None})
    assert r.provenance["polarity_detected"] == [1, 1, -1]


def test_summary_mentions_group_label_and_rates():
    s = make(group="1000W", label="inter_turn").summary()
    assert "1000W" in s and "inter_turn" in s and "10000Hz" in s


def test_summary_marks_simulated_recordings():
    s = make(source="sim_gem", simulated=True).summary()
    assert "SIMULATED" in s
