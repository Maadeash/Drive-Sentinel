"""
D2 KAIST adapter.

Unit tests on the parsing and the signal maths always run. Integration tests
against `data_ext/kaist_pmsm/` skip when the dataset is absent, and are marked
`slow` because a full load reads 13 GB.
"""

import math
import os

import numpy as np
import pytest

from drivesentinel.adapters import kaist_pmsm as K

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_ext", "kaist_pmsm")
needs_data = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, "current")),
    reason="D2 not present (data_ext/ is gitignored)",
)


def three_phase(mag=(1.0, 1.0, 1.0), phase_deg=(0.0, -120.0, 120.0)):
    return [m * complex(math.cos(math.radians(p)), math.sin(math.radians(p)))
            for m, p in zip(mag, phase_deg)]


def faulted_3wire(neg_frac: float, pos: complex = 1.0 + 0j):
    """
    A physically valid faulted set for a THREE-WIRE machine.

    Scaling one phase's magnitude, which is the obvious way to fake a fault, is
    not valid here: it gives a non-zero zero-sequence component, and in a
    three-wire machine the three currents sum to zero by Kirchhoff whatever the
    winding is doing.  A real stator asymmetry shows up as negative sequence on
    top of positive sequence, and both of those sum to zero by construction.

    Building it correctly matters because `detect_polarity` minimises |I0|: if
    the synthetic fault carried spurious zero sequence, the detector would be
    asked to cancel the fault, and the test would be measuring the wrong thing.
    """
    a = K._A
    p, n = pos, neg_frac * pos
    return [p + n, p * a * a + n * a, p * a + n * a * a]


# ---------------------------------------------------------------------------
# filename parsing -- both severity spellings, and the `coil` token
# ---------------------------------------------------------------------------

def test_parses_underscore_severity():
    m = K.parse_filename("1000W_0_68_current_intercoil.tdms")
    assert (m["rating_w"], m["severity_pct"], m["fault_type"]) == (1000, 0.68, "intercoil")


def test_parses_dotted_severity():
    """Exactly one file in the dataset writes the decimal point as a dot."""
    m = K.parse_filename("1000W_1.01_current_intercoil.tdms")
    assert m["severity_pct"] == 1.01
    assert m["fault_type"] == "intercoil"


def test_both_spellings_of_the_same_severity_agree():
    a = K.parse_filename("1000W_1_01_vibration_intercoil.tdms")["severity_pct"]
    b = K.parse_filename("1000W_1.01_current_intercoil.tdms")["severity_pct"]
    assert a == b == 1.01


def test_coil_token_folds_to_intercoil():
    """8 vibration files say `coil` where their current partners say `intercoil`."""
    m = K.parse_filename("3000W_13_30_vibration_coil.tdms")
    assert m["fault_type"] == "intercoil"
    assert m["fault_type_token"] == "coil"


def test_parses_multi_digit_severity():
    assert K.parse_filename("1500W_37_66_current_intercoil.tdms")["severity_pct"] == 37.66


def test_unparseable_filename_raises():
    with pytest.raises(ValueError, match="unparseable"):
        K.parse_filename("something_else.tdms")


def test_severity_zero_is_healthy():
    assert K.label_for("intercoil", 0.0) == "healthy"
    assert K.label_for("interturn", 0.0) == "healthy"


def test_nonzero_severity_keeps_its_fault_type():
    assert K.label_for("interturn", 2.26) == "inter_turn"
    assert K.label_for("intercoil", 0.68) == "inter_coil"


# ---------------------------------------------------------------------------
# sequence components
# ---------------------------------------------------------------------------

def test_balanced_set_has_no_negative_or_zero_sequence():
    i0, i1, i2 = K.sequence_components(three_phase())
    assert i0 == pytest.approx(0.0, abs=1e-12)
    assert max(i1, i2) == pytest.approx(1.0, rel=1e-9)
    assert min(i1, i2) == pytest.approx(0.0, abs=1e-12)


def test_reversed_rotation_is_still_balanced():
    """ACB wiring swaps which component is 'positive'. min/max is invariant."""
    assert K.negative_sequence_ratio(
        three_phase(phase_deg=(0.0, 120.0, -120.0))) == pytest.approx(0.0, abs=1e-12)


def test_injected_negative_sequence_raises_the_ratio():
    assert K.negative_sequence_ratio(three_phase()) == pytest.approx(0.0, abs=1e-12)
    assert K.negative_sequence_ratio(faulted_3wire(0.08)) == pytest.approx(0.08, rel=1e-9)


def test_a_valid_faulted_set_has_no_zero_sequence():
    """Kirchhoff: three wires, so I0 == 0 however bad the winding is."""
    i0, _, _ = K.sequence_components(faulted_3wire(0.25))
    assert i0 == pytest.approx(0.0, abs=1e-12)


def test_ratio_grows_monotonically_with_injected_unbalance():
    prev = -1.0
    for frac in (0.0, 0.01, 0.03, 0.08, 0.2):
        r = K.negative_sequence_ratio(faulted_3wire(frac))
        assert r > prev
        prev = r


def test_ratio_is_invariant_to_uniform_amplitude_scaling():
    """The whole point of a ratio feature on probe-volt data."""
    a = K.negative_sequence_ratio(three_phase(mag=(1.2, 1.0, 0.9)))
    b = K.negative_sequence_ratio(three_phase(mag=(12.0, 10.0, 9.0)))
    assert a == pytest.approx(b, rel=1e-9)


# ---------------------------------------------------------------------------
# polarity detection
# ---------------------------------------------------------------------------

def test_detector_leaves_a_clean_set_alone():
    signs, before, after = K.detect_polarity(three_phase())
    assert signs == (1, 1, 1)
    assert after == pytest.approx(before)


def test_detector_recovers_a_planted_single_flip():
    ph = three_phase()
    ph[2] = -ph[2]
    signs, before, after = K.detect_polarity(ph)
    assert signs == (1, 1, -1)
    assert before > 0.9           # the artefact it is there to catch
    assert after < 1e-9


def test_detector_recovers_a_planted_double_flip():
    ph = three_phase()
    ph[1], ph[2] = -ph[1], -ph[2]
    signs, before, after = K.detect_polarity(ph)
    assert signs == (1, -1, -1)
    assert before > 0.9
    assert after < 1e-9


def test_detector_holds_the_first_channel_positive():
    """A global flip is physically meaningless; pinning ch0 keeps it unique."""
    ph = [-p for p in three_phase()]
    signs, _, _ = K.detect_polarity(ph)
    assert signs[0] == 1


@pytest.mark.parametrize("flip", [(1, 1, -1), (1, -1, 1), (1, -1, -1)])
def test_detector_survives_a_real_fault_on_top_of_a_flip(flip):
    """
    The sign fix must recover the wiring AND leave the fault intact. This is the
    test that matters: a detector that cancelled the fault along with the
    artefact would look perfect by its own metric and destroy the branch.
    """
    clean = faulted_3wire(0.09)
    ph = [flip[i] * clean[i] for i in range(3)]
    signs, before, after = K.detect_polarity(ph)
    assert signs == flip
    assert before > 0.5                       # the artefact it exists to catch
    assert after < 1e-9                       # and it is gone
    corrected = [signs[i] * ph[i] for i in range(3)]
    assert K.negative_sequence_ratio(corrected) == pytest.approx(0.09, rel=1e-9)


# ---------------------------------------------------------------------------
# gain calibration -- built, measured, disabled
# ---------------------------------------------------------------------------

def test_gain_calibration_is_off_by_default():
    import inspect
    assert inspect.signature(K.load).parameters["gain_calibration"].default is False


def test_gain_calibration_can_zero_i0_by_construction():
    """
    Two free real parameters against a 2-DOF complex residual. This test exists
    to document WHY the 'after' figure is not a diagnostic: it is algebra.
    """
    ph = three_phase(mag=(1.0, 0.9, 1.15))
    gains, i0_after = K.calibrate_gains(ph)
    assert i0_after < 1e-9
    assert gains[0] == 1.0


def test_gain_calibration_clamps_implausible_corrections():
    ph = three_phase(mag=(1.0, 0.3, 3.0))
    gains, _ = K.calibrate_gains(ph, max_trim=0.15)
    assert all(0.85 - 1e-9 <= g <= 1.15 + 1e-9 for g in gains[1:])


def test_gain_calibration_can_move_the_fault_feature():
    """The measured reason it is disabled. If this stops being true, revisit."""
    ph = three_phase(mag=(1.0, 0.93, 0.88))
    before = K.negative_sequence_ratio(ph)
    gains, _ = K.calibrate_gains(ph)
    after = K.negative_sequence_ratio([gains[i] * ph[i] for i in range(3)])
    assert abs(after - before) > 0.02


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_current_unit_is_not_amps():
    assert K.CURRENT_UNIT == "probe_V"


def test_absolute_amplitude_helper_refuses():
    with pytest.raises(AssertionError, match="probe volts"):
        K.assert_ratio_only("rms_feature")


# ---------------------------------------------------------------------------
# index / dedup  (reads only headers, so it is fast)
# ---------------------------------------------------------------------------

@needs_data
def test_index_finds_48_entries():
    assert len(K.index(ROOT)) == 48


@needs_data
def test_dedup_marks_exactly_three():
    dups = [e for e in K.index(ROOT) if e["is_duplicate"]]
    assert len(dups) == 3
    assert {e["rating_w"] for e in dups} == {1000, 1500, 3000}


@needs_data
def test_dedup_keeps_the_intercoil_named_copy():
    for e in K.index(ROOT):
        if e["is_duplicate"]:
            assert "interturn" in e["current_file"]
            assert "intercoil" in e["duplicate_of"]


@needs_data
def test_duplicates_are_verified_byte_identical():
    for e in K.index(ROOT):
        if e["is_duplicate"]:
            assert e["duplicate_verified_identical"]


@needs_data
def test_unique_recording_counts():
    uniq = [e for e in K.index(ROOT) if not e["is_duplicate"]]
    assert len(uniq) == 45
    by = {}
    for e in uniq:
        by[e["label"]] = by.get(e["label"], 0) + 1
    assert by == {"healthy": 3, "inter_coil": 21, "inter_turn": 21}


@needs_data
def test_the_grid_is_complete():
    """3 ratings x 2 fault types x 8 severities, one run per cell."""
    ent = K.index(ROOT)
    assert len(ent) == 48
    for rating in (1000, 1500, 3000):
        for ft in ("intercoil", "interturn"):
            sev = [e["severity_pct"] for e in ent
                   if e["rating_w"] == rating and e["fault_type"] == ft]
            assert len(sev) == 8, (rating, ft)
            assert len(set(sev)) == 8
            assert 0.0 in sev


# ---------------------------------------------------------------------------
# full load  (slow: reads 13 GB)
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.slow
def test_load_returns_45_recordings():
    recs = K.load(ROOT, limit=4)
    assert all(r.source == "kaist_pmsm" and r.stage == "S4" for r in recs)
    assert all(set(r.signals) == {"ia", "ib", "ic"} for r in recs)


@needs_data
@pytest.mark.slow
def test_load_never_hardcodes_channel_names():
    """
    The seven cDAQ5Mod1 recordings use ai0/ai1/ai3. Load one and confirm three
    channels came back with the alternate names.
    """
    ent = [e for e in K.index(ROOT)
           if e["rating_w"] == 3000 and e["fault_type"] == "intercoil"
           and e["severity_pct"] == 7.12]
    assert len(ent) == 1
    names, data, _, _, _ = K._read_tdms(ent[0]["current_path"], "A")
    assert len(data) == 3
    assert all("cDAQ5Mod1" in n for n in names)
    assert [n.split("/")[-1] for n in names] == ["ai0", "ai1", "ai3"]


@needs_data
@pytest.mark.slow
def test_loaded_recordings_are_decimated_to_10khz():
    r = K.load(ROOT, limit=1)[0]
    assert set(r.fs.values()) == {10_000.0}
    assert r.provenance["decimate_factor"] == 10
    assert r.duration() > 100


@needs_data
@pytest.mark.slow
def test_provenance_records_every_inferred_decision():
    p = K.load(ROOT, limit=1)[0].provenance
    for key in ("polarity_signs", "gains_computed", "gains_applied",
                "gain_calibration_enabled", "i0rel_raw", "i0rel_after_sign",
                "i0rel_residual", "channels_found", "units_actual"):
        assert key in p, key
    assert p["units_actual"] == "probe_V"
    assert p["gains_applied"] == [1.0, 1.0, 1.0]
