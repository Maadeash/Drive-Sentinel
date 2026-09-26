"""
`drivesentinel.rtl.model` is the reference the RTL is verified against, so it has
to be verified too.

Two claims carry the whole verification flow, and if either is wrong every number
in `docs/results_rtl.md` is measured against the wrong thing:

  1. ``forward(requant="float")`` is **exactly** `golden_reference.run_int8`. Not
     close to it -- exactly, bit for bit, including the float64 rounding, because
     the golden reference IS the specification and a model that is merely nearly
     it would quietly absolve the RTL of a real disagreement.
  2. ``round_half_to_even`` is **exactly** `numpy.rint` on a division by a power of
     two. That is the operation `rtl_spec.md` section 4 names as the single most
     likely source of a mismatch that looks like a weight-loading bug.

Everything else in this file supports those two.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "artifacts", "int8_export")

needs_export = pytest.mark.skipif(
    not os.path.exists(os.path.join(EXPORT, "scales.json")),
    reason="INT8 export absent")


@pytest.fixture(scope="module")
def golden():
    spec = importlib.util.spec_from_file_location(
        "golden_reference", os.path.join(EXPORT, "golden_reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def exp():
    from drivesentinel.rtl import model as M
    return M.Export()


# ---------------------------------------------------------------------------
# claim 1: the float path IS the golden reference
# ---------------------------------------------------------------------------

@needs_export
def test_float_path_reproduces_the_golden_reference_exactly(golden, exp):
    from drivesentinel.rtl import model as M

    rng = np.random.default_rng(0)
    # Around the conditioned DSP output's own distribution (input_norm.json puts
    # the per-channel means near 0.75 and the standard deviations near 0.4), plus
    # a deliberately wild batch to push the clamps.
    x = np.concatenate([
        rng.normal(0.75, 0.40, size=(24, 5, 512)),
        rng.normal(0.00, 4.00, size=(8, 5, 512)),
    ])
    want = golden.run_int8(x)
    got = M.forward(M.quantise_input(x, exp), exp, requant="float")

    assert np.array_equal(want.argmax(1), got["pred"])
    assert np.abs(want - got["logits"]).max() == 0.0, \
        "the float path is not byte-identical to the golden reference"


@needs_export
def test_quantise_input_matches_the_golden_reference_front_end(golden, exp):
    """
    The PS front end (declarations D-4) has to hand the fabric exactly the bytes
    the reference would have produced internally, or the verified boundary is not
    the boundary the reference defines.
    """
    from drivesentinel.rtl import model as M

    rng = np.random.default_rng(1)
    x = rng.normal(0.75, 0.4, size=(8, 5, 512))
    a = (x - golden.INPUT_MEAN[None]) / golden.INPUT_STD[None]
    want = np.clip(np.rint(a / golden.LAYERS[0]["s_x"]), -128, 127).astype(np.int8)
    assert np.array_equal(want, M.quantise_input(x, exp))


# ---------------------------------------------------------------------------
# claim 2: round-half-to-even
# ---------------------------------------------------------------------------

def test_round_half_to_even_matches_numpy_rint_over_a_dense_sweep():
    from drivesentinel.rtl import model as M

    for shift in range(1, 12):
        p = np.arange(-4000, 4001, dtype=np.int64)
        got = M.round_half_to_even(p, np.full_like(p, shift))
        want = np.rint(p.astype(np.float64) / (2.0 ** shift)).astype(np.int64)
        bad = np.flatnonzero(got != want)
        assert bad.size == 0, (
            f"shift {shift}: first disagreement at p={p[bad[0]]}, "
            f"got {got[bad[0]]} want {want[bad[0]]}")


def test_round_half_to_even_is_not_floor_x_plus_half():
    """
    The shortcut rtl_spec.md section 4 names as wrong. If these ever agreed on
    ties, this file would be asserting nothing.
    """
    from drivesentinel.rtl import model as M

    ties = np.array([-3, -1, 1, 3, 5, 7], dtype=np.int64)      # k + 0.5 at shift 1
    one = np.ones_like(ties)
    rhe = M.round_half_to_even(ties, one)
    shortcut = np.floor(ties.astype(np.float64) / 2.0 + 0.5).astype(np.int64)
    assert np.array_equal(rhe, np.rint(ties / 2.0).astype(np.int64))
    assert not np.array_equal(rhe, shortcut), \
        "floor(x + 0.5) now agrees with rint on ties -- the warning is stale"


def test_round_half_to_even_handles_large_products():
    """The real datapath shifts by up to 40 with a 58-bit product."""
    from drivesentinel.rtl import model as M

    rng = np.random.default_rng(3)
    for shift in (26, 33, 40):
        p = rng.integers(-(1 << 52), 1 << 52, size=4096, dtype=np.int64)
        got = M.round_half_to_even(p, np.full_like(p, shift))
        # float64 is exact for |p| < 2**53, so the reference divide is exact too
        want = np.rint(p.astype(np.float64) / (2.0 ** shift)).astype(np.int64)
        assert np.array_equal(got, want), f"shift {shift}"


# ---------------------------------------------------------------------------
# the multiplier split
# ---------------------------------------------------------------------------

@needs_export
def test_multiplier_mantissa_is_normalised_to_the_top_of_the_range(exp):
    from drivesentinel.rtl import model as M

    for F in (8, 16, 26, 32):
        for m in exp.multipliers():
            m0, shift = M.quantise_multiplier(m, F)
            assert np.all(m0 >= (1 << (F - 1)))
            assert np.all(m0 < (1 << F))
            assert np.all(shift >= 1)
            # and it approximates what it claims to approximate
            rel = np.abs(m0 * 2.0 ** -shift - m) / m
            assert rel.max() < 2.0 ** -(F - 2), f"F={F} relative error {rel.max()}"


@needs_export
def test_pool_multiplier_carries_the_one_over_thirty_two(exp):
    """
    rtl_spec.md section 4: the 1/32 of the global average pool is folded into the
    requantisation multiplier so the design has no divider. If it were dropped,
    every fc input would be 32x too large and the argmax would still often be
    right -- which is exactly the kind of wrong that ships.
    """
    Ms = exp.multipliers()
    i = exp.pool_index
    L, nxt = exp.layers[i], exp.layers[i + 1]["s_x"]
    want = L["s_W"] * L["s_x"] / (32.0 * nxt)
    assert np.allclose(Ms[i], want, rtol=0, atol=0)


@needs_export
def test_fc_class_scales_differ_so_argmax_must_use_them(exp):
    """The premise behind ds_fc. Asserted, not assumed."""
    Ms = exp.multipliers()
    fc = Ms[-1]
    assert fc.min() != fc.max()
    # ... and within one octave, which is what lets the three share a shift
    assert fc.max() / fc.min() < 2.0


# ---------------------------------------------------------------------------
# the int path at the shipped width
# ---------------------------------------------------------------------------

@needs_export
def test_shipped_width_is_bit_exact_on_a_sample(exp):
    """
    A fast version of what scripts/rtl/confirm_bit_exact.py measured over the
    whole cache: at the shipped fractional width the integer datapath must agree
    with the float specification on every accumulator, not merely on the argmax.
    """
    from drivesentinel.rtl import model as M

    params_path = os.path.join(ROOT, "artifacts", "rtl", "rtl_params.json")
    if not os.path.exists(params_path):
        pytest.skip("artifacts/rtl/rtl_params.json absent")
    with open(params_path) as fh:
        F = json.load(fh)["frac_width"]

    rng = np.random.default_rng(7)
    xq = M.quantise_input(rng.normal(0.75, 0.4, size=(48, 5, 512)), exp)
    a = M.forward(xq, exp, requant="float", trace=True)
    b = M.forward(xq, exp, requant="int", frac_width=F, trace=True)
    for i, L in enumerate(exp.layers):
        assert np.array_equal(a["acc"][i], b["acc"][i]), L["name"]
    for u, v in zip(a["act"], b["act"]):
        assert np.array_equal(u, v)


@needs_export
def test_accumulators_stay_inside_the_widths_the_spec_claims(exp):
    """
    rtl_spec.md section 4 says int32 accumulators cannot overflow, and the RTL
    carries no saturation logic in the MAC path on the strength of that. The
    bound is architectural -- 192 terms of 127x128 -- so this checks the
    architecture, not the calibration set.
    """
    S = exp.scales["accumulator"]
    for name, a in S.items():
        assert a["peak_abs"] < 2 ** 31, name
        assert a["fits_int32"], name
    worst = max(a["terms"] for a in S.values()) * 127 * 128
    assert worst < 2 ** 31
    # the pooled sum too
    assert 32 * S["conv4"]["peak_abs"] < 2 ** 31
