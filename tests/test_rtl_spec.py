"""
`docs/rtl_spec.md` must agree with the artefacts it specifies.

The spec is a contract between the golden reference and RTL that does not exist yet.
If the INT8 export is ever regenerated, the spec's shapes, accumulator widths and MAC
counts go stale silently -- and a stale hardware spec is worse than none, because it
looks authoritative. These tests recompute every load-bearing number from
`artifacts/int8_export/` and assert the document still contains it.

Rounding is pinned too. Three transcription slips (conv2 M range, the fc output scale)
and one mis-rounded macro-F1 were caught this way before the spec was first committed.
"""

import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "artifacts", "int8_export")
SPEC = os.path.join(ROOT, "docs", "rtl_spec.md")

needs_export = pytest.mark.skipif(
    not os.path.exists(os.path.join(EXPORT, "scales.json")),
    reason="INT8 export absent")


def spec_text():
    if not os.path.exists(SPEC):
        pytest.skip("rtl_spec.md absent")
    with open(SPEC, encoding="utf-8") as fh:
        return fh.read()


def scales():
    with open(os.path.join(EXPORT, "scales.json")) as fh:
        return json.load(fh)


def sci(v):
    """The spec's exponent style: 1.2598e-3, not 1.2598e-03."""
    return "{:.4e}".format(v).replace("e-0", "e-")


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

@needs_export
def test_layer_shapes_and_mac_counts_match_the_export():
    doc, S = spec_text(), scales()
    L_in, total = 512, 0
    for spec in S["layers"]:
        W = spec["W_shape"]
        if spec["kind"] == "conv":
            O, I, K = W
            out = (L_in + 2 * spec["padding"] - K) // spec["stride"] + 1
            assert f"{I}x{L_in} | {O}x{out}" in doc, spec["name"]
            assert f"({O},{I},{K})" in doc, spec["name"]
            mac = O * I * K * out
            assert f"{mac:,}" in doc, spec["name"]
            L_in = out
        else:
            mac = W[0] * W[1]
        total += mac
    assert f"{total:,}" in doc, "total MAC count"


@needs_export
def test_weight_and_bias_counts_match():
    doc, q = spec_text(), scales()["quantisation"]
    assert f"{q['n_weights']:,}" in doc
    assert str(q["n_biases"]) in doc


# ---------------------------------------------------------------------------
# arithmetic contract
# ---------------------------------------------------------------------------

@needs_export
def test_accumulator_peaks_and_widths_match():
    doc, acc = spec_text(), scales()["accumulator"]
    for name, a in acc.items():
        assert f"{a['peak_abs']:,}" in doc, name
        assert a["fits_int32"], f"{name} no longer fits int32 -- spec section 4 is wrong"


@needs_export
def test_pooled_sum_bound_is_stated():
    doc, acc = spec_text(), scales()["accumulator"]
    pooled = 32 * acc["conv4"]["peak_abs"]
    assert f"{pooled:,}" in doc
    assert pooled < 2 ** 31


@needs_export
def test_requantisation_multiplier_ranges_match():
    """M[o] = s_W[o] * s_x / s_x_next, with 1/32 folded in at conv4."""
    doc, S = spec_text(), scales()
    for i, spec in enumerate(S["layers"][:-1]):
        M = np.array(spec["s_W"]) * spec["s_x"] / S["layers"][i + 1]["s_x"]
        if spec["name"] == "conv4":
            M = M / 32.0
        assert sci(M.min()) in doc, f"{spec['name']} min"
        assert sci(M.max()) in doc, f"{spec['name']} max"


@needs_export
def test_fc_output_scales_differ_so_argmax_needs_scaling():
    """
    The spec says argmax must be taken on the scaled value. That is only true
    because the three class scales are not equal -- assert the premise.
    """
    doc, S = spec_text(), scales()
    fl = S["layers"][-1]
    sc = np.array(fl["s_W"]) * fl["s_x"]
    assert sc.min() != sc.max(), "class scales equal -- spec section 5 premise is gone"
    assert sci(sc.min()) in doc
    assert sci(sc.max()) in doc


def test_round_half_to_even_is_what_the_golden_reference_does():
    """
    The spec tells the RTL to round ties to even. If numpy ever stopped doing
    that, the spec would be instructing hardware to differ from its reference.
    """
    assert [float(np.rint(v)) for v in (0.5, 1.5, 2.5, 3.5)] == [0.0, 2.0, 2.0, 4.0]
    doc = spec_text()
    assert "round_half_to_even" in doc
    assert "floor(x + 0.5)" in doc      # named as the wrong shortcut


# ---------------------------------------------------------------------------
# honesty labelling -- the reason this project has a claims audit
# ---------------------------------------------------------------------------

def test_spec_does_not_claim_an_unmeasured_accuracy_gain():
    doc = spec_text()
    assert "does not improve accuracy" in doc
    p = os.path.join(ROOT, "artifacts", "runs", "lobo_summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            assert f"{json.load(fh)['pooled']['macro_f1']:.4f}" in doc


def test_the_two_things_that_still_have_no_measurement_say_so():
    """
    The spec's ESTIMATE labelling has been partly superseded: synthesis and
    implementation have run, so the resource and clock figures are measurements
    now. Two things have NOT been measured and the document must keep saying so
    plainly, because they are the two a slide would most like to imply.
    """
    doc = spec_text()
    assert "ESTIMATE" in doc, "the arithmetic timing table is still an estimate"
    # no board, no bitstream -- stated at the top and again in section 10
    assert "No bitstream has been written and nothing has run on a board" in doc
    assert "Nothing has run on hardware" in doc
    # power and end-to-end latency remain unmeasured
    assert "Power: `NOT RUN`" in doc
    assert "end-to-end latency is not yet measured" in doc
    assert "End-to-end latency: `NOT MEASURED`" in doc


def test_spec_was_not_rewritten_to_match_the_outcome():
    """
    Sections 1-8 were declared before the RTL existed and must still read that
    way; the measurements were added as marked notes, not edited in over the
    predictions. The clearest case is the section 2 / section 7 contradiction the
    build exposed -- it is resolved in rtl_declarations.md D-4 and deliberately
    NOT edited away here, because a specification that quietly agrees with its
    own implementation has stopped being evidence of anything.
    """
    doc = spec_text()
    assert "This document has not been rewritten to match the outcome" in doc
    # the original, now-superseded sentence is still present
    assert ("Standardisation `x = (x - mean) / std` per channel is applied "
            "**inside** the accelerator") in doc
    assert "rtl_declarations.md` D-4" in doc


def test_all_five_open_questions_are_answered():
    doc = spec_text()
    assert "## 9. Open questions — ALL FIVE ANSWERED" in doc
    for n in range(1, 6):
        assert f"### 9.{n} " in doc, f"section 9.{n} missing"
    # and each answer names its evidence rather than asserting itself
    for artefact in ("artifacts/rtl/requant_sweep.json",
                     "artifacts/rtl/bit_exact_confirm.json",
                     "constraints/ds_top.xdc",
                     "docs/rtl_declarations.md` D-3"):
        assert artefact in doc, artefact


def test_rtl_acceptance_bar_is_not_the_int8_vs_float_bar():
    """
    100 % RTL-vs-golden, not the 98 % / 0.99877 INT8-vs-float figure. Confusing
    the two would let a rounding bug ship as 'within spec'.
    """
    doc = spec_text()
    assert "**100.000 %**" in doc
    assert "100 %, not 99" in doc


def test_spec_scopes_itself_to_the_bearing_stage():
    doc = spec_text()
    assert "covers stage S5 (bearing) only" in doc
