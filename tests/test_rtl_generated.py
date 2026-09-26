"""
The generated RTL must agree with the export it was generated from.

`rtl/ds_gen.vh` and `rtl/mem/requant_params.mem` are derived from
`artifacts/int8_export/scales.json`. If the model is ever re-exported and those are
not regenerated, the Verilog goes on computing the old multipliers against the new
weights -- and it does it silently, because the shapes would still match. That is
the same failure `tests/test_rtl_spec.py` guards the specification against; this
guards the implementation.

The weights and biases themselves are NOT generated. The RTL reads the exported
`.mem` files unchanged, which is what makes a disagreement with the golden reference
a logic bug rather than a transcription error (`rtl_spec.md` section 6). The tests
below assert that property too, by checking the header points at those files and at
no others.
"""

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "artifacts", "int8_export")
RTL = os.path.join(ROOT, "rtl")
GEN = os.path.join(RTL, "ds_gen.vh")
PARAM_MEM = os.path.join(RTL, "mem", "requant_params.mem")
PARAMS_JSON = os.path.join(ROOT, "artifacts", "rtl", "rtl_params.json")

needs_export = pytest.mark.skipif(
    not os.path.exists(os.path.join(EXPORT, "scales.json")),
    reason="INT8 export absent")
needs_gen = pytest.mark.skipif(
    not os.path.exists(GEN), reason="rtl/ds_gen.vh absent -- run gen_rtl_params.py")


def gen_text():
    with open(GEN, encoding="utf-8") as fh:
        return fh.read()


def defines():
    """`DS_NAME -> value, as text."""
    out = {}
    for m in re.finditer(r"^`define\s+(\S+)\s+(.+?)\s*(?://.*)?$",
                         gen_text(), re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def scales():
    with open(os.path.join(EXPORT, "scales.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the regeneration check -- the one that actually catches drift
# ---------------------------------------------------------------------------

@needs_export
@needs_gen
def test_generated_files_match_a_fresh_regeneration():
    """
    Re-run the generator in --check mode. It regenerates into memory and exits
    non-zero if either committed file differs by so much as a byte.

    This is the test that matters: everything below it checks one property at a
    time, and this checks all of them at once, including the ones nobody thought
    to write a test for.
    """
    r = subprocess.run(
        [sys.executable, os.path.join("scripts", "rtl", "gen_rtl_params.py"),
         "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        "rtl/ds_gen.vh or rtl/mem/requant_params.mem is stale against "
        "artifacts/int8_export/. Re-run scripts/rtl/gen_rtl_params.py.\n"
        + r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# the header points at the EXPORTED weights, unchanged
# ---------------------------------------------------------------------------

@needs_export
@needs_gen
def test_header_loads_the_exported_mem_files_and_not_copies():
    d = defines()
    for i, spec in enumerate(scales()["layers"]):
        assert d[f"DS_W{i}_FILE"] == f'"artifacts/int8_export/{spec["W_file"]}"'
        assert d[f"DS_B{i}_FILE"] == f'"artifacts/int8_export/{spec["b_file"]}"'


@needs_export
@needs_gen
def test_weight_base_offsets_tile_the_unified_memory_exactly():
    """
    The weights of the five layers are concatenated into one memory. If the
    offsets did not tile it exactly, one layer would read another's weights --
    the right number of terms from the wrong addresses, which looks like a
    training bug and is not.
    """
    d, S = defines(), scales()
    base = 0
    for i, spec in enumerate(S["layers"]):
        assert int(d[f"DS_W{i}_BASE"]) == base, spec["name"]
        n = 1
        for dim in spec["W_shape"]:
            n *= dim
        base += n
    assert base == int(d["DS_W_DEPTH"]) == S["quantisation"]["n_weights"]

    base = 0
    for i, spec in enumerate(S["layers"]):
        assert int(d[f"DS_B{i}_BASE"]) == base, spec["name"]
        base += spec["b_shape"][0]
    assert base == int(d["DS_B_DEPTH"]) == S["quantisation"]["n_biases"]


# ---------------------------------------------------------------------------
# the layer table
# ---------------------------------------------------------------------------

@needs_export
@needs_gen
def test_layer_table_matches_the_export():
    d, S = defines(), scales()
    L_in = 512
    for i, spec in enumerate(S["layers"]):
        W = spec["W_shape"]
        if spec["kind"] == "conv":
            O, I, K = W
            L_out = (L_in + 2 * spec["padding"] - K) // spec["stride"] + 1
        else:
            O, I, K, L_out = W[0], W[1], 1, 1
        assert int(d[f"DS_L{i}_O"]) == O, spec["name"]
        assert int(d[f"DS_L{i}_I"]) == I, spec["name"]
        assert int(d[f"DS_L{i}_K"]) == K, spec["name"]
        assert int(d[f"DS_L{i}_LIN"]) == L_in, spec["name"]
        assert int(d[f"DS_L{i}_LOUT"]) == L_out, spec["name"]
        assert int(d[f"DS_L{i}_STRIDE"]) == spec["stride"], spec["name"]
        assert int(d[f"DS_L{i}_PAD"]) == spec["padding"], spec["name"]
        assert int(d[f"DS_L{i}_TERMS"]) == I * K, spec["name"]
        pool = bool(int(d[f"DS_L{i}_POOL"]))
        # The pool collapses the order axis, so the next layer sees one position
        # per channel and the destination stride is 1 rather than L_out.
        assert int(d[f"DS_L{i}_DSTL"]) == (1 if pool else L_out), spec["name"]
        L_in = 1 if pool else L_out


@needs_export
@needs_gen
def test_exactly_one_pool_layer_and_one_last_layer():
    d, S = defines(), scales()
    n = len(S["layers"])
    assert sum(int(d[f"DS_L{i}_POOL"]) for i in range(n)) == 1
    assert sum(int(d[f"DS_L{i}_LAST"]) for i in range(n)) == 1
    # and the pool is the layer immediately before the linear one
    pool = [i for i in range(n) if int(d[f"DS_L{i}_POOL"])][0]
    last = [i for i in range(n) if int(d[f"DS_L{i}_LAST"])][0]
    assert pool == last - 1


# ---------------------------------------------------------------------------
# the requantisation parameter image
# ---------------------------------------------------------------------------

@needs_gen
def test_param_mem_has_one_word_per_output_channel():
    d = defines()
    with open(PARAM_MEM) as fh:
        words = [ln for ln in fh if ln.strip()]
    assert len(words) == int(d["DS_B_DEPTH"])
    assert all(len(w.strip()) == 8 for w in words), "expected 32-bit hex words"


@needs_gen
def test_param_word_fields_fit_and_the_mantissa_is_normalised():
    """
    Each word is {shift, m0}. m0 must be normalised into [2**(F-1), 2**F) -- that
    is what makes the fractional width mean "significant bits" -- and shift must
    fit the 6-bit field ds_defs.vh declares.
    """
    d = defines()
    F = int(d["DS_FRAC_W"])
    shift_w = 32 - F
    assert shift_w >= 6, "the shift field has been squeezed below 6 bits"
    with open(PARAM_MEM) as fh:
        for n, line in enumerate(fh):
            if not line.strip():
                continue
            w = int(line.strip(), 16)
            m0, shift = w & ((1 << F) - 1), w >> F
            assert 1 <= shift < (1 << shift_w), f"entry {n}: shift {shift}"
            assert (1 << (F - 1)) <= m0 < (1 << F), f"entry {n}: m0 {m0}"


@needs_gen
def test_the_three_fc_classes_share_one_shift():
    """
    ds_fc compares the three raw products directly, so they must be in the same
    units. Per-class shifts would put them in three different ones -- which is
    the mistake rtl_spec.md section 5 warns about, reintroduced one level down.
    """
    d = defines()
    F, n_b = int(d["DS_FRAC_W"]), int(d["DS_B_DEPTH"])
    n_classes = int(d["DS_L4_O"])
    with open(PARAM_MEM) as fh:
        words = [int(ln.strip(), 16) for ln in fh if ln.strip()]
    fc = words[n_b - n_classes:]
    shifts = {w >> F for w in fc}
    assert len(shifts) == 1, f"fc shifts differ: {shifts}"
    assert shifts.pop() == int(d["DS_FC_SHIFT"])


# ---------------------------------------------------------------------------
# the build ID, and why it is not in the design
# ---------------------------------------------------------------------------

@needs_gen
def test_the_build_id_is_not_compiled_into_the_rtl():
    """
    docs/rtl_declarations.md D-3: the hardware computes the build ID by walking
    the memories it actually loaded. A constant emitted by the same script that
    produced those memories would make V-6 true by construction and prove
    nothing -- so the expected value lives in the artefact JSON, for the
    testbench, and must not appear in any Verilog source.
    """
    if not os.path.exists(PARAMS_JSON):
        pytest.skip("artifacts/rtl/rtl_params.json absent")
    with open(PARAMS_JSON) as fh:
        expected = json.load(fh)["expected_build_id_hex"]
    for name in os.listdir(RTL):
        if not name.endswith((".v", ".vh")):
            continue
        with open(os.path.join(RTL, name), encoding="utf-8") as fh:
            text = fh.read().lower()
        assert expected.lower() not in text, (
            f"{name} contains the expected build ID as a literal; V-6 would be "
            f"true by construction")


@needs_export
def test_build_id_algorithm_is_crc32_of_the_exported_bytes():
    """
    The Python side of V-6. The RTL walks the loaded memories; this walks the
    files. They are independent implementations of the same definition, which is
    what makes the comparison a check rather than a tautology -- so the
    definition itself is pinned here.
    """
    if not os.path.exists(PARAMS_JSON):
        pytest.skip("artifacts/rtl/rtl_params.json absent")
    import binascii
    import numpy as np

    with open(PARAMS_JSON) as fh:
        params = json.load(fh)

    def read_mem(path, bits):
        with open(path) as fh:
            vals = [int(ln.strip(), 16) for ln in fh if ln.strip()]
        half, full = 1 << (bits - 1), 1 << bits
        return [v - full if v >= half else v for v in vals]

    S = scales()
    blob = bytearray()
    for spec in S["layers"]:
        blob += bytes(v & 0xFF for v in
                      read_mem(os.path.join(EXPORT, spec["W_file"]), 8))
    for spec in S["layers"]:
        for v in read_mem(os.path.join(EXPORT, spec["b_file"]), 32):
            blob += int(v).to_bytes(4, "little", signed=True)
    assert (binascii.crc32(bytes(blob)) & 0xFFFFFFFF) == params["expected_build_id"]
    assert np is not None


# ---------------------------------------------------------------------------
# the width came from a measurement
# ---------------------------------------------------------------------------

@needs_gen
def test_the_fractional_width_came_from_a_sweep_not_a_default():
    if not os.path.exists(PARAMS_JSON):
        pytest.skip("artifacts/rtl/rtl_params.json absent")
    with open(PARAMS_JSON) as fh:
        params = json.load(fh)
    assert params["frac_width_source"] in ("bit_exact_confirm.json",
                                           "requant_sweep.json"), \
        "the width did not come from a measured sweep"
    assert int(defines()["DS_FRAC_W"]) == params["frac_width"]

    conf = os.path.join(ROOT, "artifacts", "rtl", "bit_exact_confirm.json")
    if os.path.exists(conf):
        with open(conf) as fh:
            c = json.load(fh)
        # The shipped width must be one the full-cache confirmation found
        # bit-exact -- not merely one that passed the 512-window curve.
        assert params["frac_width"] in c["bit_exact_widths"]
