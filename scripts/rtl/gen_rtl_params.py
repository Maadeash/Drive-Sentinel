"""
Generate `rtl/ds_gen.vh` and `rtl/mem/requant_params.mem` from the INT8 export.

WHAT IS GENERATED AND WHAT IS NOT
---------------------------------
Generated here: the requantisation multipliers and shifts, the layer table, the
memory depths and base offsets, the chosen fractional width, and the paths to the
exported files. All of them are properties of the EXPORTED MODEL, so re-exporting
must change them and must not require anybody to edit Verilog.

Not generated here, and deliberately: the weights and the biases. The RTL loads
`artifacts/int8_export/layer_*_{W,b}.mem` unchanged, which is what makes a
mismatch against the golden reference a logic bug rather than a transcription
error (`rtl_spec.md` section 6). Nothing in this script rewrites them.

Also not generated: the build ID. `docs/rtl_declarations.md` D-3 requires the
hardware to compute it from the memories it actually loaded. This script does
compute the expected value -- into `artifacts/rtl/rtl_params.json`, for the
testbench to compare against -- but it never puts it in the design, because a
constant in the design would make V-6 true by construction.

THE FRACTIONAL WIDTH
--------------------
Read from `artifacts/rtl/bit_exact_confirm.json` if it exists, else from
`artifacts/rtl/requant_sweep.json`. Not from an argument with a default, because
a default is a number nobody measured.
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.rtl import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTL_DIR = os.path.join(ROOT, "rtl")
MEM_DIR = os.path.join(RTL_DIR, "mem")
OUT_DIR = os.path.join(C.ARTIFACT_DIR, "rtl")

# Path the RTL uses in $readmemh. Relative, so it is the same string on every
# machine; both XSim and Vivado are launched from the repository root.
EXPORT_REL = "artifacts/int8_export"
PARAM_REL = "rtl/mem/requant_params.mem"

SHIFT_W = 6          # must match `DS_SHIFT_W in rtl/ds_defs.vh
PARAM_WORD_W = 32    # one 32-bit word per channel: {shift, m0}


def choose_width() -> tuple:
    """The width the measurements chose, and why."""
    conf = os.path.join(OUT_DIR, "bit_exact_confirm.json")
    sweep = os.path.join(OUT_DIR, "requant_sweep.json")
    if os.path.exists(conf):
        with open(conf) as fh:
            c = json.load(fh)
        if c.get("smallest_bit_exact_width"):
            return int(c["smallest_bit_exact_width"]), "bit_exact_confirm.json"
    if os.path.exists(sweep):
        with open(sweep) as fh:
            s = json.load(fh)
        if s.get("chosen_frac_width"):
            return int(s["chosen_frac_width"]), "requant_sweep.json"
    sys.exit("FATAL: no measured fractional width. Run scripts/rtl/sweep_requant.py "
             "and scripts/rtl/confirm_bit_exact.py first -- this script will not "
             "invent one.")


def crc32_of_export(exp: M.Export) -> int:
    """
    The expected build ID: CRC-32/ISO-HDLC over all int8 weights in layer order,
    then all int32 biases little-endian in layer order.

    ds_weight_crc.v walks the loaded memories in the same order. The two paths
    are independent -- this one parses the hex text with Python, that one reads
    what $readmemh put in the fabric -- which is what makes V-6 a check rather
    than a tautology.
    """
    blob = bytearray()
    for L in exp.layers:
        blob += bytes(int(v) & 0xFF for v in L["W"].reshape(-1))
    for L in exp.layers:
        for v in L["b"].reshape(-1):
            blob += int(v).to_bytes(4, "little", signed=True)
    return binascii.crc32(bytes(blob)) & 0xFFFFFFFF


def layer_table(exp: M.Export) -> List[dict]:
    rows, L_in = [], C.N_ORDER_BINS
    wbase = bbase = 0
    for idx, lay in enumerate(exp.layers):
        W = lay["W"].shape
        if lay["kind"] == "conv":
            O, I, K = W
            L_out = (L_in + 2 * lay["padding"] - K) // lay["stride"] + 1
        else:
            # fc as a degenerate convolution: a 64-term reduction over one
            # position. ds_conv1d runs it on the same datapath (see its header).
            O, I, K = W[0], W[1], 1
            L_out = 1
        is_pool = (idx == exp.pool_index)
        is_last = bool(lay["is_last"])
        rows.append({
            "index": idx, "name": lay["name"],
            "O": int(O), "I": int(I), "K": int(K),
            "L_in": int(L_in), "L_out": int(L_out),
            # Destination stride between output channels: L_out normally, 1 when
            # the pool collapses the axis, so the next layer's channel-major read
            # addressing lands on the right byte without a special case.
            "dst_l": 1 if is_pool else int(L_out),
            "stride": int(lay["stride"]), "pad": int(lay["padding"]),
            "terms": int(I) * int(K),
            "wbase": wbase, "bbase": bbase,
            "is_pool": is_pool, "is_last": is_last,
            "macs": int(O) * int(I) * int(K) * int(L_out),
            # cycles = O * (2 + L_out * (I*K + 3)) + (pool ? O : 0); the engine's
            # per-position overhead is POS_INIT, DRAIN and OUT.
            "cycles_model": int(O) * (2 + int(L_out) * (int(I) * int(K) + 3))
                            + (int(O) if is_pool else 0),
        })
        wbase += int(np.prod(W))
        bbase += int(O)
        # The pool collapses the order axis, so the NEXT layer sees one position
        # per channel. Getting this wrong gives the fc layer L_in = 32 and it
        # reads the 64 pooled bytes at a stride of 32 -- the right number of
        # terms from the wrong addresses, which looks like a weight bug.
        L_in = 1 if is_pool else (L_out if lay["kind"] == "conv" else 1)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac-width", type=int, default=None,
                    help="override; normally the measured width is used")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into memory and fail if the committed files "
                         "differ, instead of writing them")
    args = ap.parse_args()

    os.makedirs(MEM_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    exp = M.Export()
    if args.frac_width:
        F, src = args.frac_width, "command line override"
    else:
        F, src = choose_width()

    Ms = exp.multipliers()
    table = layer_table(exp)

    # ---- requantisation parameters, one word per output channel -------------
    words, meta = [], []
    for idx, (lay, m) in enumerate(zip(exp.layers, Ms)):
        m0, shift = M.quantise_multiplier(m, F)
        if lay["is_last"]:
            # ds_fc compares the three raw products, so the three classes must
            # share one shift (see ds_fc.v). The smallest shift normalises the
            # largest multiplier, so no mantissa overflows F bits.
            s = int(shift.min())
            m0 = np.rint(m * (2.0 ** s)).astype(np.int64)
            shift = np.full_like(shift, s)
            assert m0.max() < (1 << F), "fc mantissa overflows the field"
        for o in range(len(m0)):
            assert 0 < int(shift[o]) < (1 << SHIFT_W), \
                f"shift {int(shift[o])} does not fit {SHIFT_W} bits"
            assert int(m0[o]) < (1 << F), "m0 does not fit the field"
            words.append((int(shift[o]) << F) | int(m0[o]))
            meta.append({"layer": lay["name"], "channel": o,
                         "M": float(m[o]), "m0": int(m0[o]),
                         "shift": int(shift[o]),
                         "rel_error": float(abs(int(m0[o]) * 2.0 ** -int(shift[o])
                                                - m[o]) / m[o])})
    assert F + SHIFT_W <= PARAM_WORD_W, \
        f"F={F} + shift={SHIFT_W} exceeds the {PARAM_WORD_W}-bit parameter word"
    assert len(words) == exp.scales["quantisation"]["n_biases"], \
        "one requant parameter per bias, or the shared index is wrong"

    param_mem = "".join(f"{w:08x}\n" for w in words)

    # ---- the logit shift ----------------------------------------------------
    fc = exp.layers[-1]
    fc_shift = int(M.quantise_multiplier(Ms[-1], F)[1].min())
    m0_fc_max = int(np.rint(Ms[-1] * (2.0 ** fc_shift)).max())
    # Architectural bound on the fc accumulator: 64 terms of 127 x 128, plus the
    # largest bias. Not the measured peak -- the register must not overflow on an
    # input nobody has seen.
    acc_bound = fc["W"].shape[1] * 127 * 128 + int(np.abs(fc["b"]).max())
    prod_bound = m0_fc_max * acc_bound
    logit_shift = max(0, int(np.ceil(np.log2(prod_bound))) - 30)
    assert prod_bound >> logit_shift < (1 << 31), "exposed logit overflows int32"

    # ---- the header ---------------------------------------------------------
    q = exp.scales["quantisation"]
    in_bytes = exp.scales["input"]["channels"] * exp.scales["input"]["bins"]
    w_depth = sum(int(np.prod(L["W"].shape)) for L in exp.layers)
    b_depth = sum(int(L["b"].shape[0]) for L in exp.layers)
    wgt_aw = max(1, int(np.ceil(np.log2(w_depth))))
    par_aw = max(1, int(np.ceil(np.log2(b_depth))))

    lines = [
        "//===========================================================================",
        "// ds_gen.vh -- GENERATED by scripts/rtl/gen_rtl_params.py. DO NOT EDIT.",
        "//===========================================================================",
        "// Everything here is a property of artifacts/int8_export/, not of the",
        "// architecture. Re-export the model and regenerate; never hand-edit.",
        "// tests/test_rtl_generated.py regenerates this file and fails if the",
        "// committed copy differs.",
        "//",
        f"// fractional width : {F}   (chosen by {src};",
        "//                     docs/rtl_declarations.md D-2 states the rule)",
        f"// weights          : {w_depth:,} int8",
        f"// biases           : {b_depth} int32",
        "//===========================================================================",
        "",
        "`ifndef DS_GEN_VH",
        "`define DS_GEN_VH",
        "",
        "// ---- requantisation -------------------------------------------------",
        f"`define DS_FRAC_W      {F}",
        f"`define DS_FC_SHIFT    {fc_shift}   // one shared shift for the 3 classes",
        f"`define DS_LOGIT_SHIFT {logit_shift}   // raw product -> the int32 register",
        f"`define DS_LOGIT_LSB_EXP {logit_shift - fc_shift}  // register LSB = 2**this,"
        " in the reference's float units",
        "",
        "// ---- memory geometry -------------------------------------------------",
        f"`define DS_W_DEPTH     {w_depth}",
        f"`define DS_B_DEPTH     {b_depth}",
        f"`define DS_WGT_AW      {wgt_aw}",
        f"`define DS_PAR_AW      {par_aw}",
        f"`define DS_IN_BYTES    {in_bytes}",
        "",
        "// ---- exported memory images, read unchanged --------------------------",
    ]
    for i, lay in enumerate(exp.layers):
        spec = exp.scales["layers"][i]
        lines.append(f'`define DS_W{i}_FILE     "{EXPORT_REL}/{spec["W_file"]}"')
    lines.append("")
    for i, lay in enumerate(exp.layers):
        spec = exp.scales["layers"][i]
        lines.append(f'`define DS_B{i}_FILE     "{EXPORT_REL}/{spec["b_file"]}"')
    lines.append("")
    lines.append(f'`define DS_PARAM_FILE   "{PARAM_REL}"   // GENERATED, not exported')
    lines.append("")
    lines.append("// ---- base offsets into the unified memories --------------------------")
    for r in table:
        lines.append(f"`define DS_W{r['index']}_BASE     {r['wbase']}")
    lines.append("")
    for r in table:
        lines.append(f"`define DS_B{r['index']}_BASE     {r['bbase']}")
    lines.append("")
    lines.append("// ---- layer table -----------------------------------------------------")
    lines.append("// O  out channels | I in channels | K taps | LIN/LOUT length along the")
    lines.append("// order axis | DSTL destination stride | TERMS = I*K, so ds_conv1d needs")
    lines.append("// no multiplier | POOL conv4 | LAST fc")
    for r in table:
        i = r["index"]
        lines += [
            f"// layer {i}: {r['name']}  {r['macs']:,} MACs",
            f"`define DS_L{i}_O        {r['O']}",
            f"`define DS_L{i}_I        {r['I']}",
            f"`define DS_L{i}_K        {r['K']}",
            f"`define DS_L{i}_LIN      {r['L_in']}",
            f"`define DS_L{i}_LOUT     {r['L_out']}",
            f"`define DS_L{i}_DSTL     {r['dst_l']}",
            f"`define DS_L{i}_STRIDE   {r['stride']}",
            f"`define DS_L{i}_PAD      {r['pad']}",
            f"`define DS_L{i}_TERMS    {r['terms']}",
            f"`define DS_L{i}_POOL     {int(r['is_pool'])}",
            f"`define DS_L{i}_LAST     {int(r['is_last'])}",
        ]
    lines += ["", "`endif // DS_GEN_VH", ""]
    header = "\n".join(lines)

    # ---- write or check -----------------------------------------------------
    gen_path = os.path.join(RTL_DIR, "ds_gen.vh")
    mem_path = os.path.join(MEM_DIR, "requant_params.mem")

    if args.check:
        bad = []
        for path, want in ((gen_path, header), (mem_path, param_mem)):
            have = open(path, encoding="utf-8", newline="").read() \
                if os.path.exists(path) else None
            if have is None or have.replace("\r\n", "\n") != want:
                bad.append(path)
        if bad:
            sys.exit("STALE: " + ", ".join(bad))
        print("  generated files match the export")
        return

    with open(gen_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
    with open(mem_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(param_mem)

    params = {
        "frac_width": F,
        "frac_width_source": src,
        "shift_width": SHIFT_W,
        "fc_shift": fc_shift,
        "logit_shift": logit_shift,
        "logit_lsb_exponent": logit_shift - fc_shift,
        "logit_lsb_value": 2.0 ** (logit_shift - fc_shift),
        "w_depth": w_depth,
        "b_depth": b_depth,
        "input_bytes": in_bytes,
        "expected_build_id": crc32_of_export(exp),
        "expected_build_id_hex": f"{crc32_of_export(exp):08x}",
        "build_id_algorithm": "CRC-32/ISO-HDLC over int8 weights in layer order, "
                              "then int32 biases little-endian in layer order",
        "layers": table,
        "total_macs": sum(r["macs"] for r in table),
        "total_cycles_model": sum(r["cycles_model"] for r in table),
        "requant_params": meta,
        "max_relative_multiplier_error": max(m["rel_error"] for m in meta),
        "on_chip_bytes": w_depth + b_depth * 4 + b_depth * 4 + 2 * 4096,
    }
    out = os.path.join(OUT_DIR, "rtl_params.json")
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2)

    print(f"  fractional width   : {F}  ({src})")
    print(f"  fc shift           : {fc_shift}   logit shift: {logit_shift}"
          f"   LSB = 2^{logit_shift - fc_shift}")
    print(f"  max rel M error    : {params['max_relative_multiplier_error']:.3e}")
    print(f"  total MACs         : {params['total_macs']:,}")
    print(f"  modelled cycles    : {params['total_cycles_model']:,}")
    print(f"  expected build id  : {params['expected_build_id_hex']}")
    print(f"  wrote {gen_path}")
    print(f"  wrote {mem_path}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
