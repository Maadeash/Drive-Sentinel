"""
Sweep the requantisation fractional width against the golden reference.

`rtl_spec.md` section 5 deliberately refuses to name a width:

    "The exact fractional width is NOT fixed by this spec. It is chosen in item (b)
     by sweeping the width against the golden reference on all 16,211 calibration
     windows and taking the smallest width that gives 100 % argmax agreement."

`docs/rtl_declarations.md` D-2 fixes the rule. This script runs it and writes
`artifacts/rtl/requant_sweep.json`; `scripts/rtl/gen_rtl_params.py` then emits the
memory image and the Verilog header from the winner. Nothing in between is typed by
hand.

Two curves are reported, not one:

  * **argmax agreement** over all 16,211 windows -- the D-2 criterion, and the thing
    V-1 will later have to reproduce in silicon.
  * **per-layer accumulator agreement** over the V-3 window set -- because argmax
    agreement can hide a broken layer (`rtl_spec.md` section 8), and V-3 asks for
    bit-exact accumulators, which is a strictly harder bar than agreeing on which of
    three classes is largest.

A width can pass the first and fail the second; if it does, that is a finding about
V-3, reported here rather than discovered in a simulator at 3 a.m.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.features import load_cache
from drivesentinel.rtl import model as M

OUT_DIR = os.path.join(C.ARTIFACT_DIR, "rtl")


def load_golden():
    path = os.path.join(C.EXPORT_DIR, "golden_reference.py")
    spec = importlib.util.spec_from_file_location("golden_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def batched_pred(fn, n, batch, label):
    out = np.empty(n, dtype=np.int64)
    t0 = time.time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        out[s:e] = fn(s, e)
    print(f"    {label}: {time.time() - t0:.1f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--min-width", type=int, default=8)
    ap.add_argument("--max-width", type=int, default=M.MAX_FRAC_WIDTH)
    ap.add_argument("--acc-windows", type=int, default=512,
                    help="windows used for the per-layer accumulator curve")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    exp = M.Export()
    golden = load_golden()

    X, y, meta = load_cache()
    n = len(X)
    print(f"  cache        : {X.shape}")
    print(f"  export       : {exp.dir}")

    # The hardware boundary (declarations D-4): the PS front end hands the fabric
    # these bytes. Computed once -- it does not depend on the requant width.
    xq = np.empty((n, X.shape[1], X.shape[2]), dtype=np.int8)
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        xq[s:e] = M.quantise_input(X[s:e].astype(np.float64), exp)

    print("  reference pass (golden_reference.predict, all windows)")
    ref = batched_pred(lambda s, e: golden.predict(X[s:e].astype(np.float64)),
                       n, args.batch, "golden")

    # The float-requant path of the model must be the golden, exactly. If it is
    # not, every agreement figure below is measured against the wrong thing.
    print("  self-check   : model(requant='float') vs golden")
    mf = batched_pred(
        lambda s, e: M.forward(xq[s:e], exp, requant="float")["pred"],
        n, args.batch, "model-float")
    n_float_mismatch = int((mf != ref).sum())
    if n_float_mismatch:
        sys.exit(f"FATAL: model float path disagrees with golden on "
                 f"{n_float_mismatch} windows -- the model is wrong, not the RTL")
    print("    identical on all windows")

    # V-3's reference accumulators, on the first `acc-windows` windows.
    ref_trace = M.forward(xq[:args.acc_windows], exp, requant="float", trace=True)

    rows = []
    for F in range(args.min_width, args.max_width + 1):
        t0 = time.time()
        pred = batched_pred(
            lambda s, e, F=F: M.forward(xq[s:e], exp, requant="int",
                                        frac_width=F)["pred"],
            n, args.batch, f"F={F}")
        agree = float((pred == ref).mean())

        tr = M.forward(xq[:args.acc_windows], exp, requant="int", frac_width=F,
                       trace=True)
        acc_exact = {}
        for li, L in enumerate(exp.layers):
            a, b = ref_trace["acc"][li], tr["acc"][li]
            acc_exact[L["name"]] = float((a == b).mean())

        M0, shift = M.quantise_multiplier(exp.multipliers()[0], F)
        rows.append({
            "frac_width": F,
            "argmax_agreement": agree,
            "disagreements": int((pred != ref).sum()),
            "layer_accumulator_exact": acc_exact,
            "all_layers_bit_exact": all(v == 1.0 for v in acc_exact.values()),
            "shift_min": int(min(int(s.min()) for s in
                                 (M.quantise_multiplier(m, F)[1]
                                  for m in exp.multipliers()))),
            "shift_max": int(max(int(s.max()) for s in
                                 (M.quantise_multiplier(m, F)[1]
                                  for m in exp.multipliers()))),
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"  F={F:2d}  argmax {agree * 100:9.5f} %  "
              f"({r['disagreements']:5d} bad)  acc bit-exact: "
              f"{'yes' if r['all_layers_bit_exact'] else 'no '}  "
              f"[{r['seconds']}s]", flush=True)

    # D-2: the smallest width that reaches 100 % AND stays there for every larger
    # width tested. "Stays there" is the part that stops a lucky single point from
    # being read as a threshold.
    chosen = None
    for i, r in enumerate(rows):
        if all(q["argmax_agreement"] == 1.0 for q in rows[i:]):
            chosen = r["frac_width"]
            break
    chosen_acc = None
    for i, r in enumerate(rows):
        if all(q["all_layers_bit_exact"] for q in rows[i:]):
            chosen_acc = r["frac_width"]
            break

    # The accumulator curve above is measured on `acc-windows` windows for memory,
    # but V-3's claim is about the datapath, not about 512 windows. Confirm the two
    # candidate widths over the WHOLE cache: every accumulator of every layer, and
    # every int8 activation handed between layers.
    confirm = {}
    for label, F in (("chosen_frac_width", chosen),
                     ("smallest_width_all_layers_bit_exact", chosen_acc)):
        if F is None or F in confirm:
            continue
        t0 = time.time()
        acc_bad = {L["name"]: 0 for L in exp.layers}
        act_bad = 0
        total = {L["name"]: 0 for L in exp.layers}
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            a = M.forward(xq[s:e], exp, requant="float", trace=True)
            b = M.forward(xq[s:e], exp, requant="int", frac_width=F, trace=True)
            for li, L in enumerate(exp.layers):
                acc_bad[L["name"]] += int((a["acc"][li] != b["acc"][li]).sum())
                total[L["name"]] += int(a["acc"][li].size)
            for u, v in zip(a["act"], b["act"]):
                act_bad += int((u != v).sum())
        confirm[F] = {
            "frac_width": F,
            "accumulators_compared": total,
            "accumulator_mismatches": acc_bad,
            "activation_mismatches": act_bad,
            "bit_exact_all_windows": (sum(acc_bad.values()) == 0 and act_bad == 0),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  full-cache confirm F={F}: "
              f"{'BIT-EXACT' if confirm[F]['bit_exact_all_windows'] else 'differs'} "
              f"{acc_bad}  [{confirm[F]['seconds']}s]", flush=True)

    report = {
        "rule": "docs/rtl_declarations.md D-2",
        "full_cache_confirmation": list(confirm.values()),
        "n_windows": int(n),
        "acc_curve_windows": int(args.acc_windows),
        "reference": "artifacts/int8_export/golden_reference.py predict()",
        "widths_tested": [r["frac_width"] for r in rows],
        "sweep": rows,
        "chosen_frac_width": chosen,
        "smallest_width_all_layers_bit_exact": chosen_acc,
        "bar": "100.000 % argmax agreement, all windows",
        "met": chosen is not None,
    }
    out = os.path.join(OUT_DIR, "requant_sweep.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    if chosen is None:
        print("  NEGATIVE RESULT: no width in the swept range reaches 100.000 %.")
        print("  Per docs/rtl_declarations.md D-2 the bar does not move.")
    else:
        print(f"  chosen frac_width           : {chosen}")
        print(f"  smallest bit-exact (V-3)    : {chosen_acc}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
