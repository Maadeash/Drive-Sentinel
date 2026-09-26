"""
Full-cache bit-exactness confirmation, per fractional width.

`sweep_requant.py` measures the per-layer accumulator curve on a 512-window slice
for memory reasons, and that slice said F = 25 was bit-exact. Over the whole cache
it is not: 64 conv4 accumulators out of 33.2 million differ, because a 1-LSB
difference in conv3's requantised int8 output propagates. 512 windows were not
enough to see it, which is the ordinary reason a partial check is misleading rather
than a dramatic one.

V-3's declared scope is the first 64 windows (`rtl_spec.md` section 8), so F = 25
would pass it. This script asks the stronger question instead -- what width makes
the integer datapath bit-identical to the float specification on EVERY accumulator
of EVERY layer of ALL 16,211 windows -- because the answer costs a handful of DSP
slices and turns V-3 from a spot check into a statement about the datapath.

Writes `artifacts/rtl/bit_exact_confirm.json`.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--widths", type=int, nargs="+",
                    default=list(range(20, M.MAX_FRAC_WIDTH + 1)))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    exp = M.Export()
    X, _, _ = load_cache()
    n = len(X)

    xq = np.empty((n, X.shape[1], X.shape[2]), dtype=np.int8)
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        xq[s:e] = M.quantise_input(X[s:e].astype(np.float64), exp)
    print(f"  {n:,} windows, widths {args.widths}")

    rows = []
    for F in args.widths:
        t0 = time.time()
        acc_bad = {L["name"]: 0 for L in exp.layers}
        acc_tot = {L["name"]: 0 for L in exp.layers}
        act_bad = 0
        pred_bad = 0
        worst_logit = 0.0
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            a = M.forward(xq[s:e], exp, requant="float", trace=True)
            b = M.forward(xq[s:e], exp, requant="int", frac_width=F, trace=True)
            for li, L in enumerate(exp.layers):
                acc_bad[L["name"]] += int((a["acc"][li] != b["acc"][li]).sum())
                acc_tot[L["name"]] += int(a["acc"][li].size)
            for u, v in zip(a["act"], b["act"]):
                act_bad += int((u != v).sum())
            pred_bad += int((a["pred"] != b["pred"]).sum())
            worst_logit = max(worst_logit,
                              float(np.abs(a["logits"] - b["logits"]).max()))
        rows.append({
            "frac_width": F,
            "accumulators_compared": acc_tot,
            "accumulator_mismatches": acc_bad,
            "activation_mismatches": act_bad,
            "argmax_mismatches": pred_bad,
            "max_abs_logit_diff": worst_logit,
            "bit_exact_all_windows": sum(acc_bad.values()) == 0 and act_bad == 0,
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"  F={F:2d}  acc mismatches {sum(acc_bad.values()):>9,}  "
              f"act {act_bad:>9,}  argmax {pred_bad:>3}  "
              f"max|dlogit| {worst_logit:.3e}  "
              f"{'BIT-EXACT' if r['bit_exact_all_windows'] else ''} "
              f"[{r['seconds']}s]", flush=True)

    exact = [r["frac_width"] for r in rows if r["bit_exact_all_windows"]]
    smallest = None
    for r in rows:
        if all(q["bit_exact_all_windows"] for q in rows[rows.index(r):]):
            smallest = r["frac_width"]
            break

    report = {
        "n_windows": int(n),
        "reference": "drivesentinel.rtl.model forward(requant='float'), which is "
                     "byte-identical to golden_reference.run_int8 (asserted in "
                     "scripts/rtl/sweep_requant.py and tests/test_rtl_model.py)",
        "widths": rows,
        "bit_exact_widths": exact,
        "smallest_bit_exact_width": smallest,
    }
    out = os.path.join(OUT_DIR, "bit_exact_confirm.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  smallest fully bit-exact width: {smallest}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
