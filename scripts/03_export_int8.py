"""
Quantise the deployment model to INT8 and write the FPGA artefacts.

Calibration and verification use DISJOINT window sets.  Calibrating and then
measuring agreement on the same windows would report how well the activation
scales fit the data they were derived from, which is not the question.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.export import export
from drivesentinel.features import load_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", type=int, default=None,
                    help="calibration windows (default from config)")
    ap.add_argument("--verify", type=int, default=2048)
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, meta = load_cache()
    rng = np.random.default_rng(C.SEED)

    n_calib = args.calib or C.QUANT_CONFIG["calibration_windows"]
    perm = rng.permutation(len(X))
    calib_idx = perm[:n_calib]
    verify_idx = perm[n_calib:n_calib + args.verify]

    print(f"  cache      : {X.shape}")
    print(f"  calibration: {len(calib_idx):,} windows")
    print(f"  verification: {len(verify_idx):,} windows (disjoint)")

    report = export(
        ckpt_path=args.ckpt, out_dir=args.out,
        calib=X[calib_idx],
        verify=X[verify_idx], verify_labels=y[verify_idx].astype(np.int64),
    )

    out_dir = args.out or C.EXPORT_DIR
    print(f"\n  weights    : {report['n_weights']:,} int8  "
          f"+ {report['n_biases']:,} int32 biases")
    print("  accumulator headroom:")
    for name, a in report["accumulator"].items():
        print(f"    {name:6} {a['terms']:4d} terms  peak {a['peak_abs']:>12,}  "
              f"{a['bits_needed']:2d} bits  int32 ok: {a['fits_int32']}")

    if "agreement" in report:
        a = report["agreement"]
        print(f"\n  INT8 vs float argmax agreement : {a['argmax_agreement'] * 100:.2f} % "
              f"(n={a['n']:,})   [methodology bar: >= 98 %]")
        print(f"  max |logit| difference         : {a['max_abs_logit_diff']:.4f}")
        if "float_accuracy" in report:
            print(f"  accuracy on the same windows   : "
                  f"float {report['float_accuracy'] * 100:.2f} %  "
                  f"int8 {report['int8_accuracy'] * 100:.2f} %")
        if a["argmax_agreement"] < 0.98:
            print("\n  WARNING: agreement is below the 98 % bar in the methodology.")

    print(f"\n  wrote {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f:24} {size:>9,} B")


if __name__ == "__main__":
    main()
