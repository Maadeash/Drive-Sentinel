"""
Verify the exported golden reference against the float model on the FULL cache.

This is the check that matters before any RTL work starts, and it is the one
that is easy to skip.  The exporter already reports agreement on a couple of
thousand windows; this runs every window in the cache through both paths, so a
disagreement that only appears on, say, one bearing's spectra cannot hide.

It loads the GENERATED golden_reference.py from the export directory -- the same
file and the same .mem images the Verilog will read -- rather than calling the
in-package quantiser.  That is the point: it tests the artefact, not the code
that made it.
"""

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from drivesentinel import config as C
from drivesentinel.features import load_cache
from drivesentinel.model import build_model, fold_batchnorm
from drivesentinel.quantize import float_forward


def load_golden(export_dir: str):
    path = os.path.join(export_dir, "golden_reference.py")
    if not os.path.exists(path):
        sys.exit(f"FATAL: {path} not found -- run scripts/03_export_int8.py first")
    spec = importlib.util.spec_from_file_location("golden_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    export_dir = args.export_dir or C.EXPORT_DIR
    ckpt = args.ckpt or os.path.join(C.RUN_DIR, "deployment_model.pt")

    golden = load_golden(export_dir)
    X, y, meta = load_cache()
    print(f"  cache      : {X.shape}")
    print(f"  golden ref : {export_dir}")

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(blob["state_dict"])
    folded = fold_batchnorm(model.eval())
    mean = np.asarray(blob["norm_mean"]).reshape(1, -1, 1)
    std = np.asarray(blob["norm_std"]).reshape(1, -1, 1)

    n = len(X)
    f_pred = np.empty(n, dtype=np.int64)
    q_pred = np.empty(n, dtype=np.int64)
    for s in range(0, n, args.batch):
        chunk = X[s:s + args.batch].astype(np.float64)
        f_pred[s:s + len(chunk)] = float_forward(folded, (chunk - mean) / std).argmax(1)
        q_pred[s:s + len(chunk)] = golden.predict(chunk)   # golden standardises internally
        if s % (args.batch * 8) == 0:
            print(f"    {s:,}/{n:,}", flush=True)

    truth = y.astype(np.int64)
    agree = float((f_pred == q_pred).mean())
    report = {
        "n_windows": int(n),
        "argmax_agreement": agree,
        "float_accuracy": float((f_pred == truth).mean()),
        "int8_accuracy": float((q_pred == truth).mean()),
        "disagreements": int((f_pred != q_pred).sum()),
        "meets_98pct_bar": bool(agree >= 0.98),
    }

    # where the disagreements live -- a cluster in one bearing is a real signal,
    # scattered singletons are just rounding at a decision boundary
    if report["disagreements"]:
        bad = meta.iloc[np.flatnonzero(f_pred != q_pred)]
        report["disagreements_by_bearing"] = bad["bearing"].value_counts().to_dict()

    out = os.path.join(export_dir, "verification_full.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  windows            : {report['n_windows']:,}")
    print(f"  argmax agreement   : {agree * 100:.3f} %   "
          f"[methodology bar: >= 98 %]  -> {'PASS' if agree >= 0.98 else 'FAIL'}")
    print(f"  disagreements      : {report['disagreements']:,}")
    print(f"  accuracy (in-sample, deployment model trained on all bearings):")
    print(f"    float {report['float_accuracy'] * 100:.2f} %   "
          f"int8 {report['int8_accuracy'] * 100:.2f} %")
    if report.get("disagreements_by_bearing"):
        print(f"  disagreements by bearing: {report['disagreements_by_bearing']}")
    print(f"\n  NOTE: the accuracy above is IN-SAMPLE and is not a generalisation")
    print(f"        estimate. The honest number is in runs/lobo_summary.json.")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
