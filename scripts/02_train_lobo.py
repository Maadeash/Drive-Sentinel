"""
Leave-one-bearing-out sweep, then the deployment model.

The headline numbers are POOLED across folds.  Each bearing is held out exactly
once, so concatenating every fold's predictions yields one out-of-sample
prediction per window over the whole dataset.  Per-fold macro-F1 would be
meaningless: a single-bearing test set contains one class, so its F1 caps at
1/n_classes and its majority baseline is always 1.0.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from drivesentinel import config as C
from drivesentinel import folds as F_
from drivesentinel.features import load_cache
from drivesentinel.model import build_model, describe
from drivesentinel.train import run_lobo, train_deployment_model


def report(summary: dict) -> None:
    s, p = summary, summary["pooled"]
    print(f"\n  === LEAVE-ONE-BEARING-OUT, POOLED ({s['n_folds']} folds) ===")
    print(f"  window accuracy    {p['window_acc']:.4f}   ({p['n_windows']:,} windows)")
    print(f"  macro F1           {p['macro_f1']:.4f}")
    print(f"  per-recording acc  {p['recording_acc']:.4f}   "
          f"({p['n_recordings']:,} recordings)")
    print(f"  majority baseline  {p['majority_baseline']:.4f}")
    print(f"  per-bearing acc    {s['per_bearing_acc_mean']:.4f} +/- "
          f"{s['per_bearing_acc_std']:.4f}  (SEM {s['per_bearing_acc_sem']:.4f}, "
          f"n={s['n_folds']} bearings)")

    print("\n  confusion (rows = true, cols = predicted):")
    print("            " + "".join(f"{l:>13}" for l in s["labels"]))
    for lab, row in zip(s["labels"], s["confusion"]):
        print(f"  {lab:>9} " + "".join(f"{v:>13,}" for v in row))

    print("\n  by damage origin (healthy joins both as the reference class):")
    for k, v in s["by_damage_origin"].items():
        print(f"    {k:11} acc={v['window_acc']:.4f}  F1={v['macro_f1']:.4f}  "
              f"base={v['majority_baseline']:.4f}  ({v['n_bearings']} bearings)")

    print("\n  hardest bearings:")
    for f in sorted(s["folds"], key=lambda r: r["window_acc"])[:6]:
        print(f"    {f['test_bearing']:5} {f['test_label']:11} {f['origin']:10} "
              f"acc={f['window_acc']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None,
                    help="first N folds only (smoke test)")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--tag", default="lobo")
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, meta = load_cache()
    print(f"  cache: {X.shape} | {meta['bearing'].nunique()} bearings "
          f"| {meta['filename'].nunique():,} recordings")

    cfg = dict(C.TRAIN_CONFIG)
    if args.epochs:
        cfg["epochs"] = args.epochs

    fold_defs = F_.make_folds(meta)
    F_.verify_folds(fold_defs, sorted(meta["bearing"].unique()))
    print(f"  folds: {len(fold_defs)} leave-one-bearing-out, verified no overlap")
    if args.folds:
        fold_defs = fold_defs[:args.folds]

    print(describe(build_model()))

    summary = run_lobo(X, y, meta, fold_defs, cfg=cfg)
    summary["train_config"] = cfg
    out = os.path.join(C.RUN_DIR, f"{args.tag}_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    report(summary)
    print(f"\n  wrote {out}  ({summary['seconds'] / 60:.1f} min)")

    if args.no_deploy:
        return

    print("\n  training deployment model on ALL bearings (no holdout)...")
    model, stats = train_deployment_model(X, y, meta, cfg=cfg)
    ckpt = os.path.join(C.RUN_DIR, "deployment_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "norm_mean": stats["mean"], "norm_std": stats["std"],
        "val_bearings": stats["val_bearings"], "val_acc": stats["val_acc"],
        "labels": C.LABELS, "model_config": C.MODEL_CONFIG,
        "lobo_pooled": summary["pooled"],
    }, ckpt)
    print(f"  val_acc {stats['val_acc']:.4f} on held-out bearings "
          f"{stats['val_bearings']}")
    print(f"  wrote {ckpt}")


if __name__ == "__main__":
    main()
