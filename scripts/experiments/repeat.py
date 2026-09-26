"""
Run the SAME config several times with different base seeds.

The point is not to find a good run.  It is to measure how much a run moves on
its own, so a headline number comes with a spread instead of resting on one
draw.  Everything in this project that looked like a finding and then evaporated
-- "patience 12 beats 20", "shorter schedules generalise" -- would have been
caught by doing this first.

Reports mean, standard deviation and min-max across repeats.  Quote the mean.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel import folds as F_
from drivesentinel.features import load_cache
from drivesentinel.train import run_lobo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="repeat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, meta = load_cache()
    fold_defs = F_.make_folds(meta)
    F_.verify_folds(fold_defs, sorted(meta["bearing"].unique()))

    cfg = dict(C.TRAIN_CONFIG)
    print(f"  cache {X.shape} | feature set {C.FEATURE_SET}")
    print(f"  config: epochs={cfg['epochs']} seeds={cfg['n_seeds']} "
          f"tta={cfg['tta_shifts']} val={cfg['val_bearings_per_class']}")
    print(f"  augment: {dict(C.AUGMENT_CONFIG)}")
    print(f"  repeats: {args.repeats} (different base seeds)\n")

    runs = []
    for r in range(args.repeats):
        seed = C.SEED + 7919 * r        # a prime stride, so seeds do not collide
        print(f"{'=' * 70}\n  REPEAT {r + 1}/{args.repeats}  base seed {seed}\n"
              f"{'=' * 70}", flush=True)
        s = run_lobo(X, y, meta, fold_defs, cfg=cfg, seed=seed)
        s["base_seed"] = seed
        runs.append(s)
        p = s["pooled"]
        print(f"\n  -> repeat {r + 1}: window {p['window_acc']:.4f}  "
              f"F1 {p['macro_f1']:.4f}  recording {p['recording_acc']:.4f}",
              flush=True)

    def stats(key):
        v = np.array([r["pooled"][key] for r in runs])
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max()

    out = {
        "n_repeats": args.repeats,
        "train_config": cfg,
        "augment_config": dict(C.AUGMENT_CONFIG),
        "feature_set": C.FEATURE_SET,
        "runs": [{"base_seed": r["base_seed"], **r["pooled"]} for r in runs],
        "summary": {k: dict(zip(("mean", "std", "min", "max"), stats(k)))
                    for k in ("window_acc", "macro_f1", "recording_acc")},
    }
    path = args.out or os.path.join(C.RUN_DIR, f"{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print(f"\n\n{'=' * 70}")
    print(f"  {args.repeats} repeats of an identical config")
    print(f"  {'metric':20}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    print(f"  {'-' * 56}")
    for k in ("window_acc", "macro_f1", "recording_acc"):
        m, sd, lo, hi = stats(k)
        print(f"  {k:20}{m:9.4f}{sd:9.4f}{lo:9.4f}{hi:9.4f}")
    print(f"\n  baseline {runs[0]['pooled']['majority_baseline']:.4f}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
