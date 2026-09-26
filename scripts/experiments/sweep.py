"""
Compare training recipes over the FULL 29-fold leave-one-bearing-out protocol.

Eight folds is not enough to compare on -- the first eight bearings are six
healthy plus two outer-race, so the pooled number is dominated by one class and
says nothing about the others. Every config here runs all 29 folds.

Nothing about the protocol changes between configs: same folds, same cache, same
pooled reporting. Only the recipe moves.
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from drivesentinel import config as C
from drivesentinel import folds as F_
from drivesentinel.features import load_cache
from drivesentinel.train import run_lobo


# Recipe deltas applied on top of C.TRAIN_CONFIG / C.AUGMENT_CONFIG.
WEAK_AUG = {"order_jitter_bins": 3, "amplitude_scale": 0.15,
            "channel_dropout": 0.0, "noise_std": 0.02, "prob": 0.5}
STRONG_AUG = {"order_jitter_bins": 4, "amplitude_scale": 0.25,
              "channel_dropout": 0.15, "noise_std": 0.05, "prob": 0.8}

# All recipes hold val_bearings_per_class at 1.  With only 6 healthy bearings,
# leave-one-bearing-out already removes one, and holding out 2 per class for
# validation would leave just 3 healthy bearings to fit on -- starving the
# smallest class to buy a slightly steadier stopping signal.  That trade is
# tested explicitly by the val2_* recipe rather than assumed either way.
# LOW-VARIANCE REGIME.
#
# Every recipe below sets val_bearings_per_class = 0.  Removing the inner
# validation split cut the run-to-run standard deviation from 0.1082 to 0.0018
# -- a 60x reduction -- because the base seed was choosing which bearings got
# held out, and with 6 healthy specimens that choice dominated everything else.
#
# That matters for reading this table: differences above ~0.005 are now real.
# Earlier sweeps in this project compared single draws against a +/-0.11 noise
# floor and their conclusions cannot be trusted -- which is why the 5-channel
# feature set is retested here rather than left rejected.
RECIPES = {
    "base":            {"train": {}, "aug": STRONG_AUG},
    "weak_aug":        {"train": {}, "aug": WEAK_AUG},
    "no_ensemble":     {"train": {"n_seeds": 1, "tta_shifts": (0,)}, "aug": STRONG_AUG},
    "tta_only":        {"train": {"n_seeds": 1}, "aug": STRONG_AUG},
    "epochs40":        {"train": {"epochs": 40}, "aug": STRONG_AUG},
    "epochs60":        {"train": {"epochs": 60}, "aug": STRONG_AUG},
    "seeds5":          {"train": {"n_seeds": 5}, "aug": STRONG_AUG},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="*", default=None,
                    help=f"subset of {list(RECIPES)}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--feature-set", default=None, choices=["v1", "v2"],
                    help="which cache to sweep on; defaults to config.FEATURE_SET")
    args = ap.parse_args()

    # Pin the feature set explicitly.  Sweeps run days apart on different caches
    # are not comparable, and the shape printed below is the only thing that
    # would otherwise reveal which one was used.
    if args.feature_set and args.feature_set != C.FEATURE_SET:
        # Also export it, so any subprocess this run spawns agrees.
        os.environ["DRIVESENTINEL_FEATURE_SET"] = args.feature_set
        C.FEATURE_SET = args.feature_set
        C.N_INPUT_CHANNELS = C._CHANNELS[args.feature_set]
        C.MODEL_CONFIG["in_channels"] = C.N_INPUT_CHANNELS
        C.CACHE_PATH, C.CACHE_META_PATH = C._cache_paths(args.feature_set)

    C.ensure_dirs()
    X, y, meta = load_cache()
    fold_defs = F_.make_folds(meta)
    F_.verify_folds(fold_defs, sorted(meta["bearing"].unique()))
    print(f"  feature set {C.FEATURE_SET} | cache {X.shape} | {len(fold_defs)} folds "
          f"| baseline {__import__('numpy').bincount(y).max() / len(y):.4f}")

    names = args.recipes or list(RECIPES)
    base_aug = copy.deepcopy(C.AUGMENT_CONFIG)
    results = {}

    for name in names:
        spec = RECIPES[name]
        cfg = dict(C.TRAIN_CONFIG)
        cfg.update(spec["train"])
        # AUGMENT_CONFIG is read from the module inside augment(), so patch it
        C.AUGMENT_CONFIG.clear()
        C.AUGMENT_CONFIG.update(spec["aug"])

        print(f"\n{'=' * 70}\n  RECIPE {name}\n  train={spec['train']}\n"
              f"  aug={spec['aug']}\n{'=' * 70}", flush=True)
        t0 = time.time()
        summary = run_lobo(X, y, meta, fold_defs, cfg=cfg)
        summary["recipe"] = name
        summary["train_config"] = cfg
        summary["augment_config"] = dict(spec["aug"])
        summary["feature_set"] = C.FEATURE_SET
        summary["cache_shape"] = list(X.shape)
        results[name] = summary

        p = summary["pooled"]
        print(f"\n  -> {name}: window {p['window_acc']:.4f}  "
              f"F1 {p['macro_f1']:.4f}  recording {p['recording_acc']:.4f}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

        out = args.out or os.path.join(C.RUN_DIR, "sweep.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)

    C.AUGMENT_CONFIG.clear()
    C.AUGMENT_CONFIG.update(base_aug)

    print(f"\n\n{'=' * 78}")
    print(f"  {'recipe':22}{'window':>9}{'macroF1':>9}{'recording':>11}"
          f"{'per-bearing':>14}{'min':>7}")
    print(f"  {'-' * 74}")
    for name, s in sorted(results.items(),
                          key=lambda kv: -kv[1]["pooled"]["window_acc"]):
        p = s["pooled"]
        print(f"  {name:22}{p['window_acc']:9.4f}{p['macro_f1']:9.4f}"
              f"{p['recording_acc']:11.4f}"
              f"{s['per_bearing_acc_mean']:8.3f}+-{s['per_bearing_acc_std']:.3f}"
              f"{s['seconds'] / 60:7.1f}")
    print(f"  baseline               {results[names[0]]['pooled']['majority_baseline']:.4f}")


if __name__ == "__main__":
    main()
