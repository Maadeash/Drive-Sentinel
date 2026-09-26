"""
Three framings of the same data, so the accuracy target is answered honestly.

  3class_lobo   healthy / inner / outer, leave-one-bearing-out   -- the hard,
                deployable number.
  binary_lobo   healthy vs damaged, leave-one-bearing-out        -- a genuinely
                easier and genuinely useful task: "is this bearing fine?" is the
                first-order product question, and the Good/Warning/Fault ladder
                needs it answered before location matters at all.
  3class_shuffled   healthy / inner / outer, windows split at RANDOM, ignoring
                bearing identity.  This is the protocol most published Paderborn
                results use, and it is why they report 99 %.  Included ONLY as a
                calibration point, clearly labelled, so the gap between it and
                the leave-one-bearing-out number is visible rather than
                mysterious. It must never be quoted as a generalisation figure.

Every framing uses the same cache, the same model and the same ensemble.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from drivesentinel import config as C
from drivesentinel import folds as F_
from drivesentinel.features import load_cache
from drivesentinel.train import (aggregation_curve, get_device, macro_f1,
                                 recording_vote, run_lobo, train_one_fold)


def shuffled_split_reference(X, y, meta, cfg, n_folds=5, seed=C.SEED):
    """
    The leaky number, computed deliberately.

    Windows are split at random with no regard to which bearing they came from,
    so the seven near-identical windows of one recording -- and the 560 windows
    of one bearing -- land on both sides of the split. The model can memorise
    bearing identity and score extremely well without having learned anything
    that transfers to a bearing it has never seen.
    """
    from drivesentinel.model import build_model
    from drivesentinel.train import _fit_one, channel_stats, predict_probs

    device = get_device()
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).long().to(device)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    folds = np.array_split(perm, n_folds)

    truth_all, pred_all = [], []
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        i_tr = torch.from_numpy(tr).to(device)
        i_te = torch.from_numpy(te).to(device)
        mean, std = channel_stats(Xt[i_tr])
        Xf, yf = (Xt[i_tr] - mean) / std, yt[i_tr]
        Xv, yv = Xf[:2048], yf[:2048]          # leaky by construction; the point
        Xte, yte = (Xt[i_te] - mean) / std, yt[i_te]
        counts = torch.bincount(yf, minlength=len(C.LABELS)).float()
        w = (counts.sum() / (len(C.LABELS) * counts.clamp_min(1))).to(device)
        model, _, _ = _fit_one(Xf, yf, Xv, yv, w, cfg, device, seed + k)
        pred = predict_probs(model, Xte).argmax(1).cpu().numpy()
        truth_all.append(yte.cpu().numpy())
        pred_all.append(pred)
        print(f"    shuffled fold {k}: acc={np.mean(pred == yte.cpu().numpy()):.4f}",
              flush=True)

    t, p = np.concatenate(truth_all), np.concatenate(pred_all)
    return {"window_acc": float((t == p).mean()),
            "macro_f1": macro_f1(t, p, len(C.LABELS)),
            "n_windows": int(t.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*",
                    default=["binary_lobo", "3class_shuffled"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, meta = load_cache()
    fold_defs = F_.make_folds(meta)
    F_.verify_folds(fold_defs, sorted(meta["bearing"].unique()))

    cfg = dict(C.TRAIN_CONFIG)
    cfg.update({"epochs": 20, "early_stop": False, "val_bearings_per_class": 1,
                "n_seeds": args.seeds, "tta_shifts": (-3, 0, 3)})
    print(f"  cache {X.shape} | feature set {C.FEATURE_SET} | cfg epochs="
          f"{cfg['epochs']} seeds={cfg['n_seeds']} tta={cfg['tta_shifts']}")

    out = {}

    if "binary_lobo" in args.tasks:
        print(f"\n{'=' * 70}\n  BINARY: healthy vs damaged, leave-one-bearing-out"
              f"\n{'=' * 70}", flush=True)
        # C.LABELS drives the metrics and the class weights, but the model's
        # output width lives in MODEL_CONFIG, which was evaluated at import.
        # Both have to move together or cross_entropy gets a 2-wide weight
        # vector against a 3-wide head.
        original_labels = list(C.LABELS)
        original_n = C.MODEL_CONFIG["n_classes"]
        y_bin = (y != C.LABEL_TO_INDEX["healthy"]).astype(np.int8)
        C.LABELS[:] = ["healthy", "damaged"]
        C.MODEL_CONFIG["n_classes"] = 2
        try:
            summary = run_lobo(X, y_bin, meta, fold_defs, cfg=cfg)
        finally:
            C.LABELS[:] = original_labels
            C.MODEL_CONFIG["n_classes"] = original_n
        out["binary_lobo"] = summary
        p = summary["pooled"]
        print(f"\n  -> binary LOBO: window {p['window_acc']:.4f}  "
              f"F1 {p['macro_f1']:.4f}  recording {p['recording_acc']:.4f}  "
              f"baseline {p['majority_baseline']:.4f}")
        print(f"     aggregation: {summary['aggregation_curve']}")

    if "3class_shuffled" in args.tasks:
        print(f"\n{'=' * 70}\n  LEAKY REFERENCE: 3-class, windows shuffled across"
              f" bearings\n  (this is the protocol that produces 99 % in the "
              f"literature)\n{'=' * 70}", flush=True)
        ref = shuffled_split_reference(X, y, meta, cfg)
        out["3class_shuffled_LEAKY"] = ref
        print(f"\n  -> shuffled 3-class: window {ref['window_acc']:.4f}  "
              f"F1 {ref['macro_f1']:.4f}   <-- NOT a generalisation estimate")

    path = os.path.join(C.RUN_DIR, "task_variants.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
