"""
Stratified 5-fold cross-validation over windows -- the standard protocol in the
Paderborn literature, and the one to quote for a hackathon comparison against
published figures.

Produces the full set of numbers a slide needs: accuracy per fold, pooled
accuracy, macro-F1, per-class precision/recall/F1, and the confusion matrix.

WHAT THIS MEASURES, PLAINLY
---------------------------
Windows are split at random, so windows from the same recording and the same
bearing appear in both train and test.  That is what makes it comparable to the
published numbers on this dataset -- and it means the result describes accuracy
on bearings the model has already seen, not on a new one.

Use `scripts/02_train_lobo.py` for the leave-one-bearing-out figure.  Both are
real; they answer different questions.  Label whichever one you show.
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
from drivesentinel.features import load_cache
from drivesentinel.train import (_fit_one, channel_stats, get_device,
                                 macro_f1, predict_probs)


def stratified_folds(y: np.ndarray, k: int, seed: int):
    """Class-balanced random partition of window indices."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in folds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--seeds", type=int, default=1,
                    help="models averaged per fold")
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, meta = load_cache()
    device = get_device()
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).long().to(device)

    cfg = dict(C.TRAIN_CONFIG)
    cfg["n_seeds"] = args.seeds
    print(f"  cache {X.shape} | feature set {C.FEATURE_SET}")
    print(f"  protocol: stratified {args.folds}-fold over WINDOWS "
          f"(random split, not by bearing)")
    print(f"  epochs={cfg['epochs']}  models/fold={args.seeds}\n")

    folds = stratified_folds(y, args.folds, args.seed)
    truth_all, pred_all, per_fold = [], [], []

    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        i_tr = torch.from_numpy(tr).to(device)
        i_te = torch.from_numpy(te).to(device)

        mean, std = channel_stats(Xt[i_tr])
        Xf, yf = (Xt[i_tr] - mean) / std, yt[i_tr]
        Xte, yte = (Xt[i_te] - mean) / std, yt[i_te]

        counts = torch.bincount(yf, minlength=len(C.LABELS)).float()
        w = (counts.sum() / (len(C.LABELS) * counts.clamp_min(1))).to(device)

        probs = torch.zeros(Xte.shape[0], len(C.LABELS), device=device)
        for m in range(args.seeds):
            model, _, _ = _fit_one(Xf, yf, Xf[:0], yf[:0], w, cfg, device,
                                   args.seed + 1000 * m + 17 * k)
            probs += predict_probs(model, Xte, cfg["tta_shifts"])
        pred = (probs / args.seeds).argmax(1).cpu().numpy()
        truth = yte.cpu().numpy()

        acc = float((pred == truth).mean())
        per_fold.append(acc)
        truth_all.append(truth)
        pred_all.append(pred)
        print(f"  fold {k}: accuracy {acc:.4f}  ({len(te):,} test windows)",
              flush=True)

    t = np.concatenate(truth_all)
    p = np.concatenate(pred_all)
    n = len(C.LABELS)

    cm = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(t, p):
        cm[a, b] += 1

    per_class = {}
    for i, lab in enumerate(C.LABELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": float(prec), "recall": float(rec),
                          "f1": float(f1), "support": int(cm[i, :].sum())}

    out = {
        "protocol": f"stratified {args.folds}-fold over windows (random split)",
        "note": ("Windows from the same recording and bearing appear in both "
                 "train and test. Comparable to published Paderborn figures; "
                 "measures accuracy on bearings already seen, not on a new one."),
        "feature_set": C.FEATURE_SET,
        "accuracy": float((t == p).mean()),
        "accuracy_per_fold": [round(a, 4) for a in per_fold],
        "accuracy_std": float(np.std(per_fold, ddof=1)),
        "macro_f1": macro_f1(t, p, n),
        "majority_baseline": float(np.bincount(t).max() / t.size),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "labels": list(C.LABELS),
        "n_windows": int(t.size),
    }
    path = os.path.join(C.RUN_DIR, "shuffled_benchmark.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'=' * 62}")
    print(f"  STRATIFIED {args.folds}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 62}")
    print(f"  accuracy        {out['accuracy']:.4f}  "
          f"(+/- {out['accuracy_std']:.4f} across folds)")
    print(f"  macro F1        {out['macro_f1']:.4f}")
    print(f"  baseline        {out['majority_baseline']:.4f}")
    print(f"\n  {'class':12}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>10}")
    for lab, m in per_class.items():
        print(f"  {lab:12}{m['precision']:11.4f}{m['recall']:9.4f}"
              f"{m['f1']:9.4f}{m['support']:10,}")
    print(f"\n  confusion (rows = true, cols = predicted):")
    print("            " + "".join(f"{l:>13}" for l in C.LABELS))
    for lab, row in zip(C.LABELS, cm):
        print(f"  {lab:>9} " + "".join(f"{v:>13,}" for v in row))
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
