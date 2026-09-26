"""
D2 CNN EXPERIMENT -- does a CNN on the full harmonic spectrum beat the scalar GBM?

STATUS: POST-HOC EXPERIMENT. Run 2026-09-19, after the gradient-boosting result
was known. Declared as such in docs/claims_audit.md. The acceptance criteria below
were fixed BEFORE the run and are not to be changed after seeing the numbers.

HYPOTHESIS
----------
A 1-D CNN on the f/f_e harmonic spectrum finds inter-coil vs inter-turn structure
that 23 scalar features miss.

ACCEPTANCE CRITERIA -- FIXED BEFORE THE RUN, NOT NEGOTIABLE AFTERWARDS
----------------------------------------------------------------------
  1. V5 accuracy must beat 0.750 -- the i0rel_residual 1-NN session-only baseline
     -- to count as evidence of winding diagnosis at all. That scalar is zero by
     Kirchhoff in a three-wire machine and therefore cannot contain winding
     information; anything that fails to beat it is not diagnosing windings.
  2. V5 accuracy must beat the GBM's 0.6251 by MORE THAN THE SEED SPREAD to count
     as an improvement.
  3. If either fails, this is a NEGATIVE RESULT and the GBM stays as the shipped
     branch.

THE RECIPE, DECLARED HERE BEFORE THE RUN
-----------------------------------------
One recipe. No hyperparameter search, no sweep, no early stopping on a test fold.
Three seeds, mean +/- std reported. Anything else would be tuning against the
acceptance criteria, which is the specific thing this experiment must not do.

  input        f/f_e spectrum, 0-20 orders, 512 bins, log-magnitude,
               median-normalised -- the existing feature contract
  architecture drivesentinel.model.OrderSpectrumCNN, conv stages
               (16,9,2) (32,7,2) (64,5,2) (64,3,2), GAP, linear.
               Same shape as the bearing model, 1 input channel, 2 classes.
  optimiser    AdamW, lr 3e-3, weight decay 1e-4, OneCycle, 30 epochs, batch 256
  loss         cross-entropy, label smoothing 0.05, class-balanced weights
  normalise    per-channel mean/std from the FIT SET ONLY
  seeds        0, 1, 2
  ensembling   none. single model, no TTA.

WHAT IS HELD IDENTICAL TO THE GBM RUN
--------------------------------------
Task (inter_coil vs inter_turn, healthy excluded), adapter (polarity corrected,
NO gain calibration), windows (1.0 s / 0.5 s hop), splits (V1 leave-one-motor-out,
V5 leave-one-session-out with 2022-08-11 excluded as degenerate, V3 shuffled).
Only the features and the model change.

THE TIER DOES NOT MOVE
----------------------
B-S4 is INDICATIVE at 2 validation groups against a floor of 3, whatever this
scores. A better number does not buy authority; the floor is on GROUPS as well as
on macro-F1, and it was pre-registered before any branch existed.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.adapters import kaist_pmsm as K
from drivesentinel.branches import winding as W
from drivesentinel.common import splits as SP
from drivesentinel.common.schema import windows as make_windows

OUT_DIR = os.path.join(C.ARTIFACT_DIR, "multistage", "winding")
CACHE = os.path.join(OUT_DIR, "spectra_cnn.npz")

N_BINS = 512
ORDER_MAX = 20.0
SEEDS = (0, 1, 2)
DEGENERATE_SESSION = "2022-08-11"

RECIPE = {
    "architecture": "OrderSpectrumCNN (16,9,2)(32,7,2)(64,5,2)(64,3,2) + GAP + linear",
    "in_channels": 1, "n_classes": 2, "n_bins": N_BINS, "order_max": ORDER_MAX,
    "epochs": 30, "batch_size": 256, "lr": 3e-3, "weight_decay": 1e-4,
    "label_smoothing": 0.05, "schedule": "OneCycle", "class_weights": True,
    "seeds": list(SEEDS), "ensemble": False, "tta": False,
    "declared_before_run": True,
}

ACCEPTANCE = {
    "criterion_1_beat_session_baseline": 0.750,
    "criterion_2_beat_gbm_by_more_than_seed_spread": 0.6251,
    "gbm_v5_accuracy": 0.6251,
    "gbm_v5_macro_f1": 0.6250,
    "session_only_baseline": 0.750,
    "declared_before_run": True,
}


# ===========================================================================
# spectra
# ===========================================================================

def spectrum(win: np.ndarray, fs: float, f_elec: float = K.F_ELEC_HZ) -> np.ndarray:
    """
    One window (3, n) -> (N_BINS,) on an f/f_e axis, log-magnitude, median-normalised.

    Mean over the three phases, then resampled onto the order axis. Median
    normalisation with a peak-relative floor -- the same guard the scalar branch
    needed, for the same reason: a spectrum whose energy sits in one or two orders
    drives the median to the floating-point noise and destroys scale invariance.
    """
    x = win.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(x, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    orders = freqs / f_elec

    grid = np.linspace(0.0, ORDER_MAX, N_BINS, endpoint=False)
    band = np.interp(grid, orders, mag, left=0.0, right=0.0)

    peak = float(band.max())
    scale = max(float(np.median(band)), 1e-6 * peak)
    return np.log1p(band / scale).astype(np.float32) if scale > 0 else band.astype(np.float32)


def build_cache(root=None, rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        d = dict(np.load(CACHE, allow_pickle=False))
        print(f"  spectra     : {d['X'].shape} (cached)")
        return d

    t0 = time.time()
    recs = [r for r in K.load(root) if r.label != "healthy"]
    X, y, group, session, sev, rec_id = [], [], [], [], [], []
    for r in recs:
        w = make_windows(r, W.PHASES, W.WINDOW_S, W.HOP_S)
        fs = r.fs[W.PHASES[0]]
        S = np.stack([spectrum(w[i], fs) for i in range(w.shape[0])])
        X.append(S)
        n = S.shape[0]
        y += [r.label] * n
        group += [r.group] * n
        session += [r.condition["session"]] * n
        sev += [r.condition["severity_pct"]] * n
        rec_id += [r.provenance["current_file"]] * n

    d = dict(X=np.concatenate(X), y=np.asarray(y), group=np.asarray(group),
             session=np.asarray(session),
             severity=np.asarray(sev, dtype=np.float64),
             recording=np.asarray(rec_id))
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(CACHE, **d)
    print(f"  spectra     : {d['X'].shape} built in {time.time()-t0:.1f}s -> {CACHE}")
    return d


# ===========================================================================
# model
# ===========================================================================

def fit_predict(Xtr, ytr, Xte, labels, seed):
    import torch
    import torch.nn as nn
    from drivesentinel.model import OrderSpectrumCNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")

    idx = {c: i for i, c in enumerate(labels)}
    ytr_i = torch.tensor([idx[v] for v in ytr], dtype=torch.long)
    Xtr_t = torch.from_numpy(Xtr).float().unsqueeze(1)
    Xte_t = torch.from_numpy(Xte).float().unsqueeze(1)

    # Fit-set statistics only.
    mean = Xtr_t.mean(dim=(0, 2), keepdim=True)
    std = Xtr_t.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    Xtr_t = (Xtr_t - mean) / std
    Xte_t = (Xte_t - mean) / std

    model = OrderSpectrumCNN(in_channels=RECIPE["in_channels"],
                             n_classes=len(labels),
                             conv_stages=C.MODEL_CONFIG["conv_stages"]).to(dev)

    counts = torch.bincount(ytr_i, minlength=len(labels)).float()
    wts = (counts.sum() / (len(labels) * counts.clamp_min(1))).to(dev)
    lossf = nn.CrossEntropyLoss(weight=wts,
                                label_smoothing=RECIPE["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    bs, ep = RECIPE["batch_size"], RECIPE["epochs"]
    nb = max(1, (len(Xtr_t) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RECIPE["lr"],
                                                total_steps=ep * nb)
    model.train()
    for _ in range(ep):
        perm = torch.randperm(len(Xtr_t))
        for b in range(nb):
            sel = perm[b * bs:(b + 1) * bs]
            if not len(sel):
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[sel]), ytr_i[sel])
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    return np.asarray([labels[i] for i in pred])


def scores(truth, pred, labels):
    truth, pred = np.asarray(truth), np.asarray(pred)
    acc = float((truth == pred).mean()) if truth.size else float("nan")
    f1s = []
    for c in labels:
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    base = (float(Counter(truth.tolist()).most_common(1)[0][1] / truth.size)
            if truth.size else float("nan"))
    return dict(accuracy=acc, macro_f1=float(np.mean(f1s)),
                majority_baseline=base, n=int(truth.size))


# ===========================================================================
# protocols -- identical splits to the GBM run
# ===========================================================================

def run_grouped(D, key, labels, seed, exclude=()):
    g = D[key]
    keep = ~np.isin(g, list(exclude))
    t_all, p_all, folds = [], [], []
    for f in SP.iter_checked(SP.leave_one_group_out(g), g):
        held = sorted(set(g[f["test_idx"]].tolist()))
        te = f["test_idx"][keep[f["test_idx"]]]
        if te.size == 0 or len(set(D["y"][te].tolist())) < 2:
            folds.append(dict(held_out=held, excluded=True))
            continue
        pred = fit_predict(D["X"][f["train_idx"]], D["y"][f["train_idx"]],
                           D["X"][te], labels, seed)
        sc = scores(D["y"][te], pred, labels)
        sc.update(held_out=held, excluded=False)
        folds.append(sc)
        t_all.append(D["y"][te]); p_all.append(pred)
    pooled = (scores(np.concatenate(t_all), np.concatenate(p_all), labels)
              if t_all else dict(accuracy=float("nan"), macro_f1=float("nan"),
                                 majority_baseline=float("nan"), n=0))
    return dict(folds=folds, pooled=pooled,
                n_groups_scored=sum(1 for f in folds if not f.get("excluded")))


def run_shuffled(D, labels, seed, n_folds=5):
    t_all, p_all = [], []
    for f in SP.shuffled(len(D["X"]), n_folds=n_folds, seed=seed, labels=D["y"]):
        pred = fit_predict(D["X"][f["train_idx"]], D["y"][f["train_idx"]],
                           D["X"][f["test_idx"]], labels, seed)
        t_all.append(D["y"][f["test_idx"]]); p_all.append(pred)
    return dict(pooled=scores(np.concatenate(t_all), np.concatenate(p_all), labels),
                leaky=True,
                caveat="(leaky reference) windows split at random, ignoring session and motor")


# ===========================================================================
# main
# ===========================================================================

def agg(runs, field="accuracy"):
    v = [r["pooled"][field] for r in runs]
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1) if len(v) > 1 else 0.0),
                values=[float(x) for x in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    D = build_cache(rebuild=a.rebuild)
    labels = list(W.LABELS)
    print(f"  task        : {labels[0]} vs {labels[1]}")
    print(f"  windows     : {dict(Counter(D['y'].tolist()))}")
    print(f"  recipe      : {RECIPE['epochs']} epochs, lr {RECIPE['lr']}, "
          f"batch {RECIPE['batch_size']}, seeds {list(SEEDS)}")
    print(f"  ACCEPTANCE  : V5 > {ACCEPTANCE['criterion_1_beat_session_baseline']} "
          f"(session baseline) AND V5 > {ACCEPTANCE['gbm_v5_accuracy']} + seed spread "
          f"(GBM)\n")

    v1, v5, v3 = [], [], []
    for s in SEEDS:
        ts = time.time()
        r1 = run_grouped(D, "group", labels, s)
        r5 = run_grouped(D, "session", labels, s, exclude=(DEGENERATE_SESSION,))
        r3 = run_shuffled(D, labels, s)
        v1.append(r1); v5.append(r5); v3.append(r3)
        print(f"  seed {s}: V1 {r1['pooled']['accuracy']:.4f}  "
              f"V5 {r5['pooled']['accuracy']:.4f}  "
              f"V3 {r3['pooled']['accuracy']:.4f} (leaky)   "
              f"[{time.time()-ts:.0f}s]", flush=True)

    res = {
        "experiment": "d2_cnn",
        "status": "POST-HOC -- run after the gradient-boosting result was known",
        "task": f"{labels[0]} vs {labels[1]}",
        "recipe": RECIPE, "acceptance": ACCEPTANCE,
        "n_windows": int(D["X"].shape[0]),
        "V1_leave_one_motor_out": {"per_seed": v1, "accuracy": agg(v1),
                                   "macro_f1": agg(v1, "macro_f1")},
        "V5_leave_one_session_out": {"per_seed": v5, "accuracy": agg(v5),
                                     "macro_f1": agg(v5, "macro_f1"),
                                     "n_groups_scored": v5[0]["n_groups_scored"]},
        "V3_shuffled_LEAKY": {"per_seed": v3, "accuracy": agg(v3),
                              "macro_f1": agg(v3, "macro_f1")},
    }

    a5, s5 = res["V5_leave_one_session_out"]["accuracy"]["mean"], \
        res["V5_leave_one_session_out"]["accuracy"]["std"]
    c1 = a5 > ACCEPTANCE["criterion_1_beat_session_baseline"]
    c2 = (a5 - ACCEPTANCE["gbm_v5_accuracy"]) > s5
    res["verdict"] = {
        "v5_accuracy_mean": a5, "v5_accuracy_std": s5,
        "criterion_1_beats_session_baseline": bool(c1),
        "criterion_1_margin": float(a5 - ACCEPTANCE["criterion_1_beat_session_baseline"]),
        "criterion_2_beats_gbm_by_more_than_seed_spread": bool(c2),
        "criterion_2_margin": float(a5 - ACCEPTANCE["gbm_v5_accuracy"]),
        "seed_spread": s5,
        "outcome": "POSITIVE" if (c1 and c2) else "NEGATIVE RESULT",
        "shipped_branch": "CNN" if (c1 and c2) else "gradient boosting (unchanged)",
        "tier_unchanged": "INDICATIVE -- 2 validation groups against a floor of 3, "
                          "whatever this scores. The floor is on GROUPS as well as "
                          "macro-F1 and was pre-registered before any branch existed.",
    }
    res["seconds"] = round(time.time() - t0, 1)

    out = os.path.join(OUT_DIR, "winding_cnn_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print(f"\n  === RESULT ===")
    for k, r, gbm in (("V1", res["V1_leave_one_motor_out"], 0.3784),
                      ("V5", res["V5_leave_one_session_out"], 0.6251),
                      ("V3", res["V3_shuffled_LEAKY"], 0.9990)):
        print(f"    {k}  CNN {r['accuracy']['mean']:.4f} +/- {r['accuracy']['std']:.4f}"
              f"   GBM {gbm:.4f}")
    print(f"\n    criterion 1 (V5 > 0.750 session baseline): "
          f"{'PASS' if c1 else 'FAIL'}  ({a5:.4f})")
    print(f"    criterion 2 (V5 > GBM + seed spread):      "
          f"{'PASS' if c2 else 'FAIL'}  "
          f"(margin {a5 - ACCEPTANCE['gbm_v5_accuracy']:+.4f} vs spread {s5:.4f})")
    print(f"    OUTCOME: {res['verdict']['outcome']}")
    print(f"    SHIPPED: {res['verdict']['shipped_branch']}")
    print(f"  wrote {out}  ({res['seconds']/60:.1f} min)")


if __name__ == "__main__":
    main()
