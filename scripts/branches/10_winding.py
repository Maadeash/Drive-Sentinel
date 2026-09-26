"""
B-S4 winding branch: feature cache + every protocol, each reported separately.

PROTOCOLS
---------
  V4  BATCH CONTROL -- the headline. Predict acquisition session / batch / DAQ
      chassis from the SAME features, instead of the fault class. Plus a
      model-free probe: 1-NN on i0rel_residual alone, a scalar that is zero by
      Kirchhoff in a three-wire machine and therefore cannot contain winding
      information.
  V1  leave-one-motor-out, 3 folds by kW rating.
  V5  leave-one-session-out over the 3 acquisition days. 2 usable folds; the
      2022-08-11 fold is pure inter_coil and is excluded from scoring.
  V3  shuffled windows -- the leaky reference.

The task is inter_coil vs inter_turn. There is no healthy number here, and there
must not be one: all three healthy recordings are from 2022-01-25, so the class is
perfectly confounded with session (docs/data_notes_d2.md §7D).

Every card carries the session-only baseline. A fault-class score that does not
clearly beat what session identity alone buys is not evidence of winding diagnosis.
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
from drivesentinel.common import splits as S

OUT_DIR = os.path.join(C.ARTIFACT_DIR, "multistage", "winding")
CACHE = os.path.join(OUT_DIR, "features.npz")

DEGENERATE_SESSION = "2022-08-11"      # pure inter_coil; cannot score a 2-class task


# ===========================================================================
# model
# ===========================================================================

def _model(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_depth=4,
        l2_regularization=1.0, random_state=seed)


def _fit_predict(Xtr, ytr, Xte, seed=0):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)                  # fit-set statistics only
    m = _model(seed).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def _scores(truth, pred, labels):
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
    base = float(Counter(truth.tolist()).most_common(1)[0][1] / truth.size) if truth.size else float("nan")
    return dict(accuracy=acc, macro_f1=float(np.mean(f1s)),
                per_class_f1=dict(zip(labels, [float(f) for f in f1s])),
                majority_baseline=base, n=int(truth.size))


def _balance(y, groups, folds, group_arr):
    out = []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        out.append({
            "fold": f["fold"],
            "held_out": sorted(set(np.asarray(group_arr)[te].tolist())),
            "train_windows": dict(Counter(np.asarray(y)[tr].tolist())),
            "test_windows": dict(Counter(np.asarray(y)[te].tolist())),
        })
    return out


# ===========================================================================
# protocols
# ===========================================================================

def run_grouped(D, group_key, labels, name, exclude_groups=(), seed=0):
    """Leave-one-group-out over `group_key`, with named groups excluded."""
    X, y = D["X"], D["y"]
    g = D[group_key]
    keep = ~np.isin(g, list(exclude_groups))
    folds = S.leave_one_group_out(g)
    per_fold, truth_all, pred_all = [], [], []

    for f in S.iter_checked(folds, g):
        held = sorted(set(g[f["test_idx"]].tolist()))
        te = f["test_idx"][keep[f["test_idx"]]]
        tr = f["train_idx"]
        if te.size == 0:
            per_fold.append(dict(fold=f["fold"], held_out=held, excluded=True,
                                 reason="group excluded from scoring"))
            continue
        te_labels = set(y[te].tolist())
        if len(te_labels) < 2:
            per_fold.append(dict(fold=f["fold"], held_out=held, excluded=True,
                                 reason=f"test set has one class only: {sorted(te_labels)}",
                                 test_windows=dict(Counter(y[te].tolist()))))
            continue
        pred = _fit_predict(X[tr], y[tr], X[te], seed)
        sc = _scores(y[te], pred, labels)
        sc.update(fold=f["fold"], held_out=held, excluded=False,
                  train_windows=dict(Counter(y[tr].tolist())),
                  test_windows=dict(Counter(y[te].tolist())))
        per_fold.append(sc)
        truth_all.append(y[te]); pred_all.append(pred)

    pooled = (_scores(np.concatenate(truth_all), np.concatenate(pred_all), labels)
              if truth_all else dict(accuracy=float("nan"), macro_f1=float("nan"),
                                     per_class_f1={}, majority_baseline=float("nan"), n=0))
    return dict(protocol=name, scheme=f"leave-one-{group_key}-out",
                group_key=group_key, labels=list(labels),
                excluded_groups=list(exclude_groups),
                n_groups_scored=sum(1 for p in per_fold if not p.get("excluded")),
                folds=per_fold, pooled=pooled)


def run_shuffled(D, labels, seed=0, n_folds=5):
    X, y = D["X"], D["y"]
    truth_all, pred_all = [], []
    for f in S.shuffled(len(X), n_folds=n_folds, seed=seed, labels=y):
        pred = _fit_predict(X[f["train_idx"]], y[f["train_idx"]], X[f["test_idx"]], seed)
        truth_all.append(y[f["test_idx"]]); pred_all.append(pred)
    sc = _scores(np.concatenate(truth_all), np.concatenate(pred_all), labels)
    return dict(protocol="V3", scheme="shuffled-windows", leaky=True,
                caveat="(leaky reference) windows split at random, ignoring session and motor",
                labels=list(labels), pooled=sc)


def run_batch_control(D, seed=0):
    """
    V4 -- the headline.

    Three model-based controls (predict session / batch / chassis from the SAME
    winding features, under leave-one-motor-out so the model cannot simply
    memorise), plus the model-free probe.
    """
    X = D["X"]
    out = {"protocol": "V4", "scheme": "batch control", "targets": {}}

    for target in ("session", "batch", "chassis"):
        t = D[target]
        labels = sorted(set(t.tolist()))
        truth_all, pred_all = [], []
        for f in S.iter_checked(S.leave_one_group_out(D["group"]), D["group"]):
            tr, te = f["train_idx"], f["test_idx"]
            if len(set(t[tr].tolist())) < 2:
                continue
            pred = _fit_predict(X[tr], t[tr], X[te], seed)
            truth_all.append(t[te]); pred_all.append(pred)
        if truth_all:
            out["targets"][target] = _scores(np.concatenate(truth_all),
                                             np.concatenate(pred_all), labels)
        else:
            out["targets"][target] = dict(note="no fold had >1 target class in training")

    # Same features, shuffled split -- how separable is session at all?
    for target in ("session", "batch", "chassis"):
        t = D[target]
        labels = sorted(set(t.tolist()))
        truth_all, pred_all = [], []
        for f in S.shuffled(len(X), n_folds=5, seed=seed, labels=t):
            pred = _fit_predict(X[f["train_idx"]], t[f["train_idx"]], X[f["test_idx"]], seed)
            truth_all.append(t[f["test_idx"]]); pred_all.append(pred)
        out["targets"][f"{target}_shuffled_LEAKY"] = _scores(
            np.concatenate(truth_all), np.concatenate(pred_all), labels)
    return out


def run_residual_probe(recs):
    """
    Model-free. 1-NN leave-one-recording-out on i0rel_residual alone.

    That scalar is zero by Kirchhoff in a three-wire machine, so it cannot carry
    winding information. Recovering session from it is a stronger and plainer
    demonstration than a trained model doing the same thing.
    """
    faulty = [r for r in recs if r.label != "healthy"]
    v = np.array([[r.provenance["i0rel_residual"]] for r in faulty])
    tgt = {
        "session": np.array([r.condition["session"] for r in faulty]),
        "batch": np.array([r.condition["batch"] for r in faulty]),
        "chassis": np.array([r.condition["daq_chassis"] for r in faulty]),
        "fault_class": np.array([r.label for r in faulty]),
    }
    motors = np.array([r.group for r in faulty])

    def loo(x, y):
        ok = 0
        for i in range(len(x)):
            m = np.ones(len(x), bool); m[i] = False
            ok += (y[m][np.linalg.norm(x[m] - x[i], axis=1).argmin()] == y[i])
        return float(ok / len(x))

    def baseline(y):
        return float(Counter(y.tolist()).most_common(1)[0][1] / len(y))

    out = {"protocol": "V4-probe", "scheme": "1-NN on i0rel_residual (model-free)",
           "note": ("i0rel_residual is |I0|/max(|I1|,|I2|) after the polarity fix. "
                    "In a three-wire machine this is zero by Kirchhoff whatever the "
                    "winding is doing, so it carries no winding information by "
                    "construction."),
           "n_recordings": len(faulty), "subsets": {}}
    for sub, mask in (("all_motors", np.ones(len(faulty), bool)),
                      ("1000W_1500W_only", np.isin(motors, ["1000W", "1500W"])),
                      ("3000W_only", motors == "3000W")):
        if mask.sum() < 3:
            continue
        out["subsets"][sub] = {
            k: dict(loo_1nn_accuracy=loo(v[mask], tgt[k][mask]),
                    majority_baseline=baseline(tgt[k][mask]))
            for k in tgt
        }
        out["subsets"][sub]["n"] = int(mask.sum())
    return out


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature cache")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    recs = K.load()
    print(K.grid_report(K.index(os.path.join(C.PROJECT_ROOT, "data_ext", "kaist_pmsm"))))

    if a.rebuild or not os.path.exists(CACHE):
        t0 = time.time()
        D = W.build_dataset(recs, drop_healthy=True)
        np.savez_compressed(CACHE, **D)
        print(f"\n  features    : {D['X'].shape} in {time.time()-t0:.1f}s -> {CACHE}")
    else:
        D = dict(np.load(CACHE, allow_pickle=False))
        print(f"\n  features    : {D['X'].shape} (cached)")

    labels = list(W.LABELS)
    print(f"  task        : {labels[0]} vs {labels[1]}  (healthy EXCLUDED -- see §7D)")
    print(f"  windows     : {dict(Counter(D['y'].tolist()))}")
    print(f"  motors      : {dict(Counter(D['group'].tolist()))}")
    print(f"  sessions    : {dict(Counter(D['session'].tolist()))}")

    results = {
        "branch": "winding",
        "dataset": "kaist_pmsm",
        "stage": "S4",
        "task": f"{labels[0]} vs {labels[1]}",
        "healthy_status": "NOT MEASURABLE",
        "healthy_reason": (
            "All three healthy recordings are from 2022-01-25 (3/0/0 across the "
            "three acquisition days), so healthy-vs-faulty is perfectly confounded "
            "with acquisition session. Under leave-one-session-out, holding out "
            "2022-01-25 leaves training with no healthy recordings, and holding out "
            "either other day leaves the test set with none. No session-aware "
            "protocol on this dataset can evaluate it. See docs/data_notes_d2.md 7D."
        ),
        "n_windows": int(D["X"].shape[0]),
        "feature_names": list(map(str, D["feature_names"])),
        "window_s": W.WINDOW_S, "hop_s": W.HOP_S,
    }

    print("\n  === V4 batch control (headline) ===", flush=True)
    results["V4"] = run_batch_control(D, a.seed)
    for k, v in results["V4"]["targets"].items():
        if "accuracy" in v:
            print(f"    predict {k:26} acc {v['accuracy']:.4f}  "
                  f"macroF1 {v['macro_f1']:.4f}  (baseline {v['majority_baseline']:.4f})")

    print("\n  === V4 model-free residual probe ===", flush=True)
    results["V4_probe"] = run_residual_probe(recs)
    for sub, d in results["V4_probe"]["subsets"].items():
        print(f"    {sub} (n={d['n']}):")
        for k in ("session", "batch", "chassis", "fault_class"):
            print(f"       -> {k:12} {d[k]['loo_1nn_accuracy']:.3f}  "
                  f"(baseline {d[k]['majority_baseline']:.3f})")

    print("\n  === V1 leave-one-motor-out ===", flush=True)
    results["V1"] = run_grouped(D, "group", labels, "V1", seed=a.seed)
    for f in results["V1"]["folds"]:
        if f.get("excluded"):
            print(f"    fold {f['fold']} {f['held_out']}: EXCLUDED ({f['reason']})")
        else:
            print(f"    fold {f['fold']} {f['held_out']}: acc {f['accuracy']:.4f} "
                  f"macroF1 {f['macro_f1']:.4f} (base {f['majority_baseline']:.4f})")
    print(f"    POOLED: acc {results['V1']['pooled']['accuracy']:.4f} "
          f"macroF1 {results['V1']['pooled']['macro_f1']:.4f}")

    print("\n  === V5 leave-one-session-out ===", flush=True)
    results["V5"] = run_grouped(D, "session", labels, "V5",
                                exclude_groups=(DEGENERATE_SESSION,), seed=a.seed)
    results["V5"]["excluded_reason"] = (
        f"{DEGENERATE_SESSION} contains only inter_coil recordings (7/0), so it "
        f"cannot score a two-class task. It remains in TRAINING for the other folds."
    )
    for f in results["V5"]["folds"]:
        if f.get("excluded"):
            print(f"    fold {f['fold']} {f['held_out']}: EXCLUDED ({f['reason']})")
        else:
            print(f"    fold {f['fold']} {f['held_out']}: acc {f['accuracy']:.4f} "
                  f"macroF1 {f['macro_f1']:.4f} (base {f['majority_baseline']:.4f})")
    print(f"    POOLED: acc {results['V5']['pooled']['accuracy']:.4f} "
          f"macroF1 {results['V5']['pooled']['macro_f1']:.4f}")

    print("\n  === V3 shuffled windows (leaky reference) ===", flush=True)
    results["V3"] = run_shuffled(D, labels, a.seed)
    print(f"    acc {results['V3']['pooled']['accuracy']:.4f} "
          f"macroF1 {results['V3']['pooled']['macro_f1']:.4f}  (LEAKY)")

    # -- fusion authority, from the pre-registered floor -------------------
    n_sessions_scored = results["V5"]["n_groups_scored"]
    v5_f1 = results["V5"]["pooled"]["macro_f1"]
    authority = C.fault_authority(n_sessions_scored, v5_f1)
    results["fusion"] = {
        "floor": C.FUSION_CONFIG["fault_authority"],
        "protocol_used": "V5 (leave-one-session-out)",
        "n_validation_groups": n_sessions_scored,
        "honest_macro_f1": v5_f1,
        "fault_authority": bool(authority),
        "status": "Fault-capable" if authority else "INDICATIVE",
        "reason": (
            f"{n_sessions_scored} scored validation group(s) against the "
            f"pre-registered minimum of "
            f"{C.FUSION_CONFIG['fault_authority']['min_validation_groups']}; "
            f"macro-F1 {v5_f1:.4f} against the minimum of "
            f"{C.FUSION_CONFIG['fault_authority']['min_macro_f1']}."
        ),
    }
    results["session_only_baseline"] = {
        "source": "V4_probe.subsets.1000W_1500W_only.fault_class",
        "note": ("1-NN on i0rel_residual alone -- a scalar that cannot contain "
                 "winding information. Any fault-class score that does not clearly "
                 "beat this is not demonstrating winding diagnosis."),
    }
    results["seconds"] = round(time.time() - t_start, 1)

    out = os.path.join(OUT_DIR, "winding_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n  FUSION: {results['fusion']['status']} -- {results['fusion']['reason']}")
    print(f"  wrote {out}  ({results['seconds']/60:.1f} min)")


if __name__ == "__main__":
    main()
