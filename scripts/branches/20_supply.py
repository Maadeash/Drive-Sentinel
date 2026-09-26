"""
B-S1 supply branch: score the threshold rule, and a learned classifier as the
leaky reference for contrast.

THE RULE IS THE DELIVERABLE. The learned model exists only to show the gap a
leaky split opens on a dataset with one recording per (motor x class).

Protocols:
  R1  the rule, leave-one-motor-out. The rule has no parameters fitted to data,
      so this is really "does it work on a motor it was not designed against" --
      and it is scored per motor anyway so the number is comparable to V1.
  R2  the rule, all recordings pooled. Same rule, so the same answer; reported
      to make the point that a rule has no train/test distinction to leak across.
  L1  learned classifier, leave-one-motor-out.
  L3  learned classifier, shuffled windows -- the LEAKY REFERENCE.
  N1  bearing / motor-identity confound, as a documented NEGATIVE RESULT.
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
from drivesentinel.adapters import thomas_motor as T
from drivesentinel.branches import supply as S
from drivesentinel.common import splits as SP

OUT_DIR = os.path.join(C.ARTIFACT_DIR, "multistage", "supply")


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
    base = (float(Counter(truth.tolist()).most_common(1)[0][1] / truth.size)
            if truth.size else float("nan"))
    return dict(accuracy=acc, macro_f1=float(np.mean(f1s)),
                per_class_f1=dict(zip(labels, [float(f) for f in f1s])),
                majority_baseline=base, n=int(truth.size))


# ===========================================================================
# window features for the LEARNED reference only
# ===========================================================================

def window_features(rec):
    """
    Per-window features for the learned classifier.

    Deliberately the same information the rule uses -- per-phase RMS, their
    ratios, and vibration RMS -- so the comparison is about the PROTOCOL, not
    about one side having better inputs.
    """
    fs = rec.fs["I1"]
    tracks = []
    for c in T.CURRENTS:
        _, r = S.rms_track(np.asarray(rec.signals[c], dtype=np.float64), fs)
        tracks.append(r)
    R = np.vstack(tracks)
    _, v = S.rms_track(np.asarray(rec.signals["vib_x"], dtype=np.float64),
                       rec.fs["vib_x"])
    v = v[:R.shape[1]] if v.size >= R.shape[1] else np.pad(
        v, (0, R.shape[1] - v.size), mode="edge")

    med = np.median(R, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(med > 0, R / med, 0.0)
    feats = np.vstack([R, ratios, R.min(axis=0) / np.maximum(R.max(axis=0), 1e-9),
                       v[None, :]]).T
    return np.nan_to_num(feats.astype(np.float32))


def build(recs):
    X, y, group, motor, scen = [], [], [], [], []
    for r in recs:
        f = window_features(r)
        verdicts = S.classify(r)
        n = min(len(f), len(verdicts))
        live = [i for i in range(n) if verdicts[i].state != "off"]
        if not live:
            continue
        X.append(f[live])
        y += [r.label] * len(live)
        group += [r.group] * len(live)
        motor += [r.condition["motor"]] * len(live)
        scen += [r.condition["scenario"]] * len(live)
    return dict(X=np.concatenate(X), y=np.asarray(y), group=np.asarray(group),
                motor=np.asarray(motor), scenario=np.asarray(scen))


def _fit_predict(Xtr, ytr, Xte, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                       max_depth=4, random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    recs = T.load()
    labels = list(S.LABELS)
    print(f"  recordings  : {len(recs)}")
    print(f"  motors      : {dict(Counter(r.group for r in recs))}")
    print(f"  labels      : {dict(Counter(r.label for r in recs))}")

    res = {
        "branch": "supply", "dataset": "thomas_motor", "stage": "S1",
        "task": " / ".join(labels),
        "deliverable": "documented threshold rule (not a learned model)",
        "rule": dict(S.RULE),
        "rule_statement": (
            "A phase is LOST in a window when its 0.2 s RMS falls below 5 % of the "
            "median of the other two. A window is OFF when every phase is below an "
            "absolute floor of 0.05 A, and OFF windows are excluded from scoring. "
            "phase_loss_running is separated from single_phasing_start by ROTATION "
            "(vib_x RMS above 0.005), not by current."),
        "why_not_learned": (
            "One recording per (motor x class) and two motors: under "
            "leave-one-motor-out a learned model sees one training example per "
            "class, so it separates two 20 s captures rather than learning phase "
            "loss. Liu et al. measure exactly this on this dataset: macro-F1 "
            "0.9682 random split, 0.5856 within-label block split."),
        "label_source": "phase-current collapse rule",
        "label_vector_used": False,
        "label_reconstruction": dict(T.LABEL_RECONSTRUCTION),
    }

    # -- R1/R2: the rule ---------------------------------------------------
    print("\n  === R1 threshold rule, per recording ===", flush=True)
    per_rec, truth_r, pred_r = [], [], []
    for r in recs:
        s = S.summarise(r)
        per_rec.append(s)
        truth_r.append(s["true_label"]); pred_r.append(s["verdict"])
        lat = s["latency"]["latency_s"] if s["latency"] else None
        print(f"    {s['file']:12} {s['motor']:8} {s['true_label']:22} -> "
              f"{s['verdict']:22} " + (f"latency {lat:.2f}s" if lat is not None else ""))
    res["R1_rule_per_recording"] = {
        "protocol": "threshold rule, every recording",
        "scores": _scores(truth_r, pred_r, labels),
        "recordings": [{k: v for k, v in s.items() if k != "latency"} | {
            "latency_s": (s["latency"] or {}).get("latency_s"),
            "latency_note": (s["latency"] or {}).get("note"),
        } for s in per_rec],
    }

    print("\n  === R2 rule scored leave-one-motor-out ===", flush=True)
    r2 = {}
    for motor in sorted({r.group for r in recs}):
        held = [r for r in recs if r.group == motor]
        t = [r.label for r in held]
        p = [S.recording_verdict(S.classify(r)) for r in held]
        r2[motor] = _scores(t, p, labels)
        print(f"    hold out {motor:16} acc {r2[motor]['accuracy']:.4f} "
              f"macroF1 {r2[motor]['macro_f1']:.4f} (n={r2[motor]['n']})")
    res["R2_rule_leave_one_motor_out"] = {
        "protocol": "threshold rule, scored on each motor in turn",
        "note": ("The rule has no parameters fitted to any recording, so holding "
                 "a motor out changes nothing about the rule -- this measures "
                 "whether it works on each motor, not whether it generalises "
                 "from one to the other."),
        "per_motor": r2,
        "pooled": _scores(truth_r, pred_r, labels),
    }

    # -- latency -----------------------------------------------------------
    lats = [(s["file"], (s["latency"] or {}).get("latency_s"))
            for s in per_rec if s["latency"]]
    res["detection_latency"] = {
        "per_event": [{"file": f, "latency_s": l} for f, l in lats],
        "window_s": S.RULE["window_s"], "hop_s": S.RULE["hop_s"],
        "note": ("Latency is measured against an amplitude crossing independent "
                 "of the rule's own threshold. It cannot be better than the hop "
                 "of 0.1 s."),
    }

    # -- L1/L3: learned, for contrast --------------------------------------
    print("\n  === L1/L3 learned classifier (contrast only) ===", flush=True)
    D = build(recs)
    print(f"    windows: {D['X'].shape} {dict(Counter(D['y'].tolist()))}")

    truth_all, pred_all, per_fold = [], [], []
    for f in SP.iter_checked(SP.leave_one_group_out(D["group"]), D["group"]):
        tr, te = f["train_idx"], f["test_idx"]
        if len(set(D["y"][tr].tolist())) < 2:
            continue
        pred = _fit_predict(D["X"][tr], D["y"][tr], D["X"][te], a.seed)
        sc = _scores(D["y"][te], pred, labels)
        sc["held_out"] = sorted(set(D["group"][te].tolist()))
        per_fold.append(sc)
        truth_all.append(D["y"][te]); pred_all.append(pred)
        print(f"    fold {sc['held_out']}: acc {sc['accuracy']:.4f} "
              f"macroF1 {sc['macro_f1']:.4f} (base {sc['majority_baseline']:.4f})")
    res["L1_learned_leave_one_motor_out"] = {
        "protocol": "learned classifier, leave-one-motor-out",
        "folds": per_fold,
        "pooled": _scores(np.concatenate(truth_all), np.concatenate(pred_all), labels),
    }
    print(f"    POOLED: acc {res['L1_learned_leave_one_motor_out']['pooled']['accuracy']:.4f}")

    t_all, p_all = [], []
    for f in SP.shuffled(len(D["X"]), n_folds=5, seed=a.seed, labels=D["y"]):
        pred = _fit_predict(D["X"][f["train_idx"]], D["y"][f["train_idx"]],
                            D["X"][f["test_idx"]], a.seed)
        t_all.append(D["y"][f["test_idx"]]); p_all.append(pred)
    res["L3_learned_shuffled_LEAKY"] = {
        "protocol": "learned classifier, shuffled windows",
        "leaky": True,
        "caveat": "(leaky reference) windows split at random, ignoring motor and recording",
        "pooled": _scores(np.concatenate(t_all), np.concatenate(p_all), labels),
    }
    print(f"    SHUFFLED (LEAKY): acc "
          f"{res['L3_learned_shuffled_LEAKY']['pooled']['accuracy']:.4f}")

    # -- N1: the confound, as a negative result ----------------------------
    print("\n  === N1 bearing / motor-identity confound (negative result) ===",
          flush=True)
    t_all, p_all = [], []
    for f in SP.shuffled(len(D["X"]), n_folds=5, seed=a.seed, labels=D["motor"]):
        pred = _fit_predict(D["X"][f["train_idx"]], D["motor"][f["train_idx"]],
                            D["X"][f["test_idx"]], a.seed)
        t_all.append(D["motor"][f["test_idx"]]); p_all.append(pred)
    mot = _scores(np.concatenate(t_all), np.concatenate(p_all), ["healthy", "faulty"])
    res["N1_bearing_confound"] = {
        "protocol": "predict MOTOR IDENTITY from the same features, shuffled split",
        "scores": mot,
        "negative_result": True,
        "statement": T.bearing_confound_note(),
    }
    print(f"    predict motor identity: acc {mot['accuracy']:.4f} "
          f"(baseline {mot['majority_baseline']:.4f})")
    print(f"    -> {T.bearing_confound_note()[:78]}...")

    # -- fusion authority ---------------------------------------------------
    n_groups = len({r.group for r in recs})
    honest_f1 = res["R2_rule_leave_one_motor_out"]["pooled"]["macro_f1"]
    auth = C.fault_authority(n_groups, honest_f1)
    res["fusion"] = {
        "floor": C.FUSION_CONFIG["fault_authority"],
        "protocol_used": "R2 (threshold rule, leave-one-motor-out)",
        "n_validation_groups": n_groups,
        "honest_macro_f1": honest_f1,
        "fault_authority": bool(auth),
        "status": "Fault-capable" if auth else "INDICATIVE",
        "reason": (
            f"{n_groups} validation groups against the pre-registered minimum of "
            f"{C.FUSION_CONFIG['fault_authority']['min_validation_groups']}; "
            f"macro-F1 {honest_f1:.4f} against the minimum of "
            f"{C.FUSION_CONFIG['fault_authority']['min_macro_f1']}. Two motors is "
            f"below the group floor whatever the rule scores, which "
            f"docs/claims_audit.md §1.1 anticipated before this branch existed."),
    }
    res["seconds"] = round(time.time() - t0, 1)

    out = os.path.join(OUT_DIR, "supply_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n  FUSION: {res['fusion']['status']} -- {res['fusion']['reason']}")
    print(f"  wrote {out}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
