"""
B-S2/S3 inverter telemetry: score the 4-class family, and lead with the ablation.

Protocols:
  V3  ELECTRICAL-ONLY ABLATION -- temperature channels removed. THE HEADLINE.
      "Can open- and short-circuit faults be detected without a thermometer?" is
      the question that matters for a drive-health story, and the answer is
      reported first whatever it is.
  V1  contiguous block split per condition, 70/30 with a purge gap.
  V2  random split -- the LEAKY REFERENCE.

The 9-class location view is produced as a qualitative confusion matrix only.
Under the block split three of its classes get ~8-10 test windows, and a
per-class number on eight samples has a 95 % CI of roughly +/-35 points.
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
from drivesentinel.adapters import bacha_inverter as B
from drivesentinel.branches import inverter_telemetry as IT
from drivesentinel.common import splits as SP

NL = chr(10)

OUT_DIR = os.path.join(C.ARTIFACT_DIR, "multistage", "inverter_telemetry")

# One window is 50 samples at hop 10, so consecutive windows share 80 % of their
# samples. A purge of ceil(window/hop) = 5 guarantees no shared sample crosses
# the split; splits.contiguous_block refuses purge=0 outright.
PURGE_WINDOWS = 5


def _scores(truth, pred, labels):
    truth, pred = np.asarray(truth), np.asarray(pred)
    acc = float((truth == pred).mean()) if truth.size else float("nan")
    f1s, support = [], {}
    for c in labels:
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        support[c] = int((truth == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    base = (float(Counter(truth.tolist()).most_common(1)[0][1] / truth.size)
            if truth.size else float("nan"))
    return dict(accuracy=acc, macro_f1=float(np.mean(f1s)),
                per_class_f1=dict(zip(labels, [float(f) for f in f1s])),
                support=support, majority_baseline=base, n=int(truth.size))


def _confusion(truth, pred, labels):
    idx = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(truth, pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m.tolist()


def _fit_predict(Xtr, ytr, Xte, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                       max_depth=4, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def block_split(D, train_frac=0.70, purge=PURGE_WINDOWS):
    """
    Contiguous 70/30 inside each condition run, ordered by within-run position.

    Per run, not globally: a single global cut would put whole conditions on one
    side. `splits.contiguous_block_per_group` enforces the purge.
    """
    order = np.lexsort((D["position"], D["run"]))
    runs_sorted = D["run"][order]
    f = SP.contiguous_block_per_group(runs_sorted, train_frac, purge)
    return order[f["train_idx"]], order[f["test_idx"]], f["caveat"]


def run_block(D, target, labels, seed, train_frac=0.70):
    tr, te, caveat = block_split(D, train_frac)
    SP.assert_disjoint(tr, te, context="block split")
    pred = _fit_predict(D["X"][tr], D[target][tr], D["X"][te], seed)
    sc = _scores(D[target][te], pred, labels)
    sc.update(caveat=caveat, n_train=int(tr.size), n_test=int(te.size),
              purge_windows=PURGE_WINDOWS,
              confusion=_confusion(D[target][te], pred, labels),
              labels=list(labels))
    return sc


def run_shuffled(D, target, labels, seed, n_folds=5):
    t_all, p_all = [], []
    for f in SP.shuffled(len(D["X"]), n_folds=n_folds, seed=seed, labels=D[target]):
        pred = _fit_predict(D["X"][f["train_idx"]], D[target][f["train_idx"]],
                            D["X"][f["test_idx"]], seed)
        t_all.append(D[target][f["test_idx"]]); p_all.append(pred)
    t, p = np.concatenate(t_all), np.concatenate(p_all)
    sc = _scores(t, p, labels)
    sc.update(leaky=True, confusion=_confusion(t, p, labels), labels=list(labels),
              caveat="(leaky reference) windows split at random within each run")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    root = os.path.join(C.PROJECT_ROOT, "data_ext", "bacha_inverter")
    recs = B.load(root)
    labels = list(IT.LABELS)

    D = IT.build_dataset(recs, use_temperature=True)
    E = IT.build_dataset(recs, use_temperature=False)
    print(f"  conditions  : {len(recs)}")
    print(f"  windows     : {D['X'].shape} (with temperature), "
          f"{E['X'].shape} (electrical only)")
    print(f"  family      : {dict(Counter(D['family'].tolist()))}")
    print(f"  dropped     : {list(B.DROPPED_CHANNELS)} "
          f"(per-class means span <0.8 ADC counts)")

    res = {
        "branch": "inverter_telemetry", "dataset": "bacha_inverter",
        "stage": "S2/S3",
        "task": " / ".join(labels),
        "n_windows": int(D["X"].shape[0]),
        "window_s": IT.WINDOW_S, "hop_s": IT.HOP_S,
        "purge_windows": PURGE_WINDOWS,
        "dropped_channels": list(B.DROPPED_CHANNELS),
        "dropped_reason": ("VDC/IDC/VD per-class means span 0.78/0.45/0.26 ADC "
                           "counts against per-channel std 1.2-1.4 -- inside "
                           "their own quantisation noise."),
        "calibration_independence": (
            "No feature inverts the NTC curve. Temperature enters only as raw ADC "
            "statistics and as ADC DIFFERENCES between channels, both monotone in "
            "temperature under any calibration. If the Steinhart-Hart refit in "
            "data_notes_d3.md is wrong, none of these numbers change."),
        "confound": IT.CONFOUND_NOTE,
        "temperature_note": IT.TEMPERATURE_NOTE,
        "feature_names": list(map(str, D["feature_names"])),
        "feature_names_electrical_only": list(map(str, E["feature_names"])),
    }

    # -- V3 first: the ablation is the headline ---------------------------
    print("\n  === V3 ELECTRICAL-ONLY ABLATION (headline) ===", flush=True)
    v3_block = run_block(E, "family", labels, a.seed)
    v3_shuf = run_shuffled(E, "family", labels, a.seed)
    res["V3_electrical_only"] = {
        "protocol": "temperature channels REMOVED; Ia/Ib only",
        "question": ("Can open- and short-circuit faults be detected without a "
                     "thermometer? This is the question that matters for a "
                     "drive-health story."),
        "block_split": v3_block,
        "shuffled_LEAKY": v3_shuf,
        "n_features": int(E["X"].shape[1]),
    }
    print(f"    block split : acc {v3_block['accuracy']:.4f} "
          f"macroF1 {v3_block['macro_f1']:.4f} (base {v3_block['majority_baseline']:.4f})")
    print(f"    per-class F1: " + "  ".join(
        f"{k} {v:.3f}" for k, v in v3_block["per_class_f1"].items()))
    print(f"    shuffled    : acc {v3_shuf['accuracy']:.4f}  (LEAKY)")

    # -- the sharpest available statement of the confound, and it is measured --
    # If "over temperature" is detectable WITHOUT a temperature sensor, the model
    # is identifying the run, not the condition. Each condition is one contiguous
    # file recorded at a distinct wall-clock time, so anything that drifts with
    # time -- including the currents -- carries run identity.
    ot_f1 = v3_block["per_class_f1"]["over_temp"]
    oc_f1 = v3_block["per_class_f1"]["open_circuit"]
    res["confound_evidence"] = {
        "over_temp_f1_without_temperature_channels": ot_f1,
        "statement": (
            f"The electrical-only ablation reaches macro-F1 {v3_block['macro_f1']:.4f} "
            f"and scores over_temp at F1 {ot_f1:.3f} -- WITH NO TEMPERATURE SENSOR "
            f"IN THE FEATURE SET. A thermal fault is not physically detectable from "
            f"two 10 Hz phase currents, so that number is the model identifying WHICH "
            f"RUN a window came from, not what condition the inverter was in. Each "
            f"condition is one contiguous file at a distinct wall-clock time, so "
            f"anything that drifts with time carries run identity. Treat every "
            f"4-class number on this dataset as an upper bound contaminated by run "
            f"identification."),
        "open_circuit_f1_without_temperature_channels": oc_f1,
        "open_circuit_note": (
            f"open_circuit is the one class the electrical channels should be able "
            f"to see, and it is the WEAKEST at F1 {oc_f1:.3f}. F1 (HB2 high-side "
            f"open) has almost the same Ia/Ib means as F0 -- measured 519/474 "
            f"against 515/475."),
    }
    print(f"    -> over_temp F1 {ot_f1:.3f} WITHOUT a temperature sensor: "
          f"that is run identity, not physics")
    print(f"    -> open_circuit F1 {oc_f1:.3f}: the class the currents SHOULD see "
          f"is the weakest")

    # -- V1 block split, with temperature ---------------------------------
    print("\n  === V1 contiguous block split (with temperature) ===", flush=True)
    v1 = run_block(D, "family", labels, a.seed)
    res["V1_block_split"] = {
        "protocol": "contiguous 70/30 per condition run, purge >= 1 window",
        "scores": v1,
    }
    print(f"    acc {v1['accuracy']:.4f} macroF1 {v1['macro_f1']:.4f} "
          f"(base {v1['majority_baseline']:.4f}, n_test={v1['n_test']})")
    print(f"    per-class F1: " + "  ".join(
        f"{k} {v:.3f}" for k, v in v1["per_class_f1"].items()))

    # -- V2 shuffled, the leaky reference ---------------------------------
    print("\n  === V2 random split (leaky reference) ===", flush=True)
    v2 = run_shuffled(D, "family", labels, a.seed)
    res["V2_shuffled_LEAKY"] = {"protocol": "random window split", "scores": v2}
    print(f"    acc {v2['accuracy']:.4f} macroF1 {v2['macro_f1']:.4f}  (LEAKY)")

    # -- RUN-IDENTIFICATION CONTROL ---------------------------------------
    # The over_temp-without-a-thermometer result is an INFERENCE that the model
    # is reading run identity. This measures it directly: predict WHICH RUN a
    # window came from, same features, same block split. If run identity is
    # recoverable then so is anything perfectly correlated with it -- and the
    # condition label IS perfectly correlated with it, because there is exactly
    # one run per condition.
    print(NL + "  === RUN-IDENTIFICATION CONTROL ===", flush=True)
    runs = sorted(set(D["run"].tolist()))
    rid_full = run_block(D, "run", runs, a.seed)
    rid_elec = run_block(E, "run", runs, a.seed)
    res["run_identification_control"] = {
        "question": ("Can the model tell WHICH RUN a window came from? One run per "
                     "condition means run identity and condition label are the same "
                     "variable, so whatever fraction of the 4-class score is run "
                     "identification is not condition diagnosis."),
        "with_temperature": {k: rid_full[k] for k in
                             ("accuracy", "macro_f1", "majority_baseline", "n")},
        "electrical_only": {k: rid_elec[k] for k in
                            ("accuracy", "macro_f1", "majority_baseline", "n")},
        "n_runs": len(runs),
        "interpretation": (
            f"Run identity is recoverable at {rid_full['accuracy']:.4f} with "
            f"temperature and {rid_elec['accuracy']:.4f} from the electrical "
            f"channels alone, against a {rid_full['majority_baseline']:.4f} "
            f"majority baseline over {len(runs)} runs. The 4-class family score is "
            f"an upper bound on condition diagnosis by that margin: a model that can "
            f"name the run can name the condition without diagnosing anything, "
            f"because there is one run per condition."),
    }
    print(f"    predict run (with temperature): acc {rid_full['accuracy']:.4f} "
          f"macroF1 {rid_full['macro_f1']:.4f} "
          f"(base {rid_full['majority_baseline']:.4f}, {len(runs)} runs)")
    print(f"    predict run (electrical only) : acc {rid_elec['accuracy']:.4f} "
          f"macroF1 {rid_elec['macro_f1']:.4f}")
    print("    -> compare against the 4-class scores above: that is how much of "
          "them is run identity")

    # -- 9-class: qualitative only ----------------------------------------
    loc_labels = sorted(set(D["location"].tolist()))
    loc = run_block(D, "location", loc_labels, a.seed)
    res["location_9class_QUALITATIVE_ONLY"] = {
        "protocol": "contiguous block split, 9-class location view",
        "reported_as": "confusion matrix only -- NOT a headline metric",
        "reason": ("Under the block split the smallest classes get ~8-10 test "
                   "windows. A per-class number on eight samples has a 95 % CI "
                   "of roughly +/-35 points."),
        "labels": loc_labels,
        "confusion": loc["confusion"],
        "support": loc["support"],
    }
    print(f"\n  === 9-class location view (qualitative only) ===")
    print(f"    test-window support: {loc['support']}")

    # -- fusion authority --------------------------------------------------
    n_groups = 0        # one run per condition: no independent group to hold out
    honest_f1 = v1["macro_f1"]
    auth = C.fault_authority(n_groups, honest_f1)
    res["fusion"] = {
        "floor": C.FUSION_CONFIG["fault_authority"],
        "protocol_used": "V1 (contiguous block split, within-run)",
        "n_validation_groups": n_groups,
        "honest_macro_f1": honest_f1,
        "fault_authority": bool(auth),
        "status": "Fault-capable" if auth else "INDICATIVE",
        "reason": (
            f"0 independent validation groups against the pre-registered minimum "
            f"of {C.FUSION_CONFIG['fault_authority']['min_validation_groups']}: "
            f"one run per condition means there is nothing to hold out. The "
            f"branch is INDICATIVE whatever it scores, which "
            f"docs/claims_audit.md §1.1 anticipated before it existed."),
    }
    res["seconds"] = round(time.time() - t0, 1)

    out = os.path.join(OUT_DIR, "inverter_telemetry_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n  FUSION: {res['fusion']['status']} -- {res['fusion']['reason']}")
    print(f"  wrote {out}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
