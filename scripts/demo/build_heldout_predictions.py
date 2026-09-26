"""
Held-out predictions for the winding and inverter replays.

WHY THIS EXISTS
---------------
An alert names a fault type. That name must come from what the branch actually
predicts on data it was not trained on -- never from the recording's label.

The bearing replays already satisfy this: each carries the probabilities of the
fold model that held that bearing out (build_scenarios.build_bearing). The
winding and inverter replays do not. The winding ramps store only a physics
feature, and the inverter replays store a one-hot of the TRUE label as their
`probs`. Feeding either through the alert pipeline would present ground truth as
a detection.

This script supplies what they lack, from each branch's OWN validated protocol --
the protocol its published metric, and therefore its tier, comes from:

  winding   V5 leave-one-session-out    (winding_results.json "V5")
  inverter  V1 contiguous 70/30 block   (inverter_telemetry_results.json
            split within each run,       "V1_block_split"; the protocol the
            purge 5 windows              fusion tier is computed from)

THE PROOF OF SAMENESS. The script refits exactly those protocols (same features,
model, hyper-parameters, seed and split) and FAILS unless the recomputed score
equals the figure already published in the results JSON. The replay therefore
shows the very predictions behind the number on the metric card, errors
included -- it cannot be flattered.

WHAT HAS NO HELD-OUT PREDICTION, AND SAYS SO
  * winding windows from session 2022-08-11: never held out by V5 (pure
    inter_coil, cannot score a two-class task). Their probs are NaN and the
    app shows "fault type not determined" for them.
  * inverter windows in the TRAINING block of each run. Only the held-out 30 %
    block is replayed through the alert pipeline.

Outputs artifacts/demo/heldout/<scenario_id>.npz. The scenario files themselves
are not touched.

CPU. V-1 runs in parallel; set OMP_NUM_THREADS=2 when calling this.
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np

from drivesentinel import config as C

DEMO = os.path.join(C.ARTIFACT_DIR, "demo")
OUT = os.path.join(DEMO, "heldout")


def _load_script(name):
    p = os.path.join(ROOT, "scripts", "branches", name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_proba(model_factory, Xtr, ytr, Xte, seed):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)                   # fit-set statistics only
    m = model_factory(seed).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), m.predict_proba(sc.transform(Xte)), \
        [str(c) for c in m.classes_]


# ===========================================================================
# winding -- V5 leave-one-session-out
# ===========================================================================

def winding(seed):
    Wscript = _load_script("10_winding.py")
    from drivesentinel.branches import winding as W
    with open(os.path.join(C.ARTIFACT_DIR, "multistage", "winding",
                           "winding_results.json")) as fh:
        R = json.load(fh)
    D = dict(np.load(os.path.join(C.ARTIFACT_DIR, "multistage", "winding",
                                  "features.npz"), allow_pickle=False))
    labels = list(W.LABELS)
    excluded = set(R["V5"]["excluded_groups"])
    sessions = sorted(set(D["session"].tolist()))

    probs = np.full((len(D["y"]), len(labels)), np.nan)
    pred = np.full(len(D["y"]), "", dtype=object)
    scored = np.zeros(len(D["y"]), bool)
    for s in sessions:
        if s in excluded:
            continue
        te = D["session"] == s
        p, pr, classes = _fit_proba(Wscript._model, D["X"][~te], D["y"][~te],
                                    D["X"][te], seed)
        order = [classes.index(l) for l in labels]
        probs[te] = pr[:, order]
        pred[te] = p
        scored |= te

    acc = float((pred[scored] == D["y"][scored]).mean())
    pub = R["V5"]["pooled"]["accuracy"]
    assert abs(acc - pub) < 1e-12, (
        f"winding V5 recomputed {acc!r} != published {pub!r}: not the same protocol")
    print(f"  winding V5 recomputed accuracy {acc:.4f} == published {pub:.4f}")
    return D, probs, labels, scored


def winding_sidecar(entry, D, probs, labels, scored):
    """Rows in exactly the order build_winding_ramp took them, then proved equal."""
    sc = dict(np.load(os.path.join(DEMO, entry["file"]), allow_pickle=False))
    motor = entry["id"].rsplit("_", 1)[1]
    fault = entry.get("fault", "inter_turn")
    sel = (D["group"] == motor) & (D["y"] == fault)
    rows = np.concatenate([np.flatnonzero(sel & (D["severity"] == v))
                           for v in np.sort(np.unique(D["severity"][sel]))])
    ns = D["X"][rows, list(D["feature_names"]).index("neg_seq_ratio")]
    assert np.array_equal(ns.astype(np.float32), sc["neg_seq_ratio"]), (
        f"{entry['id']}: rows do not line up with the stored scenario")
    assert np.array_equal(D["recording"][rows], sc["recording"])
    return dict(protocol="V5 leave-one-session-out (winding_results.json)",
                labels=np.asarray(labels), probs=probs[rows].astype(np.float32),
                has_heldout=scored[rows], rows=rows.astype(np.int64),
                true_label=np.asarray(D["y"][rows]))


# ===========================================================================
# inverter -- V1 contiguous block split, with temperature
# ===========================================================================

def inverter(seed):
    Iscript = _load_script("30_inverter_telemetry.py")
    from drivesentinel.adapters import bacha_inverter as BI
    from drivesentinel.branches import inverter_telemetry as IT
    with open(os.path.join(C.ARTIFACT_DIR, "multistage", "inverter_telemetry",
                           "inverter_telemetry_results.json")) as fh:
        R = json.load(fh)
    recs = BI.load(os.path.join(C.PROJECT_ROOT, "data_ext", "bacha_inverter"))
    D = IT.build_dataset(recs, use_temperature=True)
    labels = list(IT.LABELS)

    def model(seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                              max_depth=4, l2_regularization=1.0,
                                              random_state=seed)

    tr, te, _ = Iscript.block_split(D)
    p, pr, classes = _fit_proba(model, D["X"][tr], D["family"][tr], D["X"][te], seed)
    acc = float((p == D["family"][te]).mean())
    pub = R["V1_block_split"]["scores"]["accuracy"]
    assert abs(acc - pub) < 1e-12 and te.size == R["V1_block_split"]["scores"]["n_test"], (
        f"inverter V1 recomputed {acc!r} (n={te.size}) != published {pub!r}")
    print(f"  inverter V1 recomputed accuracy {acc:.4f} on {te.size} held-out "
          f"windows == published {pub:.4f}")
    order = [classes.index(l) for l in labels]
    return D, te, pr[:, order], labels


def inverter_sidecar(f_code, D, te, probs, labels):
    rows = te[D["location"][te] == f_code]
    sel = np.isin(te, rows)
    order = np.argsort(D["position"][rows])
    return dict(protocol="V1 contiguous 70/30 block split within the run, purge 5 "
                         "(inverter_telemetry_results.json)",
                labels=np.asarray(labels), probs=probs[sel][order].astype(np.float32),
                position=D["position"][rows][order].astype(np.int64),
                has_heldout=np.ones(order.size, bool),
                true_label=np.asarray(D["family"][rows][order]))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DEMO, "manifest.json")) as fh:
        man = json.load(fh)
    seed = C.SEED

    wind = [s for s in man["scenarios"] if s["branch"] == "winding" and s.get("file")]
    inv = [s for s in man["scenarios"] if s["branch"] == "inverter_telemetry"
           and s.get("file")]

    if wind:
        D, probs, labels, scored = winding(seed)
        for s in wind:
            out = winding_sidecar(s, D, probs, labels, scored)
            np.savez_compressed(os.path.join(OUT, f"{s['id']}.npz"), **out)
            n = int(out["has_heldout"].sum())
            agree = float((np.asarray(labels)[np.nanargmax(np.nan_to_num(
                out["probs"][out["has_heldout"]], nan=-1), axis=1)]
                == out["true_label"][out["has_heldout"]]).mean()) if n else float("nan")
            print(f"    {s['id']:34} {n}/{len(out['rows'])} windows held out, "
                  f"held-out agreement with truth {agree:.3f}")
    if inv:
        D, te, probs, labels = inverter(seed)
        for s in inv:
            code = s.get("f_code") or s["id"].split("_")[1]
            out = inverter_sidecar(code, D, te, probs, labels)
            np.savez_compressed(os.path.join(OUT, f"{s['id']}.npz"), **out)
            agree = float((np.asarray(labels)[out["probs"].argmax(1)]
                           == out["true_label"]).mean())
            print(f"    {s['id']:34} {len(out['position'])} held-out windows, "
                  f"agreement with truth {agree:.3f}")


if __name__ == "__main__":
    main()
