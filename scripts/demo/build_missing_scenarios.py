"""
Build ONLY the scenarios the technician web app needs that do not exist yet.

The existing replays in artifacts/demo/ are reused exactly as built and are never
rewritten: the bearing replays are out-of-sample by construction and must stay
that way. `scripts/demo/build_scenarios.py main()` rebuilds everything and
rewrites the manifest, so it is NOT run here. This script calls the SAME builder
functions, with the same rules, for the missing cases only, and appends to the
manifest. An id that is already present is skipped, never overwritten.

    .venv/Scripts/python.exe scripts/demo/build_missing_scenarios.py

THE FIVE
  inverter short circuit          build_inverter("F3")   HB1 low-side short
  inverter over-temperature       build_inverter("F6")   HB1 over-temperature
  winding inter-coil short        build_winding_ramp("1000W", "inter_coil")
  supply single-phasing at start  build_supply(5)        healthy motor, FILE 5
  inner-race bearing, detected    build_bearing("KI18")  fold model that held KI18 out

WHY KI18. Of the eleven inner-race bearings, KI18 has real (not artificially
induced) damage and scored window accuracy 1.0000 when held out
(artifacts/runs/lobo_summary.json). It is the clearest honest example of a
correct detection. KI05, the documented miss, is already in the demo.

WHY 1000W FOR INTER-COIL. The 3000W motor's inter-coil faults are all in session
2022-08-11, which the winding branch's leave-one-session-out protocol never holds
out, so no out-of-sample prediction exists for them. 1000W inter-coil recordings
are in the two scored sessions.

CPU. V-1 runs in parallel on this machine, so torch is held to 2 threads.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

import build_scenarios as BS
from drivesentinel import config as C

OUT_DIR = BS.OUT_DIR


def _save(sc, manifest, extra):
    p = os.path.join(OUT_DIR, f"{sc['scenario']}.npz")
    np.savez_compressed(p, **sc)
    entry = {"id": sc["scenario"], "stage": sc["stage"], "branch": sc["branch"],
             "file": os.path.basename(p),
             "out_of_sample": bool(sc.get("out_of_sample", False)),
             "mb": round(os.path.getsize(p) / 1e6, 2),
             "added_for": "technician web app",
             "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    entry.update(extra)
    manifest["scenarios"].append(entry)
    print(f"  + {sc['scenario']:38} {entry['mb']:5.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bearing", default="KI18")
    ap.add_argument("--skip-bearing", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    have = {s["id"] for s in manifest["scenarios"]}
    before = {f: os.path.getmtime(os.path.join(OUT_DIR, f))
              for f in os.listdir(OUT_DIR) if f.endswith(".npz")}

    # -- inverter -------------------------------------------------------------
    for code in ("F3", "F6"):
        sc = BS.build_inverter(code)
        if sc is None:
            print(f"  inverter {code}: D3 absent -- SKIPPED")
            continue
        if sc["scenario"] in have:
            print(f"  = {sc['scenario']:38} already present, kept")
            continue
        _save(sc, manifest, {"deliverable": sc["deliverable"], "f_code": code,
                             "true_label": str(sc["true_label"])})

    # -- winding inter-coil ---------------------------------------------------
    sc = BS.build_winding_ramp("1000W", "inter_coil")
    if sc is None:
        print("  winding inter_coil: feature cache absent -- SKIPPED")
    elif sc["scenario"] in have:
        print(f"  = {sc['scenario']:38} already present, kept")
    else:
        _save(sc, manifest, {"fault": "inter_coil"})

    # -- supply single-phasing at start ---------------------------------------
    sc = BS.build_supply(5)
    if sc is None:
        print("  supply FILE 5: D4 absent -- SKIPPED")
    elif sc["scenario"] in have:
        print(f"  = {sc['scenario']:38} already present, kept")
    else:
        _save(sc, manifest, {"deliverable": "threshold rule",
                             "true_label": str(sc["true_label"]),
                             "verdict": str(sc["verdict"]),
                             "latency_s": float(sc["latency_s"])})

    # -- inner-race bearing, detected -----------------------------------------
    bid = f"bearing_{a.bearing}"
    if a.skip_bearing:
        print(f"  {bid}: skipped by request")
    elif bid in have:
        print(f"  = {bid:38} already present, kept")
    else:
        import torch
        torch.set_num_threads(a.threads)
        from drivesentinel.features import load_cache
        from drivesentinel.train import get_device
        X, y, meta = load_cache()
        sc = BS.build_bearing(a.bearing, X, y, meta, get_device(),
                              dict(C.TRAIN_CONFIG), max_windows=120)
        # build_bearing asserts this already; asserted again here because the
        # whole value of a bearing replay rests on it.
        assert sc["out_of_sample"] is True
        assert a.bearing not in [str(b) for b in sc["train_bearings"]], \
            f"OUT-OF-SAMPLE VIOLATION: {a.bearing} is in its own fold's training set"
        _save(sc, manifest, {"bearing": a.bearing, "true_label": sc["true_label"],
                             "window_acc": sc["window_acc"],
                             "n_windows_stored": sc["n_windows_stored"],
                             "n_windows_in_fold": sc["n_windows_in_fold"],
                             "n_train_bearings": sc["n_train_bearings"]})

    # The existing scenario files must be untouched.
    for f, t in before.items():
        assert os.path.getmtime(os.path.join(OUT_DIR, f)) == t, \
            f"{f} was modified -- existing scenarios must be reused as built"

    manifest["extended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  manifest: {len(manifest['scenarios'])} scenarios")


if __name__ == "__main__":
    main()
