"""
The deployed INT8 network against the held-out fold models, on the stored
bearing windows the web app streams.

The network the FPGA runs was trained on every Paderborn bearing
(train.train_deployment_model), so on these windows its verdict is in-sample.
This measures how far that would overstate the bearing stage compared with the
fold model that held each bearing out -- the reason the web app keeps the
held-out verdict for recorded windows (drivesentinel/engines.py).

    .venv/Scripts/python.exe scripts/webapp/check_deployed_in_sample.py
    -> artifacts/webapp/deployed_model_in_sample.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivesentinel.engines import SoftwareEngine  # noqa: E402


def main() -> None:
    sw = SoftwareEngine()
    out = {"what": ("window accuracy on the stored demo windows: the deployed INT8 "
                    "network (bit-accurate model, trained on all bearings -> IN-SAMPLE, "
                    "leaky reference) vs the leave-one-bearing-out fold model that held "
                    "the bearing out (honest)"),
           "bearings": {}}
    for p in sorted(glob.glob(os.path.join(ROOT, "artifacts", "demo", "bearing_*.npz"))):
        z = np.load(p, allow_pickle=False)
        truth = z["truth"]
        dep = np.array([sw.infer(w)["class"] for w in z["spectra"]])
        out["bearings"][str(z["bearing"])] = {
            "true_label": str(z["true_label"]), "windows": int(len(truth)),
            "deployed_int8_in_sample_acc": round(float((dep == truth).mean()), 4),
            "held_out_fold_acc": round(float((z["pred"] == truth).mean()), 4)}
    path = os.path.join(ROOT, "artifacts", "webapp", "deployed_model_in_sample.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
