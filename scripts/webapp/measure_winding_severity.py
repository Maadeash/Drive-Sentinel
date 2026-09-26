"""
Can the winding branch tell how SEVERE a short is? Measured, not assumed.

The technician app's winding alert has a HOW BAD field. A severity band
(early / moderate / severe) is only honest if something the branch measures
tracks the recorded severity. The natural candidate is the negative-sequence
current ratio -- the physics feature the existing severity-ramp replay shows.

This script measures, per fault type and per motor, the Spearman rank
correlation between that ratio and the recorded severity, plus the median ratio
at each severity level, and writes artifacts/webapp/winding_severity_check.json.

Result on the D2 feature cache (2026-09-21): |rho| <= 0.34 everywhere and the
medians alternate with ACQUISITION SESSION rather than rising with severity --
the confound documented in docs/data_notes_d2.md section 7 again. So the app
does NOT show a severity band for winding faults; it says severity is not
estimated and links this measurement. Choosing thresholds anyway would be
inventing a number.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import spearmanr

from drivesentinel import config as C

OUT = os.path.join(C.ARTIFACT_DIR, "webapp", "winding_severity_check.json")


def main():
    D = dict(np.load(os.path.join(C.ARTIFACT_DIR, "multistage", "winding",
                                  "features.npz"), allow_pickle=False))
    ns = D["X"][:, list(D["feature_names"]).index("neg_seq_ratio")]
    rows = []
    for fault in ("inter_turn", "inter_coil"):
        for motor in sorted(set(D["group"].tolist())):
            m = (D["y"] == fault) & (D["group"] == motor)
            if m.sum() < 10:
                continue
            lv = np.sort(np.unique(D["severity"][m]))
            rows.append({
                "fault": fault, "motor": motor, "n_windows": int(m.sum()),
                "spearman_rho": float(spearmanr(D["severity"][m], ns[m]).correlation),
                "severity_levels": [float(v) for v in lv],
                "median_neg_seq_ratio": [float(np.median(ns[m & (D["severity"] == v)]))
                                         for v in lv],
                "median_by_level_session": [
                    sorted(set(D["session"][m & (D["severity"] == v)].tolist()))
                    for v in lv],
            })
    worst = max(abs(r["spearman_rho"]) for r in rows)
    out = {
        "question": "Does the negative-sequence ratio track recorded winding-fault "
                    "severity within a motor?",
        "feature": "neg_seq_ratio (artifacts/multistage/winding/features.npz)",
        "rows": rows,
        "max_abs_rho": worst,
        "conclusion": ("No. The largest within-motor |rank correlation| is "
                       f"{worst:.2f}, and the medians move with acquisition session, "
                       "not severity. The technician app therefore shows no "
                       "severity band for winding faults."),
        "severity_band_shown": False,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(f"  {r['fault']:10} {r['motor']:5} rho {r['spearman_rho']:+.2f}")
    print(f"  max |rho| {worst:.2f} -> severity band shown: False")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
