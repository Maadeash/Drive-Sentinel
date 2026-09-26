"""
How often does the winding stage name the right fault type on the replayed rows?

Both winding demo alerts name the wrong type (the inter-turn ramp is called
inter-coil, and the inter-coil recording inter-turn). This measures whether
that is the stage being wrong throughout, or the alert firing early on windows
that lean the wrong way. It uses the held-out V5 (leave-one-session-out)
predictions the app replays, from artifacts/demo/heldout/, and the per-window
top class of the fusion replay at the moment each alert fired.

    .venv/Scripts/python.exe scripts/webapp/check_winding_type.py
    -> artifacts/webapp/winding_type_check.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivesentinel import alerts as A          # noqa: E402
from drivesentinel import fusion as FU         # noqa: E402
from drivesentinel.sources import RecordedScenarioSource  # noqa: E402


def main() -> None:
    src = RecordedScenarioSource()
    metrics = FU.load_branch_metrics()
    out = {"protocol": "V5 (leave-one-session-out), held-out rows only", "recordings": {}}
    for u in src.units():
        if u.branch != "winding":
            continue
        z = np.load(os.path.join(ROOT, "artifacts", "demo", "heldout", f"{u.scenario}.npz"),
                    allow_pickle=False)
        labels = [str(x) for x in z["labels"]]
        ho = z["has_heldout"].astype(bool)
        pred = np.array(labels)[z["probs"][ho].argmax(1)]
        true = np.asarray(z["true_label"]).astype(str)[ho]
        st, em, first = None, A.AlertEmitter(u.unit_no), None
        for ob in src.observations(u.unit_no):
            if ob.probs is None:
                continue
            st = st or FU.BranchState(metric=metrics[u.branch], labels=ob.labels)
            a = em.observe(st.update(ob.probs), ob.t_s, ob.context)
            if a and first is None:
                first = {"t_s": ob.t_s, "fault_named": a.fault, "status": a.status}
        out["recordings"][u.scenario] = {
            "unit": u.unit_id, "recorded_type": str(true[0]),
            "held_out_rows": int(ho.sum()),
            "window_agreement": round(float((pred == true).mean()), 4),
            "windows_per_class": {k: int((pred == k).sum()) for k in sorted(set(pred))},
            "first_alert": first}
    path = os.path.join(ROOT, "artifacts", "webapp", "winding_type_check.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
