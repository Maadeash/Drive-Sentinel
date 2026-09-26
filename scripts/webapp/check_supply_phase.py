"""
Is the phase the supply alert names the phase that was actually lost?

The app names the phase from the rule's own per-phase measurement. This check
does not reuse that path: it reads the raw D4 currents with the adapter's
low-level reader, computes a plain windowed RMS per phase, and asks which
phase sits near zero. D4 has no metadata naming the disconnected phase
(data_notes_d4.md: "no channel names"), so the raw current is the only ground
truth there is.

    .venv/Scripts/python.exe scripts/webapp/check_supply_phase.py
    -> artifacts/webapp/supply_phase_check.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivesentinel.adapters import thomas_motor as T  # noqa: E402

N_WIN = 40
DEAD = 0.2       # a window counts as collapsed below 20 % of the healthiest phase's median


def main() -> None:
    root = T.default_root()
    out = {"method": (f"raw D4 I1/I2/I3, DC removed, RMS over {N_WIN} equal windows; "
                      f"a phase is collapsed in a window when its RMS is below "
                      f"{DEAD:.0%} of the largest per-phase median RMS"),
           "files": {}}
    for f in (2, 5):
        p = os.path.join(root, f"FILE {f}.mat")
        rms = {}
        for k, c in enumerate(("I1", "I2", "I3")):
            x = T.read_channel(p, c)
            x = x - x.mean()
            seg = len(x) // N_WIN
            rms[f"L{k + 1}"] = np.sqrt(np.mean(x[: seg * N_WIN].reshape(N_WIN, seg) ** 2, axis=1))
        ref = max(np.median(r) for r in rms.values())
        dead = {ph: int((r < DEAD * ref).sum()) for ph, r in rms.items()}
        # the lost phase is dead in the most windows; the others are dead only
        # when the whole motor is off
        lost = max(dead, key=dead.get)
        out["files"][f"FILE {f}"] = {
            "median_rms": {ph: round(float(np.median(r)), 4) for ph, r in rms.items()},
            "collapsed_windows": dead, "of_windows": N_WIN, "lost_phase": lost}
    path = os.path.join(ROOT, "artifacts", "webapp", "supply_phase_check.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
