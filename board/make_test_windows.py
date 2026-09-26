"""
Package real bearing windows and the golden reference for the board.

The PYNQ-Z2 will not have the 150 MB order-spectrum cache or the drivesentinel
package, so this writes a small, self-contained test set next to the overlay:

    board/test_windows.npz
        x_float    (N, 5, 512) float32  conditioned DSP output -- what the PS front
                                        end would hand the quantiser
        x_int8     (N, 5, 512) int8     the bytes the AXI-Stream carries, from the
                                        D-4 quantiser (drivesentinel.rtl.model)
        pred_rtl   (N,) int             class the RTL model predicts at F = 26
        logit_reg  (N, 3) int64         the LOGIT0..2 register values it predicts
        pred_float (N,) int             golden_reference.predict on x_float
        label      (N,) int             ground-truth class
        window     (N,) int             index into the full 16,211-window cache
        bearing    (N,) str             which bearing it came from
    board/golden/
        golden_reference.py + scales.json + input_norm.json + the ten .mem files,
        copied unchanged from artifacts/int8_export/, so the notebook can run the
        specification on the board itself.

WHICH WINDOWS. 40 per class, drawn with a fixed seed across as many different
bearings as the class has, so the board test is not 120 near-identical windows
from one recording. The selection is written into the file (window, bearing), so
it can be traced back to the cache.

`pred_rtl` and `pred_float` must agree on every window -- that is the claim V-1
is checking on all 16,211 in simulation -- and this script asserts it for the
ones it packages, so a board mismatch cannot be blamed on the test file.
"""

import importlib.util
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from drivesentinel import config as C
from drivesentinel.features import load_cache
from drivesentinel.rtl import model as M

BOARD = os.path.join(ROOT, "board")
PER_CLASS = 40
SEED = 20260921


def main() -> None:
    X, y, meta = load_cache()
    rng = np.random.default_rng(SEED)
    picks = []
    for c in range(len(C.LABELS)):
        idx = np.flatnonzero(y == c)
        bearings = meta.iloc[idx]["bearing"].to_numpy()
        # round-robin over bearings so every bearing of the class is represented
        # before any bearing is used twice
        by_b = {b: rng.permutation(idx[bearings == b]).tolist()
                for b in sorted(set(bearings))}
        order = rng.permutation(sorted(by_b))
        chosen = []
        while len(chosen) < PER_CLASS:
            for b in order:
                if by_b[b] and len(chosen) < PER_CLASS:
                    chosen.append(by_b[b].pop())
        picks += chosen
    picks = np.array(sorted(picks))

    exp = M.Export()
    xf = X[picks].astype(np.float32)
    xq = M.quantise_input(xf.astype(np.float64), exp)
    params = __import__("json").load(open(os.path.join(C.ARTIFACT_DIR, "rtl",
                                                       "rtl_params.json")))
    r = M.forward(xq, exp, requant="int", frac_width=params["frac_width"])
    logit_reg = r["logits_raw"] >> np.int64(params["logit_shift"])

    spec = importlib.util.spec_from_file_location(
        "golden_reference", os.path.join(C.EXPORT_DIR, "golden_reference.py"))
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    pred_float = golden.predict(xf.astype(np.float64))

    assert np.array_equal(r["pred"], pred_float), \
        "RTL model and golden reference disagree on the packaged windows"

    np.savez_compressed(
        os.path.join(BOARD, "test_windows.npz"),
        x_float=xf, x_int8=xq, pred_rtl=r["pred"].astype(np.int64),
        logit_reg=logit_reg.astype(np.int64), pred_float=pred_float.astype(np.int64),
        label=y[picks].astype(np.int64), window=picks.astype(np.int64),
        bearing=meta.iloc[picks]["bearing"].to_numpy().astype(str),
        build_id=np.int64(params["expected_build_id"]),
        expected_cycles=np.int64(1786724),
    )

    gdir = os.path.join(BOARD, "golden")
    os.makedirs(gdir, exist_ok=True)
    names = ["golden_reference.py", "scales.json", "input_norm.json"] + \
        [f"layer_{i}_{k}.mem" for i in range(5) for k in ("W", "b")]
    for n in names:
        shutil.copyfile(os.path.join(C.EXPORT_DIR, n), os.path.join(gdir, n))

    print(f"  {len(picks)} windows, classes {np.bincount(y[picks])}, "
          f"{len(set(meta.iloc[picks]['bearing']))} bearings")
    print(f"  RTL model and golden agree on all {len(picks)}")
    print(f"  wrote board/test_windows.npz and board/golden/ ({len(names)} files)")


if __name__ == "__main__":
    main()
