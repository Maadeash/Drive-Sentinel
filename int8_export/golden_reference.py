"""
DriveSentinel INT8 Golden Reference  (auto-generated -- do not edit)
====================================================================
Standalone NumPy model of the quantised order-spectrum CNN.  Depends on numpy
and nothing else: no torch, no drivesentinel package.

PURPOSE
-------
Bit-level specification for the RTL.  The Verilog implementation must reproduce
the argmax of run_int8() for every input window.  Both this file and the RTL
load the SAME .mem files sitting next to it, so a disagreement is a bug in the
logic and cannot be a weight-transcription error.

INPUT CONTRACT
--------------
    x : float array, shape (5, 512)
        Conditioned order spectra, channel 0 = current sideband fold,
        channel 1 = vibration envelope, produced by the DSP front end:
        window 1.0s hop 0.5s | 5 x 512 bins over 0-16.0 orders (0.03125 orders/bin) | classes ['healthy', 'inner_race', 'outer_race']
    The input is standardised inside run_int8() using INPUT_MEAN / INPUT_STD,
    so callers pass the DSP output directly.

ARITHMETIC
----------
    x_int8    = clamp(round(x / s_x), -128, 127)
    acc_int32 = sum(W_int8 * x_int8) + b_int32          # exact integer
    y_float   = s_W[out_channel] * s_x * acc_int32
    relu, then the next layer  (omitted on the last layer)

Accumulators are int32.  The widest layer sums 192 products, worst case
3,121,152, which needs 23 bits -- so no saturation logic is required.

CLASSES: {0: 'healthy', 1: 'inner_race', 2: 'outer_race'}

Generated from: deployment_model.pt
Contract fingerprint: n/a
"""

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {0: 'healthy', 1: 'inner_race', 2: 'outer_race'}
N_CHANNELS = 5
N_BINS = 512

with open(os.path.join(_HERE, "scales.json")) as _fh:
    SCALES = json.load(_fh)
with open(os.path.join(_HERE, "input_norm.json")) as _fh:
    _NORM = json.load(_fh)

INPUT_MEAN = np.array(_NORM["mean"], dtype=np.float64).reshape(N_CHANNELS, 1)
INPUT_STD = np.array(_NORM["std"], dtype=np.float64).reshape(N_CHANNELS, 1)


def _read_mem(path, shape, signed_bits):
    with open(path) as fh:
        vals = [int(ln.strip(), 16) for ln in fh if ln.strip()]
    half = 1 << (signed_bits - 1)
    full = 1 << signed_bits
    out = np.array([v - full if v >= half else v for v in vals], dtype=np.int64)
    return out.reshape(shape)


def _load_layers():
    layers = []
    for spec in SCALES["layers"]:
        W = _read_mem(os.path.join(_HERE, spec["W_file"]), spec["W_shape"], 8)
        b = _read_mem(os.path.join(_HERE, spec["b_file"]), spec["b_shape"], 32)
        layers.append({
            "name": spec["name"], "kind": spec["kind"],
            "W": W.astype(np.int32), "b": b.astype(np.int32),
            "s_W": np.array(spec["s_W"], dtype=np.float64),
            "s_x": float(spec["s_x"]),
            "stride": spec["stride"], "padding": spec["padding"],
        })
    return layers


LAYERS = _load_layers()


def _quant(x, s):
    return np.clip(np.rint(x / s), -128, 127).astype(np.int32)


def run_int8(x):
    """
    x : (N, 5, 512) or (5, 512) float, raw DSP output.
    returns (N, 3) float logits.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    a = (a - INPUT_MEAN[None]) / INPUT_STD[None]

    for L in LAYERS:
        if L["kind"] == "conv":
            xq = _quant(a, L["s_x"])
            pad = L["padding"]
            if pad:
                xq = np.pad(xq, ((0, 0), (0, 0), (pad, pad)))
            k = L["W"].shape[2]
            stride = L["stride"]
            out_len = (xq.shape[2] - k) // stride + 1
            cols = np.stack(
                [xq[:, :, i * stride:i * stride + k] for i in range(out_len)], axis=1
            )
            acc = np.einsum("blik,oik->bol", cols, L["W"], optimize=True)
            acc = acc + L["b"][None, :, None]
            a = acc.astype(np.float64) * (L["s_W"][None, :, None] * L["s_x"])
            a = np.maximum(a, 0.0)
        else:
            a = a.mean(axis=2)
            xq = _quant(a, L["s_x"])
            acc = xq @ L["W"].T + L["b"][None, :]
            a = acc.astype(np.float64) * (L["s_W"][None, :] * L["s_x"])
    return a


def predict(x):
    """Class indices. Map through CLASS_NAMES for labels."""
    return run_int8(x).argmax(axis=1)


if __name__ == "__main__":
    print("DriveSentinel INT8 golden reference")
    print(f"  input      : ({N_CHANNELS}, {N_BINS})")
    print(f"  classes    : {CLASS_NAMES}")
    for L in LAYERS:
        print(f"  {L['name']:6} {L['kind']:6} W{tuple(L['W'].shape)} "
              f"s_x={L['s_x']:.6g}")
    probe = np.zeros((1, N_CHANNELS, N_BINS))
    print(f"  zero input -> class {int(predict(probe)[0])} "
          f"({CLASS_NAMES[int(predict(probe)[0])]})")
