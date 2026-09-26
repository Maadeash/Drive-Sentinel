"""
tests/test_model_export.py -- the graph, the fold, the quantiser, the memories.

The chain these tests protect is: trained model -> BatchNorm folded -> INT8
quantised -> written to .mem -> read back by the golden reference.  Every link
is a place where a silent transcription or scaling error would produce a model
that trains well and deploys wrong, so each link gets its own assertion.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from drivesentinel import config as C
from drivesentinel.model import build_model, fold_batchnorm, describe
from drivesentinel import quantize as Q
from drivesentinel import export as E


# ===========================================================================
# graph
# ===========================================================================

def test_shapes_and_budget():
    m = build_model()
    x = torch.randn(3, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)
    assert m(x).shape == (3, len(C.LABELS))
    # The FPGA budget in the architecture note assumed these two numbers.
    assert m.parameter_count() < 40_000, m.parameter_count()
    assert m.mac_count() < 2_000_000, m.mac_count()


def test_no_dropout_in_graph():
    """Train and deploy graphs must be identical -- see model.py docstring."""
    assert not any("Dropout" in type(mod).__name__ for mod in build_model().modules())


# ===========================================================================
# BatchNorm folding
# ===========================================================================

def test_folding_is_numerically_exact():
    m = build_model().eval()
    # random BN statistics, otherwise the fold is trivially the identity
    for stage in m.features:
        stage.bn.running_mean.normal_(0, 0.5)
        stage.bn.running_var.uniform_(0.5, 2.0)
        stage.bn.weight.data.uniform_(0.5, 1.5)
        stage.bn.bias.data.normal_(0, 0.3)

    x = torch.randn(8, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)
    with torch.no_grad():
        before, after = m(x), fold_batchnorm(m)(x)
    assert torch.allclose(before, after, atol=1e-5), \
        f"fold changed the function by {(before - after).abs().max():.2e}"


def test_folded_graph_has_no_batchnorm():
    folded = fold_batchnorm(build_model().eval())
    assert not any("BatchNorm" in type(mod).__name__ for mod in folded.modules())


# ===========================================================================
# quantiser
# ===========================================================================

def _folded_and_quantised(seed: int = 0):
    torch.manual_seed(seed)
    m = fold_batchnorm(build_model().eval())
    calib = torch.randn(128, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)
    layers, meta = Q.quantize_model(m, calib)
    return m, layers, meta, calib


def test_weights_are_int8_and_use_the_range():
    _, layers, _, _ = _folded_and_quantised()
    for L in layers:
        assert L.W_int8.dtype == np.int8
        assert L.b_int32.dtype == np.int32
        # per-channel scaling should push at least one weight per channel near
        # the rail; if not, the range is being wasted
        peak = np.abs(L.W_int8.reshape(L.n_out, -1)).max(axis=1)
        assert (peak >= 100).all(), f"{L.name}: under-used INT8 range {peak.min()}"


def test_per_channel_scales_have_one_per_output():
    _, layers, _, _ = _folded_and_quantised()
    for L in layers:
        assert L.s_W.shape == (L.n_out,), f"{L.name}: {L.s_W.shape}"


def test_accumulator_fits_int32():
    _, layers, meta, _ = _folded_and_quantised()
    for name, a in meta["accumulator"].items():
        assert a["fits_int32"], f"{name} overflows int32: {a}"
        assert a["bits_needed"] <= 32


def test_int8_tracks_float():
    """The whole point of the quantiser: same decisions as the float model."""
    m, layers, _, _ = _folded_and_quantised()
    x = np.random.default_rng(0).normal(size=(64, C.N_INPUT_CHANNELS, C.N_ORDER_BINS))
    a = Q.agreement(m, layers, x)
    assert a["argmax_agreement"] >= 0.98, a


def test_quantize_folds_defensively():
    """Passing an UNfolded model must not silently calibrate the wrong tensors."""
    m = build_model().eval()
    calib = torch.randn(32, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)
    layers, _ = Q.quantize_model(m, calib)          # not folded by the caller
    x = np.random.default_rng(1).normal(size=(32, C.N_INPUT_CHANNELS, C.N_ORDER_BINS))
    a = Q.agreement(fold_batchnorm(m), layers, x)
    assert a["argmax_agreement"] >= 0.98, a


# ===========================================================================
# .mem round-trip -- the transcription link
# ===========================================================================

@pytest.mark.parametrize("values", [
    np.array([0, 1, -1, 127, -128, 42, -99], dtype=np.int8),
    np.arange(-128, 128, dtype=np.int8),
])
def test_int8_mem_roundtrip(values):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.mem")
        E.write_mem_int8(p, values)
        back = E.read_mem_int8(p, values.shape)
    assert np.array_equal(values, back)


def test_int32_mem_roundtrip():
    values = np.array([0, 1, -1, 2 ** 31 - 1, -2 ** 31, 123456, -987654], dtype=np.int32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.mem")
        E.write_mem_int32(p, values)
        back = E.read_mem_int32(p, values.shape)
    assert np.array_equal(values, back)


def test_mem_file_is_readmemh_shaped():
    """One bare hex token per line, fixed width -- what $readmemh expects."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.mem")
        E.write_mem_int8(p, np.array([-1, 0, 127], dtype=np.int8))
        lines = open(p).read().strip().split("\n")
    assert lines == ["ff", "00", "7f"]
    assert all(len(ln) == 2 and not ln.startswith("0x") for ln in lines)


# ===========================================================================
# end to end: export -> generated golden reference -> same answers
# ===========================================================================

def test_export_and_golden_reference_agree():
    import importlib.util

    torch.manual_seed(3)
    model = build_model().eval()
    rng = np.random.default_rng(5)
    calib = rng.normal(size=(256, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)).astype(np.float32)
    probe = rng.normal(size=(64, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)).astype(np.float32)

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "norm_mean": np.zeros(C.N_INPUT_CHANNELS),
            "norm_std": np.ones(C.N_INPUT_CHANNELS),
            "labels": C.LABELS, "model_config": C.MODEL_CONFIG,
        }, ckpt)

        out = os.path.join(d, "export")
        report = E.export(ckpt_path=ckpt, out_dir=out, calib=calib, verify=probe)

        for f in ("scales.json", "input_norm.json", "golden_reference.py",
                  "layer_0_W.mem", "layer_4_b.mem"):
            assert os.path.exists(os.path.join(out, f)), f"missing {f}"

        spec = importlib.util.spec_from_file_location(
            "golden_ref_under_test", os.path.join(out, "golden_reference.py"))
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)

        # the generated file, reading .mem from disk, must match the in-memory
        # quantiser it was generated from
        folded = fold_batchnorm(model)
        layers, _ = Q.quantize_model(folded, torch.from_numpy(calib))
        ours = Q.int8_forward(layers, probe).argmax(1)
        theirs = golden.predict(probe)

    assert np.array_equal(ours, theirs), (
        f"generated golden reference disagrees with the exporter on "
        f"{(ours != theirs).sum()}/{len(ours)} samples"
    )
    assert report["agreement"]["argmax_agreement"] >= 0.98


def test_export_rejects_per_bin_normalisation():
    """Per-bin statistics cannot be a cheap hardware stage; fail loudly."""
    model = build_model().eval()
    rng = np.random.default_rng(7)
    calib = rng.normal(size=(32, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "norm_mean": np.zeros((1, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)),
            "norm_std": np.ones((1, C.N_INPUT_CHANNELS, C.N_ORDER_BINS)),
            "labels": C.LABELS, "model_config": C.MODEL_CONFIG,
        }, ckpt)
        with pytest.raises(AssertionError, match="per-channel"):
            E.export(ckpt_path=ckpt, out_dir=os.path.join(d, "x"), calib=calib)
