"""
The board package must be consistent with itself, with the verified RTL, and honest
about never having touched a board.

Nothing here tests hardware -- nothing can, without a board. What it tests:

  * the committed notebook is exactly what board/make_notebook.py generates, so
    the register offsets in it came from rtl/ds_defs.vh and not from a hand edit;
  * the notebook states its board-run status truthfully, and any board_run.json
    in the repository was produced by the real pynq on a board;
  * the .hwh puts the core where the notebook will look for it, at 100 MHz;
  * the notebook runs end to end against the software stand-in for pynq;
  * that stand-in is strict enough to be worth running: a notebook that polled
    the RTL's sticky DONE bit instead of BUSY would FAIL against it (negative
    control), because that is the one protocol trap the verified RTL is known to
    set;
  * the checksum manifest matches the files.
"""

import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")
NB = os.path.join(BOARD, "drivesentinel_overlay.ipynb")
HWH = os.path.join(BOARD, "drivesentinel.hwh")
SYS = os.path.join(ROOT, "artifacts", "rtl", "system", "system.json")

needs_nb = pytest.mark.skipif(not os.path.exists(NB), reason="notebook absent")
needs_hwh = pytest.mark.skipif(not os.path.exists(HWH), reason="overlay not built")
needs_windows = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOARD, "test_windows.npz")),
    reason="board test windows absent")


# ---------------------------------------------------------------------------
# honesty
# ---------------------------------------------------------------------------

@needs_nb
def test_notebook_states_its_board_run_truthfully():
    """The header says the notebook ran on a board only if a genuine board run is
    committed, and then points at it; the committed file carries no outputs, so
    the only record of a run is board_run.json itself."""
    nb = json.load(open(NB, encoding="utf-8"))
    first = "".join(nb["cells"][0]["source"])
    p = os.path.join(BOARD, "board_run.json")
    ran = os.path.exists(p) and json.load(open(p)).get("ran_on_hardware") is True
    if ran:
        assert "has been run on a PYNQ-Z2" in first and "board_run.json" in first
        assert "HAS NOT BEEN RUN" not in first
    else:
        assert "has been run on" not in first
    assert nb["metadata"]["drivesentinel"]["board_run"] == "board/board_run.json"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c.get("outputs", []) == [], "committed notebook has saved outputs"


@needs_nb
def test_notebook_cannot_label_a_mock_run_as_hardware():
    """The final cell derives ran_on_hardware from the pynq it imported."""
    src = "".join("".join(c["source"]) for c in
                  json.load(open(NB, encoding="utf-8"))["cells"])
    assert "'ran_on_hardware': ON_HARDWARE" in src
    assert "startswith('MOCK')" in src
    assert "'ran_on_hardware': True" not in src


def test_any_committed_board_run_came_from_a_board():
    p = os.path.join(BOARD, "board_run.json")
    if not os.path.exists(p):
        pytest.skip("no board run yet -- the expected state")
    r = json.load(open(p))
    assert r["ran_on_hardware"] is True
    assert not str(r.get("pynq_version", "")).startswith("MOCK")


@pytest.mark.skipif(not os.path.exists(SYS), reason="overlay not built")
def test_system_json_never_claims_hardware_and_keeps_scopes_apart():
    s = json.load(open(SYS))
    assert s["ran_on_hardware"] is False
    assert s["scope"].startswith("FULL SYSTEM")
    assert "core only" in s["not_comparable_with"]
    if s["bitstream_written"]:
        # a bitstream is only written when the full system closed timing
        assert s["timing_met"] is True
        assert float(s["wns_ns"]) >= 0 and float(s["whs_ns"]) >= 0


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------

@needs_nb
def test_committed_notebook_matches_its_generator(tmp_path):
    before = open(NB, "rb").read()
    subprocess.run([sys.executable, os.path.join(BOARD, "make_notebook.py")],
                   cwd=ROOT, check=True, capture_output=True)
    after = open(NB, "rb").read()
    assert before == after, "board/drivesentinel_overlay.ipynb was edited by hand " \
                            "or is stale; regenerate with board/make_notebook.py"


@needs_nb
def test_notebook_register_offsets_match_ds_defs():
    txt = open(os.path.join(ROOT, "rtl", "ds_defs.vh"), encoding="utf-8").read()
    regs = {m.group(1): int(m.group(2), 16) for m in
            re.finditer(r"`define DS_REG_(\w+)\s+8'h([0-9A-Fa-f]+)", txt)}
    src = "".join("".join(c["source"]) for c in
                  json.load(open(NB, encoding="utf-8"))["cells"])
    for name, off in regs.items():
        assert re.search(rf"REG_{name}\s*=\s*0x{off:02X}\b", src), name
    assert "EXPECTED_BUILD_ID = 0x23EB56BF" in src


@needs_nb
def test_notebook_bincount_is_safe_on_32bit_arm():
    """
    The PYNQ-Z2 is armv7l: its numpy refuses `np.bincount` on an int64 array
    ("Cannot cast array data from dtype('int64') to dtype('int32')"), which the
    board hit on 2026-09-24. The dry run executes on 64-bit, where the same
    call works, so it cannot catch this -- this static check does.
    """
    src = "".join("".join(c["source"]) for c in
                  json.load(open(NB, encoding="utf-8"))["cells"]
                  if c["cell_type"] == "code")
    calls = re.findall(r"np\.bincount\([^\n]*", src)
    assert calls, "expected the class-count print in section 3"
    for call in calls:
        assert ".astype(np.intp)" in call, call


@needs_nb
@needs_windows
def test_notebook_golden_runs_under_the_boards_older_numpy_pad():
    """
    The board's numpy predates 1.17, where `np.pad`'s `mode` became optional;
    golden_reference.py (generated, frozen) omits it, which the board hit on
    2026-09-24. Section 3 supplies the default. Run that cell under a `np.pad`
    with the old signature -- after first checking that golden_reference fails
    under it without the shim, so this test can tell the two apart.
    """
    nb = json.load(open(NB, encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "import golden_reference" in "".join(c["source"]))
    gdir = os.path.join(BOARD, "golden")
    T = np.load(os.path.join(BOARD, "test_windows.npz"))
    real_pad, dont_write = np.pad, sys.dont_write_bytecode

    def old_numpy_pad(array, pad_width, mode, **kwargs):     # numpy < 1.17: mode required
        return real_pad(array, pad_width, mode, **kwargs)

    sys.dont_write_bytecode = True
    np.pad = old_numpy_pad
    sys.modules.pop("golden_reference", None)
    try:
        sys.path.insert(0, gdir)
        import golden_reference as unpatched
        with pytest.raises(TypeError, match="mode"):
            unpatched.predict(T["x_float"][:1].astype(np.float64))
        sys.modules.pop("golden_reference", None)

        ns = {"np": np, "os": os, "sys": sys, "HERE": BOARD}
        exec(compile(cell, "<notebook section 3>", "exec"), ns)
        pred = ns["golden"].predict(T["x_float"][:4].astype(np.float64))
        assert pred.tolist() == T["pred_float"][:4].tolist()
    finally:
        np.pad, sys.dont_write_bytecode = real_pad, dont_write
        sys.modules.pop("golden_reference", None)
        sys.path[:] = [p for p in sys.path if p != gdir]


@needs_hwh
def test_hwh_places_the_core_where_the_notebook_looks():
    h = open(HWH, encoding="utf-8", errors="replace").read()
    m = re.search(r'<MEMRANGE [^>]*BASEVALUE="(0x[0-9A-Fa-f]+)" HIGHNAME="C_HIGHADDR" '
                  r'HIGHVALUE="(0x[0-9A-Fa-f]+)" INSTANCE="ds_0"', h)
    assert m, "ds_0 has no address range in the .hwh"
    base, high = int(m.group(1), 16), int(m.group(2), 16)
    assert base == 0x43C00000
    assert high - base + 1 == 0x1000, "core window is not 4 KB"
    assert 'INSTANCE="axi_dma_0"' in h
    assert re.search(r'PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ" VALUE="100\.0+"', h), \
        "fabric clock in the handoff is not 100 MHz"


def test_wrapper_instantiates_the_verified_core_unchanged():
    w = open(os.path.join(BOARD, "rtl", "ds_pynq_wrap.v"), encoding="utf-8").read()
    assert re.search(r"ds_top\s*#\(\s*\.LANES\s*\(1\),[^)]*\.DEBUG_TRACE\s*\(0\)", w)
    # the wrapper is shell only: no always blocks, no assigns, no arithmetic
    code = re.sub(r"//.*", "", w)
    assert "always" not in code and "assign" not in code


@needs_windows
def test_packaged_windows_agree_between_rtl_model_and_golden():
    T = np.load(os.path.join(BOARD, "test_windows.npz"))
    assert len(T["window"]) == 120
    assert np.array_equal(T["pred_rtl"], T["pred_float"])
    assert np.bincount(T["label"]).tolist() == [40, 40, 40]
    assert int(T["build_id"]) == 0x23EB56BF


def test_mock_replicates_the_observed_hardware_naming_quirk():
    """
    Pins the exact shape a real board produced on 2026-09-24 (`ol.ip_dict` keyed
    'ds_0/s_axi', not 'ds_0'; `ol.ds_0` a bare proxy with no register access),
    so a future edit to the mock cannot quietly go back to being more lenient
    than the hardware it stands in for.
    """
    sys.path.insert(0, os.path.join(BOARD, "dryrun"))
    try:
        import importlib
        pynq = importlib.import_module("pynq")
        importlib.reload(pynq)
        ol = pynq.Overlay(os.path.join(BOARD, "drivesentinel.bit"))
        assert sorted(ol.ip_dict) == ["axi_dma_0", "ds_0/s_axi"]
        assert not (hasattr(ol.ds_0, "read") and hasattr(ol.ds_0, "write"))
        assert hasattr(ol.ds_0.s_axi, "read") and hasattr(ol.ds_0.s_axi, "write")
        info = ol.ip_dict["ds_0/s_axi"]
        assert {"phys_addr", "addr_range"} <= set(info)
    finally:
        sys.path.remove(os.path.join(BOARD, "dryrun"))
        sys.modules.pop("pynq", None)


def test_resolve_mmio_finds_the_core_by_hierarchy_and_by_raw_address():
    """board/mmio_resolve.py against the mock: the hierarchy path it actually
    takes here, and the address-only fallback it would take if that path were
    unavailable -- both must reach the same registers."""
    sys.path.insert(0, os.path.join(BOARD, "dryrun"))
    try:
        import importlib
        pynq = importlib.import_module("pynq")
        importlib.reload(pynq)
        sys.path.insert(0, BOARD)
        try:
            import mmio_resolve
            importlib.reload(mmio_resolve)
            ol = pynq.Overlay(os.path.join(BOARD, "drivesentinel.bit"))
            via_hierarchy = mmio_resolve.resolve_mmio(ol, "ds_0")
            assert via_hierarchy is ol.ds_0.s_axi

            class NoHierarchy:              # only ip_dict -- forces the MMIO fallback
                ip_dict = ol.ip_dict
            via_address = mmio_resolve.resolve_mmio(NoHierarchy(), "ds_0")
            assert via_address.read(ol.ds_0.s_axi.R["MAGIC"]) == \
                via_hierarchy.read(ol.ds_0.s_axi.R["MAGIC"])

            class Empty:
                ip_dict = {}
            with pytest.raises(RuntimeError, match="could not find a register interface"):
                mmio_resolve.resolve_mmio(Empty(), "ds_0")
        finally:
            sys.path.remove(BOARD)
            sys.modules.pop("mmio_resolve", None)
    finally:
        sys.path.remove(os.path.join(BOARD, "dryrun"))
        sys.modules.pop("pynq", None)


def test_manifest_matches_the_files():
    p = os.path.join(BOARD, "MANIFEST.sha256")
    if not os.path.exists(p):
        pytest.skip("manifest absent")
    for line in open(p):
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        with open(os.path.join(BOARD, name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == digest, name


# ---------------------------------------------------------------------------
# the dry run, and proof it can fail
# ---------------------------------------------------------------------------

@needs_nb
@needs_windows
def test_notebook_runs_end_to_end_against_the_mock(tmp_path):
    """
    Runs board/dryrun/run_dryrun.py unchanged, in a subprocess, with its output
    directory redirected to tmp_path. The runner writes dryrun_result.json next
    to itself (`HERE`); pointing HERE at tmp_path keeps the committed
    board/dryrun/dryrun_result.json untouched by the test suite. The mock pynq
    is still imported from board/dryrun, where it lives.
    """
    dry = os.path.join(BOARD, "dryrun")
    driver = "; ".join([
        "import importlib.util, sys",
        f"sys.path.insert(0, {dry!r})",
        f"spec = importlib.util.spec_from_file_location('run_dryrun', {os.path.join(dry, 'run_dryrun.py')!r})",
        "m = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(m)",
        f"m.HERE = {str(tmp_path)!r}",
        "sys.exit(m.main())"])
    r = subprocess.run([sys.executable, "-c", driver],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = json.load(open(tmp_path / "dryrun_result.json"))
    assert out["all_cells_ran"]
    res = out["result"]
    assert res["ran_on_hardware"] is False and res["mock"] is True
    assert res["class_agree_with_golden"] == 120
    assert res["logit_registers_exact"] == 120


@needs_windows
def test_mock_catches_a_notebook_that_polls_the_sticky_done_bit():
    """
    Negative control. The verified RTL clears DONE only on a CTRL.start write, and
    a DMA frame starts an inference without one, so after the first window DONE
    is stale. A notebook that waited on DONE would read the result registers
    while they are still cleared. If the mock let that pass, the dry run above
    would prove nothing about the protocol.
    """
    sys.path.insert(0, os.path.join(BOARD, "dryrun"))
    try:
        import importlib
        pynq = importlib.import_module("pynq")
        importlib.reload(pynq)
        assert pynq.__version__.startswith("MOCK")
        ol = pynq.Overlay("drivesentinel.bit")
        # ol.ds_0 is a bare hierarchy proxy on this mock (matching the real
        # board, board/mmio_resolve.py); the registers are at ol.ds_0.s_axi.
        core, dma = ol.ds_0.s_axi, ol.axi_dma_0
        R, B = core.R, core.B
        T = np.load(os.path.join(BOARD, "test_windows.npz"))
        buf = pynq.allocate(shape=(2560,), dtype=np.uint8)

        def send(i):
            buf[:] = T["x_int8"][i].reshape(-1).view(np.uint8)
            dma.sendchannel.transfer(buf)

        def read_logits():
            return [core.read(R[f"LOGIT{k}"]) for k in range(3)]

        def infer_polling_done(i):
            send(i)
            while not (core.read(R["STATUS"]) >> B["DONE"]) & 1:
                pass
            return core.read(R["CLASS"]), read_logits()

        def infer_busy_protocol(i):
            send(i)
            while not (core.read(R["STATUS"]) >> B["BUSY"]) & 1:
                pass
            while (core.read(R["STATUS"]) >> B["BUSY"]) & 1:
                pass
            return core.read(R["CLASS"]), read_logits()

        def expected(i):
            return int(T["pred_rtl"][i]), [int(v) & 0xFFFFFFFF for v in T["logit_reg"][i]]

        # the notebook's protocol: right on every window
        for i in range(4):
            assert infer_busy_protocol(i) == expected(i)
        # the wrong protocol: the first window can pass, later ones cannot
        wrong = [infer_polling_done(i) != expected(i) for i in range(4, 10)]
        assert any(wrong), "the mock let a DONE-polling notebook pass"
    finally:
        sys.path.remove(os.path.join(BOARD, "dryrun"))
        sys.modules.pop("pynq", None)
