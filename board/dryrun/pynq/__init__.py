"""
A SOFTWARE STAND-IN FOR `pynq`. NOT HARDWARE. NOT A SIMULATION OF THE FPGA.

Exists for one purpose: to execute board/drivesentinel_overlay.ipynb end to end
on a development machine, so that a bug in the NOTEBOOK -- a wrong register
offset, a signed/unsigned slip in the logit decode, a broken BUSY/DONE protocol,
a shape error in the DMA buffer -- is found here and not on the board.

It cannot find a bug in the hardware, and nothing it produces is evidence about
the hardware. The "core" behind it is drivesentinel.rtl.model, the bit-accurate
Python model, so class and logit agreement with the golden reference is true by
construction here. What is NOT true by construction, and is therefore what this
mock actually tests:

  * the notebook reads the right offsets for MAGIC, BUILD_ID, CONFIG, CLASS,
    LOGIT0..2, CYCLES, STATUS (the mock serves them from the same ds_defs.vh
    offsets the RTL decodes, parsed independently of the notebook generator);
  * the notebook's BUSY-rise-then-fall protocol works against the RTL's real
    DONE semantics -- DONE here is sticky exactly as in ds_axil_ctrl.v, set by
    the first inference and cleared only by a CTRL.start write, so a notebook
    that polled DONE would return stale results and be caught;
  * the DMA buffer is laid out channel-major, 2,560 bytes, as ds_axis_in.v
    expects (the mock rebuilds the (5, 512) window from the raw bytes);
  * negative logits survive the 32-bit register round trip;
  * the notebook resolves the core's registers through whatever path PYNQ
    actually exposes them by, rather than assuming `ol.ds_0` is itself the
    register file.

THE LAST ONE IS NOT A GUESS. `board/build_overlay.tcl` adds the core as a bare
RTL module reference (`create_bd_cell -type module -reference ds_pynq_wrap
ds_0`), never packaged as an IP-XACT core, and a real board run on 2026-09-24
showed PYNQ folding its one AXI4-Lite interface into `ip_dict['ds_0/s_axi']`
rather than `ip_dict['ds_0']`, with `ol.ds_0` a bare hierarchy proxy that has
no `.read()`/`.write()` of its own (`board/mmio_resolve.py` has the full
traceback and the fix). This mock reproduces exactly that shape below, so a
notebook that assumes a flat `ol.ds_0` fails the dry run instead of the board.

CYCLES returns the simulated 1,786,724: this mock does not model time, and its
timing cell output is meaningless and labelled so by board/dryrun/run_dryrun.py.
"""

import json
import os
import re
import sys
import time

import numpy as np

__version__ = "MOCK-not-pynq"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)


def _regs():
    txt = open(os.path.join(_ROOT, "rtl", "ds_defs.vh"), encoding="utf-8").read()
    r = {m.group(1): int(m.group(2), 16) for m in
         re.finditer(r"`define DS_REG_(\w+)\s+8'h([0-9A-Fa-f]+)", txt)}
    b = {m.group(1): int(m.group(2)) for m in
         re.finditer(r"`define DS_ST_(\w+)\s+(\d+)", txt)}
    magic = int(re.search(r"`define DS_MAGIC\s+32'h([0-9A-Fa-f]+)", txt).group(1), 16)
    return r, b, magic


class _Buffer(np.ndarray):
    def flush(self):
        pass

    def invalidate(self):
        pass


def allocate(shape, dtype):
    return np.zeros(shape, dtype=dtype).view(_Buffer)


class _Hierarchy:
    """
    A bare PYNQ-style hierarchy proxy: attribute access to named children only,
    no register access of its own -- exactly what `ol.ds_0` turned out to be on
    the real board (see the module docstring). Accessing an unknown attribute
    (in particular `.read`/`.write`) raises the ordinary `AttributeError` a
    plain object raises, which is all `mmio_resolve.resolve_mmio`'s `hasattr`
    checks need; it does not try to reproduce PYNQ's own wording.
    """

    def __init__(self, **children):
        self.__dict__.update(children)


_mmio_devices = {}      # phys_addr -> a mock device with .read(off)/.write(off, v)


class MMIO:
    """
    Stand-in for `pynq.MMIO`, for `mmio_resolve.resolve_mmio`'s address-based
    fallback path. Real `pynq.MMIO` maps any physical address; this mock has no
    real memory to map, so it only knows the addresses `Overlay.__init__` below
    handed out -- which is every address this dry run could plausibly ask for.
    """

    def __init__(self, base_addr, length=4):
        self.base_addr, self.length = base_addr, length
        try:
            self._dev = _mmio_devices[base_addr]
        except KeyError:
            raise RuntimeError(
                f"MOCK MMIO: no mock device at 0x{base_addr:08x}; known: "
                f"{[hex(a) for a in _mmio_devices]}")

    def read(self, offset=0, length=4):
        return self._dev.read(offset)

    def write(self, offset, value):
        self._dev.write(offset, value)


class _Core:
    """The register file of ds_axil_ctrl.v, with the model standing in for the engine."""

    BUSY_READS = 3      # STATUS reads that report BUSY after a frame

    def __init__(self):
        from drivesentinel.rtl import model as M
        self.M = M
        self.exp = M.Export()
        p = json.load(open(os.path.join(_ROOT, "artifacts", "rtl", "rtl_params.json")))
        self.F, self.shift, self.build_id = p["frac_width"], p["logit_shift"], \
            p["expected_build_id"]
        self.R, self.B, self.magic = _regs()
        self.done = 0
        self.busy_left = 0
        self.cls = 0
        self.logits = [0, 0, 0]
        self.cycles = 0
        self.pending = (0, [0, 0, 0])

    # -- the DMA delivers a frame: ds_axis_in auto-starts the engine ----------
    def frame(self, raw_bytes):
        assert raw_bytes.size == 2560, f"frame is {raw_bytes.size} bytes, not 2,560"
        x = raw_bytes.view(np.int8).reshape(5, 512)          # channel-major
        r = self.M.forward(x[None], self.exp, requant="int", frac_width=self.F)
        regs = (r["logits_raw"][0] >> np.int64(self.shift)).astype(np.int64)
        # The real core clears CLASS and LOGIT0..2 when an inference starts
        # (ds_fc's `clr`) and writes them only at the very end. Mirror that: the
        # results are held back until BUSY falls. A mock that published them at
        # once would let a notebook that polls the stale DONE bit pass here and
        # fail on the board -- exactly the bug this mock exists to catch.
        self.pending = (int(r["pred"][0]), [int(v) & 0xFFFFFFFF for v in regs])
        self.cls, self.logits = 0, [0, 0, 0]
        self.cycles = 0          # the counter restarts at the start pulse
        self.busy_left = self.BUSY_READS
        # NB: auto-start does NOT clear DONE -- the real RTL's behaviour

    def read(self, off):
        R, B = self.R, self.B
        if off == R["STATUS"]:
            busy = 1 if self.busy_left > 0 else 0
            if self.busy_left > 0:
                self.busy_left -= 1
                if self.busy_left == 0:
                    self.done = 1
                    self.cls, self.logits = self.pending
                    self.cycles = 1786724     # simulated figure, not modelled time
            return (self.done << B["DONE"]) | (busy << B["BUSY"]) | \
                   ((1 - busy) << B["IDLE"]) | (1 << B["IN_VALID"]) | \
                   (1 << B["CRC_DONE"])
        return {
            R["CTRL"]: 0, R["CLASS"]: self.cls,
            R["LOGIT0"]: self.logits[0], R["LOGIT1"]: self.logits[1],
            R["LOGIT2"]: self.logits[2], R["BUILDID"]: self.build_id,
            R["CYCLES"]: self.cycles, R["MAGIC"]: self.magic,
            R["CONFIG"]: (self.shift << 16) | (1 << 8) | self.F,
        }.get(off, 0xDEADBEEF)

    def write(self, off, v):
        if off == self.R["CTRL"] and v & 1:
            self.done = 0


class _Channel:
    def __init__(self, core):
        self.core = core

    def transfer(self, buf):
        self.core.frame(np.asarray(buf).view(np.uint8).reshape(-1))

    def wait(self):
        pass


class _DMA:
    def __init__(self, core):
        self.sendchannel = _Channel(core)


class Overlay:
    def __init__(self, bitfile):
        self.bitfile = bitfile
        hwh = os.path.splitext(bitfile)[0] + ".hwh"
        self.found_bit = os.path.exists(bitfile)
        self.found_hwh = os.path.exists(hwh)
        base, rng = 0x43C00000, 0x1000
        amap = os.path.join(_ROOT, "artifacts", "rtl", "system", "address_map.txt")
        if os.path.exists(amap):
            for line in open(amap):
                if "ds_0" in line:
                    parts = line.split()
                    base, rng = int(parts[1], 16), int(parts[3], 16)
        self._core = _Core()
        # 'ds_0' is a hierarchy with no registers of its own; the register
        # file is one level down, at 'ds_0.s_axi' -- matching the real board
        # (module docstring). A notebook or server that reads 'ol.ds_0'
        # directly fails here exactly as it failed there.
        self.ds_0 = _Hierarchy(s_axi=self._core)
        self.axi_dma_0 = _DMA(self._core)
        self.ip_dict = {"ds_0/s_axi": {"phys_addr": base, "addr_range": rng},
                        "axi_dma_0": {"phys_addr": 0x40400000, "addr_range": 0x10000}}
        _mmio_devices[base] = self._core
