"""
DriveSentinel inference server for the PYNQ-Z2 -- RUN ON THE BOARD, 2026-09-24.

Runs on the board's ARM cores, next to drivesentinel.bit / .hwh. It loads the
overlay, refuses to start unless the core's build-ID register reads
0x23eb56bf, and then serves one bearing window per HTTP request through the
AXI DMA:

    GET  /health   -> {"ok", "magic", "build_id", "config", "expected_cycles",
                       "frontend": {available, fft, error}, "python", "numpy", "scipy"}
    POST /process  body: RAW signal -- 4 s of motor current (2 phases) and
                   vibration at 64 kHz; see decode_raw() for the layout. The
                   board runs the whole DSP front end itself (board/frontend.py),
                   quantises, and sends every 1 s window through the FPGA:
                   -> {"windows": [{class, label, logits, cycles, fpga_ms, rpm,
                       int8 (base64), view (the vibration envelope, 256 bins)}],
                       "timings": {parse_ms, decimate_ms, envelope_ms, spectra_ms,
                       quantise_ms, fpga_ms, total_ms}, ...}
    POST /infer    body: one window, either
                     2,560 bytes  int8, channel-major (5 x 512), already quantised
                    10,240 bytes  float32, channel-major, the DSP front end's output;
                                  quantised here by the PS quantiser (D-4), exactly
                                  as the notebook does
                   -> {"class": 0..2, "label": ..., "logits": [int32 x 3],
                       "cycles": int, "engine": "FPGA (PYNQ-Z2)", "build_id": "0x..."}

Start it on the board (see board/SERVER.md):

    sudo -E python3 server.py --host 0.0.0.0 --port 8765

PROTOCOL. A completed DMA frame starts an inference automatically. The STATUS
DONE bit is cleared only by a CTRL.start write, so after the first window it
stays set and cannot tell one inference from the next (claims_audit.md 1.21,
handoff 6.7). This server therefore does what the notebook does: wait for BUSY
to rise, then to fall, and report CYCLES so the caller can check that a whole
inference ran (the simulated count is 1,786,724 for every window).

REGISTER ACCESS. `ol.ds_0` is not itself the register file -- see
`board/mmio_resolve.py` for why (a real, board-observed naming quirk) and what
`resolve_mmio` tries instead. Bundle `mmio_resolve.py` next to this file on the
board (`board/SERVER.md`).

Dependencies: numpy and pynq, both on every PYNQ image; the standard library
for HTTP. One inference at a time -- the core has one engine -- so requests
are serialised by a lock.

ON THE BOARD (2026-09-24): this file (as of fc67f0d) served the web app from a
PYNQ-Z2 -- every bearing window, and on 600 of them answers bit-exact against
the software model (artifacts/webapp/board_server_check.json, claims_audit
1.27). Its build-ID REFUSAL path has been exercised only against the software
stand-in (tests/test_board_server.py), since the real board's ID matched.
"""

import argparse
import base64
import json
import os
import platform
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

try:
    from http.server import ThreadingHTTPServer
except ImportError:                  # Python 3.6 -- the PYNQ 2.5 image; added in 3.7
    from http.server import HTTPServer
    from socketserver import ThreadingMixIn

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The DSP front end needs scipy (sosfilt); the rest of this server does not. If
# it cannot load, /infer still works and /process says why it cannot.
sys.path.insert(0, HERE)
try:
    import frontend as FRONTEND
    FRONTEND_ERROR = None
except Exception as _e:                       # scipy absent, params file missing, ...
    FRONTEND, FRONTEND_ERROR = None, "%s: %s" % (type(_e).__name__, _e)

RAW_MAX_BYTES = 16 * 1024 * 1024              # 3 channels x 4 s x 64 kHz x float64 = 6.1 MB
VIEW_CHANNEL = 2                              # vibration envelope, 500-2500 Hz: the page's view


def versions() -> dict:
    try:
        import scipy
        scipy_v = scipy.__version__
    except ImportError:
        scipy_v = None
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_v,
            "frontend": {"available": FRONTEND is not None,
                         "fft": FRONTEND.FFT_BACKEND if FRONTEND else None,
                         "workers": FRONTEND.WORKERS if FRONTEND else 0,
                         "error": FRONTEND_ERROR},
            "machine": machine()}


def _sys(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def machine() -> dict:
    """What a timing needs beside it (docs/frontend_timing_plan.md 4.1): the CPU, its
    governor and clock, and the die temperature -- None where the OS does not say."""
    try:
        import pynq
        pynq_v = getattr(pynq, "__version__", None)
    except ImportError:
        pynq_v = None
    temp = _sys("/sys/class/thermal/thermal_zone0/temp")
    freq = _sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return {"arch": platform.machine(), "cpus": os.cpu_count(), "pynq": pynq_v,
            "governor": _sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "cpu0_mhz": int(freq) / 1000.0 if freq and freq.isdigit() else None,
            "temp_c": int(temp) / 1000.0 if temp and temp.lstrip("-").isdigit() else None}


def decode_raw(body: bytes):
    """
    [4-byte little-endian header length][header JSON][channel arrays], where the
    header is {"fs": 64000.0, "dtype": "<f8", "n": [n_current_1, n_current_2,
    n_vibration]} and the arrays follow in that order, little-endian.
    """
    if len(body) < 4:
        raise ValueError("raw body too short")
    (hlen,) = struct.unpack("<I", body[:4])
    if hlen > 4096 or 4 + hlen > len(body):
        raise ValueError("bad raw header length %d" % hlen)
    header = json.loads(body[4:4 + hlen].decode("utf-8"))
    dtype = np.dtype(header.get("dtype", "<f8"))
    if dtype.kind != "f":
        raise ValueError("raw samples must be floating point")
    ns = [int(n) for n in header["n"]]
    if len(ns) != 3 or min(ns) <= 0:
        raise ValueError("expected three channels")
    if 4 + hlen + sum(ns) * dtype.itemsize != len(body):
        raise ValueError("raw body is %d bytes; the header describes %d"
                         % (len(body), 4 + hlen + sum(ns) * dtype.itemsize))
    out, off = [], 4 + hlen
    for n in ns:
        out.append(np.frombuffer(body, dtype=dtype, count=n, offset=off).astype(np.float64))
        off += n * dtype.itemsize
    return float(header["fs"]), out

# From rtl/ds_defs.vh via the generated notebook (board/drivesentinel_overlay.ipynb
# cell 2). Copied, not re-derived, because the board has no rtl/ folder;
# tests/test_board_server.py asserts they still equal ds_defs.vh.
REG = {"CTRL": 0x00, "STATUS": 0x04, "CLASS": 0x08, "LOGIT0": 0x0C, "LOGIT1": 0x10,
       "LOGIT2": 0x14, "BUILDID": 0x18, "CYCLES": 0x1C, "MAGIC": 0x20, "CONFIG": 0x24}
ST = {"DONE": 0, "BUSY": 1, "IDLE": 2, "IN_VALID": 3, "CRC_DONE": 4}
MAGIC = 0x44533031                 # 'DS01'
EXPECTED_BUILD_ID = 0x23EB56BF
EXPECTED_CYCLES = 1786724          # simulated, every window
EXPECTED_CONFIG = (26, 1, 16)      # frac width, lanes, logit shift
CLASSES = ["healthy", "inner_race", "outer_race"]
ENGINE = "FPGA (PYNQ-Z2)"
WINDOW_SHAPE = (5, 512)
INT8_BYTES = 5 * 512
FLOAT_BYTES = 5 * 512 * 4


class BuildIdMismatch(RuntimeError):
    pass


def signed32(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


class Accelerator:
    """The overlay, checked, and the busy-bit inference protocol."""

    def __init__(self, bitfile: str, overlay_factory=None, allocate=None):
        if overlay_factory is None or allocate is None:
            from pynq import Overlay, allocate as _alloc   # on the board only
            overlay_factory = overlay_factory or Overlay
            allocate = allocate or _alloc
        self.ol = overlay_factory(bitfile)
        sys.path.insert(0, HERE)
        from mmio_resolve import resolve_mmio      # bundled next to this file
        self.core = resolve_mmio(self.ol, "ds_0")
        self.dma = self.ol.axi_dma_0
        self.buf = allocate(shape=(INT8_BYTES,), dtype=np.uint8)
        self.lock = threading.Lock()
        self.identity = self._check()
        self._quant = None

    def rd(self, off: int) -> int:
        return int(self.core.read(off))

    def _check(self) -> dict:
        t0 = time.time()
        while not (self.rd(REG["STATUS"]) >> ST["CRC_DONE"]) & 1:
            if time.time() - t0 > 1.0:
                raise RuntimeError("build-ID walk did not finish within 1 s")
        magic, build_id, config = self.rd(REG["MAGIC"]), self.rd(REG["BUILDID"]), \
            self.rd(REG["CONFIG"])
        cfg = (config & 0xFF, (config >> 8) & 0xFF, (config >> 16) & 0xFF)
        if magic != MAGIC:
            raise RuntimeError(f"MAGIC 0x{magic:08x}: not the DriveSentinel core")
        if build_id != EXPECTED_BUILD_ID:
            raise BuildIdMismatch(
                f"BUILD_ID 0x{build_id:08x}, expected 0x{EXPECTED_BUILD_ID:08x}: the "
                f"weights in the fabric are not the export this package claims. "
                f"Refusing to serve.")
        if cfg != EXPECTED_CONFIG:
            raise RuntimeError(f"CONFIG {cfg}, expected {EXPECTED_CONFIG}")
        return {"magic": f"0x{magic:08x}", "build_id": f"0x{build_id:08x}",
                "config": {"frac_width": cfg[0], "lanes": cfg[1], "logit_shift": cfg[2]}}

    # -- the PS-side quantiser (docs/rtl_declarations.md D-4), as in the notebook
    def quantise(self, x: np.ndarray) -> np.ndarray:
        if self._quant is None:
            sys.path.insert(0, os.path.join(HERE, "golden"))
            import golden_reference as golden          # numpy only
            self._quant = (golden.INPUT_MEAN, golden.INPUT_STD, golden.LAYERS[0]["s_x"])
        mean, std, s_x = self._quant
        a = (np.asarray(x, np.float64) - mean) / std
        return np.clip(np.rint(a / s_x), -128, 127).astype(np.int8)

    def infer(self, window_int8: np.ndarray, timeout_s: float = 1.0) -> dict:
        w = np.ascontiguousarray(window_int8, dtype=np.int8).reshape(WINDOW_SHAPE)
        with self.lock:
            self.buf[:] = w.reshape(-1).view(np.uint8)
            if hasattr(self.buf, "flush"):
                self.buf.flush()
            self.dma.sendchannel.transfer(self.buf)
            self.dma.sendchannel.wait()
            t0 = time.time()
            while not (self.rd(REG["STATUS"]) >> ST["BUSY"]) & 1:     # wait for the start
                if time.time() - t0 > 0.05:
                    raise RuntimeError("inference did not start after the frame")
            while (self.rd(REG["STATUS"]) >> ST["BUSY"]) & 1:         # wait for the end
                if time.time() - t0 > timeout_s:
                    raise RuntimeError("inference did not finish")
            cls = self.rd(REG["CLASS"]) & 0x3
            logits = [signed32(self.rd(REG[k])) for k in ("LOGIT0", "LOGIT1", "LOGIT2")]
            cycles = self.rd(REG["CYCLES"])
        return {"class": cls, "label": CLASSES[cls], "logits": logits, "cycles": cycles,
                "engine": ENGINE, "build_id": self.identity["build_id"]}

    def process_raw(self, body: bytes) -> dict:
        """The whole chain on the board: raw signal -> DSP (ARM) -> quantise ->
        FPGA, per window, with each stage timed."""
        if FRONTEND is None:
            raise RuntimeError("the DSP front end is not available on this board: %s"
                               % FRONTEND_ERROR)
        t0 = time.perf_counter()
        fs, (c1, c2, vib) = decode_raw(body)
        if abs(fs - FRONTEND.FS) > 1e-6:
            raise ValueError("signal is at %s Hz; the front end expects %s Hz" % (fs, FRONTEND.FS))
        t1 = time.perf_counter()
        specs, diags, dsp_t = FRONTEND.process_recording(c1, c2, vib)
        t2 = time.perf_counter()
        q = [self.quantise(w) for w in specs]
        t3 = time.perf_counter()
        windows = []
        for w, x, d in zip(specs, q, diags):
            f0 = time.perf_counter()
            r = self.infer(x)
            view = w[VIEW_CHANNEL].reshape(-1, 2).mean(axis=1)
            r.update({"fpga_ms": (time.perf_counter() - f0) * 1e3,
                      "rpm": d["rpm"], "f_elec_hz": d["f_elec_hz"],
                      "int8": base64.b64encode(x.tobytes()).decode("ascii"),
                      "view": [round(float(v), 4) for v in view]})
            windows.append(r)
        t4 = time.perf_counter()
        timings = {"parse_ms": (t1 - t0) * 1e3, "quantise_ms": (t3 - t2) * 1e3,
                   "fpga_ms": (t4 - t3) * 1e3, "total_ms": (t4 - t0) * 1e3}
        timings.update(dsp_t)
        return {"windows": windows, "timings": timings, "samples": [c1.size, c2.size, vib.size],
                "fs": fs, "engine": ENGINE, "build_id": self.identity["build_id"],
                "fft": FRONTEND.FFT_BACKEND}

    def infer_bytes(self, body: bytes) -> dict:
        if len(body) == INT8_BYTES:
            x = np.frombuffer(body, dtype=np.int8)
        elif len(body) == FLOAT_BYTES:
            x = self.quantise(np.frombuffer(body, dtype="<f4").reshape(WINDOW_SHAPE))
        else:
            raise ValueError(f"window must be {INT8_BYTES} bytes (int8) or "
                             f"{FLOAT_BYTES} bytes (float32), got {len(body)}")
        return self.infer(x)


def make_handler(acc: Accelerator):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DriveSentinelBoard/1"

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                out = {"ok": True, "engine": ENGINE, "expected_cycles": EXPECTED_CYCLES}
                out.update(acc.identity)
                out.update(versions())
                return self._json(200, out)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/infer", "/process"):
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            limit = RAW_MAX_BYTES if self.path == "/process" else FLOAT_BYTES
            if n > limit:
                return self._json(413, {"error": "body too large"})
            try:
                body = self.rfile.read(n)
                if self.path == "/process":
                    return self._json(200, acc.process_raw(body))
                return self._json(200, acc.infer_bytes(body))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except RuntimeError as e:
                return self._json(503, {"error": str(e)})
            except Exception as e:              # say what broke, rather than drop the socket
                return self._json(500, {"error": "%s: %s" % (type(e).__name__, e)})

        def log_message(self, fmt, *args):      # quiet by default
            if os.environ.get("DS_BOARD_LOG"):
                super().log_message(fmt, *args)
    return Handler


def serve(acc: Accelerator, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(acc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "drivesentinel.bit"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workers", type=int, default=2,
                    help="processes for the DSP front end (the Zynq-7020 has 2 ARM cores)")
    a = ap.parse_args(argv)
    if FRONTEND is not None:
        # Forked now, before the overlay is loaded, so the workers hold no pynq state.
        FRONTEND.start_workers(a.workers)
    try:
        acc = Accelerator(a.bit)
    except BuildIdMismatch as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    print(f"DriveSentinel core ok: {acc.identity}", flush=True)
    v = versions()
    print("python %s, numpy %s, scipy %s; DSP front end: %s" % (
        v["python"], v["numpy"], v["scipy"],
        ("available, FFT from %s, %d worker(s)" % (v["frontend"]["fft"], v["frontend"]["workers"]))
        if v["frontend"]["available"]
        else "NOT AVAILABLE -- " + str(v["frontend"]["error"])), flush=True)
    httpd = serve(acc, a.host, a.port)
    print(f"serving on http://{a.host}:{a.port}  (GET /health, POST /process, POST /infer)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if FRONTEND is not None:
            FRONTEND.stop_workers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
