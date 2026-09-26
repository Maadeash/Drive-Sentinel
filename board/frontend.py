"""
The bearing DSP front end, running on the PYNQ-Z2's ARM cores.

Raw 64 kHz motor current (two phases) and vibration in; the model's input out:
one (5, 512) float32 picture per 1 s window, 0.5 s hop -- 7 windows per 4 s of
signal. This is drivesentinel/dsp.py `process_recording`, ported line for line,
so the board computes what built the training cache instead of receiving it:

    raw 64 kHz
      -> decimate (current / 16, vibration / 2), anti-aliased
      -> shaft speed from the current spectrum (never a tachometer)
      -> current: supply-notched sideband fold + raw order spectrum
      -> vibration: band-pass (low, high) -> Hilbert envelope -> order spectrum,
         plus a log-frequency raw spectrum
      -> median-normalise + log1p, per channel

WHAT IS DIFFERENT FROM dsp.py, AND WHY IT IS NOT A DIFFERENCE IN THE MATHS
--------------------------------------------------------------------------
* Constants and filters come from frontend_params.json, generated on the laptop
  from config.py by make_frontend_params.py (a test pins the file to it).
* scipy.signal.decimate / sosfiltfilt / hilbert are re-expressed with the one
  compiled routine the board's older scipy is sure to share with the laptop's,
  `sosfilt`, plus numpy: odd-extension padding, forward and backward passes
  from the steady-state initial state, exactly scipy's sosfiltfilt; and the
  FFT form of the analytic signal, exactly scipy's hilbert.
* The FFT is scipy.fft when the board's scipy has it (1.4+), else numpy.fft --
  with Bluestein's algorithm for lengths with a large prime factor, which
  numpy 1.13 would otherwise take hours over (see use_fft below). Both compute
  the same transform; they can differ in the last bits.
  tests/test_board_frontend.py measures how often that changes a quantised
  input byte, and scripts/webapp/check_board_frontend.py measures it on the
  board against the laptop, recording by recording.

Python 3.6 / numpy 1.13 / 32-bit ARM compatible: no postponed annotations, no
dtype-sensitive bincounts, explicit dtypes throughout.
"""

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# FFT. On the board, numpy.fft needs help with prime lengths.
# ---------------------------------------------------------------------------
# The Hilbert envelope transforms the whole decimated vibration signal at once:
# 128,001 samples = 3 x 42,667, and 42,667 is prime. scipy.fft (pocketfft) handles
# a large prime factor with Bluestein's algorithm; numpy 1.13's fftpack, on the
# board, does not -- it is O(n x p) there, ~5e9 complex operations per transform,
# four per 4 s of signal. Measured on the PYNQ-Z2 on 2026-09-24: one data file did
# not finish in 280 s. So on the numpy path this module applies Bluestein itself:
# the same DFT, computed as a convolution through power-of-two FFTs. Lengths whose
# prime factors are all small go straight to numpy, as before.

SMOOTH = 64                    # largest prime factor numpy's fftpack is left to handle
_CHIRPS = {}


def _largest_prime_factor(n):
    p, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            p, n = d, n // d
        d += 1
    return max(p, n)


def _chirp(n):
    c = _CHIRPS.get(n)
    if c is None:
        k = np.arange(n, dtype=np.int64)          # int64 explicitly: k*k overflows int32
        w = np.exp(-1j * np.pi * ((k * k) % (2 * n)).astype(np.float64) / n)
        m = 1 << int(np.ceil(np.log2(2 * n - 1)))
        b = np.zeros(m, dtype=np.complex128)
        b[:n] = np.conj(w)
        b[m - n + 1:] = np.conj(w[1:])[::-1]
        c = (w, m, np.fft.fft(b))
        _CHIRPS[n] = c
    return c


def bluestein_fft(x):
    """The length-n DFT of x through power-of-two FFTs (Bluestein / chirp-z)."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    w, m, fb = _chirp(n)
    a = np.zeros(m, dtype=np.complex128)
    a[:n] = x * w
    return w * np.fft.ifft(np.fft.fft(a) * fb)[:n]


def _np_fft(x):
    return np.fft.fft(x) if _largest_prime_factor(x.size) <= SMOOTH else bluestein_fft(x)


def _np_ifft(x):
    if _largest_prime_factor(x.size) <= SMOOTH:
        return np.fft.ifft(x)
    return np.conj(bluestein_fft(np.conj(x))) / x.size


def _np_rfft(x):
    if _largest_prime_factor(x.size) <= SMOOTH:
        return np.fft.rfft(x)
    return bluestein_fft(x)[:x.size // 2 + 1]


def use_fft(backend):
    """Select the FFT: "scipy.fft" (the laptop's, when the board has it) or
    "numpy.fft" (numpy, with Bluestein for long prime factors)."""
    global _fft, _ifft, _rfft, FFT_BACKEND
    if backend == "scipy.fft":
        from scipy.fft import fft as _fft, ifft as _ifft, rfft as _rfft
    elif backend == "numpy.fft":
        _fft, _ifft, _rfft = _np_fft, _np_ifft, _np_rfft
    else:
        raise ValueError(backend)
    FFT_BACKEND = backend


try:                                        # the laptop's FFT, when the board has it
    use_fft("scipy.fft")
except ImportError:                         # scipy < 1.4 -- the PYNQ-Z2 has 0.19.1
    use_fft("numpy.fft")
from scipy.signal import sosfilt           # scipy >= 0.16; the only compiled routine used

with open(os.path.join(HERE, "frontend_params.json")) as _fh:
    P = json.load(_fh)

_F = {name: (np.asarray(f["sos"], dtype=np.float64), np.asarray(f["zi"], dtype=np.float64),
             int(f["edge"])) for name, f in P["filters"].items()}
FS = float(P["fs"])
FS_CUR = FS / P["decimate_current"]
FS_VIB = FS / P["decimate_vibration"]
N_BINS = int(P["n_bins"])
ORDER_MAX = float(P["order_max"])
EPS = float(P["spectrum_eps"])
_RAW_EDGES = (float(P["raw_spectrum_edges_hz"]["fs"]),
              np.asarray(P["raw_spectrum_edges_hz"]["edges"], dtype=np.float64))


# ---------------------------------------------------------------------------
# filters: scipy.signal.sosfiltfilt and decimate, from sosfilt alone
# ---------------------------------------------------------------------------

def _odd_ext(x, n):
    """scipy.signal._arraytools.odd_ext along the last axis of a 1-D array."""
    left = 2.0 * x[0] - x[n:0:-1]
    right = 2.0 * x[-1] - x[-2:-(n + 2):-1]
    return np.concatenate((left, x, right))


def sosfiltfilt(name, x):
    sos, zi, edge = _F[name]
    if x.size <= edge:
        raise ValueError("signal of %d samples is shorter than the pad of %d" % (x.size, edge))
    ext = _odd_ext(x, edge)
    y, _ = sosfilt(sos, ext, zi=zi * ext[0])
    y, _ = sosfilt(sos, y[::-1], zi=zi * y[-1])
    return y[::-1][edge:-edge]


def decimate(x, q):
    x = np.asarray(x, dtype=np.float64)
    if q <= 1:
        return x
    return sosfiltfilt("decimate_%d" % q, x)[::q]


def analytic_envelope(x, band):
    """Band-pass, |analytic signal| (scipy.signal.hilbert's FFT form), minus DC."""
    filtered = sosfiltfilt(band, np.asarray(x, dtype=np.float64))
    n = filtered.size
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    env = np.abs(_ifft(_fft(filtered) * h))
    return env - env.mean()


# ---------------------------------------------------------------------------
# spectra -- dsp.py, unchanged in substance
# ---------------------------------------------------------------------------

def _mag(x, fs, demean=True):
    x = np.asarray(x, dtype=np.float64)
    if demean:
        x = x - x.mean()
    n = x.size
    return np.abs(_rfft(x * np.hanning(n))), np.fft.rfftfreq(n, 1.0 / fs), n


def estimate_shaft_speed(current, fs):
    mag, freqs, _ = _mag(current, fs)
    band = (freqs >= P["fundamental_lo_hz"]) & (freqs <= P["fundamental_hi_hz"])
    if not band.any():
        raise ValueError("no spectral bins in the fundamental band at fs=%s" % fs)
    idx = np.flatnonzero(band)[np.argmax(mag[band])]
    if 0 < idx < mag.size - 1:
        a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    f_elec = float((idx + delta) * (freqs[1] - freqs[0]))
    return f_elec / P["pole_pairs"], f_elec


def _notch(mag, freqs, f_elec):
    for k in range(1, P["supply_harmonics_notched"] + 1):
        centre = k * f_elec
        if centre > freqs[-1]:
            break
        mag[np.abs(freqs - centre) <= P["supply_notch_halfwidth_hz"]] = 0.0
    return mag


def _orders():
    return np.arange(N_BINS, dtype=np.float64) * (ORDER_MAX / N_BINS)


def envelope_order_spectrum(envelope, fs, fr_hz):
    x = np.asarray(envelope, dtype=np.float64)
    n = x.size
    mag = np.abs(_rfft(x * np.hanning(n))) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    if fr_hz <= 0:
        return np.zeros(N_BINS, dtype=np.float64)
    target_hz = _orders() * fr_hz
    out = np.interp(target_hz, freqs, mag, left=0.0, right=0.0)
    out[target_hz > freqs[-1]] = 0.0
    return out


def current_sideband_order_spectrum(current, fs, fr_hz, f_elec):
    mag, freqs, n = _mag(current, fs)
    mag = mag * (2.0 / n)
    if fr_hz <= 0 or f_elec <= 0:
        return np.zeros(N_BINS, dtype=np.float64)
    fundamental = float(np.interp(f_elec, freqs, mag))
    notched = _notch(mag.copy(), freqs, f_elec)
    offsets = _orders() * fr_hz
    upper = np.interp(f_elec + offsets, freqs, notched, left=0.0, right=0.0)
    lower = np.interp(np.abs(f_elec - offsets), freqs, notched, left=0.0, right=0.0)
    upper[f_elec + offsets > freqs[-1]] = 0.0
    return (upper + lower) / (fundamental + EPS)


def raw_order_spectrum(signal, fs, fr_hz, f_elec=None):
    mag, freqs, n = _mag(signal, fs)
    mag = mag * (2.0 / n)
    if fr_hz <= 0:
        return np.zeros(N_BINS, dtype=np.float64)
    if f_elec and f_elec > 0:
        mag = _notch(mag.copy(), freqs, f_elec)
    target = _orders() * fr_hz
    out = np.interp(target, freqs, mag, left=0.0, right=0.0)
    out[target > freqs[-1]] = 0.0
    return out


def raw_log_spectrum(signal, fs):
    lo, hi_band = P["raw_spectrum_hz"]
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    power = np.abs(_rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    hi = min(hi_band, 0.98 * fs / 2)
    if hi <= lo:
        return np.zeros(N_BINS, dtype=np.float64)
    if fs == _RAW_EDGES[0]:
        edges = _RAW_EDGES[1]                  # the laptop's own edges (make_frontend_params)
    else:                                      # geomspace with its endpoints pinned, as numpy >= 1.16
        edges = 10.0 ** np.linspace(np.log10(lo), np.log10(hi), N_BINS + 1)
        edges[0], edges[-1] = lo, hi
    idx = np.searchsorted(freqs, edges)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    out = cumulative[np.clip(idx[1:], 0, len(power))] - \
        cumulative[np.clip(idx[:-1], 0, len(power))]
    return np.sqrt(np.maximum(out, 0.0))


def condition_spectrum(spectrum):
    s = np.asarray(spectrum, dtype=np.float64)
    med = np.median(s)
    if not np.isfinite(med) or med <= EPS:
        med = float(np.mean(np.abs(s))) or 1.0
    return np.log1p(s / (med + EPS))


# ---------------------------------------------------------------------------
# a whole recording
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# both ARM cores
# ---------------------------------------------------------------------------
# The Zynq-7020 has two Cortex-A9 cores; numpy 1.13 uses one. Measured on the
# PYNQ-Z2 on 2026-09-24, single process: 5.2 s of board time per 4 s of signal --
# behind real time -- of which 3.3 s were the two vibration envelopes and 1.1 s
# the seven windows' spectra. Those are independent of one another, so with
# start_workers(2) they run in two worker processes: the same functions on the
# same numbers, so the output is identical to the single-process path (tested).
# The workers are forked before the overlay is loaded and hold no pynq state.

_POOL = None
WORKERS = 1


def start_workers(n):
    global _POOL, WORKERS
    if n > 1 and _POOL is None:
        import multiprocessing
        _POOL = multiprocessing.Pool(n)
        WORKERS = n
    return WORKERS


def stop_workers():
    global _POOL, WORKERS
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
    _POOL, WORKERS = None, 1


def _window(c1, c2, env_low, env_high, vib):
    """One window's five channels and its diagnostics."""
    fr_hz, f_elec = estimate_shaft_speed(c1, FS_CUR)
    channels = [
        np.mean([current_sideband_order_spectrum(s, FS_CUR, fr_hz, f_elec)
                 for s in (c1, c2)], axis=0),
        np.mean([raw_order_spectrum(s, FS_CUR, fr_hz, f_elec) for s in (c1, c2)], axis=0),
        envelope_order_spectrum(env_low, FS_VIB, fr_hz),
        envelope_order_spectrum(env_high, FS_VIB, fr_hz),
        raw_log_spectrum(vib, FS_VIB),
    ]
    spec = np.stack([condition_spectrum(c) for c in channels]).astype(np.float32)
    return spec, {"fr_hz": fr_hz, "f_elec_hz": f_elec, "rpm": fr_hz * 60.0}


def _window_args(args):
    return _window(*args)


def process_recording(current_1, current_2, vibration):
    """
    4 s of raw 64 kHz signal -> ((n_windows, 5, 512) float32, per-window
    diagnostics, stage timings in ms). Decimation and demodulation run once over
    the whole signal and the result is windowed, as dsp.process_recording does.
    With workers, the two envelopes and then the windows are split across them;
    the current decimation runs in this process meanwhile, so its time is booked
    to "decimate" and the rest of the wait to "envelope".
    """
    t0 = time.perf_counter()
    vib = decimate(vibration, P["decimate_vibration"])
    if _POOL is not None:
        jobs = [_POOL.apply_async(analytic_envelope, (vib, b)) for b in ("band_low", "band_high")]
    cur_1 = decimate(current_1, P["decimate_current"])
    cur_2 = decimate(current_2, P["decimate_current"])
    t1 = time.perf_counter()
    if _POOL is not None:
        env_low, env_high = [j.get() for j in jobs]
    else:
        env_low = analytic_envelope(vib, "band_low")
        env_high = analytic_envelope(vib, "band_high")
    t2 = time.perf_counter()
    # Wall time, split so the three add up: "decimate" is this process's own
    # decimation; "envelope" is the rest until both envelopes are in hand.
    decimate_ms = (t1 - t0) * 1e3
    envelope_ms = (t2 - t0) * 1e3 - decimate_ms

    n_cur, hop_cur = int(round(P["window_s"] * FS_CUR)), int(round(P["hop_s"] * FS_CUR))
    n_vib, hop_vib = int(round(P["window_s"] * FS_VIB)), int(round(P["hop_s"] * FS_VIB))
    vib_len = min(env_low.size, env_high.size)
    if min(cur_1.size, cur_2.size) < n_cur or vib_len < n_vib:
        empty = np.zeros((0, 5, N_BINS), dtype=np.float32)
        return empty, [], {"decimate_ms": decimate_ms, "envelope_ms": envelope_ms,
                           "spectra_ms": 0.0}
    n_windows = min(1 + (min(cur_1.size, cur_2.size) - n_cur) // hop_cur,
                    1 + (vib_len - n_vib) // hop_vib)

    tstart = time.perf_counter()
    args = []
    for w in range(n_windows):
        cs, vs = w * hop_cur, w * hop_vib
        args.append((cur_1[cs:cs + n_cur], cur_2[cs:cs + n_cur], env_low[vs:vs + n_vib],
                     env_high[vs:vs + n_vib], vib[vs:vs + n_vib]))
    out = _POOL.map(_window_args, args) if _POOL is not None else [_window(*a) for a in args]
    specs, diags = [o[0] for o in out], [o[1] for o in out]
    return (np.stack(specs).astype(np.float32), diags,
            {"decimate_ms": decimate_ms, "envelope_ms": envelope_ms,
             "spectra_ms": (time.perf_counter() - tstart) * 1e3})
