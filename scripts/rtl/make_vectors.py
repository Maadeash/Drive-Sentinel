"""
Build the simulation vectors for V-1 .. V-6.

Every vector set is derived from the same two things the RTL is derived from --
the order-spectrum cache and `artifacts/int8_export/` -- so the testbench never
compares the RTL against a number that was typed anywhere.

THE SETS
--------
v1  All 16,211 cache windows, optionally split into shards so several XSim
    processes can run disjoint ranges in parallel. `docs/rtl_declarations.md` D-6
    allows splitting and forbids subsampling: the union of the shards is V-1, a
    subset is `PARTIAL`.
v3  The first N windows (default 64, the spec's number) plus every int32
    accumulator of every layer in the order ds_conv1d emits them, for the
    bit-exact per-layer comparison. 14,339 accumulators per window.
v4  Directed requantisation vectors: exact ties at every shift the export uses,
    on both signs, with the floor landing both even and odd, plus the values
    either side of each tie. Driven into ds_requant directly by tb_requant.sv --
    an exact tie cannot be reached reliably by choosing an input window, and a
    test that cannot reach the case it is named after is not a test.
v5  Saturation probes: windows chosen so the int8 clamp fires at every layer. The
    per-layer clamp counts are written into the meta file, because a saturation
    test that never saturates passes for the wrong reason.

FILE FORMATS
------------
`in_*.hex`   640 lines per window, one 32-bit word as 8 hex digits, little-endian
             within the word -- byte 4n of the frame is bits [7:0] of line n.
`exp_*.txt`  one line per window: class, then the three exposed logit registers
             as decimal int32, i.e. (M0[c] * acc[c]) >>> DS_LOGIT_SHIFT.
`acc.txt`    one accumulator per line, decimal, in emission order:
             layer 0..4, then output channel, then position.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.features import load_cache
from drivesentinel.rtl import model as M

OUT_ROOT = os.path.join(C.ARTIFACT_DIR, "rtl", "vectors")
PARAMS = os.path.join(C.ARTIFACT_DIR, "rtl", "rtl_params.json")

_HEX = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)


def write_hex32(path: str, words: np.ndarray) -> None:
    """
    Write uint32 values as 8 lowercase hex digits per line, fast.

    np.savetxt takes minutes on the ten million lines V-1 needs; building the
    character array directly takes seconds. The nibbles are extracted
    most-significant first so the text reads the way $fscanf("%h") parses it.
    """
    w = np.ascontiguousarray(words, dtype=np.uint32)
    nib = np.empty((w.size, 9), dtype=np.uint8)
    for d in range(8):
        nib[:, d] = _HEX[(w >> np.uint32(4 * (7 - d))) & np.uint32(0xF)]
    nib[:, 8] = ord("\n")
    with open(path, "wb") as fh:
        fh.write(nib.tobytes())


def window_words(xq_w: np.ndarray) -> np.ndarray:
    """One (5, 512) int8 window -> 640 little-endian uint32 beats."""
    return np.ascontiguousarray(xq_w).reshape(-1).view(np.uint8).view("<u4")


def expected(xq: np.ndarray, exp: M.Export, F: int, logit_shift: int,
             fc_shift: int):
    """Class and exposed logit registers, from the bit-accurate RTL model."""
    r = M.forward(xq, exp, requant="int", frac_width=F, trace=True)
    # int64 arithmetic shift, matching Verilog's >>> on a signed value: the raw
    # product reaches 2**46 here, well inside int64, and a float64 cast would
    # round exactly where two classes are closest.
    shifted = r["logits_raw"] >> np.int64(logit_shift)
    return r, shifted


def load_params() -> dict:
    if not os.path.exists(PARAMS):
        sys.exit("FATAL: artifacts/rtl/rtl_params.json missing -- run "
                 "scripts/rtl/gen_rtl_params.py first")
    with open(PARAMS) as fh:
        return json.load(fh)


def emit_set(name: str, xq: np.ndarray, exp: M.Export, p: dict,
             shards: int = 1, with_acc: bool = False, extra: dict = None) -> dict:
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    F, ls, fs = p["frac_width"], p["logit_shift"], p["fc_shift"]

    n = len(xq)
    edges = np.linspace(0, n, shards + 1).astype(int)
    meta = {"set": name, "n_windows": int(n), "shards": [],
            "frac_width": F, "logit_shift": ls, "fc_shift": fs,
            "logit_lsb_value": p["logit_lsb_value"],
            "expected_build_id": p["expected_build_id"],
            "expected_build_id_hex": p["expected_build_id_hex"]}
    if extra:
        meta.update(extra)

    acc_rows = []
    for s in range(shards):
        lo, hi = int(edges[s]), int(edges[s + 1])
        words = np.concatenate([window_words(xq[i]) for i in range(lo, hi)])
        write_hex32(os.path.join(out_dir, f"in_{s}.hex"), words)

        r, shifted = expected(xq[lo:hi], exp, F, ls, fs)
        with open(os.path.join(out_dir, f"exp_{s}.txt"), "w", newline="\n") as fh:
            for j in range(hi - lo):
                fh.write(f"{int(r['pred'][j])} {shifted[j, 0]} "
                         f"{shifted[j, 1]} {shifted[j, 2]}\n")

        if with_acc:
            for j in range(hi - lo):
                for li in range(len(exp.layers)):
                    acc_rows.append(r["acc"][li][j].reshape(-1))

        meta["shards"].append({"shard": s, "first_window": lo,
                               "n_windows": hi - lo,
                               "in": f"in_{s}.hex", "exp": f"exp_{s}.txt"})
        print(f"    shard {s}: windows {lo}..{hi - 1}", flush=True)

    if with_acc:
        flat = np.concatenate(acc_rows)
        np.savetxt(os.path.join(out_dir, "acc.txt"), flat, fmt="%d")
        meta["acc_file"] = "acc.txt"
        meta["acc_per_window"] = int(flat.size // n)
        print(f"    {flat.size:,} accumulators "
              f"({meta['acc_per_window']:,} per window)")

    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


# ---------------------------------------------------------------------------
# v4 -- directed requantisation vectors
# ---------------------------------------------------------------------------

def make_v4(exp: M.Export, p: dict) -> dict:
    """
    Directed vectors for round-half-to-even, driven straight into ds_requant.

    A FINDING FIRST, BECAUSE IT SHAPES THE TEST
    -------------------------------------------
    At the chosen fractional width the exported shifts are 34..40, and the
    mantissas are 26-bit values with no large power-of-two factor. An exact tie
    needs `acc * m0 == q * 2**shift + 2**(shift-1)`; for an odd `m0` the
    solutions for `acc` are spaced `2**shift` apart, which at shift >= 34 means
    there is no solution inside the int32 accumulator range at all. **The tie
    case is therefore unreachable from the AXI-Stream input.** It is still
    reachable by the hardware -- a different export, a different width, a
    different layer would reach it -- so it has to be tested, and it has to be
    tested at the unit level. That is what this set is.

    THREE FAMILIES
      1. m0 = 1, shift 1..31. The product is then the accumulator itself, so
         every exact tie, every value either side of one, both signs and both
         parities of the floor are directly constructible. This covers the
         rounder completely as a function of (product, shift), which is all it
         is a function of.
      2. m0 = 2**25, shift 26..40. A mantissa with 25 factors of two makes ties
         reachable at the shifts the export actually uses:
         acc = q * 2**(shift-25) + 2**(shift-26) gives a product that is exactly
         a tie, and stays inside int32.
      3. The 179 real exported (m0, shift) pairs, with accumulators swept across
         the int8 clamp boundary in both directions -- no ties, by the finding
         above, but the arithmetic the datapath actually performs.

    Expected values come from drivesentinel.rtl.model.round_half_to_even, the
    same three lines ds_requant.v implements.
    """
    out_dir = os.path.join(OUT_ROOT, "v4")
    os.makedirs(out_dir, exist_ok=True)
    F = p["frac_width"]
    real_shifts = sorted({int(e["shift"]) for e in p["requant_params"]})

    vectors = []                       # (acc, m0, shift)

    # family 1 -- m0 = 1, so product == acc
    for s in range(1, 32):
        half = 1 << (s - 1)
        for q in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 17, -17):
            tie = q * (1 << s) + half
            if abs(tie) >= (1 << 30):
                continue
            for d in (-1, 0, 1):
                vectors.append((tie + d, 1, s))

    # family 2 -- m0 = 2**25, ties reachable at the exported shifts
    m0_pow = 1 << (F - 1)
    for s in range(F, 41):
        step = 1 << (s - (F - 1))
        half_acc = 1 << (s - F)
        for q in (-3, -2, -1, 0, 1, 2, 3, 9):
            acc = q * step + half_acc
            if abs(acc) >= (1 << 30):
                continue
            for d in (-1, 0, 1):
                vectors.append((acc + d, m0_pow, s))

    # family 3 -- the real parameters, across the clamp boundary
    for e in p["requant_params"]:
        m0, s = int(e["m0"]), int(e["shift"])
        unit = max((1 << s) // m0, 1)          # accumulator worth about 1 LSB
        for mult in (0, 1, 2, 63, 64, 126, 127, 128, 129, 255,
                     -1, -64, -127, -128, -129):
            acc = mult * unit
            if abs(acc) < (1 << 30):
                vectors.append((acc, m0, s))

    rows, n_ties = [], 0
    for acc, m0, s in vectors:
        prod = acc * m0
        if (prod & ((1 << s) - 1)) == (1 << (s - 1)):
            n_ties += 1
        q = int(M.round_half_to_even(np.int64(prod), np.int64(s)))
        rows.append((acc, m0, s, q, int(np.clip(q, -128, 127))))

    with open(os.path.join(out_dir, "requant_vectors.txt"), "w", newline="\n") as fh:
        for acc, m0, s, q, cl in rows:
            fh.write(f"{acc} {m0} {s} {q} {cl}\n")

    meta = {
        "set": "v4",
        "n_vectors": len(rows),
        "n_exact_ties": n_ties,
        "real_shifts": real_shifts,
        "frac_width": F,
        "file": "requant_vectors.txt",
        "columns": "acc m0 shift expected_q expected_clamped",
        "finding": "At frac_width %d the exported shifts are %d..%d with odd "
                   "26-bit mantissas, so an exact tie has no solution inside the "
                   "int32 accumulator range. The tie case is unreachable from "
                   "the AXI-Stream input and is exercised here at the unit level."
                   % (F, min(real_shifts), max(real_shifts)),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"    {meta['n_vectors']} vectors, {n_ties} exact ties, "
          f"real shifts {min(real_shifts)}..{max(real_shifts)}")
    return meta


# ---------------------------------------------------------------------------
# v5 -- saturation probes
# ---------------------------------------------------------------------------

def clamp_counts(xq: np.ndarray, exp: M.Export, F: int) -> dict:
    """
    How many values the int8 clamp actually caught, per layer.

    Comes straight out of the model, not from a second implementation here: a
    duplicate of the requantisation would be one more thing that can drift away
    from the datapath it is supposed to describe.
    """
    r = M.forward(xq, exp, requant="int", frac_width=F)
    names = [L["name"] for L in exp.layers if not L["is_last"]]
    return dict(zip(names, r["clamped"]))


def _search_pool_clamp(exp: M.Export, F: int, iters: int = 6000) -> np.ndarray:
    """
    Find an input that makes the POOL layer's clamp fire.

    The other three clamps fire readily -- the activation scales were fitted at
    the 99.9th percentile, so roughly one value in a thousand of ordinary data
    saturates. conv4 is different: the global average pool divides by 32 and the
    fc input scale is generous, so nothing in the cache and none of the obvious
    extremes gets past 127. The architectural bound is 915..1370 per channel, so
    it IS reachable; it just needs looking for.

    A greedy hill climb over int8 values, fixed seed, stopping as soon as a probe
    exceeds the clamp. Measured, not hand-placed -- a hand-placed vector is a
    constant somebody would have to re-derive after any re-export, and this
    converges in well under a second.
    """
    Ms = exp.multipliers()
    m0, shift = M.quantise_multiplier(Ms[exp.pool_index], F)

    def pool_q(xq):
        r = M.forward(xq, exp, requant="int", frac_width=F, trace=True)
        psum = np.maximum(r["acc"][exp.pool_index], 0).sum(axis=2)
        return M.round_half_to_even(m0[None, :] * psum, shift[None, :])

    rng = np.random.default_rng(7)
    x = np.full((1, 5, 512), 127, np.int8)
    best = int(pool_q(x).max())
    choices = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
    for _ in range(iters):
        if best > 127:
            break
        n = int(rng.integers(1, 40))
        cand = x.copy()
        cand[0, rng.integers(0, 5, size=n), rng.integers(0, 512, size=n)] = \
            rng.choice(choices, size=n)
        v = int(pool_q(cand).max())
        if v >= best:
            best, x = v, cand
    return x[0], best


def make_v5(X: np.ndarray, exp: M.Export, p: dict) -> np.ndarray:
    """Probe windows: the int8 extremes, structured patterns, and amplified data."""
    rng = np.random.default_rng(20260920)
    probes = [
        np.full((5, 512), 127, dtype=np.int8),
        np.full((5, 512), -128, dtype=np.int8),
        np.tile(np.where(np.arange(5) % 2 == 0, 127, -128)[:, None],
                (1, 512)).astype(np.int8),
        np.tile(np.where(np.arange(512) % 2 == 0, 127, -128), (5, 1))
          .astype(np.int8),
        rng.integers(-128, 128, size=(5, 512)).astype(np.int8),
        rng.choice(np.array([-128, 127], dtype=np.int8), size=(5, 512)),
    ]
    # plus real windows pushed hard enough that the clamp is guaranteed to fire
    sel = rng.choice(len(X), size=26, replace=False)
    for i in sel:
        for gain in (4.0, 16.0):
            probes.append(np.clip(
                np.rint(M.quantise_input(X[i:i + 1].astype(np.float64), exp)[0]
                        * gain), -128, 127).astype(np.int8))
    pool_probe, pool_peak = _search_pool_clamp(exp, p["frac_width"])
    probes.append(pool_probe)
    print(f"    pool-clamp search: peak pre-clamp value {pool_peak} "
          f"({'fires' if pool_peak > 127 else 'DOES NOT FIRE'})")
    return np.stack(probes)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["v1", "v3", "v4", "v5"])
    ap.add_argument("--shards", type=int, default=4,
                    help="parallel XSim processes for V-1")
    ap.add_argument("--v3-windows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    exp = M.Export()
    p = load_params()
    print(f"  frac width {p['frac_width']}, logit shift {p['logit_shift']}, "
          f"build id {p['expected_build_id_hex']}")

    need_cache = any(s in args.sets for s in ("v1", "v3", "v5"))
    xq = None
    if need_cache:
        X, _, _ = load_cache()
        xq = np.empty((len(X), X.shape[1], X.shape[2]), dtype=np.int8)
        for s in range(0, len(X), args.batch):
            e = min(s + args.batch, len(X))
            xq[s:e] = M.quantise_input(X[s:e].astype(np.float64), exp)
        print(f"  cache {X.shape} -> int8 at the hardware boundary")

    summary = {}
    if "v3" in args.sets:
        print("  v3: per-layer accumulators")
        summary["v3"] = emit_set("v3", xq[:args.v3_windows], exp, p,
                                 shards=1, with_acc=True)
    if "v4" in args.sets:
        print("  v4: directed round-half-to-even vectors")
        summary["v4"] = make_v4(exp, p)
    if "v5" in args.sets:
        print("  v5: saturation probes")
        probes = make_v5(X, exp, p)
        counts = clamp_counts(probes, exp, p["frac_width"])
        print(f"    clamp fires per layer: {counts}")
        summary["v5"] = emit_set("v5", probes, exp, p, shards=1,
                                 extra={"clamp_counts": counts})
    if "v1" in args.sets:
        print(f"  v1: all windows, {args.shards} shards")
        summary["v1"] = emit_set("v1", xq, exp, p, shards=args.shards)

    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "shards"}
                   for k, v in summary.items()}, fh, indent=2)
    print(f"  wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
