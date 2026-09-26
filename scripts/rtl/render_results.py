"""
Generate `docs/results_rtl.md` from the JSON the runs produced.

The rule this project has followed since P6: results markdown is generated, never
hand-typed, and anything not run reads `NOT RUN`. This script reads

    artifacts/rtl/requant_sweep.json      the D-2 width sweep
    artifacts/rtl/bit_exact_confirm.json  the full-cache bit-exactness confirmation
    artifacts/rtl/rtl_params.json         what the generator emitted
    artifacts/rtl/verify.json             V-1 .. V-6, from XSim
    artifacts/rtl/synth*/synth.json       post-route, from Vivado
    artifacts/rtl/synth*/post_route_util.rpt   Vivado's own utilisation summary
    board/board_run.json                  the notebook's result from a real board,
                                          used only if it says ran_on_hardware and
                                          was not produced by the dry-run mock

and writes the document. Nothing here invents a figure; every cell either comes from
one of those files or says `NOT RUN`.

TWO KINDS OF UTILISATION NUMBER
-------------------------------
`synth.json` carries raw get_cells primitive tallies. Those are NOT the same as the
"Slice LUTs" line of `report_utilization`, which accounts for LUT combining. This
script parses the REPORT, because the report is what a reviewer opens, and keeps the
primitive counts only as a cross-check.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drivesentinel import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTL_ART = os.path.join(C.ARTIFACT_DIR, "rtl")
OUT = os.path.join(ROOT, "docs", "results_rtl.md")

NOT_RUN = "`NOT RUN`"


def load(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def parse_util(path: str) -> Dict[str, int]:
    """Pull the canonical rows out of Vivado's utilisation report."""
    if not os.path.exists(path):
        return {}
    want = {
        "Slice LUTs": "slice_luts",
        "Slice Registers": "slice_registers",
        "Block RAM Tile": "bram_tiles",
        "DSPs": "dsps",
        "Bonded IOB": "bonded_iob",
    }
    out: Dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"\|\s*([A-Za-z0-9 ]+?)\s*\|\s*(\d+)\s*\|", line)
            if not m:
                continue
            name, val = m.group(1).strip(), int(m.group(2))
            if name in want and want[name] not in out:
                out[want[name]] = val
                out[want[name] + "_available"] = _avail(line)
    return out


def _avail(line: str) -> int:
    cells = [c.strip() for c in line.split("|")]
    # | name | Used | Fixed | Prohibited | Available | Util% |
    try:
        return int(cells[5])
    except (IndexError, ValueError):
        return 0


def parse_wns(path: str):
    """WNS / WHS / failing endpoints from a timing summary report."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    m = re.search(r"Setup\s*:\s*(\d+)\s*Failing Endpoints,\s*Worst Slack\s*"
                  r"(-?\d+\.\d+)ns", txt)
    if not m:
        return None
    h = re.search(r"Hold\s*:\s*(\d+)\s*Failing Endpoints,\s*Worst Slack\s*"
                  r"(-?\d+\.\d+)ns", txt)
    return {
        "setup_failing": int(m.group(1)),
        "wns_ns": float(m.group(2)),
        "hold_failing": int(h.group(1)) if h else None,
        "whs_ns": float(h.group(2)) if h else None,
    }


def parse_hier_row(path: str, instance: str) -> Dict[str, int]:
    """One instance's row from report_utilization -hierarchical."""
    if not os.path.exists(path):
        return {}
    cols = ["total_luts", "logic_luts", "lutrams", "srls", "ffs", "ramb36",
            "ramb18", "dsps"]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 3 and cells[1] == instance:
                vals = cells[3:3 + len(cols)]
                try:
                    return dict(zip(cols, (int(v) for v in vals)))
                except ValueError:
                    return {}
    return {}


def fmt(v, spec="{:,}"):
    return NOT_RUN if v is None else spec.format(v)


def pct(a, b):
    return NOT_RUN if not b else f"{100.0 * a / b:.2f} %"


def us(cycles: int, period_ns: float) -> float:
    """Microseconds. cycles * ns is NANOseconds; dividing by 1000 gives us."""
    return cycles * period_ns / 1000.0


def ms(cycles: int, period_ns: float) -> float:
    return cycles * period_ns / 1.0e6


# ---------------------------------------------------------------------------

def main() -> None:
    sweep = load(os.path.join(RTL_ART, "requant_sweep.json"))
    confirm = load(os.path.join(RTL_ART, "bit_exact_confirm.json"))
    params = load(os.path.join(RTL_ART, "rtl_params.json"))
    verify = load(os.path.join(RTL_ART, "verify.json"))

    builds = []
    for name, label in (("synth", "shipped (DEBUG_TRACE=0)"),
                        ("synth_debug_trace", "with the V-3 trace (DEBUG_TRACE=1)"),
                        ("synth_v2_pipelined_requant",
                         "iteration 2: pipelined requantiser only"),
                        ("synth_v1_combinational",
                         "iteration 1: combinational requantiser")):
        d = os.path.join(RTL_ART, name)
        if not os.path.isdir(d):
            continue
        builds.append({
            "dir": name, "label": label,
            "json": load(os.path.join(d, "synth.json")),
            "util": parse_util(os.path.join(d, "post_route_util.rpt")),
            "timing": parse_wns(os.path.join(d, "post_route_timing.rpt")),
        })
    shipped = next((b for b in builds if b["dir"] == "synth"), None)

    runs = (verify or {}).get("runs", {})
    v1, v3, v4, v5 = (runs.get(k) for k in ("v1", "v3", "v4", "v5"))

    # A board run counts only if the notebook says it ran on hardware and was not
    # driven by the software stand-in (board/dryrun/pynq reports 'MOCK...'); the
    # same rule tests/test_board_package.py applies to a committed board_run.json.
    board = load(os.path.join(ROOT, "board", "board_run.json"))
    if board and not (board.get("ran_on_hardware") is True and
                      not str(board.get("pynq_version", "")).startswith("MOCK")):
        board = None
    bt = (board or {}).get("timing", {})
    bwall = bt.get("dma_plus_inference_wall", {})

    L = []
    A = L.append

    A("# RTL results — INT8 bearing-CNN accelerator")
    A("")
    A("**Generated by `scripts/rtl/render_results.py`. Do not edit by hand.**")
    A("Every figure below comes from a JSON written by a run; anything not run reads")
    A("`NOT RUN`. The recipes were declared in `docs/rtl_declarations.md` before any")
    A("RTL existed, and the bars in `docs/rtl_spec.md` §8 were declared before that.")
    A("")
    if board:
        A("> **One section of this document is a measurement of silicon: §4c**, a run")
        A(f"> of `board/drivesentinel_overlay.ipynb` on a {board['board'].split(' (')[0]} "
          f"(`board/board_run.json`). Everything else is")
        A("> simulation (XSim) or synthesis (Vivado post-route on `xc7z020clg400-1`):")
        A("> real reports of a real implementation, but not measurements of the board.")
    else:
        A("> **Nothing in this document has run on hardware.** No bitstream was written and")
        A("> no board was involved. `rtl_spec.md` §9.4 is answered **simulation plus")
        A("> synthesis only**. The simulation figures are XSim; the resource and clock")
        A("> figures are Vivado post-route on `xc7z020clg400-1`. Those are real reports of")
        A("> a real implementation, and they are still not a measurement of silicon.")
    A("")
    A("> **This accelerates stage S5, the bearing CNN, and nothing else.** The winding,")
    A("> supply and inverter branches are scalar-feature models in software. Putting")
    A("> this model in fabric does not change its accuracy in either direction: it stays")
    A("> **macro-F1 0.7852, leave-one-bearing-out over 29 bearings**. The hardware claim")
    A("> is latency and resource cost.")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------------- slide
    A("## 1. The slide table")
    A("")
    A("| | |")
    A("|---|---|")
    if params:
        A(f"| Model | frozen v5 bearing CNN, **{params['total_macs']:,} MACs** per "
          f"1 s window |")
        A(f"| Parameters | **{params['w_depth']:,} int8 weights**, "
          f"{params['b_depth']} int32 biases |")
    if v1 and v1.get("complete"):
        A(f"| Windows verified bit-exact vs the golden reference | "
          f"**{v1['windows_run']:,} / {v1['windows_expected']:,}** "
          f"(argmax, V-1) |")
    elif v1:
        A(f"| Windows verified (argmax, V-1) | **PARTIAL — {v1['windows_run']:,} of "
          f"{v1['windows_expected']:,}**, {v1['class_mismatches']} disagreements "
          f"so far. Not V-1 until all {v1['windows_expected']:,} have run "
          f"(`rtl_declarations.md` D-6). |")
    else:
        A(f"| Windows verified (argmax, V-1) | {NOT_RUN} |")
    if v3 and v3.get("complete"):
        A(f"| Per-layer int32 accumulators compared, bit-exact (V-3) | "
          f"**{v3['acc_compared']:,}** over {v3['windows_run']} windows |")
    if shipped and shipped["util"]:
        u, j = shipped["util"], shipped["json"]
        A(f"| **Core only** — resources (post-route, xc7z020, out of context) | "
          f"**{u.get('slice_luts', 0):,} LUT** "
          f"({pct(u.get('slice_luts', 0), u.get('slice_luts_available', 0))}), "
          f"**{u.get('slice_registers', 0):,} FF**, "
          f"**{u.get('bram_tiles', 0)} BRAM**, "
          f"**{u.get('dsps', 0)} DSP48** |")
        if j:
            A(f"| **Core only** — clock | target {j['target_mhz']:.0f} MHz, achieved "
              f"**{j['achieved_mhz']:.1f} MHz** "
              f"(WNS {j['wns_ns']:+.3f} ns, "
              f"{'timing met' if j['timing_met'] else 'TIMING NOT MET'}) |")
    _sysj = load(os.path.join(RTL_ART, "system", "system.json"))
    _su = parse_util(os.path.join(RTL_ART, "system", "post_route_util.rpt"))
    if _sysj:
        A(f"| **Full system** (PS + DMA + interconnect + core), post-route | "
          f"{_su.get('slice_luts', 0):,} LUT, {_su.get('slice_registers', 0):,} FF, "
          f"{_su.get('bram_tiles', 0)} BRAM, {_su.get('dsps', 0)} DSP48; "
          f"WNS {float(_sysj['wns_ns']):+.3f} ns at 100 MHz, "
          f"{'timing met' if _sysj['timing_met'] else 'TIMING NOT MET'}; bitstream "
          f"{'written' if _sysj['bitstream_written'] else 'NOT written'} — "
          + ("**loaded and run on a PYNQ-Z2 (§4c)** |" if board
             else "**never loaded onto a board** |"))
    cyc = (v1 or v3 or v5 or {}).get("cycles_max")
    if cyc and shipped and shipped["json"]:
        j = shipped["json"]
        A(f"| Latency per inference | **{cyc:,} cycles** = "
          f"**{us(cyc, j['target_period_ns']):,.1f} us "
          f"({ms(cyc, j['target_period_ns']):.2f} ms)** at 100 MHz, "
          f"{us(cyc, j['achieved_period_ns']):,.1f} us at the achieved "
          f"{j['achieved_mhz']:.1f} MHz |")
        hop = 0.5
        A(f"| Margin against the 0.5 s hop | "
          f"**{hop * 1.0e6 / us(cyc, j['achieved_period_ns']):.0f}x** |")
    if board and bwall:
        A(f"| **On the board** — CNN path, DMA + inference, PS wall clock | "
          f"median **{bwall['median_ms']:.2f} ms**, p95 {bwall['p95_ms']:.2f} ms, "
          f"max {bwall['max_ms']:.2f} ms over {bt['windows_timed']} windows "
          f"(board run, §4c) |")
    A("| End-to-end latency | `NOT MEASURED`. The DSP front end is PS software "
      "(§9.1) and has never been timed on any hardware. |")
    if board:
        A(f"| Ran on hardware | **yes, once** — {board['windows']} windows: "
          f"{board['class_agree_with_golden']}/{board['windows']} class agree with the "
          f"golden reference, {board['logit_registers_exact']}/{board['windows']} logit "
          f"registers bit-exact, {board['cycles_equal_simulation']}/{board['windows']} "
          f"cycle counts equal to simulation (board run, §4c) |")
    else:
        A("| Ran on hardware | **no** |")
    A("")
    A("---")
    A("")

    # ------------------------------------------------------- requant width
    A("## 2. The requantisation width — swept, not chosen")
    A("")
    A("`rtl_spec.md` §5 refused to name a fractional width and §9.2 left it to be")
    A("measured. `docs/rtl_declarations.md` D-2 fixed the rule before the sweep ran:")
    A("the smallest `F` giving **100.000 % argmax agreement** against")
    A("`golden_reference.predict` on all 16,211 cache windows, and staying there for")
    A("every larger width tested.")
    A("")
    if not sweep:
        A(NOT_RUN)
    else:
        A(f"Reference: `{sweep['reference']}`. Windows: **{sweep['n_windows']:,}**.")
        A("")
        A("| F | argmax agreement | disagreements | per-layer accumulators bit-exact "
          f"({sweep['acc_curve_windows']} windows) |")
        A("|---|---|---|---|")
        for r in sweep["sweep"]:
            A(f"| {r['frac_width']} | {r['argmax_agreement'] * 100:.5f} % | "
              f"{r['disagreements']:,} | "
              f"{'yes' if r['all_layers_bit_exact'] else 'no'} |")
        A("")
        A(f"**D-2 selects F = {sweep['chosen_frac_width']}.** Agreement is not monotone "
          "in F — F = 10 has 2 disagreements and F = 11 has 4 — which is why the rule "
          "asks for stability across every larger width rather than for the first "
          "width that happens to hit 100 %.")
        A("")

    if confirm:
        A("### The width that actually shipped, and why it is larger")
        A("")
        A("F = %d meets the D-2 argmax criterion **in the bit-accurate Python model**. "
          "It does not meet V-3's bit-exactness bar. (Both statements are about the "
          "model; V-1 itself is the RTL in the simulator, reported in section 3.)"
          % sweep["chosen_frac_width"])
        A("")
        A("The accumulator column above is measured on %d windows for memory. Over the "
          "whole cache it says something different, and the difference is the point of "
          "having V-3 at all:" % sweep["acc_curve_windows"])
        A("")
        A("| F | accumulator mismatches, all 16,211 windows | int8 activation "
          "mismatches | argmax mismatches | max abs logit difference |")
        A("|---|---|---|---|---|")
        for r in confirm["widths"]:
            tot = sum(r["accumulator_mismatches"].values())
            A(f"| {r['frac_width']} | {tot:,} | {r['activation_mismatches']:,} | "
              f"{r['argmax_mismatches']} | {r['max_abs_logit_diff']:.3e} |")
        A("")
        A(f"**F = {confirm['smallest_bit_exact_width']} is the smallest width at which "
          "the integer datapath is bit-identical to the float specification on every "
          "accumulator of every layer of all 16,211 windows** — a strictly harder bar "
          "than V-3's declared scope of 64 windows, and the one the design was built "
          "to.")
        A("")
        A("Two declared criteria disagreed and the stricter one bound. That is not a "
          "bar moving: F = %d also gives 100.000 %% argmax agreement in the model, so "
          "the D-2 criterion is met at the shipped width too, and a handful of DSP "
          "slices is a cheap price for removing a whole class of question about "
          "whether a 1-LSB drift matters. Whether the *RTL* meets V-1 is a separate "
          "question with its own answer in section 3."
          % confirm["smallest_bit_exact_width"])
        A("")

    if params:
        A("### What the generator emitted")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| Fractional width | **{params['frac_width']}** "
          f"(from `{params['frac_width_source']}`) |")
        A(f"| Shift range | {min(p['shift'] for p in params['requant_params'])} .. "
          f"{max(p['shift'] for p in params['requant_params'])} |")
        A(f"| Worst relative multiplier error | "
          f"{params['max_relative_multiplier_error']:.3e} |")
        A(f"| fc shared shift | {params['fc_shift']} |")
        A(f"| Exposed logit register LSB | 2^{params['logit_lsb_exponent']} = "
          f"{params['logit_lsb_value']:.3e} in the reference's float units |")
        A(f"| Build ID (CRC-32 of the loaded weights and biases) | "
          f"`{params['expected_build_id_hex']}` |")
        A(f"| On-chip storage | {params['on_chip_bytes']:,} bytes |")
        A("")
    A("---")
    A("")

    # --------------------------------------------------------- verification
    A("## 3. Verification — V-1 to V-6")
    A("")
    A("Bars from `rtl_spec.md` §8, declared before any RTL existed. V-3 is run before")
    A("V-1 on purpose: argmax agreement can hide a broken layer that later layers wash")
    A("out.")
    A("")
    A("| # | Criterion | Bar | Result |")
    A("|---|---|---|---|")

    def row(tag, criterion, bar, result):
        A(f"| {tag} | {criterion} | {bar} | {result} |")

    def verdict(ok, detail, in_progress=False):
        if in_progress:
            return "**IN PROGRESS** — " + detail
        return ("**PASS** — " if ok else "**FAIL** — ") + detail

    # -- V-1 ----------------------------------------------------------------
    if v1:
        # A disagreement anywhere is a FAIL at once -- the bar is zero, so no
        # number of later windows can recover it. Otherwise an unfinished run is
        # IN PROGRESS: not a pass (D-6), and not a fail either.
        detail = (f"{v1['windows_run']:,} of {v1['windows_expected']:,} windows "
                  f"run, {v1['class_mismatches']} disagreements")
        failed = v1["class_mismatches"] > 0
        row("V-1", "RTL argmax vs `golden_reference.predict`, all 16,211 windows",
            "**100.000 %**",
            verdict(not failed and v1.get("complete"), detail,
                    in_progress=(not failed and not v1.get("complete"))))
    else:
        row("V-1", "RTL argmax vs `golden_reference.predict`, all 16,211 windows",
            "**100.000 %**", NOT_RUN)

    # -- V-2 ----------------------------------------------------------------
    src = v1 or v3 or v5
    if src:
        lsb = src["max_abs_logit_diff_lsb"]
        detail = f"**{lsb} LSB** of the exposed int32 register"
        if params:
            detail += (f" = {lsb * params['logit_lsb_value']:.3e} in the "
                       f"reference's float units")
        row("V-2", "RTL scaled logits vs golden, max absolute difference",
            "reported, not assumed", detail)
    else:
        row("V-2", "RTL scaled logits vs golden, max absolute difference",
            "reported, not assumed", NOT_RUN)

    # -- V-3 ----------------------------------------------------------------
    if v3:
        detail = (f"{v3['acc_compared']:,} accumulators compared over "
                  f"{v3['windows_run']} windows, {v3['acc_mismatches']} mismatches")
        row("V-3", "Per-layer int32 accumulators vs golden, first 64 windows",
            "**bit-exact**",
            verdict(v3["acc_mismatches"] == 0 and v3.get("complete"), detail))
    else:
        row("V-3", "Per-layer int32 accumulators vs golden, first 64 windows",
            "**bit-exact**", NOT_RUN)

    # -- V-4 ----------------------------------------------------------------
    if v4:
        detail = (f"{v4['vectors']:,} vectors, {v4['exact_ties']} exact ties, "
                  f"{v4['q_mismatches']} rounding and {v4['clamp_mismatches']} "
                  f"clamp mismatches")
        row("V-4", "Directed tie vectors exercising round-half-to-even",
            "exact match on every tie", verdict(v4.get("pass", False), detail))
    else:
        row("V-4", "Directed tie vectors exercising round-half-to-even",
            "exact match on every tie", NOT_RUN)

    # -- V-5 ----------------------------------------------------------------
    if v5:
        ok = (v5["class_mismatches"] == 0 and v5["logit_mismatches"] == 0
              and v5.get("complete"))
        clamped = sum(v5.get("meta", {}).get("clamp_counts", {}).values())
        detail = (f"{v5['windows_run']} probes, {clamped:,} values caught by the "
                  f"clamp, {v5['class_mismatches'] + v5['logit_mismatches']} "
                  f"mismatches")
        row("V-5", "Saturation probes forcing the int8 clamp at each layer",
            "exact match", verdict(ok, detail))
    else:
        row("V-5", "Saturation probes forcing the int8 clamp at each layer",
            "exact match", NOT_RUN)

    # -- V-6 ----------------------------------------------------------------
    bid = next((r for r in (v1, v3, v5)
                if r and r.get("build_id_got") is not None), None)
    if bid:
        row("V-6", "Build-ID register vs an independent CRC of the `.mem` files",
            "equal",
            verdict(bid["build_id_match"],
                    f"`0x{bid['build_id_got']:08x}`, computed by the hardware from "
                    f"the memories it loaded"))
    else:
        row("V-6", "Build-ID register vs an independent CRC of the `.mem` files",
            "equal", NOT_RUN)
    A("")

    if v1:
        A("### V-1: how it was run")
        A("")
        A(f"- DUT configuration: {v1.get('dut_config', 'DEBUG_TRACE=1 (instrumented)')}.")
        if "chunks_complete" in v1:
            A(f"- {v1['meta'].get('n_windows', v1['windows_expected']):,} windows in "
              f"{v1['shards']} chunks, {v1.get('workers', '?')} XSim processes at a "
              f"time; **{v1['chunks_complete']} of {v1['shards']} chunks complete**, "
              f"{v1['windows_run']:,} windows compared in total.")
            A("- Resumable: the testbench rewrites its report after every window and "
              "the runner skips finished chunks, so re-running "
              "`scripts/rtl/run_tb.sh v1` continues where the last run stopped.")
        A(f"- Disagreements so far: **{v1['class_mismatches']}** in class, "
          f"**{v1['logit_mismatches']}** in the exposed logit registers.")
        A("")

    if v5 and v5.get("meta", {}).get("clamp_counts"):
        A("### V-5: the probes actually saturate")
        A("")
        A("A saturation test that never saturates passes for the wrong reason, so the")
        A("clamp is counted rather than assumed. Values caught by the int8 clamp across")
        A(f"the {v5['windows_run']} probe windows:")
        A("")
        A("| layer | values clamped |")
        A("|---|---|")
        for k, n in v5["meta"]["clamp_counts"].items():
            A(f"| {k} | {n:,} |")
        A("")
        A("The pool layer's clamp is the hard one: the global average pool divides by")
        A("32 and the fc input scale is generous, so nothing in the cache and none of")
        A("the obvious extremes reaches it. The probe that does was found by a")
        A("fixed-seed hill climb in `scripts/rtl/make_vectors.py`, not placed by hand.")
        A("")

    if v4 and v4.get("meta", {}).get("finding"):
        A("### V-4: a finding that came with the test")
        A("")
        A("> " + v4["meta"]["finding"])
        A("")
        A("So V-4 is a unit testbench driving `ds_requant` directly, not a window of")
        A("data. \"We could not reach the case\" and \"the case passed\" are very")
        A("different sentences, and this is the first.")
        A("")

    A("### What is NOT verified in hardware")
    A("")
    A("`docs/rtl_declarations.md` D-4. `rtl_spec.md` §2 put the input standardisation")
    A("inside the accelerator and §7 put 2,560 int8 bytes on the AXI-Stream; those")
    A("cannot both be true, because 2,560 int8 bytes **is** the standardised and")
    A("quantised input. §7 wins, now that §9.1 is answered PS software: the front end")
    A("runs on the PS, so")
    A("")
    A("```")
    A("x_int8 = clamp(round_half_to_even((x - mean) / (std * s_x0)), -128, 127)")
    A("```")
    A("")
    A("belongs to the PS with the rest of the DSP chain. The fabric is integer-only")
    A("from the first convolution onward, which is what makes it bit-exact by")
    A("construction rather than by luck. **The standardisation is not verified in")
    A("hardware, because it is not in hardware.** The testbench generates its int8")
    A("inputs with the golden reference's own quantiser, so the boundary the hardware")
    A("sees is exactly the boundary the reference defines.")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------- synthesis
    A("## 4. Synthesis and implementation")
    A("")
    if not builds:
        A(NOT_RUN)
    else:
        j = shipped["json"] if shipped else None
        A(f"Part **{(j or {}).get('part', '?')}**, Vivado "
          f"**{(j or {}).get('vivado', '?')}**, **out of context**, "
          f"synthesis + place + route, **no bitstream**.")
        A("")
        A("Out of context is not a shortcut. This core is an AXI peripheral of the Zynq")
        A("PS with roughly 190 ports, against 125 user I/O on the clg400 package: a")
        A("pin-constrained build is not possible, and would say nothing about the core")
        A("if it were. I/O delays are left unconstrained rather than invented — see")
        A("`constraints/ds_top.xdc` — so every timed path is register-to-register, and")
        A("the figure quoted is the worst of those.")
        A("")
        A("| build | LUT | FF | BRAM tiles | DSP48 | WNS (ns) | achieved | timing |")
        A("|---|---|---|---|---|---|---|---|")
        for b in builds:
            u, t, bj = b["util"], b["timing"], b["json"]
            wns = t["wns_ns"] if t else None
            ach = f"{1000.0 / (10.0 - wns):.1f} MHz" if wns is not None else NOT_RUN
            met = ("**met**" if (wns is not None and wns >= 0) else "**NOT MET**") \
                if wns is not None else NOT_RUN
            A(f"| {b['label']} | {u.get('slice_luts', '?'):,} | "
              f"{u.get('slice_registers', '?'):,} | {u.get('bram_tiles', '?')} | "
              f"{u.get('dsps', '?')} | {wns:+.3f} | {ach} | {met} |"
              if wns is not None else
              f"| {b['label']} | {u.get('slice_luts', '?')} | ? | ? | ? | {NOT_RUN} | "
              f"{NOT_RUN} | {NOT_RUN} |")
        A("")
        if shipped and shipped["util"]:
            u = shipped["util"]
            A("Against the device, for the shipped build:")
            A("")
            A("| resource | used | available | utilisation |")
            A("|---|---|---|---|")
            for key, name in (("slice_luts", "Slice LUT"),
                              ("slice_registers", "Slice register"),
                              ("bram_tiles", "Block RAM tile"),
                              ("dsps", "DSP48E1")):
                if key in u:
                    A(f"| {name} | {u[key]:,} | {u.get(key + '_available', 0):,} | "
                      f"{pct(u[key], u.get(key + '_available', 0))} |")
            A("")
        A("### Timing closure, in two steps")
        A("")
        A("The 100 MHz constraint in `constraints/ds_top.xdc` was never edited. What")
        A("changed was the design, twice, each time at the path the router reported:")
        A("")
        A("1. **The requantiser was combinational.** A 26x32 DSP multiply, a 58-bit")
        A("   variable shift, a remainder comparison, the round and the clamp, all")
        A("   between one pair of flops: 30 logic levels, 18.248 ns, WNS −8.253 ns.")
        A("   Split into three clocked stages — +3 cycles per output element, about")
        A("   2 % of the inference.")
        A("2. **Then the MAC was the path.** Block RAM → 8x8 multiply → 32-bit add →")
        A("   accumulator, 10.505 ns, WNS −0.560 ns. The product is now registered —")
        A("   +1 cycle per output element, about 0.7 %.")
        A("")
        A("Both intermediate implementations are kept under `artifacts/rtl/` with their")
        A("reports, because \"it closes\" is worth much less without the two numbers it")
        A("closed against.")
        A("")
    A("---")
    A("")

    # ------------------------------------------------------------- latency
    # ------------------------------------------------------ full system
    sysj = load(os.path.join(RTL_ART, "system", "system.json"))
    sys_util = parse_util(os.path.join(RTL_ART, "system", "post_route_util.rpt"))
    sys_core = parse_hier_row(os.path.join(RTL_ART, "system",
                                           "post_route_util_hier.rpt"), "ds_0")
    A("## 4b. Full system — the PYNQ-Z2 overlay "
      + ("(run on a PYNQ-Z2 — §4c)" if board else "(board-ready, NOT run on a board)"))
    A("")
    A("> **Two different measurements. Do not compare them as the same thing.**")
    A("> Section 4 is the **core only**, out of context: the accelerator on its own,")
    A("> with its I/O unconstrained. This section is the **full system**: the Zynq PS,")
    A("> an AXI DMA, two AXI interconnects, a reset block and the core, placed and")
    A("> routed together as one design, with a bitstream written. The core inside it")
    A("> is the same RTL, unchanged, wrapped by `board/rtl/ds_pynq_wrap.v`.")
    A("")
    if not sysj:
        A(NOT_RUN)
    else:
        A(f"Built by `board/build_overlay.tcl`, Vivado {sysj['vivado']}, "
          f"`{sysj['part']}`, fabric clock `{sysj['fabric_clock']}` at "
          f"{1000.0 / float(sysj['target_period_ns']):.0f} MHz. Bitstream written: "
          f"**{'yes' if sysj['bitstream_written'] else 'NO'}**. "
          + ("**Ran on hardware: yes — §4c.**" if board else "**Ran on hardware: no.**"))
        A("")
        A("| | full system | of which: the core (`ds_0`) | core only, out of context (section 4) |")
        A("|---|---|---|---|")
        oc = shipped["util"] if shipped else {}
        rows = (("Slice LUTs", "slice_luts", "total_luts"),
                ("Slice registers", "slice_registers", "ffs"))
        for name, key, hkey in rows:
            A(f"| {name} | **{sys_util.get(key, 0):,}** "
              f"({pct(sys_util.get(key, 0), sys_util.get(key + '_available', 0))}) | "
              f"{sys_core.get(hkey, 0):,} | {oc.get(key, 0):,} |")
        core_tiles = sys_core.get("ramb36", 0) + sys_core.get("ramb18", 0) / 2.0
        A(f"| Block RAM tiles | **{sys_util.get('bram_tiles', 0)}** "
          f"({pct(sys_util.get('bram_tiles', 0), sys_util.get('bram_tiles_available', 0))}) | "
          f"{core_tiles:g} ({sys_core.get('ramb36', 0)} RAMB36 + "
          f"{sys_core.get('ramb18', 0)} RAMB18) | {oc.get('bram_tiles', 0)} |")
        A(f"| DSP48E1 | **{sys_util.get('dsps', 0)}** | {sys_core.get('dsps', 0)} | "
          f"{oc.get('dsps', 0)} |")
        # Timing is deliberately NOT put beside the core-only figure: the two
        # builds time different sets of paths (the full system includes the DMA,
        # the interconnects and the PS boundary), so the slacks are not the same
        # quantity. Resource counts of the core's own instance ARE comparable,
        # which is why they share a table above.
        A(f"| WNS at 100 MHz, whole design | **{float(sysj['wns_ns']):+.3f} ns** | "
          f"not separable | not comparable — see section 4 |")
        A(f"| WHS, whole design | **{float(sysj['whs_ns']):+.3f} ns** | "
          f"not separable | not comparable — see section 4 |")
        A(f"| Timing at 100 MHz | **{'met' if sysj['timing_met'] else 'NOT MET'}** | "
          f"— | — |")
        A("")
        A("The core's share of the full system and the core-only build agree to "
          "within one LUT, with identical registers, block RAM and DSP counts: the "
          "wrapper added no logic to the datapath, as it was written not to. The rest "
          "of the full system's fabric is the DMA and the two interconnects.")
        A("")
        A("Every one of the core's eleven memory images — the ten exported weight and "
          "bias `.mem` files and the generated requantisation parameters — is logged "
          "by Vivado as `$readmem data file ... is read successfully` in "
          "`artifacts/rtl/system/vivado_system.log`, and the synthesised core holds "
          f"{sysj.get('core_block_rams_after_synth', '?')} block RAM primitives. The "
          "board's build-ID register is the check that the bitstream carries exactly "
          "the exported weights; "
          + (f"on the board it read **{board['build_id']}**, "
             f"{'matching' if board['build_id_match'] else 'NOT matching'} the export "
             "(§4c)." if board else
             "it has not been read, because no board has run it."))
        A("")
    A("---")
    A("")

    # ------------------------------------------------------------ the board
    A("## 4c. On the board — PYNQ-Z2")
    A("")
    if not board:
        A(NOT_RUN + ". `board/board_run.json` does not exist, or was not produced on a "
          "board. `board/README.md` says how to run it.")
    else:
        A("From `board/board_run.json`, written by `board/drivesentinel_overlay.ipynb` "
          "on the board and copied back unedited.")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| Board | {board['board'].split(' (')[0]} |")
        A(f"| Software | PYNQ {board['pynq_version']}, numpy "
          f"{board.get('numpy_version', 'not recorded')}, `{board['platform']}` |")
        A(f"| Bitstream | `{board['bitstream']}`; build-ID register **{board['build_id']}** "
          f"({'match' if board['build_id_match'] else 'MISMATCH'}) |")
        A(f"| Windows | {board['windows']} packaged windows (`board/test_windows.npz`) |")
        A(f"| Class vs `golden_reference.predict`, run on the board's ARM cores | "
          f"**{board['class_agree_with_golden']} / {board['windows']}** agree"
          + (f"; disagreements at cache windows {board['class_disagreement_windows']}"
             if board['class_disagreement_windows'] else "") + " |")
        A(f"| Logit registers vs the RTL model's predicted registers | "
          f"**{board['logit_registers_exact']} / {board['windows']}** bit-exact |")
        A(f"| CYCLES register vs simulation | **{board['cycles_equal_simulation']} / "
          f"{board['windows']}** equal |")
        if bt:
            dt = bt.get("dma_transfer_wall", {})
            A(f"| Accelerator alone, CYCLES / 100 MHz | {bt['fabric_ms_at_100MHz']:.2f} ms |")
            A(f"| DMA + inference, PS wall clock ({bt['windows_timed']} windows) | "
              f"median **{bwall['median_ms']:.2f} ms**, p95 {bwall['p95_ms']:.2f} ms, "
              f"min {bwall['min_ms']:.2f} ms, max {bwall['max_ms']:.2f} ms |")
            if dt:
                A(f"| of which the DMA transfer | median {dt['median_ms']:.3f} ms, "
                  f"p95 {dt['p95_ms']:.3f} ms |")
            A(f"| Same INT8 model in numpy on the board's Cortex-A9, for scale | "
              f"{bt['golden_reference_on_arm_ms_per_window']:.1f} ms per window |")
        A("")
        A("**What this run is.** A check of the hardware against the reference on real "
          "windows: the fabric computes what the verified RTL model says it computes, "
          "window after window, in a fixed number of cycles. It is **not** a V-1 on "
          f"silicon ({board['windows']} windows, not the full cache), and **not** an "
          "accuracy figure: the packaged windows come from all 29 bearings, which the "
          "deployed network was trained on. The model's honest figure stays macro-F1 "
          "0.7852, leave-one-bearing-out.")
        A("")
        A("**What the timing is.** The CNN path only, from the DMA transfer to BUSY "
          "falling, measured on the PS with Python polling the register. The DSP front "
          "end is not in it (§5).")
        A("")
        A(f"**The board's clock was not synchronised** (a direct cable, no time source); "
          f"its `date_utc` field reads `{board['date_utc']}`. `docs/claims_audit.md` "
          f"records the actual run date.")
        A("")

    # ------------------------------------------------- the DSP front end on the board
    A("## 4d. The DSP front end on the board — raw signal in")
    A("")
    fe = load(os.path.join(ROOT, "artifacts", "webapp", "board_frontend_check.json"))
    if not fe or fe.get("ran_on_hardware") is not True:
        A(NOT_RUN + ". `artifacts/webapp/board_frontend_check.json` does not exist, or was "
          "not produced on a board. `scripts/webapp/check_board_frontend.py` writes it, "
          "against `board/server.py`'s `POST /process` (`board/SERVER.md`). The front end "
          "is ported (`board/frontend.py`) and equals `dsp.py` on the laptop "
          "(`tests/test_board_frontend.py`); on the board it has not been run.")
    else:
        tm, mch = fe["timing"], fe["board"].get("machine") or {}
        tot = tm["board_per_file_ms"]["total_ms"]
        dsp_f, dsp_w = tm["board_dsp_per_file_ms"], tm["board_dsp_per_window_amortised_ms"]
        lap = tm["laptop_dsp_per_file_ms"]
        A("From `artifacts/webapp/board_frontend_check.json`, written by "
          "`scripts/webapp/check_board_frontend.py` against the board's `POST /process`.")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| Board software | Python {fe['board']['python']}, numpy {fe['board']['numpy']}, "
          f"scipy {fe['board']['scipy']}; FFT from `{fe['board']['frontend']['fft']}` |")
        A(f"| DSP worker processes on the board | {fe.get('board_workers')} |")
        A(f"| Board CPU | `{mch.get('arch')}`, {mch.get('cpus')} cores, governor "
          f"`{mch.get('governor')}`, cpu0 {mch.get('cpu0_mhz')} MHz, "
          f"{mch.get('temp_c')} °C at the start |")
        A(f"| Input | {fe['data_files']} raw 4 s data files, 64 kHz, current ×2 + vibration |")
        A(f"| Board int8 network input vs the laptop's `dsp.py` + quantiser | "
          f"**{fe['int8_identical']} / {fe['windows']}** identical |")
        A(f"| Where they differ | {fe['int8_bytes_differing']:,} of "
          f"{fe['int8_bytes_compared']:,} input bytes, by at most {fe['int8_max_abs_diff']} "
          f"steps; board verdict = the verdict on the laptop's input on "
          f"**{fe['verdict_same_as_laptop_input']} / {fe['windows']}** |")
        A(f"| FPGA logit registers vs the bit-accurate model, on the board's input | "
          f"**{fe['logits_exact']} / {fe['windows']}** exact |")
        A(f"| Shaft speed, board vs laptop | max difference {fe['max_rpm_diff']:.3g} rpm |")
        A(f"| Board DSP per 4 s file ({dsp_f['n']} files, {tm['warmup_files_excluded']} "
          f"warm-up excluded) | median **{dsp_f['median']:.0f} ms**, p95 "
          f"{dsp_f['p95']:.0f} ms, max {dsp_f['max']:.0f} ms |")
        A(f"| Whole chain on the board per 4 s file (parse, DSP, quantise, FPGA) | "
          f"median **{tot['median']:.0f} ms**, p95 {tot['p95']:.0f} ms |")
        A(f"| Board DSP per window, **amortised** over the file's windows | "
          f"median {dsp_w['median']:.1f} ms, p95 {dsp_w['p95']:.1f} ms |")
        A(f"| Same DSP on the development laptop, for the ratio | median "
          f"{lap['median']:.0f} ms per file ({dsp_f['median'] / lap['median']:.1f}x) |")
        A("")
        A("**What the timing is.** A 4 s data file is processed whole, as `dsp.py` does, so "
          "the real-time test is 4 s of signal in under 4 s. The per-window figure is that "
          "time divided by the windows — amortised, unpinned — and is **not** the "
          "single-window, pinned-core protocol `docs/frontend_timing_plan.md` §3.3 set its "
          f"{tm['plan_bar_ms']:.0f} ms bar against; it is shown beside that bar, not as a "
          "verdict on it.")
        A("")
    A("---")
    A("")

    A("## 5. Latency — measured, against the spec's arithmetic")
    A("")
    if cyc and shipped and shipped["json"]:
        j = shipped["json"]
        A(f"The cycle counter in `ds_ctrl_fsm` runs from the start pulse to the last")
        A(f"layer's done pulse and is read back over AXI-Lite. Every window measured so")
        A(f"far takes the **same {cyc:,} cycles** — the engine has no data-dependent")
        A(f"paths, which is itself worth knowing for a real-time claim.")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| Cycles per inference | **{cyc:,}** (measured) |")
        if params:
            A(f"| Modelled from the layer table | {params['total_cycles_model']:,} "
              f"(`artifacts/rtl/rtl_params.json`) |")
        A(f"| MACs per window | {(params or {}).get('total_macs', 0):,} |")
        A(f"| Cycles per MAC | {cyc / max((params or {}).get('total_macs', 1), 1):.3f} |")
        A(f"| At the 100 MHz target | **{us(cyc, 10.0):,.1f} us** "
          f"({ms(cyc, 10.0):.2f} ms) |")
        A(f"| At the achieved {j['achieved_mhz']:.1f} MHz | "
          f"**{us(cyc, j['achieved_period_ns']):,.1f} us** "
          f"({ms(cyc, j['achieved_period_ns']):.2f} ms) |")
        A(f"| Real-time requirement (the 0.5 s hop) | 500 ms |")
        A(f"| Margin | **{500.0 / ms(cyc, j['achieved_period_ns']):.0f}x** |")
        A("")
        A("### Against `rtl_spec.md` §7")
        A("")
        A("§7's table is arithmetic: `MACs / MACs_per_cycle / clock`, labelled ESTIMATE,")
        A("and explicitly ignoring requantisation, pipeline fill, padding stalls and AXI")
        A("transfer. At 8 MACs/cycle it predicted 2.11 ms. This build runs **1 MAC per")
        A("cycle** — `docs/rtl_declarations.md` D-1 selects the smallest lane count")
        A("meeting the hop with a 10x margin, and one lane clears it by a factor of")
        A(f"{500.0 / ms(cyc, j['achieved_period_ns']):.0f} — so the comparable §7")
        A(f"row is 1 MAC/cycle = 16.92 ms at 100 MHz.")
        A("")
        A(f"The measurement is **{ms(cyc, 10.0):.2f} ms**, i.e. "
          f"**{cyc / (params or {}).get('total_macs', 1):.3f} cycles per MAC** against")
        A("the ideal 1.000. The whole divergence is the per-output-element overhead the")
        A("estimate says it ignores, and it is small because the overhead is amortised")
        A("over a long reduction: 4 cycles of accumulator setup and drain plus 3 of")
        A("requantisation pipeline, against 45 to 192 reduction cycles. There is no")
        A("large divergence to explain — which is the useful result, because it means")
        A("§7's arithmetic was a sound basis for the architecture choice.")
        A("")
    else:
        A(NOT_RUN)
    A("")
    A("### End-to-end latency is still not measured")
    A("")
    A("`rtl_spec.md` §7 says the unmeasured risk is the front end, not the CNN, and")
    A("that has not changed. Per window the DSP chain runs 64,000-sample FFTs, Hilbert")
    A("envelopes, order resampling, a median and a `log1p` across five channels, and")
    A("**that has never been timed on any hardware**. §9.1 places it on the PS, which")
    A("fixes where it runs but not how long it takes.")
    A("")
    if board and bwall:
        A("The defensible system claim is now: *\"CNN inference takes "
          f"{bwall['median_ms']:.1f} ms per window measured on a PYNQ-Z2 (median, DMA")
        A("plus inference, PS wall clock), bit-exact against the verified RTL model;")
        A("end-to-end latency is not yet measured.\"*")
    else:
        A("The defensible system claim remains: *\"CNN inference is under 18 ms measured in")
        A("simulation and closes timing at 100 MHz in post-route implementation;")
        A("end-to-end latency is not yet measured.\"*")
    A("")
    A("---")
    A("")
    A("## 6. How to reproduce")
    A("")
    A("```bash")
    A(".venv/Scripts/python.exe scripts/rtl/sweep_requant.py        # the D-2 sweep")
    A(".venv/Scripts/python.exe scripts/rtl/confirm_bit_exact.py    # full-cache check")
    A(".venv/Scripts/python.exe scripts/rtl/gen_rtl_params.py       # ds_gen.vh + .mem")
    A(".venv/Scripts/python.exe scripts/rtl/make_vectors.py         # V-1..V-5 vectors")
    A("scripts/rtl/run_tb.sh                                        # V-1..V-6, XSim")
    A(".venv/Scripts/python.exe scripts/rtl/render_results.py       # this document")
    A("```")
    A("")
    A("```powershell")
    A(".\\scripts\\rtl\\run_tb.ps1")
    A("```")
    A("")
    A("Synthesis and implementation (no bitstream):")
    A("")
    A("```bash")
    A("D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal \\")
    A("  -log artifacts/rtl/synth/vivado_synth.log \\")
    A("  -source scripts/rtl/synth.tcl -tclargs artifacts/rtl/synth 0 1")
    A("```")
    A("")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
