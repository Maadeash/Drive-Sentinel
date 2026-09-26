"""
Run the RTL verification plan: compile, elaborate, simulate, collect.

One entry point for V-1 .. V-6 of `rtl_spec.md` section 8. `scripts/rtl/run_tb.ps1`
and `scripts/rtl/run_tb.sh` are one-line wrappers around it, so re-running the whole
verification is one command on either shell.

    .venv/Scripts/python.exe scripts/rtl/run_verify.py --stages all

Everything it produces lands in `artifacts/rtl/verify.json`, and
`scripts/rtl/render_results.py` generates `docs/results_rtl.md` from that. No
verification number is ever typed by hand.

WHY EACH SHARD GETS ITS OWN SNAPSHOT
------------------------------------
V-1 is 16,211 windows at roughly 1.74 million cycles each, which is about a day of
XSim on one core. `docs/rtl_declarations.md` D-6 allows splitting it across
processes over disjoint window ranges and forbids subsampling. Two XSim processes
sharing one elaborated snapshot directory write over each other's run files, so
each shard is elaborated separately -- twenty seconds of elaboration to avoid a
class of failure that would look like a simulation bug.

ORDER
-----
V-3 before V-1, per the spec: argmax agreement can hide a broken layer, so the
per-layer accumulator comparison is the one you want to fail first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drivesentinel import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTL = os.path.join(ROOT, "rtl")
TB = os.path.join(ROOT, "tb")
SIM = os.path.join(C.ARTIFACT_DIR, "rtl", "sim")
VEC = os.path.join(C.ARTIFACT_DIR, "rtl", "vectors")
OUT = os.path.join(C.ARTIFACT_DIR, "rtl")

# Source order matters only for readability; xvlog resolves module references at
# elaboration. Kept bottom-up so a reader meets the leaves first.
RTL_FILES = [
    "ds_mac_array.v", "ds_requant.v", "ds_relu_clamp.v", "ds_global_pool.v",
    "ds_fc.v", "ds_weight_mem.v", "ds_bias_mem.v", "ds_param_mem.v",
    "ds_act_mem.v", "ds_conv1d.v", "ds_ctrl_fsm.v", "ds_axis_in.v",
    "ds_axil_ctrl.v", "ds_weight_crc.v", "ds_core.v", "ds_top.v",
]
TB_FILES = ["tb_requant.sv", "tb_ds_top.sv"]

DEFAULT_VIVADO = r"D:\Vivado\2023.2\bin"


def tool(vivado_bin: str, name: str) -> str:
    return os.path.join(vivado_bin, name + ".bat")


def run(cmd: List[str], label: str, log: str = None) -> int:
    t0 = time.time()
    with open(log, "w") if log else _null() as fh:
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh or subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
    print(f"    {label}: exit {p.returncode} [{time.time() - t0:.0f}s]", flush=True)
    return p.returncode


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def compile_all(vivado_bin: str) -> None:
    print("  compiling")
    rc = run([tool(vivado_bin, "xvlog"), "-i", "rtl", "--nolog"]
             + [os.path.join("rtl", f) for f in RTL_FILES],
             "xvlog rtl", os.path.join(SIM, "xvlog_rtl.log"))
    if rc:
        sys.exit("xvlog failed on the RTL -- see artifacts/rtl/sim/xvlog_rtl.log")
    rc = run([tool(vivado_bin, "xvlog"), "-sv", "-i", "rtl", "--nolog"]
             + [os.path.join("tb", f) for f in TB_FILES],
             "xvlog tb", os.path.join(SIM, "xvlog_tb.log"))
    if rc:
        sys.exit("xvlog failed on the testbenches -- see artifacts/rtl/sim/xvlog_tb.log")


def elaborate(vivado_bin: str, snapshot: str, top: str) -> None:
    rc = run([tool(vivado_bin, "xelab"), "--nolog", "--timescale", "1ns/1ps",
              "-debug", "off", "-s", snapshot, f"work.{top}"],
             f"xelab {snapshot}", os.path.join(SIM, f"xelab_{snapshot}.log"))
    if rc:
        sys.exit(f"xelab failed for {snapshot}")


def write_cfg(cfg_id: int, kv: Dict[str, str]) -> str:
    path = os.path.join(SIM, f"cfg_{cfg_id}.txt")
    with open(path, "w", newline="\n") as fh:
        for k, v in kv.items():
            fh.write(f"{k} {v}\n")
    return path


def parse_report(path: str) -> Dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                k, v = parts
                out[k] = int(v) if (v.lstrip("-").isdigit()) else v
    return out


def launch(vivado_bin: str, snapshot: str, cfg_id: int, log: str):
    return subprocess.Popen(
        [tool(vivado_bin, "xsim"), snapshot, "--nolog", "--runall",
         "--testplusarg", f"CFG{cfg_id}"],
        cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def vec_meta(name: str) -> Dict:
    with open(os.path.join(VEC, name, "meta.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------

def stage_v4(vivado_bin: str) -> Dict:
    m = vec_meta("v4")
    write_cfg(40, {"vec": "artifacts/rtl/vectors/v4/requant_vectors.txt",
                   "result": "artifacts/rtl/sim/v4.txt"})
    p = launch(vivado_bin, "sim_requant", 40, os.path.join(SIM, "v4.log"))
    p.wait()
    r = parse_report(os.path.join(SIM, "v4.txt"))
    r["meta"] = m
    r["bar"] = "exact match on every tie"
    r["pass"] = bool(r) and r.get("q_mismatches") == 0 \
        and r.get("clamp_mismatches") == 0 and r.get("vectors", 0) > 0 \
        and r.get("exact_ties", 0) > 0
    return r


def stage_windows(vivado_bin: str, name: str, shards: int, with_acc: bool,
                  build_id: int) -> Dict:
    m = vec_meta(name)
    procs, cfgs = [], []
    base = {"v3": 30, "v5": 50, "v1": 100}[name]
    for sh in m["shards"]:
        s = sh["shard"]
        snapshot = f"sim_top_{s}" if shards > 1 else "sim_top"
        kv = {
            "set": f"{name}.{s}",
            "in": f"artifacts/rtl/vectors/{name}/{sh['in']}",
            "exp": f"artifacts/rtl/vectors/{name}/{sh['exp']}",
            "nwin": str(sh["n_windows"]),
            "first": str(sh["first_window"]),
            "buildid": str(build_id),
            "result": f"artifacts/rtl/sim/{name}_{s}.txt",
        }
        if with_acc:
            kv["acc"] = f"artifacts/rtl/vectors/{name}/{m['acc_file']}"
        write_cfg(base + s, kv)
        cfgs.append(base + s)
        procs.append(launch(vivado_bin, snapshot, base + s,
                            os.path.join(SIM, f"{name}_{s}.log")))
        print(f"    launched {name} shard {s}: windows "
              f"{sh['first_window']}..{sh['first_window'] + sh['n_windows'] - 1}",
              flush=True)

    t0 = time.time()
    for p in procs:
        p.wait()
    elapsed = time.time() - t0

    parts = [parse_report(os.path.join(SIM, f"{name}_{sh['shard']}.txt"))
             for sh in m["shards"]]
    return aggregate(name, m, parts, elapsed)


def aggregate(name: str, m: Dict, parts: List[Dict], elapsed: float) -> Dict:
    """
    Sum per-shard reports into one verdict.

    Only reports that actually ran count. The first version of this function
    compared `build_id_got == build_id_expected` over every shard, including
    shards with no report at all -- where both sides are None, so the
    comparison said True. A run killed before any shard finished therefore
    reported `build_id_match: true` with nothing behind it. A vacuous pass is
    the one kind of pass this project cannot afford, so an empty set of
    evidence now reports False.
    """
    ran = [p for p in parts if p.get("windows_run", 0) > 0]
    total = {
        "set": name,
        "shards": len(parts),
        "shards_with_evidence": len(ran),
        "seconds": round(elapsed, 1),
        "windows_run": sum(p.get("windows_run", 0) for p in parts),
        "windows_expected": m["n_windows"],
        "class_mismatches": sum(p.get("class_mismatches", 0) for p in parts),
        "logit_mismatches": sum(p.get("logit_mismatches", 0) for p in parts),
        "max_abs_logit_diff_lsb": max([p.get("max_abs_logit_diff_lsb", 0)
                                       for p in ran] or [0]),
        "acc_compared": sum(p.get("acc_compared", 0) for p in parts),
        "acc_mismatches": sum(p.get("acc_mismatches", 0) for p in parts),
        "build_id_match": bool(ran) and all(
            p.get("build_id_got") is not None
            and p.get("build_id_got") == p.get("build_id_expected") for p in ran),
        "build_id_got": ran[0].get("build_id_got") if ran else None,
        "cycles_min": min([p["cycles_min"] for p in ran] or [0]),
        "cycles_max": max([p["cycles_max"] for p in ran] or [0]),
        "per_shard": parts,
        "meta": {k: v for k, v in m.items() if k != "shards"},
    }
    total["complete"] = total["windows_run"] == total["windows_expected"]
    if not total["complete"]:
        total["status_note"] = (
            "PARTIAL: %d of %d windows ran. docs/rtl_declarations.md D-6 forbids "
            "calling a subset V-1." % (total["windows_run"],
                                       total["windows_expected"]))
    return total


# ---------------------------------------------------------------------------
# V-1, chunked and resumable
# ---------------------------------------------------------------------------

def snapshot_v1() -> Dict:
    """
    Aggregate whatever V-1 chunk reports exist right now, without launching or
    touching anything.

    The testbench rewrites its report after every window, so this reads the
    evidence of a run that is still going -- or of one that was killed -- and
    lets it be committed as PARTIAL with the exact count. It is a read-only view;
    the running job keeps going and writes its own final aggregate when done.
    """
    m = vec_meta("v1")
    parts = [parse_report(os.path.join(SIM, f"v1_{sh['shard']}.txt"))
             for sh in m["shards"]]
    out = aggregate("v1", m, parts, 0.0)
    out["chunks_complete"] = sum(1 for sh in m["shards"] if chunk_done(sh))
    out["dut_config"] = "DEBUG_TRACE=0 (the shipped, synthesised configuration)"
    out["snapshot"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["workers"] = "see artifacts/rtl/bench_workers.json (2 by default)"
    return out


def compile_shipped_tb(vivado_bin: str) -> None:
    """
    The testbench again, into its own library, with TB_NO_TRACE defined so the
    DUT is built with DEBUG_TRACE = 0 -- the configuration synthesis reports and
    the one that ships. V-1's verdict should be about that design.

    Its own library, because defining the macro into `work` would silently turn
    the V-3 accumulator monitor off for every other snapshot as well. The RTL is
    compiled into that library too, from the same sources: xelab did not resolve
    ds_top across libraries with -L, and a self-contained library removes the
    question of which compiled copy of the design a snapshot picked up.
    """
    rc = run([tool(vivado_bin, "xvlog"), "-i", "rtl", "--nolog", "--work", "ship"]
             + [os.path.join("rtl", f) for f in RTL_FILES],
             "xvlog rtl (shipped config)", os.path.join(SIM, "xvlog_ship_rtl.log"))
    if rc:
        sys.exit("xvlog failed on the RTL for the shipped-config library")
    rc = run([tool(vivado_bin, "xvlog"), "-sv", "-i", "rtl", "--nolog",
              "--define", "TB_NO_TRACE", "--work", "ship",
              os.path.join("tb", "tb_ds_top.sv")],
             "xvlog tb (shipped config)", os.path.join(SIM, "xvlog_ship.log"))
    if rc:
        sys.exit("xvlog failed on the shipped-config testbench")


def elaborate_shipped(vivado_bin: str, snapshot: str) -> None:
    rc = run([tool(vivado_bin, "xelab"), "--nolog", "--timescale", "1ns/1ps",
              "-debug", "off", "-L", "ship", "-s", snapshot, "ship.tb_ds_top"],
             f"xelab {snapshot}", os.path.join(SIM, f"xelab_{snapshot}.log"))
    if rc:
        sys.exit(f"xelab failed for {snapshot}")


def chunk_done(sh: Dict) -> bool:
    r = parse_report(os.path.join(SIM, f"v1_{sh['shard']}.txt"))
    return r.get("windows_run", 0) == sh["n_windows"]


def stage_v1_chunked(vivado_bin: str, workers: int, build_id: int,
                     max_hours: float = None) -> Dict:
    """
    V-1 as a queue of small chunks, run `workers` at a time, resumable.

    WHY. The first full V-1 attempt ran 4 shards of ~4,050 windows for 4.5 hours,
    reached window 768 of every shard with zero mismatches -- and recorded none
    of it, because the testbench only wrote its report at $finish and the run
    was killed before that. On this machine V-1 is most of a day. So:

      * the testbench now rewrites its report after every window;
      * the windows are split into ~250-window chunks
        (scripts/rtl/make_vectors.py --sets v1 --shards N);
      * a chunk whose report says it finished is skipped on the next run, so a
        killed run loses at most the chunks that were in flight.

    Re-running this command resumes. The union of finished chunks is V-1; any
    shortfall is reported as PARTIAL with the exact count (declarations D-6),
    never as V-1.
    """
    m = vec_meta("v1")
    todo = [sh for sh in m["shards"] if not chunk_done(sh)]
    done_before = len(m["shards"]) - len(todo)
    print(f"    {len(m['shards'])} chunks, {done_before} already complete, "
          f"{len(todo)} to run on {workers} workers", flush=True)

    free = [f"sim_ship_{w}" for w in range(workers)]
    running = []            # (proc, snapshot, chunk, t_start)
    t0 = time.time()
    deadline = t0 + max_hours * 3600 if max_hours else None
    log_path = os.path.join(SIM, "v1_progress.log")

    while todo or running:
        while free and todo and (deadline is None or time.time() < deadline):
            sh = todo.pop(0)
            snap = free.pop(0)
            s = sh["shard"]
            write_cfg(1000 + s, {
                "set": f"v1.{s}",
                "in": f"artifacts/rtl/vectors/v1/{sh['in']}",
                "exp": f"artifacts/rtl/vectors/v1/{sh['exp']}",
                "nwin": str(sh["n_windows"]),
                "first": str(sh["first_window"]),
                "buildid": str(build_id),
                "result": f"artifacts/rtl/sim/v1_{s}.txt",
            })
            running.append((launch(vivado_bin, snap, 1000 + s,
                                   os.path.join(SIM, f"v1_{s}.log")),
                            snap, sh, time.time()))
        if not running:
            break
        time.sleep(5)
        still = []
        for proc, snap, sh, ts in running:
            if proc.poll() is None:
                still.append((proc, snap, sh, ts))
                continue
            free.append(snap)
            r = parse_report(os.path.join(SIM, f"v1_{sh['shard']}.txt"))
            line = (f"chunk {sh['shard']:3d} windows {sh['first_window']}.."
                    f"{sh['first_window'] + sh['n_windows'] - 1}: ran "
                    f"{r.get('windows_run', 0)}/{sh['n_windows']}, class "
                    f"mismatches {r.get('class_mismatches', '?')}, logit "
                    f"mismatches {r.get('logit_mismatches', '?')} "
                    f"[{time.time() - ts:.0f}s]")
            print("    " + line, flush=True)
            with open(log_path, "a") as fh:
                fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line + "\n")
        running = still

    parts = [parse_report(os.path.join(SIM, f"v1_{sh['shard']}.txt"))
             for sh in m["shards"]]
    out = aggregate("v1", m, parts, time.time() - t0)
    out["workers"] = workers
    out["chunks_complete"] = sum(1 for sh in m["shards"] if chunk_done(sh))
    out["dut_config"] = "DEBUG_TRACE=0 (the shipped, synthesised configuration)"
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vivado-bin", default=os.environ.get("VIVADO_BIN",
                                                           DEFAULT_VIVADO))
    ap.add_argument("--stages", nargs="+",
                    default=["compile", "v4", "v3", "v5", "v1"],
                    help="'all' runs compile v4 v3 v5 v1, in that order")
    # Two, not four, on this 2-core / 4-thread machine: measured by
    # scripts/rtl/bench_workers.py -- 0.260 windows/s at 2 workers against
    # 0.245 at 4 (artifacts/rtl/bench_workers.json).
    ap.add_argument("--shards", type=int, default=2,
                    help="parallel XSim workers for V-1")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop launching new V-1 chunks after this long; the "
                         "run can be resumed by re-running the same command")
    ap.add_argument("--out", default=os.path.join(OUT, "verify.json"))
    args = ap.parse_args()

    stages = ["compile", "v4", "v3", "v5", "v1"] if args.stages == ["all"] \
        else args.stages
    os.makedirs(SIM, exist_ok=True)

    with open(os.path.join(OUT, "rtl_params.json")) as fh:
        params = json.load(fh)
    build_id = params["expected_build_id"]

    report = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            report = json.load(fh)
    report.setdefault("runs", {})
    report["vivado"] = args.vivado_bin
    report["rtl_params"] = {k: params[k] for k in
                            ("frac_width", "frac_width_source", "logit_shift",
                             "fc_shift", "logit_lsb_value", "total_macs",
                             "total_cycles_model", "expected_build_id_hex")}

    if "compile" in stages:
        compile_all(args.vivado_bin)
        elaborate(args.vivado_bin, "sim_requant", "tb_requant")
        elaborate(args.vivado_bin, "sim_top", "tb_ds_top")

    if "compile" in stages or "v1" in stages:
        compile_shipped_tb(args.vivado_bin)
        for w in range(args.shards):
            elaborate_shipped(args.vivado_bin, f"sim_ship_{w}")

    if "v1snap" in stages:
        report["runs"]["v1"] = snapshot_v1()
        r = report["runs"]["v1"]
        print(f"  v1 snapshot: {r['windows_run']:,} / {r['windows_expected']:,} "
              f"windows, {r['chunks_complete']} chunks complete, class mismatches "
              f"{r['class_mismatches']}, logit mismatches {r['logit_mismatches']}")

    for name in ("v4", "v3", "v5", "v1"):
        if name not in stages:
            continue
        print(f"  {name}", flush=True)
        if name == "v4":
            report["runs"]["v4"] = stage_v4(args.vivado_bin)
        else:
            if name == "v1":
                report["runs"]["v1"] = stage_v1_chunked(
                    args.vivado_bin, args.shards, build_id, args.max_hours)
            else:
                report["runs"][name] = stage_windows(args.vivado_bin, name, 1,
                                                     with_acc=(name == "v3"),
                                                     build_id=build_id)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        r = report["runs"][name]
        print(f"    -> {json.dumps({k: v for k, v in r.items() if k not in ('per_shard', 'meta')})}",
              flush=True)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
