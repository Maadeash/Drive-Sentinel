"""
How many XSim processes this machine should run at once for V-1.

V-1 is 16,211 windows at ~1.79 million cycles each, and on this machine (i3-1115G4,
2 cores / 4 threads) that is most of a day. The worker count is therefore worth
measuring rather than guessing: 4 threads suggests 4 workers, and the measurement
says otherwise.

Runs the shipped-config snapshot (`sim_ship_*`, built by run_verify.py) on 12
saturation-probe windows at 1, 2 and 4 concurrent processes and writes
`artifacts/rtl/bench_workers.json`. Measured 2026-09-21:

    1 worker   0.180 windows/s   full V-1 ~25.0 h
    2 workers  0.260 windows/s   full V-1 ~17.3 h   <- the default
    4 workers  0.245 windows/s   full V-1 ~18.4 h

Four is slower than two: the two extra processes land on hyperthreads of cores
already busy, and XSim's kernel is not the kind of workload that benefits.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import run_verify as R  # noqa: E402

N = 12


def main() -> None:
    vb = os.environ.get("VIVADO_BIN", R.DEFAULT_VIVADO)
    out = {}
    for workers in (1, 2, 4):
        for w in range(workers):
            R.write_cfg(2000 + w, {
                "set": f"bench.{workers}.{w}",
                "in": "artifacts/rtl/vectors/v5/in_0.hex",
                "exp": "artifacts/rtl/vectors/v5/exp_0.txt",
                "nwin": str(N), "first": "0",
                "buildid": str(json.load(open(os.path.join(
                    R.OUT, "rtl_params.json")))["expected_build_id"]),
                "result": f"artifacts/rtl/sim/bench_{workers}_{w}.txt",
            })
        t0 = time.time()
        procs = [R.launch(vb, f"sim_ship_{w}", 2000 + w,
                          os.path.join(R.SIM, f"bench_{workers}_{w}.log"))
                 for w in range(workers)]
        for p in procs:
            p.wait()
        dt = time.time() - t0
        ok = all(R.parse_report(os.path.join(R.SIM, f"bench_{workers}_{w}.txt"))
                 .get("windows_run") == N for w in range(workers))
        rate = workers * N / dt
        out[workers] = {"seconds": round(dt, 1), "windows_per_s": round(rate, 4),
                        "full_v1_hours": round(16211 / rate / 3600, 1),
                        "all_ok": ok}
        print(workers, out[workers], flush=True)
    with open(os.path.join(R.OUT, "bench_workers.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
