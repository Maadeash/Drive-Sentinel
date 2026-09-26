
# Front-end timing — a measurement plan, not a measurement

**Nothing in this document is a result.** It states a method so that the one open
question in the system latency story has a stated way of being answered, instead of
being answered by assertion later. No code here has been written and no timing has
been taken. Every figure this plan would produce reads `NOT RUN` until it does.

---

## 1. The question, stated precisely

`rtl_spec.md` §7 has said from the start that **the unmeasured risk is the front end,
not the CNN**, and after the RTL build that is still the only thing standing between
this project and an end-to-end latency claim:

| Stage | Status |
|---|---|
| DSP front end — FFT, Hilbert envelope, order resampling, median, `log1p`, 5 channels | **NEVER TIMED ON ANY HARDWARE** |
| Input standardise + quantise to int8 (`rtl_declarations.md` D-4) | **NEVER TIMED ON ANY HARDWARE** |
| AXI-Stream transfer of 2,560 bytes | 2,560 cycles in simulation, PS-side DMA cost unmeasured |
| CNN inference in the PL | **measured: 1,786,724 cycles = 17.87 ms at 100 MHz** |

The question is one number with a pass/fail attached:

> **Does one hop's worth of front-end work complete in under 0.5 s on the PYNQ-Z2's
> PS (dual-core ARM Cortex-A9 at 650 MHz), for all five channels, with enough margin
> to survive a real workload?**

If it does, the system is real-time and the end-to-end figure is
`front_end + transfer + 17.87 ms`. If it does not, the front end has to move — to the
PL, to a lower resolution, or to a longer hop — and *that* is a design decision this
project has not had to take because it has never had the number.

---

## 2. Why this cannot be answered on the development machine

`CLAUDE.md` is explicit that the "minutes on a 3050" timings in older documents do not
transfer to this machine and must be re-measured. The same discipline forbids the
reverse substitution: **a timing taken on this i3-1115G4 would say nothing about a
650 MHz Cortex-A9.** The differences are not a constant factor —

- vector width (AVX2 versus NEON) and how well the FFT library uses it,
- cache hierarchy, against a 64,000-sample working set per channel,
- memory bandwidth, which is where order resampling and the median will live,
- the FFT implementation itself: `scipy.fft` (pocketfft) on x86 versus whatever the
  PS runs.

So a desktop number would be an estimate dressed as a measurement. This project has a
name for that and does not ship it.

**This plan therefore requires a board.** `rtl_spec.md` §9.4 is answered *no board*,
so this plan is **blocked, and is declared blocked** rather than approximated. That is
the honest status: the method is ready, the hardware is not available.

---

## 3. What would be measured

### 3.1 The unit under test

`drivesentinel.dsp.order_spectrum_window()` and its callees, for **one 1.0 s window of
five channels at 64 kHz**, ending at the int8 bytes the AXI-Stream carries. Concretely
the chain in `dsp.py`, per channel:

| # | Step | Dominant cost |
|---|---|---|
| 1 | Decimate to the working rate | FIR, O(N) |
| 2 | Band-pass / notch the supply harmonics | filter, O(N) |
| 3 | Hilbert envelope (channels 2, 3) | **two 64k FFTs** |
| 4 | FFT to the frequency axis | **one 64k FFT** |
| 5 | Order resampling onto 512 bins over 0–16 orders | interpolation, memory-bound |
| 6 | Median normalise, then `log1p` | O(bins), a partial sort |
| 7 | Standardise and quantise to int8 (D-4) | O(bins), trivial |

Steps 3 and 4 are the arithmetic; step 5 is where an unpleasant surprise would most
likely live, because it is a gather and not a stream.

### 3.2 The four numbers to report

1. **Wall-clock per window**, over ≥ 200 consecutive windows drawn from a real
   recording — not synthetic data, because the median and the resampling are
   data-dependent in their memory access patterns.
2. **The distribution, not the mean.** Report min / median / p95 / max. A real-time
   claim is about the tail; a mean that fits the hop while the p95 does not is a system
   that drops windows.
3. **Per-step breakdown** at the granularity of the table above, so that a failure is
   actionable rather than just a verdict.
4. **Single-core and both-cores.** The A9 is dual-core and five channels are
   embarrassingly parallel, so a 2x is available if it is needed. Measure single-core
   first — a plan that needs both cores to pass has no margin left for the rest of the
   application.

### 3.3 The declared bar, fixed now

Fixed here, before any measurement, in the style of `rtl_declarations.md`:

> **The front end passes if the p95 per-window wall-clock, single-core, is at or below
> 250 ms** — half the 0.5 s hop, leaving the other half for the CNN's measured 17.87 ms,
> the AXI transfers, the four scalar-feature branches, the fusion layer and the
> operating system.
>
> If the p95 exceeds 250 ms but the median does not, that is reported as a **marginal**
> result, with the distribution, and is *not* rounded down to a pass.
> If it exceeds 500 ms the front end does not meet the hop and that is a negative
> result, reported the way the D2 CNN was.

The bar is 250 ms and not 500 ms on purpose: a system that exactly fills its hop has no
margin, and the hop is the *real-time* requirement, not the *latency* requirement.

---

## 4. Method

### 4.1 Instrumentation

A script — `scripts/rtl/time_frontend.py`, **not written** — that:

1. loads one Paderborn recording from the cache-building path, so the input is the same
   data the trained model saw;
2. iterates windows with `config.WINDOW_SECONDS` / `config.HOP_SECONDS`;
3. times each step with `time.perf_counter_ns()` around it, accumulating per step;
4. discards the first 20 windows as warm-up (FFT plan creation, page faults, governor
   ramp) and says so in the output;
5. writes `artifacts/rtl/frontend_timing.json` with every per-window figure, not just
   the summary, so the distribution can be re-derived without re-running;
6. records `platform.machine()`, `platform.processor()`, the CPU clock as reported by
   the OS, the governor setting, the numpy and scipy versions and their BLAS/FFT
   backends — because a timing without its machine is not a measurement.

`docs/results_rtl.md` would gain a section generated from that JSON. Until the JSON
exists it reads `NOT RUN`, per the standing rule.

### 4.2 Controls that make the number mean something

These are the parts that are easy to skip and are the reason a timing is believed:

- **Pin the CPU governor to `performance`** and record that it was pinned. A PYNQ image
  defaults to `ondemand`; a 200-window loop on an idle board will ramp the clock
  partway through and produce a bimodal distribution that looks like a cache effect.
- **Pin the process to one core** (`taskset`/`sched_setaffinity`) for the single-core
  figure, so the scheduler cannot quietly hand it two.
- **Report the thermal state.** The Zynq PS throttles; a 200-window run is long enough
  to matter. Read `/sys/class/thermal/thermal_zone0/temp` at the start and the end.
- **Run the same script on this development machine as well** and publish both,
  clearly labelled. Not because the desktop number is the answer — §2 says it is not —
  but because the *ratio* between them is the thing that would let a future reader
  sanity-check a claim made about a third platform.
- **Check the output, not just the clock.** Assert that the int8 bytes the timed path
  produces are identical to `drivesentinel.rtl.model.quantise_input` on the same
  window. A fast path that computes something slightly different is not a measurement
  of this system.

### 4.3 What would then be composed, and how carefully

If and only if the front end passes, the end-to-end figure is:

```
end_to_end = front_end_p95            (measured, PS)
           + axis_transfer            (measured, PS-side DMA; NOT the 2,560 simulated cycles)
           + 17.87 ms                 (measured, PL, at 100 MHz)
           + fusion + scalar branches (measured, PS)
```

Each term measured separately and **added with its own p95**, not with its mean. The
sum of medians is not the median of the sum, and a real-time claim built from means is
a claim about the average second rather than about the worst one.

---

## 5. Status

| | |
|---|---|
| Plan | **written** (this document) |
| Implementation | **NOT WRITTEN** |
| Measurement | **NOT RUN** |
| Blocker | no PYNQ-Z2 available; `rtl_spec.md` §9.4 answered *no board* |
| Bar | **p95 ≤ 250 ms per window, single-core**, fixed above before any measurement |

Until this runs, the system claim stays exactly where `docs/results_rtl.md` §5 leaves
it:

> *"CNN inference is 17.87 ms, measured in simulation and closing timing at 100 MHz
> post-route. End-to-end latency is not yet measured, because the DSP front end has
> never been timed on any hardware."*
