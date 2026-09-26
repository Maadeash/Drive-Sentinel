# RTL specification — INT8 bearing-CNN accelerator (PYNQ-Z2 / Zynq XC7Z020)

**Status: BUILT; V-2..V-6 PASS IN SIMULATION, V-1 IN PROGRESS; SYNTHESISED AND
IMPLEMENTED; NOT RUN ON HARDWARE.** This was item (a) of the RTL track — the specification, written before any
Verilog existed, for the same reason every other recipe in this project is declared
first: a bar moved after seeing the result is not a bar. The RTL is now in `rtl/`, the
testbenches in `tb/`, and the results in **`docs/results_rtl.md`**, which is generated
from JSON and is the authority for every measured figure.

**This document has not been rewritten to match the outcome.** Sections 1–8 stand as
they were declared. What has changed is only this: §9's five open questions are now
answered, and figures that were labelled ESTIMATE are marked with the measurement that
replaced them and a pointer to the report. Everything else is the original text,
including the parts the measurements went on to complicate — §2 against §7 being the
clearest, resolved in `docs/rtl_declarations.md` D-4 and *not* edited away here.

Every figure below is derived from `artifacts/int8_export/` and the frozen v5 pipeline,
not from the plan's prose. Derivations are reproducible from `golden_reference.py` and
`scales.json`. **No bitstream has been written and nothing has run on a board**, so no
power figure and no end-to-end latency figure exists, and the resource and clock
figures come from a Vivado post-route report rather than from silicon.

---

## 1. Scope

| In scope (this spec) | Out of scope (separate specs) |
|---|---|
| The 5-layer INT8 CNN inference core | The DSP front end (FFT, envelope, order resampling, median, log1p) |
| Weight/bias storage and load | Sensor acquisition and ADC interface |
| Requantisation and argmax | The fusion layer, trip gating, dashboard |
| AXI-Lite control + AXI-Stream data path | The multi-stage branches (S1–S4) |
| Bit-agreement verification against the golden reference | Timing closure sign-off |

**The accelerator covers stage S5 (bearing) only.** The winding, supply and inverter
telemetry branches are scalar-feature models running in software and are not part of
this hardware. Anyone reading the demo must not conclude the FPGA does all five stages.

### What this accelerates, honestly

The model is the frozen v5 3-class bearing CNN. Its honest score is **macro-F1 0.7852,
leave-one-bearing-out over 29 bearings** (`artifacts/runs/lobo_summary.json`). The
`0.9915` INT8 accuracy in `verification_full.json` is agreement on the pooled window set
and is **not** a generalisation estimate. Putting this model in hardware does not change
its accuracy in either direction. The hardware claim is latency and power, nothing else.

---

## 2. Input contract

Fixed by `dsp.py` and `config.py`; the accelerator does not get to renegotiate it.

| Property | Value | Source |
|---|---|---|
| Shape | `(5, 512)` float32 | `golden_reference.N_CHANNELS/N_BINS` |
| Axis | shaft order, 0–16.0 orders, **0.03125 orders/bin** | `config.ORDER_RESOLUTION` |
| Window / hop | **1.0 s / 0.5 s** | `config.WINDOW_SECONDS`, `HOP_SECONDS` |
| Source rate | 64 kHz (phase currents, vibration) | `config.FS_FAST` |
| Conditioning | median-normalise then `log1p`, per channel | `dsp.condition_spectrum` |
| Classes | `healthy`, `inner_race`, `outer_race` | `config.LABELS` |

Channels, in order: (0) current sideband fold, (1) raw current order spectrum,
(2) vibration envelope 500–2500 Hz, (3) vibration envelope 2500–10000 Hz,
(4) raw log vibration spectrum.

Standardisation `x = (x - mean) / std` per channel is applied **inside** the accelerator
boundary, using `input_norm.json`. The host passes conditioned DSP output directly.

---

## 3. Topology — measured, not quoted

| Layer | Kind | In | Out | Weights (O,I,K) | Stride | Pad | int8 weights | MACs |
|---|---|---|---|---|---|---|---|---|
| conv1 | conv1d | 5x512 | 16x256 | (16,5,9) | 2 | 4 | 720 | 184,320 |
| conv2 | conv1d | 16x256 | 32x128 | (32,16,7) | 2 | 3 | 3,584 | 458,752 |
| conv3 | conv1d | 32x128 | 64x64 | (64,32,5) | 2 | 2 | 10,240 | 655,360 |
| conv4 | conv1d | 64x64 | 64x32 | (64,64,3) | 2 | 1 | 12,288 | 393,216 |
| pool | global mean over length 32 | 64x32 | 64 | — | — | — | 0 | 0 |
| fc | linear | 64 | 3 | (3,64) | — | — | 192 | 192 |
| **total** | | | | | | | **27,024** | **1,691,840** |

Biases: 16+32+64+64+3 = **179**, int32.

Padding is zero padding applied to the **quantised** activation, not the float one
(`golden_reference.run_int8`). Padding with the float zero and padding with integer zero
coincide here only because the quantisation is symmetric with no zero-point offset. That
is a property of this export, and the RTL may rely on it.

---

## 4. Arithmetic contract — the part that must be exact

Per layer, for output channel `o`:

```
x_int8    = clamp(round_half_to_even(x_float / s_x), -128, 127)
acc_int32 = sum(W_int8 * x_int8) + b_int32                 # exact, no rounding
y_float   = s_W[o] * s_x * acc_int32
y_relu    = max(y_float, 0)                                # every layer except fc
```

Four properties the RTL must honour:

1. **Symmetric quantisation, no zero point.** There is no offset term anywhere. The MAC
   is a plain signed 8x8 product sum.
2. **Per-output-channel weight scales.** `s_W` is a vector, not a scalar
   (`scales.json: quantisation.per_channel_weights = true`). A scalar-scale
   implementation will not match.
3. **Round-half-to-even.** `numpy.rint` rounds ties to even: `rint(0.5) = 0`,
   `rint(1.5) = 2`, `rint(2.5) = 2`. The common hardware shortcut `floor(x + 0.5)`
   gives 1, 2, 3 and **will disagree on ties**. This is the single most likely source of
   a mismatch that looks like a weight-loading bug and is not one.
4. **ReLU before requantisation**, on the dequantised value. Since all scales are
   positive this is equivalent to clamping `acc_int32` at 0, which is what the RTL
   should do.

### Accumulator widths — no saturation logic needed

Measured peaks over the calibration set (`scales.json: accumulator`):

| Layer | Terms summed | Peak abs | Bits | Fits int32 |
|---|---|---|---|---|
| conv1 | 45 | 733,886 | 21 | yes |
| conv2 | 112 | 1,839,894 | 22 | yes |
| conv3 | 160 | 2,642,495 | 23 | yes |
| conv4 | 192 | 3,149,695 | 23 | yes |
| fc | 64 | 1,041,656 | 21 | yes |

The absolute worst case is bounded independently of the data: 192 terms of 127x128 is
3,121,152, needing 23 bits. **int32 accumulators cannot overflow**, so the RTL carries no
saturation logic in the MAC path. The clamp at the int8 output is the only saturating
operation in the design.

### Global average pool

The golden reference pools the **dequantised, ReLU'd** conv4 output over length 32, then
requantises for the fc layer. Because the per-channel scale is constant along that axis,
the RTL may sum the 32 int32 accumulators first and fold `1/32` into the requantisation
multiplier — mathematically identical, and it avoids a divider.

Worst case pooled sum: `32 x 3,149,695 = 100,790,240` -> **28 bits signed, fits int32.**

---

## 5. Requantisation — the one real design decision

Consecutive layers chain through a single per-output-channel multiplier:

```
M[o] = s_W[o] * s_x / s_x_next            (conv1..conv3)
M[o] = s_W[o] * s_x / (32 * s_x_next)     (conv4, pool folded in)
x_int8_next = clamp(round_half_to_even(M[o] * acc_int32), -128, 127)
```

Measured multiplier ranges:

| Layer | min M | max M | log2 range |
|---|---|---|---|
| conv1 | 1.7591e-3 | 4.2667e-3 | -9.15 to -7.87 |
| conv2 | 1.2598e-3 | 4.3729e-3 | -9.63 to -7.84 |
| conv3 | 1.4396e-3 | 5.2528e-3 | -9.44 to -7.57 |
| conv4 (pool folded in) | 5.5157e-5 | 1.0290e-4 | -14.15 to -13.25 |

The fc layer is **not** requantised: its output scale `s_W[o] * s_x` spans 8.2081e-5 to
9.4877e-5 across the three classes. Because those three scales differ, the argmax **must**
be taken on the scaled value, not on the raw accumulator. Taking argmax on `acc_int32`
directly is wrong, and it is wrong quietly — the classes are close enough that it would
mostly agree.

**Proposed representation:** per-channel `M0[o]` as an unsigned Q0.31 integer plus a
per-channel right shift, so `M[o] = M0[o] * 2^-(31+shift[o])`. Relative error is then
about `2^-31`; with `acc` bounded at `2^23`, the absolute error in the product is roughly
`2^-8` LSB, so disagreement with the golden reference is possible only when the exact
product lands within `2^-8` of a `.5` tie boundary.

Requantisation happens once per output element, not per MAC: 4,096 + 4,096 + 4,096 + 64 +
3 = **12,355 requant operations per window**, against 1,691,840 MACs. A single
multiplier suffices; this is not a throughput concern.

**The exact fractional width is NOT fixed by this spec.** It is chosen in item (b) by
sweeping the width against the golden reference on all 16,211 calibration windows and
taking the smallest width that gives 100 % argmax agreement. Writing a number here that
has not been swept would be exactly the kind of unmeasured constant this project does not
ship.

---

## 6. Storage budget

| Item | Size | Placement |
|---|---|---|
| int8 weights | 27,024 B | BRAM, `$readmemh` from the exported `.mem` files |
| int32 biases | 716 B (179 x 4) | BRAM or distributed RAM |
| Requant `M0` + `shift` | 176 channels x 5 B (ESTIMATE) | distributed RAM |
| Input buffer | 2,560 B (5x512 int8) | BRAM |
| Activation ping-pong | 2 x 4,096 B | BRAM |
| `input_norm` mean/std | 10 constants | constants |

The largest intermediate activation is **4,096 int8 bytes** — conv1, conv2 and conv3
outputs are each exactly 4,096, and conv4 is 2,048. Two 4 KB buffers ping-ponged cover
the whole network. **Total on-chip storage is under 40 KB** against 4.9 Mb of BRAM on the
XC7Z020 (datasheet), so the design is not memory-bound and needs no DDR traffic for
weights.

The RTL loads **the same `.mem` files the golden reference loads**. A weight
transcription error is therefore impossible by construction, and any disagreement is a
logic bug. That property is what makes the verification in section 8 worth running.

---

## 7. Interface and timing envelope

### Interface (proposed)

- **AXI-Lite slave** — control: `start`, `done`, `idle`, the class index, three int32
  scaled logits, and a build-ID register holding a fingerprint of the loaded weights.
- **AXI-Stream slave** — one window in: 2,560 int8 beats, or 640 beats at 32 bits.
- Weights initialise from BRAM contents at bitstream load. No runtime weight reload in v1.

The build-ID register is not optional. `claims_audit.md` §1.4 records that an earlier
weight set was superseded; a bitstream that cannot say which weights it holds cannot be
matched to a claim.

### Timing — ESTIMATES, not measurements

> **MEASURED SINCE.** The built core runs **1 MAC per cycle** — `rtl_declarations.md`
> D-1 selects the smallest lane count meeting the hop with a 10x margin, and one lane
> clears it by 30x — so the comparable row below is 1 MAC/cycle, 1,691,840 cycles,
> 16.92 ms. The simulator measures **1,786,724 cycles = 17.87 ms at 100 MHz**
> (`artifacts/rtl/verify.json`), i.e. 1.056 cycles per MAC against the ideal 1.000.
> The 5.6 % is the per-output-element overhead this table says it ignores.
> Post-route the design achieves **106.6 MHz**, so timing closes at the 100 MHz target
> (`artifacts/rtl/synth/`). Resource and latency figures: `docs/results_rtl.md`.

The real-time requirement is set by the hop, not the window: **one inference per 0.5 s**.

| MACs/cycle | Cycles/window | At 100 MHz | Headroom vs 0.5 s |
|---|---|---|---|
| 8 | 211,480 | 2.11 ms | 236x |
| 32 | 52,870 | 0.53 ms | 945x |
| 64 | 26,435 | 0.26 ms | 1,891x |
| 128 | 13,218 | 0.13 ms | 3,783x |

Arithmetic only: `1,691,840 / MACs_per_cycle / 100e6`. It ignores requantisation, pipeline
fill, padding stalls and AXI transfer, all of which are real. Even so the conclusion
survives a 10x pessimism factor: **the CNN is nowhere near the bottleneck.** The XC7Z020
carries 220 DSP48E1 slices (datasheet), so even the 128-MAC row is reachable — but there
is no reason to build it. The right choice is the smallest array that closes timing,
leaving fabric for the front end.

**The unmeasured risk is the front end, not the CNN.** Per window the pipeline runs
64,000-sample FFTs, Hilbert envelopes, order resampling, a median and a `log1p` across
five channels. Whether that fits in 0.5 s on the PS (ARM Cortex-A9) **has never been
timed, here or anywhere**. Per `CLAUDE.md`, the "minutes on a 3050" style timings in the
docs do not transfer to this machine and must be re-measured; the same discipline applies
here. Until it is measured, the honest system claim is *"CNN inference is sub-millisecond;
end-to-end latency is not yet measured."*

> **STILL TRUE, and the wording has been sharpened rather than relaxed.** The front end
> has still never been timed on any hardware. §9.1 is now answered — it runs on the PS —
> which fixes *where* it runs and says nothing about *how long it takes*. The measured
> claim is now *"CNN inference is 17.87 ms in simulation and closes timing at 100 MHz
> post-route; end-to-end latency is not yet measured."* Note that 17.87 ms is not
> sub-millisecond: the arithmetic above assumed at least 8 MACs per cycle, and one lane
> is what the hop budget actually justified. `docs/results_rtl.md` §5 carries the
> comparison and `docs/rtl_declarations.md` D-1 carries the sizing rule.

---

## 8. Verification plan — declared before the RTL exists

Acceptance is **not** the 98 % bar in `methodology_v3.md` §F, and not the 0.99877
INT8-vs-float agreement in `verification_full.json`. Those compare *quantised against
float*, a modelling question already settled. This compares *RTL against the golden
reference*, an implementation question with a different right answer.

| # | Criterion | Bar |
|---|---|---|
| V-1 | RTL argmax vs `golden_reference.predict`, all 16,211 calibration windows | **100.000 %** — zero disagreements |
| V-2 | RTL scaled logits vs golden, max absolute difference | within the requant width chosen in item (b); reported, not assumed |
| V-3 | Per-layer int32 accumulators vs golden, first 64 windows | **bit-exact**, every layer, every channel |
| V-4 | Directed tie vectors exercising round-half-to-even | exact match on every tie |
| V-5 | Saturation probe: inputs forcing the int8 clamp at each layer | exact match |
| V-6 | Weight-load self-check: build-ID register vs hash of the `.mem` files | equal |

**V-1 is 100 %, not 99.x %.** The golden reference *is* the specification; an
implementation that disagrees with its own specification on any window has a bug, and
"close enough" is how a rounding-mode error ships. If V-1 cannot be met, that is reported
as a negative result the way the D2 CNN was — not accommodated by relaxing V-1.

V-3 matters more than V-1 while debugging: argmax agreement can hide a broken layer that
later layers wash out.

### Reporting rules (inherited, non-negotiable)

- Results go to a JSON; `docs/results_rtl.md` is generated from it, never hand-typed.
- Anything not run reads `NOT RUN`.
- Resource and timing figures read `ESTIMATE` until a synthesis or board report exists,
  and then they cite that report.
- Every headline number is registered in `claims_audit.md` before it reaches a slide.

---

## 9. Open questions — ALL FIVE ANSWERED

Answered after the fact, each with its answer and what the answer cost. The original
questions are kept verbatim; the answers are additions, not replacements.

### 9.1 Front-end placement — **PS SOFTWARE**

> *PS software, PL hardware, or split? This decides whether the AXI-Stream input is
> order spectra (this spec) or raw samples (a much larger spec). Blocking for any
> end-to-end latency claim; **not** blocking for the CNN core.*

**Answered: PS software.** The AXI-Stream input is order spectra per §2, and there is
no PL front end in this version.

This exposed a contradiction inside this document that had to be settled rather than
papered over. §2 says standardisation happens *inside* the accelerator; §7 says the
stream carries 2,560 int8 bytes. Both cannot be true, because 2,560 int8 bytes **is**
the standardised-and-quantised input. §7 wins, and the affine map

```
x_int8 = clamp(round_half_to_even((x - mean) / (std * s_x0)), -128, 127)
```

belongs to the PS with the rest of the DSP chain. The fabric is then integer-only from
the first convolution onward, which is what makes it bit-exact **by construction**
rather than by luck.

The cost is a narrowing of the verified boundary, declared in
`docs/rtl_declarations.md` D-4 and repeated in `docs/results_rtl.md`: the
standardisation is *not* verified in hardware, because it is not in hardware.

### 9.2 Requantisation fractional width — **F = 26**, and F = 15 would have been wrong

> *Swept in item (b) per section 5. Not guessed here.*

Swept over F = 8..32 against `golden_reference.predict` on all 16,211 cache windows
(`artifacts/rtl/requant_sweep.json`). **The sweep runs the bit-accurate Python model of
the datapath (`drivesentinel/rtl/model.py`), not the RTL** — it chooses the width the RTL
is then built at. Every statement in this subsection is about that model; whether the
RTL itself agrees is V-1 and V-3, reported in `docs/results_rtl.md`.

- **D-2's rule selects F = 15** — the smallest width with 100.000 % argmax agreement.
- **F = 15 would fail V-3's bar.** In the model at that width, 810,265 conv2 and
  2,926,089 conv3 accumulators differ from the float specification. The argmax survives; the datapath does not.
- **F = 26** is the smallest width at which every accumulator of every layer of all
  16,211 windows is bit-identical to the specification
  (`artifacts/rtl/bit_exact_confirm.json`).

F = 26 ships. Two declared criteria disagreed and the stricter one bound — which is not
a bar moving, since F = 26 also gives 100.000 % argmax agreement in the model. Agreement is also not
monotone in F (F = 10 has 2 disagreements, F = 11 has 4), which is why D-2 asks for
stability across every larger width rather than for the first width that happens to hit
100 %.

### 9.3 Clock target — **100 MHz, CLOSED at 106.6 MHz**

> *100 MHz is assumed throughout as a conventional PYNQ-Z2 figure. Not yet constrained,
> not yet closed.*

Constrained in `constraints/ds_top.xdc` and closed post-route on `xc7z020clg400-1`,
out of context: **WNS +0.615 ns, zero failing endpoints, achieved 106.6 MHz**.

The constraint was never edited. The design changed twice, each time at the path the
router reported:

| build | WNS | achieved | |
|---|---|---|---|
| combinational requantiser | −8.253 ns | 54.8 MHz | fail |
| pipelined requantiser | −0.560 ns | 94.7 MHz | fail |
| + registered MAC product | **+0.615 ns** | **106.6 MHz** | **met** |

Both intermediate implementations are kept with their reports under `artifacts/rtl/`.
The cost of closure was 4 extra cycles per output element, about 2.9 % of the
inference.

### 9.4 Board access — **SIMULATION AND SYNTHESIS ONLY**

> *Is a physical PYNQ-Z2 available for this competition, or does the track stop at
> simulation plus a synthesis report? This changes which of V-1..V-6 can actually be
> run, and the answer belongs on the slide either way.*

**Answered: no board.** The track stops at simulation plus synthesis. No bitstream has
been written and nothing has run on hardware.

All six criteria still run — V-1..V-6 are all simulation criteria, and V-6 in
particular is a hardware self-check that a simulator executes perfectly well. What
does *not* exist, and is reported as `NOT RUN` rather than estimated: power, and any
end-to-end system latency.

Place and route were run anyway, without writing a bitstream. A post-synthesis timing
number has no routing delay in it — it is an estimate of an estimate — and the point of
this step was to replace §7's arithmetic with something a reviewer can check.

### 9.5 Bitstream provenance — **CRC-32 walked over the loaded memories**

> *Which weight export a given bitstream holds, given §1.4 of the claims audit.
> Addressed by the build-ID register; the fingerprint scheme is undecided.*

**Answered** (`docs/rtl_declarations.md` D-3): CRC-32/ISO-HDLC over all int8 weights in
exported layer order, then all int32 biases little-endian in layer order — computed by
`ds_weight_crc.v` **at reset, from the memories actually loaded into the fabric**, not
baked in by the generator. The value for the current export is `0x23eb56bf`.

A generated constant would have made V-6 true by construction and proved nothing. That
is not a hypothetical: the first version of the walk registered its own memory address,
so every byte was read twice and the register came out `0x8a36ab8f`. V-6 caught it. A
constant would have agreed with itself.

`tests/test_rtl_generated.py` asserts the expected value appears in no Verilog source.

---

## 10. What this hardware may not be claimed to do

- It does not improve accuracy. macro-F1 stays 0.7852 LOBO.
- It does not monitor five stages. It monitors S5.
- It does not deliver a verdict "within a single AC cycle" — `workflow_v2.md` §12
  already flags that phrasing for removal.
- It has no measured end-to-end latency, power or resource figure, and will not have one
  until §9.4 is answered and a report exists.

> **UPDATED, and the list got shorter by exactly one item.** §9.4 is answered — no
> board — and a Vivado post-route report now exists, so there *is* a measured resource
> figure and a measured clock (`docs/results_rtl.md` §4). The other three bullets stand
> unchanged, and two of the four things in the last bullet stand with them:
>
> - **Power: `NOT RUN`.** No estimate, no figure, nothing to quote.
> - **End-to-end latency: `NOT MEASURED`.** The DSP front end runs on the PS and has
>   never been timed on any hardware. §9.1 fixed where it runs, not how long it takes.
> - **Nothing has run on hardware.** The resource and clock figures are a real report
>   of a real implementation of this design, and they are still not silicon.
