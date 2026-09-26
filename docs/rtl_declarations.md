# RTL track — recipes declared before the runs

Written **2026-09-20T06:56:03Z**, before any RTL existed, any sweep was run, any
simulation was launched and any synthesis report was produced. Committed on its own so
the commit timestamp is the evidence.

This project has one standing failure mode and it is not a bug: choosing the bar after
seeing the number. `claims_audit.md` §1.2 records the one time it happened. Every
decision below is fixed here, in advance, with the rule stated rather than the answer.

---

## D-1 — MAC array size

`rtl_spec.md` §7 sets the real-time requirement from the **hop**, not the window: one
inference per 0.5 s. §7 also states the selection criterion — *"the right choice is the
smallest array that closes timing, leaving fabric for the front end"* — because the
unmeasured risk is the DSP front end, not the CNN.

**Rule.** Build the design with a parameterised lane count `LANES`. Choose the smallest
`LANES` in {1, 2, 4, 8, 16, 32} whose **measured** cycle count per window, at the
**achieved** post-synthesis clock, leaves at least a **10x** margin against the 0.5 s
hop.

- The cycle count is measured by the testbench (a cycle counter in the top wrapper),
  not computed from the MAC total.
- The margin is evaluated at the achieved clock from the Vivado timing report, not at
  the 100 MHz target.
- If no `LANES` in the set meets 10x, report the best achieved margin as a negative
  result and do not move the 10x.

By arithmetic (§7) `LANES = 1` gives 1,691,840 MAC cycles = 16.9 ms at 100 MHz, a 29x
margin, so the rule is expected to select `LANES = 1`. Declaring the rule rather than the
answer is the point: if the measured overhead turns out to be 20x the arithmetic, the
rule picks a bigger array without anybody's judgement being involved.

## D-2 — Requantisation fractional width

`rtl_spec.md` §5 and §9.2. The width is swept, not guessed.

**Rule.** For `F` in 8..32 inclusive, build the per-output-channel multiplier as
`M0[o] = round_half_to_even(M[o] * 2^shift[o])` with `shift[o]` the unique value placing
`M0[o]` in `[2^(F-1), 2^F)`, and evaluate a bit-accurate integer model of the RTL over
**all 16,211 cache windows**. Take the **smallest `F` with 100.000 % argmax agreement
against `golden_reference.predict`**. Report the whole sweep, not just the winner.

- Ties in the sweep do not exist: agreement is monotone in `F` only in expectation, so
  if a smaller `F` agrees and a larger one does not, both are reported and the smallest
  `F` from which agreement is 100 % *and stays 100 % for every larger F tested* is taken.
- If no `F <= 32` reaches 100 %, that is a negative result against V-1 and is reported
  as one. The bar does not move to 99.9 %.

## D-3 — Build-ID / bitstream provenance (closes §9.5)

**Rule.** The build ID is a **CRC-32/ISO-HDLC** computed by the hardware itself, at
reset, by walking every weight and bias memory word actually loaded into the fabric —
not a constant baked in by the generator. Byte order is the layer order of
`scales.json`: all int8 weights of layer *l*, then the int32 biases of layer *l*
little-endian, for *l* = 0..4.

A constant would make V-6 true by construction and would prove nothing. Walking the
loaded memories means a swapped `.mem` file changes the register.

## D-4 — Verified boundary

`rtl_spec.md` §2 says input standardisation happens inside the accelerator; §7 says the
AXI-Stream input is 2,560 int8 bytes. Those two cannot both be true — 2,560 int8 bytes
*is* the quantised input, after standardisation.

**Decision.** §7 wins, because §9.1 has since been answered **PS software**: the front
end is PS-side, so the affine standardise-and-quantise
`x_int8 = clamp(round_half_to_even((x - mean) / (std * s_x0)), -128, 127)` belongs to the
PS front end with the rest of the DSP. The PL implements the golden reference from the
first convolution onward and is integer-only, hence bit-exact by construction rather than
by luck.

This is a **narrowing of the verified boundary** and is reported as one. The testbench
generates its int8 inputs with the golden reference's own `_quant` on its own
`INPUT_MEAN`/`INPUT_STD`, so the boundary the hardware sees is exactly the boundary the
reference defines — but the standardisation itself is *not* verified in hardware,
because it is not in hardware. `results_rtl.md` says so in those words.

## D-5 — What counts as a pass

Unchanged from `rtl_spec.md` §8. V-1 is **100.000 %**. If it cannot be met it is written
up as a negative result, the way the D2 CNN was, and the next item starts.

## D-6 — Simulation scope

V-1 runs all 16,211 windows through the RTL under XSim. If that does not fit the session,
the run is **split across parallel XSim processes over disjoint window ranges** and the
union reported — it is not subsampled. A subsampled V-1 is reported as `PARTIAL` with the
exact count, never as V-1.
