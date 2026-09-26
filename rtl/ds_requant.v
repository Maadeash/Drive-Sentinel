//=============================================================================
// ds_requant -- fixed-point requantisation with round-half-to-even, pipelined
//=============================================================================
// Computes  q = rint(M0 * acc / 2**shift)  with ties rounded to EVEN, in three
// clocked stages.
//
// WHY THIS MODULE IS THE RISKY ONE
// --------------------------------
// rtl_spec.md section 4, property 3: numpy's rint -- and therefore the golden
// reference, and therefore the specification -- rounds ties to even.
//     rint(0.5) = 0, rint(1.5) = 2, rint(2.5) = 2
// The usual hardware shortcut floor(x + 0.5) gives 1, 2, 3. It disagrees only on
// exact ties, so it passes a smoke test, passes most of a regression, and then
// fails on a handful of windows in a pattern that looks exactly like a weight
// loading bug. This module is where that is got right, once.
//
// HOW THE THREE CASES ARE EXACT
// -----------------------------
// For signed `p` and shift `s > 0`:
//     q   = p >>> s                    arithmetic shift = floor(p / 2**s)
//     rem = p & (2**s - 1)             two's-complement remainder, in [0, 2**s)
//     half= 2**(s-1)
// then
//     rem >  half  ->  q + 1
//     rem <  half  ->  q
//     rem == half  ->  q + (q & 1)     q is the floor, so this lands on even
// Negative values need no special case: the shift already floors them, and the
// remainder of a two's-complement number is still its distance above the floor.
// Worked examples, against numpy:
//     p = -3, s = 1 -> q = -2, rem = 1 = half, q&1 = 0 -> -2   rint(-1.5) = -2
//     p = -1, s = 1 -> q = -1, rem = 1 = half, q&1 = 1 -> 0    rint(-0.5) = -0
//     p =  5, s = 1 -> q =  2, rem = 1 = half, q&1 = 0 ->  2   rint( 2.5) =  2
// The identical three lines are in drivesentinel/rtl/model.py:round_half_to_even,
// and tb/tb_requant.sv drives both against each other over 3,966 directed
// vectors including 427 exact ties (V-4).
//
// WHY IT IS PIPELINED, MEASURED RATHER THAN GUESSED
// -------------------------------------------------
// The first version of this module was combinational. Post-route it was the
// critical path of the whole design by a wide margin -- 30 logic levels,
// 18.248 ns against a 10 ns target, WNS -8.253 ns, reported in
// artifacts/rtl/synth_v1_combinational/. The path was
//     FSM state -> 26x32 DSP multiply -> 58-bit variable shift -> remainder
//     compare -> round -> clamp -> activation write
// which is a multiplier and a barrel shifter and two wide carry chains between
// one pair of flops. Splitting it three ways costs three cycles per output
// element, 12,355 elements per window, about 2 % of the inference; the
// alternative was a core that runs at 55 MHz.
//
//   stage 1   p1 = acc * m0                      (the DSP)
//   stage 2   floor, remainder, the two compares (the barrel shifter)
//   stage 3   the conditional increment, and the product passed through
//
// The multiplier M0 is UNSIGNED and per output channel (rtl_spec.md section 4,
// property 2: s_W is a vector, not a scalar). `shift` is per output channel too,
// because the multipliers span nearly seven octaves and one shared fixed point
// would throw away most of them.
//=============================================================================

module ds_requant #(
    parameter integer ACC_W   = 32,   // accumulator width in, signed
    parameter integer FRAC_W  = 26,   // M0 width, unsigned, normalised to the top
    parameter integer SHIFT_W = 6,    // width of the shift field
    parameter integer PROD_W  = 58    // ACC_W + FRAC_W
) (
    input  wire                      clk,

    input  wire                      in_valid, // start one requantisation
    input  wire signed [ACC_W-1:0]   acc,      // int32 accumulator, already relu'd
                                               // for conv layers (ds_relu_clamp)
    input  wire        [FRAC_W-1:0]  m0,       // mantissa, in [2**(F-1), 2**F)
    input  wire        [SHIFT_W-1:0] shift,    // right shift, >= 1 by construction

    output reg                       out_valid,// 3 cycles after in_valid
    output reg  signed [PROD_W-1:0]  prod,     // M0 * acc, untruncated: the fc
                                               // path takes argmax on this
    output reg  signed [ACC_W-1:0]   q         // rint(prod / 2**shift), ties even
);

    // NO RESET. These three stages are pure dataflow: they hold whatever was
    // last pushed through them and nothing reads an output unless `out_valid`
    // says so. ds_conv1d cannot reach its S_RQW wait state within three cycles
    // of reset, so no uninitialised value can reach a result -- and leaving the
    // reset off keeps it out of the datapath's critical path, which is the whole
    // point of this module being pipelined at all.
    //
    // ---- stage 1: the multiply --------------------------------------------
    wire signed [FRAC_W:0] m0_s = {1'b0, m0};   // unsigned, zero-extended by one

    reg                      v1;
    reg signed [PROD_W-1:0]  p1;
    reg        [SHIFT_W-1:0] s1;

    always @(posedge clk) begin
        v1 <= in_valid;
        p1 <= acc * m0_s;
        s1 <= shift;
    end

    // ---- stage 2: floor, remainder, and the two comparisons ---------------
    // Built as masks rather than with a variable part-select, because the shift
    // amount is a run-time value.
    wire [PROD_W-1:0] mask2 = ({{(PROD_W-1){1'b0}}, 1'b1} << s1) - 1'b1;
    wire [PROD_W-1:0] half2 = {{(PROD_W-1){1'b0}}, 1'b1} << (s1 - 1'b1);
    wire [PROD_W-1:0] rem2  = p1 & mask2;

    reg                     v2;
    reg signed [PROD_W-1:0] floor2;
    reg signed [PROD_W-1:0] p2;
    reg                     above2, tie2;

    always @(posedge clk) begin
        v2     <= v1;
        p2     <= p1;
        floor2 <= p1 >>> s1;
        above2 <= (rem2 > half2);
        tie2   <= (rem2 == half2);
    end

    // ---- stage 3: the conditional increment -------------------------------
    // floor2[0] is the parity of the floor, so adding it on a tie lands on even.
    wire signed [PROD_W-1:0] rounded =
        floor2 + (above2 ? {{(PROD_W-1){1'b0}}, 1'b1}
                         : (tie2 ? {{(PROD_W-1){1'b0}}, floor2[0]}
                                 : {PROD_W{1'b0}}));

    always @(posedge clk) begin
        out_valid <= v2;
        prod      <= p2;
        // Truncation to ACC_W cannot lose information: the requantised value is
        // by construction within the int8 range plus the small excursion the
        // clamp exists to catch. Checked in simulation below rather than assumed.
        q         <= rounded[ACC_W-1:0];
    end

    // synthesis translate_off
    // `shift` is >= 1 for every generated channel (every multiplier is far below
    // 1.0, so the mantissa normalisation always shifts right), and the
    // requantised value stays within a few hundred of zero. Both are properties
    // of the export, not of the architecture, so they are checked in simulation
    // rather than assumed in silence.
    always @(posedge clk) begin
        if (in_valid && shift == 0)
            $fatal(1, "ds_requant: shift = 0, which would underflow `half`");
        if (v2 && (rounded > $signed({{(PROD_W-ACC_W+1){1'b0}}, {(ACC_W-1){1'b1}}})
                || rounded < -$signed({{(PROD_W-ACC_W+1){1'b0}}, {(ACC_W-1){1'b1}}})))
            $fatal(1, "ds_requant: requantised value %0d does not fit ACC_W",
                   rounded);
    end
    // synthesis translate_on

endmodule
