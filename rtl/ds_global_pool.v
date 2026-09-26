//=============================================================================
// ds_global_pool -- global average pool over the order axis, as an integer sum
//=============================================================================
// The golden reference pools the DEQUANTISED, relu'd conv4 output over length 32
// and then requantises the mean for the fc layer. This module sums the 32 relu'd
// int32 accumulators instead and leaves the 1/32 to be folded into the
// requantisation multiplier:
//
//     mean_o = (1/32) * sum_t relu(acc[o,t]) * s_W[o] * s_x
//     x_fc   = clamp(rint(mean_o / s_x_fc))
//            = clamp(rint( M[o] * sum_t relu(acc[o,t]) )),
//       with M[o] = s_W[o] * s_x / (32 * s_x_fc)
//
// rtl_spec.md section 4 sanctions this explicitly: the per-channel scale is
// constant along the pooled axis, so the rearrangement is exact and it avoids a
// divider. scripts/rtl/gen_rtl_params.py emits M[o] with the 1/32 already in it,
// so there is no divide-by-32 anywhere in the design.
//
// WIDTH. Worst case 32 x 3,149,695 = 100,790,240 -> 28 bits signed, inside the
// same int32 the rest of the datapath carries (rtl_spec.md section 4). The sum is
// of relu'd values so it is non-negative in practice; it is still carried signed,
// because a pool accumulator that has silently gone negative is a bug worth being
// able to see rather than one wrapped into a huge positive number.
//
// There is no averaging, no rounding and no truncation in this module. That is
// the point of folding the 1/32 downstream: the only rounding in the whole
// activation path happens once, in ds_requant.
//=============================================================================

module ds_global_pool #(
    parameter integer ACC_W = 32    // must hold DS_POOL_N accumulators
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clr,      // start a new output channel: load
                                             // `acc_in` instead of adding to it
    input  wire                    acc_en,   // add `acc_in` into the running sum
    input  wire signed [ACC_W-1:0] acc_in,   // one relu'd conv4 accumulator
    output reg  signed [ACC_W-1:0] sum       // sum over the positions seen so far
);

    always @(posedge clk) begin
        if (!rst_n)
            sum <= {ACC_W{1'b0}};
        else if (clr && acc_en)
            sum <= acc_in;                   // first position of a channel
        else if (clr)
            sum <= {ACC_W{1'b0}};
        else if (acc_en)
            sum <= sum + acc_in;
    end

endmodule
