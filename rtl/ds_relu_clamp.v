//=============================================================================
// ds_relu_clamp -- the two non-linearities that bracket requantisation
//=============================================================================
// One module, two combinational functions, because they are the two halves of a
// single boundary in the golden reference and separating them would invite
// someone to apply one and forget the other:
//
//     y_float = s_W[o] * s_x * acc          (dequantise)
//     y_relu  = max(y_float, 0)             <-- RELU, before requantisation
//     x_next  = clamp(rint(y_relu / s_x'), -128, 127)   <-- CLAMP, after
//
// WHY RELU CAN BE DONE ON THE INTEGER
// -----------------------------------
// rtl_spec.md section 4, property 4. The reference relu's the DEQUANTISED value,
// but every scale in this export is strictly positive, so
//     max(acc * s, 0) == max(acc, 0) * s   for s > 0
// and clamping the int32 accumulator at zero is exactly equivalent. That is a
// property of symmetric quantisation with no zero point; it would NOT hold if a
// zero-point offset were ever added to this export.
//
// WHY THE CLAMP IS THE ONLY SATURATING OPERATION
// ----------------------------------------------
// The MAC path cannot overflow (ds_mac_array header). The activation scales were
// fitted at the 99.9th percentile, so roughly one value in a thousand genuinely
// exceeds the int8 range and the clamp is doing real work, not defending against
// an impossible case.
//
// Note that after relu the lower clamp at -128 is unreachable; it is kept because
// the fc layer is not relu'd, and because an unreachable branch that matches the
// specification is cheaper than a comment explaining why it was removed.
//=============================================================================

module ds_relu_clamp #(
    parameter integer ACC_W = 32,   // accumulator / requantised value width
    parameter integer OUT_W = 8     // quantised activation width (int8)
) (
    // --- relu half: applied to the raw accumulator, before requantisation ---
    input  wire signed [ACC_W-1:0] acc_in,     // int32 accumulator, any sign
    input  wire                    relu_en,    // 1 on every layer except fc
    output wire signed [ACC_W-1:0] acc_relu,   // relu_en ? max(acc_in,0) : acc_in

    // --- clamp half: applied to the requantised value, after ---
    input  wire signed [ACC_W-1:0] q_in,       // ds_requant output
    output wire signed [OUT_W-1:0] q_clamp     // clamp(q_in, -2**(OUT_W-1),
                                               //               2**(OUT_W-1)-1)
);

    assign acc_relu = (relu_en && acc_in[ACC_W-1]) ? {ACC_W{1'b0}} : acc_in;

    localparam signed [ACC_W-1:0] Q_MAX =  (1 <<< (OUT_W-1)) - 1;   //  127
    localparam signed [ACC_W-1:0] Q_MIN = -(1 <<< (OUT_W-1));       // -128

    assign q_clamp = (q_in > Q_MAX) ? Q_MAX[OUT_W-1:0] :
                     (q_in < Q_MIN) ? Q_MIN[OUT_W-1:0] :
                                      q_in[OUT_W-1:0];

endmodule
