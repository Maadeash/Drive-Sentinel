//=============================================================================
// ds_fc -- the classifier tail: scale the three accumulators, then argmax
//=============================================================================
// The fc REDUCTION is not here. Layer 4 is a 64-term inner product, which is the
// same reduction the convolution engine already performs (I = 64, K = 1,
// L_in = L_out = 1), so it runs on the same multiplier; building a second
// datapath for 192 of the model's 1,691,840 MACs would be waste. What IS here is
// the part of the fc layer that differs from a convolution: it is not requantised
// to int8, and its output feeds an argmax instead of the next layer.
//
// WHY THE ARGMAX MUST BE TAKEN ON THE SCALED VALUE
// ------------------------------------------------
// rtl_spec.md section 5. The fc layer's per-class output scale s_W[c] * s_x spans
// 8.2081e-5 to 9.4877e-5 -- a factor of 1.16 between the smallest and the largest.
// Taking argmax on the raw int32 accumulators compares three numbers that are in
// three different units. It would be WRONG, and it would be wrong quietly: the
// three classes are close enough that it agrees most of the time.
//
// ONE SHIFT FOR ALL THREE CLASSES
// -------------------------------
// The conv layers use a per-output-channel shift, because their multipliers span
// seven octaves. Here the three multipliers are within one octave of each other,
// so ds_fc uses a SINGLE shift (DS_FC_SHIFT, generated) and compares the raw
// products M0[c] * acc[c] directly. Per-class shifts would put the three
// comparands back into three different units -- the exact mistake described
// above, reintroduced one level down.
//
// TIE BREAKING
// ------------
// numpy's argmax returns the FIRST maximum. The comparison below is strictly
// greater-than and the classes arrive in index order, so a tie keeps the earlier
// class, matching the reference.
//
// THE EXPOSED LOGITS
// ------------------
// AXI-Lite carries int32, and the raw product is DS_LOGIT_W bits. The register
// holds prod >>> DS_LOGIT_SHIFT, so its LSB is worth 2**(DS_LOGIT_SHIFT -
// DS_FC_SHIFT); ds_gen.vh records that exponent as DS_LOGIT_LSB_EXP and
// docs/results_rtl.md reports V-2's max absolute difference in the reference's
// own float units, using it. The shift is arithmetic truncation, not
// round-half-to-even: these bits are a reporting convenience, and the argmax --
// the thing the design actually decides -- is taken on the untruncated product.
//=============================================================================

`include "ds_defs.vh"

// The three logit registers are separate ports rather than an array because
// they are three separate AXI-Lite addresses (ds_defs.vh) and Verilog-2001 has
// no array ports. N_CLASS is therefore effectively fixed at 3 here; the
// parameter exists so the class-index width tracks it and so a fourth class
// would fail loudly at elaboration rather than silently drop a logit.
module ds_fc #(
    parameter integer N_CLASS   = 3,
    parameter integer IDX_W     = 2,    // $clog2(N_CLASS), rounded up
    parameter integer PROD_W    = 58,   // width of M0[c] * acc[c]
    parameter integer LOGIT_SH  = 16    // right shift into the int32 register
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clr,        // start a new inference
    input  wire                     valid,      // `prod` is class `idx`'s product
    input  wire [IDX_W-1:0]         idx,        // class index, 0..N_CLASS-1
    input  wire signed [PROD_W-1:0] prod,       // M0[idx] * acc[idx], untruncated
    output reg  [IDX_W-1:0]         class_idx,  // argmax over the classes seen
    output reg  signed [31:0]       logit0,     // prod >>> LOGIT_SH, per class
    output reg  signed [31:0]       logit1,
    output reg  signed [31:0]       logit2
);

    reg signed [PROD_W-1:0] best;
    reg                     have;

    // synthesis translate_off
    initial if (N_CLASS != 3)
        $fatal(1, "ds_fc has three AXI-Lite logit registers; N_CLASS = %0d",
               N_CLASS);
    // synthesis translate_on

    wire signed [PROD_W-1:0] shifted = prod >>> LOGIT_SH;

    always @(posedge clk) begin
        if (!rst_n || clr) begin
            best      <= {PROD_W{1'b0}};
            have      <= 1'b0;
            class_idx <= {IDX_W{1'b0}};
            logit0    <= 32'sd0;
            logit1    <= 32'sd0;
            logit2    <= 32'sd0;
        end else if (valid) begin
            if (!have || prod > best) begin     // strictly >: keeps the first max
                best      <= prod;
                class_idx <= idx;
            end
            have <= 1'b1;
            case (idx)
                {IDX_W{1'b0}}:              logit0 <= shifted[31:0];
                {{(IDX_W-1){1'b0}}, 1'b1}:  logit1 <= shifted[31:0];
                {{(IDX_W-2){1'b0}}, 2'b10}: logit2 <= shifted[31:0];
                default: ;
            endcase
        end
    end

endmodule
