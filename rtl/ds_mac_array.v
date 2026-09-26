//=============================================================================
// ds_mac_array -- LANES signed 8x8 multipliers and the adder tree over them
//=============================================================================
// The only arithmetic in the MAC path. Deliberately trivial, and deliberately
// separate from the sequencer that feeds it, because the two have different
// failure modes: this module can only be wrong about signedness or width, and
// ds_conv1d can only be wrong about addresses.
//
// NO SATURATION. rtl_spec.md section 4 bounds the widest reduction at 192 terms
// of 127 x 128 = 3,121,152, which needs 23 bits, so an int32 accumulator cannot
// overflow on any input whatsoever -- not just on the calibration set. Adding a
// saturating adder here would be dead logic that quietly changes the arithmetic
// if the bound were ever wrong, which is the opposite of what you want: you want
// it to break loudly.
//
// LANES is a parameter and the module is written for any value, but the shipped
// core instantiates it with LANES = 1. The reason is in docs/rtl_declarations.md
// D-1: the hop budget is 0.5 s and one lane already meets it with a wide margin,
// and the fabric is better spent on the DSP front end, which is the part of the
// system that has never been timed. What LANES > 1 additionally needs is a
// weight memory that can deliver LANES bytes per cycle; see ds_weight_mem.v.
//=============================================================================

module ds_mac_array #(
    parameter integer LANES = 1,    // multipliers instantiated in parallel
    parameter integer A_W   = 8,    // activation width, signed
    parameter integer W_W   = 8,    // weight width, signed
    parameter integer SUM_W = 32    // output width, signed; must hold LANES terms
) (
    // Activations, packed little-lane-first: lane j occupies bits [j*A_W +: A_W].
    input  wire [LANES*A_W-1:0]  act,
    // Weights, same packing.
    input  wire [LANES*W_W-1:0]  wgt,
    // Per-lane enable. A disabled lane contributes exactly zero. Used for zero
    // padding at the ends of the order axis and, when LANES does not divide the
    // kernel length, for the ragged last group.
    input  wire [LANES-1:0]      lane_en,
    // Sum of the enabled lanes' products, signed.
    output wire signed [SUM_W-1:0] sum
);

    // Per-lane products, sign-extended to the output width so the reduction below
    // is a plain signed add and the tree shape is left to the synthesiser.
    wire signed [SUM_W-1:0] term [0:LANES-1];

    genvar j;
    generate
        for (j = 0; j < LANES; j = j + 1) begin : g_lane
            wire signed [A_W-1:0] a = act[j*A_W +: A_W];
            wire signed [W_W-1:0] w = wgt[j*W_W +: W_W];
            wire signed [A_W+W_W-1:0] p = a * w;
            assign term[j] = lane_en[j] ? {{(SUM_W-A_W-W_W){p[A_W+W_W-1]}}, p}
                                        : {SUM_W{1'b0}};
        end
    endgenerate

    integer n;
    reg signed [SUM_W-1:0] acc;
    always @* begin
        acc = {SUM_W{1'b0}};
        for (n = 0; n < LANES; n = n + 1)
            acc = acc + term[n];
    end
    assign sum = acc;

endmodule
