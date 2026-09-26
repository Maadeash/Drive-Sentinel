//=============================================================================
// ds_bias_mem -- the int32 bias ROM, loaded from the EXPORTED .mem files
//=============================================================================
// 179 biases (16 + 32 + 64 + 64 + 3), one per output channel of the five layers,
// in layer order at the generated offsets. Same principle as ds_weight_mem: the
// exported files are read unchanged.
//
// The bias is ADDED INTO THE ACCUMULATOR, not into the requantised result:
//     acc_int32 = sum(W_int8 * x_int8) + b_int32
// ds_conv1d preloads the accumulator with the bias at the start of each output
// position, so the adder that accumulates products is the adder that applies the
// bias and there is no separate add.
//=============================================================================

`include "ds_defs.vh"

module ds_bias_mem #(
    parameter integer DEPTH = 179,
    parameter integer AW    = 8
) (
    input  wire              clk,
    input  wire [AW-1:0]     addr,   // flat output-channel index across layers
    output reg signed [31:0] q       // bias, one cycle after `addr`
);

    reg [31:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(`DS_B0_FILE, mem, `DS_B0_BASE);
        $readmemh(`DS_B1_FILE, mem, `DS_B1_BASE);
        $readmemh(`DS_B2_FILE, mem, `DS_B2_BASE);
        $readmemh(`DS_B3_FILE, mem, `DS_B3_BASE);
        $readmemh(`DS_B4_FILE, mem, `DS_B4_BASE);
    end

    always @(posedge clk)
        q <= mem[addr];

endmodule
