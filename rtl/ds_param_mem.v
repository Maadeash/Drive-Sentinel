//=============================================================================
// ds_param_mem -- per-output-channel requantisation multiplier and shift
//=============================================================================
// 179 entries, indexed exactly like ds_bias_mem: one per output channel of the
// five layers, in layer order at the same offsets. Each word is
//
//     { shift[DS_SHIFT_W-1:0], m0[DS_FRAC_W-1:0] }
//
// so that M[o] = m0[o] * 2**-shift[o] approximates the exact float multiplier
//
//     conv1..conv3 : s_W[o] * s_x / s_x_next
//     conv4        : s_W[o] * s_x / (32 * s_x_next)     -- pool folded in
//     fc           : s_W[c] * s_x                       -- output scale
//
// UNLIKE THE WEIGHTS, THIS IMAGE IS GENERATED, NOT EXPORTED. It is derived from
// scales.json by scripts/rtl/gen_rtl_params.py at the fractional width the sweep
// chose (docs/rtl_declarations.md D-2, artifacts/rtl/requant_sweep.json).
// tests/test_rtl_generated.py regenerates it and fails if the committed image
// differs, so it cannot drift away from the export it was derived from -- the
// same guard tests/test_rtl_spec.py applies to the specification document.
//
// The fc entries carry the one shared shift described in ds_fc.v; all three hold
// the same value, and the generator asserts it.
//=============================================================================

`include "ds_defs.vh"

module ds_param_mem #(
    parameter integer DEPTH = 179,
    parameter integer AW    = 8,
    parameter integer WORD_W = 32     // DS_FRAC_W + DS_SHIFT_W, rounded up
) (
    input  wire              clk,
    input  wire [AW-1:0]     addr,    // flat output-channel index across layers
    output reg [WORD_W-1:0]  q        // {shift, m0}, one cycle after `addr`
);

    reg [WORD_W-1:0] mem [0:DEPTH-1];

    initial $readmemh(`DS_PARAM_FILE, mem);

    always @(posedge clk)
        q <= mem[addr];

endmodule
