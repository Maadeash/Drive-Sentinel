//=============================================================================
// ds_weight_mem -- the int8 weight ROM, loaded from the EXPORTED .mem files
//=============================================================================
// One unified byte-addressed memory holding all DS_W_DEPTH int8 weights, filled
// by five $readmemh calls at the generated per-layer offsets (ds_gen.vh). The
// flat index of a weight is
//
//     addr = base[layer] + (o * I + i) * K + k
//
// which is the row-major (O, I, K) order the exporter wrote, so no reordering
// happens anywhere between the file and the multiplier.
//
// WHY IT LOADS THE EXPORT DIRECTLY
// --------------------------------
// rtl_spec.md section 6: "The RTL loads the same .mem files the golden reference
// loads. A weight transcription error is therefore impossible by construction,
// and any disagreement is a logic bug." That property is the reason the
// verification in section 8 is worth running at all, and it is why this module
// does not pre-process, re-pack, byte-swap or bank the files. It reads them as
// the exporter wrote them.
//
// WHAT LANES > 1 WOULD COST
// -------------------------
// A single memory has one read port and therefore delivers one weight per cycle,
// which is what pins the shipped core at LANES = 1 (docs/rtl_declarations.md
// D-1 -- and one lane already meets the 0.5 s hop budget with a wide margin).
// Going wider means banking this memory, and banking it means either generating
// per-bank .mem files -- giving up the property above -- or copying a flat
// $readmemh image into banks in an initial block, whose synthesis behaviour is
// tool-dependent. Neither is free, and neither buys anything the hop budget
// needs. It is written down here so the next person does not rediscover it.
//=============================================================================

`include "ds_defs.vh"

module ds_weight_mem #(
    parameter integer DEPTH = 27024,
    parameter integer AW    = 15      // $clog2(DEPTH) rounded up
) (
    input  wire            clk,
    input  wire [AW-1:0]   addr,      // flat weight index
    output reg  [7:0]      q          // weight byte, one cycle after `addr`
);

    reg [7:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(`DS_W0_FILE, mem, `DS_W0_BASE);
        $readmemh(`DS_W1_FILE, mem, `DS_W1_BASE);
        $readmemh(`DS_W2_FILE, mem, `DS_W2_BASE);
        $readmemh(`DS_W3_FILE, mem, `DS_W3_BASE);
        $readmemh(`DS_W4_FILE, mem, `DS_W4_BASE);
    end

    always @(posedge clk)
        q <= mem[addr];

endmodule
