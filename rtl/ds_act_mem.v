//=============================================================================
// ds_act_mem -- one activation buffer: 4,096 int8 bytes, one write and one read
//=============================================================================
// Two of these are ping-ponged across the five layers. rtl_spec.md section 6:
// the largest intermediate activation is 4,096 int8 bytes (conv1, conv2 and conv3
// each produce exactly that), the 5 x 512 input is 2,560, and conv4's output
// never reaches a buffer at all because the pool consumes it -- so two 4 KB
// buffers cover the whole network and nothing spills to DDR.
//
// LAYOUT. addr = channel * L + position, channel-major. Chosen because the
// reduction loop's inner index is the kernel tap k, and for a fixed input channel
// the K taps of a kernel sit at K CONSECUTIVE addresses. The alternative
// (position-major) would stride by L on every single cycle of the innermost loop.
//
// PORTS. Simple dual port: the engine reads the source buffer every cycle and
// writes the destination buffer occasionally, and the two are never the same
// buffer, so no read/write collision is possible and the write-first versus
// read-first question does not arise. Both ports are registered, which is what
// lets Vivado infer a block RAM rather than distributed RAM.
//
// There is no initialisation. A buffer holds whatever the previous layer wrote,
// and the padding at the ends of the order axis is supplied by ds_conv1d forcing
// the MAC lane to zero (ds_mac_array.lane_en) rather than by reading a zeroed
// location -- so out-of-range positions are never addressed at all.
//=============================================================================

module ds_act_mem #(
    parameter integer DEPTH = 4096,
    parameter integer AW    = 12,
    parameter integer DW    = 8
) (
    input  wire          clk,
    // write port
    input  wire          we,
    input  wire [AW-1:0] wa,
    input  wire [DW-1:0] wd,
    // read port, one cycle of latency
    input  wire [AW-1:0] ra,
    output reg  [DW-1:0] rq
);

    reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we)
            mem[wa] <= wd;
        rq <= mem[ra];
    end

endmodule
