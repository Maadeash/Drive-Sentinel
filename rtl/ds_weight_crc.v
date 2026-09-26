//=============================================================================
// ds_weight_crc -- the build ID, computed from the memories that were loaded
//=============================================================================
// Walks every weight byte and every bias word actually present in the fabric,
// after reset, and reduces them to a CRC-32. The result is the BUILD_ID register
// on AXI-Lite (rtl_spec.md section 7).
//
// WHY IT WALKS THE MEMORIES INSTEAD OF BEING A GENERATED CONSTANT
// ---------------------------------------------------------------
// docs/rtl_declarations.md D-3. A constant emitted by the same script that
// produced the memory images would make V-6 -- "build-ID register vs hash of the
// .mem files" -- true by construction, and would therefore prove nothing at all.
// Reading back what was loaded means a swapped, truncated or missing .mem file
// changes the register, which is the whole point: claims_audit.md section 1.4
// records that an earlier weight set was superseded, and a bitstream that cannot
// say which weights it holds cannot be matched to a claim.
//
// ALGORITHM. CRC-32/ISO-HDLC, the one zlib and Python's binascii.crc32 compute:
// reflected, polynomial 0xEDB88320 in its reflected form, init 0xFFFFFFFF, final
// XOR 0xFFFFFFFF. Bit-serial, eight steps unrolled combinationally per byte, so
// one byte per cycle and no lookup table.
//
// BYTE ORDER. All DS_W_DEPTH int8 weights in exported layer order, then all
// DS_B_DEPTH int32 biases little-endian, also in layer order -- the same order
// scripts/rtl/gen_rtl_params.py uses when it computes the expected value for the
// testbench to compare against. The order is arbitrary but it is written down in
// exactly two places and a test checks they agree.
//
// COST. DS_W_DEPTH + 5 * DS_B_DEPTH cycles once per reset -- about 28,000 cycles,
// 0.28 ms at 100 MHz. The core refuses to start until `done` is high, so this can
// never overlap an inference and the address muxes it needs are never contended.
//=============================================================================

`include "ds_defs.vh"

module ds_weight_crc #(
    parameter integer W_DEPTH = 27024,
    parameter integer B_DEPTH = 179,
    parameter integer WGT_AW  = 15,
    parameter integer PAR_AW  = 8
) (
    input  wire              clk,
    input  wire              rst_n,

    // Read ports, muxed onto the parameter memories while `busy`. Both addresses
    // are COMBINATIONAL from the index counters, not registered: the memories
    // register their own output, so an address that is itself registered arrives
    // a cycle late and every byte gets read twice. That bug shipped in the first
    // version of this module and V-6 is what caught it -- which is the argument
    // for D-3 in one sentence, because a generated constant would have agreed
    // with itself and said nothing.
    output reg               busy,
    output wire [WGT_AW-1:0] wgt_addr,
    input  wire [7:0]        wgt_q,
    output wire [PAR_AW-1:0] bias_addr,
    input  wire [31:0]       bias_q,

    output reg               done,       // sticky once the walk has finished
    output reg  [31:0]       build_id
);

    // Sized copies: part-selecting an integer parameter is not portable.
    localparam [WGT_AW:0] W_DEPTH_C = W_DEPTH[WGT_AW:0];
    localparam [PAR_AW:0] B_DEPTH_C = B_DEPTH[PAR_AW:0];

    localparam S_IDLE = 2'd0;
    localparam S_W    = 2'd1;   // weights, one byte per cycle
    localparam S_B    = 2'd2;   // biases, one word every five cycles
    localparam S_END  = 2'd3;

    reg [1:0]        state;
    reg [WGT_AW:0]   w_idx;      // address presented THIS cycle; wgt_q holds w_idx-1
    reg [PAR_AW:0]   b_idx;
    reg [2:0]        b_sub;     // 0 presents the address, 1..4 feed the bytes
    reg [31:0]       crc;
    reg              feed;      // a byte is on `byte_in` this cycle
    reg [7:0]        byte_in;

    // --- one byte of CRC-32, eight reflected steps, combinational ----------
    function [31:0] crc32_byte;
        input [31:0] c_in;
        input [7:0]  d;
        reg   [31:0] c;
        integer      n;
        begin
            c = c_in ^ {24'd0, d};
            for (n = 0; n < 8; n = n + 1)
                c = c[0] ? ((c >> 1) ^ 32'hEDB88320) : (c >> 1);
            crc32_byte = c;
        end
    endfunction

    assign wgt_addr  = w_idx[WGT_AW-1:0];
    assign bias_addr = b_idx[PAR_AW-1:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b1;         // the walk starts immediately out of reset
            done      <= 1'b0;
            crc       <= 32'hFFFFFFFF;
            w_idx     <= {(WGT_AW+1){1'b0}};
            b_idx     <= {(PAR_AW+1){1'b0}};
            b_sub     <= 3'd0;
            feed      <= 1'b0;
            build_id  <= 32'd0;
        end else begin
            feed <= 1'b0;
            if (feed)
                crc <= crc32_byte(crc, byte_in);

            case (state)
            // One priming cycle: address 0 is already presented, so its byte
            // lands on wgt_q at the end of this cycle.
            S_IDLE: begin
                w_idx <= {{WGT_AW{1'b0}}, 1'b1};
                state <= S_W;
            end

            S_W: begin
                // wgt_q holds mem[w_idx - 1]; wgt_addr already presents w_idx
                byte_in <= wgt_q;
                feed    <= 1'b1;
                if (w_idx == W_DEPTH_C) begin
                    b_idx <= {(PAR_AW+1){1'b0}};
                    b_sub <= 3'd0;
                    state <= S_B;
                end else begin
                    w_idx <= w_idx + 1'b1;
                end
            end

            S_B: begin
                case (b_sub)
                3'd0: b_sub <= 3'd1;                  // wait for bias_q
                3'd1: begin byte_in <= bias_q[7:0];   feed <= 1'b1; b_sub <= 3'd2; end
                3'd2: begin byte_in <= bias_q[15:8];  feed <= 1'b1; b_sub <= 3'd3; end
                3'd3: begin byte_in <= bias_q[23:16]; feed <= 1'b1; b_sub <= 3'd4; end
                3'd4: begin
                    byte_in <= bias_q[31:24];
                    feed    <= 1'b1;
                    b_sub   <= 3'd0;
                    if (b_idx == B_DEPTH_C - 1'b1) begin
                        state <= S_END;
                    end else begin
                        b_idx <= b_idx + 1'b1;
                    end
                end
                default: b_sub <= 3'd0;
                endcase
            end

            S_END: begin
                // one cycle for the last `feed` to land in `crc`
                if (!feed) begin
                    build_id <= crc ^ 32'hFFFFFFFF;
                    busy     <= 1'b0;
                    done     <= 1'b1;
                end
            end

            default: state <= S_END;
            endcase
        end
    end

endmodule
