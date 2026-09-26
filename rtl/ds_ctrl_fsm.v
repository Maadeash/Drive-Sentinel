//=============================================================================
// ds_ctrl_fsm -- runs the five layers in order and owns the ping-pong
//=============================================================================
// Holds the layer table, hands ds_conv1d one descriptor at a time, and selects
// which activation buffer is the source and which is the destination. The table
// itself lives in ds_gen.vh and is GENERATED from artifacts/int8_export/
// scales.json: shapes, strides, paddings and the weight and bias base offsets are
// properties of the exported model, not of the architecture, so re-exporting the
// model must change them and must not require editing this file.
//
// THE PING-PONG
// -------------
//   layer 0 (conv1)  buffer 0 -> buffer 1      buffer 0 holds the 5 x 512 input
//   layer 1 (conv2)  buffer 1 -> buffer 0
//   layer 2 (conv3)  buffer 0 -> buffer 1
//   layer 3 (conv4)  buffer 1 -> buffer 0      writes the 64 pooled values
//   layer 4 (fc)     buffer 0 -> (no write)    reads those 64 values
// so the source is buffer (layer & 1) and the destination is the other one. The
// fc layer lands on buffer 0 by the same rule, which is why the pool writes there.
//
// WHY THERE IS A GAP STATE
// ------------------------
// ds_conv1d registers its write port, so the final byte of a layer is written one
// cycle AFTER the layer reports done. Swapping the buffers on the done pulse would
// send that byte to the wrong buffer -- and the symptom would be a single wrong
// value at the end of every layer, which looks like an off-by-one in the address
// generator and is not. C_GAP holds everything still for three cycles instead.
//
// The cycle counter runs from the start pulse to the last layer's done and is
// exposed on AXI-Lite. It is the MEASURED latency; docs/results_rtl.md quotes it
// rather than the arithmetic estimate in rtl_spec.md section 7, and
// docs/rtl_declarations.md D-1 selects the MAC array size from it.
//=============================================================================

`include "ds_defs.vh"

module ds_ctrl_fsm #(
    parameter integer WGT_AW = 15,
    parameter integer PAR_AW = 8,
    parameter integer CFG_W  = `DS_CFG_W   // shared with ds_conv1d
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              start,       // one-cycle pulse
    output reg               busy,
    output reg               done,        // one-cycle pulse when layer 4 finishes
    output reg  [31:0]       cycles,      // clocks from `start` to that pulse

    // ---- descriptor out to ds_conv1d ---------------------------------------
    output reg  [7:0]        cfg_o_ch,
    output reg  [7:0]        cfg_in_ch,
    output reg  [3:0]        cfg_k,
    output reg  [CFG_W-1:0]  cfg_l_in,
    output reg  [CFG_W-1:0]  cfg_l_out,
    output reg  [CFG_W-1:0]  cfg_dst_l,
    output reg  [1:0]        cfg_stride,
    output reg  [3:0]        cfg_pad,
    output reg  [CFG_W-1:0]  cfg_terms,
    output reg  [WGT_AW-1:0] cfg_wbase,
    output reg  [PAR_AW-1:0] cfg_pbase,
    output reg               cfg_is_pool,
    output reg               cfg_is_last,
    output reg  [2:0]        cfg_layer,

    // ---- handshake with ds_conv1d ------------------------------------------
    output reg               layer_go,
    input  wire              layer_done,

    // ---- buffer selection --------------------------------------------------
    output wire              src_sel,     // 0 = buffer A, 1 = buffer B
    output wire              dst_sel
);

    localparam C_IDLE  = 3'd0;
    localparam C_SETUP = 3'd1;   // descriptor presented, one cycle to settle
    localparam C_GO    = 3'd2;
    localparam C_WAIT  = 3'd3;
    localparam C_GAP   = 3'd4;   // let the last registered write land
    localparam C_DONE  = 3'd5;

    reg [2:0] state;
    reg [2:0] layer;
    reg [1:0] gap;

    assign src_sel = layer[0];
    assign dst_sel = ~layer[0];

    // ---- the generated layer table -----------------------------------------
    always @* begin
        case (layer)
        3'd0: begin
            cfg_o_ch = `DS_L0_O;  cfg_in_ch = `DS_L0_I;  cfg_k     = `DS_L0_K;
            cfg_l_in = `DS_L0_LIN; cfg_l_out = `DS_L0_LOUT;
            cfg_dst_l = `DS_L0_DSTL; cfg_stride = `DS_L0_STRIDE;
            cfg_pad  = `DS_L0_PAD; cfg_terms = `DS_L0_TERMS;
            cfg_wbase = `DS_W0_BASE; cfg_pbase = `DS_B0_BASE;
            cfg_is_pool = `DS_L0_POOL; cfg_is_last = `DS_L0_LAST;
        end
        3'd1: begin
            cfg_o_ch = `DS_L1_O;  cfg_in_ch = `DS_L1_I;  cfg_k     = `DS_L1_K;
            cfg_l_in = `DS_L1_LIN; cfg_l_out = `DS_L1_LOUT;
            cfg_dst_l = `DS_L1_DSTL; cfg_stride = `DS_L1_STRIDE;
            cfg_pad  = `DS_L1_PAD; cfg_terms = `DS_L1_TERMS;
            cfg_wbase = `DS_W1_BASE; cfg_pbase = `DS_B1_BASE;
            cfg_is_pool = `DS_L1_POOL; cfg_is_last = `DS_L1_LAST;
        end
        3'd2: begin
            cfg_o_ch = `DS_L2_O;  cfg_in_ch = `DS_L2_I;  cfg_k     = `DS_L2_K;
            cfg_l_in = `DS_L2_LIN; cfg_l_out = `DS_L2_LOUT;
            cfg_dst_l = `DS_L2_DSTL; cfg_stride = `DS_L2_STRIDE;
            cfg_pad  = `DS_L2_PAD; cfg_terms = `DS_L2_TERMS;
            cfg_wbase = `DS_W2_BASE; cfg_pbase = `DS_B2_BASE;
            cfg_is_pool = `DS_L2_POOL; cfg_is_last = `DS_L2_LAST;
        end
        3'd3: begin
            cfg_o_ch = `DS_L3_O;  cfg_in_ch = `DS_L3_I;  cfg_k     = `DS_L3_K;
            cfg_l_in = `DS_L3_LIN; cfg_l_out = `DS_L3_LOUT;
            cfg_dst_l = `DS_L3_DSTL; cfg_stride = `DS_L3_STRIDE;
            cfg_pad  = `DS_L3_PAD; cfg_terms = `DS_L3_TERMS;
            cfg_wbase = `DS_W3_BASE; cfg_pbase = `DS_B3_BASE;
            cfg_is_pool = `DS_L3_POOL; cfg_is_last = `DS_L3_LAST;
        end
        default: begin
            cfg_o_ch = `DS_L4_O;  cfg_in_ch = `DS_L4_I;  cfg_k     = `DS_L4_K;
            cfg_l_in = `DS_L4_LIN; cfg_l_out = `DS_L4_LOUT;
            cfg_dst_l = `DS_L4_DSTL; cfg_stride = `DS_L4_STRIDE;
            cfg_pad  = `DS_L4_PAD; cfg_terms = `DS_L4_TERMS;
            cfg_wbase = `DS_W4_BASE; cfg_pbase = `DS_B4_BASE;
            cfg_is_pool = `DS_L4_POOL; cfg_is_last = `DS_L4_LAST;
        end
        endcase
        cfg_layer = layer;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= C_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            layer_go <= 1'b0;
            layer    <= 3'd0;
            cycles   <= 32'd0;
        end else begin
            done     <= 1'b0;
            layer_go <= 1'b0;
            if (busy)
                cycles <= cycles + 32'd1;

            case (state)
            C_IDLE:
                if (start) begin
                    busy   <= 1'b1;
                    layer  <= 3'd0;
                    cycles <= 32'd0;
                    state  <= C_SETUP;
                end

            C_SETUP: state <= C_GO;

            C_GO: begin
                layer_go <= 1'b1;
                state    <= C_WAIT;
            end

            C_WAIT:
                if (layer_done) begin
                    gap   <= 2'd3;
                    state <= C_GAP;
                end

            C_GAP:
                if (gap == 2'd0) begin
                    if (layer == `DS_N_LAYERS - 1) begin
                        state <= C_DONE;
                    end else begin
                        layer <= layer + 3'd1;
                        state <= C_SETUP;
                    end
                end else begin
                    gap <= gap - 2'd1;
                end

            C_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= C_IDLE;
            end

            default: state <= C_IDLE;
            endcase
        end
    end

endmodule
