//=============================================================================
// ds_core -- the accelerator proper: memories, engine, sequencer, build ID
//=============================================================================
// Everything except the bus adapters. ds_top adds AXI-Lite and AXI-Stream around
// this; the split exists so the testbench can drive the core directly when it is
// the arithmetic under test and through the buses when the buses are.
//
// CONTENTS
//   ds_weight_mem   27,024 int8 weights, from the exported .mem files
//   ds_bias_mem        179 int32 biases, likewise
//   ds_param_mem       179 {shift, M0} requantisation parameters, GENERATED
//   ds_act_mem  x2   two 4 KB ping-pong activation buffers
//   ds_conv1d        the single datapath every layer runs on
//   ds_fc            the scaled-logit argmax
//   ds_ctrl_fsm      the layer sequencer
//   ds_weight_crc    the build-ID walk
//
// TOTAL ON-CHIP STORAGE  27,024 + 716 + 179*4 + 2*4,096 = 36,656 bytes, against
// the 4.9 Mb of block RAM on an XC7Z020 -- rtl_spec.md section 6's "under 40 KB"
// budget, now instantiated rather than estimated. Nothing streams from DDR.
//
// TWO ARBITRATIONS, BOTH TRIVIAL
//   * The weight and bias memories are shared between ds_conv1d and
//     ds_weight_crc. The CRC walk runs once, immediately after reset, and the
//     core refuses to start until it has finished, so the mux below is never
//     contended and needs no arbiter.
//   * Activation buffer A is written by ds_axis_in (the input frame) and by
//     ds_conv1d (layers 1 and 3). Those never overlap either: the frame lands
//     before the start pulse. The mux is one bit wide and is documented rather
//     than defended.
//=============================================================================

`include "ds_defs.vh"

module ds_core #(
    parameter integer LANES = 1,
    parameter integer DEBUG_TRACE = 1   // see ds_conv1d
) (
    input  wire               clk,
    input  wire               rst_n,

    // ---- control -----------------------------------------------------------
    input  wire               start,        // one-cycle pulse; ignored unless idle
                                            // and the build-ID walk has finished
    output wire               busy,
    output wire               done,         // one-cycle pulse per inference
    output wire [31:0]        cycles,       // measured, start pulse to done pulse

    // ---- input frame, from ds_axis_in --------------------------------------
    input  wire               in_we,
    input  wire [`DS_ACT_AW-1:0] in_wa,
    input  wire [7:0]         in_wd,

    // ---- result ------------------------------------------------------------
    output wire [1:0]         class_idx,
    output wire signed [31:0] logit0,
    output wire signed [31:0] logit1,
    output wire signed [31:0] logit2,

    // ---- provenance --------------------------------------------------------
    output wire [31:0]        build_id,
    output wire               crc_done,

    // ---- V-3 trace ---------------------------------------------------------
    output wire               dbg_acc_valid,
    output wire [2:0]         dbg_layer,
    output wire [`DS_CH_W-1:0]  dbg_ch,
    output wire [`DS_CFG_W-1:0] dbg_pos,
    output wire signed [`DS_ACC_W-1:0] dbg_acc
);

    localparam integer ACT_W  = `DS_ACT_W;
    localparam integer WGT_W  = `DS_WGT_W;
    localparam integer ACC_W  = `DS_ACC_W;
    localparam integer FRAC_W = `DS_FRAC_W;
    localparam integer SHIFT_W= `DS_SHIFT_W;
    localparam integer PROD_W = `DS_PROD_W;
    localparam integer ACT_AW = `DS_ACT_AW;
    localparam integer WGT_AW = `DS_WGT_AW;
    localparam integer PAR_AW = `DS_PAR_AW;
    localparam integer CFG_W  = `DS_CFG_W;

    // ---- descriptor wires --------------------------------------------------
    wire [7:0]        cfg_o_ch, cfg_in_ch;
    wire [3:0]        cfg_k, cfg_pad;
    wire [CFG_W-1:0]  cfg_l_in, cfg_l_out, cfg_dst_l, cfg_terms;
    wire [1:0]        cfg_stride;
    wire [WGT_AW-1:0] cfg_wbase;
    wire [PAR_AW-1:0] cfg_pbase;
    wire              cfg_is_pool, cfg_is_last;
    wire [2:0]        cfg_layer;
    wire              layer_go, layer_done;
    wire              src_sel, dst_sel;

    // ---- memory ports ------------------------------------------------------
    wire [WGT_AW-1:0] conv_wgt_addr, crc_wgt_addr, wgt_addr;
    wire [7:0]        wgt_q;
    wire [PAR_AW-1:0] conv_bias_addr, crc_bias_addr, bias_addr, par_addr;
    wire signed [31:0] bias_q;
    wire [31:0]       par_q;
    wire              crc_busy;

    assign wgt_addr  = crc_busy ? crc_wgt_addr  : conv_wgt_addr;
    assign bias_addr = crc_busy ? crc_bias_addr : conv_bias_addr;

    ds_weight_mem #(.DEPTH (`DS_W_DEPTH), .AW (WGT_AW)) u_wmem (
        .clk (clk), .addr (wgt_addr), .q (wgt_q));

    ds_bias_mem #(.DEPTH (`DS_B_DEPTH), .AW (PAR_AW)) u_bmem (
        .clk (clk), .addr (bias_addr), .q (bias_q));

    ds_param_mem #(.DEPTH (`DS_B_DEPTH), .AW (PAR_AW), .WORD_W (32)) u_pmem (
        .clk (clk), .addr (par_addr), .q (par_q));

    // ---- activation ping-pong ---------------------------------------------
    wire              conv_we;
    wire [ACT_AW-1:0] conv_wa, conv_ra;
    wire [ACT_W-1:0]  conv_wd;
    wire [ACT_W-1:0]  q_a, q_b;

    // Buffer A takes writes from the stream (the input frame) and from the engine
    // on the layers whose destination it is. See the header: they never overlap.
    wire a_we = in_we | (conv_we & ~dst_sel);
    wire [ACT_AW-1:0] a_wa = in_we ? in_wa : conv_wa;
    wire [ACT_W-1:0]  a_wd = in_we ? in_wd : conv_wd;

    ds_act_mem #(.DEPTH (`DS_ACT_DEPTH), .AW (ACT_AW), .DW (ACT_W)) u_act_a (
        .clk (clk), .we (a_we), .wa (a_wa), .wd (a_wd),
        .ra (conv_ra), .rq (q_a));

    ds_act_mem #(.DEPTH (`DS_ACT_DEPTH), .AW (ACT_AW), .DW (ACT_W)) u_act_b (
        .clk (clk), .we (conv_we & dst_sel), .wa (conv_wa), .wd (conv_wd),
        .ra (conv_ra), .rq (q_b));

    wire [ACT_W-1:0] conv_rq = src_sel ? q_b : q_a;

    // ---- the engine --------------------------------------------------------
    wire              fc_valid;
    wire [1:0]        fc_idx;
    wire signed [PROD_W-1:0] fc_prod;

    ds_conv1d #(
        .LANES (LANES), .ACT_W (ACT_W), .WGT_W (WGT_W), .ACC_W (ACC_W),
        .FRAC_W (FRAC_W), .SHIFT_W (SHIFT_W), .PROD_W (PROD_W),
        .ACT_AW (ACT_AW), .WGT_AW (WGT_AW), .PAR_AW (PAR_AW), .CFG_W (CFG_W),
        .DEBUG_TRACE (DEBUG_TRACE)
    ) u_engine (
        .clk (clk), .rst_n (rst_n),
        .cfg_o_ch (cfg_o_ch), .cfg_in_ch (cfg_in_ch), .cfg_k (cfg_k),
        .cfg_l_in (cfg_l_in), .cfg_l_out (cfg_l_out), .cfg_dst_l (cfg_dst_l),
        .cfg_stride (cfg_stride), .cfg_pad (cfg_pad), .cfg_terms (cfg_terms),
        .cfg_wbase (cfg_wbase), .cfg_pbase (cfg_pbase),
        .cfg_is_pool (cfg_is_pool), .cfg_is_last (cfg_is_last),
        .cfg_layer (cfg_layer),
        .go (layer_go), .busy (), .done (layer_done),
        .act_ra (conv_ra), .act_rq (conv_rq),
        .act_we (conv_we), .act_wa (conv_wa), .act_wd (conv_wd),
        .wgt_addr (conv_wgt_addr), .wgt_q (wgt_q),
        .bias_addr (conv_bias_addr), .bias_q (bias_q),
        .par_addr (par_addr), .par_q (par_q),
        .fc_valid (fc_valid), .fc_idx (fc_idx), .fc_prod (fc_prod),
        .dbg_acc_valid (dbg_acc_valid), .dbg_layer (dbg_layer),
        .dbg_ch (dbg_ch), .dbg_pos (dbg_pos), .dbg_acc (dbg_acc)
    );

    // ---- the classifier tail ----------------------------------------------
    wire core_start = start & crc_done & ~busy;

    ds_fc #(.N_CLASS (3), .PROD_W (PROD_W), .LOGIT_SH (`DS_LOGIT_SHIFT)) u_fc (
        .clk (clk), .rst_n (rst_n),
        .clr (core_start), .valid (fc_valid), .idx (fc_idx), .prod (fc_prod),
        .class_idx (class_idx),
        .logit0 (logit0), .logit1 (logit1), .logit2 (logit2)
    );

    // ---- the sequencer -----------------------------------------------------
    ds_ctrl_fsm #(.WGT_AW (WGT_AW), .PAR_AW (PAR_AW), .CFG_W (CFG_W)) u_ctrl (
        .clk (clk), .rst_n (rst_n),
        .start (core_start), .busy (busy), .done (done), .cycles (cycles),
        .cfg_o_ch (cfg_o_ch), .cfg_in_ch (cfg_in_ch), .cfg_k (cfg_k),
        .cfg_l_in (cfg_l_in), .cfg_l_out (cfg_l_out), .cfg_dst_l (cfg_dst_l),
        .cfg_stride (cfg_stride), .cfg_pad (cfg_pad), .cfg_terms (cfg_terms),
        .cfg_wbase (cfg_wbase), .cfg_pbase (cfg_pbase),
        .cfg_is_pool (cfg_is_pool), .cfg_is_last (cfg_is_last),
        .cfg_layer (cfg_layer),
        .layer_go (layer_go), .layer_done (layer_done),
        .src_sel (src_sel), .dst_sel (dst_sel)
    );

    // ---- provenance --------------------------------------------------------
    ds_weight_crc #(
        .W_DEPTH (`DS_W_DEPTH), .B_DEPTH (`DS_B_DEPTH),
        .WGT_AW (WGT_AW), .PAR_AW (PAR_AW)
    ) u_crc (
        .clk (clk), .rst_n (rst_n),
        .busy (crc_busy), .wgt_addr (crc_wgt_addr), .wgt_q (wgt_q),
        .bias_addr (crc_bias_addr), .bias_q (bias_q),
        .done (crc_done), .build_id (build_id)
    );

endmodule
