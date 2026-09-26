//=============================================================================
// ds_top -- AXI-Lite control, AXI-Stream input, and the core between them
//=============================================================================
// The synthesis top level. rtl_spec.md section 7's interface, built:
//
//   AXI4-Lite slave   start / status / class / three scaled logits / BUILD_ID /
//                     the measured cycle count / a magic word / the build config
//   AXI4-Stream slave one window: 640 beats of 32 bits = 2,560 int8 bytes
//   weights           initialised from BRAM contents at bitstream load; there is
//                     no runtime weight reload in v1
//
// AUTO-START. A completed input frame starts an inference by itself, so the
// steady-state host loop is "push 640 beats, poll STATUS.done, read CLASS" with
// no per-window control write. Writing CTRL.start re-runs whatever window is
// already in the buffer, which is what the directed tie and saturation tests
// (V-4, V-5) use to run many vectors without re-streaming identical data.
//
// SCOPE, SO THE DEMO CANNOT OVERSTATE IT. This accelerates stage S5, the bearing
// CNN, and nothing else (rtl_spec.md section 1). The winding, supply and inverter
// branches are scalar-feature models in software. Putting this model in fabric
// does not change its accuracy in either direction: it stays macro-F1 0.7852
// leave-one-bearing-out. The hardware claim is latency and resource cost.
//=============================================================================

`include "ds_defs.vh"

module ds_top #(
    parameter integer LANES = 1,       // MAC lanes; see docs/rtl_declarations.md D-1
    parameter integer DEBUG_TRACE = 1  // V-3 accumulator trace; 0 in the
                                       // synthesised build. See ds_conv1d.
) (
    input  wire        clk,         // single clock domain, 100 MHz target
    input  wire        rst_n,       // active-low synchronous reset

    // ---- AXI4-Lite slave, 256-byte aperture --------------------------------
    input  wire [7:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [7:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,

    // ---- AXI4-Stream slave: the order-spectrum window ----------------------
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // ---- sideband, for a board LED or an interrupt -------------------------
    output wire        irq,          // one-cycle pulse when an inference ends
    output wire [1:0]  class_idx_o,  // the same value as the CLASS register

    // ---- V-3 trace: driven only when DEBUG_TRACE, held at zero otherwise ---
    output wire        dbg_acc_valid,
    output wire [2:0]  dbg_layer,
    output wire [`DS_CH_W-1:0]  dbg_ch,
    output wire [`DS_CFG_W-1:0] dbg_pos,
    output wire signed [`DS_ACC_W-1:0] dbg_acc
);

    wire        axil_start, soft_rst;
    wire        core_busy, core_done;
    wire [31:0] cycles, build_id;
    wire        crc_done;
    wire [1:0]  class_idx;
    wire signed [31:0] logit0, logit1, logit2;

    wire        in_we, frame_done, frame_valid, frame_error;
    wire [`DS_ACT_AW-1:0] in_wa;
    wire [7:0]  in_wd;

    wire core_rst_n = rst_n & ~soft_rst;

    ds_axis_in #(.N_BYTES (`DS_IN_BYTES), .ACT_AW (`DS_ACT_AW)) u_axis (
        .clk (clk), .rst_n (core_rst_n),
        .s_axis_tdata (s_axis_tdata), .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready), .s_axis_tlast (s_axis_tlast),
        .act_we (in_we), .act_wa (in_wa), .act_wd (in_wd),
        .frame_done (frame_done), .frame_valid (frame_valid),
        .frame_error (frame_error)
    );

    // A completed frame starts an inference; so does CTRL.start. The core itself
    // gates both on being idle and on the build-ID walk having finished.
    wire start = frame_done | axil_start;

    ds_core #(.LANES (LANES), .DEBUG_TRACE (DEBUG_TRACE)) u_core (
        .clk (clk), .rst_n (core_rst_n),
        .start (start), .busy (core_busy), .done (core_done), .cycles (cycles),
        .in_we (in_we), .in_wa (in_wa), .in_wd (in_wd),
        .class_idx (class_idx),
        .logit0 (logit0), .logit1 (logit1), .logit2 (logit2),
        .build_id (build_id), .crc_done (crc_done),
        .dbg_acc_valid (dbg_acc_valid), .dbg_layer (dbg_layer),
        .dbg_ch (dbg_ch), .dbg_pos (dbg_pos), .dbg_acc (dbg_acc)
    );

    // CONFIG reports what was actually built, so a host reading a bitstream it
    // did not compile can tell which requantisation width is inside it -- the
    // same provenance argument as BUILD_ID, one level up.
    // Assembled through 8-bit wires rather than by part-selecting the macros:
    // a part-select on an unsized constant is not portable across tools.
    wire [7:0] cfg_frac_w     = `DS_FRAC_W;
    wire [7:0] cfg_lanes      = LANES;
    wire [7:0] cfg_logit_shft = `DS_LOGIT_SHIFT;
    wire [31:0] config_word = {8'd0, cfg_logit_shft, cfg_lanes, cfg_frac_w};

    ds_axil_ctrl u_axil (
        .clk (clk), .rst_n (rst_n),
        .s_axi_awaddr (s_axi_awaddr), .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready), .s_axi_wdata (s_axi_wdata),
        .s_axi_wstrb (s_axi_wstrb), .s_axi_wvalid (s_axi_wvalid),
        .s_axi_wready (s_axi_wready), .s_axi_bresp (s_axi_bresp),
        .s_axi_bvalid (s_axi_bvalid), .s_axi_bready (s_axi_bready),
        .s_axi_araddr (s_axi_araddr), .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready), .s_axi_rdata (s_axi_rdata),
        .s_axi_rresp (s_axi_rresp), .s_axi_rvalid (s_axi_rvalid),
        .s_axi_rready (s_axi_rready),
        .start (axil_start), .soft_rst (soft_rst),
        .busy (core_busy), .core_done (core_done),
        .frame_valid (frame_valid & ~frame_error), .crc_done (crc_done),
        .class_idx (class_idx),
        .logit0 (logit0), .logit1 (logit1), .logit2 (logit2),
        .build_id (build_id), .cycles (cycles), .config_word (config_word)
    );

    assign irq         = core_done;
    assign class_idx_o = class_idx;

endmodule
