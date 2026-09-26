//=============================================================================
// ds_pynq_wrap -- the verified core, unchanged, with the interface metadata a
//                 Vivado block design needs
//=============================================================================
// This file adds NOTHING to the datapath. It instantiates rtl/ds_top.v exactly
// as simulated and synthesised, with DEBUG_TRACE = 0 (the configuration V-1 runs
// and the out-of-context build reports), and adds three things only a block
// design needs:
//
//   1. X_INTERFACE attributes, so Vivado recognises s_axi as AXI4-Lite and
//      s_axis as AXI4-Stream, and knows both are clocked by `clk` and reset by
//      `rst_n`. Without them apply_bd_automation cannot connect the core, and
//      the attributes cannot go in rtl/ because rtl/ is the verified artefact and
//      is not edited for packaging.
//   2. AWPROT / ARPROT inputs, which the AXI interconnect drives and the core
//      ignores. AXI4-Lite requires them on the interface; the core never
//      needed them, because it grants every access the same way.
//   3. Address truncation: the interconnect presents 32-bit addresses and the
//      core decodes 8. The block design assigns this core a 64 KB window, so
//      the 256-byte register map repeats inside it -- harmless, and noted in
//      board/README.md so nobody reads it as a decode bug.
//
// Deliberately NOT connected: the V-3 trace outputs (DEBUG_TRACE = 0 holds them
// at zero), `class_idx_o` (the CLASS register carries the same value), and
// `irq`. The interrupt is a one-cycle pulse, and a pulse into a level-sensitive
// GIC input can be missed; the notebook polls STATUS instead, which is slower by
// microseconds against a 17.9 ms inference.
//=============================================================================

module ds_pynq_wrap (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis, ASSOCIATED_RESET rst_n, FREQ_HZ 100000000" *)
    input  wire        clk,            // fabric clock, FCLK_CLK0 = 100 MHz
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        rst_n,          // active-low, synchronous to clk

    // ---- AXI4-Lite slave: control, status, result, provenance -------------
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 32" *)
    input  wire [31:0] s_axi_awaddr,   // byte address; low 8 bits decoded
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWPROT" *)
    input  wire [2:0]  s_axi_awprot,   // required by AXI4-Lite, ignored
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output wire        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output wire        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output wire        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [31:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARPROT" *)
    input  wire [2:0]  s_axi_arprot,   // required by AXI4-Lite, ignored
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output wire        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output wire        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire        s_axi_rready,

    // ---- AXI4-Stream slave: one window of 640 x 32-bit beats, from the DMA --
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)
    (* X_INTERFACE_PARAMETER = "TDATA_NUM_BYTES 4, HAS_TLAST 1, HAS_TKEEP 0, HAS_TSTRB 0" *)
    input  wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
    input  wire        s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
    output wire        s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST" *)
    input  wire        s_axis_tlast
);

    ds_top #(
        .LANES       (1),   // docs/rtl_declarations.md D-1
        .DEBUG_TRACE (0)    // the shipped configuration
    ) u_core (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axi_awaddr  (s_axi_awaddr[7:0]),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_axi_wdata),
        .s_axi_wstrb   (s_axi_wstrb),
        .s_axi_wvalid  (s_axi_wvalid),
        .s_axi_wready  (s_axi_wready),
        .s_axi_bresp   (s_axi_bresp),
        .s_axi_bvalid  (s_axi_bvalid),
        .s_axi_bready  (s_axi_bready),
        .s_axi_araddr  (s_axi_araddr[7:0]),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata   (s_axi_rdata),
        .s_axi_rresp   (s_axi_rresp),
        .s_axi_rvalid  (s_axi_rvalid),
        .s_axi_rready  (s_axi_rready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .irq           (),
        .class_idx_o   (),
        .dbg_acc_valid (),
        .dbg_layer     (),
        .dbg_ch        (),
        .dbg_pos       (),
        .dbg_acc       ()
    );

endmodule
