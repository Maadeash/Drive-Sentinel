//=============================================================================
// ds_axil_ctrl -- AXI4-Lite slave: start, status, result, provenance
//=============================================================================
// A minimal, strictly non-blocking AXI4-Lite slave. One address decode, no burst,
// no protection checking, and every read returns in a single cycle, because the
// registers behind it are all flops. The map is in ds_defs.vh.
//
//   0x00 CTRL     W   bit0 start (self-clearing), bit1 soft reset
//   0x04 STATUS   R   done / busy / idle / input-frame-valid / crc-done
//   0x08 CLASS    R   argmax index, 0..2
//   0x0C..0x14    R   the three scaled logits, int32
//   0x18 BUILD_ID R   CRC-32 of the weights and biases actually loaded
//   0x1C CYCLES   R   clock cycles of the last inference, measured
//   0x20 MAGIC    R   "DS01"
//   0x24 CONFIG   R   {LOGIT_SHIFT, LANES, FRAC_W}
//
// WHY BUILD_ID IS NOT OPTIONAL
// ----------------------------
// rtl_spec.md section 7: claims_audit.md section 1.4 records that an earlier
// weight set was superseded, so a bitstream that cannot say which weights it
// holds cannot be matched to a claim. docs/rtl_declarations.md D-3 goes further
// and requires the value to be COMPUTED BY THE HARDWARE from the memories it
// actually loaded, not baked in by the generator -- a constant would make V-6
// true by construction and prove nothing.
//
// WHY CYCLES IS A REGISTER
// ------------------------
// Because every latency figure in this project has to come from a run.
// rtl_spec.md section 7's table is arithmetic and labelled ESTIMATE; this
// register is the measurement that replaces it, and D-1 sizes the MAC array from
// it rather than from the arithmetic.
//=============================================================================

`include "ds_defs.vh"

module ds_axil_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // ---- AXI4-Lite slave ---------------------------------------------------
    input  wire [7:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [7:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,

    // ---- to and from the core ----------------------------------------------
    output reg         start,        // one-cycle pulse
    output reg         soft_rst,     // one-cycle pulse
    input  wire        busy,
    input  wire        core_done,    // one-cycle pulse: latch DONE
    input  wire        frame_valid,
    input  wire        crc_done,
    input  wire [1:0]  class_idx,
    input  wire signed [31:0] logit0,
    input  wire signed [31:0] logit1,
    input  wire signed [31:0] logit2,
    input  wire [31:0] build_id,
    input  wire [31:0] cycles,
    input  wire [31:0] config_word
);

    // ---- write channel -----------------------------------------------------
    reg        aw_hold, w_hold;
    reg [7:0]  aw_addr;
    reg [31:0] w_data;

    assign s_axi_awready = !aw_hold;
    assign s_axi_wready  = !w_hold;

    wire do_write = aw_hold && w_hold;

    // ---- DONE is sticky until the next start -------------------------------
    reg done_flag;

    always @(posedge clk) begin
        if (!rst_n) begin
            aw_hold      <= 1'b0;
            w_hold       <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;
            start        <= 1'b0;
            soft_rst     <= 1'b0;
            done_flag    <= 1'b0;
        end else begin
            start    <= 1'b0;
            soft_rst <= 1'b0;

            if (core_done)
                done_flag <= 1'b1;

            if (s_axi_awvalid && !aw_hold) begin
                aw_hold <= 1'b1;
                aw_addr <= s_axi_awaddr;
            end
            if (s_axi_wvalid && !w_hold) begin
                w_hold <= 1'b1;
                w_data <= s_axi_wdata;
            end

            if (do_write && !s_axi_bvalid) begin
                if (aw_addr == `DS_REG_CTRL) begin
                    if (w_data[0]) begin
                        start     <= 1'b1;
                        done_flag <= 1'b0;   // cleared by arming, not by reading
                    end
                    if (w_data[1])
                        soft_rst <= 1'b1;
                end
                s_axi_bresp  <= 2'b00;       // OKAY for every address: the map is
                s_axi_bvalid <= 1'b1;        // read-mostly and a write elsewhere
                aw_hold      <= 1'b0;        // is a host bug, not a bus error
                w_hold       <= 1'b0;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // ---- read channel ------------------------------------------------------
    reg ar_hold;
    reg [7:0] ar_addr;
    assign s_axi_arready = !ar_hold;

    // Built bit by bit from the ds_defs.vh positions rather than as one packed
    // concatenation with a padding literal in front of it: a concatenation puts
    // the bits in the reverse of the order the map lists them, and its padding
    // width has to be recounted by hand every time a status bit is added.
    wire [31:0] status;
    assign status[`DS_ST_DONE]     = done_flag;
    assign status[`DS_ST_BUSY]     = busy;
    assign status[`DS_ST_IDLE]     = ~busy;
    assign status[`DS_ST_IN_VALID] = frame_valid;
    assign status[`DS_ST_CRC_DONE] = crc_done;
    assign status[31:`DS_ST_CRC_DONE + 1] = {(31 - `DS_ST_CRC_DONE){1'b0}};

    always @(posedge clk) begin
        if (!rst_n) begin
            ar_hold      <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_rresp  <= 2'b00;
            s_axi_rdata  <= 32'd0;
        end else begin
            if (s_axi_arvalid && !ar_hold) begin
                ar_hold <= 1'b1;
                ar_addr <= s_axi_araddr;
            end
            if (ar_hold && !s_axi_rvalid) begin
                case (ar_addr)
                `DS_REG_CTRL:    s_axi_rdata <= 32'd0;
                `DS_REG_STATUS:  s_axi_rdata <= status;
                `DS_REG_CLASS:   s_axi_rdata <= {30'd0, class_idx};
                `DS_REG_LOGIT0:  s_axi_rdata <= logit0;
                `DS_REG_LOGIT1:  s_axi_rdata <= logit1;
                `DS_REG_LOGIT2:  s_axi_rdata <= logit2;
                `DS_REG_BUILDID: s_axi_rdata <= build_id;
                `DS_REG_CYCLES:  s_axi_rdata <= cycles;
                `DS_REG_MAGIC:   s_axi_rdata <= `DS_MAGIC;
                `DS_REG_CONFIG:  s_axi_rdata <= config_word;
                default:         s_axi_rdata <= 32'hDEADBEEF;
                endcase
                s_axi_rresp  <= 2'b00;
                s_axi_rvalid <= 1'b1;
                ar_hold      <= 1'b0;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
