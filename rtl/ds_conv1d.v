//=============================================================================
// ds_conv1d -- one layer of the network, start to finish
//=============================================================================
// Runs a complete layer from a descriptor supplied by ds_ctrl_fsm: address
// generation, the reduction, the bias, relu, requantisation, the global average
// pool, and the write back into the ping-pong buffer. Every layer of the model
// goes through this module, including the fc layer, which is a 64-term inner
// product -- the same reduction with I = 64, K = 1, L_in = L_out = 1. Building a
// second datapath for 192 of the model's 1,691,840 MACs would be waste; what the
// fc layer genuinely does differently (no requantisation, argmax on the scaled
// value) lives in ds_fc.v.
//
// LOOP ORDER  for o in 0..O-1:  for pos in 0..L_out-1:  for i, k
// -------------------------------------------------------------
// Output channel outermost. Three things fall out of that and none of them are
// accidental:
//   * the bias and the requantisation multiplier are read ONCE per output
//     channel, not once per output element;
//   * the global average pool needs exactly ONE accumulator, because all 32
//     positions of a channel are visited consecutively (a position-outer loop
//     would need 64);
//   * the weights for one output channel are I*K CONSECUTIVE addresses in the
//     exported row-major (O, I, K) image, so the weight pointer is an increment,
//     never a multiply.
//
// ADDRESSING WITHOUT MULTIPLIERS
// ------------------------------
// All four indices are maintained incrementally:
//   w_oc_base += I*K   per output channel      (weights are (o,i,k) row-major)
//   w_ptr     += 1     per reduction term      (k is innermost)
//   a_row     += L_in  per input channel       (activations are channel-major)
//   t_base    += stride per output position    (t_base = pos*stride - pad)
// and the activation address is a_row + (t_base + k). The only comparison that
// needs the raw position is the zero-padding test below.
//
// ZERO PADDING IS A LANE MASK, NOT A MEMORY READ
// ----------------------------------------------
// When t = t_base + k falls outside [0, L_in) the term is outside the spectrum.
// Rather than reading a zeroed location, the MAC lane is disabled, so the product
// is exactly zero and the out-of-range address is never presented. rtl_spec.md
// section 3 notes that padding with the float zero and with the integer zero
// coincide here only because the quantisation is symmetric with no zero point;
// that is a property of this export and the RTL is allowed to rely on it.
//
// PIPELINE. Three stages, and they are why DRAIN and DRAIN2 exist:
//   cycle n   : counters drive act_ra / wgt_addr combinationally; the padding
//               test is computed and registered
//   cycle n+1 : the memories present their data and ds_mac_array multiplies;
//               the product is registered
//   cycle n+2 : the accumulator adds the product
// so two cycles after the last address is issued the accumulator is still short.
// S_DRAIN and S_DRAIN2 are those cycles. Getting this off by one is the classic
// way to lose the last tap of every kernel, which shows up as a
// plausible-looking but wrong result -- exactly what V-3 is for.
//
// The product is registered rather than added straight into the accumulator
// because with both in one cycle the path block RAM -> multiply -> 32-bit add
// was the design's critical path at 10.505 ns against a 10 ns target.
//
// CYCLES PER LAYER
//   pooling layer : O * (2 + L_out * (I*K + 4) + 4)
//   every other   : O * (2 + L_out * (I*K + 7))
// The per-position overhead is POS_INIT, DRAIN, DRAIN2, OUT and the
// requantiser's three pipeline stages; the pooling layer pays the last three
// once per channel
// instead of once per position, because only the pooled sum is requantised. The
// MEASURED total is in artifacts/rtl/verify.json -- this comment is arithmetic
// and the simulator is the authority.
//=============================================================================

`include "ds_defs.vh"

module ds_conv1d #(
    parameter integer LANES   = 1,
    parameter integer ACT_W   = 8,
    parameter integer WGT_W   = 8,
    parameter integer ACC_W   = 32,
    parameter integer FRAC_W  = 25,
    parameter integer SHIFT_W = 7,
    parameter integer PROD_W  = 57,
    parameter integer ACT_AW  = 12,
    parameter integer WGT_AW  = 15,
    parameter integer PAR_AW  = 8,
    // Width of the length/count fields in the layer descriptor. 10 bits
    // holds the 512-bin input axis and the 192-term widest reduction.
    parameter integer CFG_W   = `DS_CFG_W,
    // Channel index width. Every layer's O and I are at most 64, so 7 bits
    // holds 0..127 with room; derived from CFG_W would be wasteful, and a bare
    // 7 scattered through the counters would be the kind of magic number this
    // datapath is meant not to have.
    parameter integer CH_W    = `DS_CH_W,
    // Kernel tap index width. K is at most 9.
    parameter integer K_W     = `DS_K_W,
    // Signed position along the order axis, INCLUDING the padding excursion:
    // t ranges over [-pad, (L_out-1)*stride - pad + K-1], i.e. [-4, 514] for
    // conv1. One bit wider than CFG_W, plus the sign.
    parameter integer T_W     = CFG_W + 2,
    // The V-3 accumulator trace. 1 in simulation, 0 in the synthesised build:
    // the trace ports are real ports at the top level, so out-of-context
    // synthesis will NOT trim them on its own, and the deployment configuration
    // should not carry 50-odd flops of debug. docs/results_rtl.md reports the
    // utilisation both ways so the cost is visible rather than assumed away.
    parameter integer DEBUG_TRACE = 1
) (
    input  wire                   clk,
    input  wire                   rst_n,

    // ---- layer descriptor, held stable by ds_ctrl_fsm while `busy` ----------
    input  wire [7:0]             cfg_o_ch,    // O, output channels
    input  wire [7:0]             cfg_in_ch,   // I, input channels
    input  wire [3:0]             cfg_k,       // K, kernel taps
    input  wire [CFG_W-1:0]       cfg_l_in,    // input length along the order axis
    input  wire [CFG_W-1:0]       cfg_l_out,   // output length
    input  wire [CFG_W-1:0]       cfg_dst_l,   // stride between output channels in
                                               // the destination buffer: L_out
                                               // normally, 1 when pooling
    input  wire [1:0]             cfg_stride,  // convolution stride
    input  wire [3:0]             cfg_pad,     // zero padding, each end
    input  wire [CFG_W-1:0]       cfg_terms,   // I*K, precomputed by the FSM so
                                               // this module needs no multiplier
    input  wire [WGT_AW-1:0]      cfg_wbase,   // first weight of the layer
    input  wire [PAR_AW-1:0]      cfg_pbase,   // first bias / requant parameter
    input  wire                   cfg_is_pool, // conv4: pool over positions
    input  wire                   cfg_is_last, // fc: scale and hand to ds_fc
    input  wire [2:0]             cfg_layer,   // layer index, for the V-3 trace

    // ---- handshake ---------------------------------------------------------
    input  wire                   go,          // one-cycle pulse; descriptor must
                                               // already be valid
    output reg                    busy,
    output reg                    done,        // one-cycle pulse at layer end

    // ---- activation buffers ------------------------------------------------
    output wire [ACT_AW-1:0]      act_ra,      // read: source buffer
    input  wire [ACT_W-1:0]       act_rq,      // ... one cycle later
    output reg                    act_we,      // write: destination buffer
    output reg  [ACT_AW-1:0]      act_wa,
    output reg  [ACT_W-1:0]       act_wd,

    // ---- parameter memories ------------------------------------------------
    output wire [WGT_AW-1:0]      wgt_addr,
    input  wire [WGT_W-1:0]       wgt_q,
    output wire [PAR_AW-1:0]      bias_addr,   // shared index with par_addr
    input  wire signed [31:0]     bias_q,
    output wire [PAR_AW-1:0]      par_addr,
    input  wire [31:0]            par_q,       // {shift, m0}

    // ---- fc tail -----------------------------------------------------------
    output reg                    fc_valid,    // one pulse per class
    output reg  [1:0]             fc_idx,
    output reg signed [PROD_W-1:0] fc_prod,    // M0[c] * acc[c], untruncated

    // ---- V-3 trace: the int32 accumulator, before requantisation -----------
    // Driven only when DEBUG_TRACE = 1, held at reset value otherwise. They are
    // real top-level ports, so out-of-context synthesis does NOT trim them on its
    // own -- which is why DEBUG_TRACE exists and the shipped build sets it to 0.
    // In simulation they are what tb/tb_ds_top.sv compares for V-3.
    output reg                    dbg_acc_valid,
    output reg  [2:0]             dbg_layer,
    output reg  [CH_W-1:0]        dbg_ch,
    output reg  [CFG_W-1:0]       dbg_pos,
    output reg signed [ACC_W-1:0] dbg_acc
);

    localparam S_IDLE     = 4'd0;
    localparam S_OC_ADDR  = 4'd1;   // present bias / requant-param address
    localparam S_OC_LOAD  = 4'd2;   // capture them
    localparam S_POS_INIT = 4'd3;   // preload the accumulator with the bias
    localparam S_ACC      = 4'd4;   // I*K reduction cycles
    localparam S_DRAIN    = 4'd5;   // the MAC pipeline's last two terms land
    localparam S_DRAIN2   = 4'd9;   // ... here (three stages, see the header)
    localparam S_OUT      = 4'd6;   // trace, pool, and start the requantiser
    localparam S_POOL_FLU = 4'd7;   // pooled channel -> requantiser, once per ch
    localparam S_RQW      = 4'd8;   // wait out the requantiser's three stages

    // Width-correct ones, so that "+ 1" never depends on a literal's default
    // width being what the target happens to be.
    localparam [CH_W-1:0]  ONE_CH  = {{(CH_W-1){1'b0}}, 1'b1};
    localparam [K_W-1:0]   ONE_K   = {{(K_W-1){1'b0}}, 1'b1};
    localparam [CFG_W-1:0] ONE_POS = {{(CFG_W-1){1'b0}}, 1'b1};

    reg [3:0] state;
    reg       from_pool;            // the in-flight requant is a pooled channel

    // ---- loop counters -----------------------------------------------------
    reg [CH_W-1:0]  o;       // output channel, 0..O-1
    reg [CFG_W-1:0] pos;     // output position, 0..L_out-1
    reg [CH_W-1:0]  i;       // input channel, 0..I-1
    reg [K_W-1:0]   k;       // kernel tap, 0..K-1

    // ---- incrementally maintained addresses --------------------------------
    reg [WGT_AW-1:0] w_oc_base;      // first weight of output channel o
    reg [WGT_AW-1:0] w_ptr;          // walking weight pointer
    reg [ACT_AW-1:0] a_row;          // i * L_in
    reg signed [T_W-1:0] t_base;     // pos * stride - pad
    reg [ACT_AW-1:0] w_row;          // o * cfg_dst_l, destination row

    // Zero-extensions of the descriptor fields to the address widths. Written as
    // explicit concatenations rather than as part-selects, because a part-select
    // wider than its source is an out-of-bounds read: the tools warn, and what
    // they fill the extra bits with is not something to rely on.
    wire [ACT_AW-1:0] l_in_ext  = {{(ACT_AW-CFG_W){1'b0}}, cfg_l_in};
    wire [ACT_AW-1:0] dst_l_ext = {{(ACT_AW-CFG_W){1'b0}}, cfg_dst_l};
    wire [ACT_AW-1:0] pos_ext   = {{(ACT_AW-CFG_W){1'b0}}, pos};
    wire [WGT_AW-1:0] terms_ext = {{(WGT_AW-CFG_W){1'b0}}, cfg_terms};

    // ---- per-output-channel constants --------------------------------------
    reg signed [31:0]    bias_reg;
    reg [FRAC_W-1:0]     m0_reg;
    reg [SHIFT_W-1:0]    shift_reg;

    // ---- accumulator and pipeline flags ------------------------------------
    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] prod_r;   // the product, registered (see the header)
    reg                    ce_q;     // a term was issued last cycle
    reg                    val_q;    // ... and it was inside the spectrum
    reg                    ce_q2;    // ... and its product is in prod_r now

    // ---- addressing --------------------------------------------------------
    wire signed [T_W-1:0] t      = t_base + {{(T_W-K_W){1'b0}}, k};
    wire               in_range  = (t >= 0) &&
                                   (t < $signed({{(T_W-CFG_W){1'b0}}, cfg_l_in}));
    wire [ACT_AW-1:0]  act_ra_c  = a_row + t[ACT_AW-1:0];

    // Out-of-range taps still present an address; it is masked at the MAC, and
    // holding it at a_row keeps it inside the buffer so no simulator reads an
    // out-of-bounds location.
    assign act_ra   = in_range ? act_ra_c : a_row;
    assign wgt_addr = w_ptr;
    assign bias_addr = cfg_pbase + {1'b0, o};
    assign par_addr  = cfg_pbase + {1'b0, o};

    // ---- the multiplier ----------------------------------------------------
    wire signed [ACC_W-1:0] mac_sum;
    ds_mac_array #(
        .LANES (LANES), .A_W (ACT_W), .W_W (WGT_W), .SUM_W (ACC_W)
    ) u_mac (
        .act     (act_rq),
        .wgt     (wgt_q),
        .lane_en (val_q),
        .sum     (mac_sum)
    );

    // ---- relu, pool, requantise, clamp -------------------------------------
    wire signed [ACC_W-1:0] acc_relu;
    wire signed [ACT_W-1:0] q_clamped;
    wire signed [ACC_W-1:0] rq_q;
    wire signed [PROD_W-1:0] rq_prod;

    wire pool_last = cfg_is_pool && (pos == cfg_l_out - ONE_POS);

    wire signed [ACC_W-1:0] pool_sum;
    ds_global_pool #(.ACC_W (ACC_W)) u_pool (
        .clk    (clk),
        .rst_n  (rst_n),
        .clr    ((state == S_OUT) && cfg_is_pool && (pos == {CFG_W{1'b0}})),
        .acc_en ((state == S_OUT) && cfg_is_pool),
        .acc_in (acc_relu),
        .sum    (pool_sum)
    );

    // Requantisation input: the pooled sum for conv4's flush, the relu'd
    // accumulator everywhere else. The fc layer passes through with relu
    // disabled and uses `rq_prod` rather than `rq_q`.
    wire signed [ACC_W-1:0] rq_in = (state == S_POOL_FLU) ? pool_sum : acc_relu;

    ds_relu_clamp #(.ACC_W (ACC_W), .OUT_W (ACT_W)) u_relu_clamp (
        .acc_in   (acc),
        .relu_en  (!cfg_is_last),
        .acc_relu (acc_relu),
        .q_in     (rq_q),
        .q_clamp  (q_clamped)
    );

    // Three clocked stages -- see ds_requant.v for why. `rq_start` is asserted
    // for one cycle in S_OUT (or S_POOL_FLU) and `rq_valid` comes back three
    // cycles later, which is what S_RQW waits out.
    wire rq_start = (state == S_OUT && !cfg_is_pool) || (state == S_POOL_FLU);
    wire rq_valid;

    ds_requant #(
        .ACC_W (ACC_W), .FRAC_W (FRAC_W), .SHIFT_W (SHIFT_W), .PROD_W (PROD_W)
    ) u_requant (
        .clk       (clk),
        .in_valid  (rq_start),
        .acc       (rq_in),
        .m0        (m0_reg),
        .shift     (shift_reg),
        .out_valid (rq_valid),
        .prod      (rq_prod),
        .q         (rq_q)
    );

    // ---- loop bookkeeping --------------------------------------------------
    wire k_last   = (k == cfg_k[K_W-1:0] - ONE_K);
    wire i_last   = (i == cfg_in_ch[CH_W-1:0] - ONE_CH);
    wire red_last = k_last && i_last;
    wire pos_last = (pos == cfg_l_out - ONE_POS);
    wire oc_last  = (o == cfg_o_ch[CH_W-1:0] - ONE_CH);

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            from_pool <= 1'b0;
            act_we   <= 1'b0;
            ce_q     <= 1'b0;
            ce_q2    <= 1'b0;
            val_q    <= 1'b0;
            fc_valid <= 1'b0;
            dbg_acc_valid <= 1'b0;
        end else begin
            act_we        <= 1'b0;
            done          <= 1'b0;
            fc_valid      <= 1'b0;
            dbg_acc_valid <= 1'b0;
            // (the trace writes below are inside `if (DEBUG_TRACE)`)
            ce_q          <= (state == S_ACC);
            val_q         <= (state == S_ACC) && in_range;

            // MAC pipeline stages 2 and 3. Splitting the product away from the
            // accumulate is a timing fix, measured: with them in one cycle the
            // path block RAM -> 8x8 multiply -> 32-bit add -> acc was 10.505 ns
            // against a 10 ns target (artifacts/rtl/synth_v2_pipelined_requant/).
            // It costs one more cycle per output element, about 0.7 % of the
            // inference.
            prod_r <= mac_sum;
            ce_q2  <= ce_q;
            if (ce_q2)
                acc <= acc + prod_r;

            case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                if (go) begin
                    busy      <= 1'b1;
                    o         <= {CH_W{1'b0}};
                    w_oc_base <= cfg_wbase;
                    w_row     <= {ACT_AW{1'b0}};
                    state     <= S_OC_ADDR;
                end
            end

            // bias_addr / par_addr are combinational from `o`; one cycle to
            // present them, one for the memories to answer.
            S_OC_ADDR: state <= S_OC_LOAD;

            S_OC_LOAD: begin
                bias_reg  <= bias_q;
                m0_reg    <= par_q[FRAC_W-1:0];
                shift_reg <= par_q[FRAC_W +: SHIFT_W];
                pos       <= {CFG_W{1'b0}};
                t_base    <= -$signed({{(T_W-4){1'b0}}, cfg_pad});
                state     <= S_POS_INIT;
            end

            // -----------------------------------------------------------
            S_POS_INIT: begin
                acc   <= bias_reg;             // preload: acc = b, then += W*x
                i     <= {CH_W{1'b0}};
                k     <= {K_W{1'b0}};
                a_row <= {ACT_AW{1'b0}};
                w_ptr <= w_oc_base;
                state <= S_ACC;
            end

            S_ACC: begin
                w_ptr <= w_ptr + 1'b1;
                if (red_last) begin
                    state <= S_DRAIN;
                end else if (k_last) begin
                    k     <= {K_W{1'b0}};
                    i     <= i + ONE_CH;
                    a_row <= a_row + l_in_ext;
                end else begin
                    k <= k + ONE_K;
                end
            end

            // Two cycles, not one: the MAC pipeline is address -> memory ->
            // product -> accumulate, so the last term issued in S_ACC only
            // reaches the accumulator two cycles later. Getting this off by one
            // silently drops the last tap of every kernel -- a plausible-looking
            // wrong answer, which is what V-3 exists to catch.
            S_DRAIN:  state <= S_DRAIN2;
            S_DRAIN2: state <= S_OUT;

            // -----------------------------------------------------------
            // The accumulator is complete here. This cycle emits the V-3 trace,
            // feeds the pool on conv4, and STARTS the requantiser (rq_start is
            // combinational from the state). Nothing is written yet: the
            // requantiser answers three cycles later, in S_RQW.
            S_OUT: begin
                if (DEBUG_TRACE) begin
                    dbg_acc_valid <= 1'b1;
                    dbg_layer     <= cfg_layer;
                    dbg_ch        <= o;
                    dbg_pos       <= pos;
                    dbg_acc       <= acc;
                end

                if (cfg_is_pool) begin
                    // u_pool consumes acc_relu this cycle. Only the last
                    // position of a channel needs requantising, so the other 31
                    // go straight on and cost nothing.
                    if (pool_last) begin
                        state <= S_POOL_FLU;
                    end else begin
                        pos    <= pos + ONE_POS;
                        t_base <= t_base + $signed({{(T_W-2){1'b0}}, cfg_stride});
                        state  <= S_POS_INIT;
                    end
                end else begin
                    from_pool <= 1'b0;
                    state     <= S_RQW;
                end
            end

            // One cycle per output CHANNEL, not per position: the pooled sum is
            // only complete after S_OUT's accumulate has landed, and this is
            // where the pooled channel enters the requantiser.
            S_POOL_FLU: begin
                from_pool <= 1'b1;
                state     <= S_RQW;
            end

            // The requantiser's three stages. Costs three cycles per output
            // element -- 12,355 elements, about 2 % of the inference -- and buys
            // the difference between 55 MHz and timing closure at 100 MHz. It
            // could be hidden entirely by overlapping it with the next position's
            // reduction, at the price of carrying the destination address and
            // the channel index down the pipeline with it; 2 % did not justify
            // that, and the choice is written here rather than left to be
            // rediscovered.
            S_RQW: if (rq_valid) begin
                if (cfg_is_last) begin
                    fc_valid <= 1'b1;
                    // The fc layer has three output channels, so the class index
                    // is the low bits of the output-channel counter. ds_fc
                    // asserts N_CLASS == 3 in simulation.
                    fc_idx   <= o[1:0];
                    fc_prod  <= rq_prod;
                end else begin
                    act_we <= 1'b1;
                    // cfg_dst_l is 1 when pooling, so the pooled channel lands
                    // at w_row and the fc layer reads it at i * 1 + 0.
                    act_wa <= from_pool ? w_row : (w_row + pos_ext);
                    act_wd <= q_clamped;
                end

                if (!from_pool && !pos_last) begin
                    pos    <= pos + ONE_POS;
                    t_base <= t_base + $signed({{(T_W-2){1'b0}}, cfg_stride});
                    state  <= S_POS_INIT;
                end else if (!oc_last) begin
                    o         <= o + ONE_CH;
                    w_oc_base <= w_oc_base + terms_ext;
                    w_row     <= w_row + dst_l_ext;
                    state     <= S_OC_ADDR;
                end else begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
