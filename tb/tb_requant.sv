//=============================================================================
// tb_requant -- V-4: round-half-to-even, at the unit level
//=============================================================================
// rtl_spec.md section 8, V-4: "Directed tie vectors exercising round-half-to-even
// -- exact match on every tie."
//
// WHY THIS IS A UNIT TESTBENCH AND NOT A WINDOW OF DATA
// ----------------------------------------------------
// At the chosen fractional width the exported shifts are 33..40 and the
// mantissas are odd 26-bit values, so `acc * m0 == q*2**s + 2**(s-1)` has no
// solution with `acc` inside int32: an exact tie CANNOT be produced by any
// input window. The rounding mode is still part of the specification and still
// has to be right -- a different export, width or layer would reach it -- so it
// is exercised here, driving ds_requant directly with constructed products.
// scripts/rtl/make_vectors.py writes that finding into the vector set's meta
// file and docs/results_rtl.md repeats it, because "we could not reach the case"
// and "the case passed" are very different sentences.
//
// The expected values come from drivesentinel.rtl.model.round_half_to_even --
// the same three lines ds_requant.v implements, in the other language. Agreement
// between them is the test.
//
// CONFIGURATION. One plusarg, +CFG<n>, naming artifacts/rtl/sim/cfg_<n>.txt with
// `vec <path>` and `result <path>`. See tb_ds_top.sv for why it is a file and not
// a plusarg per setting -- Vivado's Windows launcher loses everything after an
// '=' in an argument.
//=============================================================================

`timescale 1ns / 1ps

`include "ds_defs.vh"

module tb_requant;

    localparam integer ACC_W   = `DS_ACC_W;
    localparam integer FRAC_W  = `DS_FRAC_W;
    localparam integer SHIFT_W = `DS_SHIFT_W;
    localparam integer PROD_W  = `DS_PROD_W;

    reg                      clk = 1'b0;
    reg                      in_valid = 1'b0;
    reg signed [ACC_W-1:0]   acc;
    reg        [FRAC_W-1:0]  m0;
    reg        [SHIFT_W-1:0] shift;
    wire                     out_valid;
    wire signed [PROD_W-1:0] prod;
    wire signed [ACC_W-1:0]  q;
    wire signed [7:0]        q_clamp;

    always #5 clk = ~clk;

    ds_requant #(
        .ACC_W (ACC_W), .FRAC_W (FRAC_W), .SHIFT_W (SHIFT_W), .PROD_W (PROD_W)
    ) dut (
        .clk (clk), .in_valid (in_valid),
        .acc (acc), .m0 (m0), .shift (shift),
        .out_valid (out_valid), .prod (prod), .q (q)
    );

    // The clamp is the other half of the boundary; V-4 checks both, because a
    // rounder that is right and a clamp that is off by one give the same symptom.
    ds_relu_clamp #(.ACC_W (ACC_W), .OUT_W (8)) u_clamp (
        .acc_in   ({ACC_W{1'b0}}),
        .relu_en  (1'b0),
        .acc_relu (),
        .q_in     (q),
        .q_clamp  (q_clamp)
    );

    string vec_path, res_path;
    int    fd, fd_res, fd_cfg, code, cfg_id;
    string k, v;
    // Read as strings and converted with atoi(), not with %d: XSim's $fscanf
    // mis-parses a negative decimal in any field but the first, silently
    // returning a short count. Half of these vectors are negative on purpose.
    int     v_acc, v_m0, v_shift, v_q, v_clamp;
    string  t_acc, t_m0, t_shift, t_q, t_clamp;
    int    n = 0, bad_q = 0, bad_clamp = 0, n_ties = 0, bad_ties = 0;
    int    shown = 0;
    // Sampled in the ACTIVE region of the edge on which out_valid is first seen
    // high. `q` is a flop that is rewritten on the very next edge, so reading it
    // after a #1 -- i.e. after the NBA update -- reads the NEXT vector's result.
    // That is a testbench race, not a DUT bug: V-3 compares the same arithmetic
    // through ds_conv1d and is bit-exact.
    logic                    seen;
    logic signed [ACC_W-1:0] got_q;
    logic signed [PROD_W-1:0] got_prod;
    logic signed [7:0]       got_clamp;

    initial begin
        vec_path = ""; res_path = "";
        if (!$value$plusargs("CFG%d", cfg_id)) $fatal(1, "+CFG<n> is required");
        fd_cfg = $fopen($sformatf("artifacts/rtl/sim/cfg_%0d.txt", cfg_id), "r");
        if (fd_cfg == 0) $fatal(1, "cannot open the config for +CFG%0d", cfg_id);
        forever begin
            code = $fscanf(fd_cfg, "%s %s\n", k, v);
            if (code != 2) break;
            case (k)
                "vec":    vec_path = v;
                "result": res_path = v;
                default:  ;
            endcase
        end
        $fclose(fd_cfg);
        if (vec_path == "") $fatal(1, "config has no 'vec'");
        $display("[cfg] vec=%0s result=%0s", vec_path, res_path);
        if (res_path == "") res_path = "artifacts/rtl/sim/v4.txt";

        fd = $fopen(vec_path, "r");
        if (fd == 0) $fatal(1, "cannot open %0s", vec_path);

        // Prime the pipeline. ds_requant has no reset -- it is three stages of
        // pure dataflow, and in ds_conv1d the enclosing FSM cannot reach S_RQW
        // within three cycles of reset, so no X ever reaches a result. Here the
        // stages start at X, and `while (!seen)` on an X exits immediately, so
        // the first vector would be compared against an uninitialised flop.
        repeat (8) @(posedge clk);

        forever begin
            // %s then atoi(), not %d: XSim's $fscanf mis-parses a negative
            // decimal in any field but the first, returning a short count and
            // no error. Half of these vectors are negative on purpose.
            code = $fscanf(fd, "%s %s %s %s %s",
                           t_acc, t_m0, t_shift, t_q, t_clamp);
            if (code != 5) break;
            v_acc   = t_acc.atoi();
            v_m0    = t_m0.atoi();
            v_shift = t_shift.atoi();
            v_q     = t_q.atoi();
            v_clamp = t_clamp.atoi();

            // One vector at a time through the three-stage pipeline. Slower
            // than streaming them back to back, and exactly as exhaustive; at
            // 3,966 vectors the whole run is microseconds either way.
            @(posedge clk);
            acc      <= v_acc[ACC_W-1:0];
            m0       <= v_m0[FRAC_W-1:0];
            shift    <= v_shift[SHIFT_W-1:0];
            in_valid <= 1'b1;
            @(posedge clk);
            in_valid <= 1'b0;
            do begin
                @(posedge clk);
                seen      = out_valid;
                got_q     = q;
                got_prod  = prod;
                got_clamp = q_clamp;
            end while (seen !== 1'b1);

            n = n + 1;
            // an exact tie: the remainder below the shift is exactly the half
            if (((got_prod & ((longint'(1) << v_shift) - 1)) == (longint'(1) << (v_shift - 1))))
                n_ties = n_ties + 1;

            if (got_q !== v_q[ACC_W-1:0]) begin
                bad_q = bad_q + 1;
                if (((got_prod & ((longint'(1) << v_shift) - 1)) ==
                     (longint'(1) << (v_shift - 1))))
                    bad_ties = bad_ties + 1;
                if (shown < 12) begin
                    shown = shown + 1;
                    $display("[V-4] acc=%0d m0=%0d shift=%0d: got q=%0d expected %0d",
                             v_acc, v_m0, v_shift, got_q, v_q);
                end
            end
            if (got_clamp !== v_clamp[7:0]) begin
                bad_clamp = bad_clamp + 1;
                if (shown < 12) begin
                    shown = shown + 1;
                    $display("[V-4] acc=%0d m0=%0d shift=%0d: got clamp=%0d expected %0d",
                             v_acc, v_m0, v_shift, got_clamp, v_clamp);
                end
            end
        end
        $fclose(fd);

        fd_res = $fopen(res_path, "w");
        $fwrite(fd_res, "set v4\n");
        $fwrite(fd_res, "vectors %0d\n", n);
        $fwrite(fd_res, "exact_ties %0d\n", n_ties);
        $fwrite(fd_res, "q_mismatches %0d\n", bad_q);
        $fwrite(fd_res, "tie_mismatches %0d\n", bad_ties);
        $fwrite(fd_res, "clamp_mismatches %0d\n", bad_clamp);
        $fclose(fd_res);

        $display("DONE v4: %0d vectors (%0d exact ties), q mismatches %0d, clamp mismatches %0d", n, n_ties, bad_q, bad_clamp);
        $finish;
    end

endmodule
