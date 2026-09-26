//=============================================================================
// tb_ds_top -- drives real order-spectrum windows through the RTL
//=============================================================================
// The verification harness for V-1, V-2, V-3, V-5 and V-6 of rtl_spec.md section
// 8. One simulation runs one vector set; scripts/rtl/run_verify.py (wrapped by
// run_tb.ps1 / run_tb.sh) runs them all and turns the reports into
// artifacts/rtl/verify.json, from which docs/results_rtl.md is generated. No number in this flow is typed by
// a human at any point.
//
// WHAT IT DRIVES
// --------------
// The AXI4-Stream input and the AXI4-Lite control interface, not the internals.
// The only internal signals it touches are the ds_conv1d accumulator trace, which
// exists for V-3 and is compiled out (DEBUG_TRACE = 0) for the V-1 run and for
// synthesis. Everything else goes through the
// buses the PS would use, so a bus bug and an arithmetic bug are both in scope.
//
// WHY V-3's TRACE MATTERS MORE THAN V-1's ARGMAX WHILE DEBUGGING
// --------------------------------------------------------------
// rtl_spec.md section 8: argmax agreement can hide a broken layer that later
// layers wash out. The accumulator monitor below compares EVERY int32
// accumulator of every layer in emission order and reports the first divergence
// with its layer, channel and position -- which is the difference between "the
// answer is wrong" and "conv2 channel 17 position 0 is wrong, so it is the left
// padding".
//
// CONFIGURATION, AND WHY IT IS A FILE
// -----------------------------------
// One plusarg, +CFG<n>, naming artifacts/rtl/sim/cfg_<n>.txt, which holds the
// paths and counts as `key value` lines. Not a plusarg per setting, because
// Vivado's Windows launcher scripts rebuild their argument list from %1 with
// shift, and cmd.exe splits a token on '=' -- so +IN=path arrives as the two
// tokens +IN and path and the simulation sees neither. A numbered flag has no
// '=' in it and survives. Written down because it looks like a stylistic choice
// and is not.
//
// Keys: set, in, exp, acc (optional -- its presence enables the V-3 monitor),
// nwin, first, buildid (decimal), result, maxfail.
//=============================================================================

`timescale 1ns / 1ps

module tb_ds_top;

    localparam integer CLK_NS   = 10;     // 100 MHz nominal; timing closure is
                                          // decided by synthesis, not by this
    localparam integer BEATS    = 640;    // 2,560 int8 bytes per window
    localparam integer MAX_ACC  = 20000;  // accumulators per window, upper bound

    // ---- DUT ports ---------------------------------------------------------
    reg         clk = 1'b0;
    reg         rst_n = 1'b0;

    reg  [7:0]  awaddr;  reg awvalid;  wire awready;
    reg  [31:0] wdata;   reg wvalid;   wire wready;  reg [3:0] wstrb;
    wire [1:0]  bresp;   wire bvalid;  reg bready;
    reg  [7:0]  araddr;  reg arvalid;  wire arready;
    wire [31:0] rdata;   wire [1:0] rresp; wire rvalid; reg rready;

    reg  [31:0] tdata;   reg tvalid;   wire tready;   reg tlast;

    wire        irq;
    wire [1:0]  class_o;
    wire        dbg_acc_valid;
    wire [2:0]  dbg_layer;
    wire [6:0]  dbg_ch;
    wire [9:0]  dbg_pos;
    wire signed [31:0] dbg_acc;

    // The V-3 accumulator trace is on by default. `+define TB_NO_TRACE` builds
    // the SHIPPED configuration instead (DEBUG_TRACE = 0, the one synthesis
    // reports), which is what V-1 runs: the argmax verdict should be about the
    // design that ships, not the instrumented one.
`ifdef TB_NO_TRACE
    localparam integer TB_TRACE = 0;
`else
    localparam integer TB_TRACE = 1;
`endif

    ds_top #(.LANES (1), .DEBUG_TRACE (TB_TRACE)) dut (
        .clk (clk), .rst_n (rst_n),
        .s_axi_awaddr (awaddr), .s_axi_awvalid (awvalid), .s_axi_awready (awready),
        .s_axi_wdata (wdata), .s_axi_wstrb (wstrb), .s_axi_wvalid (wvalid),
        .s_axi_wready (wready), .s_axi_bresp (bresp), .s_axi_bvalid (bvalid),
        .s_axi_bready (bready),
        .s_axi_araddr (araddr), .s_axi_arvalid (arvalid), .s_axi_arready (arready),
        .s_axi_rdata (rdata), .s_axi_rresp (rresp), .s_axi_rvalid (rvalid),
        .s_axi_rready (rready),
        .s_axis_tdata (tdata), .s_axis_tvalid (tvalid), .s_axis_tready (tready),
        .s_axis_tlast (tlast),
        .irq (irq), .class_idx_o (class_o),
        .dbg_acc_valid (dbg_acc_valid), .dbg_layer (dbg_layer),
        .dbg_ch (dbg_ch), .dbg_pos (dbg_pos), .dbg_acc (dbg_acc)
    );

    always #(CLK_NS/2) clk = ~clk;

    // ---- configuration -----------------------------------------------------
    string in_path, exp_path, acc_path, res_path, set_name, shown_acc;
    int    n_win, first_win, max_fail;
    longint exp_build_id;
    int    fd_in, fd_exp, fd_acc, fd_res, fd_cfg;
    int    have_acc;
    int    cfg_id;

    // ---- tallies -----------------------------------------------------------
    int    windows_run      = 0;
    int    class_mismatch   = 0;
    int    logit_mismatch   = 0;
    longint max_logit_diff  = 0;
    longint acc_compared    = 0;
    longint acc_mismatch    = 0;
    int    first_bad_window = -1;
    int    first_bad_layer  = -1;
    int    first_bad_ch     = -1;
    int    first_bad_pos    = -1;
    longint first_bad_exp   = 0;
    longint first_bad_got   = 0;
    int    cur_window       = 0;
    int    cycles_min       = 0;
    int    cycles_max       = 0;
    int    cycles_first     = 0;
    int    got_build_id     = 0;
    int    build_id_ok      = 0;

    // =======================================================================
    // AXI4-Lite
    // =======================================================================
    task automatic axil_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            awaddr <= addr; awvalid <= 1'b1;
            wdata  <= data; wvalid  <= 1'b1; wstrb <= 4'hF;
            bready <= 1'b1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            awvalid <= 1'b0; wvalid <= 1'b0;
            while (!bvalid) @(posedge clk);
            @(posedge clk);
            bready <= 1'b0;
        end
    endtask

    task automatic axil_read(input [7:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            araddr <= addr; arvalid <= 1'b1; rready <= 1'b1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            arvalid <= 1'b0;
            while (!rvalid) @(posedge clk);
            data = rdata;
            @(posedge clk);
            rready <= 1'b0;
        end
    endtask

    // =======================================================================
    // AXI4-Stream: one window
    // =======================================================================
    task automatic push_window();
        int b, code;
        logic [31:0] word;
        begin
            for (b = 0; b < BEATS; b = b + 1) begin
                code = $fscanf(fd_in, "%h\n", word);
                if (code != 1)
                    $fatal(1, "vector file ran out at window %0d beat %0d",
                           cur_window, b);
                @(posedge clk);
                tdata  <= word;
                tvalid <= 1'b1;
                tlast  <= (b == BEATS - 1);
                @(posedge clk);
                while (!tready) @(posedge clk);
                tvalid <= 1'b0;
                tlast  <= 1'b0;
            end
        end
    endtask

    // =======================================================================
    // V-3: every int32 accumulator, in emission order
    // =======================================================================
    // The engine emits one accumulator per output element, layer by layer, then
    // channel, then position. The expected file is written in exactly that order
    // by scripts/rtl/make_vectors.py, so this is a straight sequential compare --
    // and a desynchronisation shows up immediately rather than as a wrong value
    // much later.
    always @(posedge clk) begin
        if (have_acc && dbg_acc_valid) begin
            longint want;
            string  tok;
            int code;
            // %s then atoi(), not %d: XSim's $fscanf mis-parses negative
            // decimals, and roughly half of these accumulators are negative.
            code = $fscanf(fd_acc, "%s", tok);
            want = tok.atoi();
            if (code != 1) begin
                $display("[V-3] accumulator file exhausted at window %0d", cur_window);
                have_acc = 0;
            end else begin
                acc_compared = acc_compared + 1;
                if (want !== longint'(dbg_acc)) begin
                    if (acc_mismatch == 0) begin
                        first_bad_window = cur_window;
                        first_bad_layer  = dbg_layer;
                        first_bad_ch     = dbg_ch;
                        first_bad_pos    = dbg_pos;
                        first_bad_exp    = want;
                        first_bad_got    = dbg_acc;
                        $display("[V-3] FIRST DIVERGENCE window %0d layer %0d channel %0d position %0d: expected %0d got %0d",
                                 cur_window, dbg_layer, dbg_ch, dbg_pos,
                                 want, dbg_acc);
                    end
                    acc_mismatch = acc_mismatch + 1;
                end
            end
        end
    end

    // =======================================================================
    // main
    // =======================================================================
    integer w, code;
    integer e_class, e_l0, e_l1, e_l2;
    string  t_class, t_l0, t_l1, t_l2;
    logic [31:0] rv, st;
    longint d;

    initial begin
        awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
        wstrb = 4'hF; tvalid = 0; tlast = 0; tdata = 0;
        have_acc = 0;

        read_config();

        if (acc_path != "") begin
            fd_acc = $fopen(acc_path, "r");
            if (fd_acc == 0) $fatal(1, "cannot open %s", acc_path);
            have_acc = 1;
        end

        fd_in  = $fopen(in_path, "r");
        fd_exp = $fopen(exp_path, "r");
        if (fd_in == 0)  $fatal(1, "cannot open %s", in_path);
        if (fd_exp == 0) $fatal(1, "cannot open %s", exp_path);

        repeat (8) @(posedge clk);
        rst_n <= 1'b1;

        // ---- V-6: the build-ID walk -------------------------------------
        // The core refuses to start until this finishes, so waiting on it is
        // not politeness, it is the handshake.
        st = 0;
        while (!st[4]) begin
            axil_read(8'h04, st);
        end
        axil_read(8'h18, rv); got_build_id = rv;
        build_id_ok = (longint'(unsigned'(rv)) === exp_build_id);
        axil_read(8'h20, rv);
        if (rv !== 32'h44533031)
            $display("[V-6] MAGIC mismatch: got %08h", rv);
        axil_read(8'h24, rv);
        $display("[cfg] frac_width=%0d lanes=%0d logit_shift=%0d",
                 rv[7:0], rv[15:8], rv[23:16]);
        $display("[V-6] build id: got %08h expected %08h  -> %0s",
                 got_build_id, exp_build_id[31:0],
                 (longint'(unsigned'(got_build_id)) === exp_build_id)
                     ? "MATCH" : "MISMATCH");

        // ---- the windows ------------------------------------------------
        for (w = 0; w < n_win; w = w + 1) begin
            cur_window = first_win + w;
            push_window();
            @(posedge irq);
            @(posedge clk);

            axil_read(8'h1C, rv);
            if (w == 0) begin
                cycles_first = rv; cycles_min = rv; cycles_max = rv;
            end else begin
                if (rv < cycles_min) cycles_min = rv;
                if (rv > cycles_max) cycles_max = rv;
            end

            axil_read(8'h04, st);
            if (!st[0])
                $display("[bus] window %0d: STATUS.done not set", cur_window);

            code = $fscanf(fd_exp, "%s %s %s %s", t_class, t_l0, t_l1, t_l2);
            if (code != 4) $fatal(1, "expected file ran out at window %0d",
                                  cur_window);
            e_class = t_class.atoi();
            e_l0    = t_l0.atoi();
            e_l1    = t_l1.atoi();
            e_l2    = t_l2.atoi();

            axil_read(8'h08, rv);
            if (rv[1:0] !== e_class[1:0]) begin
                class_mismatch = class_mismatch + 1;
                if (class_mismatch <= max_fail)
                    $display("[V-1] window %0d: class %0d, expected %0d",
                             cur_window, rv[1:0], e_class);
            end

            check_logit(8'h0C, e_l0, 0);
            check_logit(8'h10, e_l1, 1);
            check_logit(8'h14, e_l2, 2);

            windows_run = windows_run + 1;
            // Rewrite the report after EVERY window. V-1 takes most of a day on
            // this machine and a run that is killed part-way must still leave
            // its evidence behind: the previous attempt reached window 768 of
            // every shard with zero mismatches and recorded none of it, because
            // the report was only written at $finish. The report is a few
            // hundred bytes against five seconds of simulation per window.
            write_report();
            if ((w % 256) == 0)
                $display("  window %0d / %0d  (class mismatches %0d, acc mismatches %0d)", w, n_win, class_mismatch,
                         acc_mismatch);
        end

        write_report();
        $display("DONE %0s: %0d windows, class mismatches %0d, logit mismatches %0d, acc mismatches %0d",
                 set_name, windows_run, class_mismatch, logit_mismatch,
                 acc_mismatch);
        $finish;
    end

    // Reads artifacts/rtl/sim/cfg_<n>.txt. Unknown keys are reported rather than
    // ignored: a typo in a runner script must not silently turn a bar into a
    // weaker one by leaving a path unset.
    task automatic read_config();
        string k, v;
        int code;
        begin
            in_path = ""; exp_path = ""; acc_path = ""; res_path = "";
            set_name = "unnamed"; n_win = 0; first_win = 0; max_fail = 8;
            exp_build_id = 0;

            if (!$value$plusargs("CFG%d", cfg_id))
                $fatal(1, "+CFG<n> is required (see the header)");
            res_path = $sformatf("artifacts/rtl/sim/cfg_%0d.txt", cfg_id);
            fd_cfg = $fopen(res_path, "r");
            if (fd_cfg == 0) $fatal(1, "cannot open %0s", res_path);
            res_path = "";

            forever begin
                code = $fscanf(fd_cfg, "%s %s\n", k, v);
                if (code != 2) break;
                case (k)
                    "set":     set_name     = v;
                    "in":      in_path      = v;
                    "exp":     exp_path     = v;
                    "acc":     acc_path     = v;
                    "result":  res_path     = v;
                    "nwin":    n_win        = v.atoi();
                    "first":   first_win    = v.atoi();
                    "maxfail": max_fail     = v.atoi();
                    "buildid": exp_build_id = v.atoi();
                    default:   $display("[cfg] unknown key '%0s' ignored", k);
                endcase
            end
            $fclose(fd_cfg);

            if (in_path == "")  $fatal(1, "config has no 'in'");
            if (exp_path == "") $fatal(1, "config has no 'exp'");
            if (n_win == 0)     $fatal(1, "config has no 'nwin'");
            if (res_path == "") res_path = "artifacts/rtl/sim/result.txt";
            // Assigned, not a ternary between a string literal and a string
            // variable: XSim's kernel aborts on that mix, and it aborts at time
            // 0 with a stack pointing into an unrelated module.
            if (acc_path == "") shown_acc = "(none)";
            else                shown_acc = acc_path;
            $display("[cfg] set=%0s nwin=%0d first=%0d acc=%0s",
                     set_name, n_win, first_win, shown_acc);
        end
    endtask

    task automatic check_logit(input [7:0] addr, input integer want,
                               input integer idx);
        logic [31:0] got;
        longint diff;
        begin
            axil_read(addr, got);
            if ($signed(got) !== want) begin
                logit_mismatch = logit_mismatch + 1;
                diff = $signed(got) - want;
                if (diff < 0) diff = -diff;
                if (diff > max_logit_diff) max_logit_diff = diff;
                if (logit_mismatch <= max_fail)
                    $display("[V-2] window %0d logit %0d: got %0d expected %0d",
                             cur_window, idx, $signed(got), want);
            end
        end
    endtask

    task automatic write_report();
        begin
            fd_res = $fopen(res_path, "w");
            $fwrite(fd_res, "set %0s\n", set_name);
            $fwrite(fd_res, "first_window %0d\n", first_win);
            $fwrite(fd_res, "windows_requested %0d\n", n_win);
            $fwrite(fd_res, "windows_run %0d\n", windows_run);
            $fwrite(fd_res, "class_mismatches %0d\n", class_mismatch);
            $fwrite(fd_res, "logit_mismatches %0d\n", logit_mismatch);
            $fwrite(fd_res, "max_abs_logit_diff_lsb %0d\n", max_logit_diff);
            $fwrite(fd_res, "acc_compared %0d\n", acc_compared);
            $fwrite(fd_res, "acc_mismatches %0d\n", acc_mismatch);
            $fwrite(fd_res, "first_bad_window %0d\n", first_bad_window);
            $fwrite(fd_res, "first_bad_layer %0d\n", first_bad_layer);
            $fwrite(fd_res, "first_bad_channel %0d\n", first_bad_ch);
            $fwrite(fd_res, "first_bad_position %0d\n", first_bad_pos);
            $fwrite(fd_res, "first_bad_expected %0d\n", first_bad_exp);
            $fwrite(fd_res, "first_bad_got %0d\n", first_bad_got);
            $fwrite(fd_res, "build_id_got %0d\n", got_build_id);
            $fwrite(fd_res, "build_id_expected %0d\n", exp_build_id);
            $fwrite(fd_res, "cycles_first %0d\n", cycles_first);
            $fwrite(fd_res, "cycles_min %0d\n", cycles_min);
            $fwrite(fd_res, "cycles_max %0d\n", cycles_max);
            $fclose(fd_res);
        end
    endtask

endmodule
