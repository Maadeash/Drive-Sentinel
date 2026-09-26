//=============================================================================
// ds_axis_in -- AXI4-Stream slave: one input window into activation buffer A
//=============================================================================
// Carries ONE window: 5 channels x 512 order bins of int8, 2,560 bytes, as 640
// beats of 32 bits. Byte 4n of the frame is TDATA[7:0] of beat n, i.e. plain
// little-endian within the beat.
//
// WHAT IS ON THE WIRE, AND WHY IT IS ALREADY int8
// -----------------------------------------------
// docs/rtl_declarations.md D-4. rtl_spec.md section 2 put the input
// standardisation inside the accelerator and section 7 put 2,560 int8 bytes on
// the stream; those cannot both be true, because 2,560 int8 bytes IS the
// standardised-and-quantised input. Section 7 wins now that section 9.1 is
// answered PS software: the front end runs on the PS, so
//
//     x_int8 = clamp(round_half_to_even((x - mean) / (std * s_x0)), -128, 127)
//
// belongs to the PS with the rest of the DSP chain, and the fabric is integer
// only from the first convolution onward -- bit-exact by construction rather than
// by luck. The cost is that the standardisation is NOT verified in hardware,
// because it is not in hardware, and docs/results_rtl.md says so in those words.
//
// ADDRESS ORDER. addr = channel * 512 + bin, matching ds_act_mem's channel-major
// layout, so the stream is simply the buffer image in order and this module does
// no reordering at all.
//
// BACKPRESSURE. The buffer is one byte wide, so a beat takes four cycles to
// write and TREADY is low for three of them. 640 beats therefore cost 2,560
// cycles against an inference of roughly 1.7 million: the input transfer is
// 0.15 % of the frame time and is not worth widening the buffer for.
//=============================================================================

module ds_axis_in #(
    parameter integer N_BYTES = 2560,   // 5 channels x 512 bins
    parameter integer ACT_AW  = 12
) (
    input  wire              clk,
    input  wire              rst_n,

    // AXI4-Stream slave
    input  wire [31:0]       s_axis_tdata,
    input  wire              s_axis_tvalid,
    output wire              s_axis_tready,
    input  wire              s_axis_tlast,

    // write port into activation buffer A
    output reg               act_we,
    output reg  [ACT_AW-1:0] act_wa,
    output reg  [7:0]        act_wd,

    // a complete frame has been written
    output reg               frame_done,   // one-cycle pulse
    output reg               frame_valid,  // sticky: a full frame is in the buffer
    // a frame that ended early or ran long -- reported, never silently accepted
    output reg               frame_error
);

    // Sized copy of the byte count: a part-select on an integer parameter is not
    // portable across simulators, and this comparison happens on every beat.
    localparam [ACT_AW:0] N_BYTES_C = N_BYTES[ACT_AW:0];

    reg [1:0]        sub;     // which byte of the held beat is being written
    reg [31:0]       beat;
    reg              hold;    // a beat is captured and being written out
    reg [ACT_AW:0]   count;   // bytes written so far in this frame
    reg              last_q;

    assign s_axis_tready = !hold;

    always @(posedge clk) begin
        if (!rst_n) begin
            act_we      <= 1'b0;
            sub         <= 2'd0;
            hold        <= 1'b0;
            count       <= {(ACT_AW+1){1'b0}};
            frame_done  <= 1'b0;
            frame_valid <= 1'b0;
            frame_error <= 1'b0;
        end else begin
            act_we     <= 1'b0;
            frame_done <= 1'b0;

            if (!hold) begin
                if (s_axis_tvalid) begin
                    beat   <= s_axis_tdata;
                    last_q <= s_axis_tlast;
                    hold   <= 1'b1;
                    sub    <= 2'd0;
                end
            end else begin
                act_we <= 1'b1;
                act_wa <= count[ACT_AW-1:0];
                case (sub)
                2'd0: act_wd <= beat[7:0];
                2'd1: act_wd <= beat[15:8];
                2'd2: act_wd <= beat[23:16];
                2'd3: act_wd <= beat[31:24];
                endcase
                count <= count + 1'b1;
                if (sub == 2'd3) begin
                    hold <= 1'b0;
                    // TLAST and the byte count must agree. A frame that ends on
                    // the wrong beat is a host bug, and a silently short window
                    // would be classified anyway, from whatever the previous
                    // frame left in the buffer.
                    if (count + 1'b1 == N_BYTES_C) begin
                        count       <= {(ACT_AW+1){1'b0}};
                        frame_done  <= 1'b1;
                        frame_valid <= 1'b1;
                        frame_error <= frame_error | ~last_q;
                    end else if (last_q) begin
                        count       <= {(ACT_AW+1){1'b0}};
                        frame_error <= 1'b1;
                    end
                end else begin
                    sub <= sub + 2'd1;
                end
            end
        end
    end

endmodule
