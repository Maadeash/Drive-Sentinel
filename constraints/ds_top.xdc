#=============================================================================
# ds_top.xdc -- timing constraints for the DriveSentinel INT8 accelerator
#=============================================================================
# Target: xc7z020clg400-1 (PYNQ-Z2), out-of-context.
#
# WHY OUT OF CONTEXT
# ------------------
# This core is an AXI peripheral of the Zynq PS. Its ~190 ports are AXI-Lite,
# AXI-Stream and sideband -- none of them ever reach a package pin, and the
# clg400 package has fewer user I/O than the core has ports, so a pin-out
# constrained build is not merely unnecessary, it is impossible and would be a
# meaningless thing to report. Out-of-context synthesis and implementation give
# the number that matters for this core: the worst register-to-register path
# inside it.
#
# rtl_spec.md section 9.3 (answered): the clock target is 100 MHz. If timing does
# not close, the achieved figure is reported. It is not relaxed silently, and the
# constraint below is not edited to make a report look better.
#=============================================================================

create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

#-----------------------------------------------------------------------------
# I/O timing
#-----------------------------------------------------------------------------
# Deliberately NOT constrained with set_input_delay / set_output_delay. In an
# out-of-context build the I/O budget depends entirely on what the core is
# connected to in the block design, so any number here would be invented -- and
# it would then dominate the timing summary and be reported as though it said
# something about this core. The interface is fully registered on both sides
# (ds_axil_ctrl and ds_axis_in flop every input they use and every output they
# drive), so the internal paths below are the real constraint.
#
# scripts/rtl/synth.tcl reports the worst clk-to-clk path separately from the
# overall summary for exactly this reason, and docs/results_rtl.md quotes the
# clk-to-clk figure as the achieved clock.
set_false_path -from [get_ports rst_n]
