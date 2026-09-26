#=============================================================================
# synth.tcl -- Vivado batch synthesis and implementation, no bitstream
#=============================================================================
#   cd C:/kone
#   D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal -log <log> \
#       -source scripts/rtl/synth.tcl -tclargs <out_dir> <debug_trace> <lanes>
#
# rtl_spec.md section 9.4 is answered SIMULATION + SYNTHESIS ONLY: no bitstream
# is written and nothing runs on a board. Place and route still run, because a
# post-synthesis timing number is an estimate of an estimate -- it has no routing
# delays in it -- and the point of this step is to replace section 7's arithmetic
# ESTIMATE table with a measurement. What is reported is therefore post-route,
# and docs/results_rtl.md says in as many words that no hardware was involved.
#
# Out of context: the core is an AXI peripheral of the PS and its ~190 ports
# never reach a package pin. See constraints/ds_top.xdc.
#
# Everything it measures lands in <out_dir>/synth.json, which
# scripts/rtl/render_results.py reads. The .rpt files are kept beside it as the
# evidence behind every number.
#=============================================================================

set out_dir   [lindex $argv 0]
set dbg_trace [lindex $argv 1]
set lanes     [lindex $argv 2]
if {$out_dir   eq ""} { set out_dir "artifacts/rtl/synth" }
if {$dbg_trace eq ""} { set dbg_trace 0 }
if {$lanes     eq ""} { set lanes 1 }

set part xc7z020clg400-1
file mkdir $out_dir

puts "=== DriveSentinel RTL: $part, LANES=$lanes, DEBUG_TRACE=$dbg_trace ==="

# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
# Relative paths: $readmemh in ds_weight_mem / ds_bias_mem / ds_param_mem refers
# to artifacts/int8_export/ and rtl/mem/ relative to the launch directory, so
# Vivado must be started from the repository root. That is also what makes the
# synthesised weights provably the exported ones.
set rtl_files {
    rtl/ds_mac_array.v rtl/ds_requant.v rtl/ds_relu_clamp.v rtl/ds_global_pool.v
    rtl/ds_fc.v rtl/ds_weight_mem.v rtl/ds_bias_mem.v rtl/ds_param_mem.v
    rtl/ds_act_mem.v rtl/ds_conv1d.v rtl/ds_ctrl_fsm.v rtl/ds_axis_in.v
    rtl/ds_axil_ctrl.v rtl/ds_weight_crc.v rtl/ds_core.v rtl/ds_top.v
}
read_verilog $rtl_files
read_xdc constraints/ds_top.xdc

# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
synth_design -top ds_top -part $part -mode out_of_context \
    -include_dirs rtl \
    -generic LANES=$lanes -generic DEBUG_TRACE=$dbg_trace

report_utilization -file $out_dir/post_synth_util.rpt
report_timing_summary -file $out_dir/post_synth_timing.rpt

# ---------------------------------------------------------------------------
# implementation -- no write_bitstream
# ---------------------------------------------------------------------------
opt_design
place_design
phys_opt_design
route_design

report_utilization      -file $out_dir/post_route_util.rpt
report_timing_summary   -file $out_dir/post_route_timing.rpt
report_clock_utilization -file $out_dir/post_route_clock.rpt
report_drc              -file $out_dir/post_route_drc.rpt

# The worst register-to-register path in the core. Reported separately from the
# summary because the summary includes unconstrained I/O paths, which say
# nothing about this core (constraints/ds_top.xdc explains why they are left
# unconstrained rather than given an invented budget).
report_timing -max_paths 20 -nworst 20 -delay_type max -sort_by slack \
    -file $out_dir/post_route_clk_to_clk.rpt

# ---------------------------------------------------------------------------
# the numbers, as JSON
# ---------------------------------------------------------------------------
# NOTE ON THE TWO KINDS OF COUNT
# The primitive counts below are raw get_cells tallies. They are NOT the same as
# the "Slice LUTs" line of report_utilization, which accounts for LUT combining
# (two LUT5s sharing one LUT6 site count as one). docs/results_rtl.md quotes the
# REPORT numbers, parsed by scripts/rtl/render_results.py from
# post_route_util.rpt -- the report is Vivado's own summary and is what a
# reviewer would look at. These are kept as a cross-check and labelled as such.

# Every timed path in this design is register-to-register on clk: the I/O is
# left unconstrained on purpose (constraints/ds_top.xdc says why), and an
# unconstrained path carries no slack and is not returned here. So the worst
# path below IS the worst clk-to-clk path, which is what docs/results_rtl.md
# quotes as the achieved clock.
set slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set period [get_property PERIOD [get_clocks clk]]
set achieved_ns [expr {$period - $slack}]
set achieved_mhz [expr {1000.0 / $achieved_ns}]

set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]

set luts  [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
set ffs   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set dsps  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ DSP48*}]]
set bram36 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB36*}]]
set bram18 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB18*}]]
set carry  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ CARRY4*}]]
set srl    [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ SRL*}]]

# Device totals for the xc7z020, from the part database rather than from memory.
set part_obj [get_parts $part]
set tot_lut  [get_property LUT_ELEMENTS   $part_obj]
set tot_ff   [get_property FLIPFLOPS      $part_obj]
set tot_dsp  [get_property DSP            $part_obj]
set tot_bram [get_property BLOCK_RAMS     $part_obj]

set fh [open $out_dir/synth.json w]
puts $fh "{"
puts $fh "  \"part\": \"$part\","
puts $fh "  \"mode\": \"out_of_context\","
puts $fh "  \"vivado\": \"[version -short]\","
puts $fh "  \"lanes\": $lanes,"
puts $fh "  \"debug_trace\": $dbg_trace,"
puts $fh "  \"bitstream_written\": false,"
puts $fh "  \"ran_on_hardware\": false,"
puts $fh "  \"target_period_ns\": $period,"
puts $fh "  \"target_mhz\": [expr {1000.0 / $period}],"
puts $fh "  \"wns_ns\": $slack,"
puts $fh "  \"whs_ns\": $whs,"
puts $fh "  \"achieved_period_ns\": $achieved_ns,"
puts $fh "  \"achieved_mhz\": $achieved_mhz,"
puts $fh "  \"timing_met\": [expr {$slack >= 0 ? "true" : "false"}],"
puts $fh "  \"lut_primitives\": $luts,"
puts $fh "  \"ff_primitives\": $ffs,"
puts $fh "  \"dsp48_primitives\": $dsps,"
puts $fh "  \"ramb36\": $bram36,"
puts $fh "  \"ramb18\": $bram18,"
puts $fh "  \"carry4\": $carry,"
puts $fh "  \"srl\": $srl,"
puts $fh "  \"device_lut\": $tot_lut,"
puts $fh "  \"device_ff\": $tot_ff,"
puts $fh "  \"device_dsp\": $tot_dsp,"
puts $fh "  \"device_bram36\": $tot_bram,"
puts $fh "  \"reports\": \"$out_dir\""
puts $fh "}"
close $fh

puts "=== achieved [format %.3f $achieved_mhz] MHz (WNS [format %.3f $slack] ns) ==="
puts "=== LUT $luts  FF $ffs  DSP48 $dsps  RAMB36 $bram36  RAMB18 $bram18 ==="
puts "=== wrote $out_dir/synth.json ==="
