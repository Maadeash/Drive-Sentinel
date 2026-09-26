#=============================================================================
# build_overlay.tcl -- the PYNQ-Z2 overlay around the verified core
#=============================================================================
#   cd C:/kone
#   D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal \
#       -log artifacts/rtl/system/vivado_system.log \
#       -source board/build_overlay.tcl
#
# Produces board/drivesentinel.bit + board/drivesentinel.hwh (the PYNQ overlay)
# and artifacts/rtl/system/ (reports + system.json).
#
# WHAT IS IN THE BLOCK DESIGN
#   Zynq PS7 ── M_AXI_GP0 ── interconnect ─┬─ AXI DMA  S_AXI_LITE   (control)
#                                          └─ ds_0     s_axi        (the core)
#   AXI DMA M_AXI_MM2S ── interconnect ── PS7 S_AXI_HP0              (reads DDR)
#   AXI DMA M_AXIS_MM2S ────────────────── ds_0 s_axis               (one window)
#   FCLK_CLK0 = 100 MHz drives everything; proc_sys_reset makes rst_n.
#
# ds_0 is board/rtl/ds_pynq_wrap.v, which instantiates rtl/ds_top.v UNCHANGED
# with DEBUG_TRACE = 0. Nothing under rtl/ is edited for packaging.
#
# THE CLOCK. 100 MHz: the frequency the core closed at out of context (WNS
# +0.615 ns, artifacts/rtl/synth/). The full system adds a DMA, two
# interconnects and the PS boundary, so it is timed again here and reported
# SEPARATELY. The constraint is the PS7's own FCLK_CLK0 period; nothing here
# overrides or relaxes it. If the full system does not close, NO BITSTREAM IS
# WRITTEN -- a bitstream that fails timing is not one to hand to a board.
#
# THE PS CONFIGURATION. No PYNQ-Z2 board file is installed, so the PS7 is
# configured property by property from the board's published settings, and each
# property that fails to apply is logged rather than silently skipped. For use as
# a PYNQ overlay this is less critical than it looks: the PS is configured at boot
# by the PYNQ image's FSBL and a PL overlay does not reconfigure DDR or MIO. What
# the overlay DOES carry that PYNQ uses is the fabric clock (from the .hwh) and
# the address map.
#
# THREADS. V-1 runs in parallel on this machine and must not be starved, so
# Vivado is held to one thread. Slower, deliberately.
#=============================================================================

set_param general.maxThreads 1

set part      xc7z020clg400-1
set proj_dir  board/build/vivado
set bd_name   drivesentinel
set out_dir   artifacts/rtl/system
set board_dir board
file mkdir $out_dir

create_project ds_overlay $proj_dir -part $part -force
set_property target_language Verilog [current_project]

# ---------------------------------------------------------------------------
# sources: the verified RTL, read from rtl/ in place, and the wrapper
# ---------------------------------------------------------------------------
set rtl_files {
    rtl/ds_mac_array.v rtl/ds_requant.v rtl/ds_relu_clamp.v rtl/ds_global_pool.v
    rtl/ds_fc.v rtl/ds_weight_mem.v rtl/ds_bias_mem.v rtl/ds_param_mem.v
    rtl/ds_act_mem.v rtl/ds_conv1d.v rtl/ds_ctrl_fsm.v rtl/ds_axis_in.v
    rtl/ds_axil_ctrl.v rtl/ds_weight_crc.v rtl/ds_core.v rtl/ds_top.v
}
add_files -norecurse $rtl_files
add_files -norecurse {rtl/ds_defs.vh rtl/ds_gen.vh}
set_property file_type {Verilog Header} [get_files {ds_defs.vh ds_gen.vh}]
add_files -norecurse board/rtl/ds_pynq_wrap.v
set_property include_dirs [file normalize rtl] [current_fileset]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------------------
# block design
# ---------------------------------------------------------------------------
# create_bd_design sources Vivado's own IP-integrator init scripts
# (scripts/ipintegrator/init.tcl), which in turn `source` a dozen helpers BY
# RELATIVE NAME. On this machine that lookup is not deterministic: three runs
# failed on three different helpers ("couldn't read file utils.tcl" / 
# "testbench.tcl" / "configure_noc.tcl"), all of which exist, while the same
# preamble run on its own succeeded. Worse, create_bd_design does not raise when
# it happens -- the failure only surfaces later as "invalid command name
# bd::utils::...". So:
#   * the one call is made with the working directory set to that scripts
#     folder, so the relative names resolve by construction, and the directory
#     is restored immediately (every other path in this script is relative to
#     the repository root, which the $readmemh paths depend on);
#   * success is judged by whether the init actually defined its commands, not
#     by the absence of an exception;
#   * a bounded retry remains as a second line, logged.
set ipi_dir [file join $::env(XILINX_VIVADO) scripts ipintegrator]
set repo_dir [pwd]
set bd_ok 0
set bd_log [open $out_dir/create_bd_attempts.log w]
for {set attempt 1} {$attempt <= 3} {incr attempt} {
    cd $ipi_dir
    set rc [catch {create_bd_design $bd_name} err]
    cd $repo_dir
    set init_ok [expr {[llength [info commands ::bd::utils::is_empty]] > 0}]
    if {$rc == 0 && $init_ok} {
        puts $bd_log "attempt $attempt ok (init commands present)"
        set bd_ok 1
        break
    }
    puts $bd_log "attempt $attempt FAILED rc=$rc init_ok=$init_ok : $err"
    catch {close_bd_design [get_bd_designs -quiet $bd_name]}
    catch {remove_files [get_files -quiet $bd_name.bd]}
    file delete -force $proj_dir/ds_overlay.srcs/sources_1/bd/$bd_name
    after 3000
}
close $bd_log
if {!$bd_ok} { error "create_bd_design failed; see $out_dir/create_bd_attempts.log" }

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]

# PYNQ-Z2 PS settings, one property at a time so a failure is visible.
set ps_cfg [dict create \
    PCW_CRYSTAL_PERIPHERAL_FREQMHZ      50 \
    PCW_APU_PERIPHERAL_FREQMHZ          650 \
    PCW_PRESET_BANK0_VOLTAGE            {LVCMOS 3.3V} \
    PCW_PRESET_BANK1_VOLTAGE            {LVCMOS 1.8V} \
    PCW_UIPARAM_DDR_PARTNO              {MT41K256M16 RE-125} \
    PCW_UIPARAM_DDR_BUS_WIDTH           {16 Bit} \
    PCW_UIPARAM_DDR_FREQ_MHZ            525 \
    PCW_QSPI_PERIPHERAL_ENABLE          1 \
    PCW_QSPI_GRP_SINGLE_SS_ENABLE       1 \
    PCW_ENET0_PERIPHERAL_ENABLE         1 \
    PCW_ENET0_ENET0_IO                  {MIO 16 .. 27} \
    PCW_ENET0_GRP_MDIO_ENABLE           1 \
    PCW_ENET0_GRP_MDIO_IO               {MIO 52 .. 53} \
    PCW_SD0_PERIPHERAL_ENABLE           1 \
    PCW_SD0_SD0_IO                      {MIO 40 .. 45} \
    PCW_UART0_PERIPHERAL_ENABLE         1 \
    PCW_UART0_UART0_IO                  {MIO 14 .. 15} \
    PCW_USB0_PERIPHERAL_ENABLE          1 \
    PCW_USB0_USB0_IO                    {MIO 28 .. 39} \
    PCW_GPIO_MIO_GPIO_ENABLE            1 \
    PCW_FPGA0_PERIPHERAL_FREQMHZ        100 \
    PCW_USE_M_AXI_GP0                   1 \
    PCW_USE_S_AXI_HP0                   1 \
]
set ps_log [open $out_dir/ps7_config.log w]
dict for {k v} $ps_cfg {
    if {[catch {set_property CONFIG.$k $v $ps} err]} {
        puts $ps_log "FAILED $k = $v : $err"
    } else {
        puts $ps_log "ok     $k = [get_property CONFIG.$k $ps]"
    }
}
close $ps_log

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" \
             Master "Disable" Slave "Disable"} $ps

# AXI DMA: memory-to-stream only. No scatter-gather: one 2,560-byte window per
# transfer is exactly what PYNQ's simple DMA driver does best.
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg              0 \
    CONFIG.c_include_mm2s            1 \
    CONFIG.c_include_s2mm            0 \
    CONFIG.c_m_axis_mm2s_tdata_width 32 \
    CONFIG.c_mm2s_burst_size         16 \
    CONFIG.c_sg_length_width         23 \
] $dma

# The verified core, via the wrapper, as a module reference.
set core [create_bd_cell -type module -reference ds_pynq_wrap ds_0]

# control plane: PS -> DMA registers, PS -> core registers
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins ds_0/s_axi]
# data plane: DMA reads DDR through HP0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/axi_dma_0/M_AXI_MM2S" Slave "/ps7/S_AXI_HP0" Clk "Auto"} \
    [get_bd_intf_pins ps7/S_AXI_HP0]
# the window itself
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins ds_0/s_axis]

# Anything the automation left unclocked goes to FCLK_CLK0 and its reset.
foreach p [get_bd_pins -quiet -of_objects [get_bd_cells {axi_dma_0 ds_0}] -filter {TYPE == clk}] {
    if {[llength [get_bd_nets -quiet -of_objects $p]] == 0} {
        connect_bd_net [get_bd_pins ps7/FCLK_CLK0] $p
    }
}
set rst_cell [lindex [get_bd_cells -quiet -filter {VLNV =~ *proc_sys_reset*}] 0]
if {[llength [get_bd_nets -quiet -of_objects [get_bd_pins ds_0/rst_n]]] == 0} {
    connect_bd_net [get_bd_pins $rst_cell/peripheral_aresetn] [get_bd_pins ds_0/rst_n]
}

assign_bd_address
# The core decodes 8 address bits (a 256-byte register map). Left to itself,
# assign_bd_address gave it a 512 MB window at 0x6000_0000 because the wrapper
# presents a 32-bit address -- and PYNQ maps an IP's whole range, so that is not
# harmless. Pin it to the smallest legal window at the conventional GP0 PL base.
set ds_seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]                 -filter {NAME =~ *ds_0*}]
set_property range  4K         $ds_seg
set_property offset 0x43C00000 $ds_seg
validate_bd_design
save_bd_design

# address map, for the record (PYNQ reads it from the .hwh; nothing hard-codes it)
set amap [open $out_dir/address_map.txt w]
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
    puts $amap [format "%-40s 0x%08X  range 0x%X" [get_property NAME $seg] \
        [get_property OFFSET $seg] [get_property RANGE $seg]]
}
close $amap

# ---------------------------------------------------------------------------
# synthesis -- in this process, from the repository root
# ---------------------------------------------------------------------------
# Global synthesis (no out-of-context IP runs) and synth_design called here
# rather than launch_runs: a run directory would change the working directory,
# and the core's $readmemh paths are relative to the repository root. That is
# what makes the bitstream's weights provably the exported .mem files.
set bd_file [get_files $bd_name.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
set bd_path [get_property NAME $bd_file]
set hwh_proj [glob -nocomplain $proj_dir/*.gen/sources_1/bd/$bd_name/hw_handoff/$bd_name.hwh]
close_project

# Synthesis runs in a fresh in-memory project, the documented non-project flow
# for a block design. The first attempt called synth_design inside the project
# session and it reported "module 'drivesentinel_wrapper' not found" although
# the wrapper had been generated and added -- the project's generated sources
# are not what an in-session synth_design reads. launch_runs would see them, but
# it synthesises inside a run directory, and the core's $readmemh paths are
# relative to the repository root. read_bd + explicit sources keeps both.
create_project -in_memory -part $part
set_property target_language Verilog [current_project]
read_verilog $rtl_files
read_verilog board/rtl/ds_pynq_wrap.v
read_bd $bd_path
set bd_file [get_files $bd_name.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
read_verilog $wrapper

synth_design -top ${bd_name}_wrapper -part $part -include_dirs [file normalize rtl]

# Did the weights load? $readmemh failing is the one failure that would make
# every verification number meaningless while looking like a working design --
# the build ID on the board would catch it, but that is too late to find out.
set bram_core [llength [get_cells -hier -quiet -filter {REF_NAME =~ RAMB* && NAME =~ *ds_0*}]]
puts "=== core block RAMs after synthesis: $bram_core ==="
report_utilization -file $out_dir/post_synth_util.rpt
report_utilization -hierarchical -file $out_dir/post_synth_util_hier.rpt

opt_design
place_design
phys_opt_design
route_design

report_utilization               -file $out_dir/post_route_util.rpt
report_utilization -hierarchical -file $out_dir/post_route_util_hier.rpt
report_timing_summary            -file $out_dir/post_route_timing.rpt
report_timing -max_paths 20 -nworst 20 -delay_type max -sort_by slack \
    -file $out_dir/post_route_worst_paths.rpt
report_clocks                    -file $out_dir/post_route_clocks.rpt
report_drc                       -file $out_dir/post_route_drc.rpt

set wns  [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs  [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
# FCLK_CLK0 is named clk_fpga_0 by the PS7's own constraints.
set clk  [get_clocks -quiet clk_fpga_0]
if {[llength $clk] == 0} { set clk [lindex [get_clocks] 0] }
set per  [get_property PERIOD $clk]
set met  [expr {$wns >= 0 && $whs >= 0}]

# ---------------------------------------------------------------------------
# bitstream -- only if the full system closed timing
# ---------------------------------------------------------------------------
set bit_written false
if {$met} {
    write_bitstream -force $board_dir/$bd_name.bit
    # The .hwh from the project-mode generation above: same block design, and
    # the one Vivado writes to the canonical hw_handoff location.
    set hwh $hwh_proj
    if {[llength $hwh] == 1} {
        file copy -force [lindex $hwh 0] $board_dir/$bd_name.hwh
        set bit_written true
    } else {
        puts "ERROR: hardware handoff (.hwh) not found; overlay incomplete"
    }
} else {
    puts "TIMING NOT MET (WNS $wns ns, WHS $whs ns): no bitstream written, by design"
}

set fh [open $out_dir/system.json w]
puts $fh "{"
puts $fh "  \"scope\": \"FULL SYSTEM: PS7 + AXI DMA + interconnects + reset + ds_0 (the core)\","
puts $fh "  \"not_comparable_with\": \"artifacts/rtl/synth/ (core only, out of context)\","
puts $fh "  \"part\": \"$part\","
puts $fh "  \"vivado\": \"[version -short]\","
puts $fh "  \"fabric_clock\": \"[get_property NAME $clk]\","
puts $fh "  \"target_period_ns\": $per,"
puts $fh "  \"wns_ns\": $wns,"
puts $fh "  \"whs_ns\": $whs,"
puts $fh "  \"timing_met\": [expr {$met ? "true" : "false"}],"
puts $fh "  \"bitstream_written\": $bit_written,"
puts $fh "  \"core_block_rams_after_synth\": $bram_core,"
puts $fh "  \"ran_on_hardware\": false,"
puts $fh "  \"reports\": \"$out_dir\""
puts $fh "}"
close $fh

puts "=== FULL SYSTEM: WNS $wns ns  WHS $whs ns  period $per ns  met=$met  bitstream=$bit_written ==="
