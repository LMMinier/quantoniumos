#!/bin/bash
echo "=============================================="
echo "RFTPU FPGA Synthesis Report"
echo "=============================================="
echo ""
echo "Target: iCE40UP5K (WebFPGA compatible)"
echo "Date: $(date)"
echo ""

yosys -p "
read_verilog -sv fpga_top.sv
synth_ice40 -top fpga_top -device up5k
stat
" 2>&1 | tee synth_report.txt | grep -E "^[0-9]|Number|SB_|Error|Warning|cells:|LUT|DFF|BRAM|Estimated"

echo ""
echo "=============================================="
echo "Resource Summary:"
grep -A 20 "=== fpga_top ===" synth_report.txt | tail -18
echo ""
echo "Full report saved to: synth_report.txt"
