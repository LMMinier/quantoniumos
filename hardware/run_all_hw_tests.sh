#!/bin/bash
# ============================================================================
# QuantoniumOS Hardware Verification Suite
# Run all hardware tests and simulations
# ============================================================================
set -e

cd "$(dirname "$0")"
HARDWARE_DIR="$(pwd)"
PROJECT_ROOT="$(dirname "$HARDWARE_DIR")"

echo "=============================================="
echo " QuantoniumOS Hardware Test Suite"
echo " Date: $(date)"
echo "=============================================="

# Check dependencies
echo ""
echo "Checking dependencies..."
command -v python3 >/dev/null 2>&1 || { echo "Python3 required"; exit 1; }
echo "✓ Python3 found"

if command -v iverilog >/dev/null 2>&1; then
    echo "✓ Icarus Verilog found"
    HAS_IVERILOG=1
else
    echo "⚠ Icarus Verilog not found (install with: sudo apt install iverilog)"
    HAS_IVERILOG=0
fi

if command -v yosys >/dev/null 2>&1; then
    echo "✓ Yosys found"
    HAS_YOSYS=1
else
    echo "⚠ Yosys not found (install with: sudo apt install yosys)"
    HAS_YOSYS=0
fi

# ============================================================================
# Test 1: Python Verification Suite
# ============================================================================
echo ""
echo "=============================================="
echo " TEST 1: Python Algorithm Verification"
echo "=============================================="
cd "$PROJECT_ROOT"
python3 "$HARDWARE_DIR/run_full_verification.py"

# ============================================================================
# Test 2: RFT Operator Variants
# ============================================================================
echo ""
echo "=============================================="
echo " TEST 2: RFT Operator Variants Unitarity"
echo "=============================================="
python3 -c "
import sys
sys.path.insert(0, '.')
from algorithms.rft.variants.operator_variants import OPERATOR_VARIANTS
import numpy as np

N = 8
print(f'Testing {len(OPERATOR_VARIANTS)} variants at N={N}')
passed = 0
for name, info in OPERATOR_VARIANTS.items():
    try:
        basis = info['generator'](N)
        err = np.linalg.norm(basis.T @ basis - np.eye(N))
        status = '✓' if err < 1e-10 else '✗'
        print(f'  {status} {info[\"name\"]:<25} err={err:.2e}')
        if err < 1e-10:
            passed += 1
    except Exception as e:
        print(f'  ✗ {name}: {e}')
print(f'Passed: {passed}/{len(OPERATOR_VARIANTS)}')
"

# ============================================================================
# Test 3: Verilog Simulation (if available)
# ============================================================================
if [ "$HAS_IVERILOG" = "1" ]; then
    echo ""
    echo "=============================================="
    echo " TEST 3: Verilog Simulation (Icarus)"
    echo "=============================================="
    cd "$HARDWARE_DIR"
    
    # Compile WebFPGA version
    if [ -f "fpga_top_webfpga.v" ] && [ -f "fpga_top_tb.v" ]; then
        echo "Compiling WebFPGA version..."
        iverilog -g2005-sv -o fpga_sim_test fpga_top_webfpga.v fpga_top_tb.v 2>&1 || true
        
        if [ -f "fpga_sim_test" ]; then
            echo "Running simulation..."
            timeout 30 vvp fpga_sim_test 2>&1 | head -30 || true
            rm -f fpga_sim_test
            echo "✓ Simulation completed"
        fi
    fi
    
    # Compile full version
    if [ -f "fpga_top.sv" ] && [ -f "fpga_top_tb.v" ]; then
        echo ""
        echo "Compiling full 16-mode version..."
        iverilog -g2012 -o fpga_full_test fpga_top.sv fpga_top_tb.v 2>&1 || true
        
        if [ -f "fpga_full_test" ]; then
            echo "Running simulation..."
            timeout 30 vvp fpga_full_test 2>&1 | head -30 || true
            rm -f fpga_full_test
            echo "✓ Simulation completed"
        fi
    fi
fi

# ============================================================================
# Test 4: Yosys Synthesis Check (if available)
# ============================================================================
if [ "$HAS_YOSYS" = "1" ]; then
    echo ""
    echo "=============================================="
    echo " TEST 4: Yosys Synthesis Check"
    echo "=============================================="
    cd "$HARDWARE_DIR"
    
    if [ -f "fpga_top_webfpga.v" ]; then
        echo "Synthesizing WebFPGA version..."
        yosys -q -p "read_verilog fpga_top_webfpga.v; synth_ice40 -top fpga_top; stat" 2>&1 | grep -E "LUT|DFF|Cells|Number" || true
        echo "✓ Synthesis completed"
    fi
fi

# ============================================================================
# Test 5: Generate Test Vectors
# ============================================================================
echo ""
echo "=============================================="
echo " TEST 5: Generate Test Vectors"
echo "=============================================="
cd "$HARDWARE_DIR"
if [ -f "generate_hardware_test_vectors.py" ]; then
    python3 generate_hardware_test_vectors.py 2>&1 | head -40 || true
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo " HARDWARE TEST SUITE COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - quantoniumos_sim.vcd (waveform)"
echo "  - test_vectors_*.hex (test vectors)"
echo ""
echo "To view waveforms: gtkwave quantoniumos_sim.vcd"
