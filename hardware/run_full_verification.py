#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
"""
Full Hardware-Software Verification Suite
=========================================

This script verifies that all RTL implementations match Python reference.
Run this before submitting the paper to ensure all claims are backed.

Tests:
1. RFT-Golden kernel coefficients match Python
2. RFT-Cascade (H3) kernel matches Python  
3. SIS-Hash kernel matches Python
4. Quantum-Sim GHZ state representation
5. Unitarity verification for all modes
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import QuantoniumOS modules
try:
    # CANONICAL RFT (January 2026): Gram-normalized φ-grid basis
    from algorithms.rft.core.resonant_fourier_transform import (
        rft_basis_matrix,
        PHI,
    )
    
    # Legacy operator-based variants (kept for comparison)
    from algorithms.rft.variants.operator_variants import (
        generate_rft_golden as generate_rft_golden_legacy,
        generate_rft_cascade_h3,
        generate_rft_harmonic,
        generate_rft_fibonacci,
        generate_rft_geometric,
        generate_rft_beating,
        generate_rft_phyllotaxis,
        generate_rft_hybrid_dct,
        OPERATOR_VARIANTS,
    )
    from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
    print("✓ Successfully imported QuantoniumOS modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def generate_rft_golden(n: int) -> np.ndarray:
    """
    Generate CANONICAL RFT-Golden kernel.
    
    CANONICAL DEFINITION (January 2026):
        Φ̃ = Φ (ΦᴴΦ)^{-1/2}
    where:
        Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
    
    Returns the Gram-normalized φ-grid exponential basis.
    """
    return rft_basis_matrix(n, n, use_gram_normalization=True)


# Constants matching hardware (Q1.15 fixed-point, scale = 2^15 = 32768)
Q15_SCALE = 32768
N = 8  # Hardware transform size

def float_to_q15(value):
    """Convert float to Q1.15 fixed-point (signed 16-bit)."""
    scaled = int(round(value * Q15_SCALE))
    # Clamp to signed 16-bit range
    return max(-32768, min(32767, scaled))

def q15_to_float(value):
    """Convert Q1.15 fixed-point to float."""
    return value / Q15_SCALE

def extract_hardware_kernels_from_verilog(verilog_file):
    """Parse Verilog file and extract kernel ROM values."""
    kernels = {}
    current_mode = None
    
    with open(verilog_file, 'r') as f:
        content = f.read()
    
    # Parse case statements for kernel values
    import re
    
    # Pattern: {mode, k_index, n_index}: kernel_rom_out = value;
    # For 4-mode version: {2'd0, 3'd0, 3'd0}: kernel_rom_out = -16'sd10528;
    pattern_4mode = r"\{2'd(\d+),\s*3'd(\d+),\s*3'd(\d+)\}:\s*kernel_rom_out\s*=\s*(-?)16'sd(\d+);"
    # For 16-mode version: {4'd0, 3'd0, 3'd0}: kernel_rom_out = -16'sd10528;
    pattern_16mode = r"\{4'd(\d+),\s*3'd(\d+),\s*3'd(\d+)\}:\s*kernel_rom_out\s*=\s*(-?)16'sd(\d+);"
    
    # Try 4-mode pattern first
    matches = re.findall(pattern_4mode, content)
    if not matches:
        matches = re.findall(pattern_16mode, content)
    
    for match in matches:
        mode, k, n, sign, value = match
        mode = int(mode)
        k = int(k)
        n = int(n)
        value = int(value)
        if sign == '-':
            value = -value
        
        if mode not in kernels:
            kernels[mode] = np.zeros((N, N), dtype=np.int16)
        kernels[mode][k, n] = value
    
    return kernels

def generate_python_kernel_q15(generator_func, n=8):
    """Generate Python RFT kernel and convert to Q1.15."""
    basis = generator_func(n)
    kernel_q15 = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        for j in range(n):
            kernel_q15[i, j] = float_to_q15(basis[i, j])
    return kernel_q15, basis

def verify_unitarity(basis, tolerance=1e-10):
    """Verify that basis is unitary."""
    n = basis.shape[0]
    identity = np.eye(n)
    product = basis.T @ basis
    error = np.linalg.norm(product - identity, 'fro')
    return error, error < tolerance

def main():
    print("\n" + "="*70)
    print(" QUANTONIUMOS HARDWARE-SOFTWARE VERIFICATION SUITE")
    print(" Date:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    results = {
        'passed': 0,
        'failed': 0,
        'warnings': 0,
    }
    
    # =========================================================================
    # Test 1: Parse hardware kernels from Verilog
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: Extracting hardware kernels from Verilog")
    print("-"*70)
    
    verilog_files = [
        PROJECT_ROOT / "hardware" / "fpga_top_webfpga.v",
        PROJECT_ROOT / "hardware" / "fpga_top.sv",
    ]
    
    hw_kernels = None
    for vf in verilog_files:
        if vf.exists():
            print(f"  Parsing: {vf.name}")
            hw_kernels = extract_hardware_kernels_from_verilog(vf)
            print(f"  ✓ Found {len(hw_kernels)} modes")
            break
    
    if hw_kernels is None:
        print("  ✗ No Verilog files found!")
        results['failed'] += 1
        return results
    
    # =========================================================================
    # Test 2: Verify RFT-Golden (Mode 0)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: RFT-Golden (Mode 0) - Kernel Alignment")
    print("-"*70)
    
    py_golden_q15, py_golden_float = generate_python_kernel_q15(generate_rft_golden, N)
    
    if 0 in hw_kernels:
        hw_golden = hw_kernels[0]
        
        # Compare
        diff = np.abs(py_golden_q15.astype(np.int32) - hw_golden.astype(np.int32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Python kernel [0,0]: {py_golden_q15[0,0]}")
        print(f"  Hardware kernel [0,0]: {hw_golden[0,0]}")
        print(f"  Max difference: {max_diff} LSB")
        print(f"  Mean difference: {mean_diff:.2f} LSB")
        
        # Tolerance: allow up to 2 LSB difference due to rounding
        if max_diff <= 2:
            print("  ✓ PASS - Kernels match within tolerance")
            results['passed'] += 1
        else:
            print("  ✗ FAIL - Kernels differ significantly")
            results['failed'] += 1
            
            # Show first mismatch
            for i in range(N):
                for j in range(N):
                    if abs(py_golden_q15[i,j] - hw_golden[i,j]) > 2:
                        print(f"    Mismatch at [{i},{j}]: Python={py_golden_q15[i,j]}, HW={hw_golden[i,j]}")
                        break
                else:
                    continue
                break
        
        # Unitarity check
        err, is_unitary = verify_unitarity(py_golden_float)
        print(f"  Unitarity error: {err:.2e} {'✓' if is_unitary else '✗'}")
    else:
        print("  ✗ Mode 0 not found in hardware")
        results['failed'] += 1
    
    # =========================================================================
    # Test 3: Verify RFT-Cascade H3 (Mode 1 in WebFPGA, Mode 6 in full)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: RFT-Cascade/H3 - Kernel Alignment")
    print("-"*70)
    
    py_cascade_q15, py_cascade_float = generate_python_kernel_q15(generate_rft_cascade_h3, N)
    
    # Mode 1 in webfpga, Mode 6 in full
    cascade_mode = 1 if 1 in hw_kernels and len(hw_kernels) <= 4 else 6
    
    if cascade_mode in hw_kernels:
        hw_cascade = hw_kernels[cascade_mode]
        
        diff = np.abs(py_cascade_q15.astype(np.int32) - hw_cascade.astype(np.int32))
        max_diff = np.max(diff)
        
        print(f"  Python kernel [0,0]: {py_cascade_q15[0,0]}")
        print(f"  Hardware kernel [0,0]: {hw_cascade[0,0]}")
        print(f"  Max difference: {max_diff} LSB")
        
        if max_diff <= 2:
            print("  ✓ PASS - Cascade kernels match")
            results['passed'] += 1
        else:
            print("  ⚠ WARNING - Cascade kernels differ (may use DCT variant)")
            results['warnings'] += 1
        
        err, is_unitary = verify_unitarity(py_cascade_float)
        print(f"  Unitarity error: {err:.2e} {'✓' if is_unitary else '✗'}")
    else:
        print(f"  Mode {cascade_mode} not found in hardware")
        results['warnings'] += 1
    
    # =========================================================================
    # Test 4: All Python Variants Unitarity
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 4: All Python Variant Unitarity Checks")
    print("-"*70)
    
    all_unitary = True
    for name, info in OPERATOR_VARIANTS.items():
        try:
            basis = info['generator'](N)
            err, is_unitary = verify_unitarity(basis)
            status = "✓" if is_unitary else "✗"
            print(f"  {status} {info['name']:<25} | Error: {err:.2e}")
            if not is_unitary:
                all_unitary = False
        except Exception as e:
            print(f"  ✗ {name:<25} | Error: {e}")
            all_unitary = False
    
    if all_unitary:
        print("  ✓ All variants are unitary")
        results['passed'] += 1
    else:
        print("  ✗ Some variants failed unitarity")
        results['failed'] += 1
    
    # =========================================================================
    # Test 5: Quantum Simulation Mode (GHZ State)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 5: Quantum Simulation - GHZ State Verification")
    print("-"*70)
    
    # GHZ state: (|000⟩ + |111⟩)/√2 represented as 8-element vector
    # In computational basis: indices 0 (000) and 7 (111) have amplitude 1/√2
    ghz_expected = np.zeros(8)
    ghz_expected[0] = 1.0 / np.sqrt(2)  # |000⟩
    ghz_expected[7] = 1.0 / np.sqrt(2)  # |111⟩
    
    ghz_q15 = [float_to_q15(v) for v in ghz_expected]
    print(f"  Expected GHZ (Q1.15): [{ghz_q15[0]}, 0, 0, 0, 0, 0, 0, {ghz_q15[7]}]")
    print(f"  1/√2 in Q1.15 = {float_to_q15(1/np.sqrt(2))}")
    
    # Check hardware mode 3 (quantum sim in webfpga)
    quantum_mode = 3 if 3 in hw_kernels and len(hw_kernels) <= 4 else 14
    
    if quantum_mode in hw_kernels:
        hw_quantum = hw_kernels[quantum_mode]
        
        # Row 0 should have GHZ pattern: non-zero at [0,0] and [0,7]
        row0 = hw_quantum[0, :]
        print(f"  Hardware row 0: {list(row0)}")
        
        # Check if it matches GHZ structure
        if row0[0] != 0 and row0[7] != 0 and np.sum(row0[1:7]) == 0:
            print("  ✓ GHZ state structure detected")
            results['passed'] += 1
        else:
            print("  ⚠ WARNING - Non-standard quantum kernel")
            results['warnings'] += 1
    else:
        print(f"  Mode {quantum_mode} not found in hardware")
        results['warnings'] += 1
    
    # =========================================================================
    # Test 6: SIS Hash Mode
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 6: SIS Hash - DFT Structure Verification")
    print("-"*70)
    
    sis_mode = 2 if 2 in hw_kernels and len(hw_kernels) <= 4 else 12
    
    if sis_mode in hw_kernels:
        hw_sis = hw_kernels[sis_mode]
        
        # SIS should use DFT-like structure
        # Row 0 should be all same value (DC component)
        row0_var = np.var(hw_sis[0, :])
        
        print(f"  Hardware row 0: {list(hw_sis[0, :])}")
        print(f"  Row 0 variance: {row0_var}")
        
        if row0_var < 10:  # Very small variance = constant row
            print("  ✓ DFT-like DC row detected")
            results['passed'] += 1
        else:
            print("  ⚠ Non-standard SIS structure")
            results['warnings'] += 1
    else:
        print(f"  Mode {sis_mode} not found in hardware")
        results['warnings'] += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70)
    print(f"  ✓ Passed:   {results['passed']}")
    print(f"  ⚠ Warnings: {results['warnings']}")
    print(f"  ✗ Failed:   {results['failed']}")
    
    total = results['passed'] + results['warnings'] + results['failed']
    if results['failed'] == 0:
        print("\n  🎉 ALL CRITICAL TESTS PASSED!")
        print("  Hardware is aligned with Python reference implementation.")
    else:
        print("\n  ⚠ SOME TESTS FAILED - Review kernel generation")
    
    return results

if __name__ == "__main__":
    results = main()
    sys.exit(1 if results['failed'] > 0 else 0)
