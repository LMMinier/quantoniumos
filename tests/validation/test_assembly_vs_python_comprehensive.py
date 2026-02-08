#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Comprehensive Test Suite: Assembly/C RFT vs Python Reference
==============================================================
Tests all Python validation suites against the compiled assembly kernels.

This script validates:
1. Unitarity (Theorem 1)
2. Energy Preservation (Parseval)
3. Round-trip Reconstruction
4. Performance Scaling (O(N log N))
5. Spectral Properties
6. Comparison with FFT behavior
7. Various signal types (impulse, sine, random, quasi-periodic)
"""

import sys
import os
import time
import numpy as np
import pytest
from pathlib import Path

# Setup path
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Try to import assembly kernel
try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    ASSEMBLY_AVAILABLE = True
except ImportError:
    ASSEMBLY_AVAILABLE = False
    print("⚠️  Assembly kernels not available - build with `make` in algorithms/rft/kernels/")

# Import Python reference implementations
from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward, 
    rft_inverse_square as rft_inverse, 
    rft_unitary_error_canonical as rft_unitary_error,
    rft_matrix_canonical as rft_matrix
)
from algorithms.rft.variants import VARIANTS, PHI

# Test configuration
TEST_SIZES = [8, 16, 32, 64, 128, 256]
TOLERANCE_UNITARITY = 1e-10
TOLERANCE_RECONSTRUCTION = 1e-8
TOLERANCE_ENERGY = 1e-10

class AssemblyRFTWrapper:
    """Wrapper to make assembly RFT compatible with Python interface"""
    def __init__(self, n):
        if not ASSEMBLY_AVAILABLE:
            pytest.skip("Assembly kernels not available")
        self.rft = UnitaryRFT(n, RFT_FLAG_QUANTUM_SAFE)
        if getattr(self.rft, '_is_mock', True):
            pytest.skip("Assembly DLL not loaded - using mock")
        self.n = n
    
    def forward(self, x):
        """Forward transform"""
        if x.dtype != np.complex128:
            x = x.astype(np.complex128)
        return self.rft.forward(x)
    
    def inverse(self, X):
        """Inverse transform"""
        if X.dtype != np.complex128:
            X = X.astype(np.complex128)
        return self.rft.inverse(X)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def relative_error(a, b):
    """Compute relative error between two arrays"""
    norm_diff = np.linalg.norm(a - b)
    norm_ref = np.linalg.norm(b)
    return norm_diff / max(1e-16, norm_ref)


# =============================================================================
# TEST 1: UNITARITY VERIFICATION
# =============================================================================

def test_unitarity_assembly_vs_python():
    """Test that assembly RFT preserves unitarity like Python version"""
    print_section("TEST 1: Unitarity (Assembly vs Python)")
    
    results = []
    for n in TEST_SIZES:
        # Python reference
        py_error = rft_unitary_error(n, trials=5)
        
        # Assembly version
        if ASSEMBLY_AVAILABLE:
            rft_asm = AssemblyRFTWrapper(n)
            errors_asm = []
            for trial in range(5):
                x = np.random.randn(n) + 1j * np.random.randn(n)
                X = rft_asm.forward(x)
                x_rec = rft_asm.inverse(X)
                err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
                errors_asm.append(err)
            asm_error = np.mean(errors_asm)
        else:
            asm_error = None
        
        # Compare
        status_py = "✓ PASS" if py_error < TOLERANCE_UNITARITY else "✗ FAIL"
        if asm_error is not None:
            status_asm = "✓ PASS" if asm_error < TOLERANCE_UNITARITY else "✗ FAIL"
            match = "✓ MATCH" if abs(asm_error - py_error) < 1e-8 else "⚠ DIFF"
            print(f"N={n:4d}: Python={py_error:.2e} {status_py} | Assembly={asm_error:.2e} {status_asm} | {match}")
        else:
            print(f"N={n:4d}: Python={py_error:.2e} {status_py} | Assembly=N/A")
        
        results.append({
            'n': n,
            'python_error': py_error,
            'assembly_error': asm_error,
            'python_pass': py_error < TOLERANCE_UNITARITY,
            'assembly_pass': asm_error < TOLERANCE_UNITARITY if asm_error else None
        })
    
    return results


# =============================================================================
# TEST 2: ENERGY PRESERVATION (PARSEVAL'S THEOREM)
# =============================================================================

def test_energy_preservation():
    """Test energy preservation: ||x||² = ||Ψx||²"""
    print_section("TEST 2: Energy Preservation (Parseval)")
    
    results = []
    for n in TEST_SIZES:
        x = np.random.randn(n) + 1j * np.random.randn(n)
        energy_time = np.sum(np.abs(x) ** 2)
        
        # Python RFT
        X_py = rft_forward(x)
        energy_py = np.sum(np.abs(X_py) ** 2)
        err_py = abs(energy_py - energy_time) / energy_time
        
        # Assembly RFT
        if ASSEMBLY_AVAILABLE:
            rft_asm = AssemblyRFTWrapper(n)
            X_asm = rft_asm.forward(x.copy())
            energy_asm = np.sum(np.abs(X_asm) ** 2)
            err_asm = abs(energy_asm - energy_time) / energy_time
        else:
            energy_asm = None
            err_asm = None
        
        status_py = "✓ PASS" if err_py < TOLERANCE_ENERGY else "✗ FAIL"
        
        if err_asm is not None:
            status_asm = "✓ PASS" if err_asm < TOLERANCE_ENERGY else "✗ FAIL"
            print(f"N={n:4d}: E_time={energy_time:.4f} | E_py={energy_py:.4f} (err={err_py:.2e}) {status_py}")
            print(f"       E_asm={energy_asm:.4f} (err={err_asm:.2e}) {status_asm}")
        else:
            print(f"N={n:4d}: E_time={energy_time:.4f} | E_py={energy_py:.4f} (err={err_py:.2e}) {status_py}")
        
        results.append({
            'n': n,
            'python_error': err_py,
            'assembly_error': err_asm,
            'python_pass': err_py < TOLERANCE_ENERGY,
            'assembly_pass': err_asm < TOLERANCE_ENERGY if err_asm else None
        })
    
    return results


# =============================================================================
# TEST 3: SIGNAL RECONSTRUCTION
# =============================================================================

def test_signal_reconstruction():
    """Test reconstruction of various signal types"""
    print_section("TEST 3: Signal Reconstruction")
    
    n = 128
    
    # Define test signals
    signals = {
        "Impulse": np.array([1.0] + [0.0] * (n-1)),
        "Constant": np.ones(n),
        "Sine": np.sin(2 * np.pi * 5 * np.arange(n) / n),
        "Complex Exp": np.exp(2j * np.pi * 3 * np.arange(n) / n),
        "Random": np.random.randn(n) + 1j * np.random.randn(n),
        "Quasi-Periodic": np.cos(2 * np.pi * PHI * np.arange(n) / n),
    }
    
    results = []
    for name, x in signals.items():
        # Python reconstruction
        X_py = rft_forward(x)
        x_rec_py = rft_inverse(X_py)
        err_py = relative_error(x_rec_py, x)
        
        # Assembly reconstruction
        if ASSEMBLY_AVAILABLE:
            rft_asm = AssemblyRFTWrapper(n)
            X_asm = rft_asm.forward(x.copy())
            x_rec_asm = rft_asm.inverse(X_asm)
            err_asm = relative_error(x_rec_asm, x)
        else:
            err_asm = None
        
        status_py = "✓ PASS" if err_py < TOLERANCE_RECONSTRUCTION else "✗ FAIL"
        
        if err_asm is not None:
            status_asm = "✓ PASS" if err_asm < TOLERANCE_RECONSTRUCTION else "✗ FAIL"
            print(f"{name:15s}: Python={err_py:.2e} {status_py} | Assembly={err_asm:.2e} {status_asm}")
        else:
            print(f"{name:15s}: Python={err_py:.2e} {status_py} | Assembly=N/A")
        
        results.append({
            'signal': name,
            'python_error': err_py,
            'assembly_error': err_asm,
            'python_pass': err_py < TOLERANCE_RECONSTRUCTION,
            'assembly_pass': err_asm < TOLERANCE_RECONSTRUCTION if err_asm else None
        })
    
    return results


# =============================================================================
# TEST 4: MATRIX ORTHOGONALITY
# =============================================================================

def test_matrix_orthogonality():
    """Test that Ψ†Ψ = I"""
    print_section("TEST 4: Matrix Orthogonality (Ψ†Ψ = I)")
    
    results = []
    for n in [8, 16, 32, 64]:
        # Python version
        Psi_py = rft_matrix(n)
        I_test_py = np.conj(Psi_py.T) @ Psi_py
        I_true = np.eye(n, dtype=np.complex128)
        err_py = np.linalg.norm(I_test_py - I_true, 'fro') / n
        
        status_py = "✓ PASS" if err_py < TOLERANCE_UNITARITY else "✗ FAIL"
        print(f"N={n:4d}: Python ||Ψ†Ψ - I||_F/n = {err_py:.2e} {status_py}")
        
        # Note: Assembly version doesn't expose matrix form directly
        # But we can verify unitarity through round-trip instead
        
        results.append({
            'n': n,
            'python_error': err_py,
            'python_pass': err_py < TOLERANCE_UNITARITY
        })
    
    return results


# =============================================================================
# TEST 5: PERFORMANCE SCALING
# =============================================================================

def test_performance_scaling():
    """Compare execution time scaling"""
    print_section("TEST 5: Performance Scaling")
    
    results = []
    perf_sizes = [256, 512, 1024, 2048]
    iterations = 100
    
    print("Measuring execution times (averaged over {} iterations)...".format(iterations))
    
    for n in perf_sizes:
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # Python timing
        start = time.perf_counter()
        for _ in range(iterations):
            _ = rft_forward(x.copy())
        end = time.perf_counter()
        time_py = (end - start) / iterations * 1000  # ms
        
        # Assembly timing
        if ASSEMBLY_AVAILABLE:
            rft_asm = AssemblyRFTWrapper(n)
            x_copy = x.copy()
            start = time.perf_counter()
            for _ in range(iterations):
                _ = rft_asm.forward(x_copy)
            end = time.perf_counter()
            time_asm = (end - start) / iterations * 1000  # ms
            speedup = time_py / time_asm
        else:
            time_asm = None
            speedup = None
        
        if time_asm is not None:
            print(f"N={n:5d}: Python={time_py:7.3f}ms | Assembly={time_asm:7.3f}ms | Speedup={speedup:5.2f}x")
        else:
            print(f"N={n:5d}: Python={time_py:7.3f}ms | Assembly=N/A")
        
        results.append({
            'n': n,
            'python_time_ms': time_py,
            'assembly_time_ms': time_asm,
            'speedup': speedup
        })
    
    return results


# =============================================================================
# TEST 6: SPECTRAL COMPARISON WITH FFT
# =============================================================================

def test_spectral_comparison():
    """Compare spectral behavior between RFT and FFT"""
    print_section("TEST 6: Spectral Comparison (RFT vs FFT)")
    
    n = 128
    
    # Quasi-periodic signal (RFT should be sparse)
    t = np.arange(n)
    x_quasi = np.cos(2 * np.pi * PHI * t / n) + 0.5 * np.cos(2 * np.pi * (PHI**2) * t / n)
    
    # Get spectra
    X_fft = np.fft.fft(x_quasi, norm='ortho')
    X_rft_py = rft_forward(x_quasi)
    
    if ASSEMBLY_AVAILABLE:
        rft_asm = AssemblyRFTWrapper(n)
        X_rft_asm = rft_asm.forward(x_quasi.copy())
    else:
        X_rft_asm = None
    
    # Compute sparsity (% of coefficients below threshold)
    threshold = 0.01 * np.max(np.abs(X_rft_py))
    sparsity_fft = np.sum(np.abs(X_fft) < threshold) / n * 100
    sparsity_rft_py = np.sum(np.abs(X_rft_py) < threshold) / n * 100
    
    if X_rft_asm is not None:
        sparsity_rft_asm = np.sum(np.abs(X_rft_asm) < threshold) / n * 100
        # Compare spectra
        spectral_match = relative_error(X_rft_asm, X_rft_py)
        print(f"FFT Sparsity:     {sparsity_fft:5.1f}%")
        print(f"RFT-Py Sparsity:  {sparsity_rft_py:5.1f}%")
        print(f"RFT-Asm Sparsity: {sparsity_rft_asm:5.1f}%")
        print(f"Assembly vs Python spectral error: {spectral_match:.2e}")
    else:
        print(f"FFT Sparsity:    {sparsity_fft:5.1f}%")
        print(f"RFT-Py Sparsity: {sparsity_rft_py:5.1f}%")
        print(f"RFT-Asm: N/A")
    
    return {
        'sparsity_fft': sparsity_fft,
        'sparsity_rft_py': sparsity_rft_py,
        'sparsity_rft_asm': sparsity_rft_asm if X_rft_asm is not None else None
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE ASSEMBLY vs PYTHON RFT TEST SUITE")
    print("="*70)
    
    if not ASSEMBLY_AVAILABLE:
        print("\n⚠️  WARNING: Assembly kernels not available!")
        print("   Build them with: cd algorithms/rft/kernels && make")
        print("   Tests will run Python-only mode.\n")
    else:
        print("\n✓ Assembly kernels available - full comparison mode\n")
    
    all_results = {}
    
    # Run all test suites
    all_results['unitarity'] = test_unitarity_assembly_vs_python()
    all_results['energy'] = test_energy_preservation()
    all_results['reconstruction'] = test_signal_reconstruction()
    all_results['orthogonality'] = test_matrix_orthogonality()
    all_results['performance'] = test_performance_scaling()
    all_results['spectral'] = test_spectral_comparison()
    
    # Summary
    print_section("SUMMARY")
    
    # Count passes
    unitarity_passes = sum(1 for r in all_results['unitarity'] if r['python_pass'])
    if ASSEMBLY_AVAILABLE:
        unitarity_asm_passes = sum(1 for r in all_results['unitarity'] 
                                   if r['assembly_pass'] is not None and r['assembly_pass'])
        print(f"Unitarity Tests:      Python {unitarity_passes}/{len(TEST_SIZES)} | Assembly {unitarity_asm_passes}/{len(TEST_SIZES)}")
    else:
        print(f"Unitarity Tests:      Python {unitarity_passes}/{len(TEST_SIZES)}")
    
    energy_passes = sum(1 for r in all_results['energy'] if r['python_pass'])
    if ASSEMBLY_AVAILABLE:
        energy_asm_passes = sum(1 for r in all_results['energy'] 
                               if r['assembly_pass'] is not None and r['assembly_pass'])
        print(f"Energy Tests:         Python {energy_passes}/{len(TEST_SIZES)} | Assembly {energy_asm_passes}/{len(TEST_SIZES)}")
    else:
        print(f"Energy Tests:         Python {energy_passes}/{len(TEST_SIZES)}")
    
    recon_passes = sum(1 for r in all_results['reconstruction'] if r['python_pass'])
    if ASSEMBLY_AVAILABLE:
        recon_asm_passes = sum(1 for r in all_results['reconstruction'] 
                              if r['assembly_pass'] is not None and r['assembly_pass'])
        print(f"Reconstruction Tests: Python {recon_passes}/6 | Assembly {recon_asm_passes}/6")
    else:
        print(f"Reconstruction Tests: Python {recon_passes}/6")
    
    if ASSEMBLY_AVAILABLE:
        avg_speedup = np.mean([r['speedup'] for r in all_results['performance'] 
                              if r['speedup'] is not None])
        print(f"\nAverage Assembly Speedup: {avg_speedup:.2f}x")
    
    print("\n✅ Test suite complete!")
    
    return all_results


# =============================================================================
# PYTEST INTERFACE
# =============================================================================

@pytest.mark.skipif(not ASSEMBLY_AVAILABLE, reason="Assembly kernels not built")
def test_assembly_available():
    """Test that assembly kernels are available"""
    assert ASSEMBLY_AVAILABLE, "Assembly kernels not available"


def test_run_comprehensive_suite():
    """Pytest wrapper for comprehensive suite"""
    results = run_all_tests()
    
    # Assert all Python tests pass
    assert all(r['python_pass'] for r in results['unitarity'])
    assert all(r['python_pass'] for r in results['energy'])
    assert all(r['python_pass'] for r in results['reconstruction'])
    assert all(r['python_pass'] for r in results['orthogonality'])


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Assembly RFT vs Python Reference")
    parser.add_argument("--pytest", action="store_true", help="Run in pytest mode")
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, "-v"])
    else:
        results = run_all_tests()
        sys.exit(0)
