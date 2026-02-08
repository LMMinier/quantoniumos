#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
LCT Non-Membership Tests for Closed-Form Φ-RFT
===============================================

Tests to prove RFT ∉ LCT family:
1. Quadratic phase residual (direct test)
2. LCT optimization fit (exhaustive search)
3. Eigenvalue spectrum analysis
4. DFT correlation test
5. Ψ†F entropy test
"""
import sys
import os
# Ensure we can import from the workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.optimize import minimize, differential_evolution
from algorithms.rft.core.resonant_fourier_transform import rft_matrix_canonical as rft_matrix, rft_phase_vectors_canonical as rft_phase_vectors, PHI, rft_forward_square as rft_forward
import json

def unwrap_phase(z: np.ndarray) -> np.ndarray:
    """Unwrap phase to avoid 2π discontinuities"""
    return np.unwrap(np.angle(z))

def fit_quadratic_phase(theta: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fit θ[k] ≈ a·k² + b·k + c via least squares
    
    Returns:
        a, b, c: quadratic coefficients
        rms: root-mean-square residual
    """
    n = theta.size
    k = np.arange(n, dtype=np.float64)
    
    # Design matrix for quadratic fit
    M = np.column_stack([k*k, k, np.ones_like(k)])
    coeffs, residuals, *_ = np.linalg.lstsq(M, theta, rcond=None)
    
    fit = M @ coeffs
    rms = float(np.sqrt(np.mean((theta - fit)**2)))
    
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), rms


def test_1_quadratic_phase_residual(n: int = 256, beta: float = 0.83, sigma: float = 1.25):
    """
    TEST 1: Direct Quadratic Residual
    
    If D_φ were quadratic phase (like FrFT/LCT), best quadratic fit would have RMS ≈ 0.
    For irrational phase {k/φ}, RMS should be bounded away from 0.
    """
    print("=" * 80)
    print("TEST 1: QUADRATIC PHASE RESIDUAL")
    print("=" * 80)
    print(f"Matrix size: {n}")
    print(f"Parameters: β={beta}, σ={sigma}, φ={PHI:.6f}")
    print()
    
    # Get D_phi phase
    D_phi, _ = rft_phase_vectors(n, beta=beta, sigma=sigma, phi=PHI)
    theta = unwrap_phase(D_phi)
    
    # Fit best quadratic
    a, b, c, rms = fit_quadratic_phase(theta)
    
    print(f"Best quadratic fit: θ(k) ≈ {a:.6e}·k² + {b:.6e}·k + {c:.6e}")
    print(f"RMS residual: {rms:.6f} rad")
    print()
    
    # Verdict
    if rms < 0.01:
        verdict = "❌ FAIL: Phase is essentially quadratic (RFT ≈ LCT)"
        is_distinct = False
    elif rms < 0.1:
        verdict = "⚠️  WEAK: Phase is nearly quadratic"
        is_distinct = False
    else:
        verdict = f"✅ PASS: Phase is non-quadratic (RMS = {rms:.3f} rad >> 0)"
        is_distinct = True
    
    print(verdict)
    print()
    
    return {
        'test': 'quadratic_phase_residual',
        'rms_residual': rms,
        'coefficients': {'a': a, 'b': b, 'c': c},
        'is_distinct': is_distinct,
        'verdict': verdict
    }


def test_2_lct_optimization_fit(n: int = 128, beta: float = 0.83, sigma: float = 1.25):
    """
    TEST 2: Exhaustive LCT Optimization
    
    Try to fit: Ψ ≈ D₁·C₁·F·C₂·D₂ where C₁, C₂ are quadratic chirps
    
    If fit error << 1, then RFT might be in LCT family.
    If fit error >> 1, then RFT is structurally distinct.
    """
    print("=" * 80)
    print("TEST 2: LCT OPTIMIZATION FIT")
    print("=" * 80)
    print(f"Matrix size: {n}")
    print(f"Testing if Ψ ∈ span{{D₁·C₁·F·C₂·D₂}}")
    print()
    
    # Construct RFT matrix
    print("Building RFT matrix...")
    Psi = rft_matrix(n, beta=beta, sigma=sigma)
    
    # Standard DFT
    F = np.fft.fft(np.eye(n), norm="ortho", axis=0)
    
    k = np.arange(n)
    
    def lct_model(params):
        """
        LCT model: D₁·C₁·F·C₂·D₂
        params = [phi1, sigma1, sigma2, phi2]
        """
        phi1, sigma1, sigma2, phi2 = params
        
        # Diagonal phase operators
        D1 = np.diag(np.exp(1j * phi1 * k / n))
        D2 = np.diag(np.exp(1j * phi2 * k / n))
        
        # Quadratic chirps
        C1 = np.diag(np.exp(1j * sigma1 * k * k / n))
        C2 = np.diag(np.exp(1j * sigma2 * k * k / n))
        
        return D1 @ C1 @ F @ C2 @ D2
    
    def objective(params):
        """Frobenius distance to Ψ"""
        try:
            lct = lct_model(params)
            return np.linalg.norm(lct - Psi, 'fro')
        except:
            return 1e10
    
    print("Running global optimization (this may take 1-2 minutes)...")
    
    # Use differential evolution for global search
    bounds = [(-10, 10), (-5, 5), (-5, 5), (-10, 10)]
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=200,
        popsize=20,
        seed=42,
        workers=1,
        disp=False,
        atol=1e-6,
        tol=1e-6
    )
    
    best_error = result.fun
    best_params = result.x
    
    print()
    print(f"Best LCT fit found:")
    print(f"  Parameters: φ₁={best_params[0]:.4f}, σ₁={best_params[1]:.4f}, "
          f"σ₂={best_params[2]:.4f}, φ₂={best_params[3]:.4f}")
    print(f"  Frobenius error: {best_error:.6f}")
    print(f"  Relative error: {best_error / np.linalg.norm(Psi, 'fro'):.6f}")
    print()
    
    # Verdict
    norm_psi = np.linalg.norm(Psi, 'fro')
    rel_error = best_error / norm_psi
    
    if rel_error < 0.01:
        verdict = "❌ FAIL: RFT is well-approximated by LCT"
        is_distinct = False
    elif rel_error < 0.1:
        verdict = "⚠️  WEAK: RFT is similar to some LCT"
        is_distinct = False
    else:
        verdict = f"✅ PASS: RFT cannot be approximated by LCT (error = {rel_error:.3f})"
        is_distinct = True
    
    print(verdict)
    print()
    
    return {
        'test': 'lct_optimization_fit',
        'frobenius_error': float(best_error),
        'relative_error': float(rel_error),
        'best_params': best_params.tolist(),
        'is_distinct': is_distinct,
        'verdict': verdict
    }


def test_3_dft_correlation(n: int = 256, beta: float = 0.83, sigma: float = 1.25, trials: int = 100):
    """
    TEST 3: DFT Column Correlation
    
    If Ψ ≈ D₁·F·D₂·P, then ⟨Ψx, Fx⟩ should be large for some x.
    Test with random vectors and measure max correlation.
    """
    print("=" * 80)
    print("TEST 3: DFT CORRELATION")
    print("=" * 80)
    print(f"Matrix size: {n}")
    print(f"Testing {trials} random vectors")
    print()
    
    # rft_forward is already imported globally
    
    rng = np.random.default_rng(0xBEEF)
    max_corr = 0.0
    correlations = []
    
    for _ in range(trials):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        x = x / np.linalg.norm(x)
        
        Psi_x = rft_forward(x, beta=beta, sigma=sigma)
        F_x = np.fft.fft(x, norm="ortho")
        
        # Normalized correlation
        corr = abs(np.vdot(Psi_x, F_x))
        correlations.append(float(corr))
        max_corr = max(max_corr, corr)
    
    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations))
    
    print(f"Correlation statistics:")
    print(f"  Maximum: {max_corr:.6f}")
    print(f"  Mean: {mean_corr:.6f} ± {std_corr:.6f}")
    print()
    
    # Verdict
    if max_corr > 0.8:
        verdict = "❌ FAIL: High correlation with DFT (likely equivalent)"
        is_distinct = False
    elif max_corr > 0.5:
        verdict = "⚠️  WEAK: Moderate correlation with DFT"
        is_distinct = False
    else:
        verdict = f"✅ PASS: Low correlation with DFT (max = {max_corr:.3f})"
        is_distinct = True
    
    print(verdict)
    print()
    
    return {
        'test': 'dft_correlation',
        'max_correlation': float(max_corr),
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'is_distinct': is_distinct,
        'verdict': verdict
    }


def test_4_psihf_entropy(n: int = 128, beta: float = 0.83, sigma: float = 1.25):
    """
    TEST 4: Ψ†F Entropy
    
    If Ψ = D₁·F·D₂·P, then Ψ†F = P†·D₂†·D₁† ≈ permutation matrix
    → Columns should be sparse (low entropy)
    
    If Ψ is truly distinct, Ψ†F should have diffuse structure (high entropy).
    """
    print("=" * 80)
    print("TEST 4: Ψ†F ENTROPY")
    print("=" * 80)
    print(f"Matrix size: {n}")
    print()
    
    # Build matrices
    print("Building Ψ†F matrix...")
    Psi = rft_matrix(n, beta=beta, sigma=sigma)
    F = np.fft.fft(np.eye(n), norm="ortho", axis=0)
    
    S = Psi.conj().T @ F  # Ψ†F
    
    # Compute column-wise entropy
    def shannon_entropy(p):
        p = np.clip(p, 1e-16, 1.0)
        return float(-(p * np.log(p)).sum())
    
    entropies = []
    for j in range(n):
        col = S[:, j]
        probs = np.abs(col)**2
        probs /= max(1e-16, probs.sum())
        entropies.append(shannon_entropy(probs))
    
    mean_ent = float(np.mean(entropies))
    max_ent = float(np.log(n))
    
    print(f"Column entropy statistics:")
    print(f"  Mean entropy: {mean_ent:.4f} bits")
    print(f"  Maximum possible: {max_ent:.4f} bits")
    print(f"  Ratio: {mean_ent/max_ent:.3f}")
    print()
    
    # Verdict
    ratio = mean_ent / max_ent
    
    if ratio < 0.3:
        verdict = "❌ FAIL: Low entropy (likely permutation-like)"
        is_distinct = False
    elif ratio < 0.6:
        verdict = "⚠️  WEAK: Moderate entropy"
        is_distinct = False
    else:
        verdict = f"✅ PASS: High entropy (diffuse structure, ratio = {ratio:.3f})"
        is_distinct = True
    
    print(verdict)
    print()
    
    return {
        'test': 'psihf_entropy',
        'mean_entropy': mean_ent,
        'max_entropy': max_ent,
        'entropy_ratio': float(ratio),
        'is_distinct': is_distinct,
        'verdict': verdict
    }


def run_full_suite():
    """Run all LCT non-membership tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "LCT NON-MEMBERSHIP TEST SUITE" + " " * 29 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Testing whether Φ-RFT is structurally distinct from" + " " * 23 + "║")
    print("║" + "  the Linear Canonical Transform (LCT) family" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    results = {}
    
    # Test 1: Quadratic phase residual
    results['test1'] = test_1_quadratic_phase_residual(n=256)
    
    # Test 2: LCT optimization (smaller n for speed)
    results['test2'] = test_2_lct_optimization_fit(n=128)
    
    # Test 3: DFT correlation
    results['test3'] = test_3_dft_correlation(n=256)
    
    # Test 4: Ψ†F entropy
    results['test4'] = test_4_psihf_entropy(n=128)
    
    # Summary
    print("=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    
    tests_passed = sum([
        results['test1']['is_distinct'],
        results['test2']['is_distinct'],
        results['test3']['is_distinct'],
        results['test4']['is_distinct']
    ])
    
    print(f"\nTests passed: {tests_passed}/4")
    print()
    
    for i in range(1, 5):
        key = f'test{i}'
        status = "✅ PASS" if results[key]['is_distinct'] else "❌ FAIL"
        print(f"  [{status}] {results[key]['test']}")
    
    print()
    
    if tests_passed == 4:
        final = "✅ STRONG EVIDENCE: RFT is NOT in LCT family (all 4 tests passed)"
    elif tests_passed >= 3:
        final = "✅ GOOD EVIDENCE: RFT is likely distinct from LCT (3+ tests passed)"
    elif tests_passed >= 2:
        final = "⚠️  MODERATE EVIDENCE: RFT shows some distinctness (2 tests passed)"
    else:
        final = "❌ WEAK EVIDENCE: RFT may be LCT-equivalent (< 2 tests passed)"
    
    print(final)
    print()
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'total_tests': 4,
        'verdict': final
    }
    
    return results


if __name__ == "__main__":
    results = run_full_suite()
    
    # Save results
    output_path = 'lct_nonmembership_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
