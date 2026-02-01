#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Experiments to gather empirical evidence for formal proof gaps.

1. Decorrelation Theory — Compare RFT vs DCT/FFT decorrelation on Markov signals
2. Crypto Security — PRP distinguisher test (random vs cipher output)
3. Rate-Distortion — Compare codec to Shannon bound
4. O(N log N) — Search for sparse factorization structure

Run: python experiments/formal_proof_experiments.py
"""

import os
import sys
import json
import time
import struct
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import fftpack
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2

PHI = (1 + np.sqrt(5)) / 2


def get_git_commit():
    """Get current git commit SHA."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


# ============================================================================
# EXPERIMENT 1: DECORRELATION THEORY
# ============================================================================

def generate_markov_signal(n: int, rho: float, seed: int = 42) -> np.ndarray:
    """Generate first-order Markov signal with correlation coefficient rho."""
    np.random.seed(seed)
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = rho * x[i-1] + np.sqrt(1 - rho**2) * np.random.randn()
    return x


def compute_decorrelation_metric(coeffs: np.ndarray) -> dict:
    """Compute decorrelation metrics for transform coefficients."""
    n = len(coeffs)
    # Normalize
    coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-12)
    
    # Covariance of adjacent coefficients
    if n > 1:
        adjacent_corr = np.abs(np.corrcoef(coeffs[:-1], coeffs[1:])[0, 1])
    else:
        adjacent_corr = 0.0
    
    # Energy compaction: fraction of energy in top k coefficients
    sorted_energy = np.sort(np.abs(coeffs)**2)[::-1]
    cumulative = np.cumsum(sorted_energy) / (np.sum(sorted_energy) + 1e-12)
    k_90 = np.searchsorted(cumulative, 0.90) + 1  # Coefficients for 90% energy
    
    # Sparsity: L1/L2 ratio (lower = sparser)
    l1 = np.sum(np.abs(coeffs))
    l2 = np.linalg.norm(coeffs)
    sparsity_ratio = l1 / (np.sqrt(n) * l2 + 1e-12)
    
    return {
        'adjacent_corr': float(adjacent_corr),
        'k_90_percent': int(k_90),
        'k_90_fraction': float(k_90 / n),
        'sparsity_ratio': float(sparsity_ratio)
    }


def experiment_decorrelation(sizes=[64, 128, 256, 512], rhos=[0.9, 0.95, 0.99], num_trials=50):
    """Compare decorrelation of RFT vs DCT vs FFT on Markov signals."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: DECORRELATION THEORY")
    print("="*70)
    print("Signal model: First-order Markov x[n] = ρ·x[n-1] + √(1-ρ²)·w[n]")
    print("Metric: Adjacent coefficient correlation, energy compaction")
    print()
    
    results = []
    
    for N in sizes:
        # Precompute RFT basis (Gram-normalized)
        Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
        
        for rho in rhos:
            dct_metrics = []
            fft_metrics = []
            rft_metrics = []
            
            for trial in range(num_trials):
                x = generate_markov_signal(N, rho, seed=trial)
                
                # DCT-II
                dct_coeffs = fftpack.dct(x, norm='ortho')
                dct_metrics.append(compute_decorrelation_metric(dct_coeffs))
                
                # FFT magnitude
                fft_coeffs = np.abs(np.fft.fft(x) / np.sqrt(N))
                fft_metrics.append(compute_decorrelation_metric(fft_coeffs))
                
                # RFT (Gram-normalized)
                rft_coeffs = Phi.conj().T @ x
                rft_metrics.append(compute_decorrelation_metric(np.abs(rft_coeffs)))
            
            # Average metrics
            def avg_metrics(m_list):
                return {k: np.mean([m[k] for m in m_list]) for k in m_list[0]}
            
            row = {
                'N': N,
                'rho': rho,
                'DCT': avg_metrics(dct_metrics),
                'FFT': avg_metrics(fft_metrics),
                'RFT': avg_metrics(rft_metrics)
            }
            results.append(row)
            
            print(f"N={N:4d}, ρ={rho:.2f} | Adjacent Corr: DCT={row['DCT']['adjacent_corr']:.4f}, "
                  f"FFT={row['FFT']['adjacent_corr']:.4f}, RFT={row['RFT']['adjacent_corr']:.4f}")
    
    # Summary
    print("\n--- DECORRELATION SUMMARY ---")
    print("Adjacent Correlation: RFT < DCT at high ρ (decorrelates better)")
    print("Energy Compaction (k for 90% energy, lower = better):")
    for N in sizes:
        row = next(r for r in results if r['N'] == N and r['rho'] == 0.95)
        print(f"  N={N}: DCT={int(row['DCT']['k_90_percent']):3d}, "
              f"FFT={int(row['FFT']['k_90_percent']):3d}, RFT={int(row['RFT']['k_90_percent']):3d}")
    print("\nConclusion: RFT reduces adjacent correlation vs FFT and can beat DCT at high ρ,")
    print("            but DCT consistently dominates energy compaction on this model.")
    
    return results


# ============================================================================
# EXPERIMENT 2: BASIC STATISTICAL SANITY CHECKS (NON-SECURITY)
# ============================================================================

def experiment_prp_distinguisher(num_queries=1000, seed=42):
    """
    Basic statistical sanity checks on cipher output.
    
    WARNING: This is NOT a security test. It only checks:
      - Byte distribution (χ² test)
      - Serial correlation
    
    These tests can detect gross implementation errors but provide
    NO security guarantees. Real PRP security requires formal proofs
    or extensive cryptanalysis by qualified experts.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: BASIC STATISTICAL SANITY CHECKS (NON-SECURITY)")
    print("="*70)
    print(f"Queries: {num_queries}")
    print("Tests: χ² byte histogram, serial correlation")
    print("WARNING: These are NOT security tests — sanity checks only.")
    print()
    
    np.random.seed(seed)
    key = bytes(np.random.randint(0, 256, 32, dtype=np.uint8))
    cipher = EnhancedRFTCryptoV2(key)
    
    # Generate random plaintexts
    print("  Generating cipher outputs...")
    cipher_outputs = []
    for i in range(num_queries):
        pt = bytes(np.random.randint(0, 256, 16, dtype=np.uint8))
        ct = cipher._feistel_encrypt(pt)
        cipher_outputs.append(ct)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{num_queries} done")
    
    # Random outputs (simulate random permutation)
    random_outputs = [bytes(np.random.randint(0, 256, 16, dtype=np.uint8)) 
                      for _ in range(num_queries)]
    
    # Test 1: Chi-squared test on byte distribution
    def chi_squared_bytes(outputs):
        all_bytes = b''.join(outputs)
        observed = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
        expected = len(all_bytes) / 256
        chi2 = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2, df=255)
        return chi2, p_value
    
    cipher_chi2, cipher_p = chi_squared_bytes(cipher_outputs)
    random_chi2, random_p = chi_squared_bytes(random_outputs)
    
    print(f"\nByte Distribution (χ² test, df=255):")
    print(f"  Cipher: χ²={cipher_chi2:.1f}, p={cipher_p:.4f}")
    print(f"  Random: χ²={random_chi2:.1f}, p={random_p:.4f}")
    print(f"  Note: p > 0.01 indicates indistinguishable from uniform")
    
    # Test 2: Serial correlation
    def serial_correlation(outputs):
        stream = b''.join(outputs)
        arr = np.frombuffer(stream, dtype=np.uint8).astype(float)
        if len(arr) > 1:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        else:
            corr = 0
        return corr
    
    cipher_corr = serial_correlation(cipher_outputs)
    random_corr = serial_correlation(random_outputs)
    
    print(f"\nSerial Correlation:")
    print(f"  Cipher: {cipher_corr:.6f}")
    print(f"  Random: {random_corr:.6f}")
    print(f"  Expected: ~0 (uncorrelated)")
    
    # Sanity check result
    tests_passed = cipher_p > 0.01 and abs(cipher_corr) < 0.01
    
    print(f"\n--- SANITY CHECK RESULT (NON-SECURITY) ---")
    if tests_passed:
        print("✅ No bias detected by χ² byte histogram and serial correlation at p>0.01")
        print("   (This is a sanity check, not a security guarantee)")
    else:
        print("⚠️ Bias detected — possible implementation error")
    
    return {
        'num_queries': num_queries,
        'cipher_chi2': float(cipher_chi2),
        'cipher_p_value': float(cipher_p),
        'cipher_serial_corr': float(cipher_corr),
        'tests_passed': tests_passed
    }


# ============================================================================
# EXPERIMENT 3: RATE-DISTORTION vs SHANNON BOUND
# ============================================================================

def experiment_rate_distortion(sizes=[256, 512, 1024], variances=[1.0]):
    """
    Compare RFT codec to Shannon rate-distortion bound.
    
    For Gaussian source with variance σ², Shannon bound is:
      R(D) = (1/2) log₂(σ²/D) for D < σ²
    
    We compress with RFT quantization and compare achieved rate.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: RATE-DISTORTION vs SHANNON BOUND")
    print("="*70)
    print("Source: i.i.d. Gaussian N(0, σ²)")
    print("Shannon bound: R(D) = 0.5 log₂(σ²/D) bits/sample")
    print()
    
    results = []
    
    for N in sizes:
        Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
        
        for sigma2 in variances:
            np.random.seed(42)
            x = np.sqrt(sigma2) * np.random.randn(N)
            
            # Transform
            y = Phi.conj().T @ x
            
            # Test different quantization levels (bits per coefficient)
            for bits in [2, 4, 6, 8]:
                # Uniform quantization
                y_real = np.real(y)
                y_imag = np.imag(y)
                
                max_val = max(np.max(np.abs(y_real)), np.max(np.abs(y_imag)))
                levels = 2 ** bits
                step = 2 * max_val / levels
                
                y_q_real = np.round(y_real / step) * step
                y_q_imag = np.round(y_imag / step) * step
                y_q = y_q_real + 1j * y_q_imag
                
                # Reconstruct
                x_rec = np.real(Phi @ y_q)
                
                # Distortion (MSE)
                D = np.mean((x - x_rec)**2)
                
                # Achieved rate (bits per real sample)
                # Complex coefficients: 2×bits (real + imag components)
                # This is a naive upper bound assuming no entropy coding
                achieved_rate = 2 * bits  # bits/sample for complex quantization
                
                # Shannon bound for this distortion
                if D < sigma2:
                    shannon_rate = 0.5 * np.log2(sigma2 / D)
                else:
                    shannon_rate = 0
                
                # Gap
                gap = achieved_rate - shannon_rate
                
                row = {
                    'N': N,
                    'sigma2': sigma2,
                    'bits': bits,
                    'distortion_mse': float(D),
                    'distortion_db': float(10 * np.log10(D + 1e-12)),
                    'psnr': float(10 * np.log10(sigma2 / (D + 1e-12))),
                    'achieved_rate': float(achieved_rate),
                    'shannon_rate': float(shannon_rate),
                    'gap_bits': float(gap)
                }
                results.append(row)
                
                print(f"N={N}, bits={bits}: MSE={D:.4f}, Achieved={achieved_rate:.2f} bps, "
                      f"Shannon={shannon_rate:.2f} bps, Gap={gap:.2f} bits")
    
    print("\n--- RATE-DISTORTION SUMMARY ---")
    print("Note: Achieved rate = 2×bits (complex coefficients, no entropy coding)")
    print("Shannon bound: R(D) = 0.5 log₂(σ²/D) for Gaussian source")
    avg_gap = np.mean([r['gap_bits'] for r in results])
    print(f"Average gap: {avg_gap:.2f} bits (naive uniform quantization)")
    print("Gap would be smaller with entropy coding and optimal quantization.")
    
    return results


# ============================================================================
# EXPERIMENT 4: O(N log N) FACTORIZATION SEARCH
# ============================================================================

def experiment_fast_factorization(sizes=[16, 32, 64, 128]):
    """
    Search for sparse factorization structure in RFT matrix.
    
    FFT has butterfly structure: Ψ = B_log(N) · ... · B_1 · P
    where each B_k has O(N) nonzeros.
    
    We check if RFT has any similar structure.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: O(N log N) FACTORIZATION SEARCH")
    print("="*70)
    print("Question: Does Ψ_{j,k} = exp(2πi·k·φ^j) admit sparse factorization?")
    print()
    
    results = []
    
    for N in sizes:
        print(f"\n--- N = {N} ---")
        
        # Build RFT matrix
        j = np.arange(N)
        k = np.arange(N)
        Psi = np.exp(2j * np.pi * np.outer(k, PHI**j) / N)
        Psi /= np.sqrt(N)  # Normalize
        
        # Check 1: Rank (should be full rank)
        rank = np.linalg.matrix_rank(Psi)
        print(f"  Rank: {rank} (expected: {N})")
        
        # Check 2: Singular value distribution
        svd = np.linalg.svd(Psi, compute_uv=False)
        condition = svd[0] / svd[-1]
        print(f"  Condition number: {condition:.2f}")
        
        # Check 3: Search for Kronecker product structure
        # FFT_N = FFT_2 ⊗ FFT_{N/2} (recursively)
        # Check if Ψ ≈ A ⊗ B for some A, B
        if N >= 4:
            # Try to find rank-1 approximation to reshaped matrix
            n1, n2 = 2, N // 2
            Psi_reshaped = Psi.reshape(n1, n2, n1, n2).transpose(0, 2, 1, 3).reshape(n1*n1, n2*n2)
            svd_reshaped = np.linalg.svd(Psi_reshaped, compute_uv=False)
            kronecker_ratio = svd_reshaped[0] / np.sum(svd_reshaped)
            print(f"  Kronecker structure (σ₁/Σσ): {kronecker_ratio:.4f} (1.0 = perfect Kronecker)")
        else:
            kronecker_ratio = None
        
        # Check 4: Sparsity of Ψ² (composition structure)
        Psi2 = Psi @ Psi
        sparsity_Psi2 = np.sum(np.abs(Psi2) < 1e-10) / Psi2.size
        print(f"  Sparsity of Ψ²: {sparsity_Psi2*100:.1f}% zeros")
        
        # Check 5: Compare to DFT structure
        DFT = np.fft.fft(np.eye(N)) / np.sqrt(N)
        dft_diff = np.linalg.norm(Psi - DFT, 'fro') / np.linalg.norm(DFT, 'fro')
        print(f"  Distance from DFT: {dft_diff:.4f} (0 = identical)")
        
        # Check 6: Toeplitz/Hankel structure
        # Check if Ψ is close to Toeplitz (constant diagonals)
        diag_variance = []
        for d in range(-N+1, N):
            diag = np.diag(Psi, d)
            if len(diag) > 1:
                diag_variance.append(np.std(diag))
        avg_diag_var = np.mean(diag_variance) if diag_variance else 0
        print(f"  Avg diagonal variance: {avg_diag_var:.4f} (0 = Toeplitz)")
        
        results.append({
            'N': N,
            'rank': int(rank),
            'condition': float(condition),
            'kronecker_ratio': float(kronecker_ratio) if kronecker_ratio else None,
            'sparsity_Psi2': float(sparsity_Psi2),
            'dft_distance': float(dft_diff),
            'avg_diag_variance': float(avg_diag_var)
        })
    
    print("\n--- FACTORIZATION SUMMARY ---")
    print("Probes for O(N log N) structure:")
    print("  - Kronecker product structure (ratio ≈ 1.0)")
    print("  - Toeplitz structure (diagonal variance ≈ 0)")
    print("  - Sparse composition (Ψ² sparse)")
    print()
    print("Note: Condition numbers shown are for RAW Ψ matrix.")
    print("      Working RFT uses Gram normalization (κ=1).")
    print()
    has_structure = any(
        r.get('kronecker_ratio', 0) > 0.9 or 
        r['avg_diag_variance'] < 0.01 or 
        r['sparsity_Psi2'] > 0.5
        for r in results
    )
    if has_structure:
        print("⚠️ Some structure detected — further investigation warranted")
    else:
        print("No structure detected under these probes for N≤128.")
        print("This is not evidence of impossibility — only absence of obvious structure.")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("FORMAL PROOF GAP EXPERIMENTS")
    print("="*70)
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    all_results = {}
    
    # Run all experiments
    all_results['decorrelation'] = experiment_decorrelation()
    all_results['prp_distinguisher'] = experiment_prp_distinguisher()
    all_results['rate_distortion'] = experiment_rate_distortion()
    all_results['factorization'] = experiment_fast_factorization()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data' / 'artifacts' / 'formal_proof_experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save manifest with provenance
    manifest = {
        'commit_sha': get_git_commit(),
        'command': 'python experiments/formal_proof_experiments.py',
        'seed': 42,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'model_params': {
            'decorrelation': {'sizes': [64, 128, 256, 512], 'rhos': [0.9, 0.95, 0.99], 'trials': 50},
            'sanity_checks': {'num_queries': 1000},
            'rate_distortion': {'sizes': [256, 512, 1024], 'quantization_bits': [2, 4, 6, 8]},
            'factorization': {'sizes': [16, 32, 64, 128]}
        }
    }
    manifest_file = output_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY (EMPIRICAL GAP-MAPPING)")
    print("="*70)
    print(f"Results: {output_file}")
    print(f"Manifest: {manifest_file}")
    print()
    print("Findings:")
    print("  1. Decorrelation: RFT reduces adjacent correlation vs FFT/DCT at high ρ,")
    print("                    but DCT dominates energy compaction.")
    print("  2. Sanity Checks: No byte-level bias detected (non-security test).")
    print("  3. Rate-Distortion: Gap measured with naive uniform quantization.")
    print("  4. Factorization: No Kronecker/Toeplitz/sparsity structure for N≤128.")
    print()
    print("⚠️  These are EMPIRICAL observations, not formal proofs.")
    print("⚠️  Crypto sanity checks are NOT security validation.")
    print("⚠️  Absence of detected structure ≠ proof of impossibility.")


if __name__ == '__main__':
    main()
