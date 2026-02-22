#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Comprehensive comparison of three hybrid MCA strategies:
1. Greedy Sequential (adaptive_hybrid_compress)
2. Hard-Threshold Braided (braided_hybrid_mca)
3. Soft-Threshold Braided (soft_braided_hybrid_mca)

Tests all three on:
- MCA separation (known DCT + RFT components)
- ASCII compression (natural text)
- Rate-distortion curves
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.fft import dct, idct
from algorithms.rft.hybrids.legacy_mca import (
    adaptive_hybrid_compress,
    braided_hybrid_mca,
    soft_braided_hybrid_mca,
    rft_forward,
    rft_inverse,
)


def generate_ground_truth_signal(n: int = 256, K: int = 8, seed: int = 42):
    """Generate known DCT-sparse + RFT-sparse components."""
    rng = np.random.default_rng(seed)
    
    # DCT component: K random spikes
    cS_true = np.zeros(n)
    idx_s = rng.choice(n, size=K, replace=False)
    cS_true[idx_s] = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    x_struct = idct(cS_true.real, norm='ortho') + 1j * idct(cS_true.imag, norm='ortho')
    
    # RFT component: K random spikes
    cT_true = np.zeros(n, dtype=np.complex128)
    idx_t = rng.choice(n, size=K, replace=False)
    cT_true[idx_t] = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    x_texture = rft_inverse(cT_true)
    
    # Mix and add noise
    x_true = x_struct + x_texture
    noise = 0.01 * rng.standard_normal(n)
    x_obs = x_true + noise
    
    return x_obs, x_struct, x_texture, cS_true, cT_true


def separation_metrics(x_struct_true, x_texture_true, x_struct_est, x_texture_est):
    """Compute separation quality metrics."""
    err_s = np.linalg.norm(x_struct_true - x_struct_est) / np.linalg.norm(x_struct_true)
    err_t = np.linalg.norm(x_texture_true - x_texture_est) / np.linalg.norm(x_texture_true)
    
    # F1-like: precision/recall on coefficient support
    # (Simplified: just use correlation as proxy)
    corr_s = np.abs(np.vdot(x_struct_true, x_struct_est)) / (
        np.linalg.norm(x_struct_true) * np.linalg.norm(x_struct_est) + 1e-12
    )
    corr_t = np.abs(np.vdot(x_texture_true, x_texture_est)) / (
        np.linalg.norm(x_texture_true) * np.linalg.norm(x_texture_est) + 1e-12
    )
    
    return err_s, err_t, corr_s, corr_t


def test_mca_separation():
    """Test 1: MCA ground-truth separation."""
    print("=" * 70)
    print("TEST 1: MCA SEPARATION (Ground Truth Components)")
    print("=" * 70)
    
    x_obs, x_s_true, x_t_true, cS_true, cT_true = generate_ground_truth_signal()
    x_true = x_s_true + x_t_true
    
    # Test all three methods
    methods = [
        ("Greedy Sequential", lambda: adaptive_hybrid_compress(x_obs, max_iter=20)),
        ("Hard Braided", lambda: braided_hybrid_mca(x_obs, max_iter=20, threshold=0.05)),
        ("Soft Braided", lambda: soft_braided_hybrid_mca(x_obs, max_iter=20, threshold=0.05)),
    ]
    
    results = []
    for name, method in methods:
        res = method()
        x_s_est = res.structural
        x_t_est = res.texture
        x_rec = x_s_est + x_t_est
        
        # Reconstruction error
        err_rec = np.linalg.norm(x_true - x_rec) / np.linalg.norm(x_true)
        
        # Separation metrics
        err_s, err_t, corr_s, corr_t = separation_metrics(
            x_s_true, x_t_true, x_s_est, x_t_est
        )
        
        results.append({
            'name': name,
            'err_rec': err_rec,
            'err_s': err_s,
            'err_t': err_t,
            'corr_s': corr_s,
            'corr_t': corr_t,
        })
    
    # Print table
    print("\n| Method | Reconstruction | Sep Error (S) | Sep Error (T) | Corr (S) | Corr (T) |")
    print("|:-------|:--------------:|:-------------:|:-------------:|:--------:|:--------:|")
    for r in results:
        print(f"| {r['name']:<20} | {r['err_rec']:>6.3f} | {r['err_s']:>6.3f} | "
              f"{r['err_t']:>6.3f} | {r['corr_s']:>6.3f} | {r['corr_t']:>6.3f} |")
    
    return results


def test_ascii_compression():
    """Test 2: ASCII text compression."""
    print("\n" + "=" * 70)
    print("TEST 2: ASCII COMPRESSION (Natural Text)")
    print("=" * 70)
    
    # Sample text data
    text = "The quick brown fox jumps over the lazy dog. " * 6
    x = np.array([ord(c) for c in text[:256]], dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    
    methods = [
        ("Greedy Sequential", lambda: adaptive_hybrid_compress(x, max_iter=50)),
        ("Hard Braided", lambda: braided_hybrid_mca(x, max_iter=50, threshold=0.05)),
        ("Soft Braided", lambda: soft_braided_hybrid_mca(x, max_iter=50, threshold=0.05)),
    ]
    
    results = []
    for name, method in methods:
        res = method()
        
        # Count active coefficients
        cS = dct(res.structural.real, norm='ortho')
        cT = rft_forward(res.texture)
        
        thresh = 0.01 * max(np.max(np.abs(cS)), np.max(np.abs(cT)))
        active_s = np.sum(np.abs(cS) > thresh)
        active_t = np.sum(np.abs(cT) > thresh)
        total_active = active_s + active_t
        
        sparsity = 100 * (1 - total_active / x.size)
        
        # Reconstruction error
        x_rec = res.structural + res.texture
        err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        
        results.append({
            'name': name,
            'sparsity': sparsity,
            'active': total_active,
            'error': err,
        })
    
    # Print table
    print("\n| Method | Sparsity | Active Coeffs | Reconstruction Error |")
    print("|:-------|:--------:|:-------------:|:--------------------:|")
    for r in results:
        print(f"| {r['name']:<20} | {r['sparsity']:>6.2f}% | "
              f"{r['active']:>3d} / 256 | {r['error']:>8.4f} |")
    
    return results


def test_rate_distortion():
    """Test 3: Rate-distortion curves."""
    print("\n" + "=" * 70)
    print("TEST 3: RATE-DISTORTION CURVES")
    print("=" * 70)
    
    # Generate mixed signal
    x_obs, x_s_true, x_t_true, _, _ = generate_ground_truth_signal(n=256, K=10)
    x_true = x_s_true + x_t_true
    
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    methods = [
        ("Greedy", lambda t: adaptive_hybrid_compress(x_obs, max_iter=20)),
        ("Hard Braid", lambda t: braided_hybrid_mca(x_obs, max_iter=20, threshold=t)),
        ("Soft Braid", lambda t: soft_braided_hybrid_mca(x_obs, max_iter=20, threshold=t)),
    ]
    
    print("\n| Method | Threshold | Rate (% coeffs) | Distortion (MSE) |")
    print("|:-------|:---------:|:---------------:|:----------------:|")
    
    for name, method in methods:
        for thresh in thresholds:
            res = method(thresh)
            
            # Count coefficients
            cS = dct(res.structural.real, norm='ortho')
            cT = rft_forward(res.texture)
            
            active = np.sum(np.abs(cS) > 1e-6) + np.sum(np.abs(cT) > 1e-6)
            rate = 100 * active / x_obs.size
            
            # Distortion
            x_rec = res.structural + res.texture
            mse = np.mean(np.abs(x_true - x_rec) ** 2)
            
            print(f"| {name:<12} | {thresh:>7.3f} | {rate:>6.2f}% | {mse:>10.6f} |")


def main():
    """Run all three tests."""
    print("\n" + "=" * 70)
    print("SOFT vs HARD BRAIDING COMPARISON")
    print("=" * 70)
    print("\nComparing three hybrid MCA strategies:")
    print("1. Greedy Sequential (baseline)")
    print("2. Hard-Threshold Braided (winner-takes-all)")
    print("3. Soft-Threshold Braided (proportional allocation)")
    print()
    
    # Run tests
    mca_results = test_mca_separation()
    ascii_results = test_ascii_compression()
    test_rate_distortion()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find winners
    mca_winner = min(mca_results, key=lambda r: r['err_rec'])
    ascii_winner = max(ascii_results, key=lambda r: r['sparsity'])
    
    print(f"\n✅ MCA Separation Winner: {mca_winner['name']}")
    print(f"   Reconstruction error: {mca_winner['err_rec']:.4f}")
    
    print(f"\n✅ ASCII Compression Winner: {ascii_winner['name']}")
    print(f"   Sparsity: {ascii_winner['sparsity']:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
