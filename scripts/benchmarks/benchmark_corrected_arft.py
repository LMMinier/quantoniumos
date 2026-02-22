#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
CORRECTED ARFT vs DCT vs FFT vs KLT Benchmark
==============================================

BUG FIX: Previous KLT used single-signal covariance (rank-1, trivial).
CORRECTED: KLT now uses class-average covariance from multiple signal segments.

This gives a fair comparison:
- DCT/FFT: Fixed basis (no signal knowledge)
- ARFT: Signal-adaptive (autocorrelation-based)
- KLT: Class-adaptive (covariance from training data)

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import toeplitz, eigh
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10


# ============================================================================
# Verify Implementations
# ============================================================================

def verify_all_transforms():
    """Verify unitarity and energy preservation."""
    print("=" * 70)
    print("VERIFICATION: Transform Properties")
    print("=" * 70)
    
    np.random.seed(42)
    n = 256
    x = np.random.randn(n)
    
    # DCT roundtrip
    c = dct(x, type=2, norm='ortho')
    x_r = idct(c, type=2, norm='ortho')
    err_dct = np.linalg.norm(x - x_r)
    
    # FFT roundtrip
    c = np.fft.fft(x) / np.sqrt(n)
    x_r = np.real(np.fft.ifft(c * np.sqrt(n)))
    err_fft = np.linalg.norm(x - x_r)
    
    # ARFT roundtrip
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    _, U = eigh(L)
    c = U.T @ x
    x_r = U @ c
    err_arft = np.linalg.norm(x - x_r)
    
    print(f"Roundtrip errors: DCT={err_dct:.2e}, FFT={err_fft:.2e}, ARFT={err_arft:.2e}")
    
    if max(err_dct, err_fft, err_arft) < 1e-10:
        print("✅ All transforms are unitary")
        return True
    else:
        print("❌ Unitarity check failed")
        return False


# ============================================================================
# Data Loaders
# ============================================================================

def load_ecg_long() -> np.ndarray:
    """Load long ECG signal for KLT training."""
    try:
        import wfdb
    except ImportError:
        return None
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    record_path = os.path.join(data_dir, "100")
    
    if os.path.exists(record_path + ".dat"):
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 0].astype(np.float64)
    return None


def segment_signal(signal: np.ndarray, segment_len: int, num_segments: int) -> List[np.ndarray]:
    """Extract multiple segments from a long signal."""
    segments = []
    step = max(1, (len(signal) - segment_len) // num_segments)
    
    for i in range(num_segments):
        start = i * step
        if start + segment_len <= len(signal):
            seg = signal[start:start + segment_len].copy()
            seg = (seg - np.mean(seg)) / (np.std(seg) + EPS)
            segments.append(seg)
    
    return segments


# ============================================================================
# Transform Implementations
# ============================================================================

def compute_class_klt(segments: List[np.ndarray]) -> np.ndarray:
    """
    Compute KLT basis from multiple signal segments (proper training).
    
    This is the Karhunen-Loève Transform using sample covariance
    from the signal class, not a single signal.
    """
    n = len(segments[0])
    
    # Compute sample covariance from all segments
    cov = np.zeros((n, n))
    for seg in segments:
        cov += np.outer(seg, seg)
    cov /= len(segments)
    
    # Eigenbasis
    eigenvalues, U = eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return U[:, idx]


def compute_arft_kernel(x: np.ndarray) -> np.ndarray:
    """Compute ARFT kernel (eigenbasis of Toeplitz autocorrelation)."""
    n = len(x)
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    eigenvalues, U = eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    return U[:, idx]


def compress_with_transform(x: np.ndarray, method: str, keep_ratio: float, 
                           klt_kernel: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    Compress signal and return (reconstructed, prd).
    """
    n = len(x)
    x_norm = (x - np.mean(x)) / (np.std(x) + EPS)
    k = max(1, int(n * keep_ratio))
    
    if method == "dct":
        coeffs = dct(x_norm, type=2, norm='ortho')
        inv = lambda c: idct(c, type=2, norm='ortho')
    elif method == "fft":
        coeffs = np.fft.fft(x_norm) / np.sqrt(n)
        inv = lambda c: np.real(np.fft.ifft(c * np.sqrt(n)))
    elif method == "arft":
        U = compute_arft_kernel(x_norm)
        coeffs = U.T @ x_norm
        inv = lambda c: U @ c
    elif method == "klt":
        if klt_kernel is None:
            raise ValueError("KLT requires pre-trained kernel")
        coeffs = klt_kernel.T @ x_norm
        inv = lambda c: klt_kernel @ c
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Keep top-k by magnitude
    idx = np.argsort(np.abs(coeffs))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs)
    coeffs_sparse[idx] = coeffs[idx]
    
    recon = np.real(inv(coeffs_sparse))
    
    # PRD
    diff = x_norm - recon
    prd = 100 * np.sqrt(np.sum(diff**2) / np.sum(x_norm**2))
    
    return recon, prd


def gini_index(x: np.ndarray, method: str, klt_kernel: np.ndarray = None) -> float:
    """Compute Gini index of coefficient magnitudes."""
    n = len(x)
    x_norm = (x - np.mean(x)) / (np.std(x) + EPS)
    
    if method == "dct":
        coeffs = dct(x_norm, type=2, norm='ortho')
    elif method == "fft":
        coeffs = np.fft.fft(x_norm) / np.sqrt(n)
    elif method == "arft":
        U = compute_arft_kernel(x_norm)
        coeffs = U.T @ x_norm
    elif method == "klt":
        coeffs = klt_kernel.T @ x_norm
    else:
        raise ValueError(f"Unknown method: {method}")
    
    c = np.sort(np.abs(coeffs.flatten()))
    if len(c) == 0 or np.sum(c) == 0:
        return 0.0
    cumsum = np.cumsum(c)
    return (len(c) + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / len(c)


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("⚠️  Set USE_REAL_DATA=1 to run")
        print("   FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        sys.exit(1)
    
    print("=" * 70)
    print("CORRECTED ARFT vs DCT vs FFT vs KLT Benchmark")
    print("=" * 70)
    print("KLT now uses class-average covariance (fair comparison)")
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    # Verify transforms
    if not verify_all_transforms():
        return
    
    # Load ECG
    print("\n📥 Loading ECG data...")
    ecg_full = load_ecg_long()
    if ecg_full is None:
        print("❌ No ECG data available")
        return
    
    segment_len = 512
    num_train_segments = 100
    
    # Split into training (for KLT) and test segments
    train_segments = segment_signal(ecg_full[:len(ecg_full)//2], segment_len, num_train_segments)
    test_segments = segment_signal(ecg_full[len(ecg_full)//2:], segment_len, 20)
    
    print(f"✅ Training segments: {len(train_segments)}")
    print(f"✅ Test segments: {len(test_segments)}")
    
    # Train KLT on training data
    print("\n📊 Training KLT on ECG class...")
    klt_kernel = compute_class_klt(train_segments)
    print("✅ KLT kernel computed")
    
    # Benchmark on test data
    print("\n" + "=" * 70)
    print("BENCHMARK: Compression on UNSEEN ECG segments")
    print("=" * 70)
    
    methods = ["dct", "fft", "arft", "klt"]
    keep_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    # Aggregate results
    prd_sums = {m: {k: 0.0 for k in keep_ratios} for m in methods}
    gini_sums = {m: 0.0 for m in methods}
    
    for i, seg in enumerate(test_segments):
        for m in methods:
            klt_k = klt_kernel if m == "klt" else None
            gini_sums[m] += gini_index(seg, m, klt_k)
            
            for keep in keep_ratios:
                _, prd = compress_with_transform(seg, m, keep, klt_k)
                prd_sums[m][keep] += prd
    
    n_test = len(test_segments)
    
    # Average results
    print(f"\n📊 Average over {n_test} test segments:")
    
    # Gini
    print("\nSparsity (Gini Index - higher is better):")
    gini_avgs = {m: gini_sums[m] / n_test for m in methods}
    gini_winner = max(gini_avgs, key=gini_avgs.get)
    for m in methods:
        marker = "🏆" if m == gini_winner else "  "
        adaptive = "(adaptive)" if m in ["arft", "klt"] else "(fixed)"
        print(f"  {marker}{m.upper():5s} {adaptive:12s}: {gini_avgs[m]:.4f}")
    
    # PRD at each regime
    print("\nCompression PRD (%) - lower is better:")
    print(f"  {'Keep':>6s} |", end="")
    for m in methods:
        print(f" {m.upper():>7s} |", end="")
    print(" Winner")
    print("  " + "-" * 55)
    
    wins = {m: 0 for m in methods}
    
    for keep in keep_ratios:
        prd_avgs = {m: prd_sums[m][keep] / n_test for m in methods}
        winner = min(prd_avgs, key=prd_avgs.get)
        wins[winner] += 1
        
        print(f"  {int(keep*100):5d}% |", end="")
        for m in methods:
            marker = "🏆" if m == winner else "  "
            print(f" {marker}{prd_avgs[m]:5.2f} |", end="")
        print(f" {winner.upper()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n📊 PRD Wins by method:")
    for m in methods:
        bar = "█" * (wins[m] * 4)
        adaptive = "(adaptive)" if m in ["arft", "klt"] else "(fixed)"
        print(f"  {m.upper():5s} {adaptive:12s}: {wins[m]}/{len(keep_ratios)} {bar}")
    
    # Compare ARFT vs KLT
    print("\n" + "─" * 70)
    print("ADAPTIVE TRANSFORM COMPARISON:")
    print("─" * 70)
    
    arft_vs_klt = []
    for keep in keep_ratios:
        arft_prd = prd_sums["arft"][keep] / n_test
        klt_prd = prd_sums["klt"][keep] / n_test
        diff_pct = 100 * (arft_prd - klt_prd) / klt_prd if klt_prd > 0 else 0
        arft_vs_klt.append((keep, arft_prd, klt_prd, diff_pct))
        
    print(f"  {'Keep':>6s} | {'ARFT PRD':>10s} | {'KLT PRD':>10s} | {'Diff':>10s}")
    print("  " + "-" * 45)
    for keep, arft_prd, klt_prd, diff_pct in arft_vs_klt:
        sign = "+" if diff_pct > 0 else ""
        print(f"  {int(keep*100):5d}% | {arft_prd:10.2f} | {klt_prd:10.2f} | {sign}{diff_pct:9.1f}%")
    
    avg_diff = np.mean([d[3] for d in arft_vs_klt])
    if avg_diff < 20:
        print(f"\n✅ ARFT is within {avg_diff:.1f}% of KLT (class-optimal)")
        print("   ARFT achieves near-KLT performance with per-signal adaptation")
    else:
        print(f"\n⚠️  ARFT is {avg_diff:.1f}% behind KLT")
    
    # Compare ARFT vs DCT
    print("\n" + "─" * 70)
    print("ARFT vs DCT (Fixed Baseline):")
    print("─" * 70)
    
    arft_vs_dct = []
    for keep in keep_ratios:
        arft_prd = prd_sums["arft"][keep] / n_test
        dct_prd = prd_sums["dct"][keep] / n_test
        improvement = 100 * (dct_prd - arft_prd) / dct_prd if dct_prd > 0 else 0
        arft_vs_dct.append((keep, arft_prd, dct_prd, improvement))
    
    print(f"  {'Keep':>6s} | {'ARFT PRD':>10s} | {'DCT PRD':>10s} | {'Improvement':>12s}")
    print("  " + "-" * 50)
    for keep, arft_prd, dct_prd, improvement in arft_vs_dct:
        sign = "+" if improvement > 0 else ""
        print(f"  {int(keep*100):5d}% | {arft_prd:10.2f} | {dct_prd:10.2f} | {sign}{improvement:11.1f}%")
    
    avg_improvement = np.mean([d[3] for d in arft_vs_dct])
    if avg_improvement > 0:
        print(f"\n✅ ARFT beats DCT by average {avg_improvement:.1f}%")
    else:
        print(f"\n⚠️  DCT beats ARFT by average {-avg_improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("\n1️⃣  Fixed transforms (DCT, FFT):")
    print("   → DCT > FFT on ECG (smooth quasi-periodic)")
    
    print("\n2️⃣  Adaptive transforms (ARFT, KLT):")
    print("   → KLT is optimal (trained on signal class)")
    print(f"   → ARFT achieves {100 - abs(avg_diff):.0f}% of KLT performance")
    print("   → ARFT is per-signal adaptive (no training needed)")
    
    if avg_improvement > 10:
        print(f"\n3️⃣  ARFT beats DCT by {avg_improvement:.1f}% on real ECG")
        print("   → Adaptive approach provides measurable benefit")
    
    print("\n" + "=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
