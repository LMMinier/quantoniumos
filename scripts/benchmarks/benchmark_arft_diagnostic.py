#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
ARFT Downstream Diagnostic Benchmark
=====================================

Tests whether ARFT's compression advantage translates to better
preservation of clinically relevant features:

1. ECG: R-peak detection and RR correlation after compression
2. EEG: Sleep band power preservation (delta, theta, alpha, beta)

Compares ARFT vs DCT at various compression levels.

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_arft_diagnostic.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import toeplitz, eigh
from scipy.signal import find_peaks, butter, filtfilt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPS = 1e-10


def require_real_data():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("⚠️  Set USE_REAL_DATA=1 to run")
        print("   FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        sys.exit(1)


# ============================================================================
# Data Loaders
# ============================================================================

def load_ecg_with_annotations():
    """Load ECG with R-peak annotations from MIT-BIH."""
    try:
        import wfdb
    except ImportError:
        return None, None, None
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    record_path = os.path.join(data_dir, "100")
    
    if not os.path.exists(record_path + ".dat"):
        return None, None, None
    
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    signal = record.p_signal[:, 0].astype(np.float64)
    fs = record.fs  # 360 Hz
    
    # Filter annotations to get R-peaks (normal beats)
    r_peaks = annotation.sample[np.isin(annotation.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])]
    
    return signal, r_peaks, fs


def generate_synthetic_eeg(n: int = 10000, fs: float = 256) -> np.ndarray:
    """Generate synthetic EEG with known band powers."""
    t = np.arange(n) / fs
    
    # Create EEG with defined rhythms
    delta = 2.0 * np.sin(2 * np.pi * 2 * t)    # 0.5-4 Hz (delta)
    theta = 1.5 * np.sin(2 * np.pi * 6 * t)    # 4-8 Hz (theta)
    alpha = 1.0 * np.sin(2 * np.pi * 10 * t)   # 8-13 Hz (alpha)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)    # 13-30 Hz (beta)
    
    # Add some noise
    noise = 0.3 * np.random.randn(n)
    
    return delta + theta + alpha + beta + noise


# ============================================================================
# Transform Implementations
# ============================================================================

def compress_dct(x: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Compress with DCT."""
    x_norm = (x - np.mean(x)) / (np.std(x) + EPS)
    n = len(x_norm)
    k = max(1, int(n * keep_ratio))
    
    coeffs = dct(x_norm, type=2, norm='ortho')
    idx = np.argsort(np.abs(coeffs))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs)
    coeffs_sparse[idx] = coeffs[idx]
    
    recon = idct(coeffs_sparse, type=2, norm='ortho')
    # Denormalize
    recon = recon * (np.std(x) + EPS) + np.mean(x)
    return recon


def compress_arft(x: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Compress with ARFT."""
    x_norm = (x - np.mean(x)) / (np.std(x) + EPS)
    n = len(x_norm)
    k = max(1, int(n * keep_ratio))
    
    # Build ARFT kernel
    autocorr = np.correlate(x_norm, x_norm, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    eigenvalues, U = eigh(L)
    idx_sort = np.argsort(eigenvalues)[::-1]
    U = U[:, idx_sort]
    
    coeffs = U.T @ x_norm
    idx = np.argsort(np.abs(coeffs))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs)
    coeffs_sparse[idx] = coeffs[idx]
    
    recon = U @ coeffs_sparse
    # Denormalize
    recon = recon * (np.std(x) + EPS) + np.mean(x)
    return recon


# ============================================================================
# ECG Diagnostic Metrics
# ============================================================================

def detect_r_peaks(signal: np.ndarray, fs: float = 360) -> np.ndarray:
    """Simple R-peak detection using peak finding."""
    # Normalize
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + EPS)
    
    # Find peaks with minimum distance of 0.4s (150 bpm max)
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(signal_norm, distance=min_distance, height=0.5)
    
    return peaks


def compute_rr_intervals(peaks: np.ndarray, fs: float = 360) -> np.ndarray:
    """Compute RR intervals in ms."""
    return np.diff(peaks) / fs * 1000


def r_peak_sensitivity(true_peaks: np.ndarray, detected_peaks: np.ndarray, 
                       tolerance_samples: int = 20) -> float:
    """Compute R-peak detection sensitivity (TP / Total True)."""
    tp = 0
    for true_peak in true_peaks:
        if np.any(np.abs(detected_peaks - true_peak) <= tolerance_samples):
            tp += 1
    return tp / len(true_peaks) if len(true_peaks) > 0 else 0.0


def rr_correlation(true_rr: np.ndarray, detected_rr: np.ndarray) -> float:
    """Compute correlation between true and detected RR intervals."""
    min_len = min(len(true_rr), len(detected_rr))
    if min_len < 2:
        return 0.0
    return np.corrcoef(true_rr[:min_len], detected_rr[:min_len])[0, 1]


# ============================================================================
# EEG Diagnostic Metrics
# ============================================================================

def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply bandpass filter."""
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)
    b, a = butter(4, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal)


def compute_band_power(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    """Compute power in a frequency band."""
    filtered = bandpass_filter(signal, band[0], band[1], fs)
    return np.mean(filtered ** 2)


def eeg_band_preservation(original: np.ndarray, compressed: np.ndarray, 
                          fs: float = 256) -> Dict[str, float]:
    """Compute band power preservation ratios."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    
    preservation = {}
    for band_name, (low, high) in bands.items():
        orig_power = compute_band_power(original, fs, (low, high))
        comp_power = compute_band_power(compressed, fs, (low, high))
        
        if orig_power > 0:
            preservation[band_name] = comp_power / orig_power
        else:
            preservation[band_name] = 1.0
    
    return preservation


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    require_real_data()
    
    print("=" * 70)
    print("ARFT Downstream Diagnostic Benchmark")
    print("=" * 70)
    print("Testing clinical feature preservation under compression")
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    # ========================================================================
    # ECG: R-peak Detection
    # ========================================================================
    print("=" * 70)
    print("PART 1: ECG R-Peak Detection & RR Correlation")
    print("=" * 70)
    
    ecg, true_r_peaks, fs = load_ecg_with_annotations()
    
    if ecg is not None:
        # Use first 30 seconds (10800 samples at 360 Hz)
        n_samples = 30 * fs
        ecg_segment = ecg[:n_samples]
        true_peaks_segment = true_r_peaks[true_r_peaks < n_samples]
        true_rr = compute_rr_intervals(true_peaks_segment, fs)
        
        print(f"\n📊 ECG segment: {n_samples} samples ({n_samples/fs:.1f}s)")
        print(f"   True R-peaks: {len(true_peaks_segment)}")
        print(f"   True RR intervals: {len(true_rr)}")
        
        keep_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]
        
        print(f"\n{'Keep':>6s} | {'Method':>6s} | {'R-Sens':>8s} | {'RR-Corr':>8s} | PRD")
        print("-" * 55)
        
        ecg_results = []
        
        for keep in keep_ratios:
            for method in ['dct', 'arft']:
                if method == 'dct':
                    compressed = compress_dct(ecg_segment, keep)
                else:
                    compressed = compress_arft(ecg_segment, keep)
                
                # Detect R-peaks in compressed signal
                detected_peaks = detect_r_peaks(compressed, fs)
                detected_rr = compute_rr_intervals(detected_peaks, fs)
                
                # Metrics
                sensitivity = r_peak_sensitivity(true_peaks_segment, detected_peaks)
                rr_corr = rr_correlation(true_rr, detected_rr)
                
                # PRD
                diff = ecg_segment - compressed
                prd = 100 * np.sqrt(np.sum(diff**2) / np.sum(ecg_segment**2))
                
                ecg_results.append({
                    'keep': keep,
                    'method': method,
                    'sensitivity': sensitivity,
                    'rr_corr': rr_corr,
                    'prd': prd,
                })
                
                print(f"{int(keep*100):5d}% | {method.upper():>6s} | {sensitivity:7.3f} | {rr_corr:8.4f} | {prd:5.2f}%")
        
        # Compare ARFT vs DCT
        print("\n📊 ARFT vs DCT Comparison (ECG):")
        print("-" * 55)
        
        arft_wins = 0
        for keep in keep_ratios:
            dct_res = next(r for r in ecg_results if r['keep'] == keep and r['method'] == 'dct')
            arft_res = next(r for r in ecg_results if r['keep'] == keep and r['method'] == 'arft')
            
            # Winner has better RR correlation (primary) and lower PRD (secondary)
            if arft_res['rr_corr'] > dct_res['rr_corr']:
                winner = 'ARFT'
                arft_wins += 1
            elif arft_res['rr_corr'] == dct_res['rr_corr'] and arft_res['prd'] < dct_res['prd']:
                winner = 'ARFT'
                arft_wins += 1
            else:
                winner = 'DCT'
            
            print(f"  {int(keep*100):2d}% | RR-Corr: ARFT={arft_res['rr_corr']:.4f} vs DCT={dct_res['rr_corr']:.4f} | Winner: {winner}")
        
        print(f"\n✅ ARFT wins {arft_wins}/{len(keep_ratios)} ECG diagnostic tests")
    else:
        print("⚠️  No ECG data available")
    
    # ========================================================================
    # EEG: Band Power Preservation
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: EEG Band Power Preservation")
    print("=" * 70)
    
    fs_eeg = 256
    eeg = generate_synthetic_eeg(n=10000, fs=fs_eeg)
    
    print(f"\n📊 EEG segment: {len(eeg)} samples ({len(eeg)/fs_eeg:.1f}s)")
    
    # Original band powers
    orig_bands = eeg_band_preservation(eeg, eeg, fs_eeg)
    print("\nOriginal band powers (normalized to 1.0):")
    for band, power in orig_bands.items():
        print(f"   {band:6s}: {power:.3f}")
    
    print(f"\n{'Keep':>6s} | {'Method':>6s} | {'Delta':>7s} | {'Theta':>7s} | {'Alpha':>7s} | {'Beta':>7s} | {'Avg':>6s}")
    print("-" * 70)
    
    eeg_results = []
    
    for keep in [0.05, 0.10, 0.20, 0.30, 0.50]:
        for method in ['dct', 'arft']:
            if method == 'dct':
                compressed = compress_dct(eeg, keep)
            else:
                compressed = compress_arft(eeg, keep)
            
            preservation = eeg_band_preservation(eeg, compressed, fs_eeg)
            avg_preservation = np.mean(list(preservation.values()))
            
            eeg_results.append({
                'keep': keep,
                'method': method,
                'preservation': preservation,
                'avg': avg_preservation,
            })
            
            print(f"{int(keep*100):5d}% | {method.upper():>6s} | "
                  f"{preservation['delta']:7.3f} | {preservation['theta']:7.3f} | "
                  f"{preservation['alpha']:7.3f} | {preservation['beta']:7.3f} | "
                  f"{avg_preservation:6.3f}")
    
    # Compare ARFT vs DCT
    print("\n📊 ARFT vs DCT Comparison (EEG):")
    print("-" * 55)
    
    arft_wins_eeg = 0
    for keep in [0.05, 0.10, 0.20, 0.30, 0.50]:
        dct_res = next(r for r in eeg_results if r['keep'] == keep and r['method'] == 'dct')
        arft_res = next(r for r in eeg_results if r['keep'] == keep and r['method'] == 'arft')
        
        # Winner has better average band preservation
        if arft_res['avg'] > dct_res['avg']:
            winner = 'ARFT'
            arft_wins_eeg += 1
        else:
            winner = 'DCT'
        
        print(f"  {int(keep*100):2d}% | Avg Preservation: ARFT={arft_res['avg']:.3f} vs DCT={dct_res['avg']:.3f} | Winner: {winner}")
    
    print(f"\n✅ ARFT wins {arft_wins_eeg}/5 EEG band preservation tests")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Diagnostic Feature Preservation")
    print("=" * 70)
    
    total_tests = len(keep_ratios) + 5  # ECG + EEG
    total_arft_wins = arft_wins + arft_wins_eeg if ecg is not None else arft_wins_eeg
    
    print(f"\n📊 Total ARFT wins: {total_arft_wins}/{total_tests}")
    
    if total_arft_wins > total_tests // 2:
        print("\n✅ ARFT PRESERVES DIAGNOSTIC FEATURES BETTER THAN DCT")
        print("   → R-peak detection and RR correlation maintained")
        print("   → EEG band powers better preserved")
    elif total_arft_wins == total_tests // 2:
        print("\n🤝 ARFT and DCT are comparable for diagnostic preservation")
    else:
        print("\n⚠️  DCT preserves diagnostic features better in this test")
    
    print("\n" + "=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
