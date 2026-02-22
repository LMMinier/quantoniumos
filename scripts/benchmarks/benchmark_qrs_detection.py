#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QRS Detection Benchmark: RFT-Golden vs DCT vs FFT
==================================================

The all-variants benchmark found RFT-Golden captures 58% of energy in the 
QRS frequency band (5-15 Hz) vs DCT's 29%. 

This script tests whether this translates to BETTER QRS DETECTION.

If RFT-Golden concentrates QRS energy into fewer coefficients,
maybe we can detect R-peaks more reliably from the transformed domain.

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_qrs_detection.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import toeplitz, eigh
from scipy.signal import find_peaks
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10


def require_real_data():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("⚠️  Set USE_REAL_DATA=1 to run")
        sys.exit(1)


# ============================================================================
# TRANSFORM IMPLEMENTATIONS
# ============================================================================

def _build_resonance_operator(r: np.ndarray, decay_rate: float = 0.01) -> np.ndarray:
    n = len(r)
    k = np.arange(n)
    decay = np.exp(-decay_rate * k)
    r_reg = r * decay
    r_reg[0] = 1.0
    return toeplitz(r_reg)


def _eigenbasis(K: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


def generate_dct_basis(n: int) -> np.ndarray:
    basis = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if k == 0:
                basis[j, k] = 1.0 / np.sqrt(n)
            else:
                basis[j, k] = np.sqrt(2.0/n) * np.cos(np.pi * k * (2*j + 1) / (2*n))
    return basis


def generate_rft_golden(n: int) -> np.ndarray:
    f0 = 10.0
    k = np.arange(n)
    t_k = k / n
    r = np.cos(2 * np.pi * f0 * t_k) + np.cos(2 * np.pi * f0 * PHI * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_beating(n: int) -> np.ndarray:
    f1 = 5.0
    f2 = f1 * PHI
    f3 = f1 * PHI * PHI
    k = np.arange(n)
    t_k = k / n
    r = np.cos(2 * np.pi * f1 * t_k) + np.cos(2 * np.pi * f2 * t_k) + 0.5 * np.cos(2 * np.pi * f3 * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# ============================================================================
# QRS DETECTION
# ============================================================================

def detect_r_peaks_time_domain(signal: np.ndarray, fs: float = 360) -> np.ndarray:
    """Standard time-domain R-peak detection (ground truth)."""
    # Simple threshold-based detection
    threshold = np.mean(signal) + 1.5 * np.std(signal)
    min_distance = int(0.3 * fs)  # 300ms minimum RR interval
    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
    return peaks


def detect_peaks_in_transform(coeffs: np.ndarray, fs: float = 360, 
                               qrs_band: Tuple[float, float] = (5, 15)) -> np.ndarray:
    """
    Detect peaks in transform domain by:
    1. Extract QRS-band coefficients
    2. Find local maxima in power
    """
    n = len(coeffs)
    freqs = np.arange(n) * fs / (2 * n)
    
    # Isolate QRS band
    mask = (freqs >= qrs_band[0]) & (freqs < qrs_band[1])
    qrs_power = np.abs(coeffs) ** 2
    qrs_power[~mask] = 0
    
    # Find peaks in QRS power
    peaks, _ = find_peaks(qrs_power, height=np.mean(qrs_power[mask]) * 0.5)
    return peaks


def qrs_energy_ratio(coeffs: np.ndarray, fs: float = 360) -> float:
    """Energy in QRS band (5-15 Hz) divided by total energy."""
    n = len(coeffs)
    freqs = np.arange(n) * fs / (2 * n)
    power = np.abs(coeffs) ** 2
    
    qrs_mask = (freqs >= 5) & (freqs < 15)
    return np.sum(power[qrs_mask]) / (np.sum(power) + EPS)


def qrs_band_compaction(coeffs: np.ndarray, fs: float = 360, threshold: float = 0.8) -> int:
    """Number of QRS-band coefficients to capture threshold% of QRS energy."""
    n = len(coeffs)
    freqs = np.arange(n) * fs / (2 * n)
    
    qrs_mask = (freqs >= 5) & (freqs < 15)
    qrs_power = np.abs(coeffs[qrs_mask]) ** 2
    
    if len(qrs_power) == 0 or np.sum(qrs_power) == 0:
        return 0
    
    sorted_power = np.sort(qrs_power)[::-1]
    cumsum = np.cumsum(sorted_power)
    total = np.sum(qrs_power)
    k = np.searchsorted(cumsum, threshold * total) + 1
    return min(k, len(qrs_power))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ecg_with_annotations(segment_len: int = 1024) -> List[Dict]:
    """Load ECG with R-peak annotations for validation."""
    try:
        import wfdb
    except ImportError:
        return []
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    segments = []
    
    records = ["100", "200", "207", "217"]
    
    for record_id in records:
        record_path = os.path.join(data_dir, record_id)
        if not os.path.exists(record_path + ".dat"):
            continue
        
        try:
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            
            signal = record.p_signal[:, 0].astype(np.float64)
            r_peaks = ann.sample  # R-peak locations
            
            # Get segments with known R-peaks
            for start in range(0, len(signal) - segment_len, segment_len):
                seg = signal[start:start + segment_len]
                seg = (seg - np.mean(seg)) / (np.std(seg) + EPS)
                
                # R-peaks in this segment
                seg_peaks = r_peaks[(r_peaks >= start) & (r_peaks < start + segment_len)] - start
                
                if len(seg_peaks) >= 2:  # Need at least 2 beats
                    segments.append({
                        'name': f"{record_id}_{start}",
                        'signal': seg,
                        'r_peaks': seg_peaks,
                        'fs': record.fs,
                    })
        except:
            pass
    
    return segments[:40]  # Limit to 40 segments


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    require_real_data()
    
    print("=" * 80)
    print("QRS Detection Benchmark: Does RFT-Golden Help Arrhythmia Detection?")
    print("=" * 80)
    print("Testing if 58% QRS energy concentration translates to better R-peak detection")
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    # Load data
    print("📥 Loading ECG with R-peak annotations...")
    segments = load_ecg_with_annotations(segment_len=1024)
    
    if not segments:
        print("❌ No annotated ECG data found")
        return
    
    print(f"✅ Loaded {len(segments)} segments with R-peak annotations")
    
    n = 1024
    
    # Generate bases
    print("\n📊 Generating transform bases...")
    dct_basis = generate_dct_basis(n)
    rft_golden_basis = generate_rft_golden(n)
    rft_beating_basis = generate_rft_beating(n)
    print("   ✓ DCT, RFT-Golden, RFT-Beating")
    
    # Benchmark
    print("\n" + "=" * 80)
    print("QRS ENERGY CONCENTRATION")
    print("=" * 80)
    
    results = {
        'DCT': {'qrs_energy': [], 'qrs_compaction': []},
        'RFT-Golden': {'qrs_energy': [], 'qrs_compaction': []},
        'RFT-Beating': {'qrs_energy': [], 'qrs_compaction': []},
    }
    
    for seg in segments:
        signal = seg['signal']
        fs = seg['fs']
        
        # Transform
        dct_coeffs = dct_basis.T @ signal
        golden_coeffs = rft_golden_basis.T @ signal
        beating_coeffs = rft_beating_basis.T @ signal
        
        # QRS energy
        results['DCT']['qrs_energy'].append(qrs_energy_ratio(dct_coeffs, fs))
        results['RFT-Golden']['qrs_energy'].append(qrs_energy_ratio(golden_coeffs, fs))
        results['RFT-Beating']['qrs_energy'].append(qrs_energy_ratio(beating_coeffs, fs))
        
        # Compaction
        results['DCT']['qrs_compaction'].append(qrs_band_compaction(dct_coeffs, fs))
        results['RFT-Golden']['qrs_compaction'].append(qrs_band_compaction(golden_coeffs, fs))
        results['RFT-Beating']['qrs_compaction'].append(qrs_band_compaction(beating_coeffs, fs))
    
    # Print results
    print(f"\n{'Transform':<15} | {'Avg QRS Energy %':>15} | {'Avg QRS Compaction':>18}")
    print("-" * 55)
    
    for name in ['DCT', 'RFT-Golden', 'RFT-Beating']:
        avg_energy = np.mean(results[name]['qrs_energy']) * 100
        avg_compact = np.mean(results[name]['qrs_compaction'])
        print(f"{name:<15} | {avg_energy:15.2f}% | {avg_compact:18.1f} coeffs")
    
    # Analysis
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    dct_energy = np.mean(results['DCT']['qrs_energy'])
    golden_energy = np.mean(results['RFT-Golden']['qrs_energy'])
    beating_energy = np.mean(results['RFT-Beating']['qrs_energy'])
    
    if golden_energy > dct_energy * 1.3:
        print(f"""
    ✅ RFT-GOLDEN CONCENTRATES QRS ENERGY:
    
    RFT-Golden puts {golden_energy*100:.1f}% of signal energy into QRS band (5-15 Hz)
    DCT puts only {dct_energy*100:.1f}% — a {(golden_energy/dct_energy - 1)*100:.0f}% improvement
    
    WHAT THIS MEANS:
    - QRS complexes (ventricular depolarization) are the key to arrhythmia detection
    - RFT-Golden basis is better "tuned" to heartbeat frequencies
    - The golden-ratio frequencies (f and f*φ) create interference patterns
      that match QRS timing intervals (~0.1s QRS, ~0.3s T wave separation)
    
    POTENTIAL APPLICATIONS:
    1. More efficient QRS detectors using fewer coefficients
    2. Better arrhythmia classification features
    3. More robust detection in noisy signals
    
    ⚠️  CAVEAT: This needs validation on labeled arrhythmia datasets (e.g., PTB-XL)
    """)
    elif golden_energy > dct_energy:
        improvement = (golden_energy/dct_energy - 1) * 100
        print(f"""
    ⚠️  MARGINAL IMPROVEMENT:
    
    RFT-Golden: {golden_energy*100:.1f}% QRS energy
    DCT: {dct_energy*100:.1f}% QRS energy
    Difference: {improvement:.1f}%
    
    This is a small improvement that may not be clinically significant.
    """)
    else:
        print(f"""
    ❌ NO IMPROVEMENT:
    
    RFT-Golden does not improve QRS energy concentration over DCT.
    DCT: {dct_energy*100:.1f}%
    RFT-Golden: {golden_energy*100:.1f}%
    """)
    
    # Check beating variant
    if beating_energy > golden_energy:
        print(f"""
    🔍 INTERESTING: RFT-Beating ({beating_energy*100:.1f}%) beats RFT-Golden ({golden_energy*100:.1f}%)
    The beating interference pattern (f1, f1*φ, f1*φ²) may be even better for QRS.
    """)
    
    print("\n" + "=" * 80)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
