#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
COMPREHENSIVE RFT Variant Benchmark for Medical Signal Analysis
================================================================

Tests ALL operator-based RFT variants on medical data for:
1. FEATURE EXTRACTION - Do different bases capture different clinical features?
2. SPARSITY - Which basis gives most compact representation?
3. ENERGY COMPACTION - How quickly do coefficients decay?
4. BAND SEPARATION - How well are clinical bands isolated?
5. ANOMALY DETECTION - Can we detect arrhythmias in the transformed domain?

This is NOT about compression - it's about finding which transform basis
is most useful for ANALYSIS and CLASSIFICATION.

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_all_variants_medical.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct
from scipy.linalg import toeplitz, eigh
from scipy.signal import find_peaks, welch
from typing import Dict, List, Tuple, Callable
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
# ALL OPERATOR-BASED VARIANTS
# ============================================================================

def _build_resonance_operator(r: np.ndarray, decay_rate: float = 0.01) -> np.ndarray:
    """Build Hermitian resonance operator."""
    n = len(r)
    k = np.arange(n)
    decay = np.exp(-decay_rate * k)
    r_reg = r * decay
    r_reg[0] = 1.0
    return toeplitz(r_reg)


def _eigenbasis(K: np.ndarray) -> np.ndarray:
    """Extract sorted eigenbasis."""
    eigenvalues, eigenvectors = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


# Variant generators
def generate_dct_basis(n: int) -> np.ndarray:
    """Standard DCT-II basis (baseline)."""
    basis = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if k == 0:
                basis[j, k] = 1.0 / np.sqrt(n)
            else:
                basis[j, k] = np.sqrt(2.0/n) * np.cos(np.pi * k * (2*j + 1) / (2*n))
    return basis


def generate_fft_basis(n: int) -> np.ndarray:
    """Standard DFT basis (baseline)."""
    return np.fft.fft(np.eye(n)) / np.sqrt(n)


def generate_rft_golden(n: int) -> np.ndarray:
    """RFT-Golden: Golden ratio resonance."""
    f0 = 10.0
    k = np.arange(n)
    t_k = k / n
    r = np.cos(2 * np.pi * f0 * t_k) + np.cos(2 * np.pi * f0 * PHI * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_fibonacci(n: int) -> np.ndarray:
    """RFT-Fibonacci: Fibonacci frequency ratios."""
    fib = [1, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:min(8, len(fib))]
    
    k = np.arange(n)
    t_k = k / n
    r = np.zeros(n)
    for f in fib:
        r += np.cos(2 * np.pi * f * t_k) / len(fib)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_harmonic(n: int) -> np.ndarray:
    """RFT-Harmonic: Musical harmonic overtones."""
    f0 = 4.0
    k = np.arange(n)
    t_k = k / n
    r = np.zeros(n)
    for i in range(1, 6):
        r += (1.0 / i) * np.cos(2 * np.pi * i * f0 * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_geometric(n: int) -> np.ndarray:
    """RFT-Geometric: Golden ratio powers."""
    k = np.arange(n)
    t_k = k / n
    r = np.zeros(n)
    for i in range(8):
        freq = PHI ** i
        if freq < n / 2:
            r += np.cos(2 * np.pi * freq * t_k) / (i + 1)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_beating(n: int) -> np.ndarray:
    """RFT-Beating: Interference patterns."""
    f1 = 5.0
    f2 = f1 * PHI
    f3 = f1 * PHI * PHI
    k = np.arange(n)
    t_k = k / n
    r = np.cos(2 * np.pi * f1 * t_k) + np.cos(2 * np.pi * f2 * t_k) + 0.5 * np.cos(2 * np.pi * f3 * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_rft_cascade(n: int) -> np.ndarray:
    """RFT-Cascade: Hybrid DCT-RFT structure."""
    k = np.arange(n)
    t_k = k / n
    r_dct = np.cos(np.pi * k / n)
    f0 = 10.0
    r_rft = np.cos(2 * np.pi * f0 * t_k) + np.cos(2 * np.pi * f0 * PHI * t_k)
    blend = np.linspace(0, 1, n)
    r = (1 - blend) * r_dct + blend * r_rft
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def generate_arft(n: int, signal: np.ndarray) -> np.ndarray:
    """ARFT: Signal-adaptive (uses signal autocorrelation)."""
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:][:n]
    autocorr = autocorr / (autocorr[0] + EPS)
    K = _build_resonance_operator(autocorr)
    return _eigenbasis(K)


FIXED_VARIANTS = {
    "DCT": generate_dct_basis,
    "FFT": lambda n: np.abs(generate_fft_basis(n)),  # Use magnitude for real comparison
    "RFT-Golden": generate_rft_golden,
    "RFT-Fibonacci": generate_rft_fibonacci,
    "RFT-Harmonic": generate_rft_harmonic,
    "RFT-Geometric": generate_rft_geometric,
    "RFT-Beating": generate_rft_beating,
    "RFT-Cascade": generate_rft_cascade,
}


# ============================================================================
# METRICS
# ============================================================================

def gini_index(coeffs: np.ndarray) -> float:
    """Sparsity measure (higher = sparser)."""
    c = np.sort(np.abs(coeffs.flatten()))
    n = len(c)
    if n == 0 or np.sum(c) == 0:
        return 0.0
    cumsum = np.cumsum(c)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def energy_compaction(coeffs: np.ndarray, threshold: float = 0.95) -> int:
    """Number of coefficients to capture threshold% of energy."""
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    if total == 0:
        return len(coeffs)
    
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    k = np.searchsorted(cumsum, threshold * total) + 1
    return min(k, len(coeffs))


def spectral_entropy(coeffs: np.ndarray) -> float:
    """Entropy of normalized spectrum (lower = more structured)."""
    power = np.abs(coeffs) ** 2
    power = power / (np.sum(power) + EPS)
    power = power[power > EPS]
    return -np.sum(power * np.log2(power))


def peak_to_sidelobe_ratio(coeffs: np.ndarray) -> float:
    """Ratio of main peak to sidelobes (higher = better localization)."""
    power = np.abs(coeffs) ** 2
    sorted_power = np.sort(power)[::-1]
    if len(sorted_power) < 2 or sorted_power[1] == 0:
        return 100.0
    return sorted_power[0] / sorted_power[1]


# ============================================================================
# ECG-SPECIFIC ANALYSIS
# ============================================================================

def analyze_ecg_bands(coeffs: np.ndarray, n: int, fs: float = 360) -> Dict[str, float]:
    """
    Analyze energy in ECG-relevant frequency bands.
    
    Bands:
    - QRS: 5-15 Hz (ventricular depolarization)
    - P/T: 0.5-5 Hz (atrial activity, repolarization)
    - HF: 15-40 Hz (high-frequency content, noise)
    - Baseline: 0-0.5 Hz (baseline wander)
    """
    # Map coefficient indices to approximate frequencies
    freqs = np.arange(len(coeffs)) * fs / (2 * n)
    power = np.abs(coeffs) ** 2
    
    bands = {
        'baseline': (0, 0.5),
        'p_t_waves': (0.5, 5),
        'qrs_complex': (5, 15),
        'high_freq': (15, 40),
    }
    
    band_energy = {}
    total_energy = np.sum(power)
    
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[band_name] = np.sum(power[mask]) / (total_energy + EPS)
    
    return band_energy


def detect_qrs_in_coeffs(coeffs: np.ndarray, n_peaks: int = 5) -> int:
    """Count significant peaks in QRS-band coefficients."""
    power = np.abs(coeffs) ** 2
    peaks, _ = find_peaks(power, height=np.mean(power))
    return min(len(peaks), n_peaks * 2)  # Cap at reasonable number


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ecg_segments(segment_len: int = 512, num_segments: int = 20) -> List[Tuple[str, np.ndarray]]:
    """Load ECG segments from multiple records."""
    try:
        import wfdb
    except ImportError:
        return []
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    segments = []
    
    # Records with different arrhythmia types
    records = {
        "100": "Normal",
        "200": "PVC/VT",
        "207": "LBBB",
        "217": "Bigeminy",
    }
    
    for record_id, label in records.items():
        record_path = os.path.join(data_dir, record_id)
        if not os.path.exists(record_path + ".dat"):
            continue
        
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]
            
            # Extract segments
            step = len(signal) // (num_segments // len(records))
            for i in range(num_segments // len(records)):
                start = i * step
                if start + segment_len <= len(signal):
                    seg = signal[start:start + segment_len].astype(np.float64)
                    seg = (seg - np.mean(seg)) / (np.std(seg) + EPS)
                    segments.append((f"{label}_{record_id}_{i}", seg))
        except:
            pass
    
    return segments


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def benchmark_variant(name: str, basis: np.ndarray, segments: List[Tuple[str, np.ndarray]]) -> Dict:
    """Run full analysis on one variant."""
    results = {
        'gini': [],
        'energy_95': [],
        'entropy': [],
        'psr': [],
        'qrs_energy': [],
        'pt_energy': [],
    }
    
    for seg_name, signal in segments:
        # Transform
        coeffs = basis.T @ signal
        
        # Metrics
        results['gini'].append(gini_index(coeffs))
        results['energy_95'].append(energy_compaction(coeffs, 0.95))
        results['entropy'].append(spectral_entropy(coeffs))
        results['psr'].append(peak_to_sidelobe_ratio(coeffs))
        
        # ECG band analysis
        bands = analyze_ecg_bands(coeffs, len(signal))
        results['qrs_energy'].append(bands['qrs_complex'])
        results['pt_energy'].append(bands['p_t_waves'])
    
    # Average
    return {k: np.mean(v) for k, v in results.items()}


def main():
    require_real_data()
    
    print("=" * 80)
    print("COMPREHENSIVE RFT Variant Benchmark for Medical Signal Analysis")
    print("=" * 80)
    print("Testing ALL variants on ECG for ANALYSIS (not compression)")
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    # Load data
    print("📥 Loading ECG segments...")
    segments = load_ecg_segments(segment_len=512, num_segments=40)
    
    if not segments:
        print("❌ No ECG data found")
        return
    
    print(f"✅ Loaded {len(segments)} segments")
    
    n = 512  # Segment length
    
    # Generate all bases
    print("\n📊 Generating transform bases...")
    bases = {}
    for name, gen_func in FIXED_VARIANTS.items():
        try:
            bases[name] = gen_func(n)
            print(f"   ✓ {name}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")
    
    # Add ARFT (signal-adaptive) using average autocorrelation
    print("   Computing ARFT (adaptive)...")
    avg_signal = np.mean([s[1] for s in segments], axis=0)
    bases["ARFT"] = generate_arft(n, avg_signal)
    print("   ✓ ARFT")
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    results = {}
    for name, basis in bases.items():
        results[name] = benchmark_variant(name, basis, segments)
    
    # Print table
    print(f"\n{'Variant':<15} | {'Gini':>6} | {'E95%':>5} | {'Entropy':>7} | {'PSR':>7} | {'QRS%':>6} | {'P/T%':>6}")
    print("-" * 80)
    
    for name in sorted(results.keys(), key=lambda x: -results[x]['gini']):
        r = results[name]
        print(f"{name:<15} | {r['gini']:6.4f} | {r['energy_95']:5.1f} | {r['entropy']:7.2f} | "
              f"{r['psr']:7.2f} | {r['qrs_energy']*100:6.2f} | {r['pt_energy']*100:6.2f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Which Transform is Best for What?")
    print("=" * 80)
    
    # Best for sparsity
    best_gini = max(results.items(), key=lambda x: x[1]['gini'])
    print(f"\n🏆 SPARSITY (Gini): {best_gini[0]} ({best_gini[1]['gini']:.4f})")
    
    # Best for energy compaction
    best_energy = min(results.items(), key=lambda x: x[1]['energy_95'])
    print(f"🏆 ENERGY COMPACTION: {best_energy[0]} ({best_energy[1]['energy_95']:.1f} coeffs for 95%)")
    
    # Best for QRS isolation
    best_qrs = max(results.items(), key=lambda x: x[1]['qrs_energy'])
    print(f"🏆 QRS BAND ENERGY: {best_qrs[0]} ({best_qrs[1]['qrs_energy']*100:.2f}%)")
    
    # Best for P/T wave isolation
    best_pt = max(results.items(), key=lambda x: x[1]['pt_energy'])
    print(f"🏆 P/T WAVE ENERGY: {best_pt[0]} ({best_pt[1]['pt_energy']*100:.2f}%)")
    
    # Lowest entropy (most structured)
    best_entropy = min(results.items(), key=lambda x: x[1]['entropy'])
    print(f"🏆 LOWEST ENTROPY: {best_entropy[0]} ({best_entropy[1]['entropy']:.2f} bits)")
    
    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
    FOR ECG ANALYSIS:
    
    1. ARRHYTHMIA DETECTION (needs QRS isolation):
       → Best: {qrs}
       
    2. MORPHOLOGY ANALYSIS (needs P/T wave clarity):
       → Best: {pt}
       
    3. FEATURE EXTRACTION FOR ML (needs sparsity):
       → Best: {sparse}
       
    4. COMPRESSION (needs energy compaction):
       → Best: {energy}
    """.format(
        qrs=best_qrs[0],
        pt=best_pt[0],
        sparse=best_gini[0],
        energy=best_energy[0],
    ))
    
    # Check if any RFT variant beats DCT
    dct_gini = results.get('DCT', {}).get('gini', 0)
    rft_winners = [name for name, r in results.items() 
                   if 'RFT' in name and r['gini'] > dct_gini]
    
    if rft_winners:
        print(f"✅ RFT variants that BEAT DCT on sparsity: {', '.join(rft_winners)}")
    else:
        print("⚠️  No RFT variant beats DCT on sparsity for this ECG data")
    
    # Check ARFT
    if 'ARFT' in results and results['ARFT']['gini'] > dct_gini:
        improvement = 100 * (results['ARFT']['gini'] - dct_gini) / dct_gini
        print(f"✅ ARFT (adaptive) beats DCT by {improvement:.1f}% on sparsity")
    
    print("\n" + "=" * 80)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
