#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Adaptive RFT + Domain-Sliced Benchmark

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

Tests:
1. Adaptive RFT: Learns optimal operator parameters per signal family
2. Domain slicing: Performance by arrhythmia type (PVC, VT, AFib, etc.)
3. Finds specific domains where RFT might beat DCT

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_adaptive_rft.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as phi_rft_forward, rft_inverse_square as phi_rft_inverse


# =============================================================================
# ADAPTIVE RFT: Learn optimal parameters per signal
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2


def adaptive_rft_forward(x: np.ndarray, beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Adaptive φ-RFT with tunable parameters."""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    k = np.arange(n)
    
    # Parameterized phase vectors
    D_phi = np.exp(2j * np.pi * beta * (k * PHI % 1))
    C_sig = np.exp(2j * np.pi * sigma * k / n)
    
    X = np.fft.fft(x, norm="ortho")
    return D_phi * (C_sig * X)


def adaptive_rft_inverse(y: np.ndarray, beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Inverse of adaptive φ-RFT."""
    y = np.asarray(y, dtype=np.complex128)
    n = len(y)
    k = np.arange(n)
    
    D_phi = np.exp(2j * np.pi * beta * (k * PHI % 1))
    C_sig = np.exp(2j * np.pi * sigma * k / n)
    
    return np.fft.ifft(np.conj(C_sig) * np.conj(D_phi) * y, norm="ortho")


def find_optimal_params(signal: np.ndarray, keep_ratio: float) -> Tuple[float, float]:
    """Find optimal (beta, sigma) that minimize PRD for a given signal."""
    
    def objective(params):
        beta, sigma = params[0], params[1]
        coeffs = adaptive_rft_forward(signal, beta, sigma)
        n_keep = max(1, int(len(coeffs) * keep_ratio))
        idx = np.argsort(np.abs(coeffs))[::-1]
        sparse = np.zeros_like(coeffs)
        sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
        recon = adaptive_rft_inverse(sparse, beta, sigma)
        prd = np.linalg.norm(signal - np.real(recon)) / np.linalg.norm(signal)
        return prd
    
    # Grid search over parameter space
    best_prd = float('inf')
    best_params = (1.0, 1.0)
    
    for beta in np.linspace(0.5, 2.0, 8):
        for sigma in np.linspace(0.5, 2.0, 8):
            prd = objective([beta, sigma])
            if prd < best_prd:
                best_prd = prd
                best_params = (beta, sigma)
    
    return best_params


def compress_adaptive_rft(signal: np.ndarray, keep_ratio: float, beta: float = None, sigma: float = None) -> Tuple[np.ndarray, float, float]:
    """Compress with adaptive RFT, optionally finding optimal params."""
    if beta is None or sigma is None:
        beta, sigma = find_optimal_params(signal, keep_ratio)
    
    coeffs = adaptive_rft_forward(signal, beta, sigma)
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    recon = np.real(adaptive_rft_inverse(sparse, beta, sigma))
    
    return recon, beta, sigma


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================

def compress_fft(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    coeffs = np.fft.fft(signal, norm='ortho')
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    return np.real(np.fft.ifft(sparse, norm='ortho'))


def compress_dct(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    coeffs = dct(signal, type=2, norm='ortho')
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    return idct(sparse, type=2, norm='ortho')


def compress_phi_rft(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    coeffs = phi_rft_forward(signal)
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    return np.real(phi_rft_inverse(sparse))


# =============================================================================
# METRICS
# =============================================================================

def compute_prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference."""
    return 100.0 * np.linalg.norm(original - reconstructed) / np.linalg.norm(original)


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    error = original - reconstructed
    mse = np.mean(error ** 2)
    signal_power = np.mean(original ** 2)
    if mse > 0:
        return 10 * np.log10(signal_power / mse)
    return float('inf')


# =============================================================================
# ECG FEATURE EXTRACTION FOR SLICING
# =============================================================================

def extract_ecg_features(signal: np.ndarray, fs: float) -> Dict:
    """Extract features to characterize ECG type."""
    # Bandpass filter
    nyq = 0.5 * fs
    b, a = butter(2, [5/nyq, min(15/nyq, 0.99)], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # R-peak detection
    diff = np.diff(filtered)
    squared = diff ** 2
    window = max(1, int(0.15 * fs))
    ma = np.convolve(squared, np.ones(window) / window, mode='same')
    
    min_distance = int(0.2 * fs)
    threshold = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = find_peaks(ma, height=threshold, distance=min_distance)
    
    # RR intervals
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs * 1000  # ms
        rr_mean = np.mean(rr)
        rr_std = np.std(rr)
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0
    else:
        rr_mean, rr_std, rr_cv = 0, 0, 0
    
    # Signal characteristics
    return {
        "hr": 60000 / rr_mean if rr_mean > 0 else 0,
        "rr_cv": rr_cv,  # High CV = irregular rhythm
        "n_beats": len(peaks),
        "signal_energy": np.sum(signal ** 2),
        "high_freq_ratio": np.sum(np.abs(np.fft.rfft(signal)[len(signal)//4:])) / np.sum(np.abs(np.fft.rfft(signal))),
    }


def classify_ecg_type(features: Dict, record_id: str) -> str:
    """Classify ECG type based on features and known record annotations."""
    # MIT-BIH record characteristics (simplified)
    record_types = {
        "100": "normal",
        "101": "normal_with_pvc",
        "200": "pvc_vt",
        "207": "lbbb_vt",
        "208": "mixed_arrhythmia",
        "217": "bigeminy",
        "219": "afib",
        "221": "aflutter",
    }
    return record_types.get(record_id, "unknown")


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

@dataclass
class SlicedResult:
    transform: str
    record_id: str
    ecg_type: str
    window_idx: int
    keep_ratio: float
    prd: float
    snr: float
    params: Optional[Tuple[float, float]] = None


def load_ecg_records() -> List[Tuple[str, np.ndarray, float]]:
    """Load all MIT-BIH records."""
    records = []
    mitbih_dir = ROOT / "data" / "physionet" / "mitbih"
    
    if not mitbih_dir.exists():
        return records
    
    try:
        import wfdb
    except ImportError:
        print("⚠ wfdb not installed")
        return records
    
    for dat_file in mitbih_dir.glob("*.dat"):
        rec_id = dat_file.stem
        try:
            record = wfdb.rdrecord(str(mitbih_dir / rec_id))
            sig = record.p_signal[:, 0]
            fs = record.fs
            records.append((rec_id, sig, fs))
        except Exception as e:
            print(f"  Warning: Could not load {rec_id}: {e}")
    
    return records


def run_sliced_benchmark():
    """Run benchmark sliced by ECG type."""
    print("=" * 80)
    print("ADAPTIVE RFT + DOMAIN-SLICED BENCHMARK")
    print("⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 80)
    
    records = load_ecg_records()
    print(f"\nLoaded {len(records)} MIT-BIH records")
    
    if not records:
        print("No data found. Run: USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py")
        return
    
    results: List[SlicedResult] = []
    keep_ratios = [0.10, 0.20, 0.30]
    window_size = 2048
    
    transforms = {
        "FFT": compress_fft,
        "DCT": compress_dct,
        "φ-RFT": compress_phi_rft,
    }
    
    for rec_id, signal, fs in records:
        ecg_type = classify_ecg_type({}, rec_id)
        print(f"\nProcessing {rec_id} ({ecg_type})...")
        
        # Take windows at different positions
        window_starts = [0, 50000, 100000, 200000, 300000]
        
        for w_idx, start in enumerate(window_starts):
            if start + window_size > len(signal):
                continue
            
            window = signal[start:start+window_size].astype(np.float64)
            
            for keep_ratio in keep_ratios:
                # Standard transforms
                for tname, tfunc in transforms.items():
                    try:
                        recon = tfunc(window, keep_ratio)
                        prd = compute_prd(window, recon)
                        snr = compute_snr(window, recon)
                        results.append(SlicedResult(
                            transform=tname,
                            record_id=rec_id,
                            ecg_type=ecg_type,
                            window_idx=w_idx,
                            keep_ratio=keep_ratio,
                            prd=prd,
                            snr=snr,
                        ))
                    except Exception as e:
                        pass
                
                # Adaptive RFT
                try:
                    recon, beta, sigma = compress_adaptive_rft(window, keep_ratio)
                    prd = compute_prd(window, recon)
                    snr = compute_snr(window, recon)
                    results.append(SlicedResult(
                        transform="Adaptive RFT",
                        record_id=rec_id,
                        ecg_type=ecg_type,
                        window_idx=w_idx,
                        keep_ratio=keep_ratio,
                        prd=prd,
                        snr=snr,
                        params=(beta, sigma),
                    ))
                except Exception as e:
                    pass
    
    print_sliced_results(results)
    return results


def print_sliced_results(results: List[SlicedResult]):
    """Print results sliced by ECG type."""
    
    transforms = ["FFT", "DCT", "φ-RFT", "Adaptive RFT"]
    ecg_types = sorted(set(r.ecg_type for r in results))
    keep_ratios = sorted(set(r.keep_ratio for r in results))
    
    print("\n" + "=" * 80)
    print("RESULTS BY ECG TYPE (Arrhythmia Domain Slicing)")
    print("=" * 80)
    
    # For each ECG type, show best transform
    for keep_ratio in keep_ratios:
        print(f"\n### Compression: Keep {keep_ratio*100:.0f}% coefficients")
        print(f"{'ECG Type':<20} {'FFT PRD':<12} {'DCT PRD':<12} {'φ-RFT PRD':<12} {'Adapt PRD':<12} {'Winner':<15}")
        print("-" * 85)
        
        for ecg_type in ecg_types:
            row = {"ecg_type": ecg_type}
            best_prd = float('inf')
            winner = ""
            
            for tname in transforms:
                matching = [r for r in results 
                           if r.transform == tname 
                           and r.ecg_type == ecg_type 
                           and r.keep_ratio == keep_ratio]
                if matching:
                    avg_prd = np.mean([r.prd for r in matching])
                    row[tname] = avg_prd
                    if avg_prd < best_prd:
                        best_prd = avg_prd
                        winner = tname
                else:
                    row[tname] = float('nan')
            
            print(f"{ecg_type:<20} {row.get('FFT', float('nan')):<12.2f} {row.get('DCT', float('nan')):<12.2f} {row.get('φ-RFT', float('nan')):<12.2f} {row.get('Adaptive RFT', float('nan')):<12.2f} {winner:<15}")
    
    # Summary: Where does RFT beat DCT?
    print("\n" + "=" * 80)
    print("ANALYSIS: Where does RFT beat DCT?")
    print("=" * 80)
    
    rft_wins = []
    dct_wins = []
    
    for keep_ratio in keep_ratios:
        for ecg_type in ecg_types:
            dct_results = [r for r in results 
                          if r.transform == "DCT" 
                          and r.ecg_type == ecg_type 
                          and r.keep_ratio == keep_ratio]
            adapt_results = [r for r in results 
                            if r.transform == "Adaptive RFT" 
                            and r.ecg_type == ecg_type 
                            and r.keep_ratio == keep_ratio]
            
            if dct_results and adapt_results:
                dct_prd = np.mean([r.prd for r in dct_results])
                adapt_prd = np.mean([r.prd for r in adapt_results])
                
                if adapt_prd < dct_prd:
                    improvement = (dct_prd - adapt_prd) / dct_prd * 100
                    rft_wins.append((ecg_type, keep_ratio, improvement, adapt_prd, dct_prd))
                else:
                    degradation = (adapt_prd - dct_prd) / dct_prd * 100
                    dct_wins.append((ecg_type, keep_ratio, degradation))
    
    if rft_wins:
        print("\n✓ Adaptive RFT beats DCT in these domains:")
        for ecg_type, keep_ratio, improvement, adapt_prd, dct_prd in sorted(rft_wins, key=lambda x: -x[2]):
            print(f"  {ecg_type} @ {keep_ratio*100:.0f}%: Adaptive RFT PRD={adapt_prd:.2f}% vs DCT PRD={dct_prd:.2f}% ({improvement:.1f}% better)")
    else:
        print("\n✗ Adaptive RFT did not beat DCT in any domain")
    
    if dct_wins:
        print(f"\nDCT wins in {len(dct_wins)} domain/compression combinations")
    
    # Optimal parameters by ECG type
    print("\n" + "=" * 80)
    print("LEARNED PARAMETERS (Adaptive RFT)")
    print("=" * 80)
    
    for ecg_type in ecg_types:
        adapt_results = [r for r in results 
                        if r.transform == "Adaptive RFT" 
                        and r.ecg_type == ecg_type 
                        and r.params is not None]
        if adapt_results:
            betas = [r.params[0] for r in adapt_results]
            sigmas = [r.params[1] for r in adapt_results]
            print(f"{ecg_type}: β={np.mean(betas):.2f}±{np.std(betas):.2f}, σ={np.mean(sigmas):.2f}±{np.std(sigmas):.2f}")
    
    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


def main():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("ERROR: USE_REAL_DATA=1 required")
        sys.exit(1)
    
    run_sliced_benchmark()


if __name__ == "__main__":
    main()
