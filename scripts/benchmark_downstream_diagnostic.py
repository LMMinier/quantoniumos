#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Downstream Diagnostic Benchmark: Does compression affect clinical accuracy?

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

Tests whether RFT compression preserves diagnostic features better than FFT/DCT:
1. ECG Arrhythmia Detection: R-peak detection + simple beat classification
2. EEG Sleep Staging: Power spectral features for wake/NREM/REM classification
3. Domain slicing: Performance by arrhythmia type, sleep stage, signal characteristics

Key question: At the same compression ratio, which transform preserves diagnostic accuracy best?

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_downstream_diagnostic.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as phi_rft_forward, rft_inverse_square as phi_rft_inverse
from algorithms.rft.variants.operator_variants import rft_forward as canonical_rft_forward, rft_inverse as canonical_rft_inverse


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


def compress_canonical_rft(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    coeffs, Phi = canonical_rft_forward(signal.astype(np.float64))
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    return np.real(canonical_rft_inverse(sparse, Phi))


TRANSFORMS = {
    "FFT": compress_fft,
    "DCT": compress_dct,
    "Canonical RFT": compress_canonical_rft,
}


# =============================================================================
# ECG ARRHYTHMIA DETECTION
# =============================================================================

def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Bandpass filter for ECG."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Simple R-peak detection using derivative + threshold."""
    # Bandpass filter
    filtered = bandpass_filter(signal, fs, 5, 15)
    
    # Derivative
    diff = np.diff(filtered)
    squared = diff ** 2
    
    # Moving average
    window = int(0.15 * fs)
    if window < 1:
        window = 1
    ma = np.convolve(squared, np.ones(window) / window, mode='same')
    
    # Find peaks with minimum distance of 200ms
    min_distance = int(0.2 * fs)
    threshold = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = find_peaks(ma, height=threshold, distance=min_distance)
    
    return peaks


def compute_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Compute RR intervals in ms."""
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000  # ms


def classify_rhythm(rr_intervals: np.ndarray) -> str:
    """Simple rhythm classification based on RR variability."""
    if len(rr_intervals) < 3:
        return "insufficient"
    
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    cv = std_rr / mean_rr if mean_rr > 0 else 0
    
    # Very simple heuristics
    if mean_rr < 500:  # HR > 120
        return "tachycardia"
    elif mean_rr > 1000:  # HR < 60
        return "bradycardia"
    elif cv > 0.15:  # High variability
        return "irregular"
    else:
        return "normal"


def ecg_diagnostic_accuracy(original: np.ndarray, reconstructed: np.ndarray, fs: float) -> Dict:
    """Compare ECG diagnostic features before/after compression."""
    # Detect peaks in both
    peaks_orig = detect_r_peaks(original, fs)
    peaks_recon = detect_r_peaks(reconstructed, fs)
    
    # Peak detection accuracy
    # Count how many original peaks have a matching reconstructed peak within 50ms
    tolerance = int(0.05 * fs)  # 50ms
    matched = 0
    for p in peaks_orig:
        if any(abs(peaks_recon - p) <= tolerance):
            matched += 1
    
    sensitivity = matched / len(peaks_orig) if len(peaks_orig) > 0 else 0
    
    # False positives
    false_pos = 0
    for p in peaks_recon:
        if not any(abs(peaks_orig - p) <= tolerance):
            false_pos += 1
    precision = matched / len(peaks_recon) if len(peaks_recon) > 0 else 0
    
    # RR interval correlation
    rr_orig = compute_rr_intervals(peaks_orig, fs)
    rr_recon = compute_rr_intervals(peaks_recon, fs)
    
    if len(rr_orig) >= 3 and len(rr_recon) >= 3:
        # Align by taking min length
        min_len = min(len(rr_orig), len(rr_recon))
        rr_corr, _ = pearsonr(rr_orig[:min_len], rr_recon[:min_len])
    else:
        rr_corr = 0
    
    # Rhythm classification match
    rhythm_orig = classify_rhythm(rr_orig)
    rhythm_recon = classify_rhythm(rr_recon)
    rhythm_match = 1 if rhythm_orig == rhythm_recon else 0
    
    return {
        "peak_sensitivity": sensitivity,
        "peak_precision": precision,
        "rr_correlation": rr_corr if not np.isnan(rr_corr) else 0,
        "rhythm_match": rhythm_match,
        "rhythm_orig": rhythm_orig,
        "rhythm_recon": rhythm_recon,
        "n_peaks_orig": len(peaks_orig),
        "n_peaks_recon": len(peaks_recon),
    }


# =============================================================================
# EEG SLEEP STAGING
# =============================================================================

def compute_band_power(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    """Compute power in a frequency band using FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1/fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2 / n
    
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.sum(psd[band_mask])


def extract_sleep_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract sleep staging features (band powers)."""
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    }
    
    features = {}
    total_power = 0
    for name, band in bands.items():
        power = compute_band_power(signal, fs, band)
        features[name] = power
        total_power += power
    
    # Relative powers
    if total_power > 0:
        for name in bands:
            features[f"{name}_rel"] = features[name] / total_power
    
    # Ratios useful for sleep staging
    if features["beta"] > 0:
        features["delta_beta_ratio"] = features["delta"] / features["beta"]
        features["theta_beta_ratio"] = features["theta"] / features["beta"]
    
    return features


def classify_sleep_stage(features: Dict[str, float]) -> str:
    """Simple sleep stage classification based on band ratios."""
    delta_rel = features.get("delta_rel", 0)
    alpha_rel = features.get("alpha_rel", 0)
    beta_rel = features.get("beta_rel", 0)
    
    # Very simplified heuristics
    if delta_rel > 0.5:
        return "deep_sleep"  # N3
    elif alpha_rel > 0.3:
        return "light_sleep"  # N1/N2
    elif beta_rel > 0.3:
        return "wake"
    else:
        return "rem"


def eeg_diagnostic_accuracy(original: np.ndarray, reconstructed: np.ndarray, fs: float) -> Dict:
    """Compare EEG diagnostic features before/after compression."""
    feat_orig = extract_sleep_features(original, fs)
    feat_recon = extract_sleep_features(reconstructed, fs)
    
    # Band power correlations
    band_errors = {}
    for band in ["delta", "theta", "alpha", "beta"]:
        if feat_orig[band] > 0:
            rel_error = abs(feat_orig[band] - feat_recon[band]) / feat_orig[band]
            band_errors[f"{band}_error"] = rel_error
    
    # Feature vector correlation
    orig_vec = np.array([feat_orig.get(f"{b}_rel", 0) for b in ["delta", "theta", "alpha", "beta"]])
    recon_vec = np.array([feat_recon.get(f"{b}_rel", 0) for b in ["delta", "theta", "alpha", "beta"]])
    
    if np.std(orig_vec) > 0 and np.std(recon_vec) > 0:
        feat_corr, _ = pearsonr(orig_vec, recon_vec)
    else:
        feat_corr = 1.0
    
    # Sleep stage match
    stage_orig = classify_sleep_stage(feat_orig)
    stage_recon = classify_sleep_stage(feat_recon)
    stage_match = 1 if stage_orig == stage_recon else 0
    
    return {
        "feature_correlation": feat_corr if not np.isnan(feat_corr) else 0,
        "stage_match": stage_match,
        "stage_orig": stage_orig,
        "stage_recon": stage_recon,
        **band_errors,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

@dataclass
class DiagnosticResult:
    transform: str
    dataset: str
    domain: str
    keep_ratio: float
    metrics: Dict = field(default_factory=dict)


def load_ecg_with_annotations() -> List[Tuple[str, np.ndarray, float, str]]:
    """Load MIT-BIH ECG with rhythm annotations."""
    signals = []
    mitbih_dir = ROOT / "data" / "physionet" / "mitbih"
    
    if not mitbih_dir.exists():
        return signals
    
    try:
        import wfdb
    except ImportError:
        print("⚠ wfdb not installed")
        return signals
    
    # MIT-BIH record types (simplified)
    record_types = {
        "100": "normal",
        "101": "normal",  # Has some PVCs
    }
    
    for rec_id, rhythm_type in record_types.items():
        if (mitbih_dir / f"{rec_id}.dat").exists():
            record = wfdb.rdrecord(str(mitbih_dir / rec_id))
            sig = record.p_signal[:, 0]
            fs = record.fs
            
            # Take multiple windows
            window_size = 2048  # ~5.7 seconds at 360 Hz
            for i, start in enumerate([0, 10000, 50000, 100000, 200000]):
                if start + window_size <= len(sig):
                    window = sig[start:start+window_size]
                    signals.append((f"ECG-{rec_id}-{rhythm_type}-w{i}", window, fs, rhythm_type))
    
    return signals


def load_eeg_with_stages() -> List[Tuple[str, np.ndarray, float, str]]:
    """Load Sleep-EDF EEG with approximate stage labels."""
    signals = []
    sleepedf_dir = ROOT / "data" / "physionet" / "sleepedf"
    
    if not sleepedf_dir.exists():
        return signals
    
    try:
        import pyedflib
    except ImportError:
        print("⚠ pyedflib not installed")
        return signals
    
    for edf_file in list(sleepedf_dir.glob("*-PSG.edf"))[:2]:
        f = pyedflib.EdfReader(str(edf_file))
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        eeg_idx = labels.index("EEG Fpz-Cz") if "EEG Fpz-Cz" in labels else 0
        eeg = f.readSignal(eeg_idx)
        fs = f.getSampleFrequency(eeg_idx)
        f.close()
        
        # 30-second epochs (standard sleep staging)
        epoch_samples = int(30 * fs)
        
        # Sample epochs from different parts of the recording
        # Early = more likely wake, middle = more likely sleep
        epoch_labels = [
            (1000, "early"),      # Likely wake
            (100000, "middle"),   # Likely sleep
            (300000, "late"),     # Mixed
        ]
        
        for start, label in epoch_labels:
            if start + epoch_samples <= len(eeg):
                window = eeg[start:start+epoch_samples]
                signals.append((f"EEG-{edf_file.stem[:7]}-{label}", window, fs, label))
    
    return signals


def run_benchmark():
    """Run the full diagnostic benchmark."""
    print("=" * 80)
    print("DOWNSTREAM DIAGNOSTIC BENCHMARK")
    print("Does compression preserve clinical features?")
    print("⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 80)
    
    # Load data
    ecg_signals = load_ecg_with_annotations()
    eeg_signals = load_eeg_with_stages()
    
    print(f"\nLoaded {len(ecg_signals)} ECG segments, {len(eeg_signals)} EEG epochs")
    
    if not ecg_signals and not eeg_signals:
        print("No data found. Run fetch scripts first.")
        return
    
    keep_ratios = [0.10, 0.20, 0.30, 0.50]
    results: List[DiagnosticResult] = []
    
    # ECG benchmark
    if ecg_signals:
        print("\n" + "=" * 80)
        print("ECG ARRHYTHMIA DETECTION BENCHMARK")
        print("=" * 80)
        
        for name, signal, fs, rhythm_type in ecg_signals:
            for keep_ratio in keep_ratios:
                for tname, tfunc in TRANSFORMS.items():
                    try:
                        recon = tfunc(signal, keep_ratio)
                        metrics = ecg_diagnostic_accuracy(signal, recon, fs)
                        metrics["rhythm_type"] = rhythm_type
                        results.append(DiagnosticResult(
                            transform=tname,
                            dataset=name,
                            domain="ECG",
                            keep_ratio=keep_ratio,
                            metrics=metrics,
                        ))
                    except Exception as e:
                        print(f"  Warning: {tname} failed on {name}: {e}")
    
    # EEG benchmark
    if eeg_signals:
        print("\n" + "=" * 80)
        print("EEG SLEEP STAGING BENCHMARK")
        print("=" * 80)
        
        for name, signal, fs, stage_hint in eeg_signals:
            for keep_ratio in keep_ratios:
                for tname, tfunc in TRANSFORMS.items():
                    try:
                        recon = tfunc(signal, keep_ratio)
                        metrics = eeg_diagnostic_accuracy(signal, recon, fs)
                        metrics["stage_hint"] = stage_hint
                        results.append(DiagnosticResult(
                            transform=tname,
                            dataset=name,
                            domain="EEG",
                            keep_ratio=keep_ratio,
                            metrics=metrics,
                        ))
                    except Exception as e:
                        print(f"  Warning: {tname} failed on {name}: {e}")
    
    # Print results
    print_diagnostic_results(results)
    
    return results


def print_diagnostic_results(results: List[DiagnosticResult]):
    """Print diagnostic benchmark results with domain slicing."""
    
    # === ECG Results ===
    ecg_results = [r for r in results if r.domain == "ECG"]
    if ecg_results:
        print("\n" + "=" * 80)
        print("ECG ARRHYTHMIA DETECTION RESULTS")
        print("=" * 80)
        
        print("\n### Overall by Transform and Compression Level")
        print(f"{'Transform':<16} {'Keep%':<8} {'Peak Sens':<12} {'Peak Prec':<12} {'RR Corr':<12} {'Rhythm Match':<12}")
        print("-" * 72)
        
        for keep_ratio in sorted(set(r.keep_ratio for r in ecg_results)):
            for tname in TRANSFORMS.keys():
                matching = [r for r in ecg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    avg_sens = np.mean([r.metrics["peak_sensitivity"] for r in matching])
                    avg_prec = np.mean([r.metrics["peak_precision"] for r in matching])
                    avg_rr = np.mean([r.metrics["rr_correlation"] for r in matching])
                    avg_rhythm = np.mean([r.metrics["rhythm_match"] for r in matching])
                    print(f"{tname:<16} {keep_ratio*100:<8.0f} {avg_sens:<12.3f} {avg_prec:<12.3f} {avg_rr:<12.3f} {avg_rhythm:<12.3f}")
            print()
        
        # Find winner at each compression level
        print("\n### WINNER BY COMPRESSION LEVEL (ECG)")
        print("-" * 50)
        for keep_ratio in sorted(set(r.keep_ratio for r in ecg_results)):
            best_transform = None
            best_score = -1
            for tname in TRANSFORMS.keys():
                matching = [r for r in ecg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    # Combined score: sensitivity + precision + RR correlation + rhythm match
                    score = (np.mean([r.metrics["peak_sensitivity"] for r in matching]) +
                            np.mean([r.metrics["peak_precision"] for r in matching]) +
                            np.mean([r.metrics["rr_correlation"] for r in matching]) +
                            np.mean([r.metrics["rhythm_match"] for r in matching])) / 4
                    if score > best_score:
                        best_score = score
                        best_transform = tname
            print(f"  {keep_ratio*100:.0f}% coefficients: {best_transform} (score: {best_score:.3f})")
    
    # === EEG Results ===
    eeg_results = [r for r in results if r.domain == "EEG"]
    if eeg_results:
        print("\n" + "=" * 80)
        print("EEG SLEEP STAGING RESULTS")
        print("=" * 80)
        
        print("\n### Overall by Transform and Compression Level")
        print(f"{'Transform':<16} {'Keep%':<8} {'Feat Corr':<12} {'Stage Match':<12} {'Delta Err':<12} {'Alpha Err':<12}")
        print("-" * 72)
        
        for keep_ratio in sorted(set(r.keep_ratio for r in eeg_results)):
            for tname in TRANSFORMS.keys():
                matching = [r for r in eeg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    avg_feat = np.mean([r.metrics["feature_correlation"] for r in matching])
                    avg_stage = np.mean([r.metrics["stage_match"] for r in matching])
                    avg_delta = np.mean([r.metrics.get("delta_error", 0) for r in matching])
                    avg_alpha = np.mean([r.metrics.get("alpha_error", 0) for r in matching])
                    print(f"{tname:<16} {keep_ratio*100:<8.0f} {avg_feat:<12.3f} {avg_stage:<12.3f} {avg_delta:<12.3f} {avg_alpha:<12.3f}")
            print()
        
        # Find winner at each compression level
        print("\n### WINNER BY COMPRESSION LEVEL (EEG)")
        print("-" * 50)
        for keep_ratio in sorted(set(r.keep_ratio for r in eeg_results)):
            best_transform = None
            best_score = -1
            for tname in TRANSFORMS.keys():
                matching = [r for r in eeg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    # Score based on feature correlation and stage match
                    score = (np.mean([r.metrics["feature_correlation"] for r in matching]) +
                            np.mean([r.metrics["stage_match"] for r in matching])) / 2
                    if score > best_score:
                        best_score = score
                        best_transform = tname
            print(f"  {keep_ratio*100:.0f}% coefficients: {best_transform} (score: {best_score:.3f})")
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: CLINICAL IMPACT")
    print("=" * 80)
    
    print("\nKey Questions Answered:")
    print("-" * 50)
    
    # ECG summary
    if ecg_results:
        # At 30% compression, who wins?
        for keep_ratio in [0.30]:
            print(f"\nECG @ {keep_ratio*100:.0f}% compression:")
            for tname in TRANSFORMS.keys():
                matching = [r for r in ecg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    sens = np.mean([r.metrics["peak_sensitivity"] for r in matching])
                    rhythm = np.mean([r.metrics["rhythm_match"] for r in matching])
                    print(f"  {tname:<16}: R-peak sensitivity={sens:.1%}, Rhythm preserved={rhythm:.1%}")
    
    if eeg_results:
        for keep_ratio in [0.30]:
            print(f"\nEEG @ {keep_ratio*100:.0f}% compression:")
            for tname in TRANSFORMS.keys():
                matching = [r for r in eeg_results 
                           if r.transform == tname and r.keep_ratio == keep_ratio]
                if matching:
                    feat = np.mean([r.metrics["feature_correlation"] for r in matching])
                    stage = np.mean([r.metrics["stage_match"] for r in matching])
                    print(f"  {tname:<16}: Feature correlation={feat:.3f}, Stage preserved={stage:.1%}")
    
    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


def main():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("ERROR: USE_REAL_DATA=1 required")
        sys.exit(1)
    
    run_benchmark()


if __name__ == "__main__":
    main()
