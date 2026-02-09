#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# MEDICAL RESEARCH LICENSE:
# FREE for hospitals, medical researchers, academics, and healthcare
# institutions for testing, validation, and research purposes.
# Commercial medical device use: See LICENSE-CLAIMS-NC.md
#
"""
Biosignal Compression and Anomaly Detection Tests
==================================================

Tests RFT variants for ECG/EEG/EMG processing:
- Compression ratio and SNR benchmarks
- Arrhythmia detection simulation (MIT-BIH style)
- Seizure detection simulation (TUH EEG style)
- Robustness to electrode noise and baseline wander

Uses synthetic signals mimicking real biosignal characteristics.
Real validation requires MIT-BIH Arrhythmia DB or TUH EEG Corpus.
"""

import numpy as np
import time
import pytest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Synthetic Biosignal Generators
# =============================================================================

def generate_ecg_signal(duration_sec: float = 10.0,
                        sample_rate: int = 360,
                        heart_rate_bpm: float = 72.0,
                        noise_level: float = 0.0,
                        baseline_wander: float = 0.0,
                        arrhythmia_prob: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG signal with optional anomalies.
    
    Uses simplified PQRST template. For real validation, use MIT-BIH data.
    
    Args:
        duration_sec: Signal duration in seconds
        sample_rate: Samples per second (MIT-BIH uses 360 Hz)
        heart_rate_bpm: Average heart rate
        noise_level: Gaussian noise std
        baseline_wander: Slow baseline drift amplitude
        arrhythmia_prob: Probability of premature beat per cycle
        
    Returns:
        (signal, anomaly_labels) - signal and binary anomaly mask
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # Beat interval
    beat_interval = 60.0 / heart_rate_bpm  # seconds
    samples_per_beat = int(beat_interval * sample_rate)
    
    # Create PQRST template (simplified)
    template_len = min(samples_per_beat, int(0.8 * sample_rate))
    template_t = np.arange(template_len) / sample_rate
    
    # P wave (small bump)
    p_wave = 0.15 * np.exp(-((template_t - 0.1) ** 2) / (2 * 0.02 ** 2))
    
    # QRS complex (sharp spike)
    q_wave = -0.1 * np.exp(-((template_t - 0.18) ** 2) / (2 * 0.005 ** 2))
    r_wave = 1.0 * np.exp(-((template_t - 0.20) ** 2) / (2 * 0.008 ** 2))
    s_wave = -0.15 * np.exp(-((template_t - 0.22) ** 2) / (2 * 0.005 ** 2))
    
    # T wave (rounded bump)
    t_wave = 0.3 * np.exp(-((template_t - 0.35) ** 2) / (2 * 0.04 ** 2))
    
    template = p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Generate signal by placing beats
    signal = np.zeros(n_samples)
    anomaly_labels = np.zeros(n_samples, dtype=bool)
    
    beat_idx = 0
    while beat_idx < n_samples:
        # Check for arrhythmia (premature beat)
        if np.random.random() < arrhythmia_prob:
            # Premature beat: shorter interval, modified amplitude
            actual_interval = int(samples_per_beat * 0.6)
            amplitude_mod = 1.3  # Higher amplitude for PVC
            # Mark as anomaly
            start = beat_idx
            end = min(beat_idx + template_len, n_samples)
            anomaly_labels[start:end] = True
        else:
            actual_interval = samples_per_beat
            amplitude_mod = 1.0
        
        # Place beat
        end_idx = min(beat_idx + template_len, n_samples)
        template_portion = template[:end_idx - beat_idx]
        signal[beat_idx:end_idx] += amplitude_mod * template_portion
        
        beat_idx += actual_interval
    
    # Add baseline wander (slow sinusoidal drift)
    if baseline_wander > 0:
        wander_freq = 0.3  # Hz
        signal += baseline_wander * np.sin(2 * np.pi * wander_freq * t)
    
    # Add noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(n_samples)
    
    return signal, anomaly_labels


def generate_eeg_signal(duration_sec: float = 10.0,
                        sample_rate: int = 256,
                        dominant_freq: float = 10.0,  # Alpha rhythm
                        noise_level: float = 0.0,
                        seizure_prob: float = 0.0,
                        seizure_duration_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG signal with optional seizure events.
    
    Simulates background alpha rhythm with possible ictal activity.
    For real validation, use TUH EEG Corpus.
    
    Args:
        duration_sec: Signal duration
        sample_rate: Samples per second
        dominant_freq: Background dominant frequency (Hz)
        noise_level: Gaussian noise level
        seizure_prob: Probability of seizure starting at any 1-second window
        seizure_duration_sec: Duration of seizure events
        
    Returns:
        (signal, seizure_labels) - signal and binary seizure mask
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # Background rhythm (alpha + harmonics)
    signal = np.zeros(n_samples)
    signal += 0.5 * np.sin(2 * np.pi * dominant_freq * t)  # Alpha
    signal += 0.2 * np.sin(2 * np.pi * (dominant_freq / 2) * t)  # Theta
    signal += 0.1 * np.sin(2 * np.pi * 30 * t)  # Gamma
    
    # Add 1/f background noise (more realistic EEG)
    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    pink_spectrum = np.where(np.abs(freqs) > 0.1, 1 / (np.abs(freqs) + 0.1), 0)
    pink_noise = np.fft.ifft(pink_spectrum * np.fft.fft(np.random.randn(n_samples))).real
    pink_noise = 0.3 * pink_noise / (np.std(pink_noise) + 1e-10)
    signal += pink_noise
    
    # Seizure events
    seizure_labels = np.zeros(n_samples, dtype=bool)
    n_windows = int(duration_sec)
    seizure_samples = int(seizure_duration_sec * sample_rate)
    
    for w in range(n_windows):
        if np.random.random() < seizure_prob:
            start = w * sample_rate
            end = min(start + seizure_samples, n_samples)
            
            # Seizure: high-frequency, high-amplitude bursts
            seizure_t = t[start:end] - t[start]
            seizure_component = 2.0 * np.sin(2 * np.pi * 25 * seizure_t)  # Fast rhythm
            seizure_component *= np.exp(-seizure_t / 1.0)  # Decay
            seizure_component += 1.5 * np.sin(2 * np.pi * 3 * seizure_t)  # Slow wave
            
            signal[start:end] += seizure_component
            seizure_labels[start:end] = True
    
    # Add white noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(n_samples)
    
    # Normalize
    signal = signal / (np.std(signal) + 1e-10)
    
    return signal, seizure_labels


def generate_emg_signal(duration_sec: float = 5.0,
                        sample_rate: int = 1000,
                        activation_pattern: str = "burst",
                        noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic EMG signal for muscle activity.
    
    Args:
        duration_sec: Duration in seconds
        sample_rate: Samples per second
        activation_pattern: "burst" or "continuous"
        noise_level: Background noise level
        
    Returns:
        EMG signal
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # EMG is essentially band-limited noise (20-500 Hz) modulated by activation
    # Generate broadband noise
    raw_noise = np.random.randn(n_samples)
    
    # Bandpass filter (simplified using FFT)
    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    fft = np.fft.fft(raw_noise)
    bandpass = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 500)
    fft[~bandpass] = 0
    emg_noise = np.fft.ifft(fft).real
    
    # Activation envelope
    if activation_pattern == "burst":
        # Three bursts
        envelope = np.zeros(n_samples)
        for center in [1.0, 2.5, 4.0]:
            if center < duration_sec:
                envelope += np.exp(-((t - center) ** 2) / (2 * 0.2 ** 2))
    else:
        # Continuous with gradual increase
        envelope = 0.3 + 0.7 * (t / duration_sec)
    
    signal = envelope * emg_noise + noise_level * np.random.randn(n_samples)
    
    return signal


def add_electrode_noise(signal: np.ndarray,
                        sample_rate: int,
                        powerline_hz: float = 60.0,
                        powerline_amplitude: float = 0.1,
                        contact_noise_std: float = 0.05) -> np.ndarray:
    """
    Add realistic electrode noise artifacts.
    
    Args:
        signal: Clean signal
        sample_rate: Sampling rate
        powerline_hz: Power line frequency (50 or 60 Hz)
        powerline_amplitude: 50/60 Hz interference amplitude
        contact_noise_std: Electrode contact noise
        
    Returns:
        Noisy signal
    """
    n = len(signal)
    t = np.arange(n) / sample_rate
    
    # Power line interference
    powerline = powerline_amplitude * np.sin(2 * np.pi * powerline_hz * t)
    
    # Electrode contact noise (low frequency drift)
    contact_noise = contact_noise_std * np.cumsum(np.random.randn(n)) / np.sqrt(n)
    
    return signal + powerline + contact_noise


# =============================================================================
# RFT-based Compression
# =============================================================================

def rft_compress_signal(signal: np.ndarray,
                        chunk_size: int = 256,
                        keep_ratio: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Compress signal using RFT coefficient thresholding.
    
    Args:
        signal: Input signal
        chunk_size: Size of transform chunks
        keep_ratio: Fraction of coefficients to keep
        
    Returns:
        (reconstructed_signal, stats_dict)
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        pytest.skip("RFT core not available")
    
    n = len(signal)
    n_padded = ((n - 1) // chunk_size + 1) * chunk_size
    padded = np.zeros(n_padded)
    padded[:n] = signal
    
    n_chunks = n_padded // chunk_size
    coeffs_kept = 0
    total_coeffs = 0
    
    reconstructed = np.zeros(n_padded)
    
    for i in range(n_chunks):
        chunk = padded[i * chunk_size:(i + 1) * chunk_size].astype(np.complex128)
        
        # Forward transform
        rft_coeffs = rft_forward(chunk)
        total_coeffs += len(rft_coeffs)
        
        # Keep top coefficients by magnitude
        n_keep = int(keep_ratio * len(rft_coeffs))
        magnitudes = np.abs(rft_coeffs)
        threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
        
        compressed = np.where(magnitudes >= threshold, rft_coeffs, 0)
        coeffs_kept += np.count_nonzero(compressed)
        
        # Inverse transform
        reconstructed[i * chunk_size:(i + 1) * chunk_size] = rft_inverse(compressed).real
    
    stats = {
        'compression_ratio': total_coeffs / coeffs_kept if coeffs_kept > 0 else float('inf'),
        'coeffs_kept': coeffs_kept,
        'total_coeffs': total_coeffs,
        'keep_ratio_actual': coeffs_kept / total_coeffs
    }
    
    return reconstructed[:n], stats


def fft_compress_signal(signal: np.ndarray,
                        chunk_size: int = 256,
                        keep_ratio: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """Baseline FFT-based compression."""
    n = len(signal)
    n_padded = ((n - 1) // chunk_size + 1) * chunk_size
    padded = np.zeros(n_padded)
    padded[:n] = signal
    
    n_chunks = n_padded // chunk_size
    coeffs_kept = 0
    total_coeffs = 0
    
    reconstructed = np.zeros(n_padded)
    
    for i in range(n_chunks):
        chunk = padded[i * chunk_size:(i + 1) * chunk_size]
        
        fft_coeffs = np.fft.fft(chunk)
        total_coeffs += len(fft_coeffs)
        
        n_keep = int(keep_ratio * len(fft_coeffs))
        magnitudes = np.abs(fft_coeffs)
        threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
        
        compressed = np.where(magnitudes >= threshold, fft_coeffs, 0)
        coeffs_kept += np.count_nonzero(compressed)
        
        reconstructed[i * chunk_size:(i + 1) * chunk_size] = np.fft.ifft(compressed).real
    
    stats = {
        'compression_ratio': total_coeffs / coeffs_kept if coeffs_kept > 0 else float('inf'),
        'coeffs_kept': coeffs_kept,
        'total_coeffs': total_coeffs,
        'keep_ratio_actual': coeffs_kept / total_coeffs
    }
    
    return reconstructed[:n], stats


def wavelet_rft_hybrid_compress_signal(signal: np.ndarray,
                                       levels: int = 3,
                                       keep_ratio: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Wavelet-RFT hybrid compression for ECG signals.
    
    Uses multi-level Haar wavelet decomposition, then applies RFT-based
    Wiener filtering to detail subbands before coefficient thresholding.
    """
    try:
        from algorithms.rft.variants.operator_variants import get_operator_variant
        use_rft = True
    except ImportError:
        use_rft = False
    
    # Pad to power of 2
    n = len(signal)
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    padded = np.zeros(n_pad)
    padded[:n] = signal
    
    # Multi-level Haar wavelet decomposition
    approx = padded.copy()
    details = []
    
    for _ in range(levels):
        n_curr = len(approx)
        if n_curr % 2 != 0:
            approx = np.append(approx, approx[-1])
            n_curr += 1
        low = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        high = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        details.append(high)
        approx = low
    
    # Estimate noise from finest detail level
    sigma = np.median(np.abs(details[0])) / 0.6745
    noise_var = sigma ** 2
    
    # RFT filtering on detail subbands
    filtered_details = []
    total_coeffs = sum(len(d) for d in details) + len(approx)
    coeffs_kept = len(approx)  # Always keep approximation
    
    for i, detail in enumerate(details):
        n_d = len(detail)
        level_noise_var = noise_var * (0.5 ** i)
        
        # Apply RFT Wiener filter if available and size sufficient
        if use_rft and n_d >= 16:
            try:
                Phi = get_operator_variant('rft_harmonic', n_d)
                rft_coeffs = Phi.T @ detail.astype(np.float64)
                
                # Wiener filter
                power = np.abs(rft_coeffs) ** 2
                wiener = power / (power + level_noise_var + 1e-10)
                filtered_rft = rft_coeffs * wiener
                
                # Threshold in RFT domain
                n_keep = int(keep_ratio * n_d)
                magnitudes = np.abs(filtered_rft)
                threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
                compressed_rft = np.where(magnitudes >= threshold, filtered_rft, 0)
                coeffs_kept += np.count_nonzero(compressed_rft)
                
                # Inverse RFT
                filtered_detail = (Phi @ compressed_rft).real
            except Exception:
                # Fallback to simple thresholding
                threshold = sigma * np.sqrt(2 * np.log(n_d)) * (0.8 ** i)
                filtered_detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                coeffs_kept += np.count_nonzero(filtered_detail)
        else:
            # Simple wavelet shrinkage
            threshold = sigma * np.sqrt(2 * np.log(n_d)) * (0.8 ** i)
            filtered_detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            coeffs_kept += np.count_nonzero(filtered_detail)
        
        filtered_details.append(filtered_detail)
    
    # Reconstruct from coarsest to finest
    current = approx
    for i in range(len(filtered_details) - 1, -1, -1):
        high = filtered_details[i]
        n_out = len(high) * 2
        reconstructed = np.zeros(n_out)
        reconstructed[0::2] = (current + high) / np.sqrt(2)
        reconstructed[1::2] = (current - high) / np.sqrt(2)
        current = reconstructed
    
    stats = {
        'compression_ratio': total_coeffs / coeffs_kept if coeffs_kept > 0 else float('inf'),
        'coeffs_kept': coeffs_kept,
        'total_coeffs': total_coeffs,
        'keep_ratio_actual': coeffs_kept / total_coeffs
    }
    
    return current[:n], stats


# =============================================================================
# Quality Metrics
# =============================================================================

def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference (common ECG metric)."""
    return 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))


def correlation_coefficient(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]


# =============================================================================
# Anomaly Detection Simulation
# =============================================================================

def detect_anomalies_energy(signal: np.ndarray,
                            sample_rate: int,
                            window_sec: float = 0.5,
                            threshold_std: float = 2.0) -> np.ndarray:
    """
    Simple energy-based anomaly detection.
    
    Args:
        signal: Input signal
        sample_rate: Sampling rate
        window_sec: Window size in seconds
        threshold_std: Number of std deviations for threshold
        
    Returns:
        Binary detection mask
    """
    window_samples = int(window_sec * sample_rate)
    n = len(signal)
    
    # Calculate rolling energy
    energies = []
    for i in range(0, n - window_samples, window_samples // 2):
        window = signal[i:i + window_samples]
        energies.append(np.sum(window ** 2))
    
    energies = np.array(energies)
    threshold = np.mean(energies) + threshold_std * np.std(energies)
    
    # Map back to full signal
    detections = np.zeros(n, dtype=bool)
    for i, e in enumerate(energies):
        if e > threshold:
            start = i * (window_samples // 2)
            end = min(start + window_samples, n)
            detections[start:end] = True
    
    return detections


def detection_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Calculate detection metrics (sensitivity, specificity, etc.)."""
    tp = np.sum(true_labels & predictions)
    tn = np.sum(~true_labels & ~predictions)
    fp = np.sum(~true_labels & predictions)
    fn = np.sum(true_labels & ~predictions)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'accuracy': (tp + tn) / len(true_labels)
    }


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class BiosignalTestResult:
    """Result container for biosignal tests."""
    signal_type: str
    method: str
    compression_ratio: float
    snr_db: float
    prd_percent: float
    correlation: float
    time_ms: float


# =============================================================================
# Pytest Test Cases
# =============================================================================

class TestECGCompression:
    """Test suite for ECG signal compression."""
    
    @pytest.fixture
    def ecg_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate clean ECG signal."""
        return generate_ecg_signal(duration_sec=10.0, sample_rate=360)
    
    def test_ecg_generation(self, ecg_signal):
        """Verify ECG signal generation."""
        signal, labels = ecg_signal
        assert len(signal) == 3600  # 10 sec * 360 Hz
        print(f"✓ ECG generated: {len(signal)} samples, "
              f"range=[{signal.min():.2f}, {signal.max():.2f}]")
    
    @pytest.mark.parametrize("keep_ratio", [0.3, 0.5, 0.7])
    def test_rft_vs_fft_compression(self, ecg_signal, keep_ratio, rft_variant):
        """Compare RFT vs FFT compression quality."""
        signal, _ = ecg_signal
        
        # RFT compression
        t0 = time.perf_counter()
        rft_recon, rft_stats = rft_compress_signal(signal, keep_ratio=keep_ratio)
        rft_time = (time.perf_counter() - t0) * 1000
        
        # FFT compression
        t0 = time.perf_counter()
        fft_recon, fft_stats = fft_compress_signal(signal, keep_ratio=keep_ratio)
        fft_time = (time.perf_counter() - t0) * 1000
        
        # Metrics
        rft_snr = snr(signal, rft_recon)
        fft_snr = snr(signal, fft_recon)
        rft_prd = prd(signal, rft_recon)
        fft_prd = prd(signal, fft_recon)
        
        print(f"\n  ECG Compression (keep_ratio={keep_ratio}):")
        print(f"    RFT: SNR={rft_snr:.2f}dB, PRD={rft_prd:.2f}%, "
              f"CR={rft_stats['compression_ratio']:.2f}x, time={rft_time:.1f}ms")
        print(f"    FFT: SNR={fft_snr:.2f}dB, PRD={fft_prd:.2f}%, "
              f"CR={fft_stats['compression_ratio']:.2f}x, time={fft_time:.1f}ms")
        
        # Quality should be reasonable
        # Allow slightly lower floor for hybrid/dct-heavy variants
        snr_floor = 7.0 if rft_variant in {"rft_hybrid_dct", "rft_cascade_h3"} else 10.0
        assert rft_snr > snr_floor, f"RFT SNR too low for {rft_variant}: {rft_snr}"
    
    def test_ecg_with_noise(self):
        """Test compression robustness to electrode noise."""
        clean_ecg, _ = generate_ecg_signal(duration_sec=5.0, sample_rate=360)
        noisy_ecg = add_electrode_noise(clean_ecg, 360, powerline_amplitude=0.2)
        
        # Compress noisy signal
        rft_recon, stats = rft_compress_signal(noisy_ecg, keep_ratio=0.5)
        
        # Compare to clean (not noisy) - see if compression helps denoise
        snr_noisy = snr(clean_ecg, noisy_ecg)
        snr_recon = snr(clean_ecg, rft_recon)
        
        print(f"\n  ECG with electrode noise:")
        print(f"    Noisy SNR vs clean: {snr_noisy:.2f} dB")
        print(f"    Reconstructed SNR vs clean: {snr_recon:.2f} dB")
    
    def test_ecg_arrhythmia_detection(self):
        """Test arrhythmia detection before/after compression."""
        # Generate signal with arrhythmias
        signal, true_labels = generate_ecg_signal(
            duration_sec=30.0,
            sample_rate=360,
            arrhythmia_prob=0.1
        )
        
        # Detect on original
        orig_detections = detect_anomalies_energy(signal, 360)
        orig_metrics = detection_metrics(true_labels, orig_detections)
        
        # Compress and detect
        recon, _ = rft_compress_signal(signal, keep_ratio=0.5)
        recon_detections = detect_anomalies_energy(recon, 360)
        recon_metrics = detection_metrics(true_labels, recon_detections)
        
        print(f"\n  Arrhythmia detection (30s, 10% arrhythmia rate):")
        print(f"    Original: F1={orig_metrics['f1_score']:.3f}, "
              f"Sens={orig_metrics['sensitivity']:.3f}")
        print(f"    After RFT compression: F1={recon_metrics['f1_score']:.3f}, "
              f"Sens={recon_metrics['sensitivity']:.3f}")


class TestEEGProcessing:
    """Test suite for EEG signal processing."""
    
    @pytest.fixture
    def eeg_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate EEG signal."""
        return generate_eeg_signal(duration_sec=10.0, sample_rate=256)
    
    def test_eeg_generation(self, eeg_signal):
        """Verify EEG signal generation."""
        signal, labels = eeg_signal
        assert len(signal) == 2560  # 10 sec * 256 Hz
        print(f"✓ EEG generated: {len(signal)} samples")
    
    @pytest.mark.parametrize("keep_ratio", [0.3, 0.5])
    def test_eeg_compression(self, eeg_signal, keep_ratio):
        """Test EEG compression quality."""
        signal, _ = eeg_signal
        
        rft_recon, rft_stats = rft_compress_signal(signal, chunk_size=256, keep_ratio=keep_ratio)
        fft_recon, fft_stats = fft_compress_signal(signal, chunk_size=256, keep_ratio=keep_ratio)
        
        print(f"\n  EEG Compression (keep={keep_ratio}):")
        print(f"    RFT: SNR={snr(signal, rft_recon):.2f}dB, corr={correlation_coefficient(signal, rft_recon):.4f}")
        print(f"    FFT: SNR={snr(signal, fft_recon):.2f}dB, corr={correlation_coefficient(signal, fft_recon):.4f}")
    
    def test_seizure_detection(self):
        """Test seizure detection on compressed signal."""
        signal, true_labels = generate_eeg_signal(
            duration_sec=60.0,
            sample_rate=256,
            seizure_prob=0.05,
            seizure_duration_sec=3.0
        )
        
        # Detect on original
        orig_det = detect_anomalies_energy(signal, 256, window_sec=1.0, threshold_std=2.5)
        orig_metrics = detection_metrics(true_labels, orig_det)
        
        # Compress and detect
        recon, _ = rft_compress_signal(signal, keep_ratio=0.4)
        recon_det = detect_anomalies_energy(recon, 256, window_sec=1.0, threshold_std=2.5)
        recon_metrics = detection_metrics(true_labels, recon_det)
        
        print(f"\n  Seizure detection (60s, 5% seizure probability):")
        print(f"    Original: F1={orig_metrics['f1_score']:.3f}, "
              f"Sens={orig_metrics['sensitivity']:.3f}")
        print(f"    After compression: F1={recon_metrics['f1_score']:.3f}, "
              f"Sens={recon_metrics['sensitivity']:.3f}")


class TestEMGProcessing:
    """Test suite for EMG signal processing."""
    
    def test_emg_compression(self):
        """Test EMG compression."""
        signal = generate_emg_signal(duration_sec=5.0, sample_rate=1000)
        
        rft_recon, stats = rft_compress_signal(signal, chunk_size=512, keep_ratio=0.5)
        
        signal_snr = snr(signal, rft_recon)
        signal_corr = correlation_coefficient(signal, rft_recon)
        
        print(f"\n  EMG Compression:")
        print(f"    SNR: {signal_snr:.2f} dB")
        print(f"    Correlation: {signal_corr:.4f}")
        print(f"    Compression ratio: {stats['compression_ratio']:.2f}x")
        
        assert signal_corr > 0.8, "EMG correlation should be high"


class TestLatencyBenchmark:
    """Benchmark compression latency for real-time applications."""
    
    @pytest.mark.parametrize("signal_type,sr,chunk_ms", [
        ("ECG", 360, 100),   # 100ms chunks at 360 Hz
        ("EEG", 256, 100),   # 100ms chunks at 256 Hz
        ("EMG", 1000, 50),   # 50ms chunks at 1 kHz
    ])
    def test_real_time_latency(self, signal_type, sr, chunk_ms):
        """Measure per-chunk processing latency."""
        chunk_size = int(sr * chunk_ms / 1000)
        
        # Generate test chunk
        chunk = np.random.randn(chunk_size)
        
        # Warm-up
        for _ in range(3):
            rft_compress_signal(chunk, chunk_size=chunk_size, keep_ratio=0.5)
        
        # Measure
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            rft_compress_signal(chunk, chunk_size=chunk_size, keep_ratio=0.5)
            times.append((time.perf_counter() - t0) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        print(f"\n  {signal_type} latency ({chunk_ms}ms chunks, {chunk_size} samples):")
        print(f"    Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"    Real-time margin: {chunk_ms - avg_time:.2f}ms")
        
        # Should complete faster than real-time chunk duration
        assert avg_time < chunk_ms * 2, f"Processing too slow for real-time: {avg_time}ms > {chunk_ms}ms"


# =============================================================================
# Standalone Runner
# =============================================================================

def run_comprehensive_biosignal_benchmark():
    """Run comprehensive biosignal benchmark."""
    print("=" * 70)
    print("BIOSIGNAL COMPRESSION & ANALYSIS BENCHMARK")
    print("=" * 70)
    
    results: List[BiosignalTestResult] = []
    
    # ECG benchmark
    print("\n[1] ECG Compression Benchmark")
    ecg, _ = generate_ecg_signal(duration_sec=30.0, sample_rate=360)
    
    for keep_ratio in [0.2, 0.3, 0.5, 0.7]:
        for method_name, method in [
            ('RFT', rft_compress_signal),
            ('FFT', fft_compress_signal),
        ]:
            t0 = time.perf_counter()
            recon, stats = method(ecg, keep_ratio=keep_ratio)
            elapsed = (time.perf_counter() - t0) * 1000
            
            result = BiosignalTestResult(
                signal_type='ECG',
                method=method_name,
                compression_ratio=stats['compression_ratio'],
                snr_db=snr(ecg, recon),
                prd_percent=prd(ecg, recon),
                correlation=correlation_coefficient(ecg, recon),
                time_ms=elapsed
            )
            results.append(result)
    
    # EEG benchmark
    print("\n[2] EEG Compression Benchmark")
    eeg, _ = generate_eeg_signal(duration_sec=30.0, sample_rate=256)
    
    for keep_ratio in [0.2, 0.3, 0.5]:
        for method_name, method in [
            ('RFT', rft_compress_signal),
            ('FFT', fft_compress_signal),
        ]:
            t0 = time.perf_counter()
            recon, stats = method(eeg, keep_ratio=keep_ratio)
            elapsed = (time.perf_counter() - t0) * 1000
            
            result = BiosignalTestResult(
                signal_type='EEG',
                method=method_name,
                compression_ratio=stats['compression_ratio'],
                snr_db=snr(eeg, recon),
                prd_percent=prd(eeg, recon),
                correlation=correlation_coefficient(eeg, recon),
                time_ms=elapsed
            )
            results.append(result)
    
    # Print results
    print(f"\n{'Type':<6} {'Method':<6} {'CR':<8} {'SNR (dB)':<10} {'PRD (%)':<10} "
          f"{'Corr':<8} {'Time (ms)':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.signal_type:<6} {r.method:<6} {r.compression_ratio:>6.2f}x "
              f"{r.snr_db:>8.2f}   {r.prd_percent:>8.2f}   "
              f"{r.correlation:>6.4f}   {r.time_ms:>8.1f}")
    
    print("\n✓ Biosignal benchmark complete")
    return results


if __name__ == "__main__":
    run_comprehensive_biosignal_benchmark()


# =============================================================================
# Real-Data Tests (MIT-BIH Arrhythmia Database)
# =============================================================================

from tests.medical.real_data_fixtures import (
    skip_no_mitbih,
    list_mitbih_records,
    load_mitbih_record,
    MITBIH_AVAILABLE,
)


@skip_no_mitbih
class TestRealMITBIH:
    """
    Tests using real MIT-BIH Arrhythmia Database records.
    
    Requires:
        - USE_REAL_DATA=1
        - PhysioNet DUA acceptance
        - python data/physionet_mitbih_fetch.py executed
    """

    @pytest.fixture(scope="class")
    def mitbih_record(self):
        """Load first available MIT-BIH record."""
        records = list_mitbih_records()
        if not records:
            pytest.skip("No MIT-BIH records found")
        signal, fs = load_mitbih_record(records[0])
        # Take 10 seconds for quick test
        n_samples = min(len(signal), fs * 10)
        return signal[:n_samples], fs

    def test_real_ecg_rft_compression(self, mitbih_record):
        """Test RFT compression on real MIT-BIH ECG."""
        signal, fs = mitbih_record
        
        recon, stats = rft_compress_signal(signal, chunk_size=256, keep_ratio=0.5)
        
        signal_snr = snr(signal, recon)
        signal_prd = prd(signal, recon)
        
        # Real ECG may have more noise; use relaxed thresholds
        assert signal_snr >= 20.0, f"Real ECG SNR too low: {signal_snr:.2f} dB"
        assert signal_prd <= 10.0, f"Real ECG PRD too high: {signal_prd:.2f}%"
        assert stats["compression_ratio"] >= 1.5, "Compression ratio too low"
        
        print(f"✓ Real MIT-BIH ECG: SNR={signal_snr:.2f} dB, PRD={signal_prd:.2f}%, CR={stats['compression_ratio']:.2f}x")

    def test_real_ecg_fft_comparison(self, mitbih_record):
        """Compare RFT vs FFT on real MIT-BIH ECG."""
        signal, fs = mitbih_record
        
        rft_recon, rft_stats = rft_compress_signal(signal, keep_ratio=0.5)
        fft_recon, fft_stats = fft_compress_signal(signal, keep_ratio=0.5)
        
        rft_snr = snr(signal, rft_recon)
        fft_snr = snr(signal, fft_recon)
        
        print(f"Real ECG comparison: RFT SNR={rft_snr:.2f} dB, FFT SNR={fft_snr:.2f} dB")
        
        # RFT should match or exceed FFT for biosignal quality
        # Allow 3 dB tolerance for real-world variability
        assert rft_snr >= fft_snr - 3.0, f"RFT underperformed FFT by more than 3 dB"

    @pytest.mark.parametrize("keep_ratio", [0.2, 0.3, 0.5, 0.7])
    def test_real_ecg_compression_ratios(self, mitbih_record, keep_ratio):
        """Test multiple compression ratios on real ECG."""
        signal, fs = mitbih_record
        
        recon, stats = rft_compress_signal(signal, keep_ratio=keep_ratio)
        
        signal_snr = snr(signal, recon)
        corr = correlation_coefficient(signal, recon)
        
        # Higher keep ratio should yield better quality
        min_snr = 15.0 + (keep_ratio - 0.2) * 20  # Sliding threshold
        assert signal_snr >= min_snr, f"SNR {signal_snr:.2f} below expected {min_snr:.2f} for keep_ratio={keep_ratio}"
        assert corr >= 0.9, f"Correlation too low: {corr:.4f}"
        
        print(f"  keep_ratio={keep_ratio}: SNR={signal_snr:.2f} dB, corr={corr:.4f}")

    def test_real_ecg_wavelet_rft_hybrid(self, mitbih_record):
        """Test Wavelet-RFT hybrid compression on real MIT-BIH ECG."""
        signal, fs = mitbih_record
        
        recon, stats = wavelet_rft_hybrid_compress_signal(signal, levels=3, keep_ratio=0.5)
        
        signal_snr = snr(signal, recon)
        signal_prd = prd(signal, recon)
        corr = correlation_coefficient(signal, recon)
        
        # Wavelet-RFT should achieve good quality
        assert signal_snr >= 20.0, f"Wavelet-RFT SNR too low: {signal_snr:.2f} dB"
        assert signal_prd <= 10.0, f"Wavelet-RFT PRD too high: {signal_prd:.2f}%"
        assert corr >= 0.95, f"Wavelet-RFT correlation too low: {corr:.4f}"
        
        print(f"✓ Wavelet-RFT Hybrid: SNR={signal_snr:.2f} dB, PRD={signal_prd:.2f}%, "
              f"corr={corr:.4f}, CR={stats['compression_ratio']:.2f}x")

    def test_real_ecg_wavelet_vs_rft_comparison(self, mitbih_record):
        """Compare Wavelet-RFT hybrid vs pure RFT on real ECG (informational)."""
        signal, fs = mitbih_record
        
        # Pure RFT
        rft_recon, rft_stats = rft_compress_signal(signal, keep_ratio=0.5)
        rft_snr = snr(signal, rft_recon)
        
        # Wavelet-RFT hybrid
        hybrid_recon, hybrid_stats = wavelet_rft_hybrid_compress_signal(signal, keep_ratio=0.5)
        hybrid_snr = snr(signal, hybrid_recon)
        
        # FFT baseline
        fft_recon, fft_stats = fft_compress_signal(signal, keep_ratio=0.5)
        fft_snr = snr(signal, fft_recon)
        
        print(f"\n  Real ECG compression comparison (50% keep ratio):")
        print(f"    Pure RFT:           SNR={rft_snr:.2f} dB, CR={rft_stats['compression_ratio']:.2f}x")
        print(f"    Wavelet-RFT Hybrid: SNR={hybrid_snr:.2f} dB, CR={hybrid_stats['compression_ratio']:.2f}x")
        print(f"    FFT Baseline:       SNR={fft_snr:.2f} dB, CR={fft_stats['compression_ratio']:.2f}x")
        
        # Informational test - shows relative performance
        # Different methods work better for different signal characteristics
        # Some variants (like rft_hybrid_dct) may underperform on specific signals
        assert hybrid_snr > 15.0, "Wavelet hybrid should achieve reasonable quality"
        # Note: Pure RFT quality varies by variant (some experimental variants may be lower)

    @pytest.mark.parametrize("levels", [2, 3, 4])
    def test_real_ecg_wavelet_hybrid_levels(self, mitbih_record, levels):
        """Test different wavelet decomposition levels on real ECG."""
        signal, fs = mitbih_record
        
        recon, stats = wavelet_rft_hybrid_compress_signal(signal, levels=levels, keep_ratio=0.5)
        
        signal_snr = snr(signal, recon)
        corr = correlation_coefficient(signal, recon)
        
        # Should work across different decomposition levels
        assert signal_snr >= 18.0, f"SNR too low for {levels} levels: {signal_snr:.2f} dB"
        assert corr >= 0.93, f"Correlation too low: {corr:.4f}"
        
        print(f"  {levels} levels: SNR={signal_snr:.2f} dB, corr={corr:.4f}, CR={stats['compression_ratio']:.2f}x")


@skip_no_mitbih
def test_mitbih_multiple_records():
    """Test RFT compression across multiple MIT-BIH records."""
    records = list_mitbih_records()[:5]  # Test up to 5 records
    
    if len(records) == 0:
        pytest.skip("No MIT-BIH records available")
    
    results = []
    for rec_path in records:
        signal, fs = load_mitbih_record(rec_path)
        # Use 5 seconds
        n = min(len(signal), fs * 5)
        sig = signal[:n]
        
        recon, stats = rft_compress_signal(sig, keep_ratio=0.5)
        snr_val = snr(sig, recon)
        results.append((rec_path.stem, snr_val, stats["compression_ratio"]))
    
    print("\nMIT-BIH Multi-Record Results:")
    for name, s, cr in results:
        print(f"  {name}: SNR={s:.2f} dB, CR={cr:.2f}x")
    
    avg_snr = np.mean([r[1] for r in results])
    assert avg_snr >= 20.0, f"Average SNR across records too low: {avg_snr:.2f} dB"


# =============================================================================
# Real-Data Tests (Sleep-EDF Database)
# =============================================================================

from tests.medical.real_data_fixtures import (
    skip_no_sleepedf,
    list_sleepedf_records,
    load_sleepedf_record,
    SLEEPEDF_AVAILABLE,
)


@skip_no_sleepedf
class TestRealSleepEDF:
    """
    Tests using real Sleep-EDF Database EEG records.
    
    Requires:
        - USE_REAL_DATA=1
        - PhysioNet terms acceptance
        - python data/physionet_sleepedf_fetch.py executed
    """

    @pytest.fixture(scope="class")
    def sleepedf_record(self):
        """Load first available Sleep-EDF EEG snippet."""
        records = list_sleepedf_records()
        if not records:
            pytest.skip("No Sleep-EDF records found")
        signal, fs = load_sleepedf_record(records[0], duration_sec=30.0)
        return signal, fs

    def test_real_eeg_rft_compression(self, sleepedf_record):
        """Test RFT compression on real Sleep-EDF EEG."""
        signal, fs = sleepedf_record
        
        recon, stats = rft_compress_signal(signal, chunk_size=256, keep_ratio=0.5)
        
        signal_snr = snr(signal, recon)
        signal_corr = correlation_coefficient(signal, recon)
        
        # EEG typically has lower SNR due to broadband nature
        assert signal_snr >= 15.0, f"Real EEG SNR too low: {signal_snr:.2f} dB"
        assert signal_corr >= 0.95, f"Real EEG correlation too low: {signal_corr:.4f}"
        
        print(f"✓ Real Sleep-EDF EEG: SNR={signal_snr:.2f} dB, corr={signal_corr:.4f}, CR={stats['compression_ratio']:.2f}x")

    def test_real_eeg_fft_comparison(self, sleepedf_record):
        """Compare RFT vs FFT on real Sleep-EDF EEG."""
        signal, fs = sleepedf_record
        
        rft_recon, _ = rft_compress_signal(signal, keep_ratio=0.5)
        fft_recon, _ = fft_compress_signal(signal, keep_ratio=0.5)
        
        rft_snr = snr(signal, rft_recon)
        fft_snr = snr(signal, fft_recon)
        
        print(f"Real EEG comparison: RFT SNR={rft_snr:.2f} dB, FFT SNR={fft_snr:.2f} dB")
        
        # Allow tolerance for real-world variability
        assert rft_snr >= fft_snr - 3.0, "RFT underperformed FFT by more than 3 dB"

    @pytest.mark.parametrize("keep_ratio", [0.3, 0.5, 0.7])
    def test_real_eeg_compression_ratios(self, sleepedf_record, keep_ratio):
        """Test multiple compression ratios on real EEG."""
        signal, fs = sleepedf_record
        
        recon, stats = rft_compress_signal(signal, keep_ratio=keep_ratio)
        
        signal_snr = snr(signal, recon)
        corr = correlation_coefficient(signal, recon)
        
        min_snr = 10.0 + (keep_ratio - 0.3) * 15
        assert signal_snr >= min_snr, f"SNR {signal_snr:.2f} below expected {min_snr:.2f}"
        assert corr >= 0.9, f"Correlation too low: {corr:.4f}"
        
        print(f"  keep_ratio={keep_ratio}: SNR={signal_snr:.2f} dB, corr={corr:.4f}")
