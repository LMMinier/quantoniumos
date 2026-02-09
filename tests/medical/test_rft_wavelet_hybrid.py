#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Test RFT-Wavelet Hybrid vs Pure Wavelet vs Pure RFT for MRI Denoising
=====================================================================

Quick benchmark to see if the hybrid approach beats pure wavelet.

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This code is for research comparison purposes only and has not been
validated for medical device or clinical diagnostic applications.

Copyright (C) 2025 Luis M. Minier / quantoniumos
Licensed under AGPL-3.0-or-later
"""

import numpy as np
import time
import sys
sys.path.insert(0, "/workspaces/quantoniumos")


def shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate Shepp-Logan phantom for MRI simulation."""
    phantom = np.zeros((size, size), dtype=np.float64)
    
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),
        (0.06, -0.605, 0.023, 0.046, 0, 0.01),
    ]
    
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    
    for x0, y0, a, b, theta_deg, intensity in ellipses:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = (x - x0) * cos_t + (y - y0) * sin_t
        y_rot = -(x - x0) * sin_t + (y - y0) * cos_t
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
        phantom[mask] += intensity
    
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-10)
    return phantom


def add_rician_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise (MRI magnitude signal noise model)."""
    real = image + sigma * np.random.randn(*image.shape)
    imag = sigma * np.random.randn(*image.shape)
    return np.sqrt(real**2 + imag**2)


def add_poisson_noise(image: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Add Poisson noise (photon counting noise for PET/CT)."""
    counts = image * scale
    noisy_counts = np.random.poisson(counts.astype(np.float64))
    return noisy_counts / scale


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def ssim_simple(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Simplified SSIM."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = np.mean(original), np.mean(reconstructed)
    sigma_x, sigma_y = np.std(original), np.std(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    
    return ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))


# Simple wavelet denoiser (baseline)
def wavelet_denoise_2d(noisy_image: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """Single-level Haar wavelet denoising."""
    rows, cols = noisy_image.shape
    
    # Horizontal pass
    h_low = (noisy_image[:, 0::2] + noisy_image[:, 1::2]) / np.sqrt(2)
    h_high = (noisy_image[:, 0::2] - noisy_image[:, 1::2]) / np.sqrt(2)
    h_transform = np.hstack([h_low, h_high])
    
    # Vertical pass
    v_low = (h_transform[0::2, :] + h_transform[1::2, :]) / np.sqrt(2)
    v_high = (h_transform[0::2, :] - h_transform[1::2, :]) / np.sqrt(2)
    coeffs = np.vstack([v_low, v_high])
    
    # Threshold
    max_mag = np.max(np.abs(coeffs[rows//2:, :]))
    threshold = threshold_ratio * max_mag
    
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[rows//2:, :] = np.where(np.abs(coeffs[rows//2:, :]) < threshold, 0, coeffs[rows//2:, :])
    coeffs_thresh[:rows//2, cols//2:] = np.where(np.abs(coeffs[:rows//2, cols//2:]) < threshold, 0, coeffs[:rows//2, cols//2:])
    
    # Inverse vertical
    v_low, v_high = coeffs_thresh[:rows//2, :], coeffs_thresh[rows//2:, :]
    inv_v = np.zeros((rows, cols))
    inv_v[0::2, :] = (v_low + v_high) / np.sqrt(2)
    inv_v[1::2, :] = (v_low - v_high) / np.sqrt(2)
    
    # Inverse horizontal
    h_low, h_high = inv_v[:, :cols//2], inv_v[:, cols//2:]
    denoised = np.zeros((rows, cols))
    denoised[:, 0::2] = (h_low + h_high) / np.sqrt(2)
    denoised[:, 1::2] = (h_low - h_high) / np.sqrt(2)
    
    return np.clip(denoised, 0, 1)


# RFT-only denoiser (from tests)
def rft_denoise_2d(noisy_image: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """RFT-based 2D denoising."""
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward, rft_inverse
    except ImportError:
        print("RFT not available, using FFT")
        rft_forward = np.fft.fft
        rft_inverse = np.fft.ifft
    
    rows, cols = noisy_image.shape
    
    # Row-wise RFT
    row_coeffs = np.zeros_like(noisy_image, dtype=np.complex128)
    for i in range(rows):
        row_coeffs[i, :] = rft_forward(noisy_image[i, :].astype(np.complex128))
    
    # Column-wise RFT
    full_coeffs = np.zeros_like(row_coeffs)
    for j in range(cols):
        full_coeffs[:, j] = rft_forward(row_coeffs[:, j])
    
    # Threshold
    max_mag = np.max(np.abs(full_coeffs))
    threshold = threshold_ratio * max_mag
    thresholded = np.where(np.abs(full_coeffs) < threshold, 0, full_coeffs)
    
    # Inverse
    inv_cols = np.zeros_like(thresholded)
    for j in range(cols):
        inv_cols[:, j] = rft_inverse(thresholded[:, j])
    
    denoised = np.zeros_like(noisy_image)
    for i in range(rows):
        denoised[i, :] = rft_inverse(inv_cols[i, :]).real
    
    return np.clip(denoised, 0, 1)


def run_comparison():
    """Run the comparison benchmark."""
    print("=" * 80)
    print("RFT-Wavelet Hybrid vs Pure Methods - MRI Denoising Benchmark")
    print("=" * 80)
    
    # Import the hybrids
    try:
        from algorithms.rft.hybrids.rft_wavelet_medical import (
            rft_wavelet_denoise_2d,
            rft_wavelet_denoise_adaptive,
        )
        hybrid_v1_available = True
    except ImportError as e:
        print(f"Hybrid v1 import error: {e}")
        hybrid_v1_available = False
    
    try:
        from algorithms.rft.hybrids.rft_wavelet_medical_v2 import (
            rft_wavelet_denoise_v2,
            rft_wavelet_denoise_v2_adaptive,
        )
        hybrid_v2_available = True
    except ImportError as e:
        print(f"Hybrid v2 import error: {e}")
        hybrid_v2_available = False
    
    hybrid_available = hybrid_v1_available  # For backwards compat
    
    # Generate phantom
    np.random.seed(42)
    phantom = shepp_logan_phantom(256)
    
    # Test scenarios
    scenarios = [
        ("Rician σ=0.05 (low)", lambda x: add_rician_noise(x, 0.05), "rician"),
        ("Rician σ=0.10 (med)", lambda x: add_rician_noise(x, 0.10), "rician"),
        ("Rician σ=0.15 (high)", lambda x: add_rician_noise(x, 0.15), "rician"),
        ("Poisson 500 (PET)", lambda x: add_poisson_noise(x, 500), "poisson"),
        ("Poisson 100 (low-dose)", lambda x: add_poisson_noise(x, 100), "poisson"),
    ]
    
    print(f"\n{'Scenario':<22} {'Method':<18} {'PSNR Before':>12} {'PSNR After':>12} {'SSIM':>8} {'Time':>10}")
    print("-" * 84)
    
    results = []
    
    for name, noise_fn, noise_model in scenarios:
        noisy = noise_fn(phantom)
        psnr_before = psnr(phantom, noisy)
        
        methods = [
            ("Wavelet (baseline)", lambda x: wavelet_denoise_2d(x, 0.1)),
            ("RFT-only", lambda x: rft_denoise_2d(x, 0.1)),
        ]
        
        if hybrid_available:
            methods.append(("Hybrid v1", 
                           lambda x, nm=noise_model: rft_wavelet_denoise_2d(x, levels=2, noise_model=nm)))
        
        if hybrid_v2_available:
            methods.append(("Hybrid v2 (fast)", 
                           lambda x, nm=noise_model: rft_wavelet_denoise_v2(x, levels=2, noise_model=nm)))
            methods.append(("Hybrid v2 (auto)", 
                           lambda x: rft_wavelet_denoise_v2_adaptive(x, phantom)[0]))
        
        for method_name, denoise_fn in methods:
            start = time.perf_counter()
            try:
                denoised = denoise_fn(noisy)
                elapsed = (time.perf_counter() - start) * 1000
                psnr_after = psnr(phantom, denoised)
                ssim_val = ssim_simple(phantom, denoised)
                
                print(f"{name:<22} {method_name:<18} {psnr_before:>10.2f} dB {psnr_after:>10.2f} dB {ssim_val:>8.3f} {elapsed:>8.1f} ms")
                
                results.append({
                    "scenario": name,
                    "method": method_name,
                    "psnr_before": psnr_before,
                    "psnr_after": psnr_after,
                    "improvement": psnr_after - psnr_before,
                    "ssim": ssim_val,
                    "time_ms": elapsed,
                })
            except Exception as e:
                print(f"{name:<22} {method_name:<18} ERROR: {e}")
        
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Method per Scenario")
    print("=" * 70)
    
    for scenario in set(r["scenario"] for r in results):
        scenario_results = [r for r in results if r["scenario"] == scenario]
        if scenario_results:
            best = max(scenario_results, key=lambda r: r["psnr_after"])
            print(f"{scenario:<25} → {best['method']:<20} (+{best['improvement']:.2f} dB)")
    
    return results


if __name__ == "__main__":
    results = run_comparison()
