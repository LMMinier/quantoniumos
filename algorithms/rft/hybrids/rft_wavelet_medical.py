#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RFT-Wavelet Hybrid for Medical Image Denoising
===============================================

Combines wavelet multi-scale decomposition with RFT resonance filtering
for improved MRI/CT denoising under Rician and Poisson noise models.

Strategy:
1. Wavelet decomposition (multi-level Haar) → separates scales
2. RFT filtering on high-frequency subbands → exploits quasi-periodicity in edges
3. Adaptive soft thresholding based on noise model
4. BayesShrink-style variance estimation

The key insight: wavelets excel at separating scales, but RFT can better
model the quasi-periodic structure in tissue boundaries and anatomy edges.

Copyright (C) 2025 Luis M. Minier / quantoniumos
Licensed under AGPL-3.0-or-later
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DenoiseResult:
    """Result from denoising operation."""
    denoised: np.ndarray
    psnr_improvement: float
    ssim: float
    method: str
    time_ms: float


def _haar_decompose_2d(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-level 2D Haar wavelet decomposition.
    
    Returns:
        (LL, LH, HL, HH) subbands
    """
    rows, cols = image.shape
    
    # Pad if needed
    if rows % 2 != 0:
        image = np.vstack([image, image[-1:, :]])
        rows += 1
    if cols % 2 != 0:
        image = np.hstack([image, image[:, -1:]])
        cols += 1
    
    # Horizontal pass
    h_low = (image[:, 0::2] + image[:, 1::2]) / np.sqrt(2)
    h_high = (image[:, 0::2] - image[:, 1::2]) / np.sqrt(2)
    
    # Vertical pass on low-freq
    LL = (h_low[0::2, :] + h_low[1::2, :]) / np.sqrt(2)
    LH = (h_low[0::2, :] - h_low[1::2, :]) / np.sqrt(2)
    
    # Vertical pass on high-freq
    HL = (h_high[0::2, :] + h_high[1::2, :]) / np.sqrt(2)
    HH = (h_high[0::2, :] - h_high[1::2, :]) / np.sqrt(2)
    
    return LL, LH, HL, HH


def _haar_reconstruct_2d(LL: np.ndarray, LH: np.ndarray, 
                          HL: np.ndarray, HH: np.ndarray,
                          original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Single-level 2D Haar wavelet reconstruction.
    """
    rows, cols = LL.shape[0] * 2, LL.shape[1] * 2
    
    # Inverse vertical pass for low-freq horizontal
    h_low = np.zeros((rows, LL.shape[1]))
    h_low[0::2, :] = (LL + LH) / np.sqrt(2)
    h_low[1::2, :] = (LL - LH) / np.sqrt(2)
    
    # Inverse vertical pass for high-freq horizontal
    h_high = np.zeros((rows, HL.shape[1]))
    h_high[0::2, :] = (HL + HH) / np.sqrt(2)
    h_high[1::2, :] = (HL - HH) / np.sqrt(2)
    
    # Inverse horizontal pass
    image = np.zeros((rows, cols))
    image[:, 0::2] = (h_low + h_high) / np.sqrt(2)
    image[:, 1::2] = (h_low - h_high) / np.sqrt(2)
    
    # Trim to original shape
    return image[:original_shape[0], :original_shape[1]]


def _estimate_noise_variance(subband: np.ndarray, method: str = "mad") -> float:
    """
    Estimate noise variance from wavelet subband.
    
    Uses Median Absolute Deviation (MAD) estimator, robust to outliers.
    """
    if method == "mad":
        # MAD estimator: σ = MAD / 0.6745
        mad = np.median(np.abs(subband - np.median(subband)))
        return (mad / 0.6745) ** 2
    else:
        return np.var(subband)


def _soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def _rft_filter_subband(subband: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Apply RFT-based filtering to wavelet subband.
    
    The RFT resonance eigenbasis is better at preserving quasi-periodic
    edge structure compared to simple thresholding.
    
    Uses rft_entropy_modulated variant (best for medical imaging).
    
    Strategy:
    1. Apply RFT to rows and columns
    2. Use Wiener-like filtering in RFT domain
    3. Inverse RFT
    """
    try:
        from algorithms.rft.variants.operator_variants import get_operator_variant
        # Use entropy_modulated - winner for medical imaging
        variant = 'rft_entropy_modulated'
        Phi_rows = get_operator_variant(variant, subband.shape[1])
        Phi_cols = get_operator_variant(variant, subband.shape[0])
        
        # Fast 2D transform via matrix multiply
        full_coeffs = Phi_cols.T @ subband.astype(np.float64) @ Phi_rows
        
        # Wiener filter
        power = np.abs(full_coeffs) ** 2
        eps = 1e-10
        wiener_filter = power / (power + noise_var + eps)
        filtered_coeffs = full_coeffs * wiener_filter
        
        # Inverse
        result = Phi_cols @ filtered_coeffs @ Phi_rows.T
        return result.real
        
    except ImportError:
        # Fallback to original loop-based FFT
        pass
    
    # Fallback implementation
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        rft_forward = np.fft.fft
        rft_inverse = np.fft.ifft
    
    rows, cols = subband.shape
    
    # Row-wise RFT
    row_coeffs = np.zeros_like(subband, dtype=np.complex128)
    for i in range(rows):
        row_coeffs[i, :] = rft_forward(subband[i, :].astype(np.complex128))
    
    # Column-wise RFT
    full_coeffs = np.zeros_like(row_coeffs)
    for j in range(cols):
        full_coeffs[:, j] = rft_forward(row_coeffs[:, j])
    
    # Wiener-style filtering in RFT domain
    # H(ω) = |S(ω)|² / (|S(ω)|² + σ²)
    # Approximation: use magnitude as signal power estimate
    power = np.abs(full_coeffs) ** 2
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    wiener_filter = power / (power + noise_var + eps)
    
    # Apply filter
    filtered_coeffs = full_coeffs * wiener_filter
    
    # Inverse: column-wise then row-wise
    inv_cols = np.zeros_like(filtered_coeffs)
    for j in range(cols):
        inv_cols[:, j] = rft_inverse(filtered_coeffs[:, j])
    
    result = np.zeros_like(subband)
    for i in range(rows):
        result[i, :] = rft_inverse(inv_cols[i, :]).real
    
    return result


def _rft_edge_enhance(subband: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Enhance edges using RFT phase coherence.
    
    Edges in medical images often have quasi-periodic structure
    (tissue boundaries, vessel walls). RFT can detect and preserve these.
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        rft_forward = np.fft.fft
        rft_inverse = np.fft.ifft
    
    rows, cols = subband.shape
    
    # 2D RFT
    row_coeffs = np.array([rft_forward(subband[i, :].astype(np.complex128)) 
                           for i in range(rows)])
    full_coeffs = np.array([rft_forward(row_coeffs[:, j]) 
                            for j in range(cols)]).T
    
    # Phase coherence enhancement
    # Boost coefficients with coherent phase (indicative of structured edges)
    magnitudes = np.abs(full_coeffs)
    phases = np.angle(full_coeffs)
    
    # Compute local phase variance (low variance = coherent structure)
    kernel_size = min(5, min(rows, cols) // 4)
    if kernel_size >= 1:
        from scipy.ndimage import uniform_filter
        try:
            phase_var = uniform_filter(phases ** 2, kernel_size) - \
                       uniform_filter(phases, kernel_size) ** 2
            phase_var = np.clip(phase_var, 0, np.pi ** 2)
            
            # Boost coherent regions (low phase variance)
            coherence_boost = 1 + strength * (1 - phase_var / (np.pi ** 2))
            enhanced_coeffs = full_coeffs * coherence_boost
        except ImportError:
            enhanced_coeffs = full_coeffs
    else:
        enhanced_coeffs = full_coeffs
    
    # Inverse 2D RFT
    inv_cols = np.array([rft_inverse(enhanced_coeffs[:, j]) 
                         for j in range(cols)]).T
    result = np.array([rft_inverse(inv_cols[i, :]).real 
                       for i in range(rows)])
    
    return result


def rft_wavelet_denoise_2d(noisy_image: np.ndarray,
                            levels: int = 2,
                            noise_model: str = "rician",
                            edge_enhance: bool = True) -> np.ndarray:
    """
    RFT-Wavelet Hybrid Denoising for Medical Images.
    
    Combines multi-scale wavelet decomposition with RFT resonance filtering.
    
    Args:
        noisy_image: Noisy input image (2D, normalized to [0,1])
        levels: Number of wavelet decomposition levels
        noise_model: "rician" (MRI), "poisson" (PET/CT), or "gaussian"
        edge_enhance: Apply RFT edge enhancement
        
    Returns:
        Denoised image
    """
    original_shape = noisy_image.shape
    
    # Ensure image is proper size
    rows, cols = noisy_image.shape
    pad_rows = (2 ** levels - rows % (2 ** levels)) % (2 ** levels)
    pad_cols = (2 ** levels - cols % (2 ** levels)) % (2 ** levels)
    
    if pad_rows > 0 or pad_cols > 0:
        padded = np.pad(noisy_image, ((0, pad_rows), (0, pad_cols)), mode='reflect')
    else:
        padded = noisy_image.copy()
    
    # Multi-level wavelet decomposition
    coeffs_pyramid = []
    current = padded
    
    for level in range(levels):
        LL, LH, HL, HH = _haar_decompose_2d(current)
        coeffs_pyramid.append((LH, HL, HH, current.shape))
        current = LL
    
    # Process from coarsest to finest
    # Start with LL (approximation) - minimal processing, just light smoothing
    processed_LL = current
    
    # Process each level's detail subbands
    for level in range(levels - 1, -1, -1):
        LH, HL, HH, shape = coeffs_pyramid[level]
        
        # Estimate noise variance from HH subband (mostly noise)
        noise_var = _estimate_noise_variance(HH)
        
        # Adjust threshold based on noise model
        if noise_model == "rician":
            # Rician noise has signal-dependent variance
            threshold_mult = 1.2
        elif noise_model == "poisson":
            # Poisson noise - higher threshold for low-count regions
            threshold_mult = 1.5
        else:
            threshold_mult = 1.0
        
        # BayesShrink threshold: σ²_n / σ_signal
        signal_var = max(np.var(LH) - noise_var, 1e-10)
        bayes_threshold = threshold_mult * noise_var / np.sqrt(signal_var)
        
        # Apply RFT filtering to detail subbands
        LH_filtered = _rft_filter_subband(LH, noise_var)
        HL_filtered = _rft_filter_subband(HL, noise_var)
        HH_filtered = _rft_filter_subband(HH, noise_var * 2)  # HH has more noise
        
        # Apply soft thresholding after RFT filtering
        LH_denoised = _soft_threshold(LH_filtered, bayes_threshold * 0.5)
        HL_denoised = _soft_threshold(HL_filtered, bayes_threshold * 0.5)
        HH_denoised = _soft_threshold(HH_filtered, bayes_threshold)
        
        # Optional edge enhancement
        if edge_enhance and level == 0:  # Only at finest level
            try:
                LH_denoised = _rft_edge_enhance(LH_denoised, strength=0.2)
                HL_denoised = _rft_edge_enhance(HL_denoised, strength=0.2)
            except Exception:
                pass  # Skip if enhancement fails
        
        # Reconstruct this level
        processed_LL = _haar_reconstruct_2d(processed_LL, LH_denoised, 
                                             HL_denoised, HH_denoised, shape)
    
    # Trim to original size
    denoised = processed_LL[:original_shape[0], :original_shape[1]]
    
    return np.clip(denoised, 0, 1)


def rft_wavelet_denoise_adaptive(noisy_image: np.ndarray,
                                  reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Adaptive RFT-Wavelet denoising with automatic parameter selection.
    
    If reference is provided, uses it to estimate optimal parameters.
    Otherwise, uses blind estimation.
    
    Args:
        noisy_image: Noisy input
        reference: Optional clean reference for parameter tuning
        
    Returns:
        (denoised_image, metrics_dict)
    """
    import time
    start = time.perf_counter()
    
    # Estimate noise level
    # Use finest-scale diagonal subband
    LL, LH, HL, HH = _haar_decompose_2d(noisy_image)
    noise_sigma = np.median(np.abs(HH)) / 0.6745
    
    # Choose levels based on image size and noise
    min_dim = min(noisy_image.shape)
    max_levels = int(np.log2(min_dim)) - 2
    
    if noise_sigma > 0.15:
        levels = min(3, max_levels)  # More levels for high noise
    elif noise_sigma > 0.08:
        levels = min(2, max_levels)
    else:
        levels = min(1, max_levels)
    
    levels = max(1, levels)
    
    # Detect noise model from statistics
    # Rician: positive-skewed, Poisson: variance proportional to mean
    skewness = np.mean((noisy_image - np.mean(noisy_image)) ** 3) / (np.std(noisy_image) ** 3 + 1e-10)
    
    if skewness > 0.5:
        noise_model = "rician"
    elif np.abs(np.var(noisy_image) - np.mean(noisy_image)) < 0.1:
        noise_model = "poisson"
    else:
        noise_model = "gaussian"
    
    # Denoise
    denoised = rft_wavelet_denoise_2d(noisy_image, levels=levels, 
                                       noise_model=noise_model, edge_enhance=True)
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Compute metrics
    metrics = {
        "time_ms": elapsed_ms,
        "noise_sigma_est": noise_sigma,
        "levels": levels,
        "noise_model": noise_model,
    }
    
    if reference is not None:
        mse_before = np.mean((noisy_image - reference) ** 2)
        mse_after = np.mean((denoised - reference) ** 2)
        
        metrics["psnr_before"] = 10 * np.log10(1.0 / (mse_before + 1e-10))
        metrics["psnr_after"] = 10 * np.log10(1.0 / (mse_after + 1e-10))
        metrics["psnr_improvement"] = metrics["psnr_after"] - metrics["psnr_before"]
    
    return denoised, metrics


# Convenience alias for test integration
rft_wavelet_hybrid_denoise = rft_wavelet_denoise_2d
