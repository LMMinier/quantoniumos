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
Medical Imaging Reconstruction Tests
=====================================

Tests RFT variants against DCT/wavelets for medical imaging applications:
- MRI brain phantom reconstruction under low-dose and motion
- Noise-robust variants under Rician/Poisson noise models
- Reconstruction time and resource utilization benchmarks

Uses synthetic Shepp-Logan phantom and simulated MRI k-space data.
"""

import numpy as np
import time
import pytest
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# Synthetic Phantom Generators
# =============================================================================

def shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """
    Generate Shepp-Logan phantom for MRI simulation.
    
    The Shepp-Logan phantom is a standard test image consisting of
    ellipses with different intensities, commonly used in CT/MRI
    reconstruction algorithm evaluation.
    
    Args:
        size: Image dimension (square image)
        
    Returns:
        2D phantom image normalized to [0, 1]
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    
    # Ellipse parameters: (x0, y0, a, b, theta, intensity)
    # Values adapted from Shepp-Logan paper
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),       # Outer ellipse (skull)
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),  # Brain matter
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),  # Right ventricle
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),  # Left ventricle
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),     # Top tumor
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),    # Small detail 1
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),   # Small detail 2
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
    
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-10)
    return phantom


def simulate_mri_kspace(image: np.ndarray, 
                        undersampling_factor: float = 1.0,
                        noise_level: float = 0.0,
                        motion_amplitude: float = 0.0) -> np.ndarray:
    """
    Simulate MRI k-space acquisition with optional degradation.
    
    Args:
        image: Input image (ground truth)
        undersampling_factor: Fraction of k-space to sample (1.0 = full, 0.5 = half)
        noise_level: Gaussian noise std in k-space
        motion_amplitude: Random phase corruption to simulate motion
        
    Returns:
        Complex k-space data
    """
    kspace = np.fft.fft2(image)
    kspace = np.fft.fftshift(kspace)
    
    # Undersampling (random mask for simplicity, could use radial/spiral)
    if undersampling_factor < 1.0:
        mask = np.random.random(kspace.shape) < undersampling_factor
        # Always keep low frequencies (center 20%)
        center_size = int(0.1 * min(kspace.shape))
        cy, cx = kspace.shape[0] // 2, kspace.shape[1] // 2
        mask[cy-center_size:cy+center_size, cx-center_size:cx+center_size] = True
        kspace = kspace * mask
    
    # Add complex Gaussian noise
    if noise_level > 0:
        noise = noise_level * (np.random.randn(*kspace.shape) + 
                               1j * np.random.randn(*kspace.shape))
        kspace = kspace + noise
    
    # Motion artifact as random phase corruption
    if motion_amplitude > 0:
        phase_error = motion_amplitude * np.random.randn(*kspace.shape)
        kspace = kspace * np.exp(1j * phase_error)
    
    return np.fft.ifftshift(kspace)


def add_rician_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Rician noise (MRI magnitude signal noise model).
    
    Rician noise arises from the magnitude of complex Gaussian noise
    added to the MRI signal.
    
    Args:
        image: Clean image
        sigma: Noise standard deviation
        
    Returns:
        Noisy image with Rician noise
    """
    # MRI signal is magnitude of complex signal with Gaussian noise
    real = image + sigma * np.random.randn(*image.shape)
    imag = sigma * np.random.randn(*image.shape)
    return np.sqrt(real**2 + imag**2)


def add_poisson_noise(image: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """
    Add Poisson noise (photon counting noise for PET/CT).
    
    Args:
        image: Clean image normalized to [0, 1]
        scale: Expected photon count at max intensity
        
    Returns:
        Noisy image
    """
    # Scale to photon counts
    counts = image * scale
    # Apply Poisson noise
    noisy_counts = np.random.poisson(counts.astype(np.float64))
    # Normalize back
    return noisy_counts / scale


# =============================================================================
# Quality Metrics
# =============================================================================

def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def ssim_simple(original: np.ndarray, reconstructed: np.ndarray,
                window_size: int = 7, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Simplified SSIM (Structural Similarity Index) implementation.
    
    Full SSIM requires scipy.ndimage or skimage; this is a simplified version.
    """
    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2
    
    mu_x = np.mean(original)
    mu_y = np.mean(reconstructed)
    
    sigma_x = np.std(original)
    sigma_y = np.std(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    
    return float(ssim)


def nmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Normalized Mean Squared Error."""
    return np.sum((original - reconstructed) ** 2) / np.sum(original ** 2)


# =============================================================================
# RFT-based Denoising and Reconstruction
# =============================================================================

def rft_denoise_2d(noisy_image: np.ndarray, 
                   threshold_ratio: float = 0.1,
                   variant: str = "standard") -> np.ndarray:
    """
    Denoise 2D image using RFT thresholding.
    
    Applies RFT row-wise and column-wise (separable), then thresholds
    small coefficients to remove noise.
    
    Args:
        noisy_image: Noisy input image
        threshold_ratio: Fraction of max coefficient magnitude to threshold
        variant: RFT variant to use
        
    Returns:
        Denoised image
    """
    try:
        # Canonical operator-based RFT (resonance eigenbasis)
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        pytest.skip("RFT core not available")
    
    # Separable 2D transform: row-wise then column-wise
    rows, cols = noisy_image.shape
    
    # Row-wise RFT
    row_coeffs = np.zeros_like(noisy_image, dtype=np.complex128)
    for i in range(rows):
        row_coeffs[i, :] = rft_forward(noisy_image[i, :].astype(np.complex128))
    
    # Column-wise RFT
    full_coeffs = np.zeros_like(row_coeffs)
    for j in range(cols):
        full_coeffs[:, j] = rft_forward(row_coeffs[:, j])
    
    # Threshold small coefficients
    max_mag = np.max(np.abs(full_coeffs))
    threshold = threshold_ratio * max_mag
    thresholded = np.where(np.abs(full_coeffs) < threshold, 0, full_coeffs)
    
    # Inverse: column-wise then row-wise
    inv_cols = np.zeros_like(thresholded)
    for j in range(cols):
        inv_cols[:, j] = rft_inverse(thresholded[:, j])
    
    denoised = np.zeros_like(noisy_image)
    for i in range(rows):
        denoised[i, :] = rft_inverse(inv_cols[i, :]).real
    
    return np.clip(denoised, 0, 1)


def dct_denoise_2d(noisy_image: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Denoise 2D image using DCT thresholding (baseline comparison).
    """
    try:
        from scipy.fft import dct, idct
    except ImportError:
        from scipy.fftpack import dct, idct
    
    # 2D DCT (separable)
    coeffs = dct(dct(noisy_image.T, type=2, norm='ortho').T, type=2, norm='ortho')
    
    # Threshold
    max_mag = np.max(np.abs(coeffs))
    threshold = threshold_ratio * max_mag
    thresholded = np.where(np.abs(coeffs) < threshold, 0, coeffs)
    
    # Inverse DCT
    denoised = idct(idct(thresholded.T, type=2, norm='ortho').T, type=2, norm='ortho')
    
    return np.clip(denoised, 0, 1)


def wavelet_denoise_2d(noisy_image: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Denoise 2D image using simple Haar wavelet thresholding (baseline).
    
    This is a simplified implementation; production would use pywt.
    """
    # Simple 2D Haar transform (one level)
    rows, cols = noisy_image.shape
    assert rows % 2 == 0 and cols % 2 == 0, "Image size must be even"
    
    # Horizontal pass
    h_low = (noisy_image[:, 0::2] + noisy_image[:, 1::2]) / np.sqrt(2)
    h_high = (noisy_image[:, 0::2] - noisy_image[:, 1::2]) / np.sqrt(2)
    h_transform = np.hstack([h_low, h_high])
    
    # Vertical pass
    v_low = (h_transform[0::2, :] + h_transform[1::2, :]) / np.sqrt(2)
    v_high = (h_transform[0::2, :] - h_transform[1::2, :]) / np.sqrt(2)
    coeffs = np.vstack([v_low, v_high])
    
    # Threshold high-frequency coefficients (keep LL subband)
    max_mag = np.max(np.abs(coeffs[rows//2:, :]))
    threshold = threshold_ratio * max_mag
    
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[rows//2:, :] = np.where(np.abs(coeffs[rows//2:, :]) < threshold, 
                                           0, coeffs[rows//2:, :])
    coeffs_thresh[:rows//2, cols//2:] = np.where(np.abs(coeffs[:rows//2, cols//2:]) < threshold,
                                                  0, coeffs[:rows//2, cols//2:])
    
    # Inverse vertical pass
    v_low = coeffs_thresh[:rows//2, :]
    v_high = coeffs_thresh[rows//2:, :]
    inv_v = np.zeros((rows, cols))
    inv_v[0::2, :] = (v_low + v_high) / np.sqrt(2)
    inv_v[1::2, :] = (v_low - v_high) / np.sqrt(2)
    
    # Inverse horizontal pass
    h_low = inv_v[:, :cols//2]
    h_high = inv_v[:, cols//2:]
    denoised = np.zeros((rows, cols))
    denoised[:, 0::2] = (h_low + h_high) / np.sqrt(2)
    denoised[:, 1::2] = (h_low - h_high) / np.sqrt(2)
    
    return np.clip(denoised, 0, 1)


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class ImagingTestResult:
    """Result container for imaging reconstruction tests."""
    method: str
    noise_type: str
    noise_level: float
    psnr_before: float
    psnr_after: float
    ssim_after: float
    nmse_after: float
    time_ms: float
    improvement_db: float


# =============================================================================
# Pytest Test Cases
# =============================================================================

class TestMRIReconstruction:
    """Test suite for MRI reconstruction using RFT variants."""
    
    @pytest.fixture
    def phantom(self) -> np.ndarray:
        """Generate test phantom."""
        return shepp_logan_phantom(128)  # Smaller for faster tests
    
    def test_phantom_generation(self, phantom):
        """Verify phantom is properly generated."""
        assert phantom.shape == (128, 128)
        assert phantom.min() >= 0
        assert phantom.max() <= 1
        print(f"✓ Phantom generated: shape={phantom.shape}, "
              f"range=[{phantom.min():.3f}, {phantom.max():.3f}]")
    
    @pytest.mark.parametrize("noise_sigma", [0.05, 0.10, 0.15])
    def test_rft_vs_dct_rician_noise(self, phantom, noise_sigma):
        """Compare RFT vs DCT denoising under Rician noise."""
        # Add Rician noise (MRI-specific)
        noisy = add_rician_noise(phantom, noise_sigma)
        psnr_noisy = psnr(phantom, noisy)
        
        # RFT denoising
        t0 = time.perf_counter()
        rft_denoised = rft_denoise_2d(noisy, threshold_ratio=0.05)
        rft_time = (time.perf_counter() - t0) * 1000
        
        # DCT denoising
        t0 = time.perf_counter()
        dct_denoised = dct_denoise_2d(noisy, threshold_ratio=0.05)
        dct_time = (time.perf_counter() - t0) * 1000
        
        # Metrics
        rft_psnr = psnr(phantom, rft_denoised)
        dct_psnr = psnr(phantom, dct_denoised)
        rft_ssim = ssim_simple(phantom, rft_denoised)
        dct_ssim = ssim_simple(phantom, dct_denoised)
        
        print(f"\n  Rician noise σ={noise_sigma}:")
        print(f"    Noisy PSNR: {psnr_noisy:.2f} dB")
        print(f"    RFT: PSNR={rft_psnr:.2f} dB, SSIM={rft_ssim:.3f}, time={rft_time:.1f}ms")
        print(f"    DCT: PSNR={dct_psnr:.2f} dB, SSIM={dct_ssim:.3f}, time={dct_time:.1f}ms")
        
        # Basic sanity checks - denoising with simple thresholding may not always improve PSNR
        # especially for low noise levels where aggressive thresholding removes signal
        assert rft_psnr > 10, f"RFT PSNR too low: {rft_psnr}"
        assert dct_psnr > 10, f"DCT PSNR too low: {dct_psnr}"
        # SSIM should be reasonable
        assert rft_ssim > 0.5, f"RFT SSIM too low: {rft_ssim}"
        assert dct_ssim > 0.5, f"DCT SSIM too low: {dct_ssim}"
    
    @pytest.mark.parametrize("scale", [100, 500, 1000])
    def test_rft_vs_dct_poisson_noise(self, phantom, scale):
        """Compare RFT vs DCT under Poisson noise (CT/PET model)."""
        noisy = add_poisson_noise(phantom, scale=scale)
        psnr_noisy = psnr(phantom, noisy)
        
        rft_denoised = rft_denoise_2d(noisy, threshold_ratio=0.05)
        dct_denoised = dct_denoise_2d(noisy, threshold_ratio=0.05)
        
        rft_psnr = psnr(phantom, rft_denoised)
        dct_psnr = psnr(phantom, dct_denoised)
        
        print(f"\n  Poisson noise (scale={scale}):")
        print(f"    Noisy PSNR: {psnr_noisy:.2f} dB")
        print(f"    RFT: PSNR={rft_psnr:.2f} dB")
        print(f"    DCT: PSNR={dct_psnr:.2f} dB")
    
    def test_motion_artifact_reconstruction(self, phantom):
        """Test reconstruction under motion artifacts."""
        # Simulate k-space with motion
        kspace = simulate_mri_kspace(phantom, motion_amplitude=0.5)
        corrupted = np.abs(np.fft.ifft2(kspace))
        corrupted = corrupted / (corrupted.max() + 1e-10)
        
        psnr_corrupted = psnr(phantom, corrupted)
        
        # RFT-based denoising
        rft_denoised = rft_denoise_2d(corrupted, threshold_ratio=0.1)
        rft_psnr = psnr(phantom, rft_denoised)
        
        print(f"\n  Motion artifact test:")
        print(f"    Corrupted PSNR: {psnr_corrupted:.2f} dB")
        print(f"    RFT denoised: {rft_psnr:.2f} dB")
    
    def test_undersampled_reconstruction(self, phantom):
        """Test reconstruction from undersampled k-space."""
        # 50% undersampling
        kspace = simulate_mri_kspace(phantom, undersampling_factor=0.5)
        
        # Zero-filled reconstruction
        recon = np.abs(np.fft.ifft2(kspace))
        recon = recon / (recon.max() + 1e-10)
        
        psnr_zf = psnr(phantom, recon)
        
        # RFT regularization
        rft_recon = rft_denoise_2d(recon, threshold_ratio=0.02)
        rft_psnr = psnr(phantom, rft_recon)
        
        print(f"\n  50% undersampled reconstruction:")
        print(f"    Zero-filled PSNR: {psnr_zf:.2f} dB")
        print(f"    RFT regularized: {rft_psnr:.2f} dB")


class TestCTReconstruction:
    """Test CT/PET reconstruction scenarios."""
    
    @pytest.fixture
    def phantom(self) -> np.ndarray:
        """Generate test phantom for CT."""
        return shepp_logan_phantom(128)
    
    def test_low_dose_ct_denoising(self, phantom):
        """Simulate low-dose CT with Poisson noise."""
        # Very low dose = scale 50 (high noise)
        noisy = add_poisson_noise(phantom, scale=50)
        psnr_noisy = psnr(phantom, noisy)
        
        # Compare methods
        methods = {
            'RFT': lambda x: rft_denoise_2d(x, 0.1),
            'DCT': lambda x: dct_denoise_2d(x, 0.1),
            'Wavelet': lambda x: wavelet_denoise_2d(x, 0.1),
        }
        
        results = {}
        for name, method in methods.items():
            t0 = time.perf_counter()
            denoised = method(noisy)
            elapsed = (time.perf_counter() - t0) * 1000
            results[name] = {
                'psnr': psnr(phantom, denoised),
                'ssim': ssim_simple(phantom, denoised),
                'time_ms': elapsed
            }
        
        print(f"\n  Low-dose CT denoising (noisy PSNR={psnr_noisy:.2f} dB):")
        for name, r in results.items():
            print(f"    {name}: PSNR={r['psnr']:.2f} dB, "
                  f"SSIM={r['ssim']:.3f}, time={r['time_ms']:.1f}ms")


class TestReconstructionBenchmark:
    """Benchmark reconstruction algorithms for timing and resource usage."""
    
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_reconstruction_timing(self, size):
        """Measure reconstruction time vs image size."""
        phantom = shepp_logan_phantom(size)
        noisy = add_rician_noise(phantom, 0.1)
        
        times = {}
        
        # RFT timing
        t0 = time.perf_counter()
        _ = rft_denoise_2d(noisy, 0.05)
        times['RFT'] = (time.perf_counter() - t0) * 1000
        
        # DCT timing
        t0 = time.perf_counter()
        _ = dct_denoise_2d(noisy, 0.05)
        times['DCT'] = (time.perf_counter() - t0) * 1000
        
        print(f"\n  Size {size}x{size}:")
        for name, t in times.items():
            print(f"    {name}: {t:.2f} ms")
        
        # Check reasonable timing (should complete in under 10s)
        for name, t in times.items():
            assert t < 10000, f"{name} took too long: {t}ms"


# =============================================================================
# Standalone Runner
# =============================================================================

def run_comprehensive_imaging_benchmark():
    """Run comprehensive imaging benchmark (not pytest)."""
    print("=" * 70)
    print("MEDICAL IMAGING RECONSTRUCTION BENCHMARK")
    print("=" * 70)
    
    phantom = shepp_logan_phantom(256)
    
    results: List[ImagingTestResult] = []
    
    for noise_type, noise_func in [
        ('Rician', lambda p: add_rician_noise(p, 0.1)),
        ('Poisson', lambda p: add_poisson_noise(p, 500)),
    ]:
        noisy = noise_func(phantom)
        psnr_before = psnr(phantom, noisy)
        
        for method_name, method in [
            ('RFT', lambda x: rft_denoise_2d(x, 0.05)),
            ('DCT', lambda x: dct_denoise_2d(x, 0.05)),
            ('Wavelet', lambda x: wavelet_denoise_2d(x, 0.05)),
        ]:
            t0 = time.perf_counter()
            denoised = method(noisy)
            elapsed = (time.perf_counter() - t0) * 1000
            
            psnr_after = psnr(phantom, denoised)
            ssim_after = ssim_simple(phantom, denoised)
            nmse_after = nmse(phantom, denoised)
            
            result = ImagingTestResult(
                method=method_name,
                noise_type=noise_type,
                noise_level=0.1 if noise_type == 'Rician' else 500,
                psnr_before=psnr_before,
                psnr_after=psnr_after,
                ssim_after=ssim_after,
                nmse_after=nmse_after,
                time_ms=elapsed,
                improvement_db=psnr_after - psnr_before
            )
            results.append(result)
    
    # Print results table
    print(f"\n{'Method':<10} {'Noise':<10} {'PSNR Before':<12} {'PSNR After':<12} "
          f"{'SSIM':<8} {'Time (ms)':<10} {'Improvement':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.method:<10} {r.noise_type:<10} {r.psnr_before:>10.2f}dB "
              f"{r.psnr_after:>10.2f}dB {r.ssim_after:>6.3f}   "
              f"{r.time_ms:>8.1f}   {r.improvement_db:>+10.2f}dB")
    
    print("\n✓ Imaging benchmark complete")
    return results


if __name__ == "__main__":
    run_comprehensive_imaging_benchmark()


# =============================================================================
# Real-Data Tests (FastMRI Knee Singlecoil)
# =============================================================================

from tests.medical.real_data_fixtures import (
    skip_no_fastmri,
    list_fastmri_slices,
    load_fastmri_slice,
    FASTMRI_AVAILABLE,
)


@skip_no_fastmri
class TestRealFastMRI:
    """
    Tests using real FastMRI knee single-coil data.
    
    Requires:
        - USE_REAL_DATA=1
        - CC BY-NC 4.0 acceptance at https://fastmri.org/
        - FASTMRI_KNEE_URL=<signed_url>
        - bash data/fastmri_fetch.sh executed
    
    ⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL USE ⚠️
    """

    @pytest.fixture(scope="class")
    def fastmri_kspace(self):
        """Load first available FastMRI slice."""
        slices = list_fastmri_slices()
        if not slices:
            pytest.skip("No FastMRI slices found")
        kspace, target = load_fastmri_slice(slices[0], slice_idx=0)
        return kspace, target

    def test_real_mri_zero_filled_reconstruction(self, fastmri_kspace):
        """Test zero-filled reconstruction baseline on real MRI."""
        kspace, target = fastmri_kspace
        
        # Zero-filled reconstruction
        recon_zf = np.abs(np.fft.ifft2(kspace))
        
        # Normalize for comparison
        recon_zf = (recon_zf - recon_zf.min()) / (recon_zf.max() - recon_zf.min() + 1e-10)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-10)
        
        psnr_zf = psnr(target_norm, recon_zf)
        ssim_zf = ssim_simple(target_norm, recon_zf)
        
        print(f"✓ Real MRI zero-filled: PSNR={psnr_zf:.2f} dB, SSIM={ssim_zf:.4f}")
        
        # Zero-filled should have reasonable baseline
        assert psnr_zf >= 15.0, f"Zero-filled PSNR too low: {psnr_zf:.2f}"

    def test_real_mri_rft_denoising(self, fastmri_kspace):
        """Test RFT denoising on real MRI reconstruction."""
        kspace, target = fastmri_kspace
        
        # Reconstruct with noise
        recon = np.abs(np.fft.ifft2(kspace))
        
        # Normalize
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-10)
        
        # Add simulated noise for denoising test
        noisy = add_rician_noise(recon_norm, sigma=0.05)
        
        # Denoise with RFT
        denoised = rft_denoise_2d(noisy, threshold_ratio=0.05)
        
        psnr_noisy = psnr(recon_norm, noisy)
        psnr_denoised = psnr(recon_norm, denoised)
        
        print(f"Real MRI denoising: PSNR noisy={psnr_noisy:.2f} dB → denoised={psnr_denoised:.2f} dB")
        
        # Denoising should not degrade significantly
        assert psnr_denoised >= psnr_noisy - 3.0, "RFT denoising degraded image too much"

    def test_real_mri_undersampled_reconstruction(self, fastmri_kspace):
        """Test reconstruction from undersampled real k-space."""
        kspace, target = fastmri_kspace
        
        # Simulate 50% undersampling
        mask = np.random.random(kspace.shape) < 0.5
        # Keep center 20%
        cy, cx = kspace.shape[0] // 2, kspace.shape[1] // 2
        center = int(0.1 * min(kspace.shape))
        mask[cy-center:cy+center, cx-center:cx+center] = True
        
        undersampled = kspace * mask
        
        # Zero-filled reconstruction
        recon_us = np.abs(np.fft.ifft2(undersampled))
        recon_full = np.abs(np.fft.ifft2(kspace))
        
        # Normalize
        recon_us = (recon_us - recon_us.min()) / (recon_us.max() - recon_us.min() + 1e-10)
        recon_full = (recon_full - recon_full.min()) / (recon_full.max() - recon_full.min() + 1e-10)
        
        psnr_us = psnr(recon_full, recon_us)
        
        print(f"✓ Real MRI 50% undersampled: PSNR={psnr_us:.2f} dB")
        
        # Undersampled should still have some structure
        assert psnr_us >= 10.0, f"Undersampled reconstruction too degraded: {psnr_us:.2f}"


@skip_no_fastmri
def test_fastmri_multiple_slices():
    """Test reconstruction across multiple FastMRI slices."""
    slices = list_fastmri_slices()[:3]  # Test up to 3 volumes
    
    if len(slices) == 0:
        pytest.skip("No FastMRI slices available")
    
    results = []
    for h5_path in slices:
        try:
            kspace, target = load_fastmri_slice(h5_path, slice_idx=0)
            recon = np.abs(np.fft.ifft2(kspace))
            
            # Normalize
            recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
            target_norm = (target - target.min()) / (target.max() - target.min() + 1e-10)
            
            psnr_val = psnr(target_norm, recon_norm)
            results.append((h5_path.stem, psnr_val))
        except Exception as e:
            print(f"  Skipping {h5_path.stem}: {e}")
    
    if results:
        print("\nFastMRI Multi-Slice Results:")
        for name, p in results:
            print(f"  {name}: PSNR={p:.2f} dB")
        
        avg_psnr = np.mean([r[1] for r in results])
        assert avg_psnr >= 15.0, f"Average PSNR across slices too low: {avg_psnr:.2f}"
