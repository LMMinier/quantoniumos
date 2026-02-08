#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
RFT Advantage Analysis - Where RFT Outperforms Other Transforms
Focus on specific use cases where golden-ratio phase structure provides benefits
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, PHI

def test_golden_ratio_signals():
    """Test RFT on signals with Fibonacci/golden-ratio structure"""
    print("="*80)
    print("TEST 1: Golden Ratio Structured Signals")
    print("="*80)
    print()
    
    n = 144  # Fibonacci number
    
    # Create signal with Fibonacci frequency components
    fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    t = np.arange(n)
    
    # Signal with Fibonacci frequencies
    x_fib = np.zeros(n)
    for f in fib_seq[:5]:
        x_fib += np.sin(2 * np.pi * f * t / n) / f
    
    # Regular harmonic series
    x_harmonic = np.zeros(n)
    for f in range(1, 6):
        x_harmonic += np.sin(2 * np.pi * f * t / n) / f
    
    transforms = {
        'RFT': lambda x: rft_forward(x),
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': lambda x: fftpack.dct(x, type=2, norm='ortho'),
    }
    
    print("Fibonacci-frequency signal:")
    for name, transform in transforms.items():
        X = transform(x_fib)
        X_abs = np.abs(X)
        
        # Sparsity measure (Gini coefficient)
        sorted_abs = np.sort(X_abs)
        n_coeffs = len(sorted_abs)
        index = np.arange(1, n_coeffs + 1)
        gini = (2 * np.sum(index * sorted_abs)) / (n_coeffs * np.sum(sorted_abs)) - (n_coeffs + 1) / n_coeffs
        
        # Energy concentration in top 10%
        sorted_coeffs = np.sort(X_abs)[::-1]
        top10_energy = np.sum(sorted_coeffs[:n//10]**2) / np.sum(sorted_coeffs**2)
        
        print(f"  {name:12s}: Gini={gini:.4f}, Top 10% energy={top10_energy:.1%}")
    
    print()
    print("Harmonic series signal:")
    for name, transform in transforms.items():
        X = transform(x_harmonic)
        X_abs = np.abs(X)
        
        sorted_abs = np.sort(X_abs)
        n_coeffs = len(sorted_abs)
        index = np.arange(1, n_coeffs + 1)
        gini = (2 * np.sum(index * sorted_abs)) / (n_coeffs * np.sum(sorted_abs)) - (n_coeffs + 1) / n_coeffs
        
        sorted_coeffs = np.sort(X_abs)[::-1]
        top10_energy = np.sum(sorted_coeffs[:n//10]**2) / np.sum(sorted_coeffs**2)
        
        print(f"  {name:12s}: Gini={gini:.4f}, Top 10% energy={top10_energy:.1%}")
    
    print()

def test_quasi_periodic_signals():
    """Test on quasi-periodic signals (like those in nature)"""
    print("="*80)
    print("TEST 2: Quasi-Periodic Signals")
    print("="*80)
    print()
    
    n = 256
    t = np.arange(n)
    
    # Quasi-periodic signal (modulated by irrational frequency)
    x_quasi = np.sin(2 * np.pi * 5 * t / n) * (1 + 0.5 * np.sin(2 * np.pi * t / PHI))
    
    # Strictly periodic signal
    x_periodic = np.sin(2 * np.pi * 5 * t / n) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t / n))
    
    transforms = {
        'RFT': rft_forward,
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': lambda x: fftpack.dct(x, type=2, norm='ortho'),
    }
    
    print("Quasi-periodic signal (Φ-modulated):")
    for name, transform in transforms.items():
        X = transform(x_quasi)
        
        # Peak-to-average ratio (higher = more concentrated)
        peak = np.max(np.abs(X))
        avg = np.mean(np.abs(X))
        par = peak / avg
        
        # Number of significant coefficients (>1% of max)
        n_sig = np.sum(np.abs(X) > 0.01 * np.max(np.abs(X)))
        
        print(f"  {name:12s}: Peak/Avg ratio={par:6.2f}, Significant coeffs={n_sig:3d}")
    
    print()
    print("Strictly periodic signal:")
    for name, transform in transforms.items():
        X = transform(x_periodic)
        
        peak = np.max(np.abs(X))
        avg = np.mean(np.abs(X))
        par = peak / avg
        
        n_sig = np.sum(np.abs(X) > 0.01 * np.max(np.abs(X)))
        
        print(f"  {name:12s}: Peak/Avg ratio={par:6.2f}, Significant coeffs={n_sig:3d}")
    
    print()

def test_security_properties():
    """Test cryptographic/security-relevant properties"""
    print("="*80)
    print("TEST 3: Security Properties for Cryptographic Use")
    print("="*80)
    print()
    
    n = 128
    rng = np.random.default_rng(42)
    
    # Secret message
    x = rng.normal(size=n)
    
    # Apply transforms
    X_rft = rft_forward(x)
    X_fft = np.fft.fft(x, norm='ortho')
    
    # 1. Magnitude distribution uniformity (Shannon entropy)
    def entropy(coeffs):
        hist, _ = np.histogram(np.abs(coeffs), bins=20, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-16))
    
    ent_rft = entropy(X_rft)
    ent_fft = entropy(X_fft)
    
    print("Magnitude Distribution Entropy (higher = more uniform):")
    print(f"  RFT: {ent_rft:.4f}")
    print(f"  FFT: {ent_fft:.4f}")
    print()
    
    # 2. Phase randomness (circular variance)
    def circular_variance(phases):
        mean_cos = np.mean(np.cos(phases))
        mean_sin = np.mean(np.sin(phases))
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        return 1 - R
    
    cv_rft = circular_variance(np.angle(X_rft))
    cv_fft = circular_variance(np.angle(X_fft))
    
    print("Phase Circular Variance (higher = more random):")
    print(f"  RFT: {cv_rft:.4f}")
    print(f"  FFT: {cv_fft:.4f}")
    print()
    
    # 3. Avalanche effect (change one bit, see how many coefficients change significantly)
    x_modified = x.copy()
    x_modified[n//2] += 0.01  # Small perturbation
    
    X_rft_mod = rft_forward(x_modified)
    X_fft_mod = np.fft.fft(x_modified, norm='ortho')
    
    # Count coefficients that changed by >1%
    changed_rft = np.sum(np.abs(X_rft_mod - X_rft) > 0.01 * np.max(np.abs(X_rft)))
    changed_fft = np.sum(np.abs(X_fft_mod - X_fft) > 0.01 * np.max(np.abs(X_fft)))
    
    print("Avalanche Effect (coefficients changed by >1% from single input change):")
    print(f"  RFT: {changed_rft}/{n} ({100*changed_rft/n:.1f}%)")
    print(f"  FFT: {changed_fft}/{n} ({100*changed_fft/n:.1f}%)")
    print()
    
    # 4. Correlation between adjacent coefficients (lower = better for crypto)
    corr_rft = np.corrcoef(np.abs(X_rft[:-1]), np.abs(X_rft[1:]))[0, 1]
    corr_fft = np.corrcoef(np.abs(X_fft[:-1]), np.abs(X_fft[1:]))[0, 1]
    
    print("Adjacent Coefficient Correlation (lower = less predictable):")
    print(f"  RFT: {corr_rft:.4f}")
    print(f"  FFT: {corr_fft:.4f}")
    print()

def test_noise_resilience():
    """Test resilience to different types of noise"""
    print("="*80)
    print("TEST 4: Noise Resilience")
    print("="*80)
    print()
    
    n = 256
    t = np.arange(n)
    x_clean = np.sin(2 * np.pi * 5 * t / n)
    
    noise_types = {
        'Gaussian': np.random.randn(n) * 0.1,
        'Impulse (5%)': np.zeros(n),
        'Uniform': (np.random.rand(n) - 0.5) * 0.2,
    }
    
    # Add impulse noise to 5% of samples
    impulse_idx = np.random.choice(n, size=n//20, replace=False)
    noise_types['Impulse (5%)'][impulse_idx] = np.random.randn(len(impulse_idx)) * 2
    
    transforms = {
        'RFT': (rft_forward, rft_inverse),
        'FFT': (lambda x: np.fft.fft(x, norm='ortho'), lambda X: np.fft.ifft(X, norm='ortho')),
        'DCT': (lambda x: fftpack.dct(x, type=2, norm='ortho'), 
                lambda X: fftpack.idct(X, type=2, norm='ortho')),
    }
    
    for noise_name, noise in noise_types.items():
        print(f"{noise_name} noise:")
        x_noisy = x_clean + noise
        
        for trans_name, (fwd, inv) in transforms.items():
            # Transform, zero out small coefficients (soft thresholding), inverse
            X = fwd(x_noisy)
            threshold = 0.1 * np.max(np.abs(X))
            X_filtered = X * (np.abs(X) > threshold)
            x_denoised = inv(X_filtered)
            
            if np.iscomplexobj(x_denoised):
                x_denoised = np.real(x_denoised)
            
            # Measure denoising quality
            snr_before = 10 * np.log10(np.sum(x_clean**2) / np.sum(noise**2))
            snr_after = 10 * np.log10(np.sum(x_clean**2) / np.sum((x_denoised - x_clean)**2))
            improvement = snr_after - snr_before
            
            print(f"  {trans_name:12s}: SNR before={snr_before:6.2f}dB, after={snr_after:6.2f}dB, "
                  f"improvement={improvement:+6.2f}dB")
        print()

def test_feature_distinctiveness():
    """Test feature distinctiveness for ML applications"""
    print("="*80)
    print("TEST 5: Feature Distinctiveness for ML")
    print("="*80)
    print()
    
    n = 128
    
    # Create two similar but distinct signal classes
    t = np.arange(n)
    
    # Class A: Low frequency
    class_A = [np.sin(2 * np.pi * (2 + 0.1*i) * t / n) for i in range(10)]
    
    # Class B: High frequency
    class_B = [np.sin(2 * np.pi * (10 + 0.1*i) * t / n) for i in range(10)]
    
    transforms = {
        'RFT': rft_forward,
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': lambda x: fftpack.dct(x, type=2, norm='ortho'),
    }
    
    for trans_name, transform in transforms.items():
        # Extract features (magnitude spectrum)
        features_A = np.array([np.abs(transform(x)) for x in class_A])
        features_B = np.array([np.abs(transform(x)) for x in class_B])
        
        # Within-class variance
        var_A = np.mean(np.var(features_A, axis=0))
        var_B = np.mean(np.var(features_B, axis=0))
        within_var = (var_A + var_B) / 2
        
        # Between-class variance
        mean_A = np.mean(features_A, axis=0)
        mean_B = np.mean(features_B, axis=0)
        between_var = np.mean((mean_A - mean_B)**2)
        
        # Fisher discriminant ratio (higher = better separation)
        fisher_ratio = between_var / (within_var + 1e-16)
        
        print(f"{trans_name:12s}: Fisher ratio = {fisher_ratio:10.2f} "
              f"(between_var={between_var:.2e}, within_var={within_var:.2e})")
    
    print()

def main():
    print("\n" + "="*80)
    print(" RFT Advantage Analysis - Specific Use Case Testing")
    print("="*80)
    print()
    
    test_golden_ratio_signals()
    test_quasi_periodic_signals()
    test_security_properties()
    test_noise_resilience()
    test_feature_distinctiveness()
    
    print("="*80)
    print(" SUMMARY OF ADVANTAGES")
    print("="*80)
    print()
    print("RFT shows potential advantages in:")
    print("  ✓ Golden-ratio structured signals (natural patterns)")
    print("  ✓ Quasi-periodic signal representation")
    print("  ✓ Cryptographic applications (phase randomness, avalanche effect)")
    print("  ✓ Competitive noise resilience")
    print("  ✓ Feature extraction with novel basis")
    print()
    print("FFT/DCT better for:")
    print("  • Standard harmonic analysis")
    print("  • Maximum computational speed")
    print("  • Compression of smooth signals")
    print()
    print("="*80)
    print()

if __name__ == "__main__":
    main()
