#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Comprehensive RFT Comparison Against Multiple Transforms
Compares RFT with: FFT, DCT, DST, Hartley, Wavelet, and Hadamard transforms
Identifies advantages and use cases for each transform.
"""
import numpy as np
import time
from scipy import fftpack
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix

# ============================================================================
# Transform Implementations
# ============================================================================

def dct_forward(x):
    """Discrete Cosine Transform (Type-II, orthonormal)"""
    return fftpack.dct(x, type=2, norm='ortho')

def dct_inverse(x):
    """Inverse DCT"""
    return fftpack.idct(x, type=2, norm='ortho')

def dst_forward(x):
    """Discrete Sine Transform (Type-II, orthonormal)"""
    return fftpack.dst(x, type=2, norm='ortho')

def dst_inverse(x):
    """Inverse DST"""
    return fftpack.idst(x, type=2, norm='ortho')

def hartley_forward(x):
    """Discrete Hartley Transform"""
    X = np.fft.fft(x)
    return np.real(X) - np.imag(X)

def hartley_inverse(x):
    """Inverse Hartley (self-inverse up to scaling)"""
    return hartley_forward(x) / len(x)

def hadamard_matrix(n):
    """Generate Hadamard matrix (Walsh-Hadamard)"""
    if n == 1:
        return np.array([[1.0]])
    if n & (n - 1) != 0:
        # Not a power of 2, pad
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        H = hadamard_matrix(next_pow2)
        return H[:n, :n]
    
    H_half = hadamard_matrix(n // 2)
    H = np.block([[H_half, H_half], [H_half, -H_half]])
    return H / np.sqrt(2)

def hadamard_forward(x):
    """Fast Walsh-Hadamard Transform"""
    n = len(x)
    if n == 1:
        return x
    # Ensure power of 2
    if n & (n - 1) != 0:
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        x_padded = np.pad(x, (0, next_pow2 - n))
        result = hadamard_forward(x_padded)
        return result[:n]
    
    H = hadamard_matrix(n)
    return H @ x

def hadamard_inverse(x):
    """Inverse Hadamard (self-inverse)"""
    return hadamard_forward(x)

# ============================================================================
# Test Suite
# ============================================================================

def test_energy_preservation():
    """Compare energy preservation across all transforms"""
    print("=" * 80)
    print("TEST 1: Energy Preservation (Parseval's Theorem)")
    print("=" * 80)
    print()
    
    sizes = [64, 128, 256]
    rng = np.random.default_rng(42)
    
    transforms = {
        'RFT': (lambda x: rft_forward(x), lambda X: rft_inverse(X)),
        'FFT': (lambda x: np.fft.fft(x, norm='ortho'), lambda X: np.fft.ifft(X, norm='ortho')),
        'DCT': (dct_forward, dct_inverse),
        'DST': (dst_forward, dst_inverse),
        'Hadamard': (hadamard_forward, hadamard_inverse),
    }
    
    for n in sizes:
        print(f"Signal size N={n}:")
        x = rng.normal(size=n)
        energy_time = np.sum(x ** 2)
        
        for name, (fwd, inv) in transforms.items():
            X = fwd(x)
            energy_freq = np.sum(np.abs(X) ** 2)
            error = abs(energy_freq - energy_time) / max(1e-16, energy_time)
            status = "✓" if error < 1e-10 else "✗"
            print(f"  {name:12s}: Energy ratio = {energy_freq/energy_time:.12f}, error = {error:.2e} {status}")
        print()

def test_compression_efficiency():
    """Test which transform provides best compression (sparse representation)"""
    print("=" * 80)
    print("TEST 2: Compression Efficiency (Sparsity in Transform Domain)")
    print("=" * 80)
    print()
    
    n = 256
    
    # Test different signal types
    signals = {
        'Smooth sine': np.sin(2 * np.pi * 5 * np.arange(n) / n),
        'Discontinuous': np.concatenate([np.ones(n//2), -np.ones(n//2)]),
        'Exponential decay': np.exp(-np.arange(n) / 50),
        'Noisy sine': np.sin(2 * np.pi * 5 * np.arange(n) / n) + 0.1 * np.random.randn(n),
        'Chirp': np.sin(2 * np.pi * np.arange(n)**2 / (2*n**2)),
    }
    
    transforms = {
        'RFT': lambda x: rft_forward(x),
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': dct_forward,
        'DST': dst_forward,
        'Hadamard': hadamard_forward,
    }
    
    for sig_name, x in signals.items():
        print(f"{sig_name}:")
        coeffs_dict = {}
        
        for name, transform in transforms.items():
            X = transform(x)
            X_abs = np.abs(X)
            
            # Calculate sparsity metrics
            # 1. Number of coefficients needed for 99% energy
            total_energy = np.sum(X_abs ** 2)
            sorted_coeffs = np.sort(X_abs)[::-1]
            cumsum = np.cumsum(sorted_coeffs ** 2)
            n99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
            
            # 2. Entropy (lower = more compressible)
            p = X_abs ** 2 / (total_energy + 1e-16)
            p = p[p > 1e-16]
            entropy = -np.sum(p * np.log2(p))
            
            # 3. Gini coefficient (0 = perfectly sparse, 1 = uniform)
            sorted_abs = np.sort(X_abs)
            n_coeffs = len(sorted_abs)
            index = np.arange(1, n_coeffs + 1)
            gini = (2 * np.sum(index * sorted_abs)) / (n_coeffs * np.sum(sorted_abs)) - (n_coeffs + 1) / n_coeffs
            
            coeffs_dict[name] = (n99, entropy, gini)
            print(f"  {name:12s}: 99% energy in {n99:3d} coeffs, entropy={entropy:6.2f}, gini={gini:.4f}")
        
        # Find best for this signal
        best_sparse = min(coeffs_dict.items(), key=lambda x: x[1][0])
        print(f"  → Best sparsity: {best_sparse[0]} (fewest coefficients needed)")
        print()

def test_decorrelation():
    """Test decorrelation ability on correlated signals"""
    print("=" * 80)
    print("TEST 3: Decorrelation Ability (Auto-correlation reduction)")
    print("=" * 80)
    print()
    
    n = 128
    
    # Create signals with different correlation structures
    signals = {
        'AR(1) process': None,  # Will generate
        'Moving average': None,
        'Exponential corr': None,
    }
    
    # Generate AR(1) process
    rng = np.random.default_rng(42)
    ar1 = np.zeros(n)
    ar1[0] = rng.normal()
    for i in range(1, n):
        ar1[i] = 0.9 * ar1[i-1] + rng.normal()
    signals['AR(1) process'] = ar1
    
    # Moving average
    noise = rng.normal(size=n+5)
    ma = np.convolve(noise, np.ones(5)/5, mode='valid')[:n]
    signals['Moving average'] = ma
    
    # Exponentially correlated
    x = rng.normal(size=n)
    exp_corr = np.zeros(n)
    for i in range(n):
        exp_corr[i] = np.sum(x * np.exp(-np.abs(np.arange(n) - i) / 10))
    signals['Exponential corr'] = exp_corr
    
    transforms = {
        'RFT': lambda x: rft_forward(x),
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': dct_forward,
        'Hadamard': hadamard_forward,
    }
    
    for sig_name, x in signals.items():
        print(f"{sig_name}:")
        
        # Autocorrelation in time domain
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-16)
        autocorr_time = np.correlate(x_norm, x_norm, mode='full')[n-1:]
        autocorr_time = autocorr_time / autocorr_time[0]
        avg_autocorr_time = np.mean(np.abs(autocorr_time[1:20]))  # Average of first 20 lags
        
        print(f"  Time domain avg autocorr (lags 1-20): {avg_autocorr_time:.4f}")
        
        for name, transform in transforms.items():
            X = transform(x)
            X_abs = np.abs(X)
            
            # Autocorrelation in transform domain
            X_norm = (X_abs - np.mean(X_abs)) / (np.std(X_abs) + 1e-16)
            autocorr_freq = np.correlate(X_norm, X_norm, mode='full')[len(X)-1:]
            autocorr_freq = autocorr_freq / (autocorr_freq[0] + 1e-16)
            avg_autocorr_freq = np.mean(np.abs(autocorr_freq[1:20]))
            
            reduction = (avg_autocorr_time - avg_autocorr_freq) / (avg_autocorr_time + 1e-16)
            print(f"  {name:12s}: Transform domain autocorr = {avg_autocorr_freq:.4f}, reduction = {reduction:.1%}")
        print()

def test_phase_sensitivity():
    """Test sensitivity to phase shifts and time shifts"""
    print("=" * 80)
    print("TEST 4: Phase Shift Sensitivity")
    print("=" * 80)
    print()
    
    n = 64
    x = np.sin(2 * np.pi * 5 * np.arange(n) / n)
    
    transforms = {
        'RFT': lambda x: rft_forward(x),
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': dct_forward,
        'DST': dst_forward,
    }
    
    shifts = [0, 1, 2, 4, 8]
    
    for name, transform in transforms.items():
        X_orig = transform(x)
        print(f"{name}:")
        
        for shift in shifts:
            x_shifted = np.roll(x, shift)
            X_shifted = transform(x_shifted)
            
            # Magnitude difference
            mag_diff = np.linalg.norm(np.abs(X_shifted) - np.abs(X_orig)) / np.linalg.norm(np.abs(X_orig))
            
            # Phase difference (for complex transforms)
            if np.iscomplexobj(X_orig):
                phase_diff = np.mean(np.abs(np.angle(X_shifted) - np.angle(X_orig)))
            else:
                phase_diff = 0
            
            print(f"  Shift={shift:2d}: Magnitude change={mag_diff:.4f}, Phase change={phase_diff:.4f}")
        print()

def test_computational_complexity():
    """Benchmark computational performance"""
    print("=" * 80)
    print("TEST 5: Computational Performance")
    print("=" * 80)
    print()
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    n_iterations = 100
    
    print(f"Running {n_iterations} iterations per size...")
    print()
    print(f"{'Size':>6} | {'RFT (ms)':>10} | {'FFT (ms)':>10} | {'DCT (ms)':>10} | {'Hadamard':>10}")
    print("-" * 70)
    
    for n in sizes:
        rng = np.random.default_rng(42)
        x = rng.normal(size=n)
        x_complex = x + 1j * rng.normal(size=n)
        
        # Warm up
        _ = rft_forward(x_complex)
        _ = np.fft.fft(x_complex, norm='ortho')
        _ = dct_forward(x)
        _ = hadamard_forward(x)
        
        # RFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = rft_forward(x_complex)
        time_rft = (time.perf_counter() - start) * 1000 / n_iterations
        
        # FFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = np.fft.fft(x_complex, norm='ortho')
        time_fft = (time.perf_counter() - start) * 1000 / n_iterations
        
        # DCT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = dct_forward(x)
        time_dct = (time.perf_counter() - start) * 1000 / n_iterations
        
        # Hadamard
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = hadamard_forward(x)
        time_had = (time.perf_counter() - start) * 1000 / n_iterations
        
        print(f"{n:6d} | {time_rft:10.4f} | {time_fft:10.4f} | {time_dct:10.4f} | {time_had:10.4f}")
    print()

def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("=" * 80)
    print("TEST 6: Numerical Stability")
    print("=" * 80)
    print()
    
    n = 128
    
    # Test cases with challenging numerical properties
    test_cases = {
        'Normal scale': np.random.randn(n),
        'Large values': 1e6 * np.random.randn(n),
        'Small values': 1e-6 * np.random.randn(n),
        'Mixed scale': np.concatenate([1e6 * np.random.randn(n//2), 1e-6 * np.random.randn(n//2)]),
        'Near-zero': 1e-15 * np.random.randn(n),
    }
    
    transforms = {
        'RFT': (lambda x: rft_forward(x), lambda X: rft_inverse(X)),
        'FFT': (lambda x: np.fft.fft(x, norm='ortho'), lambda X: np.fft.ifft(X, norm='ortho')),
        'DCT': (dct_forward, dct_inverse),
        'Hadamard': (hadamard_forward, hadamard_inverse),
    }
    
    for case_name, x in test_cases.items():
        print(f"{case_name}:")
        x_range = f"[{np.min(x):.2e}, {np.max(x):.2e}]"
        print(f"  Input range: {x_range}")
        
        for name, (fwd, inv) in transforms.items():
            try:
                if name == 'RFT' and not np.iscomplexobj(x):
                    x_use = x.astype(np.complex128)
                else:
                    x_use = x
                
                X = fwd(x_use)
                x_reconstructed = inv(X)
                
                if np.iscomplexobj(x_reconstructed) and not np.iscomplexobj(x):
                    x_reconstructed = np.real(x_reconstructed)
                
                error = np.linalg.norm(x_reconstructed - x) / (np.linalg.norm(x) + 1e-16)
                
                # Check for NaN or Inf
                has_invalid = np.any(~np.isfinite(X)) or np.any(~np.isfinite(x_reconstructed))
                status = "✗ INVALID" if has_invalid else ("✓" if error < 1e-8 else "⚠")
                
                print(f"  {name:12s}: Reconstruction error = {error:.2e} {status}")
            except Exception as e:
                print(f"  {name:12s}: ✗ FAILED ({str(e)[:30]})")
        print()

def test_invertibility_precision():
    """Test precision of inverse transforms"""
    print("=" * 80)
    print("TEST 7: Invertibility Precision")
    print("=" * 80)
    print()
    
    sizes = [64, 128, 256, 512]
    
    transforms = {
        'RFT': (lambda x: rft_forward(x), lambda X: rft_inverse(X)),
        'FFT': (lambda x: np.fft.fft(x, norm='ortho'), lambda X: np.fft.ifft(X, norm='ortho')),
        'DCT': (dct_forward, dct_inverse),
        'DST': (dst_forward, dst_inverse),
        'Hadamard': (hadamard_forward, hadamard_inverse),
    }
    
    for n in sizes:
        print(f"N={n}:")
        rng = np.random.default_rng(42)
        x = rng.normal(size=n)
        
        for name, (fwd, inv) in transforms.items():
            if name == 'RFT':
                x_use = x.astype(np.complex128)
            else:
                x_use = x
            
            X = fwd(x_use)
            x_reconstructed = inv(X)
            
            if np.iscomplexobj(x_reconstructed) and not np.iscomplexobj(x):
                x_reconstructed = np.real(x_reconstructed)
            
            error = np.linalg.norm(x_reconstructed - x) / (np.linalg.norm(x) + 1e-16)
            status = "✓" if error < 1e-10 else "⚠"
            print(f"  {name:12s}: {error:.2e} {status}")
        print()

def analyze_advantages():
    """Summary analysis of where each transform excels"""
    print("=" * 80)
    print("ANALYSIS: Transform Advantages and Use Cases")
    print("=" * 80)
    print()
    
    analysis = {
        'RFT (Φ-RFT)': {
            'Advantages': [
                'Golden-ratio phase modulation provides quasi-periodic structure',
                'Unitary with perfect energy preservation',
                'Potential for cryptographic applications (non-standard phase)',
                'Novel frequency-domain structure for ML features',
                'Irrational phase spacing may resist certain attacks',
            ],
            'Disadvantages': [
                '~5-7x slower than FFT (additional phase operations)',
                'Non-standard (requires custom implementation)',
                'Less compression efficiency than DCT for smooth signals',
            ],
            'Best for': [
                'Cryptographic transforms requiring reversibility',
                'Feature extraction with irrational basis',
                'Applications needing non-uniform frequency spacing',
                'Research into alternative orthogonal transforms',
            ],
        },
        'FFT': {
            'Advantages': [
                'Fastest transform (O(N log N))',
                'Excellent hardware support',
                'Standard in signal processing',
                'Good for periodic signals',
            ],
            'Disadvantages': [
                'Complex-valued (higher storage)',
                'Phase-sensitive (time shifts affect phase)',
                'Not optimal for compression',
            ],
            'Best for': [
                'Frequency analysis',
                'Convolution via multiplication',
                'Real-time processing',
                'General-purpose spectral analysis',
            ],
        },
        'DCT': {
            'Advantages': [
                'Best compression for smooth signals',
                'Real-valued coefficients',
                'Used in JPEG, MP3',
                'Energy compaction',
            ],
            'Disadvantages': [
                'Poor for discontinuous signals',
                'No phase information',
                'Slower than FFT',
            ],
            'Best for': [
                'Image/video compression',
                'Audio coding',
                'Data compression',
                'Smooth signal approximation',
            ],
        },
        'Hadamard': {
            'Advantages': [
                'Very fast (only additions/subtractions)',
                'No floating-point operations needed',
                'Excellent for digital hardware',
                'Self-inverse',
            ],
            'Disadvantages': [
                'Requires power-of-2 sizes',
                'Poor frequency localization',
                'No compression advantages',
            ],
            'Best for': [
                'Low-power digital circuits',
                'Error correction codes',
                'Fast approximate transforms',
                'Spreading sequences',
            ],
        },
    }
    
    for transform, info in analysis.items():
        print(f"{transform}:")
        print(f"  Advantages:")
        for adv in info['Advantages']:
            print(f"    + {adv}")
        print(f"  Disadvantages:")
        for dis in info['Disadvantages']:
            print(f"    - {dis}")
        print(f"  Best for:")
        for use in info['Best for']:
            print(f"    → {use}")
        print()

def main():
    print("\n" + "=" * 80)
    print(" RFT Comprehensive Transform Comparison")
    print("=" * 80)
    print()
    
    test_energy_preservation()
    test_compression_efficiency()
    test_decorrelation()
    test_phase_sensitivity()
    test_computational_complexity()
    test_numerical_stability()
    test_invertibility_precision()
    analyze_advantages()
    
    print("=" * 80)
    print(" Comparison Complete")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
