#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Test and benchmark closed-form RFT against standard FFT
Validates:
- Transform correctness
- Unitarity
- Performance comparison
- Frequency domain behavior
"""
import numpy as np
import time
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_unitary_error_canonical as rft_unitary_error, rft_matrix_canonical as rft_matrix

def test_unitarity():
    """Test that RFT is a unitary transform"""
    sizes = [8, 16, 32, 64, 128, 256]
    for n in sizes:
        error = rft_unitary_error(n, trials=10)
        assert error < 1e-10, f"Unitarity failed for N={n}: error={error:.2e}"

def test_parseval():
    """Test Parseval's theorem (energy preservation)"""
    sizes = [8, 16, 32, 64, 128]
    for n in sizes:
        # Generate random signal
        rng = np.random.default_rng(42)
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        
        # Time domain energy
        energy_time = np.sum(np.abs(x) ** 2)
        
        # RFT frequency domain energy
        X_rft = rft_forward(x)
        energy_rft = np.sum(np.abs(X_rft) ** 2)
        
        # Compute relative error
        err_rft = abs(energy_rft - energy_time) / max(1e-16, energy_time)
        assert err_rft < 1e-10, f"Parseval failed for N={n}: error={err_rft:.2e}"

def test_orthogonality():
    """Test matrix orthogonality: Ψ† Ψ = I"""
    sizes = [8, 16, 32, 64]
    for n in sizes:
        Psi = rft_matrix(n)
        PsiH = np.conj(Psi.T)
        
        # Should be identity
        I_test = PsiH @ Psi
        I_true = np.eye(n, dtype=np.complex128)
        
        error = np.linalg.norm(I_test - I_true, 'fro') / n
        assert error < 1e-10, f"Orthogonality failed for N={n}: error={error:.2e}"

def test_signal_reconstruction():
    """Test reconstruction of known signals"""
    n = 64
    
    # Test signals
    signals = {
        "Impulse": np.array([1.0] + [0.0] * (n-1)),
        "Constant": np.ones(n),
        "Sine wave": np.sin(2 * np.pi * 5 * np.arange(n) / n),
        "Complex exp": np.exp(2j * np.pi * 3 * np.arange(n) / n),
        "Random": np.random.default_rng(123).normal(size=n) + 1j * np.random.default_rng(456).normal(size=n)
    }
    
    for name, x in signals.items():
        # RFT round trip
        X_rft = rft_forward(x)
        x_rec_rft = rft_inverse(X_rft)
        err_rft = np.linalg.norm(x_rec_rft - x) / max(1e-16, np.linalg.norm(x))
        assert err_rft < 1e-10, f"Reconstruction failed for {name}: error={err_rft:.2e}"

def test_spectrum_comparison():
    """Compare RFT and FFT spectral characteristics - norms should match"""
    n = 64
    # Create a test signal with multiple frequencies
    t = np.arange(n)
    x = (np.sin(2 * np.pi * 5 * t / n) + 
         0.5 * np.sin(2 * np.pi * 10 * t / n) + 
         0.25 * np.cos(2 * np.pi * 3 * t / n))
    
    X_rft = rft_forward(x)
    X_fft = np.fft.fft(x, norm="ortho")
    
    # The spectra will differ due to phase modulation, but norms should match
    norm_diff = abs(np.linalg.norm(X_rft) - np.linalg.norm(X_fft))
    assert norm_diff < 1e-10, f"Spectrum norm mismatch: diff={norm_diff:.2e}"

def _benchmark_performance():
    """Benchmark RFT vs FFT performance (run via __main__, not pytest)"""
    sizes = [64, 128, 256, 512, 1024, 2048]
    n_iterations = 100
    
    print(f"Running {n_iterations} iterations per size...\n")
    print(f"{'Size':>6} | {'RFT (ms)':>10} | {'FFT (ms)':>10} | {'Ratio':>8}")
    print("-" * 50)
    
    for n in sizes:
        rng = np.random.default_rng(789)
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        
        # Warm up
        _ = rft_forward(x)
        _ = np.fft.fft(x, norm="ortho")
        
        # Benchmark RFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = rft_forward(x)
        time_rft = (time.perf_counter() - start) * 1000 / n_iterations
        
        # Benchmark FFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = np.fft.fft(x, norm="ortho")
        time_fft = (time.perf_counter() - start) * 1000 / n_iterations
        
        ratio = time_rft / time_fft
        print(f"{n:6d} | {time_rft:10.4f} | {time_fft:10.4f} | {ratio:8.2f}x")


def test_linearity():
    """Test that RFT is linear: RFT(ax + by) = a*RFT(x) + b*RFT(y)"""
    n = 64
    rng = np.random.default_rng(999)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    a, b = 2.5 + 1.5j, -1.2 + 0.8j
    
    # Direct computation
    lhs = rft_forward(a * x + b * y)
    
    # Linear combination
    rhs = a * rft_forward(x) + b * rft_forward(y)
    
    error = np.linalg.norm(lhs - rhs) / max(1e-16, np.linalg.norm(lhs))
    assert error < 1e-10, f"Linearity failed: error={error:.2e}"


if __name__ == "__main__":
    # Run benchmark when executed directly
    _benchmark_performance()
