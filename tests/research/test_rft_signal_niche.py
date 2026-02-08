#!/usr/bin/env python3
"""
RFT Signal Niche Analysis
==========================

Honest benchmark: Where does RFT win vs FFT/DCT/Wavelets?

Signal Classes:
1. Golden-ratio quasi-periodic (RFT should win)
2. Integer-periodic (FFT should win)  
3. Transients/edges (Wavelets should win)
4. White noise (All fail equally)
5. Real biosignals (TBD)
"""
import numpy as np
from scipy.fft import fft, dct
from scipy.signal import morlet2
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

PHI = (1 + np.sqrt(5)) / 2

def sparsity(x, threshold=0.01):
    """Fraction of coefficients > threshold of max."""
    return 1.0 - np.count_nonzero(np.abs(x) > threshold * np.max(np.abs(x))) / len(x)

def _check_signal_class(signal, name):
    """Compare RFT/FFT/DCT sparsity on a signal."""
    N = len(signal)
    
    # Transforms
    rft = CanonicalTrueRFT(N)
    X_rft = rft.forward_transform(signal.astype(complex))
    X_fft = fft(signal, norm='ortho')
    X_dct = dct(signal, norm='ortho')
    
    # Sparsity (fraction of near-zero coefficients)
    s_rft = sparsity(X_rft)
    s_fft = sparsity(X_fft)
    s_dct = sparsity(X_dct)
    
    # Winner
    best = max(s_rft, s_fft, s_dct)
    winner = 'RFT' if s_rft == best else 'FFT' if s_fft == best else 'DCT'
    
    print(f"{name:30s} | RFT: {s_rft:.3f} | FFT: {s_fft:.3f} | DCT: {s_dct:.3f} | Winner: {winner}")
    return winner

def run_benchmark():
    """Run full signal niche benchmark."""
    N = 256
    np.random.seed(42)
    
    print("=" * 80)
    print("RFT SIGNAL NICHE ANALYSIS")
    print("=" * 80)
    print(f"{'Signal Type':<30s} | Sparsity Scores (higher=better)")
    print("-" * 80)
    
    wins = {'RFT': 0, 'FFT': 0, 'DCT': 0}
    
    # Test 1: Golden-ratio signals (RFT should win)
    t = np.arange(N) / N
    for f in [PHI, PHI**2, PHI**3]:
        signal = np.sin(2*np.pi*f*t)
        winner = _check_signal_class(signal, f"Golden freq {f:.3f}")
        wins[winner] += 1
    
    # Test 2: Integer frequencies (FFT should win)
    for k in [5, 10, 20]:
        signal = np.sin(2*np.pi*k*t)
        winner = _check_signal_class(signal, f"Integer freq {k}")
        wins[winner] += 1
    
    # Test 3: DC + few harmonics (DCT might win)
    signal = 1.0 + 0.5*np.cos(2*np.pi*t) + 0.3*np.cos(4*np.pi*t)
    winner = _check_signal_class(signal, "DC + low harmonics")
    wins[winner] += 1
    
    # Test 4: White noise (all should fail)
    signal = np.random.randn(N)
    winner = _check_signal_class(signal, "White noise")
    wins[winner] += 1
    
    # Test 5: Quasi-periodic chirp
    signal = np.sin(2*np.pi*t*PHI*t)
    winner = _check_signal_class(signal, "φ-chirp")
    wins[winner] += 1
    
    # Test 6: Fibonacci sequence (quasi-periodic)
    fib = [1, 1]
    for _ in range(N-2):
        fib.append(fib[-1] + fib[-2])
    signal = np.array(fib[:N], dtype=float)
    signal = (signal - np.mean(signal)) / np.std(signal)  # Normalize
    winner = _check_signal_class(signal, "Fibonacci sequence")
    wins[winner] += 1
    
    print("-" * 80)
    print(f"\nFinal Score: RFT={wins['RFT']}, FFT={wins['FFT']}, DCT={wins['DCT']}")
    print()
    print("CONCLUSION:")
    if wins['RFT'] > wins['FFT'] and wins['RFT'] > wins['DCT']:
        print("  ✓ RFT wins overall - but only on golden-ratio structured signals")
    elif wins['RFT'] > 0:
        print(f"  → RFT wins on {wins['RFT']}/{sum(wins.values())} signals (domain-specific)")
    else:
        print("  ✗ RFT does not show sparsity advantage")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
