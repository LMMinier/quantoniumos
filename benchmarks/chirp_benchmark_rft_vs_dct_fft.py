#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Chirp Signal Benchmark: RFT vs DCT vs FFT
==========================================

Test hypothesis: Φ-RFT achieves superior sparsity on chirp signals
compared to DCT/FFT due to golden-ratio phase modulation.

Chirps are common in:
- Audio: FM synthesis, Doppler shifts
- Radar: frequency-modulated continuous wave (FMCW)
- Medical: ultrasound imaging
- Seismic: exploration signals
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import time
from scipy.fftpack import dct, idct
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
import json
from typing import Dict, List
import matplotlib.pyplot as plt

def generate_chirp_signals(n: int = 4096) -> Dict[str, np.ndarray]:
    """
    Generate various chirp signals where DCT/FFT are known to be sparse-inefficient
    """
    t = np.linspace(0, 1, n)
    signals = {}
    
    # Linear chirp: f(t) = f0 + (f1-f0)*t
    signals['linear_chirp'] = np.sin(2*np.pi * (100*t + 200*t**2))
    
    # Quadratic chirp: f(t) = f0 + a*t^2 + b*t^3
    signals['quadratic_chirp'] = np.sin(2*np.pi * (50*t + 100*t**2 + 150*t**3))
    
    # Exponential chirp: f(t) = f0 * exp(k*t)
    signals['exponential_chirp'] = np.sin(2*np.pi * 50 * np.exp(2*t))
    
    # Golden ratio chirp: f(t) modulated by phi
    phi = (1 + np.sqrt(5)) / 2
    signals['golden_ratio_chirp'] = np.sin(2*np.pi * 100 * t * phi**t)
    
    # Hyperbolic chirp: f(t) = 1/(a + b*t)
    signals['hyperbolic_chirp'] = np.sin(2*np.pi * 500 / (1 + 5*t))
    
    # Pulse-modulated chirp (radar-like)
    envelope = np.exp(-((t - 0.5)**2) / 0.05)
    signals['radar_chirp'] = envelope * np.sin(2*np.pi * (200*t + 300*t**2))
    
    return signals


def benchmark_transform(signal: np.ndarray, method: str = 'rft', 
                       trials: int = 100, keep_fraction: float = 0.05) -> Dict:
    """
    Benchmark a single transform method
    
    Metrics:
    1. Encoding time (μs)
    2. Sparsity (coefficient energy concentration)
    3. Reconstruction error (PSNR)
    4. Top-k compression effectiveness
    """
    n = len(signal)
    k = int(keep_fraction * n)
    
    # Convert to complex if needed
    if method == 'rft':
        signal_complex = signal.astype(np.complex128)
    else:
        signal_complex = signal
    
    # Timing benchmark
    start = time.perf_counter()
    for _ in range(trials):
        if method == 'rft':
            coeffs = rft_forward(signal_complex)
        elif method == 'dct':
            coeffs = dct(signal, norm='ortho')
        elif method == 'fft':
            coeffs = np.fft.fft(signal) / np.sqrt(n)
    elapsed = (time.perf_counter() - start) / trials
    
    # Get coefficients (recompute once for analysis)
    if method == 'rft':
        coeffs = rft_forward(signal_complex)
    elif method == 'dct':
        coeffs = dct(signal, norm='ortho')
    elif method == 'fft':
        coeffs = np.fft.fft(signal) / np.sqrt(n)
    
    # Sparsity analysis: top-k coefficient energy
    coeff_magnitudes = np.abs(coeffs)
    total_energy = np.sum(coeff_magnitudes**2)
    
    # Keep top k coefficients
    idx = np.argsort(coeff_magnitudes)[-k:]
    sparse = np.zeros_like(coeffs)
    sparse[idx] = coeffs[idx]
    
    top_k_energy = np.sum(np.abs(sparse)**2)
    energy_concentration = top_k_energy / max(total_energy, 1e-16)
    
    # Reconstruction
    if method == 'rft':
        recon = rft_inverse(sparse).real
    elif method == 'dct':
        recon = idct(sparse, norm='ortho')
    elif method == 'fft':
        recon = np.fft.ifft(sparse * np.sqrt(n)).real
    
    # Error metrics
    mse = np.mean((signal - recon)**2)
    signal_power = np.mean(signal**2)
    psnr = 10 * np.log10(signal_power / max(mse, 1e-16))
    
    # Relative L2 error
    relative_error = np.linalg.norm(signal - recon) / np.linalg.norm(signal)
    
    return {
        'method': method,
        'time_us': elapsed * 1e6,
        'psnr_db': float(psnr),
        'relative_error': float(relative_error),
        'energy_concentration': float(energy_concentration),
        'coeffs_kept': k,
        'keep_fraction': keep_fraction
    }


def run_chirp_benchmark_suite(n: int = 4096, keep_fractions: List[float] = [0.01, 0.05, 0.10, 0.20]):
    """Run comprehensive chirp benchmark"""
    print("=" * 80)
    print("CHIRP SIGNAL BENCHMARK: RFT vs DCT vs FFT")
    print("=" * 80)
    print(f"Signal length: {n}")
    print(f"Keep fractions: {keep_fractions}")
    print()
    
    signals = generate_chirp_signals(n)
    all_results = {}
    
    for signal_name, signal in signals.items():
        print(f"\n{'='*80}")
        print(f"Testing: {signal_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        signal_results = {}
        
        for keep_frac in keep_fractions:
            print(f"\n  Keep fraction: {keep_frac:.1%}")
            print(f"  {'Method':<8} {'Time (μs)':<12} {'PSNR (dB)':<12} {'Rel Error':<12} {'Energy %':<12}")
            print(f"  {'-'*70}")
            
            frac_results = {}
            for method in ['rft', 'dct', 'fft']:
                result = benchmark_transform(signal, method=method, keep_fraction=keep_frac)
                frac_results[method] = result
                
                print(f"  {method.upper():<8} "
                      f"{result['time_us']:<12.2f} "
                      f"{result['psnr_db']:<12.2f} "
                      f"{result['relative_error']:<12.4f} "
                      f"{result['energy_concentration']*100:<12.1f}")
            
            signal_results[f'keep_{keep_frac}'] = frac_results
        
        all_results[signal_name] = signal_results
    
    return all_results


def analyze_results(results: Dict) -> Dict:
    """Analyze which method wins on each signal type"""
    print("\n" + "="*80)
    print("ANALYSIS: RFT SUPERIORITY ASSESSMENT")
    print("="*80)
    
    analysis = {}
    
    for signal_name, signal_results in results.items():
        print(f"\n{signal_name.upper().replace('_', ' ')}:")
        
        signal_analysis = {}
        
        for keep_key, methods in signal_results.items():
            keep_frac = float(keep_key.split('_')[1])
            
            # Compare energy concentration (higher is better)
            rft_energy = methods['rft']['energy_concentration']
            dct_energy = methods['dct']['energy_concentration']
            fft_energy = methods['fft']['energy_concentration']
            
            best_method = max(methods.items(), key=lambda x: x[1]['energy_concentration'])[0]
            
            # Calculate advantage
            if best_method == 'rft':
                vs_dct = (rft_energy - dct_energy) / dct_energy * 100
                vs_fft = (rft_energy - fft_energy) / fft_energy * 100
                advantage_str = f"RFT wins: +{vs_dct:.1f}% vs DCT, +{vs_fft:.1f}% vs FFT"
            else:
                advantage_str = f"{best_method.upper()} wins (RFT not superior)"
            
            print(f"  Keep {keep_frac:.1%}: {advantage_str}")
            
            signal_analysis[keep_key] = {
                'winner': best_method,
                'rft_vs_dct_pct': float((rft_energy - dct_energy) / dct_energy * 100),
                'rft_vs_fft_pct': float((rft_energy - fft_energy) / fft_energy * 100),
                'best_energy': float(methods[best_method]['energy_concentration'])
            }
        
        analysis[signal_name] = signal_analysis
    
    return analysis


def generate_visualizations(results: Dict, output_dir: str = "tests/benchmarks/chirp_results"):
    """Generate comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # For each signal type, plot energy concentration vs keep fraction
    for signal_name, signal_results in results.items():
        keep_fracs = []
        rft_energies = []
        dct_energies = []
        fft_energies = []
        
        for keep_key in sorted(signal_results.keys()):
            keep_frac = float(keep_key.split('_')[1])
            keep_fracs.append(keep_frac * 100)
            
            rft_energies.append(signal_results[keep_key]['rft']['energy_concentration'] * 100)
            dct_energies.append(signal_results[keep_key]['dct']['energy_concentration'] * 100)
            fft_energies.append(signal_results[keep_key]['fft']['energy_concentration'] * 100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(keep_fracs, rft_energies, 'o-', label='RFT', linewidth=2)
        plt.plot(keep_fracs, dct_energies, 's-', label='DCT', linewidth=2)
        plt.plot(keep_fracs, fft_energies, '^-', label='FFT', linewidth=2)
        
        plt.xlabel('Coefficients Kept (%)', fontsize=12)
        plt.ylabel('Energy Captured (%)', fontsize=12)
        plt.title(f'Sparsity Comparison: {signal_name.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_dir}/{signal_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")


def main():
    """Run full benchmark suite"""
    results = run_chirp_benchmark_suite(n=4096, keep_fractions=[0.01, 0.05, 0.10, 0.20])
    analysis = analyze_results(results)
    
    # Save results
    output_data = {
        'results': results,
        'analysis': analysis,
        'config': {
            'n': 4096,
            'keep_fractions': [0.01, 0.05, 0.10, 0.20],
            'methods': ['rft', 'dct', 'fft']
        }
    }
    
    output_path = 'chirp_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Generate visualizations
    try:
        generate_visualizations(results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Summary verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    rft_wins = 0
    total_tests = 0
    
    for signal_name, signal_analysis in analysis.items():
        for keep_key, keep_analysis in signal_analysis.items():
            total_tests += 1
            if keep_analysis['winner'] == 'rft':
                rft_wins += 1
    
    win_rate = rft_wins / total_tests * 100
    
    print(f"\nRFT wins: {rft_wins}/{total_tests} tests ({win_rate:.1f}%)")
    
    if win_rate >= 80:
        print("✅ STRONG EVIDENCE: RFT significantly outperforms DCT/FFT on chirp signals")
    elif win_rate >= 60:
        print("✅ GOOD EVIDENCE: RFT shows superior sparsity on most chirp types")
    elif win_rate >= 40:
        print("⚠️  MODERATE: RFT competitive but not dominant")
    else:
        print("❌ WEAK: RFT does not show clear advantage over DCT/FFT")
    
    print("\nNext steps:")
    print("1. If RFT wins ≥60%: Use these results in paper/pitch")
    print("2. Test on real-world datasets (audio, radar, medical)")
    print("3. Optimize RFT implementation for speed parity")


if __name__ == "__main__":
    main()
