#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Visualization Suite - Matplotlib/Seaborn
Generates comprehensive plots for RFT analysis and comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fftpack
import time
from pathlib import Path

from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_unitary_error_canonical as rft_unitary_error, rft_matrix_canonical as rft_matrix, PHI
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create output directory
OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_unitarity_error():
    """Plot unitarity error vs transform size"""
    print("Generating unitarity error plot...")
    
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    rft_errors = []
    fft_errors = []
    
    for n in sizes:
        # RFT error
        rft_err = rft_unitary_error(n, trials=20)
        rft_errors.append(rft_err)
        
        # FFT error
        rng = np.random.default_rng(42)
        fft_err_list = []
        for _ in range(20):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            X = np.fft.fft(x, norm='ortho')
            x_rec = np.fft.ifft(X, norm='ortho')
            err = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
            fft_err_list.append(err)
        fft_errors.append(np.mean(fft_err_list))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(sizes, rft_errors, 'o-', label='RFT (Φ-RFT)', linewidth=2, markersize=8)
    ax.semilogy(sizes, fft_errors, 's-', label='FFT', linewidth=2, markersize=8)
    ax.axhline(y=1e-15, color='r', linestyle='--', alpha=0.5, label='Machine precision')
    
    ax.set_xlabel('Transform Size (N)', fontweight='bold')
    ax.set_ylabel('Round-trip Error (relative)', fontweight='bold')
    ax.set_title('Unitarity Error: RFT vs FFT', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "unitarity_error.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "unitarity_error.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/unitarity_error.png")
    plt.close()

def plot_performance_benchmark():
    """Plot performance comparison across multiple transforms"""
    print("Generating performance benchmark plot...")
    
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    n_iterations = 50
    
    rft_times = []
    fft_times = []
    dct_times = []
    
    for n in sizes:
        print(f"  Benchmarking N={n}...")
        rng = np.random.default_rng(42)
        x_complex = rng.normal(size=n) + 1j * rng.normal(size=n)
        x_real = rng.normal(size=n)
        
        # RFT
        _ = rft_forward(x_complex)  # Warm up
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = rft_forward(x_complex)
        rft_times.append((time.perf_counter() - start) * 1000 / n_iterations)
        
        # FFT
        _ = np.fft.fft(x_complex, norm='ortho')
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = np.fft.fft(x_complex, norm='ortho')
        fft_times.append((time.perf_counter() - start) * 1000 / n_iterations)
        
        # DCT
        _ = fftpack.dct(x_real, type=2, norm='ortho')
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = fftpack.dct(x_real, type=2, norm='ortho')
        dct_times.append((time.perf_counter() - start) * 1000 / n_iterations)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute times
    ax1.loglog(sizes, rft_times, 'o-', label='RFT (Φ-RFT)', linewidth=2, markersize=8)
    ax1.loglog(sizes, fft_times, 's-', label='FFT', linewidth=2, markersize=8)
    ax1.loglog(sizes, dct_times, '^-', label='DCT', linewidth=2, markersize=8)
    ax1.set_xlabel('Transform Size (N)', fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontweight='bold')
    ax1.set_title('Transform Performance Comparison', fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Relative speedup (FFT as baseline)
    rft_ratio = np.array(rft_times) / np.array(fft_times)
    dct_ratio = np.array(dct_times) / np.array(fft_times)
    
    ax2.semilogx(sizes, rft_ratio, 'o-', label='RFT / FFT', linewidth=2, markersize=8)
    ax2.semilogx(sizes, dct_ratio, '^-', label='DCT / FFT', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='FFT baseline')
    ax2.set_xlabel('Transform Size (N)', fontweight='bold')
    ax2.set_ylabel('Relative Time (vs FFT)', fontweight='bold')
    ax2.set_title('Computational Overhead', fontweight='bold')
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_benchmark.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "performance_benchmark.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/performance_benchmark.png")
    plt.close()

def plot_spectrum_comparison():
    """Compare spectral characteristics of different signals"""
    print("Generating spectrum comparison plots...")
    
    n = 256
    t = np.arange(n)
    
    # Different signal types
    signals = {
        'Pure Sine (f=5)': np.sin(2 * np.pi * 5 * t / n),
        'Multi-tone': (np.sin(2 * np.pi * 5 * t / n) + 
                       0.5 * np.sin(2 * np.pi * 12 * t / n) + 
                       0.3 * np.cos(2 * np.pi * 20 * t / n)),
        'Chirp': np.sin(2 * np.pi * t**2 / (2 * n**2)),
        'Step Function': np.concatenate([np.ones(n//2), -np.ones(n//2)]),
    }
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    for idx, (sig_name, x) in enumerate(signals.items()):
        # Time domain
        axes[idx, 0].plot(t, x, 'b-', linewidth=1.5)
        axes[idx, 0].set_title(f'{sig_name}', fontweight='bold')
        axes[idx, 0].set_xlabel('Sample')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # RFT spectrum
        X_rft = rft_forward(x)
        axes[idx, 1].stem(np.arange(n), np.abs(X_rft), basefmt=' ', linefmt='r-', markerfmt='ro')
        axes[idx, 1].set_title('RFT Spectrum', fontweight='bold')
        axes[idx, 1].set_xlabel('Frequency Bin')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # FFT spectrum
        X_fft = np.fft.fft(x, norm='ortho')
        axes[idx, 2].stem(np.arange(n), np.abs(X_fft), basefmt=' ', linefmt='b-', markerfmt='bs')
        axes[idx, 2].set_title('FFT Spectrum', fontweight='bold')
        axes[idx, 2].set_xlabel('Frequency Bin')
        axes[idx, 2].set_ylabel('Magnitude')
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectrum_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "spectrum_comparison.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/spectrum_comparison.png")
    plt.close()

def plot_compression_efficiency():
    """Plot compression efficiency for different transforms"""
    print("Generating compression efficiency plot...")
    
    n = 512
    
    # Test signals
    signals = {
        'Smooth Sine': np.sin(2 * np.pi * 5 * np.arange(n) / n),
        'Polynomial': (np.arange(n) / n) ** 2,
        'Exponential': np.exp(-np.arange(n) / 100),
        'Noisy Sine': np.sin(2 * np.pi * 5 * np.arange(n) / n) + 0.2 * np.random.randn(n),
        'Step': np.concatenate([np.ones(n//2), -np.ones(n//2)]),
        'Random': np.random.randn(n),
    }
    
    transforms = {
        'RFT': lambda x: rft_forward(x),
        'FFT': lambda x: np.fft.fft(x, norm='ortho'),
        'DCT': lambda x: fftpack.dct(x, type=2, norm='ortho'),
        'DST': lambda x: fftpack.dst(x, type=2, norm='ortho'),
    }
    
    # Compute compression ratios (coefficients needed for 99% energy)
    results = {transform: [] for transform in transforms}
    
    for sig_name, x in signals.items():
        for trans_name, transform in transforms.items():
            X = transform(x)
            X_abs = np.abs(X)
            
            # Sort by magnitude
            sorted_coeffs = np.sort(X_abs)[::-1]
            total_energy = np.sum(sorted_coeffs ** 2)
            cumsum = np.cumsum(sorted_coeffs ** 2)
            
            # Find number of coefficients for 99% energy
            n99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
            compression_ratio = n / n99
            
            results[trans_name].append(compression_ratio)
    
    # Create grouped bar plot
    x_pos = np.arange(len(signals))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for idx, (trans_name, ratios) in enumerate(results.items()):
        offset = (idx - 1.5) * width
        ax.bar(x_pos + offset, ratios, width, label=trans_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Signal Type', fontweight='bold')
    ax.set_ylabel('Compression Ratio (N / N₉₉)', fontweight='bold')
    ax.set_title('Compression Efficiency: Coefficients for 99% Energy', fontweight='bold', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(signals.keys(), rotation=15, ha='right')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "compression_efficiency.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "compression_efficiency.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/compression_efficiency.png")
    plt.close()

def plot_phase_structure():
    """Visualize the unique phase structure of RFT"""
    print("Generating phase structure visualization...")
    
    sizes = [32, 64, 128]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, n in enumerate(sizes):
        k = np.arange(n)
        
        # RFT phase (golden ratio based)
        theta_rft = 2.0 * np.pi * np.modf(k / PHI)[0]
        
        # FFT phase (uniform)
        theta_fft = 2.0 * np.pi * k / n
        
        # Plot phase vs index
        axes[0, idx].plot(k, theta_rft, 'r-', linewidth=2, label='RFT (Φ-based)')
        axes[0, idx].plot(k, theta_fft, 'b--', linewidth=2, label='FFT (uniform)')
        axes[0, idx].set_xlabel('Index k', fontweight='bold')
        axes[0, idx].set_ylabel('Phase (radians)', fontweight='bold')
        axes[0, idx].set_title(f'Phase Structure (N={n})', fontweight='bold')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        
        # Plot phase distribution (histogram)
        axes[1, idx].hist(theta_rft, bins=30, alpha=0.7, color='red', label='RFT', density=True)
        axes[1, idx].hist(theta_fft, bins=30, alpha=0.7, color='blue', label='FFT', density=True)
        axes[1, idx].set_xlabel('Phase (radians)', fontweight='bold')
        axes[1, idx].set_ylabel('Density', fontweight='bold')
        axes[1, idx].set_title(f'Phase Distribution (N={n})', fontweight='bold')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase_structure.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "phase_structure.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/phase_structure.png")
    plt.close()

def plot_matrix_structure():
    """Visualize RFT matrix structure"""
    print("Generating matrix structure visualization...")
    
    n = 64
    
    # Generate matrices
    Psi_rft = rft_matrix(n)
    F_fft = np.fft.fft(np.eye(n), norm='ortho')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RFT magnitude
    im1 = axes[0, 0].imshow(np.abs(Psi_rft), cmap='hot', aspect='auto')
    axes[0, 0].set_title('RFT Matrix |Ψ| (Magnitude)', fontweight='bold')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RFT phase
    im2 = axes[0, 1].imshow(np.angle(Psi_rft), cmap='twilight', aspect='auto')
    axes[0, 1].set_title('RFT Matrix ∠Ψ (Phase)', fontweight='bold')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # RFT real part
    im3 = axes[0, 2].imshow(np.real(Psi_rft), cmap='RdBu', aspect='auto')
    axes[0, 2].set_title('RFT Matrix Re(Ψ)', fontweight='bold')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # FFT magnitude
    im4 = axes[1, 0].imshow(np.abs(F_fft), cmap='hot', aspect='auto')
    axes[1, 0].set_title('FFT Matrix |F| (Magnitude)', fontweight='bold')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # FFT phase
    im5 = axes[1, 1].imshow(np.angle(F_fft), cmap='twilight', aspect='auto')
    axes[1, 1].set_title('FFT Matrix ∠F (Phase)', fontweight='bold')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # FFT real part
    im6 = axes[1, 2].imshow(np.real(F_fft), cmap='RdBu', aspect='auto')
    axes[1, 2].set_title('FFT Matrix Re(F)', fontweight='bold')
    axes[1, 2].set_xlabel('Column')
    axes[1, 2].set_ylabel('Row')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matrix_structure.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "matrix_structure.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/matrix_structure.png")
    plt.close()

def plot_energy_compaction():
    """Plot energy compaction curves"""
    print("Generating energy compaction plot...")
    
    n = 512
    t = np.arange(n)
    
    # Test signal: damped multi-frequency
    x = (np.sin(2 * np.pi * 5 * t / n) * np.exp(-t / 200) +
         0.5 * np.sin(2 * np.pi * 15 * t / n) * np.exp(-t / 300))
    
    transforms = {
        'RFT': rft_forward(x),
        'FFT': np.fft.fft(x, norm='ortho'),
        'DCT': fftpack.dct(x, type=2, norm='ortho'),
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, X in transforms.items():
        X_abs = np.abs(X)
        sorted_coeffs = np.sort(X_abs)[::-1]
        
        # Cumulative energy
        cumulative_energy = np.cumsum(sorted_coeffs ** 2)
        cumulative_energy /= cumulative_energy[-1]  # Normalize
        
        ax.plot(np.arange(1, n+1), cumulative_energy, linewidth=2, label=name, marker='o', 
                markevery=n//20, markersize=6)
    
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% energy')
    ax.set_xlabel('Number of Coefficients', fontweight='bold')
    ax.set_ylabel('Cumulative Energy Fraction', fontweight='bold')
    ax.set_title('Energy Compaction: Damped Multi-tone Signal', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, shadow=True, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n//2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "energy_compaction.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "energy_compaction.pdf", bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/energy_compaction.png")
    plt.close()

def generate_latex_data():
    """Generate data files for LaTeX/TikZ plotting"""
    print("Generating LaTeX data files...")
    
    latex_dir = OUTPUT_DIR / "latex_data"
    latex_dir.mkdir(exist_ok=True)
    
    # 1. Unitarity error data
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    with open(latex_dir / "unitarity_data.dat", 'w') as f:
        f.write("# N RFT_error FFT_error\n")
        for n in sizes:
            rft_err = rft_unitary_error(n, trials=10)
            # FFT error (simplified)
            rng = np.random.default_rng(42)
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            X = np.fft.fft(x, norm='ortho')
            x_rec = np.fft.ifft(X, norm='ortho')
            fft_err = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
            f.write(f"{n} {rft_err:.6e} {fft_err:.6e}\n")
    
    # 2. Performance data
    sizes = [64, 128, 256, 512, 1024]
    with open(latex_dir / "performance_data.dat", 'w') as f:
        f.write("# N RFT_time FFT_time DCT_time\n")
        for n in sizes:
            rng = np.random.default_rng(42)
            x_complex = rng.normal(size=n) + 1j * rng.normal(size=n)
            x_real = rng.normal(size=n)
            
            start = time.perf_counter()
            for _ in range(50):
                _ = rft_forward(x_complex)
            t_rft = (time.perf_counter() - start) / 50
            
            start = time.perf_counter()
            for _ in range(50):
                _ = np.fft.fft(x_complex, norm='ortho')
            t_fft = (time.perf_counter() - start) / 50
            
            start = time.perf_counter()
            for _ in range(50):
                _ = fftpack.dct(x_real, type=2, norm='ortho')
            t_dct = (time.perf_counter() - start) / 50
            
            f.write(f"{n} {t_rft:.6e} {t_fft:.6e} {t_dct:.6e}\n")
    
    print(f"  Saved LaTeX data to: {latex_dir}/")

def main():
    print("\n" + "="*70)
    print(" RFT Visualization Suite - Generating Plots")
    print("="*70 + "\n")
    
    plot_unitarity_error()
    plot_performance_benchmark()
    plot_spectrum_comparison()
    plot_compression_efficiency()
    plot_phase_structure()
    plot_matrix_structure()
    plot_energy_compaction()
    generate_latex_data()
    
    print("\n" + "="*70)
    print(f" All figures saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
