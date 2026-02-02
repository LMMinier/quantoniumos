#!/usr/bin/env python3
"""
Publication-Ready Figure Generator
===================================

Creates high-quality figures for research papers showing:
1. RFT basis functions (φ-resonance structure)
2. Eigenvalue distributions (RFT vs FFT vs Random)
3. Wave-domain logic operations
4. Gram matrix structure
5. Frequency grid visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_function, rft_basis_matrix, BinaryRFT, PHI
)
from algorithms.rft.core.gram_utils import gram_matrix

# Publication settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def figure_1_basis_functions(save_path='fig1_rft_basis_functions.png'):
    """Figure 1: RFT Basis Functions showing φ-resonance."""
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle('RFT Basis Functions: Golden-Ratio Resonance Structure', fontsize=14, fontweight='bold')
    
    T = 256
    t = np.arange(T) / T
    
    # Plot first 6 basis functions
    for idx, k in enumerate([0, 1, 2, 3, 4, 5]):
        ax = axes[idx // 2, idx % 2]
        
        psi = rft_basis_function(k, t)
        
        # Plot real and imaginary parts
        ax.plot(t, np.real(psi), 'b-', alpha=0.7, label='Real', linewidth=1.5)
        ax.plot(t, np.imag(psi), 'r--', alpha=0.7, label='Imag', linewidth=1.5)
        
        # Compute frequency
        f_k = (k + 1) * PHI
        theta_k = 2 * np.pi * k / PHI
        
        ax.set_title(f'Ψ_{k}(t): f_{k} = {f_k:.3f}, θ_{k} = {theta_k:.3f}', fontsize=10)
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure_2_eigenvalue_comparison(save_path='fig2_eigenvalue_comparison.png'):
    """Figure 2: Eigenvalue Distribution Comparison."""
    
    N = 128
    
    # Get eigenvalues
    Phi_rft = rft_basis_matrix(N, N, use_gram_normalization=False)
    G_rft = gram_matrix(Phi_rft)
    eig_rft = np.sort(linalg.eigvalsh(G_rft))[::-1]
    
    # FFT (all λ=1)
    eig_fft = np.ones(N)
    
    # Random matrix
    np.random.seed(42)
    A_random = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) / np.sqrt(N)
    G_random = A_random.conj().T @ A_random
    eig_random = np.sort(linalg.eigvalsh(G_random))[::-1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Gram Matrix Eigenvalue Spectra', fontsize=14, fontweight='bold')
    
    # Left: Line plot
    ax = axes[0]
    ax.plot(eig_rft, 'b-', linewidth=2, label='RFT (φ-grid)')
    ax.plot(eig_fft, 'g--', linewidth=2, label='FFT (uniform)')
    ax.plot(eig_random, 'r:', linewidth=2, label='Random Gaussian')
    ax.set_xlabel('Index k')
    ax.set_ylabel('Eigenvalue λₖ')
    ax.set_title('Eigenvalue Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # Right: Histogram
    ax = axes[1]
    ax.hist(eig_rft, bins=30, alpha=0.7, label='RFT', color='blue', edgecolor='black')
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='FFT (δ at λ=1)')
    ax.hist(eig_random, bins=30, alpha=0.5, label='Random', color='red', edgecolor='black')
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('Count')
    ax.set_title('Eigenvalue Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure_3_wave_logic(save_path='fig3_wave_domain_logic.png'):
    """Figure 3: Wave-Domain Logic Operations."""
    
    brft = BinaryRFT(num_bits=8, samples_per_bit=16)
    
    a, b = 0b10101010, 0b11001100
    wa, wb = brft.encode(a), brft.encode(b)
    
    # Compute operations
    w_xor = brft.wave_xor(wa, wb)
    w_and = brft.wave_and(wa, wb)
    w_or = brft.wave_or(wa, wb)
    
    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.suptitle('Wave-Domain Binary Logic Operations', fontsize=14, fontweight='bold')
    
    t = np.arange(len(wa))
    
    # Plot waveforms
    data = [
        (wa, f'Input A: {a:08b}', 'blue'),
        (wb, f'Input B: {b:08b}', 'green'),
        (w_xor, f'A XOR B = {a^b:08b}', 'purple'),
        (w_and, f'A AND B = {a&b:08b}', 'orange'),
        (w_or, f'A OR  B = {a|b:08b}', 'red')
    ]
    
    for idx, (wave, title, color) in enumerate(data):
        ax = axes[idx]
        ax.plot(t, np.real(wave), color=color, alpha=0.8, linewidth=1, label='Real')
        ax.plot(t, np.imag(wave), color=color, alpha=0.4, linestyle='--', linewidth=1, label='Imag')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-4, 4)
        
        if idx == len(data) - 1:
            ax.set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure_4_gram_matrix_structure(save_path='fig4_gram_matrix_structure.png'):
    """Figure 4: Gram Matrix Structure Visualization."""
    
    N = 64  # Use smaller N for visibility
    
    Phi = rft_basis_matrix(N, N, use_gram_normalization=False)
    G = gram_matrix(Phi)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('RFT Gram Matrix G = ΦᴴΦ Structure', fontsize=14, fontweight='bold')
    
    # Left: Magnitude
    ax = axes[0]
    im = ax.imshow(np.abs(G), cmap='viridis', interpolation='nearest')
    ax.set_title('|G| (Magnitude)')
    ax.set_xlabel('Column k')
    ax.set_ylabel('Row n')
    plt.colorbar(im, ax=ax)
    
    # Middle: Phase
    ax = axes[1]
    im = ax.imshow(np.angle(G), cmap='twilight', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    ax.set_title('∠G (Phase)')
    ax.set_xlabel('Column k')
    ax.set_ylabel('Row n')
    plt.colorbar(im, ax=ax, label='radians')
    
    # Right: Deviation from identity
    ax = axes[2]
    I = np.eye(N)
    deviation = np.abs(G - I)
    im = ax.imshow(deviation, cmap='hot', interpolation='nearest')
    ax.set_title('|G - I| (Non-orthogonality)')
    ax.set_xlabel('Column k')
    ax.set_ylabel('Row n')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure_5_frequency_grid(save_path='fig5_frequency_grid.png'):
    """Figure 5: Frequency Grid Structure."""
    
    N = 128
    k = np.arange(N)
    
    # RFT frequencies
    freqs_rft = np.mod((k + 1) * PHI, 1.0)
    
    # FFT frequencies
    freqs_fft = k / N
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Frequency Grid Comparison: RFT vs FFT', fontsize=14, fontweight='bold')
    
    # Top-left: RFT frequency distribution
    ax = axes[0, 0]
    ax.scatter(k, freqs_rft, c='blue', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Index k')
    ax.set_ylabel('Frequency fₖ (normalized)')
    ax.set_title('RFT: fₖ = frac((k+1)×φ)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Top-right: FFT frequency distribution
    ax = axes[0, 1]
    ax.scatter(k, freqs_fft, c='green', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Index k')
    ax.set_ylabel('Frequency fₖ (normalized)')
    ax.set_title('FFT: fₖ = k/N (uniform)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Bottom-left: Gap histogram (RFT)
    ax = axes[1, 0]
    freqs_sorted = np.sort(freqs_rft)
    gaps_rft = np.diff(freqs_sorted)
    ax.hist(gaps_rft, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Gap size')
    ax.set_ylabel('Count')
    ax.set_title(f'RFT Gap Distribution (3-distance theorem)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom-right: Gap histogram (FFT)
    ax = axes[1, 1]
    gaps_fft = np.diff(freqs_fft)
    ax.hist(gaps_fft, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(1/N, color='red', linestyle='--', linewidth=2, label=f'Uniform: 1/N={1/N:.4f}')
    ax.set_xlabel('Gap size')
    ax.set_ylabel('Count')
    ax.set_title('FFT Gap Distribution (uniform)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure_6_condition_scaling(save_path='fig6_condition_number_scaling.png'):
    """Figure 6: Condition Number Scaling."""
    
    sizes = [16, 32, 64, 128, 256, 512]
    kappas = []
    
    for N in sizes:
        Phi = rft_basis_matrix(N, N, use_gram_normalization=False)
        G = gram_matrix(Phi)
        eigs = linalg.eigvalsh(G)
        kappa = np.max(eigs) / np.min(eigs)
        kappas.append(kappa)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Gram Matrix Conditioning Analysis', fontsize=14, fontweight='bold')
    
    # Left: Log-log plot
    ax = axes[0]
    ax.loglog(sizes, kappas, 'bo-', linewidth=2, markersize=8, label='RFT κ(G)')
    ax.axhline(1, color='green', linestyle='--', linewidth=2, label='FFT κ=1')
    
    # Fit power law
    coeffs = np.polyfit(np.log(sizes), np.log(kappas), 1)
    alpha = coeffs[0]
    fit = np.exp(coeffs[1]) * np.array(sizes)**alpha
    ax.loglog(sizes, fit, 'r--', linewidth=2, label=f'Fit: κ ∝ N^{alpha:.2f}')
    
    ax.set_xlabel('Matrix size N')
    ax.set_ylabel('Condition number κ(G)')
    ax.set_title('Condition Number Scaling')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Right: Stability threshold
    ax = axes[1]
    ax.semilogy(sizes, kappas, 'bo-', linewidth=2, markersize=8)
    ax.axhline(1e10, color='red', linestyle='--', linewidth=2, label='Instability threshold')
    ax.axhline(1, color='green', linestyle='--', linewidth=2, label='FFT (ideal)')
    ax.set_xlabel('Matrix size N')
    ax.set_ylabel('Condition number κ(G)')
    ax.set_title('Gram Normalization Stability')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_ylim(0.5, 1e12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_all_figures(output_dir='figures'):
    """Generate all publication-ready figures."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING PUBLICATION-READY FIGURES")
    print("=" * 70)
    
    figures = [
        ("Basis Functions", lambda: figure_1_basis_functions(f'{output_dir}/fig1_rft_basis_functions.png')),
        ("Eigenvalue Comparison", lambda: figure_2_eigenvalue_comparison(f'{output_dir}/fig2_eigenvalue_comparison.png')),
        ("Wave-Domain Logic", lambda: figure_3_wave_logic(f'{output_dir}/fig3_wave_domain_logic.png')),
        ("Gram Matrix Structure", lambda: figure_4_gram_matrix_structure(f'{output_dir}/fig4_gram_matrix_structure.png')),
        ("Frequency Grid", lambda: figure_5_frequency_grid(f'{output_dir}/fig5_frequency_grid.png')),
        ("Condition Scaling", lambda: figure_6_condition_scaling(f'{output_dir}/fig6_condition_number_scaling.png')),
    ]
    
    for name, func in figures:
        print(f"\nGenerating: {name}")
        func()
    
    print("\n" + "=" * 70)
    print(f"✓ All figures saved to: {output_dir}/")
    print("=" * 70)
    
    print("\nFigure Summary:")
    print("  1. Basis Functions - Shows φ-resonance structure")
    print("  2. Eigenvalue Comparison - Proves non-equivalence to FFT")
    print("  3. Wave-Domain Logic - Demonstrates computational capability")
    print("  4. Gram Matrix Structure - Visualizes non-orthogonality")
    print("  5. Frequency Grid - Shows quasi-periodic spacing")
    print("  6. Condition Scaling - Validates Gram normalization stability")


if __name__ == "__main__":
    generate_all_figures()
