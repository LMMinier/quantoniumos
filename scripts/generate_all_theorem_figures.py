#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate All Theorem Figures
Comprehensive visualization suite for all RFT theorems and variants
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path
import time
from typing import Dict, List, Tuple

# Import RFT modules
from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix, PHI
)
from algorithms.rft.hybrid_basis import (
    adaptive_hybrid_compress,
    hybrid_decomposition,
    braided_hybrid_mca,
    soft_braided_hybrid_mca,
)

# Output directories
BASE_DIR = Path("./figures/theorems")
BASE_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(" COMPREHENSIVE THEOREM FIGURE GENERATOR")
print("="*80)
print()

# ============================================================================
# THEOREM 10: HYBRID DECOMPOSITION
# ============================================================================

def generate_theorem10_rate_distortion():
    """
    Section 4: ASCII Bottleneck Rate-Distortion Analysis
    Shows DCT baseline, RFT catastrophic failure, Hybrid solution
    """
    print("Theorem 10 Figure 1: Rate-Distortion (ASCII Bottleneck)")
    
    # Simulate rate-distortion data (from Section 4 table)
    # In practice, run actual verify_rate_distortion.py
    
    transforms = ['DCT Only', 'RFT Only', 'Hybrid']
    rates = [4.83, 7.72, 4.96]  # Bits Per Pixel
    distortions = [0.0007, 0.0011, 0.0006]  # MSE
    colors = ['blue', 'red', 'green']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Rate comparison
    bars1 = ax1.bar(transforms, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Rate (Bits Per Pixel)', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Rate @ Iso-Distortion', fontsize=14, fontweight='bold')
    ax1.axhline(y=4.83, color='blue', linestyle='--', alpha=0.5, label='DCT Baseline')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Annotate catastrophic failure - fixed position
    ax1.annotate('CATASTROPHIC\nFAILURE\n+60% rate', 
                xy=(1, 7.72), xytext=(1, 9.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.5))
    
    # Annotate solution - fixed position
    ax1.annotate('SOLVED\nDCT parity', 
                xy=(2, 4.96), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.5))
    
    # Right: Distortion comparison
    bars2 = ax2.bar(transforms, distortions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Distortion (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.0007, color='blue', linestyle='--', alpha=0.5, label='DCT Baseline')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add overhead annotation - better placement
    overhead = 4.96 - 4.83
    ax1.text(2, 2.0, f'Overhead:\n{overhead:.2f} BPP', 
            ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8, pad=0.4))
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'theorem10_rate_distortion.png', dpi=300, bbox_inches='tight')
    plt.savefig(BASE_DIR / 'theorem10_rate_distortion.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {BASE_DIR}/theorem10_rate_distortion.[png/pdf]")


def generate_theorem10_greedy_vs_braided():
    """
    Section 5.3: Catastrophic Failure of Braided Parallel Competition
    Shows 30× error increase, 2× sparsity penalty, Pareto domination
    """
    print("Theorem 10 Figure 2: Greedy vs Braided Comparison")
    
    # Data from Section 5.3 tables
    datasets = ['Natural\nText', 'Python\nCode', 'Random\nASCII', 'Mixed\nSignal']
    
    # Compression efficiency (% coefficients used)
    greedy_sparsity = [41.60, 40.62, 44.34, 41.80]
    braided_sparsity = [81.05, 71.48, 73.05, 72.46]
    dct_baseline = [41.41, 41.02, 59.38, 24.61]
    
    # Reconstruction error
    greedy_error = [4.3e-3, 3.8e-3, 4.2e-3, 4.3e-3]
    braided_error = [0.526, 0.485, 0.263, 0.857]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Compression efficiency
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax1.bar(x - width, dct_baseline, width, label='DCT Baseline', 
                   color='blue', alpha=0.6, edgecolor='black')
    bars2 = ax1.bar(x, greedy_sparsity, width, label='Greedy (Ours)', 
                   color='green', alpha=0.7, edgecolor='black', linewidth=2)
    bars3 = ax1.bar(x + width, braided_sparsity, width, label='Braided (Failed)', 
                   color='red', alpha=0.6, edgecolor='black')
    
    ax1.set_ylabel('% Coefficients Used', fontsize=12, fontweight='bold')
    ax1.set_title('Test 1: Compression Efficiency (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Annotate catastrophic failure - better positioning
    ax1.text(1.5, 92, 'CATASTROPHIC FAILURE\n2x MORE COEFFICIENTS', 
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.5))
    ax1.set_ylim(0, 100)
    
    # Middle left: Reconstruction error (log scale)
    ax2 = fig.add_subplot(gs[1, 0])
    bars4 = ax2.bar(x - width/2, greedy_error, width, label='Greedy', 
                   color='green', alpha=0.7, edgecolor='black', linewidth=2)
    bars5 = ax2.bar(x + width/2, braided_error, width, label='Braided', 
                   color='red', alpha=0.6, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_ylabel('Reconstruction Error', fontsize=12, fontweight='bold')
    ax2.set_title('Reconstruction Quality', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Annotate 100x worse - fixed position
    ax2.text(1.5, 0.2, '100x HIGHER\nERROR', 
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.4))
    
    # Middle right: MCA Separation error
    ax3 = fig.add_subplot(gs[1, 1])
    sparsity_levels = ['Ks=4\nKt=4', 'Ks=4\nKt=8', 'Ks=8\nKt=4', 'Ks=8\nKt=8']
    greedy_mca = [0.032, 0.032, 0.032, 0.032]
    braided_mca = [0.914, 0.915, 0.920, 0.920]
    
    x_mca = np.arange(len(sparsity_levels))
    bars6 = ax3.bar(x_mca - width/2, greedy_mca, width, label='Greedy', 
                   color='green', alpha=0.7, edgecolor='black', linewidth=2)
    bars7 = ax3.bar(x_mca + width/2, braided_mca, width, label='Braided', 
                   color='red', alpha=0.6, edgecolor='black')
    ax3.set_ylabel('Separation Error', fontsize=12, fontweight='bold')
    ax3.set_title('Test 2: Source Separation (MCA)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_mca)
    ax3.set_xticklabels(sparsity_levels, fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Annotate 30x worse - fixed position
    ax3.text(1.5, 0.5, '30x WORSE\nSEPARATION', 
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.4))
    
    # Bottom: Rate-Distortion Pareto frontier
    ax4 = fig.add_subplot(gs[2, :])
    
    # Data from Test 3
    thresholds = [0.01, 0.05, 0.10, 0.20]
    greedy_rates = [0.410, 0.410, 0.410, 0.410]
    greedy_mse = [9.6e-6, 9.6e-6, 9.6e-6, 9.6e-6]
    braided_rates = [0.736, 0.738, 0.738, 0.566]
    braided_mse = [0.210, 0.199, 0.186, 0.176]
    
    ax4.scatter(greedy_rates, greedy_mse, s=200, marker='o', color='green', 
               edgecolor='black', linewidth=2, label='Greedy (Pareto Optimal)', zorder=3)
    ax4.scatter(braided_rates, braided_mse, s=200, marker='X', color='red', 
               edgecolor='black', linewidth=2, label='Braided (Dominated)', zorder=3)
    
    # Connect with arrows
    for i in range(len(thresholds)):
        ax4.annotate('', xy=(greedy_rates[i], greedy_mse[i]), 
                    xytext=(braided_rates[i], braided_mse[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))
    
    ax4.set_xlabel('Rate (Fraction of Coefficients)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Distortion (MSE)', fontsize=12, fontweight='bold')
    ax4.set_title('Test 3: Rate-Distortion Pareto Frontier (Lower-Left is Better)', 
                 fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, which='both')
    
    # Annotate domination - fixed position
    ax4.text(0.6, 0.05, 'DOMINATED ON\nBOTH AXES', 
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.5))
    
    # Add overall verdict - better spacing
    fig.text(0.5, 0.02, 'VERDICT: Braided catastrophically worse on ALL metrics', 
            ha='center', fontsize=13, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.95, pad=0.6))
    
    plt.savefig(BASE_DIR / 'theorem10_greedy_vs_braided.png', dpi=300, bbox_inches='tight')
    plt.savefig(BASE_DIR / 'theorem10_greedy_vs_braided.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {BASE_DIR}/theorem10_greedy_vs_braided.[png/pdf]")


def generate_theorem10_soft_braided():
    """
    Section 5.3.2: Soft Thresholding Partial Success
    Shows 1.17× improvement over hard, but still 18× worse than greedy
    """
    print("Theorem 10 Figure 3: Soft vs Hard Braided")
    
    methods = ['Greedy\n(Sequential)', 'Hard Braid\n(Parallel)', 'Soft Braid\n(Parallel)']
    reconstruction = [0.0402, 0.8518, 0.7255]
    sep_error_s = [1.51, 0.97, 0.91]
    sep_error_t = [1.00, 0.89, 0.89]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Reconstruction error
    colors = ['green', 'darkred', 'orange']
    bars1 = ax1.bar(methods, reconstruction, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Reconstruction Error', fontsize=12, fontweight='bold')
    ax1.set_title('Total Reconstruction Quality', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, which='both')
    
    # Annotate improvements - fixed overlap
    ax1.annotate('', xy=(2, 0.7255), xytext=(1, 0.8518),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.text(1.5, 0.85, '1.17x better\nSoft > Hard', 
            ha='center', fontsize=9, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.4))
    
    ax1.annotate('', xy=(0, 0.0402), xytext=(2, 0.7255),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax1.text(1, 0.25, '18x better\nGreedy > Soft', 
            ha='center', fontsize=9, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.4))
    
    # Right: Separation errors
    x = np.arange(len(methods))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, sep_error_s, width, label='Struct Error (S)', 
                   color='blue', alpha=0.6, edgecolor='black')
    bars3 = ax2.bar(x + width/2, sep_error_t, width, label='Texture Error (T)', 
                   color='red', alpha=0.6, edgecolor='black')
    
    ax2.set_ylabel('Component Separation Error', fontsize=12, fontweight='bold')
    ax2.set_title('Source Separation Quality', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add verdict - better formatting
    fig.text(0.5, 0.03, 
            'Scientific: Soft braiding proves parallel competition CAN work with proper smoothing\n'
            'Engineering: Greedy remains practical choice (18x better error)', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.9, pad=0.6))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(BASE_DIR / 'theorem10_soft_braided.png', dpi=300, bbox_inches='tight')
    plt.savefig(BASE_DIR / 'theorem10_soft_braided.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {BASE_DIR}/theorem10_soft_braided.[png/pdf]")


def generate_theorem10_phase_variants():
    """
    Section 6: RFT Kernel Variants (Standard, LogPhi, Mixed)
    Shows phase distributions and performance comparison
    """
    print("Theorem 10 Figure 4: RFT Phase Variants")
    
    n = 256
    k = np.arange(n)
    beta = 0.83
    
    # Standard phase
    theta_std = 2 * np.pi * beta * (k / PHI) % 1
    
    # Log-periodic phase
    theta_log = 2 * np.pi * beta * np.log(1 + k) / np.log(1 + n)
    
    # Mixed phase (alpha=0.5)
    alpha = 0.5
    theta_mix = (1 - alpha) * theta_std + alpha * theta_log
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Phase distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(k, theta_std, 'b-', linewidth=2, label='Standard φ-RFT')
    ax1.set_xlabel('Index k')
    ax1.set_ylabel('Phase θ(k) [rad]')
    ax1.set_title('Standard: frac(k/φ)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2*np.pi, color='k', linestyle='--', alpha=0.3)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k, theta_log, 'r-', linewidth=2, label='Log-Periodic')
    ax2.set_xlabel('Index k')
    ax2.set_ylabel('Phase θ(k) [rad]')
    ax2.set_title('LogPhi: log(1+k)/log(1+N)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=2*np.pi, color='k', linestyle='--', alpha=0.3)
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(k, theta_mix, 'g-', linewidth=2, label='Mixed (α=0.5)')
    ax3.set_xlabel('Index k')
    ax3.set_ylabel('Phase θ(k) [rad]')
    ax3.set_title('Mixed: (1-α)·std + α·log', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=2*np.pi, color='k', linestyle='--', alpha=0.3)
    ax3.legend()
    
    # Middle row: Unit circle representation
    for idx, (theta, title, color, ax_pos) in enumerate([
        (theta_std, 'Standard', 'blue', gs[1, 0]),
        (theta_log, 'LogPhi', 'red', gs[1, 1]),
        (theta_mix, 'Mixed', 'green', gs[1, 2])
    ]):
        ax = fig.add_subplot(ax_pos)
        phases = np.exp(1j * theta)
        scatter = ax.scatter(np.real(phases), np.imag(phases), 
                           c=k, cmap='twilight', s=20, alpha=0.7)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'{title} on Unit Circle', fontweight='bold')
        ax.set_aspect('equal')
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Index k')
    
    # Bottom: Performance comparison on different datasets
    ax7 = fig.add_subplot(gs[2, :])
    
    datasets = ['Natural\nText', 'Python\nCode', 'Random\nASCII', 'Mixed\nSignal']
    std_sparsity = [41.60, 40.62, 44.34, 41.80]
    log_sparsity = [41.60, 40.62, 44.34, 41.80]  # Same (from table)
    mix_sparsity = [41.60, 40.62, 44.34, 41.80]  # Same (from table)
    dct_baseline = [41.41, 41.02, 59.38, 24.61]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    ax7.bar(x - 1.5*width, dct_baseline, width, label='DCT Baseline', color='gray', alpha=0.5)
    ax7.bar(x - 0.5*width, std_sparsity, width, label='Standard φ-RFT', color='blue', alpha=0.7)
    ax7.bar(x + 0.5*width, log_sparsity, width, label='LogPhi RFT', color='red', alpha=0.7)
    ax7.bar(x + 1.5*width, mix_sparsity, width, label='Mixed RFT', color='green', alpha=0.7)
    
    ax7.set_ylabel('% Coefficients (99% Energy)', fontsize=12, fontweight='bold')
    ax7.set_title('Sparsity Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(datasets)
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)
    
    # Add observation - better spacing
    ax7.text(1.5, 70, 'All variants achieve PARITY on pure text\nAdaptive routing dominates', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.5))
    ax7.set_ylim(0, 80)
    
    plt.savefig(BASE_DIR / 'theorem10_phase_variants.png', dpi=300, bbox_inches='tight')
    plt.savefig(BASE_DIR / 'theorem10_phase_variants.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {BASE_DIR}/theorem10_phase_variants.[png/pdf]")


def generate_theorem10_mca_failure():
    """
    Section 5.2: MCA Source Separation Failure Visualization
    Shows why greedy captures everything in DCT
    """
    print("Theorem 10 Figure 5: MCA Separation Failure Analysis")
    
    # Simulate ground truth components
    np.random.seed(42)
    n = 256
    t = np.linspace(0, 4*np.pi, n)
    
    # DCT-sparse: Step function
    x_struct = np.zeros(n)
    x_struct[50:100] = 1.0
    x_struct[150:200] = -0.5
    
    # RFT-sparse: Golden ratio wave
    x_texture = 0.3 * np.sin(PHI * t) + 0.2 * np.sin(PHI**2 * t)
    
    # Mixture
    x_mixed = x_struct + x_texture
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Row 1: Ground truth components
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x_struct, 'b-', linewidth=2)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Ground Truth: Structure (DCT-sparse)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, x_texture, 'r-', linewidth=2)
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Ground Truth: Texture (RFT-sparse)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, x_mixed, 'g-', linewidth=2)
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Mixture: x = x_s + x_t', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Spectrum analysis
    from scipy.fft import dct, fft
    
    dct_struct = dct(x_struct, norm='ortho')
    rft_texture = rft_forward(x_texture)
    dct_mixed = dct(x_mixed, norm='ortho')
    rft_mixed = rft_forward(x_mixed)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(np.abs(dct_struct), 'b-', linewidth=1, alpha=0.7, label='Structure')
    ax4.plot(np.abs(dct_mixed), 'k--', linewidth=1, alpha=0.5, label='Mixture')
    ax4.set_ylabel('|DCT Coefficient|')
    ax4.set_title('DCT Domain', fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(np.abs(rft_texture), 'r-', linewidth=1, alpha=0.7, label='Texture')
    ax5.plot(np.abs(rft_mixed), 'k--', linewidth=1, alpha=0.5, label='Mixture')
    ax5.set_ylabel('|RFT Coefficient|')
    ax5.set_title('RFT Domain', fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Coherence visualization
    ax6 = fig.add_subplot(gs[1, 2])
    # Show overlap in coefficient magnitudes
    ax6.scatter(np.abs(dct_mixed), np.abs(rft_mixed), alpha=0.3, s=10)
    ax6.set_xlabel('|DCT[k]|')
    ax6.set_ylabel('|RFT[k]|')
    ax6.set_title('Mutual Coherence', fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.plot([1e-3, 10], [1e-3, 10], 'k--', alpha=0.3, label='Equal energy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Metrics comparison
    metrics = ['Total\nError', 'Struct\nError', 'Texture\nError', 'DCT\nSupport F1', 'RFT\nSupport F1']
    greedy_vals = [0.05, 1.5, 1.0, 0.65, 0.05]
    target_vals = [0.0, 0.0, 0.0, 1.0, 1.0]
    
    ax7 = fig.add_subplot(gs[2, :])
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, target_vals, width, label='Target (Perfect)', 
                   color='green', alpha=0.3, edgecolor='green', linewidth=2)
    bars2 = ax7.bar(x + width/2, greedy_vals, width, label='Greedy (Actual)', 
                   color='red', alpha=0.7, edgecolor='black', linewidth=2)
    
    ax7.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax7.set_title('MCA Separation Performance (Lower Error, Higher F1 is Better)', 
                 fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend(fontsize=11)
    ax7.grid(axis='y', alpha=0.3)
    ax7.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    # Annotate failures - fixed positions
    ax7.annotate('WORKS', xy=(0, 0.05), xytext=(0, 0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=9, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.3))
    
    ax7.annotate('FAILS', xy=(4, 0.05), xytext=(4, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.3))
    
    # Row 4: Root cause explanation
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    explanation = """ROOT CAUSE ANALYSIS: Why Greedy Sequential Fails at Separation

1. DCT BIAS: DCT captures BOTH structure AND texture energy
2. GREEDY SUBTRACTION: DCT wins iteration 1, subtracts ALL energy
3. RFT STARVATION: RFT never claims its rightful atoms
4. RESULT: "DCT-First Codec" not true MCA separator

What WORKS: Total reconstruction (err ~0.05) - Good for COMPRESSION
What FAILS: Component isolation (RFT F1 ~0.05) - Bad for SEPARATION

SOLUTION: Replace greedy/parallel with L1-minimization (BPDN)
    min ||s||_1 + ||t||_1  s.t.  ||x - (Psi_S*s + Psi_T*t)||_2 < eps
"""
    
    ax8.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=10, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
    
    plt.savefig(BASE_DIR / 'theorem10_mca_failure.png', dpi=300, bbox_inches='tight')
    plt.savefig(BASE_DIR / 'theorem10_mca_failure.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {BASE_DIR}/theorem10_mca_failure.[png/pdf]")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("Generating comprehensive theorem visualizations...")
    print()
    
    # Theorem 10: Hybrid Decomposition
    print("─" * 80)
    print(" THEOREM 10: HYBRID Φ-RFT / DCT DECOMPOSITION")
    print("─" * 80)
    
    generate_theorem10_rate_distortion()
    generate_theorem10_greedy_vs_braided()
    generate_theorem10_soft_braided()
    generate_theorem10_phase_variants()
    generate_theorem10_mca_failure()
    
    print()
    print("="*80)
    print(f" All figures saved to: {BASE_DIR}/")
    print("="*80)
    print()
    print("Generated files:")
    for f in sorted(BASE_DIR.glob("*.png")):
        size = f.stat().st_size / 1024
        print(f"  {f.name:50s} {size:6.1f} KB")
    print()
    print("✅ COMPLETE: All theorem figures generated")
    print()


if __name__ == "__main__":
    main()
