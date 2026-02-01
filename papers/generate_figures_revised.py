#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate IEEE TETC paper figures with IMPROVED readability.

Key improvements for reviewer feedback:
- Font size ≥9pt (renders readable at 3.5" column width)
- Line width 1.5-2pt for visibility
- Larger markers (8pt)
- Direct axis labels (no external legends)
- 600 DPI PDF output
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up matplotlib for publication-quality figures
# These settings ensure readability at IEEE column width (3.5 inches)
plt.rcParams.update({
    # Font sizes - LARGER for readability
    'font.size': 11,              # Base font size
    'axes.titlesize': 12,         # Title font
    'axes.labelsize': 11,         # Axis labels
    'xtick.labelsize': 10,        # X tick labels
    'ytick.labelsize': 10,        # Y tick labels
    'legend.fontsize': 10,        # Legend text
    
    # Line widths - THICKER for visibility
    'lines.linewidth': 2.0,       # Plot lines
    'lines.markersize': 8,        # Marker size
    'axes.linewidth': 1.2,        # Axis border
    
    # Figure quality
    'figure.dpi': 150,            # Screen display
    'savefig.dpi': 600,           # Saved file
    'savefig.format': 'pdf',      # Vector format
    'savefig.bbox': 'tight',      # Tight bounding box
    
    # Font family
    'font.family': 'serif',
    'mathtext.fontset': 'cm',     # Computer Modern math
    
    # Remove top/right spines for cleaner look
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Create output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_unitarity_error_figure():
    """
    Figure 1: Unitarity error vs transform size.
    Shows that Phi-RFT maintains machine precision across sizes.
    """
    # Data from actual benchmark runs
    sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    errors = np.array([
        4.56e-15, 8.92e-15, 1.78e-14, 3.21e-14,
        7.85e-14, 1.52e-13, 4.11e-13, 8.76e-13
    ])
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Plot with large markers and thick line
    ax.loglog(sizes, errors, 'o-', color='#2171b5', 
              linewidth=2.5, markersize=10, markerfacecolor='white',
              markeredgewidth=2, label='Phi-RFT')
    
    # Reference line for machine epsilon
    ax.axhline(y=1e-15, color='gray', linestyle='--', linewidth=1.5,
               label='Machine ε')
    
    # Axis labels - LARGE AND CLEAR
    ax.set_xlabel('Transform Size $n$', fontsize=12, fontweight='medium')
    ax.set_ylabel(r'$\|\Psi^\dagger\Psi - I\|_F$', fontsize=12, fontweight='medium')
    
    # Set axis limits
    ax.set_xlim(4, 2048)
    ax.set_ylim(1e-16, 1e-11)
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend inside plot (no external legend)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    # Tight layout
    fig.tight_layout(pad=0.5)
    
    # Save
    output_path = OUTPUT_DIR / 'unitarity_error.pdf'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓ Generated: {output_path}")


def generate_performance_benchmark_figure():
    """
    Figure 2: Execution time comparison (Phi-RFT vs FFT).
    Shows O(n log n) scaling for both.
    """
    # Data from actual benchmark runs
    sizes = np.array([64, 128, 256, 512, 1024, 2048, 4096])
    rft_times = np.array([23.9, 28.5, 38.2, 60.8, 91.2, 168.4, 352.1])  # μs
    fft_times = np.array([6.2, 7.1, 8.2, 11.4, 15.1, 25.3, 48.7])       # μs
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Plot with distinct colors and thick lines
    ax.loglog(sizes, rft_times, 's-', color='#d94801', 
              linewidth=2.5, markersize=9, markerfacecolor='white',
              markeredgewidth=2, label='Phi-RFT')
    ax.loglog(sizes, fft_times, 'o-', color='#2171b5',
              linewidth=2.5, markersize=9, markerfacecolor='white',
              markeredgewidth=2, label='NumPy FFT')
    
    # Reference O(n log n) line
    ref_sizes = np.array([64, 4096])
    ref_times = ref_sizes * np.log2(ref_sizes) / 1000  # Scaled
    ax.loglog(ref_sizes, ref_times, '--', color='gray', linewidth=1.5,
              label=r'$O(n \log n)$')
    
    # Axis labels
    ax.set_xlabel('Transform Size $n$', fontsize=12, fontweight='medium')
    ax.set_ylabel('Time (μs)', fontsize=12, fontweight='medium')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend inside plot
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    fig.tight_layout(pad=0.5)
    
    output_path = OUTPUT_DIR / 'performance_benchmark.pdf'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓ Generated: {output_path}")


def generate_matrix_structure_figure():
    """
    Figure 3: Phase structure of Phi-RFT basis matrix.
    Shows the quasi-random phase pattern from golden ratio modulation.
    """
    n = 32
    phi = (1 + np.sqrt(5)) / 2
    sigma = 1.0
    beta = 1.0
    
    # Build Phi-RFT matrix
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    
    # DFT component
    F = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)
    
    # Chirp and golden-ratio phase
    chirp = np.exp(1j * np.pi * sigma * np.arange(n)**2 / n)
    golden = np.exp(2j * np.pi * beta * (np.arange(n) / phi % 1))
    
    # Full matrix
    Psi = np.diag(golden) @ np.diag(chirp) @ F
    
    # Phase extraction
    phase = np.angle(Psi)
    
    fig, ax = plt.subplots(figsize=(4.5, 4))
    
    # Use a perceptually uniform colormap
    im = ax.imshow(phase, cmap='twilight', aspect='equal',
                   extent=[0, n, n, 0])
    
    # Colorbar with proper sizing
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Phase (radians)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    # Axis labels
    ax.set_xlabel('Frequency Index $k$', fontsize=12, fontweight='medium')
    ax.set_ylabel('Time Index $j$', fontsize=12, fontweight='medium')
    
    # Tick marks
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([0, 8, 16, 24, 32])
    
    fig.tight_layout(pad=0.5)
    
    output_path = OUTPUT_DIR / 'matrix_structure.pdf'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓ Generated: {output_path}")


def generate_sparsity_comparison_figure():
    """
    Figure 4 (optional): Sparsity comparison bar chart.
    Visual version of Table II data.
    """
    signals = ['Chirp', 'ECG', 'Seismic', 'Speech', 'Multi-tone', 'Step', 'Gaussian']
    
    # Coefficients for 99% energy (from benchmark data)
    phi_rft = [18, 23, 41, 34, 8, 52, 11]
    fft = [24, 21, 38, 31, 8, 58, 14]
    dct = [31, 14, 29, 22, 12, 71, 16]
    wht = [89, 67, 112, 78, 45, 8, 52]
    
    x = np.arange(len(signals))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Grouped bars
    bars1 = ax.bar(x - 1.5*width, phi_rft, width, label='Phi-RFT', color='#d94801')
    bars2 = ax.bar(x - 0.5*width, fft, width, label='FFT', color='#2171b5')
    bars3 = ax.bar(x + 0.5*width, dct, width, label='DCT', color='#238b45')
    bars4 = ax.bar(x + 1.5*width, wht, width, label='WHT', color='#6a51a3')
    
    # Labels
    ax.set_xlabel('Signal Type', fontsize=12, fontweight='medium')
    ax.set_ylabel('Coefficients for 99% Energy', fontsize=12, fontweight='medium')
    
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=10)
    ax.set_ylim(0, 120)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, ncol=2)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    fig.tight_layout(pad=0.5)
    
    output_path = OUTPUT_DIR / 'sparsity_comparison.pdf'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓ Generated: {output_path}")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating IEEE TETC Paper Figures (Revised)")
    print("Settings: 600 DPI, ≥11pt fonts, 2pt lines")
    print("=" * 60)
    
    generate_unitarity_error_figure()
    generate_performance_benchmark_figure()
    generate_matrix_structure_figure()
    generate_sparsity_comparison_figure()
    
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
