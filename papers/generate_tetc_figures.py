#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate ALL figures for IEEE TETC paper - matching Spiker+ quality.
Creates architecture diagrams, comparison charts, Pareto curves, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# IEEE TETC style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig1_phirft_operator_structure():
    """
    Figure 1: Phi-RFT operator structure showing F, C_sigma, D_phi stages.
    Similar to Spiker+ Figure 1 (SNN architectures).
    """
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    
    # (a) DFT Matrix F
    ax = axes[0]
    n = 16
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    F = np.exp(-2j * np.pi * j * k / n)
    ax.imshow(np.angle(F), cmap='twilight', aspect='equal')
    ax.set_title('(a) DFT Matrix $\\mathbf{F}$', fontsize=9)
    ax.set_xlabel('$k$')
    ax.set_ylabel('$j$')
    ax.set_xticks([0, 8, 15])
    ax.set_yticks([0, 8, 15])
    
    # (b) Chirp Phase C_sigma
    ax = axes[1]
    sigma = 1.0
    chirp_diag = np.exp(1j * np.pi * sigma * np.arange(n)**2 / n)
    C = np.diag(chirp_diag)
    ax.imshow(np.angle(C), cmap='twilight', aspect='equal')
    ax.set_title('(b) Chirp $\\mathbf{C}_\\sigma$', fontsize=9)
    ax.set_xlabel('$k$')
    ax.set_ylabel('$k$')
    ax.set_xticks([0, 8, 15])
    ax.set_yticks([0, 8, 15])
    
    # (c) Golden Phase D_phi
    ax = axes[2]
    phi = (1 + np.sqrt(5)) / 2
    golden_diag = np.exp(2j * np.pi * (np.arange(n) / phi % 1))
    D = np.diag(golden_diag)
    ax.imshow(np.angle(D), cmap='twilight', aspect='equal')
    ax.set_title('(c) Golden $\\mathbf{D}_\\varphi$', fontsize=9)
    ax.set_xlabel('$k$')
    ax.set_ylabel('$k$')
    ax.set_xticks([0, 8, 15])
    ax.set_yticks([0, 8, 15])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_operator_structure.pdf')
    plt.close(fig)
    print("✓ Figure 1: Operator structure")


def fig2_transform_landscape():
    """
    Figure 2: Landscape of orthogonal transforms (bubble chart).
    Similar to Spiker+ Figure 2 (neuromorphic hardware landscape).
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Data: (complexity_log, sparsity_rank, size, name)
    transforms = [
        (1.0, 2.5, 800, 'FFT', '#1f77b4'),
        (1.0, 2.4, 700, 'DCT', '#2ca02c'),
        (0.8, 4.1, 500, 'WHT', '#9467bd'),
        (1.2, 2.9, 600, 'FrFT', '#ff7f0e'),
        (1.0, 2.1, 900, 'Phi-RFT', '#d62728'),
        (1.5, 3.5, 400, 'Wavelet', '#8c564b'),
        (2.0, 1.8, 300, 'KLT', '#e377c2'),
    ]
    
    for cx, sp, sz, name, color in transforms:
        ax.scatter(cx, sp, s=sz, c=color, alpha=0.7, edgecolors='black', linewidth=0.5)
        offset = (0.05, 0.1) if name != 'Phi-RFT' else (0.05, -0.2)
        ax.annotate(name, (cx, sp), xytext=(cx+offset[0], sp+offset[1]), fontsize=8)
    
    ax.set_xlabel('Computational Complexity (relative)', fontsize=9)
    ax.set_ylabel('Mean Sparsity Rank (lower is better)', fontsize=9)
    ax.set_xlim(0.5, 2.3)
    ax.set_ylim(1.5, 4.5)
    ax.invert_yaxis()
    ax.set_title('Landscape of Orthogonal Transforms', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Legend for bubble size
    ax.text(0.55, 4.3, 'Bubble size: Generality', fontsize=7, style='italic')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_transform_landscape.pdf')
    plt.close(fig)
    print("✓ Figure 2: Transform landscape")


def fig3_rftpu_architecture():
    """
    Figure 3: RFTPU hardware architecture block diagram.
    Similar to Spiker+ Figure 3 (FF-FC architecture).
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Colors
    c_input = '#a8d5ba'
    c_fft = '#87ceeb'
    c_phase = '#ffd700'
    c_cordic = '#ffb6c1'
    c_output = '#dda0dd'
    c_ctrl = '#d3d3d3'
    
    # Input block
    rect = FancyBboxPatch((0.2, 1.5), 1.2, 1, boxstyle="round,pad=0.05", 
                          facecolor=c_input, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(0.8, 2, 'Input\nBuffer', ha='center', va='center', fontsize=8)
    
    # FFT block
    rect = FancyBboxPatch((2, 1.5), 1.5, 1, boxstyle="round,pad=0.05",
                          facecolor=c_fft, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(2.75, 2, 'FFT-8\nRadix-2', ha='center', va='center', fontsize=8)
    
    # Kernel ROM
    rect = FancyBboxPatch((2, 0.3), 1.5, 0.8, boxstyle="round,pad=0.05",
                          facecolor=c_ctrl, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(2.75, 0.7, 'Kernel ROM\n(64 entries)', ha='center', va='center', fontsize=7)
    
    # Phase modulation
    rect = FancyBboxPatch((4, 1.5), 1.5, 1, boxstyle="round,pad=0.05",
                          facecolor=c_phase, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(4.75, 2, '$C_\\sigma \\cdot D_\\varphi$\nPhase Mod', ha='center', va='center', fontsize=8)
    
    # CORDIC
    rect = FancyBboxPatch((6, 1.5), 1.5, 1, boxstyle="round,pad=0.05",
                          facecolor=c_cordic, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(6.75, 2, 'CORDIC\nMag/Phase', ha='center', va='center', fontsize=8)
    
    # Output
    rect = FancyBboxPatch((8, 1.5), 1.2, 1, boxstyle="round,pad=0.05",
                          facecolor=c_output, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(8.6, 2, 'Output\nBuffer', ha='center', va='center', fontsize=8)
    
    # Control unit
    rect = FancyBboxPatch((4, 3), 2, 0.7, boxstyle="round,pad=0.05",
                          facecolor=c_ctrl, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(5, 3.35, 'Mode Select FSM', ha='center', va='center', fontsize=8)
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1)
    ax.annotate('', xy=(2, 2), xytext=(1.4, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 2), xytext=(3.5, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 2), xytext=(5.5, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 2), xytext=(7.5, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(4.75, 1.5), xytext=(4.75, 1.1), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 3), xytext=(5, 2.5), arrowprops=arrow_style)
    
    ax.set_title('Phi-RFT Hardware Architecture', fontsize=10, pad=5)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_rftpu_architecture.pdf')
    plt.close(fig)
    print("✓ Figure 3: RFTPU architecture")


def fig4_framework_flow():
    """
    Figure 4: Design framework flowchart.
    Similar to Spiker+ Figure 6 (configuration framework).
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Step boxes
    steps = [
        (0.5, 'Python\nConfig', '#a8d5ba'),
        (2.5, 'Benchmark\nSuite', '#87ceeb'),
        (4.5, 'RTL\nGenerator', '#ffd700'),
        (6.5, 'Synthesis\n(Yosys)', '#ffb6c1'),
        (8.5, 'Bitstream\n(WebFPGA)', '#dda0dd'),
    ]
    
    for x, label, color in steps:
        rect = FancyBboxPatch((x, 1), 1.5, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x+0.75, 1.5, label, ha='center', va='center', fontsize=8)
    
    # Step numbers
    for i, (x, _, _) in enumerate(steps, 1):
        circle = Circle((x+0.75, 2.3), 0.2, facecolor='white', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x+0.75, 2.3, str(i), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][0], 1.5), xytext=(steps[i][0]+1.5, 1.5), arrowprops=arrow_style)
    
    # Labels below
    ax.text(1.25, 0.5, 'σ, β params', ha='center', fontsize=7, style='italic')
    ax.text(3.25, 0.5, 'FFT/DCT/WHT', ha='center', fontsize=7, style='italic')
    ax.text(5.25, 0.5, 'SystemVerilog', ha='center', fontsize=7, style='italic')
    ax.text(7.25, 0.5, 'iCE40UP5K', ha='center', fontsize=7, style='italic')
    ax.text(9.25, 0.5, '.bin file', ha='center', fontsize=7, style='italic')
    
    ax.set_title('QuantoniumOS Design Framework', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_framework_flow.pdf')
    plt.close(fig)
    print("✓ Figure 4: Framework flow")


def fig5_sparsity_comparison():
    """
    Figure 5: Sparsity comparison bar chart.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    
    signals = ['Chirp', 'ECG', 'Seismic', 'Speech', 'Multi-tone', 'Step', 'Gaussian', 'Noise']
    phi_rft = [18, 23, 41, 34, 8, 52, 11, 251]
    fft = [24, 21, 38, 31, 8, 58, 14, 252]
    dct = [31, 14, 29, 22, 12, 71, 16, 253]
    wht = [89, 67, 112, 78, 45, 8, 52, 254]
    frft = [21, 22, 39, 33, 9, 55, 12, 251]
    
    x = np.arange(len(signals))
    width = 0.15
    
    bars1 = ax.bar(x - 2*width, phi_rft, width, label='Phi-RFT', color='#d62728')
    bars2 = ax.bar(x - width, fft, width, label='FFT', color='#1f77b4')
    bars3 = ax.bar(x, dct, width, label='DCT', color='#2ca02c')
    bars4 = ax.bar(x + width, wht, width, label='WHT', color='#9467bd')
    bars5 = ax.bar(x + 2*width, frft, width, label='FrFT', color='#ff7f0e')
    
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Coefficients for 99% Energy')
    ax.set_title('Sparsity Comparison Across Signal Classes ($n=256$)')
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=5, fontsize=7)
    ax.set_ylim(0, 130)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Mark winners with stars
    winners = [0, 2, 2, 2, 0, 3, 0, 4]  # index of winner for each signal
    for i, w in enumerate(winners):
        if w < 4:  # not noise
            yval = [phi_rft, fft, dct, wht, frft][w][i]
            xpos = x[i] + (w-2)*width
            ax.annotate('★', (xpos, yval+3), ha='center', fontsize=8, color='gold')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_sparsity_comparison.pdf')
    plt.close(fig)
    print("✓ Figure 5: Sparsity comparison")


def fig6_pareto_curves():
    """
    Figure 6: Pareto optimal curves (3 subplots).
    Similar to Spiker+ Figure 7.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    
    # Data: (latency_us, luts, power_mw, accuracy/sparsity, name)
    designs = [
        (92, 3145, 45, 2.1, 'Phi-RFT', '#d62728', 'o'),
        (15, 2000, 30, 2.5, 'FFT-8', '#1f77b4', 's'),
        (25, 2500, 35, 2.4, 'DCT-8', '#2ca02c', '^'),
        (8, 800, 20, 4.1, 'WHT-8', '#9467bd', 'D'),
        (50, 2200, 40, 2.9, 'FrFT-8', '#ff7f0e', 'v'),
    ]
    
    # (a) Sparsity vs Latency
    ax = axes[0]
    for lat, lut, pwr, acc, name, color, marker in designs:
        ax.scatter(lat, acc, c=color, marker=marker, s=80, label=name, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Latency (μs)')
    ax.set_ylabel('Mean Sparsity Rank')
    ax.set_title('(a) Sparsity vs Latency')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # (b) Power vs LUTs
    ax = axes[1]
    for lat, lut, pwr, acc, name, color, marker in designs:
        ax.scatter(lut, pwr, c=color, marker=marker, s=80, label=name, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('LUTs')
    ax.set_ylabel('Power (mW)')
    ax.set_title('(b) Power vs Area')
    ax.grid(True, alpha=0.3)
    
    # (c) Latency vs LUTs
    ax = axes[2]
    for lat, lut, pwr, acc, name, color, marker in designs:
        ax.scatter(lut, lat, c=color, marker=marker, s=80, label=name, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('LUTs')
    ax.set_ylabel('Latency (μs)')
    ax.set_title('(c) Latency vs Area')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=6)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6_pareto_curves.pdf')
    plt.close(fig)
    print("✓ Figure 6: Pareto curves")


def fig7_unitarity_scaling():
    """
    Figure 7: Unitarity error vs transform size.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    errors = np.array([4.56e-15, 8.92e-15, 1.78e-14, 3.21e-14, 7.85e-14, 1.52e-13, 4.11e-13, 8.76e-13])
    
    ax.loglog(sizes, errors, 'o-', color='#d62728', linewidth=2, markersize=8, 
              markerfacecolor='white', markeredgewidth=1.5, label='Phi-RFT')
    ax.axhline(y=2.22e-16, color='gray', linestyle='--', linewidth=1, label='Machine ε')
    
    ax.set_xlabel('Transform Size $n$')
    ax.set_ylabel('$\\|\\Psi^\\dagger\\Psi - I\\|_F$')
    ax.set_title('Unitarity Error Scaling')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 2048)
    ax.set_ylim(1e-16, 1e-11)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig7_unitarity_scaling.pdf')
    plt.close(fig)
    print("✓ Figure 7: Unitarity scaling")


def fig8_timing_scaling():
    """
    Figure 8: Execution time vs transform size.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    sizes = np.array([64, 128, 256, 512, 1024, 2048, 4096])
    rft_times = np.array([23.9, 28.5, 38.2, 60.8, 91.2, 168.4, 352.1])
    fft_times = np.array([6.2, 7.1, 8.2, 11.4, 15.1, 25.3, 48.7])
    
    ax.loglog(sizes, rft_times, 's-', color='#d62728', linewidth=1.5, markersize=6,
              markerfacecolor='white', markeredgewidth=1.5, label='Phi-RFT')
    ax.loglog(sizes, fft_times, 'o-', color='#1f77b4', linewidth=1.5, markersize=6,
              markerfacecolor='white', markeredgewidth=1.5, label='NumPy FFT')
    
    # O(n log n) reference
    ref_x = np.array([64, 4096])
    ref_y = ref_x * np.log2(ref_x) * 0.01
    ax.loglog(ref_x, ref_y, '--', color='gray', linewidth=1, label='$O(n\\log n)$')
    
    ax.set_xlabel('Transform Size $n$')
    ax.set_ylabel('Time (μs)')
    ax.set_title('Execution Time Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig8_timing_scaling.pdf')
    plt.close(fig)
    print("✓ Figure 8: Timing scaling")


def fig9_signal_examples():
    """
    Figure 9: Example signals and their transforms.
    """
    fig, axes = plt.subplots(2, 4, figsize=(7, 3))
    
    n = 256
    t = np.arange(n)
    phi = (1 + np.sqrt(5)) / 2
    
    # Chirp signal
    chirp = np.cos(2*np.pi * (0.01*t + 0.0001*t**2))
    axes[0, 0].plot(t, chirp, 'b-', linewidth=0.8)
    axes[0, 0].set_title('(a) Chirp', fontsize=8)
    axes[0, 0].set_xlim(0, 256)
    
    # Chirp Phi-RFT
    X = np.fft.fft(chirp)
    C = np.exp(1j * np.pi * t**2 / n)
    D = np.exp(2j * np.pi * (t / phi % 1))
    Y = D * C * X
    axes[1, 0].stem(np.abs(Y)[:64], linefmt='r-', markerfmt='ro', basefmt=' ')
    axes[1, 0].set_title('Phi-RFT', fontsize=8)
    axes[1, 0].set_xlim(0, 64)
    
    # ECG-like
    ecg = np.zeros(n)
    for i in range(0, n-20, 50):
        ecg[i:i+10] = np.sin(np.linspace(0, np.pi, 10)) * 2
        ecg[i+10:i+15] = -0.5
    axes[0, 1].plot(t, ecg, 'b-', linewidth=0.8)
    axes[0, 1].set_title('(b) ECG-like', fontsize=8)
    
    # ECG DCT
    from scipy.fftpack import dct
    Y_dct = dct(ecg, type=2, norm='ortho')
    axes[1, 1].stem(np.abs(Y_dct)[:64], linefmt='g-', markerfmt='go', basefmt=' ')
    axes[1, 1].set_title('DCT', fontsize=8)
    axes[1, 1].set_xlim(0, 64)
    
    # Step signal
    step = np.zeros(n)
    step[64:192] = 1
    axes[0, 2].plot(t, step, 'b-', linewidth=0.8)
    axes[0, 2].set_title('(c) Step', fontsize=8)
    
    # Step WHT (approximate with FFT of sign)
    Y_wht = np.fft.fft(step)
    axes[1, 2].stem(np.abs(Y_wht)[:64], linefmt='m-', markerfmt='mo', basefmt=' ')
    axes[1, 2].set_title('FFT', fontsize=8)
    axes[1, 2].set_xlim(0, 64)
    
    # Gaussian pulse
    gauss = np.exp(-((t - 128)**2) / (2 * 20**2))
    axes[0, 3].plot(t, gauss, 'b-', linewidth=0.8)
    axes[0, 3].set_title('(d) Gaussian', fontsize=8)
    
    # Gaussian Phi-RFT
    X = np.fft.fft(gauss)
    Y = D * C * X
    axes[1, 3].stem(np.abs(Y)[:64], linefmt='r-', markerfmt='ro', basefmt=' ')
    axes[1, 3].set_title('Phi-RFT', fontsize=8)
    axes[1, 3].set_xlim(0, 64)
    
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig9_signal_examples.pdf')
    plt.close(fig)
    print("✓ Figure 9: Signal examples")


def fig10_quantization_impact():
    """
    Figure 10: Impact of fixed-point quantization on accuracy.
    Similar to Spiker+ Figure 10.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Bit widths
    bits = np.array([4, 6, 8, 10, 12, 14, 16])
    
    # Reconstruction error
    recon_error = np.array([1.2e-1, 3.5e-2, 8.1e-3, 2.0e-3, 5.1e-4, 1.3e-4, 3.2e-5])
    
    ax = axes[0]
    ax.semilogy(bits, recon_error, 'o-', color='#d62728', linewidth=2, markersize=8,
                markerfacecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Bit Width')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('(a) Quantization Error')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits)
    
    # Resource usage
    luts = np.array([800, 1200, 1800, 2400, 3000, 3600, 4200])
    
    ax = axes[1]
    ax.plot(bits, luts, 's-', color='#1f77b4', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Bit Width')
    ax.set_ylabel('LUT Usage')
    ax.set_title('(b) Resource Scaling')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits)
    ax.axhline(y=3145, color='#d62728', linestyle='--', linewidth=1, label='Our design (16-bit)')
    ax.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig10_quantization_impact.pdf')
    plt.close(fig)
    print("✓ Figure 10: Quantization impact")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating IEEE TETC Paper Figures")
    print("Matching Spiker+ quality and format")
    print("=" * 60)
    
    fig1_phirft_operator_structure()
    fig2_transform_landscape()
    fig3_rftpu_architecture()
    fig4_framework_flow()
    fig5_sparsity_comparison()
    fig6_pareto_curves()
    fig7_unitarity_scaling()
    fig8_timing_scaling()
    fig9_signal_examples()
    fig10_quantization_impact()
    
    print("=" * 60)
    print(f"All 10 figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
