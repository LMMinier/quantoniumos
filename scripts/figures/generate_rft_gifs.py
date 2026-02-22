#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Generate Animated GIFs for RFT Visualization
Shows dynamic behavior of transforms, phase evolution, and signal processing
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix, PHI

# Create output directory
OUTPUT_DIR = Path("./figures/gifs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating animated GIFs for RFT visualization...")
print(f"Output directory: {OUTPUT_DIR}/")
print()

def create_transform_evolution_gif():
    """Animate signal transform from time to frequency domain"""
    print("1. Creating transform evolution animation...")
    
    n = 128
    t = np.arange(n)
    
    # Signal: sum of two frequencies
    x = np.sin(2 * np.pi * 5 * t / n) + 0.5 * np.sin(2 * np.pi * 12 * t / n)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RFT Transform Evolution', fontsize=16, fontweight='bold')
    
    # Setup plots
    line_time, = axes[0, 0].plot([], [], 'b-', linewidth=2)
    axes[0, 0].set_xlim(0, n)
    axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    stem_rft = axes[0, 1].stem([0], [0], basefmt=' ')
    axes[0, 1].set_xlim(0, n)
    axes[0, 1].set_ylim(0, 8)
    axes[0, 1].set_xlabel('Frequency Bin')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('RFT Spectrum (Building)')
    axes[0, 1].grid(True, alpha=0.3)
    
    stem_fft = axes[1, 0].stem([0], [0], basefmt=' ')
    axes[1, 0].set_xlim(0, n)
    axes[1, 0].set_ylim(0, 8)
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('FFT Spectrum (Reference)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase plot
    scatter_phase = axes[1, 1].scatter([], [], c=[], cmap='twilight', s=50, alpha=0.6)
    axes[1, 1].set_xlim(-8, 8)
    axes[1, 1].set_ylim(-8, 8)
    axes[1, 1].set_xlabel('Real')
    axes[1, 1].set_ylabel('Imaginary')
    axes[1, 1].set_title('RFT Complex Plane')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    # Pre-compute transforms
    X_rft = rft_forward(x)
    X_fft = np.fft.fft(x, norm='ortho')
    
    def init():
        line_time.set_data([], [])
        return line_time, stem_rft, stem_fft, scatter_phase
    
    def animate(frame):
        # Progress from 0 to n
        progress = int((frame / 60) * n)
        
        # Time domain - reveal signal progressively
        line_time.set_data(t[:progress], x[:progress])
        
        # RFT spectrum - build up
        X_rft_partial = X_rft.copy()
        X_rft_partial[progress:] = 0
        stem_rft.markerline.set_data(np.arange(n), np.abs(X_rft_partial))
        
        # Update stemlines efficiently for LineCollection
        segments_rft = []
        for i, val in enumerate(np.abs(X_rft_partial)):
            segments_rft.append([[i, 0], [i, val]])
        stem_rft.stemlines.set_segments(segments_rft)
        
        # FFT spectrum - for comparison
        X_fft_partial = X_fft.copy()
        X_fft_partial[progress:] = 0
        stem_fft.markerline.set_data(np.arange(n), np.abs(X_fft_partial))
        
        # Update stemlines efficiently for LineCollection
        segments_fft = []
        for i, val in enumerate(np.abs(X_fft_partial)):
            segments_fft.append([[i, 0], [i, val]])
        stem_fft.stemlines.set_segments(segments_fft)
        
        # Complex plane
        X_show = X_rft_partial[np.abs(X_rft_partial) > 0.01]
        if len(X_show) > 0:
            scatter_phase.set_offsets(np.c_[np.real(X_show), np.imag(X_show)])
            scatter_phase.set_array(np.angle(X_show))
        
        return line_time, stem_rft, stem_fft, scatter_phase
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60, 
                                   interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'transform_evolution.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/transform_evolution.gif")

def create_phase_rotation_gif():
    """Animate the golden ratio phase structure"""
    print("2. Creating phase rotation animation...")
    
    n = 64
    k = np.arange(n)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Golden Ratio Phase Evolution', fontsize=16, fontweight='bold')
    
    # RFT phase
    scatter_rft = axes[0].scatter([], [], c=[], cmap='twilight', s=100, alpha=0.7)
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Imaginary')
    axes[0].set_title('RFT Phase (Φ-based)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    circle_rft = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    axes[0].add_patch(circle_rft)
    
    # FFT phase
    scatter_fft = axes[1].scatter([], [], c=[], cmap='twilight', s=100, alpha=0.7)
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].set_title('FFT Phase (Uniform)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    circle_fft = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    axes[1].add_patch(circle_fft)
    
    def init():
        scatter_rft.set_offsets(np.empty((0, 2)))
        scatter_fft.set_offsets(np.empty((0, 2)))
        return scatter_rft, scatter_fft
    
    def animate(frame):
        # Rotate through one full cycle
        beta_offset = 2 * np.pi * frame / 60
        
        # RFT phases (golden ratio based)
        theta_rft = 2.0 * np.pi * np.modf(k / PHI)[0] + beta_offset
        phases_rft = np.exp(1j * theta_rft)
        
        # FFT phases (uniform)
        theta_fft = 2.0 * np.pi * k / n + beta_offset
        phases_fft = np.exp(1j * theta_fft)
        
        # Update RFT
        scatter_rft.set_offsets(np.c_[np.real(phases_rft), np.imag(phases_rft)])
        scatter_rft.set_array(theta_rft % (2*np.pi))
        
        # Update FFT
        scatter_fft.set_offsets(np.c_[np.real(phases_fft), np.imag(phases_fft)])
        scatter_fft.set_array(theta_fft % (2*np.pi))
        
        return scatter_rft, scatter_fft
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60,
                                   interval=50, blit=True)
    
    writer = PillowWriter(fps=20)
    anim.save(OUTPUT_DIR / 'phase_rotation.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/phase_rotation.gif")

def create_signal_reconstruction_gif():
    """Animate signal reconstruction from frequency domain"""
    print("3. Creating signal reconstruction animation...")
    
    n = 128
    t = np.arange(n)
    
    # Original signal
    x = np.sin(2 * np.pi * 5 * t / n) + 0.3 * np.sin(2 * np.pi * 15 * t / n)
    X_rft = rft_forward(x)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Signal Reconstruction via Inverse RFT', fontsize=16, fontweight='bold')
    
    # Frequency domain
    stem_freq = axes[0].stem(np.arange(n), np.abs(X_rft), basefmt=' ')
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, np.max(np.abs(X_rft)) * 1.1)
    axes[0].set_xlabel('Frequency Bin')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('RFT Spectrum (Selecting Coefficients)')
    axes[0].grid(True, alpha=0.3)
    
    # Time domain reconstruction
    line_original, = axes[1].plot(t, x, 'gray', linewidth=1, alpha=0.5, label='Original')
    line_recon, = axes[1].plot([], [], 'r-', linewidth=2, label='Reconstructed')
    axes[1].set_xlim(0, n)
    axes[1].set_ylim(-2, 2)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Time Domain Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    def init():
        line_recon.set_data([], [])
        return stem_freq, line_recon
    
    def animate(frame):
        # Progress from 0 to n coefficients
        n_coeffs = int((frame / 60) * n)
        
        # Reconstruct using first n_coeffs
        X_partial = X_rft.copy()
        X_partial[n_coeffs:] = 0
        x_recon = rft_inverse(X_partial)
        
        # Update frequency domain (highlight used coefficients)
        colors = ['red' if i < n_coeffs else 'lightgray' for i in range(n)]
        stem_freq.stemlines.set_colors(colors)
        # markerline is Line2D, cannot set individual colors easily.
        stem_freq.markerline.set_color('blue')
        
        # Update reconstruction
        line_recon.set_data(t, np.real(x_recon))
        
        return stem_freq, line_recon
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60,
                                   interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'signal_reconstruction.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/signal_reconstruction.gif")

def create_compression_demo_gif():
    """Animate compression by removing small coefficients"""
    print("4. Creating compression demonstration animation...")
    
    n = 256
    t = np.arange(n)
    
    # Signal with rich frequency content
    x = (np.sin(2 * np.pi * 3 * t / n) + 
         0.5 * np.sin(2 * np.pi * 7 * t / n) +
         0.3 * np.sin(2 * np.pi * 15 * t / n) +
         0.1 * np.random.randn(n))
    
    X_rft = rft_forward(x)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('RFT Compression (Removing Small Coefficients)', fontsize=16, fontweight='bold')
    
    # Sorted coefficients
    line_sorted, = axes[0].plot([], [], 'b-', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    threshold_line = axes[0].axhline(y=0, color='g', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, np.max(np.abs(X_rft)) * 1.1)
    axes[0].set_xlabel('Coefficient Index (sorted)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Sorted Coefficient Magnitudes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Original vs compressed
    line_original, = axes[1].plot(t, x, 'b-', linewidth=1, alpha=0.7, label='Original')
    line_compressed, = axes[1].plot([], [], 'r-', linewidth=2, label='Compressed')
    axes[1].set_xlim(0, n)
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Signal Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Error
    line_error, = axes[2].plot([], [], 'purple', linewidth=2)
    axes[2].set_xlim(0, n)
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Error')
    axes[2].set_title('Reconstruction Error')
    axes[2].grid(True, alpha=0.3)
    
    # Pre-sort coefficients
    sorted_indices = np.argsort(np.abs(X_rft))[::-1]
    sorted_mags = np.abs(X_rft)[sorted_indices]
    
    text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes, 
                       verticalalignment='top', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        line_sorted.set_data([], [])
        line_compressed.set_data([], [])
        line_error.set_data([], [])
        return line_sorted, line_compressed, line_error, threshold_line, text
    
    def animate(frame):
        # Threshold increases from 0 to max
        threshold = (frame / 60) * np.max(sorted_mags)
        
        # Keep coefficients above threshold
        X_compressed = X_rft.copy()
        X_compressed[np.abs(X_compressed) < threshold] = 0
        
        n_kept = np.sum(np.abs(X_compressed) > 0)
        compression_ratio = n / max(1, n_kept)
        
        # Reconstruct
        x_compressed = rft_inverse(X_compressed)
        error = np.real(x_compressed) - x
        
        # Update plots
        line_sorted.set_data(np.arange(n), sorted_mags)
        threshold_line.set_ydata([threshold, threshold])
        
        line_compressed.set_data(t, np.real(x_compressed))
        line_error.set_data(t, error)
        
        text.set_text(f'Coefficients: {n_kept}/{n}\nCompression: {compression_ratio:.1f}:1')
        
        return line_sorted, line_compressed, line_error, threshold_line, text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60,
                                   interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'compression_demo.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/compression_demo.gif")

def create_unitarity_demo_gif():
    """Animate unitarity demonstration (energy preservation)"""
    print("5. Creating unitarity demonstration animation...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RFT Unitarity: Energy Preservation', fontsize=16, fontweight='bold')
    
    n = 128
    
    # Pre-generate random signals
    np.random.seed(42)
    signals = [np.random.randn(n) + 1j * np.random.randn(n) for _ in range(60)]
    
    scatter1 = axes[0, 0].scatter([], [], alpha=0.6, s=50)
    axes[0, 0].plot([0, 10], [0, 10], 'r--', linewidth=2, alpha=0.5)
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_xlabel('Time Domain Energy')
    axes[0, 0].set_ylabel('Frequency Domain Energy')
    axes[0, 0].set_title('Energy Conservation (E_time = E_freq)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    line_time, = axes[0, 1].plot([], [], 'b-', linewidth=2)
    axes[0, 1].set_xlim(0, n)
    axes[0, 1].set_ylim(-3, 3)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Time Domain Signal')
    axes[0, 1].grid(True, alpha=0.3)
    
    stem_freq = axes[1, 0].stem([0], [0], basefmt=' ')
    axes[1, 0].set_xlim(0, n)
    axes[1, 0].set_ylim(0, 2)
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('RFT Spectrum')
    axes[1, 0].grid(True, alpha=0.3)
    
    bar_energy = axes[1, 1].bar([0, 1], [0, 0], color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_xlim(-0.5, 1.5)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Time', 'Frequency'])
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Energy Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    energies_time = []
    energies_freq = []
    
    def init():
        scatter1.set_offsets(np.empty((0, 2)))
        line_time.set_data([], [])
        return scatter1, line_time, stem_freq, bar_energy
    
    def animate(frame):
        x = signals[frame]
        X = rft_forward(x)
        
        # Calculate energies
        e_time = np.sum(np.abs(x) ** 2)
        e_freq = np.sum(np.abs(X) ** 2)
        
        energies_time.append(e_time)
        energies_freq.append(e_freq)
        
        # Update scatter
        scatter1.set_offsets(np.c_[energies_time, energies_freq])
        scatter1.set_array(np.arange(len(energies_time)))
        
        # Update time domain
        line_time.set_data(np.arange(n), np.real(x))
        
        # Update frequency domain
        stem_freq.markerline.set_data(np.arange(n), np.abs(X))
        
        segments = []
        for i, val in enumerate(np.abs(X)):
            segments.append([[i, 0], [i, val]])
        stem_freq.stemlines.set_segments(segments)
        
        # Update bars
        bar_energy[0].set_height(e_time)
        bar_energy[1].set_height(e_freq)
        
        return scatter1, line_time, stem_freq, bar_energy
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60,
                                   interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'unitarity_demo.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/unitarity_demo.gif")

def main():
    print("="*70)
    print(" RFT Animated GIF Generator")
    print("="*70)
    print()
    
    create_transform_evolution_gif()
    create_phase_rotation_gif()
    create_signal_reconstruction_gif()
    create_compression_demo_gif()
    create_unitarity_demo_gif()
    
    print()
    print("="*70)
    print(f" All GIFs saved to: {OUTPUT_DIR}/")
    print("="*70)
    print()
    print("Generated GIFs:")
    print("  1. transform_evolution.gif    - Signal transform animation")
    print("  2. phase_rotation.gif         - Golden ratio phase structure")
    print("  3. signal_reconstruction.gif  - Inverse transform building signal")
    print("  4. compression_demo.gif       - Coefficient thresholding demo")
    print("  5. unitarity_demo.gif         - Energy preservation verification")
    print()

if __name__ == "__main__":
    main()
