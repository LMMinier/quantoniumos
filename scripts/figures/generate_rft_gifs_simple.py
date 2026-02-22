#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Generate Animated GIFs for RFT Visualization (Simplified)
Shows dynamic behavior of transforms using simpler plotting methods
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, PHI

# Create output directory
OUTPUT_DIR = Path("./figures/gifs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating animated GIFs for RFT visualization...")
print(f"Output directory: {OUTPUT_DIR}/")
print()

def create_phase_rotation_gif():
    """Animate the golden ratio phase structure"""
    print("1. Creating phase rotation animation...")
    
    n = 64
    k = np.arange(n)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Golden Ratio Phase Evolution', fontsize=16, fontweight='bold')
    
    # Setup axes
    for ax in axes:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    axes[0].set_title('RFT Phase (Φ-based)')
    axes[1].set_title('FFT Phase (Uniform)')
    
    scatter_rft = axes[0].scatter([], [], c=[], cmap='twilight', s=100, alpha=0.7, vmin=0, vmax=2*np.pi)
    scatter_fft = axes[1].scatter([], [], c=[], cmap='twilight', s=100, alpha=0.7, vmin=0, vmax=2*np.pi)
    
    def animate(frame):
        beta_offset = 2 * np.pi * frame / 60
        
        # RFT phases
        theta_rft = 2.0 * np.pi * np.modf(k / PHI)[0] + beta_offset
        phases_rft = np.exp(1j * theta_rft)
        scatter_rft.set_offsets(np.c_[np.real(phases_rft), np.imag(phases_rft)])
        scatter_rft.set_array(theta_rft % (2*np.pi))
        
        # FFT phases
        theta_fft = 2.0 * np.pi * k / n + beta_offset
        phases_fft = np.exp(1j * theta_fft)
        scatter_fft.set_offsets(np.c_[np.real(phases_fft), np.imag(phases_fft)])
        scatter_fft.set_array(theta_fft % (2*np.pi))
        
        return scatter_rft, scatter_fft
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=50, blit=True)
    
    writer = PillowWriter(fps=20)
    anim.save(OUTPUT_DIR / 'phase_rotation.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/phase_rotation.gif")

def create_transform_spectrum_gif():
    """Animate spectrum building up"""
    print("2. Creating transform spectrum animation...")
    
    n = 128
    t = np.arange(n)
    x = np.sin(2 * np.pi * 5 * t / n) + 0.5 * np.sin(2 * np.pi * 12 * t / n)
    
    X_rft = rft_forward(x)
    X_fft = np.fft.fft(x, norm='ortho')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RFT Transform Evolution', fontsize=16, fontweight='bold')
    
    # Time domain
    line_time, = axes[0, 0].plot([], [], 'b-', linewidth=2)
    axes[0, 0].set_xlim(0, n)
    axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RFT spectrum
    line_rft, = axes[0, 1].plot([], [], 'r-', linewidth=1)
    axes[0, 1].set_xlim(0, n)
    axes[0, 1].set_ylim(0, 8)
    axes[0, 1].set_xlabel('Frequency Bin')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('RFT Spectrum')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FFT spectrum
    line_fft, = axes[1, 0].plot([], [], 'b-', linewidth=1)
    axes[1, 0].set_xlim(0, n)
    axes[1, 0].set_ylim(0, 8)
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('FFT Spectrum')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Complex plane
    scatter_phase = axes[1, 1].scatter([], [], c=[], cmap='twilight', s=30, alpha=0.6, vmin=0, vmax=2*np.pi)
    axes[1, 1].set_xlim(-8, 8)
    axes[1, 1].set_ylim(-8, 8)
    axes[1, 1].set_xlabel('Real')
    axes[1, 1].set_ylabel('Imaginary')
    axes[1, 1].set_title('RFT Complex Plane')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    def animate(frame):
        progress = int((frame / 60) * n)
        
        line_time.set_data(t[:progress], x[:progress])
        
        X_rft_show = X_rft.copy()
        X_rft_show[progress:] = 0
        line_rft.set_data(np.arange(n), np.abs(X_rft_show))
        
        X_fft_show = X_fft.copy()
        X_fft_show[progress:] = 0
        line_fft.set_data(np.arange(n), np.abs(X_fft_show))
        
        X_sig = X_rft_show[np.abs(X_rft_show) > 0.01]
        if len(X_sig) > 0:
            scatter_phase.set_offsets(np.c_[np.real(X_sig), np.imag(X_sig)])
            scatter_phase.set_array(np.angle(X_sig) % (2*np.pi))
        
        return line_time, line_rft, line_fft, scatter_phase
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=True)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'transform_evolution.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/transform_evolution.gif")

def create_signal_reconstruction_gif():
    """Animate signal reconstruction"""
    print("3. Creating signal reconstruction animation...")
    
    n = 128
    t = np.arange(n)
    x = np.sin(2 * np.pi * 5 * t / n) + 0.3 * np.sin(2 * np.pi * 15 * t / n)
    X_rft = rft_forward(x)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Signal Reconstruction via Inverse RFT', fontsize=16, fontweight='bold')
    
    # Frequency domain
    line_freq, = axes[0].plot(np.arange(n), np.abs(X_rft), 'gray', linewidth=1, alpha=0.3)
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, np.max(np.abs(X_rft)) * 1.1)
    axes[0].set_xlabel('Frequency Bin')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('RFT Spectrum (Selecting Coefficients)')
    axes[0].grid(True, alpha=0.3)
    
    # Create fill container
    fill_container = [None]
    
    # Time domain
    line_original, = axes[1].plot(t, x, 'gray', linewidth=1, alpha=0.5, label='Original')
    line_recon, = axes[1].plot([], [], 'r-', linewidth=2, label='Reconstructed')
    axes[1].set_xlim(0, n)
    axes[1].set_ylim(-2, 2)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Time Domain Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    def animate(frame):
        n_coeffs = int((frame / 60) * n)
        
        X_partial = X_rft.copy()
        X_partial[n_coeffs:] = 0
        x_recon = rft_inverse(X_partial)
        
        # Remove old fill
        if fill_container[0] is not None:
            fill_container[0].remove()
        
        # Add new fill
        if n_coeffs > 0:
            fill_container[0] = axes[0].fill_between(np.arange(n_coeffs), np.abs(X_rft[:n_coeffs]), 0, color='red', alpha=0.5)
        
        line_recon.set_data(t, np.real(x_recon))
        
        return line_recon,
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'signal_reconstruction.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/signal_reconstruction.gif")

def create_unitarity_demo_gif():
    """Animate unitarity demonstration"""
    print("4. Creating unitarity demonstration animation...")
    
    n = 128
    np.random.seed(42)
    signals = [np.random.randn(n) + 1j * np.random.randn(n) for _ in range(60)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RFT Unitarity: Energy Preservation', fontsize=16, fontweight='bold')
    
    scatter1 = axes[0, 0].scatter([], [], alpha=0.6, s=50, c=[])
    axes[0, 0].plot([0, 10], [0, 10], 'r--', linewidth=2, alpha=0.5)
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_xlabel('Time Domain Energy')
    axes[0, 0].set_ylabel('Frequency Domain Energy')
    axes[0, 0].set_title('Energy Conservation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    line_time, = axes[0, 1].plot([], [], 'b-', linewidth=2)
    axes[0, 1].set_xlim(0, n)
    axes[0, 1].set_ylim(-3, 3)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Time Domain Signal')
    axes[0, 1].grid(True, alpha=0.3)
    
    line_freq, = axes[1, 0].plot([], [], 'r-', linewidth=1)
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
    
    def animate(frame):
        x = signals[frame]
        X = rft_forward(x)
        
        e_time = np.sum(np.abs(x) ** 2)
        e_freq = np.sum(np.abs(X) ** 2)
        
        energies_time.append(e_time)
        energies_freq.append(e_freq)
        
        scatter1.set_offsets(np.c_[energies_time, energies_freq])
        scatter1.set_array(np.arange(len(energies_time)))
        
        line_time.set_data(np.arange(n), np.real(x))
        line_freq.set_data(np.arange(n), np.abs(X))
        
        bar_energy[0].set_height(e_time)
        bar_energy[1].set_height(e_freq)
        
        return scatter1, line_time, line_freq
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'unitarity_demo.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/unitarity_demo.gif")

def main():
    print("="*70)
    print(" RFT Animated GIF Generator")
    print("="*70)
    print()
    
    create_phase_rotation_gif()
    create_transform_spectrum_gif()
    create_signal_reconstruction_gif()
    create_unitarity_demo_gif()
    
    print()
    print("="*70)
    print(f" All GIFs saved to: {OUTPUT_DIR}/")
    print("="*70)
    print()
    print("Generated GIFs:")
    print("  1. phase_rotation.gif         - Golden ratio phase evolution")
    print("  2. transform_evolution.gif    - Signal transform animation")
    print("  3. signal_reconstruction.gif  - Inverse transform building signal")
    print("  4. unitarity_demo.gif         - Energy preservation verification")
    print()

if __name__ == "__main__":
    main()
