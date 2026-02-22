#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Generate Detailed Hardware/Implementation GIFs for RFT
Shows actual test results, matrix operations, and real computations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path
import time

from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix, rft_phase_vectors_canonical as rft_phase_vectors, PHI
)

OUTPUT_DIR = Path("./figures/gifs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating detailed hardware/implementation GIFs...")
print()

def create_actual_matrix_multiplication_gif():
    """Show actual matrix multiplication happening"""
    print("1. Creating actual matrix multiplication visualization...")
    
    n = 16  # Small enough to see details
    
    # Generate actual RFT matrix
    Psi = rft_matrix(n)
    
    # Test signal
    x = np.zeros(n)
    x[3] = 1.0  # Impulse at position 3
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Actual RFT Matrix Multiplication: Y = Ψ·x', fontsize=16, fontweight='bold')
    
    # Input vector
    im1 = axes[0, 0].imshow(x.reshape(-1, 1), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('Input Vector x')
    axes[0, 0].set_ylabel('Element')
    axes[0, 0].set_xticks([])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RFT Matrix
    im2 = axes[0, 1].imshow(np.abs(Psi), cmap='hot', aspect='auto')
    axes[0, 1].set_title('RFT Matrix |Ψ| (16×16)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Output vector (will animate)
    Y = Psi @ x
    im3 = axes[0, 2].imshow(np.abs(Y).reshape(-1, 1), cmap='plasma', aspect='auto', vmin=0, vmax=np.max(np.abs(Y)))
    axes[0, 2].set_title('Output Vector |Y|')
    axes[0, 2].set_ylabel('Element')
    axes[0, 2].set_xticks([])
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row being computed
    im4 = axes[1, 0].imshow(np.zeros((1, n)), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Current Row Being Computed')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_yticks([])
    
    # Intermediate computation
    line1, = axes[1, 1].plot([], [], 'b-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlim(0, n)
    axes[1, 1].set_ylim(-0.3, 0.3)
    axes[1, 1].set_xlabel('Element')
    axes[1, 1].set_ylabel('Real Part')
    axes[1, 1].set_title('Row·Vector Product (Real)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Output building up
    line2, = axes[1, 2].plot([], [], 'ro-', linewidth=2, markersize=6)
    axes[1, 2].set_xlim(0, n)
    axes[1, 2].set_ylim(0, np.max(np.abs(Y)) * 1.1)
    axes[1, 2].set_xlabel('Output Index')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].set_title('Output Vector (Building)')
    axes[1, 2].grid(True, alpha=0.3)
    
    text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        row_idx = frame % n
        
        # Show current row
        current_row = np.abs(Psi[row_idx, :])
        im4.set_data(current_row.reshape(1, -1))
        
        # Show row * vector computation
        row_vec_prod = Psi[row_idx, :] * x
        line1.set_data(np.arange(n), np.real(row_vec_prod))
        
        # Show output building
        Y_partial = np.zeros(n, dtype=complex)
        for i in range(row_idx + 1):
            Y_partial[i] = np.dot(Psi[i, :], x)
        line2.set_data(np.arange(n), np.abs(Y_partial))
        
        # Highlight current row in output
        im3_data = np.abs(Y_partial).reshape(-1, 1)
        im3.set_data(im3_data)
        
        # Update text
        y_val = Y_partial[row_idx]
        text.set_text(f'Computing row {row_idx}: Y[{row_idx}] = {np.abs(y_val):.4f}∠{np.angle(y_val):.2f}rad')
        
        return im4, line1, line2, im3, text
    
    anim = animation.FuncAnimation(fig, animate, frames=n*2, interval=300, blit=False)
    
    writer = PillowWriter(fps=3)
    anim.save(OUTPUT_DIR / 'actual_matrix_multiply.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/actual_matrix_multiply.gif")

def create_live_unitarity_test_gif():
    """Show actual unitarity test running in real-time"""
    print("2. Creating live unitarity test visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Live RFT Unitarity Test (Running Now)', fontsize=16, fontweight='bold')
    
    # Test signal display
    line_signal, = axes[0, 0].plot([], [], 'b-', linewidth=2)
    axes[0, 0].set_xlim(0, 128)
    axes[0, 0].set_ylim(-3, 3)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Test Signal (Random)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Transform spectrum
    line_spectrum, = axes[0, 1].plot([], [], 'r-', linewidth=1)
    axes[0, 1].set_xlim(0, 128)
    axes[0, 1].set_ylim(0, 3)
    axes[0, 1].set_xlabel('Frequency Bin')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('RFT Spectrum')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstructed signal
    line_recon, = axes[1, 0].plot([], [], 'g-', linewidth=2, alpha=0.7, label='Reconstructed')
    line_orig, = axes[1, 0].plot([], [], 'b--', linewidth=1, alpha=0.5, label='Original')
    axes[1, 0].set_xlim(0, 128)
    axes[1, 0].set_ylim(-3, 3)
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Inverse Transform (Reconstruction)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error plot
    line_error, = axes[1, 1].plot([], [], 'purple', linewidth=2)
    axes[1, 1].set_xlim(0, 60)
    axes[1, 1].set_ylim(1e-18, 1e-14)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Test Number')
    axes[1, 1].set_ylabel('Reconstruction Error')
    axes[1, 1].set_title('Unitarity Error Over Time')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    axes[1, 1].axhline(y=1e-15, color='r', linestyle='--', alpha=0.5, label='Machine ε')
    axes[1, 1].legend()
    
    text = fig.text(0.5, 0.02, '', ha='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    errors = []
    np.random.seed(42)
    
    def animate(frame):
        n = 128
        
        # Generate random test signal
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # Time it
        t_start = time.perf_counter()
        
        # Forward transform
        X = rft_forward(x)
        
        # Inverse transform
        x_recon = rft_inverse(X)
        
        # Compute error
        error = np.linalg.norm(x_recon - x) / np.linalg.norm(x)
        errors.append(error)
        
        t_elapsed = (time.perf_counter() - t_start) * 1000
        
        # Update plots
        line_signal.set_data(np.arange(n), np.real(x))
        line_spectrum.set_data(np.arange(n), np.abs(X))
        line_recon.set_data(np.arange(n), np.real(x_recon))
        line_orig.set_data(np.arange(n), np.real(x))
        
        if len(errors) > 1:
            line_error.set_data(np.arange(len(errors)), errors)
        
        # Status text
        status = "✓ PASS" if error < 1e-10 else "✗ FAIL"
        text.set_text(f'Test #{frame+1}/60 | Error: {error:.2e} | Time: {t_elapsed:.3f}ms | {status}')
        
        return line_signal, line_spectrum, line_recon, line_orig, line_error, text
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=200, blit=False)
    
    writer = PillowWriter(fps=5)
    anim.save(OUTPUT_DIR / 'live_unitarity_test.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/live_unitarity_test.gif")

def create_phase_vectors_computation_gif():
    """Show how phase vectors are actually computed"""
    print("3. Creating phase vector computation visualization...")
    
    n = 64
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('RFT Phase Vector Computation (Real Implementation)', fontsize=16, fontweight='bold')
    
    # Golden ratio computation
    k = np.arange(n)
    
    # Step 1: k/φ
    line1, = axes[0, 0].plot([], [], 'b-', linewidth=2)
    axes[0, 0].set_xlim(0, n)
    axes[0, 0].set_ylim(0, 40)
    axes[0, 0].set_xlabel('Index k')
    axes[0, 0].set_ylabel('k / φ')
    axes[0, 0].set_title('Step 1: Divide by Golden Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=PHI, color='gold', linestyle='--', alpha=0.5, label=f'φ={PHI:.4f}')
    axes[0, 0].legend()
    
    # Step 2: frac(k/φ)
    line2, = axes[0, 1].plot([], [], 'r-', linewidth=2)
    axes[0, 1].set_xlim(0, n)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel('Index k')
    axes[0, 1].set_ylabel('frac(k / φ)')
    axes[0, 1].set_title('Step 2: Fractional Part (mod 1)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Step 3: θ = 2π·frac(k/φ)
    line3, = axes[1, 0].plot([], [], 'g-', linewidth=2)
    axes[1, 0].set_xlim(0, n)
    axes[1, 0].set_ylim(0, 2*np.pi)
    axes[1, 0].set_xlabel('Index k')
    axes[1, 0].set_ylabel('θ (radians)')
    axes[1, 0].set_title('Step 3: Phase Angle θ = 2πβ·frac(k/φ)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=2*np.pi, color='k', linestyle='--', alpha=0.3)
    
    # Step 4: Complex exponential
    scatter_exp = axes[1, 1].scatter([], [], c=[], cmap='twilight', s=50, alpha=0.7, vmin=0, vmax=2*np.pi)
    axes[1, 1].set_xlim(-1.2, 1.2)
    axes[1, 1].set_ylim(-1.2, 1.2)
    axes[1, 1].set_xlabel('Real')
    axes[1, 1].set_ylabel('Imaginary')
    axes[1, 1].set_title('Step 4: D_φ = exp(iθ)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].add_patch(circle)
    
    # Chirp phase
    line4, = axes[2, 0].plot([], [], 'm-', linewidth=2)
    axes[2, 0].set_xlim(0, n)
    axes[2, 0].set_ylim(0, 2*np.pi)
    axes[2, 0].set_xlabel('Index k')
    axes[2, 0].set_ylabel('φ (radians)')
    axes[2, 0].set_title('Chirp Phase: φ = πσk²/N')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Final combined phase
    scatter_final = axes[2, 1].scatter([], [], c=[], cmap='hsv', s=50, alpha=0.7, vmin=0, vmax=2*np.pi)
    axes[2, 1].set_xlim(-1.2, 1.2)
    axes[2, 1].set_ylim(-1.2, 1.2)
    axes[2, 1].set_xlabel('Real')
    axes[2, 1].set_ylabel('Imaginary')
    axes[2, 1].set_title('Final: D_φ ⊙ C_σ (Element-wise product)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_aspect('equal')
    circle2 = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    axes[2, 1].add_patch(circle2)
    
    text = fig.text(0.5, 0.01, '', ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8))
    
    def animate(frame):
        progress = int((frame / 60) * n)
        
        if progress > 0:
            k_show = k[:progress]
            
            # Step 1
            val1 = k_show / PHI
            line1.set_data(k_show, val1)
            
            # Step 2
            val2 = np.modf(val1)[0]
            val2 = np.where(val2 < 0, val2 + 1, val2)
            line2.set_data(k_show, val2)
            
            # Step 3
            theta = 2 * np.pi * val2
            line3.set_data(k_show, theta)
            
            # Step 4
            D_phi = np.exp(1j * theta)
            scatter_exp.set_offsets(np.c_[np.real(D_phi), np.imag(D_phi)])
            scatter_exp.set_array(theta)
            
            # Chirp
            chirp_phase = np.pi * (k_show ** 2) / n
            line4.set_data(k_show, chirp_phase % (2*np.pi))
            
            # Combined
            C_sig = np.exp(1j * chirp_phase)
            combined = D_phi * C_sig
            scatter_final.set_offsets(np.c_[np.real(combined), np.imag(combined)])
            scatter_final.set_array(np.angle(combined) % (2*np.pi))
            
            text.set_text(f'Computing phase vectors: {progress}/{n} elements | φ = {PHI:.6f}')
        
        return line1, line2, line3, scatter_exp, line4, scatter_final, text
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    anim.save(OUTPUT_DIR / 'phase_computation.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/phase_computation.gif")

def create_performance_benchmark_live_gif():
    """Show actual performance benchmark running"""
    print("4. Creating live performance benchmark visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Live Performance Benchmark: RFT vs FFT', fontsize=16, fontweight='bold')
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    rft_times = []
    fft_times = []
    
    # Benchmark progress
    bars = axes[0, 0].bar([0, 1], [0, 0], color=['red', 'blue'], alpha=0.7, width=0.6)
    axes[0, 0].set_ylim(0, 2)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['RFT', 'FFT'])
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Current Test (Running...)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative results
    line_rft, = axes[0, 1].plot([], [], 'ro-', linewidth=2, markersize=8, label='RFT')
    line_fft, = axes[0, 1].plot([], [], 'bo-', linewidth=2, markersize=8, label='FFT')
    axes[0, 1].set_xlabel('Transform Size')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_title('Benchmark Results')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_yscale('log')
    
    # Speedup ratio
    line_ratio, = axes[1, 0].plot([], [], 'go-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Parity')
    axes[1, 0].set_xlabel('Transform Size')
    axes[1, 0].set_ylabel('RFT / FFT Ratio')
    axes[1, 0].set_title('Performance Overhead')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)
    
    # Summary table
    axes[1, 1].axis('off')
    table_data = []
    table = axes[1, 1].table(cellText=[['', '', '']], 
                            colLabels=['Size', 'RFT (ms)', 'FFT (ms)'],
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    def animate(frame):
        if frame >= len(sizes):
            if len(rft_times) > 0:
                text.set_text(f'✓ Benchmark Complete! RFT is {np.mean(np.array(rft_times)/np.array(fft_times)):.1f}× slower than FFT')
            return line_rft, line_fft, line_ratio, text
        
        size_idx = frame
        n = sizes[size_idx]
        
        # Run actual benchmarks
        np.random.seed(42)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # RFT
        start = time.perf_counter()
        for _ in range(20):
            _ = rft_forward(x)
        t_rft = (time.perf_counter() - start) * 1000 / 20
        rft_times.append(t_rft)
        
        # FFT
        start = time.perf_counter()
        for _ in range(20):
            _ = np.fft.fft(x, norm='ortho')
        t_fft = (time.perf_counter() - start) * 1000 / 20
        fft_times.append(t_fft)
        
        # Update current test bars
        bars[0].set_height(t_rft)
        bars[1].set_height(t_fft)
        axes[0, 0].set_ylim(0, max(t_rft, t_fft) * 1.2)
        
        # Update cumulative plot - ensure arrays match
        sizes_so_far = np.array(sizes[:len(rft_times)])
        rft_array = np.array(rft_times)
        fft_array = np.array(fft_times)
        
        line_rft.set_data(sizes_so_far, rft_array)
        line_fft.set_data(sizes_so_far, fft_array)
        
        y_max = max(max(rft_times), max(fft_times)) * 1.5
        axes[0, 1].set_ylim(0.001, y_max)
        axes[0, 1].set_xlim(sizes[0] * 0.8, sizes[-1] * 1.2)
        
        # Update ratio
        if len(rft_times) > 0:
            ratios = rft_array / fft_array
            line_ratio.set_data(sizes_so_far, ratios)
            axes[1, 0].set_ylim(0, max(ratios) * 1.2)
            axes[1, 0].set_xlim(sizes[0] * 0.8, sizes[-1] * 1.2)
        
        # Update table
        table_data.append([f'{n}', f'{t_rft:.4f}', f'{t_fft:.4f}'])
        axes[1, 1].clear()
        axes[1, 1].axis('off')
        new_table = axes[1, 1].table(cellText=table_data[-5:],  # Last 5 rows
                                     colLabels=['Size', 'RFT (ms)', 'FFT (ms)'],
                                     cellLoc='center', loc='center')
        new_table.auto_set_font_size(False)
        new_table.set_fontsize(9)
        new_table.scale(1, 2)
        
        ratio = t_rft / t_fft
        text.set_text(f'Testing N={n} | RFT: {t_rft:.4f}ms | FFT: {t_fft:.4f}ms | Ratio: {ratio:.2f}×')
        
        return line_rft, line_fft, line_ratio, text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(sizes), interval=1000, blit=False, repeat=True)
    
    writer = PillowWriter(fps=1)
    anim.save(OUTPUT_DIR / 'live_benchmark.gif', writer=writer)
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/live_benchmark.gif")

def main():
    print("="*70)
    print(" RFT Detailed Hardware/Implementation GIF Generator")
    print("="*70)
    print()
    
    create_actual_matrix_multiplication_gif()
    create_live_unitarity_test_gif()
    create_phase_vectors_computation_gif()
    # Skip benchmark - has animation issues with table updates
    # create_performance_benchmark_live_gif()
    
    print()
    print("="*70)
    print(f" All detailed GIFs saved to: {OUTPUT_DIR}/")
    print("="*70)
    print()
    print("Generated GIFs:")
    print("  1. actual_matrix_multiply.gif - Real matrix operations step-by-step")
    print("  2. live_unitarity_test.gif    - Actual tests running in real-time  ")
    print("  3. phase_computation.gif      - Phase vector calculation process")
    print()
    print("These show the ACTUAL implementation running with real data!")
    print()

if __name__ == "__main__":
    main()
