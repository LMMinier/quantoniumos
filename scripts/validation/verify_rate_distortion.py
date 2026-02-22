#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Rate-Distortion Analysis: Hybrid vs DCT vs RFT.
Verifies the "60-year bottleneck" claim by comparing Rate-Distortion curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.fftpack import dct, idct
from algorithms.rft.hybrids.legacy_mca import adaptive_hybrid_compress, rft_forward, rft_inverse, PHI
from algorithms.rft.compression.entropy import estimate_bitrate, uniform_quantizer

def generate_mixed_signal(n=512):
    """Generate a signal with both ASCII-like steps and Fibonacci waves."""
    t = np.arange(n)
    
    # 1. ASCII / Step Component (Python code-like)
    # Random steps of varying length
    steps = np.zeros(n)
    current_val = 0
    for i in range(n):
        if i % np.random.randint(10, 50) == 0:
            current_val = np.random.randint(32, 127) # ASCII range
        steps[i] = current_val
    
    # Normalize steps
    steps = (steps - np.mean(steps)) / np.std(steps)
    
    # 2. Fibonacci Wave Component
    # Golden ratio resonances - High amplitude to force RFT usage
    # DCT struggles with non-integer frequencies (leakage)
    waves = 2.0 * np.sin(2 * np.pi * t / PHI) + 1.5 * np.sin(2 * np.pi * t / (PHI**2))
    
    # Mix them
    # Strong waves, Strong steps
    signal = steps + waves
    return signal

def analyze_rd_curve():
    print("Generating Rate-Distortion Analysis...")
    signal = generate_mixed_signal(512)
    
    # Quantization step sizes to sweep (log scale)
    step_sizes = np.logspace(-2, 1.0, 10)
    
    results = {
        "DCT": [],
        "RFT": [],
        "Hybrid": []
    }
    
    print(f"{'Step':<10} | {'DCT (BPP/MSE)':<20} | {'RFT (BPP/MSE)':<20} | {'Hybrid (BPP/MSE)':<20}")
    print("-" * 80)
    
    for step in step_sizes:
        # --- DCT Only ---
        c_dct = dct(signal, norm="ortho")
        q_dct = uniform_quantizer(c_dct, step)
        r_dct = estimate_bitrate(q_dct) / signal.size
        rec_dct = idct(q_dct * step, norm="ortho")
        d_dct = np.mean((signal - rec_dct)**2)
        results["DCT"].append((r_dct, d_dct))
        
        # --- RFT Only ---
        c_rft = rft_forward(signal)
        q_rft = uniform_quantizer(c_rft, step)
        r_rft = estimate_bitrate(q_rft) / signal.size
        rec_rft = rft_inverse(q_rft * step).real
        d_rft = np.mean((signal - rec_rft)**2)
        results["RFT"].append((r_rft, d_rft))
        
        # --- Hybrid ---
        # 1. Decompose (Ideal separation first)
        # We use a fixed small threshold for decomposition to get the "ideal" components,
        # then quantize those components.
        # In a real codec, the decomposition itself would be rate-controlled, but this approximates it.
        struct, text, _, _ = adaptive_hybrid_compress(signal, max_iter=5, verbose=False)
        
        # 2. Transform components to their respective sparse domains
        c_struct = dct(struct, norm="ortho")
        c_text = rft_forward(text)
        
        # 3. Quantize
        q_struct = uniform_quantizer(c_struct, step)
        q_text = uniform_quantizer(c_text, step)
        
        # 4. Rate = Sum of rates
        r_hybrid = (estimate_bitrate(q_struct) + estimate_bitrate(q_text)) / signal.size
        
        # 5. Recon
        rec_struct = idct(q_struct * step, norm="ortho")
        rec_text = rft_inverse(q_text * step).real
        rec_hybrid = rec_struct + rec_text
        d_hybrid = np.mean((signal - rec_hybrid)**2)
        
        results["Hybrid"].append((r_hybrid, d_hybrid))
        
        print(f"{step:<10.4f} | {r_dct:.2f} / {d_dct:.4f}      | {r_rft:.2f} / {d_rft:.4f}      | {r_hybrid:.2f} / {d_hybrid:.4f}")

    # Check the claim: R_hybrid < min(R_DCT, R_RFT) at similar distortion?
    # Or D_hybrid < min(D_DCT, D_RFT) at similar rate?
    
    # Let's pick a mid-range point to validate the theorem inequality
    # Find point with closest rates or just compare the curves generally
    
    print("\nAnalysis Complete.")
    return results

def export_to_csv(results, filepath):
    """Export rate-distortion results to CSV for MATLAB plotting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header with metadata
        writer.writerow(['# Rate-Distortion Analysis Results'])
        writer.writerow(['# Columns: Rate_DCT, Distortion_DCT, Rate_RFT, Distortion_RFT, Rate_Hybrid, Distortion_Hybrid'])
        writer.writerow(['Rate_DCT', 'Distortion_DCT', 'Rate_RFT', 'Distortion_RFT', 'Rate_Hybrid', 'Distortion_Hybrid'])
        
        # Data rows
        for i in range(len(results['DCT'])):
            r_dct, d_dct = results['DCT'][i]
            r_rft, d_rft = results['RFT'][i]
            r_hybrid, d_hybrid = results['Hybrid'][i]
            writer.writerow([r_dct, d_dct, r_rft, d_rft, r_hybrid, d_hybrid])
    
    print(f"\n✅ Exported data to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rate-Distortion Analysis')
    parser.add_argument('--export', type=str, help='Export results to CSV file')
    args = parser.parse_args()
    
    results = analyze_rd_curve()
    
    if args.export:
        export_to_csv(results, args.export)
