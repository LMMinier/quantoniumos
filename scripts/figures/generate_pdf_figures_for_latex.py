# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# 1. Plot Rate Distortion
try:
    # Skip the first 2 lines (comments)
    df_rd = pd.read_csv('figures/latex_data/rate_distortion.csv', skiprows=2)
    
    # Clean up complex number strings if present
    def parse_complex(s):
        if isinstance(s, str):
            return abs(complex(s.replace('(', '').replace(')', '')))
        return s

    # Apply parsing to Distortion columns if they are object type
    if df_rd['Distortion_Hybrid'].dtype == object:
        df_rd['Distortion_Hybrid'] = df_rd['Distortion_Hybrid'].apply(parse_complex)

    plt.figure(figsize=(8, 6))
    # Use correct column names from the CSV
    plt.semilogy(df_rd['Rate_RFT'], df_rd['Distortion_RFT'], 's--', label='RFT Only', linewidth=2)
    plt.semilogy(df_rd['Rate_DCT'], df_rd['Distortion_DCT'], 'o-', label='DCT Only', linewidth=2)
    plt.semilogy(df_rd['Rate_Hybrid'], df_rd['Distortion_Hybrid'], '^-', label='Hybrid (Theorem 10)', linewidth=2)
    
    plt.xlabel('Rate (Bits Per Pixel)')
    plt.ylabel('Distortion (MSE)')
    plt.title('Rate-Distortion: Hybrid vs Single Basis')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/rft_rate_distortion_matlab.pdf')
    print("Generated figures/rft_rate_distortion_matlab.pdf")
except Exception as e:
    print(f"Error plotting Rate Distortion: {e}")

# 2. Plot Wave Computer
try:
    # Skip the first 2 lines (comments)
    df_wave = pd.read_csv('figures/latex_data/wave_computer.csv', skiprows=2)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(df_wave['Modes'], df_wave['RFT_MSE'], 's--', label='Graph RFT', linewidth=2)
    plt.semilogy(df_wave['Modes'], df_wave['FFT_MSE'], 'o-', label='FFT', linewidth=2)
    plt.xlabel('Number of Modes')
    plt.ylabel('Reconstruction MSE')
    plt.title('Wave Computer Benchmark')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/rft_wave_computer_matlab.pdf')
    print("Generated figures/rft_wave_computer_matlab.pdf")
except Exception as e:
    print(f"Error plotting Wave Computer: {e}")

# 3. Plot Unitarity Errors
try:
    df_unit = pd.read_csv('figures/latex_data/unitarity_errors.csv')
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(df_unit['Variant'], df_unit['Unitarity_Error'], color='skyblue', edgecolor='black')
    
    plt.yscale('log')
    plt.ylabel('Unitarity Error (Frobenius Norm)')
    plt.title('Unitarity Verification for 7 RFT Variants')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{height:.1e}',
                ha='center', va='bottom', rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/rft_unitarity_errors.pdf')
    print("Generated figures/rft_unitarity_errors.pdf")
except Exception as e:
    print(f"Error plotting Unitarity Errors: {e}")
