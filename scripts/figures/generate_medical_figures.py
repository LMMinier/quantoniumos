#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate all figures for the Medical Validation Report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Colors
RFT_BLUE = '#3465a4'
FFT_RED = '#cc0000'
DCT_GREEN = '#4e9a06'
WAVELET_PURPLE = '#75507b'

# Output directory
OUTPUT_DIR = Path('/workspaces/quantoniumos/figures/medical')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def figure_ecg_snr_comparison():
    """Figure 1: ECG Compression SNR Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    retention = [0.3, 0.5, 0.7]
    rft_snr = [38.20, 51.47, 61.30]
    fft_snr = [21.53, 24.84, 27.78]
    
    ax.plot(retention, rft_snr, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=10, label='RFT', markeredgewidth=2, markeredgecolor='white')
    ax.plot(retention, fft_snr, 's-', color=FFT_RED, linewidth=3, 
            markersize=10, label='FFT', markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Coefficient Retention Ratio', fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontweight='bold')
    ax.set_title('ECG Compression Quality: RFT vs FFT', fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_ylim([15, 65])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ecg_snr_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: ecg_snr_comparison.png")

def figure_ecg_prd_comparison():
    """Figure 2: ECG PRD (Distortion) Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    retention = [0.3, 0.5, 0.7]
    rft_prd = [1.23, 0.27, 0.09]
    fft_prd = [8.39, 5.73, 4.08]
    
    ax.plot(retention, rft_prd, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=10, label='RFT', markeredgewidth=2, markeredgecolor='white')
    ax.plot(retention, fft_prd, 's-', color=FFT_RED, linewidth=3, 
            markersize=10, label='FFT', markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Coefficient Retention Ratio', fontweight='bold')
    ax.set_ylabel('PRD (%)', fontweight='bold')
    ax.set_title('ECG Distortion: RFT vs FFT (Lower is Better)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ecg_prd_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: ecg_prd_comparison.png")

def figure_eeg_snr_comparison():
    """Figure 3: EEG Compression SNR Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    retention = [0.3, 0.5]
    rft_snr = [28.49, 35.53]
    fft_snr = [25.88, 31.10]
    
    ax.plot(retention, rft_snr, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=10, label='RFT', markeredgewidth=2, markeredgecolor='white')
    ax.plot(retention, fft_snr, 's-', color=FFT_RED, linewidth=3, 
            markersize=10, label='FFT', markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Coefficient Retention Ratio', fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontweight='bold')
    ax.set_title('EEG Compression Quality: RFT vs FFT', fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.3, 0.5])
    ax.set_ylim([24, 38])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eeg_snr_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: eeg_snr_comparison.png")

def figure_battery_life():
    """Figure 4: Battery Life Projections"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    devices = ['STM32F4', 'ESP32', 'nRF52840', 'Pi Pico']
    battery_days = [56.2, 3.1, 97.5, 29.9]
    colors = [RFT_BLUE, FFT_RED, DCT_GREEN, WAVELET_PURPLE]
    
    bars = ax.bar(devices, battery_days, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} days',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Battery Life (days)', fontweight='bold')
    ax.set_title('Continuous ECG Monitoring Battery Life (2000 mAh)', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'battery_life.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: battery_life.png")

def figure_memory_footprint():
    """Figure 5: Memory Footprint by Signal Length"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    signal_lengths = [64, 128, 256, 512]
    memory_kb = [2.25, 4.50, 9.00, 18.00]
    
    ax.plot(signal_lengths, memory_kb, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=12, markeredgewidth=2, markeredgecolor='white')
    
    # Add value labels
    for x, y in zip(signal_lengths, memory_kb):
        ax.text(x, y + 0.5, f'{y:.2f} KB', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Signal Length (samples)', fontweight='bold')
    ax.set_ylabel('Memory Required (KB)', fontweight='bold')
    ax.set_title('RFT Memory Footprint', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, 530])
    ax.set_ylim([0, 20])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'memory_footprint.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: memory_footprint.png")

def figure_device_ram_usage():
    """Figure 6: RAM Usage by Device (256 samples)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    devices = ['STM32F4\n(128KB)', 'ESP32\n(320KB)', 'nRF52840\n(256KB)', 'Pi Pico\n(264KB)']
    ram_percent = [6.7, 2.5, 5.0, 3.4]
    colors = [RFT_BLUE, FFT_RED, DCT_GREEN, WAVELET_PURPLE]
    
    bars = ax.bar(devices, ram_percent, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('RAM Usage (%)', fontweight='bold')
    ax.set_title('RFT RAM Usage on Embedded Devices (256 samples)', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 9])
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='10% threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'device_ram_usage.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: device_ram_usage.png")

def figure_kmer_energy_comparison():
    """Figure 7: K-mer Transform Energy Compaction"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [3, 4, 5]
    rft_energy = [0.985, 0.957, 0.904]
    fft_energy = [0.996, 0.976, 0.916]
    dct_energy = [0.997, 0.979, 0.918]
    
    x = np.arange(len(k_values))
    width = 0.25
    
    ax.bar(x - width, rft_energy, width, label='RFT', color=RFT_BLUE, edgecolor='black', linewidth=1.5)
    ax.bar(x, fft_energy, width, label='FFT', color=FFT_RED, edgecolor='black', linewidth=1.5)
    ax.bar(x + width, dct_energy, width, label='DCT', color=DCT_GREEN, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('K-mer Size (k)', fontweight='bold')
    ax.set_ylabel('Top-10 Energy Compaction', fontweight='bold')
    ax.set_title('K-mer Transform Comparison (Higher is Better)', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['k=3', 'k=4', 'k=5'])
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kmer_energy_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: kmer_energy_comparison.png")

def figure_ct_denoising():
    """Figure 8: CT Denoising Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Noisy\nInput', 'RFT', 'DCT', 'Wavelet']
    psnr_values = [22.63, 14.69, 15.33, 23.89]
    colors = ['gray', RFT_BLUE, DCT_GREEN, WAVELET_PURPLE]
    
    bars = ax.bar(methods, psnr_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} dB',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('CT Low-Dose Denoising: Wavelet Dominates', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 27])
    
    # Highlight winner
    ax.axhline(y=22.63, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Original quality')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ct_denoising.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: ct_denoising.png")

def figure_mri_rician_noise():
    """Figure 9: MRI Rician Noise Denoising"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise_levels = [0.05, 0.10, 0.15]
    rft_psnr = [15.53, 14.85, 13.61]
    dct_psnr = [15.95, 14.73, 13.34]
    
    ax.plot(noise_levels, rft_psnr, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=10, label='RFT', markeredgewidth=2, markeredgecolor='white')
    ax.plot(noise_levels, dct_psnr, 's-', color=DCT_GREEN, linewidth=3, 
            markersize=10, label='DCT', markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Noise Level (σ)', fontweight='bold')
    ax.set_ylabel('Denoised PSNR (dB)', fontweight='bold')
    ax.set_title('MRI Rician Noise Denoising: RFT vs DCT', fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(noise_levels)
    ax.set_ylim([13, 16.5])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mri_rician_noise.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: mri_rician_noise.png")

def figure_byzantine_resilience():
    """Figure 10: Byzantine Attack Resilience"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    attack_rates = [0, 10, 20, 30]
    mean_error = [0.032, 4.698, 4.946, 12.685]
    median_error = [0.037, 0.029, 0.038, 0.039]
    rft_error = [0.032, 0.482, 1.075, 12.685]
    
    ax.plot(attack_rates, mean_error, 'x-', color=FFT_RED, linewidth=3, 
            markersize=12, label='Mean (Vulnerable)', markeredgewidth=3)
    ax.plot(attack_rates, median_error, 'o-', color=DCT_GREEN, linewidth=3, 
            markersize=10, label='Median (Robust)', markeredgewidth=2, markeredgecolor='white')
    ax.plot(attack_rates, rft_error, 's-', color=RFT_BLUE, linewidth=3, 
            markersize=10, label='RFT-Filter', markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('Malicious Client Percentage (%)', fontweight='bold')
    ax.set_ylabel('Aggregation Error', fontweight='bold')
    ax.set_title('Federated Learning: Byzantine Attack Resilience', fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(attack_rates)
    ax.set_ylim([0, 14])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'byzantine_resilience.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: byzantine_resilience.png")

def figure_processing_speed():
    """Figure 11: Processing Speed Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['FFT', 'DCT', 'RFT']
    ecg_times = [0.7, 0.6, 157.4]  # At 30% retention
    colors = [FFT_RED, DCT_GREEN, RFT_BLUE]
    
    bars = ax.bar(methods, ecg_times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Processing Time (ms)', fontweight='bold')
    ax.set_title('ECG Compression Speed (30% retention)', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'processing_speed.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: processing_speed.png")

def figure_contact_map_compression():
    """Figure 12: Contact Map Compression Quality"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    retention = [0.3, 0.5, 0.7]
    f1_scores = [0.995, 1.000, 1.000]
    
    ax.plot(retention, f1_scores, 'o-', color=RFT_BLUE, linewidth=3, 
            markersize=12, markeredgewidth=2, markeredgecolor='white')
    
    # Add value labels
    for x, y in zip(retention, f1_scores):
        ax.text(x, y - 0.001, f'{y:.3f}', ha='center', va='top', fontweight='bold')
    
    ax.set_xlabel('Coefficient Retention Ratio', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Protein Contact Map Compression (RFT)', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_ylim([0.99, 1.001])
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect reconstruction')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'contact_map_compression.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: contact_map_compression.png")

def figure_clinical_feature_preservation():
    """Figure 13: Clinical Feature Preservation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Arrhythmia detection
    methods = ['Original', 'RFT\nCompressed']
    f1_scores = [0.819, 0.819]
    sensitivity = [0.729, 0.729]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, f1_scores, width, label='F1 Score', 
                    color=RFT_BLUE, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, sensitivity, width, label='Sensitivity', 
                    color=DCT_GREEN, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('ECG Arrhythmia Detection Preserved', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.7, 0.85])
    
    # Seizure detection
    methods2 = ['Original', 'RFT\nCompressed']
    seizure_f1 = [0.615, 0.615]
    seizure_sens = [0.444, 0.444]
    
    bars3 = ax2.bar(x - width/2, seizure_f1, width, label='F1 Score', 
                    color=RFT_BLUE, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, seizure_sens, width, label='Sensitivity', 
                    color=WAVELET_PURPLE, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('EEG Seizure Detection Preserved', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods2)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.4, 0.7])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clinical_feature_preservation.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: clinical_feature_preservation.png")

def figure_domain_summary():
    """Figure 14: RFT Performance by Domain"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    domains = ['ECG\nCompression', 'EEG\nCompression', 'Edge\nDevices', 
               'CT\nDenoising', 'K-mer\nAnalysis', 'Contact\nMaps', 
               'Crypto\nHash', 'Byzantine\nResilience']
    
    # Win/Loss/Mixed (as percentages)
    wins = [100, 100, 100, 0, 0, 100, 100, 80]  # Percentage favorable
    mixed = [0, 0, 0, 0, 0, 0, 0, 20]
    losses = [0, 0, 0, 100, 100, 0, 0, 0]
    
    x = np.arange(len(domains))
    width = 0.6
    
    p1 = ax.bar(x, wins, width, label='RFT Wins', color=DCT_GREEN, edgecolor='black', linewidth=1.5)
    p2 = ax.bar(x, mixed, width, bottom=wins, label='Mixed/Competitive', 
                color='orange', edgecolor='black', linewidth=1.5)
    p3 = ax.bar(x, losses, width, bottom=np.array(wins) + np.array(mixed), 
                label='RFT Loses', color=FFT_RED, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Performance Category (%)', fontweight='bold')
    ax.set_title('RFT Performance Summary Across Medical Domains', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_summary.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: domain_summary.png")

def main():
    print("Generating Medical Validation Figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    figure_ecg_snr_comparison()
    figure_ecg_prd_comparison()
    figure_eeg_snr_comparison()
    figure_battery_life()
    figure_memory_footprint()
    figure_device_ram_usage()
    figure_kmer_energy_comparison()
    figure_ct_denoising()
    figure_mri_rician_noise()
    figure_byzantine_resilience()
    figure_processing_speed()
    figure_contact_map_compression()
    figure_clinical_feature_preservation()
    figure_domain_summary()
    
    print()
    print(f"✓ All 14 figures generated successfully!")
    print(f"✓ Location: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
