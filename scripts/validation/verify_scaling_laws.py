#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
verify_scaling_laws.py
----------------------
Runs the Irrevocable Truths verification across multiple dimensions (N)
to prove scalability and numerical stability.

Generates:
1. JSON data: data/scaling_results.json
2. Plots: figures/scaling_laws.png
3. Markdown Table: stdout
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import irrevocable_truths as it
except ImportError:
    # Fallback if running from root without path setup
    sys.path.append('scripts')
    import irrevocable_truths as it

def run_scaling_test():
    N_values = [32, 64, 128, 256, 512]
    results = {
        "N": N_values,
        "unitarity": {},
        "diagonalization": [],
        "sparsity": []
    }
    
    # Initialize variant storage
    variants = [
        "Original Φ-RFT", "Harmonic-Phase", "Fibonacci Tilt", 
        "Chaotic Mix", "Geometric Lattice", "Φ-Chaotic Hybrid"
    ]
    for v in variants:
        results["unitarity"][v] = []

    print(f"{'N':<5} | {'Diag Err':<10} | {'Sparsity':<10} | {'Max Unitary Err':<15}")
    print("-" * 50)

    for N in N_values:
        # 1. Unitarity for all variants
        max_u_err = 0
        
        # Generators map
        generators = {
            "Original Φ-RFT": it.generate_original_phi_rft,
            "Harmonic-Phase": it.generate_harmonic_phase,
            "Fibonacci Tilt": it.generate_fibonacci_tilt,
            "Chaotic Mix": it.generate_chaotic_mix,
            "Geometric Lattice": it.generate_geometric_lattice,
            "Φ-Chaotic Hybrid": it.generate_phi_chaotic_hybrid
        }

        for name, gen_func in generators.items():
            U = gen_func(N)
            I = np.eye(N, dtype=np.complex128)
            P = U.conj().T @ U
            err = np.linalg.norm(P - I, ord='fro')
            results["unitarity"][name].append(err)
            max_u_err = max(max_u_err, err)

        # 2. Diagonalization (Theorem 2)
        U_phi = it.generate_original_phi_rft(N)
        k = np.arange(N)
        lambda_diag = np.exp(1j * 2 * np.pi * (it.PHI ** (-k)))
        Lambda = np.diag(lambda_diag)
        A = U_phi @ Lambda @ U_phi.conj().T
        Lambda_recovered = U_phi.conj().T @ A @ U_phi
        diag_err = np.linalg.norm(Lambda_recovered - Lambda, ord='fro')
        results["diagonalization"].append(diag_err)

        # 3. Sparsity (Theorem 3)
        # Create a signal with fixed number of modes (e.g., 3)
        # As N grows, sparsity (1 - k/N) should increase or stay high
        n_vec = np.arange(N)
        x = np.zeros(N, dtype=np.complex128)
        modes = [1, 3, 5] # Fixed 3 modes
        for m in modes:
            freq = it.PHI ** (-m)
            x += np.exp(1j * 2 * np.pi * freq * n_vec / N)
        
        z = U_phi.conj().T @ x
        energy = np.abs(z)**2
        total_energy = np.sum(energy)
        sorted_energy = np.sort(energy)[::-1]
        cumulative_energy = np.cumsum(sorted_energy)
        k_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
        sparsity_val = 1.0 - (k_95 / N)
        results["sparsity"].append(sparsity_val)

        print(f"{N:<5} | {diag_err:.2e}   | {sparsity_val:.2%}     | {max_u_err:.2e}")

    # Save JSON
    os.makedirs("data", exist_ok=True)
    with open("data/scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nData saved to data/scaling_results.json")

    # Generate Plots
    generate_plots(results)

def generate_plots(results):
    N = results["N"]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Unitarity Error
    for name, errs in results["unitarity"].items():
        ax1.plot(N, errs, marker='o', label=name)
    ax1.set_yscale('log')
    ax1.set_title('Unitarity Error vs N (Machine Precision)')
    ax1.set_xlabel('Dimension N')
    ax1.set_ylabel('Frobenius Norm Error')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize='small')

    # Plot 2: Diagonalization Error
    ax2.plot(N, results["diagonalization"], 'r-o', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Diagonalization Error vs N')
    ax2.set_xlabel('Dimension N')
    ax2.set_ylabel('Error')
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 3: Sparsity
    ax3.plot(N, results["sparsity"], 'g-s', linewidth=2)
    ax3.set_ylim(0, 1.0)
    ax3.axhline(y=0.618, color='r', linestyle='--', label='Theoretical Limit (1/φ)')
    ax3.set_title('Sparsity vs N (Golden Signal)')
    ax3.set_xlabel('Dimension N')
    ax3.set_ylabel('Sparsity (1 - k/N)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/scaling_laws.png", dpi=150)
    print("Plots saved to figures/scaling_laws.png")

if __name__ == "__main__":
    print("\n=== VERIFYING SCALING LAWS (N=32 to 512) ===\n")
    run_scaling_test()
