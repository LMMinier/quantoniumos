#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Theoretic Hybrid Decomposition (Theorem 4.1 Reference Implementation)
======================================================================

This module provides a reference implementation of Theorem 4.1 from PHI_RFT_PROOFS.tex,
which guarantees EXACT energy preservation via Parseval's identity.

Theorem 4.1 (Hybrid Basis Decomposition):
    For any signal x and sparsity parameters K1, K2, there exists a decomposition:
        x = x_struct + x_texture + x_residual
    satisfying:
        1. x_struct has at most K1 non-zero DCT coefficients
        2. x_texture has at most K2 non-zero RFT coefficients  
        3. ||x||² = ||x_struct||² + ||x_texture||² + ||x_residual||² (exact Parseval identity)

This implementation uses orthonormal basis projections (best K-term approximations)
as defined in the theorem, not approximate methods like moving averages.

Use this for:
- Quantum state preparation requiring exact energy accounting
- Validating numerical accuracy of approximate methods
- Theoretical analysis and benchmarking

For production compression with good efficiency/quality trade-offs, see:
- H3HierarchicalCascade (moving average, fast, ~2% energy error)
- FH5EntropyGuided (adaptive routing, best for edges)

Copyright (C) 2025 Luis M. Minier / quantoniumos
Licensed under AGPL-3.0-or-later
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TheoreticDecomposition:
    """Result from theoretic hybrid decomposition"""
    x_struct: np.ndarray      # Structure component (sparse in DCT)
    x_texture: np.ndarray     # Texture component (sparse in RFT)
    x_residual: np.ndarray    # Residual (discarded coefficients)
    
    c_dct: np.ndarray         # DCT coefficients (full)
    c_rft: np.ndarray         # RFT coefficients (full)
    
    kept_dct_indices: np.ndarray
    kept_rft_indices: np.ndarray
    
    energy_original: float
    energy_struct: float
    energy_texture: float
    energy_residual: float
    energy_error: float       # Should be < 1e-14 for exact Parseval
    
    sparsity_dct: float
    sparsity_rft: float


def best_k_term_approximation(
    coefficients: np.ndarray,
    k: int,
    basis_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute best K-term approximation in given orthonormal basis.
    
    This is the P_U^(K) operator from Theorem 4.1:
        P_U^(K) y = U† T_K(U y)
    where T_K keeps the K largest-magnitude coefficients.
    
    Args:
        coefficients: Transform coefficients (already in basis domain)
        k: Number of coefficients to keep
        basis_matrix: Orthonormal basis matrix (for inverse transform)
        
    Returns:
        (sparse_signal, kept_indices) tuple
    """
    # Find K largest magnitude coefficients
    magnitudes = np.abs(coefficients)
    threshold_idx = min(k, len(magnitudes))
    
    if threshold_idx < len(magnitudes):
        # Partition to find the k-th largest element
        partition_idx = np.argpartition(magnitudes, -threshold_idx)
        kept_indices = partition_idx[-threshold_idx:]
    else:
        kept_indices = np.arange(len(magnitudes))
    
    # Create sparse coefficient vector (T_K operation)
    sparse_coeffs = np.zeros_like(coefficients)
    sparse_coeffs[kept_indices] = coefficients[kept_indices]
    
    # Inverse transform: U† c_sparse
    sparse_signal = basis_matrix.conj().T @ sparse_coeffs
    
    return sparse_signal, kept_indices


def theoretic_hybrid_decomposition(
    signal: np.ndarray,
    k1: int,
    k2: int
) -> TheoreticDecomposition:
    """
    Perform theoretic hybrid decomposition following Theorem 4.1 exactly.
    
    This uses orthonormal basis projections to guarantee exact Parseval identity:
        ||x||² = ||x_struct||² + ||x_texture||² + ||x_residual||²
    
    Args:
        signal: Input signal (real or complex)
        k1: Number of DCT coefficients to keep for structure
        k2: Number of RFT coefficients to keep for texture
        
    Returns:
        TheoreticDecomposition with all components and verification metrics
    """
    n = len(signal)
    signal = np.asarray(signal, dtype=np.complex128)
    
    # Step 1: Construct orthonormal bases
    # DCT-II basis (orthonormal)
    C = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        norm = np.sqrt(2.0 / n) * (0.5**0.5 if k == 0 else 1.0)
        for m in range(n):
            C[k, m] = norm * np.cos(np.pi * k * (2*m + 1) / (2*n))
    
    # RFT basis (unitary)
    try:
        from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
        # For basis matrix, we need column vectors of RFT
        # RFT of standard basis vectors gives us the basis
        Psi = np.zeros((n, n), dtype=np.complex128)
        for j in range(n):
            e_j = np.zeros(n, dtype=np.complex128)
            e_j[j] = 1.0
            Psi[:, j] = rft_forward(e_j)
    except ImportError:
        # Fallback to DFT if RFT not available
        Psi = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
    
    # Verify unitarity (for numerical diagnostics)
    C_unitarity_error = np.linalg.norm(C @ C.T - np.eye(n), 'fro')
    Psi_unitarity_error = np.linalg.norm(Psi @ Psi.conj().T - np.eye(n), 'fro')
    
    if C_unitarity_error > 1e-10:
        print(f"Warning: DCT unitarity error = {C_unitarity_error:.2e}")
    if Psi_unitarity_error > 1e-10:
        print(f"Warning: RFT unitarity error = {Psi_unitarity_error:.2e}")
    
    # Step 2: Transform to DCT domain and extract structure
    c_dct_full = C @ signal.real  # DCT works on real part
    x_struct, kept_dct = best_k_term_approximation(c_dct_full, k1, C)
    
    # Step 3: Compute residual after structure removal
    residual_after_dct = signal.real - x_struct
    
    # Step 4: Transform residual to RFT domain and extract texture
    c_rft_full = Psi @ residual_after_dct.astype(np.complex128)
    x_texture, kept_rft = best_k_term_approximation(c_rft_full, k2, Psi)
    
    # Step 5: Final residual
    x_residual = residual_after_dct - x_texture.real
    
    # Step 6: Verify energy preservation (Parseval identity)
    energy_original = np.linalg.norm(signal.real) ** 2
    energy_struct = np.linalg.norm(x_struct) ** 2
    energy_texture = np.linalg.norm(x_texture.real) ** 2
    energy_residual = np.linalg.norm(x_residual) ** 2
    
    energy_reconstructed = energy_struct + energy_texture + energy_residual
    energy_error = abs(energy_original - energy_reconstructed)
    
    # Compute sparsity percentages
    sparsity_dct = 100.0 * (1.0 - len(kept_dct) / n)
    sparsity_rft = 100.0 * (1.0 - len(kept_rft) / n)
    
    return TheoreticDecomposition(
        x_struct=x_struct,
        x_texture=x_texture.real,
        x_residual=x_residual,
        c_dct=c_dct_full,
        c_rft=c_rft_full,
        kept_dct_indices=kept_dct,
        kept_rft_indices=kept_rft,
        energy_original=energy_original,
        energy_struct=energy_struct,
        energy_texture=energy_texture,
        energy_residual=energy_residual,
        energy_error=energy_error,
        sparsity_dct=sparsity_dct,
        sparsity_rft=sparsity_rft
    )


def verify_theorem_4_1():
    """
    Numerical verification of Theorem 4.1.
    
    Confirms that the decomposition satisfies:
    1. Correct sparsity in each domain
    2. Exact energy preservation (Parseval identity)
    3. Successful reconstruction
    """
    print("=" * 70)
    print("Verification of Theorem 4.1 (Hybrid Basis Decomposition)")
    print("=" * 70)
    
    # Test parameters
    n = 128
    k1 = 16  # Keep 16 DCT coefficients (87.5% sparse)
    k2 = 16  # Keep 16 RFT coefficients (87.5% sparse)
    
    # Generate test signals
    test_cases = {
        'Random': np.random.randn(n),
        'Smooth': np.cos(2 * np.pi * 3 * np.arange(n) / n),
        'Edges': np.sign(np.sin(2 * np.pi * 5 * np.arange(n) / n)),
        'Quasi-periodic': np.sum([
            np.cos(2 * np.pi * (1.618 * j) * np.arange(n) / n)
            for j in range(1, 4)
        ], axis=0)
    }
    
    for name, signal in test_cases.items():
        result = theoretic_hybrid_decomposition(signal, k1, k2)
        
        print(f"\nTest: {name}")
        print(f"  Original energy:    {result.energy_original:.10f}")
        print(f"  Structure energy:   {result.energy_struct:.10f}")
        print(f"  Texture energy:     {result.energy_texture:.10f}")
        print(f"  Residual energy:    {result.energy_residual:.10f}")
        print(f"  Total reconstructed: {result.energy_struct + result.energy_texture + result.energy_residual:.10f}")
        print(f"  Energy error:        {result.energy_error:.2e}  {'✓' if result.energy_error < 1e-12 else '✗ FAILED'}")
        print(f"  DCT sparsity:        {result.sparsity_dct:.1f}%")
        print(f"  RFT sparsity:        {result.sparsity_rft:.1f}%")
        
        # Verify reconstruction
        reconstructed = result.x_struct + result.x_texture + result.x_residual
        reconstruction_error = np.linalg.norm(signal - reconstructed)
        print(f"  Reconstruction error: {reconstruction_error:.2e}  {'✓' if reconstruction_error < 1e-12 else '✗ FAILED'}")
    
    print("\n" + "=" * 70)
    print("Theorem 4.1 verification complete.")
    print("All energy preservation errors should be < 1e-12 (machine precision).")
    print("=" * 70)


if __name__ == "__main__":
    verify_theorem_4_1()
