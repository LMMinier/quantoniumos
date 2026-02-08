#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Experimental fixes for hybrid MCA coherence problem.

Tests 6 hypotheses on ASCII bottleneck:
1. Coherence-aware routing
2. Phase-adaptive RFT
3. Hierarchical cascade
4. Quantum-inspired superposition
5. Attention-based soft gating
6. Dictionary learning bridge

Baseline: Current greedy hybrid (7.72 BPP failure)
Target: Match or beat DCT (4.83 BPP)
"""

import numpy as np
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse


@dataclass
class ExperimentResult:
    """Results from one hypothesis test."""
    name: str
    bpp: float
    psnr: float
    time_ms: float
    sparsity_pct: float
    reconstruction_error: float
    coherence_violation: float  # Measure of basis interference


# =============================================================================
# Baseline: Current Greedy Hybrid (for comparison)
# =============================================================================

def baseline_greedy_hybrid(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """Current approach: independent greedy selection per bin."""
    start = time.time()
    
    N = len(signal)
    
    # Compute both transforms
    C_dct = np.fft.rfft(signal, norm='ortho')  # Using FFT as DCT proxy
    C_rft = rft_forward(signal)
    
    # Greedy selection: pick larger magnitude per bin
    C_hybrid = np.zeros(N, dtype=complex)
    mask_dct = np.abs(C_dct[:N//2]) > np.abs(C_rft[:N//2])
    
    C_hybrid[:N//2][mask_dct] = C_dct[:N//2][mask_dct]
    C_hybrid[:N//2][~mask_dct] = C_rft[:N//2][~mask_dct]
    
    # Hard threshold to target sparsity
    threshold = np.percentile(np.abs(C_hybrid), target_sparsity * 100)
    C_hybrid[np.abs(C_hybrid) < threshold] = 0
    
    # Reconstruct (simplified - just use FFT inverse)
    reconstructed = np.fft.irfft(C_hybrid[:N//2], n=N, norm='ortho')
    
    elapsed_ms = (time.time() - start) * 1000
    
    return ExperimentResult(
        name="Baseline_Greedy_Hybrid",
        bpp=compute_bpp(C_hybrid),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_hybrid == 0) / len(C_hybrid) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft, mask_dct)
    )


# =============================================================================
# Hypothesis 1: Coherence-Aware Routing
# =============================================================================

def compute_coherence_matrix(N: int) -> np.ndarray:
    """
    Compute mutual coherence between DCT and RFT bases.
    M[i,j] = |⟨DCT_i, RFT_j⟩| measures interference.
    """
    # Create basis vectors
    DCT_basis = np.zeros((N, N))
    RFT_basis = np.zeros((N, N))
    
    for k in range(N):
        # DCT basis vector
        impulse = np.zeros(N)
        impulse[k] = 1.0
        DCT_basis[k] = np.fft.irfft(impulse, n=N, norm='ortho')
        
        # RFT basis vector (approximation)
        RFT_basis[k] = rft_forward(impulse)
    
    # Compute coherence
    M = np.abs(DCT_basis @ RFT_basis.T)
    return M


def find_coherence_blocks(M: np.ndarray, threshold: float = 0.3) -> List[List[int]]:
    """
    Group coefficients that strongly interfere.
    
    Returns blocks where max coherence within block > threshold.
    """
    N = M.shape[0]
    visited = np.zeros(N, dtype=bool)
    blocks = []
    
    for i in range(N):
        if visited[i]:
            continue
        
        # Find all j that interfere with i
        block = [i]
        interfering = np.where(M[i] > threshold)[0]
        
        for j in interfering:
            if not visited[j]:
                block.append(j)
                visited[j] = True
        
        visited[i] = True
        blocks.append(block)
    
    return blocks


def hypothesis1_coherence_aware(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Coherence-aware routing: group interfering coefficients and jointly optimize.
    """
    start = time.time()
    
    N = len(signal)
    
    # Pre-compute coherence structure (could be cached offline)
    M_coherence = compute_coherence_matrix(N)
    blocks = find_coherence_blocks(M_coherence, threshold=0.3)
    
    # Compute both transforms
    C_dct = np.fft.rfft(signal, norm='ortho')
    C_rft = rft_forward(signal)
    
    # Process each coherence block jointly
    C_hybrid = np.zeros(N, dtype=complex)
    
    for block in blocks:
        if len(block) == 1:
            # No interference - use greedy
            idx = block[0]
            if idx < len(C_dct):
                C_hybrid[idx] = C_dct[idx] if np.abs(C_dct[idx]) > np.abs(C_rft[idx]) else C_rft[idx]
        else:
            # Joint optimization: minimize reconstruction error for this block
            # Simplified: use energy-weighted combination
            block = [b for b in block if b < len(C_dct)]
            E_dct = np.sum(np.abs(C_dct[block])**2)
            E_rft = np.sum(np.abs(C_rft[block])**2)
            
            w_dct = E_dct / (E_dct + E_rft + 1e-10)
            
            for idx in block:
                C_hybrid[idx] = w_dct * C_dct[idx] + (1 - w_dct) * C_rft[idx]
    
    # Hard threshold
    threshold = np.percentile(np.abs(C_hybrid), target_sparsity * 100)
    C_hybrid[np.abs(C_hybrid) < threshold] = 0
    
    # Reconstruct
    reconstructed = np.fft.irfft(C_hybrid[:N//2], n=N, norm='ortho')
    
    elapsed_ms = (time.time() - start) * 1000
    
    mask_dct = np.abs(C_dct[:N//2]) > np.abs(C_rft[:N//2])
    
    return ExperimentResult(
        name="H1_Coherence_Aware",
        bpp=compute_bpp(C_hybrid),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_hybrid == 0) / len(C_hybrid) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft, mask_dct)
    )


# =============================================================================
# Hypothesis 2: Phase-Adaptive RFT
# =============================================================================

def detect_edges(signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Detect discontinuities in signal."""
    gradient = np.abs(np.gradient(signal))
    edges = gradient > (threshold * np.max(gradient))
    
    # Dilate edge mask
    kernel_size = 5
    edges_dilated = np.convolve(edges.astype(float), np.ones(kernel_size), mode='same') > 0
    
    return edges_dilated


def adaptive_phase_rft(signal: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """
    Apply RFT with spatially-varying golden ratio parameter.
    Near edges: φ→1 (DFT-like)
    Elsewhere: φ=golden ratio
    """
    N = len(signal)
    phi = (1 + np.sqrt(5)) / 2
    
    # Modulate phi based on edges
    phi_local = np.where(edge_mask, 1.0, phi)
    
    # Apply windowed RFT with varying phi (simplified)
    # In practice, would need full spatially-varying implementation
    # Here we use a weighted combination
    
    # Standard RFT
    C_rft_full = rft_forward(signal)
    
    # DFT (phi=1 limit)
    C_dft = np.fft.rfft(signal, norm='ortho')
    
    # Fix: Ensure matching lengths by padding to N
    min_len = min(len(C_rft_full), len(C_dft))
    C_rft_padded = np.pad(C_rft_full, (0, max(0, N - len(C_rft_full))), mode='constant')
    C_dft_padded = np.pad(C_dft, (0, max(0, N - len(C_dft))), mode='constant')
    
    # Blend based on edge proximity (improved: use spatial weighting)
    edge_weight = np.mean(edge_mask)
    C_adaptive = edge_weight * C_dft_padded[:N] + (1 - edge_weight) * C_rft_padded[:N]
    
    return C_adaptive


def hypothesis2_phase_adaptive(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Phase-adaptive RFT: gracefully degrade to DFT near discontinuities.
    """
    start = time.time()
    
    N = len(signal)
    
    # Detect edges
    edges = detect_edges(signal)
    
    # Compute DCT
    C_dct = np.fft.rfft(signal, norm='ortho')
    
    # Compute adaptive RFT
    C_rft_adaptive = adaptive_phase_rft(signal, edges)
    
    # Greedy selection
    C_hybrid = np.zeros(N, dtype=complex)
    mask_dct = np.abs(C_dct[:N//2]) > np.abs(C_rft_adaptive[:N//2])
    
    C_hybrid[:N//2][mask_dct] = C_dct[:N//2][mask_dct]
    C_hybrid[:N//2][~mask_dct] = C_rft_adaptive[:N//2][~mask_dct]
    
    # Hard threshold
    threshold = np.percentile(np.abs(C_hybrid), target_sparsity * 100)
    C_hybrid[np.abs(C_hybrid) < threshold] = 0
    
    # Reconstruct
    reconstructed = np.fft.irfft(C_hybrid[:N//2], n=N, norm='ortho')
    
    elapsed_ms = (time.time() - start) * 1000
    
    return ExperimentResult(
        name="H2_Phase_Adaptive",
        bpp=compute_bpp(C_hybrid),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_hybrid == 0) / len(C_hybrid) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft_adaptive, mask_dct)
    )


# =============================================================================
# Hypothesis 3: Hierarchical Cascade (MOST PROMISING)
# =============================================================================

def wavelet_decomposition(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple wavelet-like decomposition into structure and texture.
    Structure: low-pass (smooth variations)
    Texture: high-pass (fine details)
    """
    N = len(signal)
    
    # Low-pass: moving average
    kernel_size = max(3, N // 32)
    kernel = np.ones(kernel_size) / kernel_size
    structure = np.convolve(signal, kernel, mode='same')
    
    # High-pass: residual
    texture = signal - structure
    
    return structure, texture


def hypothesis3_hierarchical_cascade(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Hierarchical cascade: DCT for structure, RFT for texture residual.
    No competition between bases!
    """
    start = time.time()
    
    N = len(signal)
    
    # Level 1: Separate structure vs texture
    structure, texture = wavelet_decomposition(signal)
    
    # Level 2: DCT for structure
    C_dct_structure = np.fft.rfft(structure, norm='ortho')
    
    # Keep top-k for structure (more coefficients since it's smoother)
    k_structure = int(N * (1 - target_sparsity) * 0.7)
    threshold_s = np.percentile(np.abs(C_dct_structure), (1 - k_structure/len(C_dct_structure)) * 100)
    C_dct_sparse = C_dct_structure.copy()
    C_dct_sparse[np.abs(C_dct_sparse) < threshold_s] = 0
    
    # Reconstruct structure
    structure_recon = np.fft.irfft(C_dct_sparse, n=N, norm='ortho')
    
    # Level 3: RFT for texture residual
    residual = texture - (structure_recon - structure)
    C_rft_texture = rft_forward(residual)
    
    # Keep top-k for texture
    k_texture = int(N * (1 - target_sparsity) * 0.3)
    threshold_t = np.percentile(np.abs(C_rft_texture), (1 - k_texture/len(C_rft_texture)) * 100)
    C_rft_sparse = C_rft_texture.copy()
    C_rft_sparse[np.abs(C_rft_sparse) < threshold_t] = 0
    
    # Reconstruct texture (simplified - just use zero for now)
    texture_recon = np.zeros(N)
    
    # Final reconstruction
    reconstructed = structure_recon + texture_recon
    
    # Compute combined sparsity
    total_coeffs = len(C_dct_sparse) + len(C_rft_sparse)
    zero_coeffs = np.sum(C_dct_sparse == 0) + np.sum(C_rft_sparse == 0)
    
    elapsed_ms = (time.time() - start) * 1000
    
    # Combine coefficients for BPP calculation
    C_combined = np.concatenate([C_dct_sparse, C_rft_sparse[:N//2]])
    
    return ExperimentResult(
        name="H3_Hierarchical_Cascade",
        bpp=compute_bpp(C_combined),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=zero_coeffs / total_coeffs * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=0.0  # No coherence issues - separate domains!
    )


# =============================================================================
# Hypothesis 4: Quantum-Inspired Superposition
# =============================================================================

def hypothesis4_quantum_superposition(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Maintain superposition of DCT and RFT until final projection.
    Uses SVD to find optimal sparse subspace.
    """
    start = time.time()
    
    N = len(signal)
    
    # Create superposition
    C_dct = np.fft.rfft(signal, norm='ortho')
    C_rft = rft_forward(signal)
    
    # Stack as columns of state matrix
    min_len = min(len(C_dct), len(C_rft))
    state_matrix = np.column_stack([
        np.pad(C_dct[:min_len], (0, N - min_len)),
        np.pad(C_rft[:min_len], (0, N - min_len))
    ])
    
    # SVD to find optimal subspace
    U, S, Vh = np.linalg.svd(state_matrix, full_matrices=False)
    
    # Project onto sparse subspace (keep top singular values)
    k_components = max(1, int((1 - target_sparsity) * len(S)))
    S_sparse = S.copy()
    S_sparse[k_components:] = 0
    
    # Reconstruct in sparse subspace
    state_sparse = U @ np.diag(S_sparse) @ Vh
    
    # Extract DCT and RFT components
    C_dct_sparse = state_sparse[:, 0]
    C_rft_sparse = state_sparse[:, 1]
    
    # Reconstruct (weighted combination)
    reconstructed_dct = np.fft.irfft(C_dct_sparse[:N//2], n=N, norm='ortho')
    # RFT inverse is approximate
    reconstructed = reconstructed_dct  # Simplified
    
    elapsed_ms = (time.time() - start) * 1000
    
    C_combined = C_dct_sparse + C_rft_sparse
    mask_dct = np.ones(min_len, dtype=bool)
    
    return ExperimentResult(
        name="H4_Quantum_Superposition",
        bpp=compute_bpp(C_combined),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(S_sparse == 0) / len(S_sparse) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft, mask_dct)
    )


# =============================================================================
# Hypothesis 5: Attention-Based Soft Gating
# =============================================================================

def attention_weights(signal: np.ndarray, C_dct: np.ndarray, C_rft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute soft attention weights for DCT vs RFT.
    Uses signal statistics as query.
    """
    N = len(signal)
    
    # Query: signal features
    Q = np.array([
        np.mean(signal),
        np.std(signal),
        np.max(np.abs(np.gradient(signal))),  # Edge strength
        np.sum(signal**2) / N  # Energy
    ])
    
    # Keys: transform statistics
    min_len = min(len(C_dct), len(C_rft))
    
    K_dct = np.array([
        np.mean(np.abs(C_dct[:min_len])),
        np.std(np.abs(C_dct[:min_len])),
        np.max(np.abs(C_dct[:min_len])),
        np.sum(np.abs(C_dct[:min_len])**2) / min_len
    ])
    
    K_rft = np.array([
        np.mean(np.abs(C_rft[:min_len])),
        np.std(np.abs(C_rft[:min_len])),
        np.max(np.abs(C_rft[:min_len])),
        np.sum(np.abs(C_rft[:min_len])**2) / min_len
    ])
    
    # Attention scores (simplified - no learned weights)
    d = len(Q)
    score_dct = np.dot(Q, K_dct) / np.sqrt(d)
    score_rft = np.dot(Q, K_rft) / np.sqrt(d)
    
    # Softmax
    exp_scores = np.exp([score_dct, score_rft] - max(score_dct, score_rft))
    weights = exp_scores / np.sum(exp_scores)
    
    return weights[0], weights[1]


def hypothesis5_attention_gating(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Soft attention-based gating: learn optimal mixing weights.
    """
    start = time.time()
    
    N = len(signal)
    
    # Compute both transforms
    C_dct = np.fft.rfft(signal, norm='ortho')
    C_rft = rft_forward(signal)
    
    # Compute attention weights
    w_dct, w_rft = attention_weights(signal, C_dct, C_rft)
    
    # Soft combination (per-coefficient adaptive weighting)
    min_len = min(len(C_dct), len(C_rft))
    
    # Use local attention: per-frequency band
    bands = 8
    band_size = min_len // bands
    
    C_hybrid = np.zeros(N, dtype=complex)
    
    for b in range(bands):
        start_idx = b * band_size
        end_idx = min((b + 1) * band_size, min_len)
        
        # Band-specific weights based on energy ratio
        E_dct = np.sum(np.abs(C_dct[start_idx:end_idx])**2)
        E_rft = np.sum(np.abs(C_rft[start_idx:end_idx])**2)
        
        w_local_dct = w_dct * E_dct / (E_dct + E_rft + 1e-10)
        w_local_rft = w_rft * E_rft / (E_dct + E_rft + 1e-10)
        
        # Normalize
        total = w_local_dct + w_local_rft
        w_local_dct /= total
        w_local_rft /= total
        
        # Soft combination
        C_hybrid[start_idx:end_idx] = (
            w_local_dct * C_dct[start_idx:end_idx] +
            w_local_rft * C_rft[start_idx:end_idx]
        )
    
    # Hard threshold
    threshold = np.percentile(np.abs(C_hybrid), target_sparsity * 100)
    C_hybrid[np.abs(C_hybrid) < threshold] = 0
    
    # Reconstruct
    reconstructed = np.fft.irfft(C_hybrid[:N//2], n=N, norm='ortho')
    
    elapsed_ms = (time.time() - start) * 1000
    
    mask_dct = np.ones(min_len, dtype=bool)  # Soft gating, no hard mask
    
    return ExperimentResult(
        name="H5_Attention_Gating",
        bpp=compute_bpp(C_hybrid),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_hybrid == 0) / len(C_hybrid) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft, mask_dct)
    )


# =============================================================================
# Hypothesis 6: Dictionary Learning Bridge
# =============================================================================

def learn_bridge_atoms(signal: np.ndarray, n_atoms: int = 32) -> np.ndarray:
    """
    Learn dictionary atoms that bridge DCT and RFT.
    Uses simple PCA on residuals.
    """
    N = len(signal)
    
    # Initial decomposition
    C_dct = np.fft.rfft(signal, norm='ortho')
    C_rft = rft_forward(signal)
    
    # Reconstruct with top-k
    k = N // 4
    C_dct_topk = np.zeros_like(C_dct)
    top_indices = np.argsort(np.abs(C_dct))[-k:]
    C_dct_topk[top_indices] = C_dct[top_indices]
    
    recon_dct = np.fft.irfft(C_dct_topk, n=N, norm='ortho')
    
    # Residual
    residual = signal - recon_dct
    
    # Extract principal components from residual
    # Create shifted versions for PCA
    window_size = N // n_atoms
    windows = np.array([
        residual[i:i+window_size] 
        for i in range(0, N - window_size, window_size)
    ])
    
    # PCA
    if len(windows) > 0:
        U, S, Vh = np.linalg.svd(windows, full_matrices=False)
        bridge_atoms = Vh[:n_atoms]
    else:
        bridge_atoms = np.random.randn(n_atoms, window_size)
    
    return bridge_atoms


def hypothesis6_dictionary_learning(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Learn overcomplete dictionary with DCT, RFT, and bridge atoms.
    """
    start = time.time()
    
    N = len(signal)
    
    # Learn bridge atoms
    bridge_atoms = learn_bridge_atoms(signal, n_atoms=16)
    
    # Compute standard transforms
    C_dct = np.fft.rfft(signal, norm='ortho')
    C_rft = rft_forward(signal)
    
    # Project onto bridge atoms (simplified sparse coding)
    # In practice, would use OMP or LASSO
    residual = signal.copy()
    C_bridge = np.zeros(len(bridge_atoms))
    
    for i, atom in enumerate(bridge_atoms):
        # Pad atom to signal length
        atom_full = np.pad(atom, (0, N - len(atom)))
        
        # Projection coefficient
        C_bridge[i] = np.dot(residual, atom_full)
        
        # Update residual
        residual -= C_bridge[i] * atom_full
    
    # Combine all coefficients
    C_combined = np.concatenate([
        np.abs(C_dct),
        np.abs(C_rft[:len(C_dct)]),
        np.abs(C_bridge)
    ])
    
    # Hard threshold
    threshold = np.percentile(C_combined, target_sparsity * 100)
    
    C_dct_sparse = C_dct.copy()
    C_dct_sparse[np.abs(C_dct_sparse) < threshold] = 0
    
    C_rft_sparse = C_rft.copy()
    C_rft_sparse[np.abs(C_rft_sparse) < threshold] = 0
    
    C_bridge[np.abs(C_bridge) < threshold] = 0
    
    # Reconstruct (simplified - just DCT component)
    reconstructed = np.fft.irfft(C_dct_sparse, n=N, norm='ortho')
    
    elapsed_ms = (time.time() - start) * 1000
    
    C_all = np.concatenate([C_dct_sparse, C_rft_sparse[:len(C_dct_sparse)], C_bridge])
    mask_dct = np.ones(len(C_dct), dtype=bool)
    
    return ExperimentResult(
        name="H6_Dictionary_Learning",
        bpp=compute_bpp(C_all),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_all == 0) / len(C_all) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=compute_coherence_violation(C_dct, C_rft, mask_dct)
    )


# =============================================================================
# Hypothesis 7: Hybrid Cascade + Attention (H3 + H5)
# =============================================================================

def hypothesis7_cascade_attention(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Combine hierarchical cascade (H3) with attention gating (H5).
    Use cascade to avoid coherence, attention to optimize within each domain.
    """
    start = time.time()
    
    N = len(signal)
    
    # Level 1: Separate structure vs texture (from H3)
    structure, texture = wavelet_decomposition(signal)
    
    # Level 2: Attention-based routing for STRUCTURE
    C_dct_structure = np.fft.rfft(structure, norm='ortho')
    C_rft_structure = rft_forward(structure)
    
    w_s_dct, w_s_rft = attention_weights(structure, C_dct_structure, C_rft_structure)
    
    # Soft combination for structure (band-wise adaptive)
    min_len_s = min(len(C_dct_structure), len(C_rft_structure))
    bands = 4
    band_size = min_len_s // bands
    
    C_structure = np.zeros(N, dtype=complex)
    for b in range(bands):
        start_idx = b * band_size
        end_idx = min((b + 1) * band_size, min_len_s)
        
        # Band-specific weights
        E_dct = np.sum(np.abs(C_dct_structure[start_idx:end_idx])**2)
        E_rft = np.sum(np.abs(C_rft_structure[start_idx:end_idx])**2)
        
        w_local_dct = w_s_dct * E_dct / (E_dct + E_rft + 1e-10)
        w_local_rft = w_s_rft * E_rft / (E_dct + E_rft + 1e-10)
        
        total = w_local_dct + w_local_rft
        w_local_dct /= (total + 1e-10)
        w_local_rft /= (total + 1e-10)
        
        C_structure[start_idx:end_idx] = (
            w_local_dct * C_dct_structure[start_idx:end_idx] +
            w_local_rft * C_rft_structure[start_idx:end_idx]
        )
    
    # Level 3: Attention-based routing for TEXTURE
    C_dct_texture = np.fft.rfft(texture, norm='ortho')
    C_rft_texture = rft_forward(texture)
    
    w_t_dct, w_t_rft = attention_weights(texture, C_dct_texture, C_rft_texture)
    
    # Soft combination for texture
    min_len_t = min(len(C_dct_texture), len(C_rft_texture))
    C_texture = np.zeros(N, dtype=complex)
    
    for b in range(bands):
        start_idx = b * band_size
        end_idx = min((b + 1) * band_size, min_len_t)
        
        E_dct = np.sum(np.abs(C_dct_texture[start_idx:end_idx])**2)
        E_rft = np.sum(np.abs(C_rft_texture[start_idx:end_idx])**2)
        
        w_local_dct = w_t_dct * E_dct / (E_dct + E_rft + 1e-10)
        w_local_rft = w_t_rft * E_rft / (E_dct + E_rft + 1e-10)
        
        total = w_local_dct + w_local_rft
        w_local_dct /= (total + 1e-10)
        w_local_rft /= (total + 1e-10)
        
        C_texture[start_idx:end_idx] = (
            w_local_dct * C_dct_texture[start_idx:end_idx] +
            w_local_rft * C_rft_texture[start_idx:end_idx]
        )
    
    # Joint sparsity optimization: pool coefficients from both domains
    all_coeffs = np.concatenate([C_structure[:N//2], C_texture[:N//2]])
    threshold = np.percentile(np.abs(all_coeffs), target_sparsity * 100)
    
    C_structure[np.abs(C_structure) < threshold] = 0
    C_texture[np.abs(C_texture) < threshold] = 0
    
    # Reconstruct both domains
    structure_recon = np.fft.irfft(C_structure[:N//2], n=N, norm='ortho')
    texture_recon = np.fft.irfft(C_texture[:N//2], n=N, norm='ortho')
    
    reconstructed = structure_recon + texture_recon
    
    elapsed_ms = (time.time() - start) * 1000
    
    # Combined coefficients
    C_combined = np.concatenate([C_structure[:N//2], C_texture[:N//2]])
    
    return ExperimentResult(
        name="H7_Cascade_Attention",
        bpp=compute_bpp(C_combined),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(C_combined == 0) / len(C_combined) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=0.0  # Cascade architecture = no coherence issues
    )


# =============================================================================
# Hypothesis 8: Aggressive Cascade with Multi-scale Decomposition
# =============================================================================

def multi_scale_decomposition(signal: np.ndarray, levels: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale wavelet-like decomposition (better than H3's simple smoothing).
    Returns structure (multi-scale smooth) and texture (high-frequency detail).
    """
    N = len(signal)
    structure_scales = []
    
    # Multi-scale smoothing
    for level in range(levels):
        kernel_size = max(3, N // (2 ** (5 - level)))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(signal, kernel, mode='same')
        structure_scales.append(smoothed)
    
    # Average across scales for robust structure
    structure = np.mean(structure_scales, axis=0)
    
    # Texture is high-frequency residual
    texture = signal - structure
    
    return structure, texture


def hypothesis8_aggressive_cascade(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Aggressive cascade with:
    1. Multi-scale structure/texture decomposition
    2. Adaptive budget allocation (not fixed 70/30)
    3. Joint threshold optimization
    """
    start = time.time()
    
    N = len(signal)
    
    # Level 1: Multi-scale decomposition
    structure, texture = multi_scale_decomposition(signal, levels=3)
    
    # Level 2: Compute both transforms for structure
    C_dct_structure = np.fft.rfft(structure, norm='ortho')
    C_rft_structure = rft_forward(structure)
    
    # Adaptive routing: choose transform based on sparsity potential
    sparsity_dct_s = np.sum(np.abs(C_dct_structure) < np.median(np.abs(C_dct_structure))) / len(C_dct_structure)
    sparsity_rft_s = np.sum(np.abs(C_rft_structure) < np.median(np.abs(C_rft_structure))) / len(C_rft_structure)
    
    # Use whichever is naturally sparser
    if sparsity_dct_s > sparsity_rft_s:
        C_structure = np.pad(C_dct_structure, (0, N - len(C_dct_structure)))
    else:
        C_structure = np.pad(C_rft_structure, (0, N - len(C_rft_structure)))
    
    # Level 3: Same for texture
    C_dct_texture = np.fft.rfft(texture, norm='ortho')
    C_rft_texture = rft_forward(texture)
    
    sparsity_dct_t = np.sum(np.abs(C_dct_texture) < np.median(np.abs(C_dct_texture))) / len(C_dct_texture)
    sparsity_rft_t = np.sum(np.abs(C_rft_texture) < np.median(np.abs(C_rft_texture))) / len(C_rft_texture)
    
    if sparsity_dct_t > sparsity_rft_t:
        C_texture = np.pad(C_dct_texture, (0, N - len(C_dct_texture)))
    else:
        C_texture = np.pad(C_rft_texture, (0, N - len(C_rft_texture)))
    
    # Joint threshold optimization: allocate budget dynamically
    all_coeffs = np.concatenate([C_structure, C_texture])
    threshold = np.percentile(np.abs(all_coeffs), target_sparsity * 100)
    
    C_structure[np.abs(C_structure) < threshold] = 0
    C_texture[np.abs(C_texture) < threshold] = 0
    
    # Reconstruct
    structure_recon = np.fft.irfft(C_structure[:N//2], n=N, norm='ortho')
    texture_recon = np.fft.irfft(C_texture[:N//2], n=N, norm='ortho')
    
    reconstructed = structure_recon + texture_recon
    
    elapsed_ms = (time.time() - start) * 1000
    
    return ExperimentResult(
        name="H8_Aggressive_Cascade",
        bpp=compute_bpp(all_coeffs),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(all_coeffs == 0) / len(all_coeffs) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=0.0
    )


# =============================================================================
# Hypothesis 9: Iterative Refinement (Best of Both Worlds)
# =============================================================================

def hypothesis9_iterative_refinement(signal: np.ndarray, target_sparsity: float = 0.95, max_iters: int = 3) -> ExperimentResult:
    """
    Iterative approach:
    1. Start with H3 (cascade for low BPP)
    2. Iteratively refine with H5 (attention for better PSNR)
    3. Stop when PSNR stops improving
    """
    start = time.time()
    
    N = len(signal)
    best_result = None
    best_psnr = -np.inf
    
    # Initial decomposition
    structure, texture = wavelet_decomposition(signal)
    
    for iteration in range(max_iters):
        # Compute transforms
        C_dct_s = np.fft.rfft(structure, norm='ortho')
        C_rft_s = rft_forward(structure)
        C_dct_t = np.fft.rfft(texture, norm='ortho')
        C_rft_t = rft_forward(texture)
        
        # Attention weights (refined each iteration)
        w_s_dct, w_s_rft = attention_weights(structure, C_dct_s, C_rft_s)
        w_t_dct, w_t_rft = attention_weights(texture, C_dct_t, C_rft_t)
        
        # Soft combination
        min_len = min(len(C_dct_s), len(C_rft_s))
        C_structure = w_s_dct * np.pad(C_dct_s[:min_len], (0, N - min_len)) + w_s_rft * np.pad(C_rft_s[:min_len], (0, N - min_len))
        
        min_len = min(len(C_dct_t), len(C_rft_t))
        C_texture = w_t_dct * np.pad(C_dct_t[:min_len], (0, N - min_len)) + w_t_rft * np.pad(C_rft_t[:min_len], (0, N - min_len))
        
        # Threshold
        all_coeffs = np.concatenate([C_structure[:N//2], C_texture[:N//2]])
        threshold = np.percentile(np.abs(all_coeffs), target_sparsity * 100)
        
        C_structure[np.abs(C_structure) < threshold] = 0
        C_texture[np.abs(C_texture) < threshold] = 0
        
        # Reconstruct
        structure_recon = np.fft.irfft(C_structure[:N//2], n=N, norm='ortho')
        texture_recon = np.fft.irfft(C_texture[:N//2], n=N, norm='ortho')
        reconstructed = structure_recon + texture_recon
        
        # Check quality
        psnr = compute_psnr(signal, reconstructed)
        
        if psnr > best_psnr:
            best_psnr = psnr
            best_result = (reconstructed, all_coeffs)
            
            # Update residual for next iteration
            residual = signal - reconstructed
            texture = residual
        else:
            # No improvement, stop
            break
    
    elapsed_ms = (time.time() - start) * 1000
    
    reconstructed, all_coeffs = best_result
    
    return ExperimentResult(
        name="H9_Iterative_Refinement",
        bpp=compute_bpp(all_coeffs),
        psnr=best_psnr,
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(all_coeffs == 0) / len(all_coeffs) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=0.0
    )


# =============================================================================
# Hypothesis 10: H3 with Quality Boost (Best BPP + Better PSNR)
# =============================================================================

def hypothesis10_quality_cascade(signal: np.ndarray, target_sparsity: float = 0.95) -> ExperimentResult:
    """
    Take H3's winning architecture but improve PSNR by:
    1. Better structure/texture split (variance-based)
    2. Smarter transform selection per domain
    3. Perceptual weighting (keep visually important coefficients)
    """
    start = time.time()
    
    N = len(signal)
    
    # Variance-based decomposition (better than simple smoothing)
    window_size = max(8, N // 64)
    local_variance = np.array([
        np.var(signal[max(0, i-window_size//2):min(N, i+window_size//2)])
        for i in range(N)
    ])
    
    # High variance = texture, low variance = structure
    variance_threshold = np.percentile(local_variance, 50)
    
    # Fix: Prevent divide-by-zero and overflow in sigmoid
    if variance_threshold == 0:
        variance_threshold = 1e-10
    
    # Soft masking with safe sigmoid (clip to prevent overflow)
    sigmoid_arg = (local_variance - variance_threshold) / (0.1 * variance_threshold + 1e-10)
    sigmoid_arg = np.clip(sigmoid_arg, -20, 20)  # Prevent exp overflow
    structure_weight = 1.0 / (1.0 + np.exp(sigmoid_arg))
    texture_weight = 1.0 - structure_weight
    
    structure = signal * structure_weight
    texture = signal * texture_weight
    
    # Transform selection: try both and pick naturally sparser one
    C_dct_s = np.fft.rfft(structure, norm='ortho')
    C_rft_s = rft_forward(structure)
    
    # Measure natural sparsity (entropy proxy)
    # Fix: Normalize and handle edge cases
    mag_dct_s = np.abs(C_dct_s)**2
    mag_dct_s = mag_dct_s / (np.sum(mag_dct_s) + 1e-10)  # Normalize to probability
    entropy_dct_s = -np.sum(mag_dct_s * np.log(mag_dct_s + 1e-10))
    
    mag_rft_s = np.abs(C_rft_s)**2
    mag_rft_s = mag_rft_s / (np.sum(mag_rft_s) + 1e-10)
    entropy_rft_s = -np.sum(mag_rft_s * np.log(mag_rft_s + 1e-10))
    
    if entropy_dct_s < entropy_rft_s:  # Lower entropy = sparser
        C_structure_chosen = C_dct_s
    else:
        C_structure_chosen = C_rft_s
    
    # Same for texture
    C_dct_t = np.fft.rfft(texture, norm='ortho')
    C_rft_t = rft_forward(texture)
    
    mag_dct_t = np.abs(C_dct_t)**2
    mag_dct_t = mag_dct_t / (np.sum(mag_dct_t) + 1e-10)
    entropy_dct_t = -np.sum(mag_dct_t * np.log(mag_dct_t + 1e-10))
    
    mag_rft_t = np.abs(C_rft_t)**2
    mag_rft_t = mag_rft_t / (np.sum(mag_rft_t) + 1e-10)
    entropy_rft_t = -np.sum(mag_rft_t * np.log(mag_rft_t + 1e-10))
    
    if entropy_dct_t < entropy_rft_t:
        C_texture_chosen = C_dct_t
    else:
        C_texture_chosen = C_rft_t
    
    # Perceptual weighting: prioritize low frequencies
    # Fix: Ensure weights match coefficient lengths
    freq_weights_s = np.exp(-np.arange(len(C_structure_chosen)) / (len(C_structure_chosen) / 4))
    freq_weights_t = np.exp(-np.arange(len(C_texture_chosen)) / (len(C_texture_chosen) / 4))
    C_structure_weighted = C_structure_chosen * freq_weights_s
    C_texture_weighted = C_texture_chosen * freq_weights_t
    
    # Joint threshold on weighted coefficients
    all_weighted = np.concatenate([C_structure_weighted, C_texture_weighted])
    threshold = np.percentile(np.abs(all_weighted), target_sparsity * 100)
    
    # Apply threshold to ORIGINAL (unweighted) coefficients
    C_structure_sparse = C_structure_chosen.copy()
    C_structure_sparse[np.abs(C_structure_weighted) < threshold] = 0
    
    C_texture_sparse = C_texture_chosen.copy()
    C_texture_sparse[np.abs(C_texture_weighted) < threshold] = 0
    
    # Reconstruct
    structure_recon = np.fft.irfft(C_structure_sparse, n=N, norm='ortho')
    texture_recon = np.fft.irfft(C_texture_sparse, n=N, norm='ortho')
    
    reconstructed = structure_recon + texture_recon
    
    elapsed_ms = (time.time() - start) * 1000
    
    # Fix: Ensure positive padding and handle length mismatches
    pad_s = max(0, N//2 - len(C_structure_sparse))
    pad_t = max(0, N//2 - len(C_texture_sparse))
    
    all_coeffs = np.concatenate([
        np.pad(C_structure_sparse, (0, pad_s))[:N//2],
        np.pad(C_texture_sparse, (0, pad_t))[:N//2]
    ])
    
    return ExperimentResult(
        name="H10_Quality_Cascade",
        bpp=compute_bpp(all_coeffs),
        psnr=compute_psnr(signal, reconstructed),
        time_ms=elapsed_ms,
        sparsity_pct=np.sum(all_coeffs == 0) / len(all_coeffs) * 100,
        reconstruction_error=np.linalg.norm(signal - reconstructed),
        coherence_violation=0.0
    )


# =============================================================================
# Utilities
# =============================================================================

def compute_bpp(coefficients: np.ndarray) -> float:
    """Estimate bits per pixel from coefficient sparsity."""
    nonzero = np.sum(coefficients != 0)
    total = len(coefficients)
    
    if nonzero == 0:
        return 0.0
    
    # Entropy coding approximation
    sparsity = nonzero / total
    entropy = -sparsity * np.log2(sparsity + 1e-10) - (1 - sparsity) * np.log2(1 - sparsity + 1e-10)
    
    # Add overhead for values
    bits_per_nonzero = 16  # Assume 16-bit quantization
    
    bpp = (nonzero * bits_per_nonzero) / total
    return bpp


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PSNR in dB."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = np.max(np.abs(original))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_coherence_violation(C_dct: np.ndarray, C_rft: np.ndarray, mask: np.ndarray) -> float:
    """
    Measure coherence violation: energy in rejected basis coefficients.
    High value = bad (throwing away correlated information).
    """
    min_len = min(len(C_dct), len(C_rft), len(mask))
    
    # Energy in DCT where we chose RFT
    E_dct_rejected = np.sum(np.abs(C_dct[:min_len][~mask[:min_len]])**2)
    
    # Energy in RFT where we chose DCT
    E_rft_rejected = np.sum(np.abs(C_rft[:min_len][mask[:min_len]])**2)
    
    # Total energy
    E_total = np.sum(np.abs(C_dct[:min_len])**2) + np.sum(np.abs(C_rft[:min_len])**2)
    
    violation = (E_dct_rejected + E_rft_rejected) / (E_total + 1e-10)
    
    return violation


# =============================================================================
# Main Experiment Runner
# =============================================================================

def load_ascii_test_signal() -> np.ndarray:
    """Load ASCII text as test signal (like Python source code)."""
    try:
        # Try to load actual Python file
        test_file = Path(__file__).parent.parent / "algorithms" / "rft" / "core" / "closed_form_rft.py"
        with open(test_file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Normalize to [-1, 1]
        signal = (data.astype(float) / 127.5) - 1.0
        
        # Truncate to power of 2 for transforms
        N = 2 ** int(np.log2(len(signal)))
        signal = signal[:N]
        
        print(f"Loaded ASCII signal: {len(signal)} samples from {test_file.name}")
        return signal
        
    except Exception as e:
        print(f"Warning: Could not load test file: {e}")
        print("Using synthetic ASCII-like signal")
        
        # Synthetic ASCII: discrete values with sharp transitions
        N = 1024
        signal = np.zeros(N)
        
        # Create text-like blocks
        block_size = 8
        for i in range(0, N, block_size):
            # Random ASCII character value
            signal[i:i+block_size] = (np.random.randint(32, 127) / 127.5) - 1.0
        
        return signal


def run_all_experiments(signal: np.ndarray, target_sparsity: float = 0.95) -> List[ExperimentResult]:
    """Run all 6 hypotheses plus baseline."""
    results = []
    
    print(f"\n{'='*80}")
    print(f"ASCII BOTTLENECK EXPERIMENT")
    print(f"Signal length: {len(signal)}, Target sparsity: {target_sparsity*100:.1f}%")
    print(f"{'='*80}\n")
    
    experiments = [
        ("Baseline", baseline_greedy_hybrid),
        ("Hypothesis 1", hypothesis1_coherence_aware),
        ("Hypothesis 2", hypothesis2_phase_adaptive),
        ("Hypothesis 3", hypothesis3_hierarchical_cascade),
        ("Hypothesis 4", hypothesis4_quantum_superposition),
        ("Hypothesis 5", hypothesis5_attention_gating),
        ("Hypothesis 6", hypothesis6_dictionary_learning),
        ("Hypothesis 7 (H3+H5)", hypothesis7_cascade_attention),
        ("Hypothesis 8 (Aggressive)", hypothesis8_aggressive_cascade),
        ("Hypothesis 9 (Iterative)", hypothesis9_iterative_refinement),
        ("Hypothesis 10 (Quality)", hypothesis10_quality_cascade),
    ]
    
    for name, func in experiments:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            result = func(signal, target_sparsity)
            results.append(result)
            print(f"✓ ({result.time_ms:.1f}ms)")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_results_table(results: List[ExperimentResult]):
    """Print formatted results table."""
    print(f"\n{'='*120}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*120}")
    
    # Header
    print(f"{'Method':<30} {'BPP':>8} {'PSNR':>8} {'Time':>10} {'Sparsity':>10} {'Recon Err':>12} {'Coherence':>12}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    
    # Sort by BPP (lower is better)
    results_sorted = sorted(results, key=lambda r: r.bpp)
    
    # Baseline for comparison
    baseline = next((r for r in results if "Baseline" in r.name), None)
    
    for result in results_sorted:
        # Highlight if better than baseline
        marker = "🏆 " if baseline and result.bpp < baseline.bpp and "Baseline" not in result.name else "   "
        
        print(f"{marker}{result.name:<27} "
              f"{result.bpp:>8.3f} "
              f"{result.psnr:>8.2f} "
              f"{result.time_ms:>9.1f}ms "
              f"{result.sparsity_pct:>9.1f}% "
              f"{result.reconstruction_error:>12.4e} "
              f"{result.coherence_violation:>12.4e}")
    
    print(f"{'='*120}\n")
    
    # Analysis
    print("ANALYSIS:")
    print(f"  Baseline BPP: {baseline.bpp:.3f} (current greedy hybrid)")
    print(f"  Target: < 4.83 BPP (DCT baseline from paper)")
    print()
    
    # Best performer
    best = results_sorted[0] if "Baseline" not in results_sorted[0].name else results_sorted[1]
    improvement = ((baseline.bpp - best.bpp) / baseline.bpp) * 100
    
    print(f"  🏆 BEST: {best.name}")
    print(f"     BPP: {best.bpp:.3f} ({improvement:+.1f}% vs baseline)")
    print(f"     PSNR: {best.psnr:.2f} dB")
    print(f"     Coherence violation: {best.coherence_violation:.4e}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if best.bpp < 4.83:
        print(f"  ✓ SUCCESS! {best.name} beats DCT baseline (4.83 BPP)")
    else:
        print(f"  ⚠ Still above DCT baseline. Best candidate: {best.name}")
    
    if best.coherence_violation < baseline.coherence_violation * 0.5:
        print(f"  ✓ Coherence violation reduced by {(1 - best.coherence_violation/baseline.coherence_violation)*100:.0f}%")
    
    print()


def main():
    """Main experiment."""
    # Load test signal
    signal = load_ascii_test_signal()
    
    # Run experiments
    results = run_all_experiments(signal, target_sparsity=0.95)
    
    # Print results
    print_results_table(results)
    
    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "hybrid_mca_experiment_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("ASCII BOTTLENECK EXPERIMENT RESULTS\n")
        f.write("="*120 + "\n\n")
        
        for result in results:
            f.write(f"{result.name}:\n")
            f.write(f"  BPP: {result.bpp:.3f}\n")
            f.write(f"  PSNR: {result.psnr:.2f} dB\n")
            f.write(f"  Time: {result.time_ms:.1f} ms\n")
            f.write(f"  Sparsity: {result.sparsity_pct:.1f}%\n")
            f.write(f"  Reconstruction error: {result.reconstruction_error:.4e}\n")
            f.write(f"  Coherence violation: {result.coherence_violation:.4e}\n")
            f.write("\n")
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
