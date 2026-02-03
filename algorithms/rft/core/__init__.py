# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Core RFT implementation subpackage.

CANONICAL DEFINITION (December 2025):
=====================================
The Resonant Fourier Transform (RFT) is a multi-carrier transform using
golden-ratio frequency and phase spacing:

    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    where:
        fₖ = (k+1) × φ       (Resonant Frequency)
        θₖ = 2π × k / φ      (Golden Phase)  
        φ = (1+√5)/2         (Golden Ratio ≈ 1.618)

This enables:
1. Binary → Wave encoding (BPSK on resonant carriers)
2. Logic operations IN THE WAVE DOMAIN (XOR, AND, OR, NOT)
3. Computation without decoding intermediate results

See: algorithms/rft/core/resonant_fourier_transform.py
"""

import numpy as np

# CANONICAL RFT - Golden-ratio multi-carrier transform
from .resonant_fourier_transform import (
    # Constants
    PHI,
    PHI_INV,
    
    # Core functions
    rft_frequency,
    rft_phase,
    rft_basis_function,
    rft_basis_matrix,
    
    # Transform functions
    rft_forward,
    rft_inverse,
    rft_forward_frame,
    rft_inverse_frame,
    rft,
    irft,
    
    # Binary RFT for wave-domain computation
    BinaryRFT,
)


def rft_forward_canonical(x: np.ndarray) -> np.ndarray:
    """Canonical RFT forward transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(x), len(x), use_gram_normalization=True)
    return rft_forward_frame(x, phi)


def rft_inverse_canonical(X: np.ndarray) -> np.ndarray:
    """Canonical RFT inverse transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(X), len(X), use_gram_normalization=True)
    return rft_inverse_frame(X, phi)

from .oscillator import Oscillator
from .geometric_container import GeometricContainer, LinearRegion
from .bloom_filter import SimplifiedBloomFilter, hash1, hash2
from .shard import Shard
from .vibrational_engine import VibrationalEngine

# Theorem implementations (Maassen-Uffink is canonical for uncertainty)
from .maassen_uffink_uncertainty import (
    measure_entropic_uncertainty,
    verify_theorem_9,
    MaassenUffinkBound,
    EntropicUncertainty,
)
from .theorem8_bootstrap_verification import (
    verify_theorem_8_bootstrap,
    Theorem8BootstrapResult,
)

# Advanced mathematical extensions
from .diophantine_rft_extension import (
    SQRT2, SQRT3, SQRT5, SILVER_RATIO,
    continued_fraction,
    convergents,
    diophantine_constant,
    diophantine_basis_matrix,
    star_discrepancy,
    analyze_equidistribution,
    davis_kahan_analysis,
    compare_k99_diophantine,
    verify_scaling_law,
    verify_universality,  # Backward compatibility alias
    verify_sharp_logn_bound,
    ScalingLawResult,
    UniversalityResult,  # Backward compatibility alias
    SharpBoundResult,
    DiscrepancyResult,
    DavisKahanResult,
    DiophantineK99Result,
)

from .sharp_coherence_bounds import (
    mutual_coherence,
    coherence_matrix,
    asymptotic_coherence_analysis,
    verify_coherence_scaling,
    verify_sqrt_n_mu_stabilization,
    gram_matrix_analysis,
    compute_sharp_mu_bound,
    measure_entropy_sum,
    verify_sharp_bound,
    riesz_thorin_analysis,
    comprehensive_sharp_verification,
    AsymptoticCoherenceResult,
    GramMatrixResult,
    SharpMUBoundResult,
    RieszThorinResult,
)

from .fibonacci_fast_rft import (
    fibonacci,
    fibonacci_sequence,
    zeckendorf,
    nearest_fibonacci,
    fast_rft_fibonacci,
    fast_rft_bluestein,
    compare_rft_algorithms,
    optimal_fibonacci_size,
    list_fibonacci_rft_sizes,
    analyze_complexity,
    FibonacciRFTResult,
    AlgorithmComparison,
    ComplexityResult,
)

__all__ = [
    'PHI',
    'PHI_INV',
    'rft_frequency',
    'rft_phase',
    'rft_basis_function',
    'rft_basis_matrix',
    'rft_forward',
    'rft_inverse',
    'rft',
    'irft',
    'BinaryRFT',
    'Oscillator',
    'GeometricContainer',
    'LinearRegion',
    'SimplifiedBloomFilter',
    'hash1',
    'hash2',
    'Shard',
    'VibrationalEngine',
    # Theorem 9 (Maassen-Uffink)
    'measure_entropic_uncertainty',
    'verify_theorem_9',
    'MaassenUffinkBound',
    'EntropicUncertainty',
    # Theorem 8 (Bootstrap CI)
    'verify_theorem_8_bootstrap',
    'Theorem8BootstrapResult',
    # Diophantine extension (Theorem 8 scaling law)
    'SQRT2', 'SQRT3', 'SQRT5', 'SILVER_RATIO',
    'continued_fraction',
    'convergents',
    'diophantine_constant',
    'diophantine_basis_matrix',
    'star_discrepancy',
    'analyze_equidistribution',
    'davis_kahan_analysis',
    'compare_k99_diophantine',
    'verify_scaling_law',
    'verify_universality',  # Backward compat alias
    'verify_sharp_logn_bound',
    'ScalingLawResult',
    'UniversalityResult',  # Backward compat alias
    'SharpBoundResult',
    'DiscrepancyResult',
    'DavisKahanResult',
    'DiophantineK99Result',
    # Sharp coherence bounds (Theorem 9 sharpening)
    'mutual_coherence',
    'coherence_matrix',
    'asymptotic_coherence_analysis',
    'verify_coherence_scaling',
    'verify_sqrt_n_mu_stabilization',
    'gram_matrix_analysis',
    'compute_sharp_mu_bound',
    'measure_entropy_sum',
    'verify_sharp_bound',
    'riesz_thorin_analysis',
    'comprehensive_sharp_verification',
    'AsymptoticCoherenceResult',
    'GramMatrixResult',
    'SharpMUBoundResult',
    'RieszThorinResult',
    # Fibonacci fast RFT
    'fibonacci',
    'fibonacci_sequence',
    'zeckendorf',
    'nearest_fibonacci',
    'fast_rft_fibonacci',
    'fast_rft_bluestein',
    'compare_rft_algorithms',
    'optimal_fibonacci_size',
    'list_fibonacci_rft_sizes',
    'analyze_complexity',
    'FibonacciRFTResult',
    'AlgorithmComparison',
    'ComplexityResult',
]
