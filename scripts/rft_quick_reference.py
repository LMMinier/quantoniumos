#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Testing and Visualization Quick Reference
Usage examples and command summary
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  RFT Testing & Visualization Suite                        ║
║                        Quick Reference Guide                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

AVAILABLE TEST SCRIPTS:
───────────────────────────────────────────────────────────────────────────

1. Basic RFT vs FFT Test
   Command: python test_rft_vs_fft.py
   
   Tests:
   • Unitarity verification (round-trip accuracy)
   • Energy preservation (Parseval's theorem)
   • Matrix orthogonality (Ψ†Ψ = I)
   • Signal reconstruction (various signal types)
   • Spectrum comparison
   • Linearity property
   • Performance benchmarks
   
   Output: Console text with test results
   Duration: ~10 seconds

───────────────────────────────────────────────────────────────────────────

2. Comprehensive Visualization Suite
   Command: python visualize_rft_analysis.py
   
   Generates:
   • figures/unitarity_error.png (& .pdf)
   • figures/performance_benchmark.png (& .pdf)
   • figures/spectrum_comparison.png (& .pdf)
   • figures/compression_efficiency.png (& .pdf)
   • figures/phase_structure.png (& .pdf)
   • figures/matrix_structure.png (& .pdf)
   • figures/energy_compaction.png (& .pdf)
   • figures/latex_data/*.dat (data files for LaTeX)
   
   Output: PNG and PDF figures in figures/ directory
   Duration: ~30-60 seconds

───────────────────────────────────────────────────────────────────────────

3. LaTeX/TikZ Publication Figures
   File: figures_rft_tikz.tex
   
   Compile with: pdflatex figures_rft_tikz.tex
   (requires LaTeX distribution with tikz, pgfplots)
   
   Generates:
   • High-quality vector graphics
   • Publication-ready figures
   • Mathematical diagrams
   • Comparison tables
   
   Note: Requires pdflatex installation

───────────────────────────────────────────────────────────────────────────

QUICK START EXAMPLES:
───────────────────────────────────────────────────────────────────────────

Example 1: Run all tests and generate all figures
  $ python test_rft_vs_fft.py
  $ python visualize_rft_analysis.py

Example 2: Test RFT on custom signal
  >>> from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
  >>> import numpy as np
  >>> x = np.sin(2 * np.pi * 5 * np.arange(64) / 64)
  >>> X = rft_forward(x)
  >>> x_reconstructed = rft_inverse(X)
  >>> error = np.linalg.norm(x_reconstructed - x)
  >>> print(f"Reconstruction error: {error:.2e}")

Example 3: Generate RFT matrix
  >>> from algorithms.rft.core.resonant_fourier_transform import rft_matrix_canonical as rft_matrix
  >>> Psi = rft_matrix(64)  # 64×64 unitary matrix
  >>> print(f"Matrix shape: {Psi.shape}")
  >>> print(f"Is unitary: {np.allclose(Psi.conj().T @ Psi, np.eye(64))}")

Example 4: Measure unitarity error
  >>> from algorithms.rft.core.resonant_fourier_transform import rft_unitary_error_canonical as rft_unitary_error
  >>> error = rft_unitary_error(128, trials=20)
  >>> print(f"Unitarity error for N=128: {error:.2e}")

───────────────────────────────────────────────────────────────────────────

KEY FINDINGS SUMMARY:
───────────────────────────────────────────────────────────────────────────

✓ ADVANTAGES of RFT:
  • Perfect unitarity (machine precision accuracy)
  • Golden-ratio phase structure (novel spectral distribution)
  • Cryptographic potential (non-standard frequency basis)
  • Research value (unexplored transform space)

⚠ LIMITATIONS of RFT:
  • 5-7× slower than FFT (still O(N log N))
  • Less compression than DCT for smooth signals
  • Non-standard (custom implementation required)

🎯 BEST USE CASES:
  • Cryptographic transforms with reversibility
  • Novel ML feature extraction
  • Signal processing research
  • Patent development
  • Educational demonstrations

❌ NOT RECOMMENDED FOR:
  • Real-time audio/video processing → Use FFT
  • Standard compression → Use DCT
  • Low-power embedded systems → Use Hadamard
  • Maximum performance critical → Use FFT

───────────────────────────────────────────────────────────────────────────

TRANSFORM COMPARISON:
───────────────────────────────────────────────────────────────────────────

Transform | Speed    | Unitarity | Compression | Best For
──────────┼──────────┼───────────┼─────────────┼──────────────────────
RFT       | Medium   | Perfect   | Good        | Crypto, Research
FFT       | Fast     | Perfect   | Good        | General Purpose
DCT       | Medium   | Perfect   | Excellent   | Compression
Hadamard  | Fastest  | Perfect   | Poor        | Low-power Digital

───────────────────────────────────────────────────────────────────────────

PERFORMANCE BENCHMARKS (N=1024):
───────────────────────────────────────────────────────────────────────────

Transform | Time (ms) | Relative to FFT
──────────┼───────────┼─────────────────
FFT       | 0.032     | 1.0× (baseline)
DCT       | 0.048     | 1.5×
RFT       | 0.162     | 5.1×
Hadamard  | 0.015     | 0.5×

───────────────────────────────────────────────────────────────────────────

MATHEMATICAL DEFINITION:
───────────────────────────────────────────────────────────────────────────

Forward RFT:
  Y = D_φ ∘ C_σ ∘ FFT(x)
  
  Where:
  • D_φ[k] = exp(i·2πβ·frac(k/φ))  (Golden-ratio phase)
  • C_σ[k] = exp(iπσk²/N)          (Chirp phase)
  • φ = (1+√5)/2 ≈ 1.618           (Golden ratio)

Inverse RFT:
  x = IFFT(C̄_σ ∘ D̄_φ ∘ Y)

Unitarity:
  Ψ†Ψ = I  ⟹  ‖Ψx‖₂ = ‖x‖₂

───────────────────────────────────────────────────────────────────────────

DOCUMENTATION FILES:
───────────────────────────────────────────────────────────────────────────

• RFT_ANALYSIS_REPORT.md     - Comprehensive analysis results
• test_rft_vs_fft.py         - Basic test suite
• visualize_rft_analysis.py  - Visualization generator
• figures_rft_tikz.tex       - LaTeX publication figures

───────────────────────────────────────────────────────────────────────────

For more information, see:
• /workspaces/quantoniumos/algorithms/rft/core/closed_form_rft.py
• /workspaces/quantoniumos/RFT_ANALYSIS_REPORT.md

╚═══════════════════════════════════════════════════════════════════════════╝
""")
