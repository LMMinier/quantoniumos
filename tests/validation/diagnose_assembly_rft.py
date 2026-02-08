#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Quick diagnostic: Is assembly RFT computing the correct transform?
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Python reference
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as py_rft_forward
from algorithms.rft.variants import PHI

# Assembly version
try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    ASSEMBLY_AVAILABLE = True
except:
    ASSEMBLY_AVAILABLE = False
    print("Assembly RFT not available")
    sys.exit(1)

N = 16

# Simple test: impulse
print("="*60)
print("Test 1: Impulse Response")
print("="*60)
x_impulse = np.zeros(N, dtype=np.complex128)
x_impulse[0] = 1.0

X_py = py_rft_forward(x_impulse)
print("Python RFT spectrum (first 8):")
print(np.abs(X_py[:8]))

rft_asm = UnitaryRFT(N, RFT_FLAG_QUANTUM_SAFE)
X_asm = rft_asm.forward(x_impulse.copy())
print("\nAssembly RFT spectrum (first 8):")
print(np.abs(X_asm[:8]))

print("\nSpectral difference:")
print(f"  Max abs difference: {np.max(np.abs(X_py - X_asm)):.2e}")
print(f"  Relative error: {np.linalg.norm(X_py - X_asm) / np.linalg.norm(X_py):.2e}")

# Test 2: Quasi-periodic
print("\n" + "="*60)
print("Test 2: Quasi-Periodic Signal (φ-based)")
print("="*60)
t = np.arange(N)
x_quasi = np.exp(2j * np.pi * PHI * t / N).astype(np.complex128)

X_py = py_rft_forward(x_quasi)
X_asm = rft_asm.forward(x_quasi.copy())

print(f"Python RFT:")
print(f"  Max magnitude: {np.max(np.abs(X_py)):.4f}")
print(f"  Sparsity (< 0.01*max): {100*np.sum(np.abs(X_py) < 0.01*np.max(np.abs(X_py)))/N:.1f}%")

print(f"\nAssembly RFT:")
print(f"  Max magnitude: {np.max(np.abs(X_asm)):.4f}")
print(f"  Sparsity (< 0.01*max): {100*np.sum(np.abs(X_asm) < 0.01*np.max(np.abs(X_asm)))/N:.1f}%")

print(f"\nSpectral match: {np.linalg.norm(X_py - X_asm) / np.linalg.norm(X_py):.2e}")

# Test 3: Check if it's just FFT
print("\n" + "="*60)
print("Test 3: Is Assembly RFT just FFT?")
print("="*60)
X_fft = np.fft.fft(x_quasi, norm='ortho')
fft_match = np.linalg.norm(X_asm - X_fft) / np.linalg.norm(X_fft)
print(f"Assembly vs FFT relative error: {fft_match:.2e}")
if fft_match < 0.01:
    print("⚠️  WARNING: Assembly RFT appears to be computing standard FFT!")
else:
    print("✓ Assembly RFT is different from standard FFT")

print("\n" + "="*60)
print("Diagnosis Complete")
print("="*60)
