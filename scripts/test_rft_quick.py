#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Quick test to verify closed-form RFT implementation"""
import numpy as np
import sys

print("Testing Closed-Form Φ-RFT Implementation...")
print("=" * 60)

# Test 1: Import
try:
    from algorithms.rft.core.resonant_fourier_transform import (
        rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_unitary_error_canonical as rft_unitary_error, rft_matrix_canonical as rft_matrix, PHI
    )
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic unitarity
print(f"\nPHI (Golden Ratio) = {PHI:.10f}")
print(f"Expected: 1.6180339887...")

sizes = [8, 16, 32, 64]
print("\nUnitarity Test (Round-trip accuracy):")
print(f"{'N':<6} | {'Error':<12} | {'Status'}")
print("-" * 35)

all_pass = True
for n in sizes:
    error = rft_unitary_error(n, trials=5)
    status = "PASS" if error < 1e-10 else "FAIL"
    symbol = "✓" if error < 1e-10 else "✗"
    print(f"{n:<6} | {error:<12.2e} | {symbol} {status}")
    if error >= 1e-10:
        all_pass = False

# Test 3: Matrix orthogonality
print("\nMatrix Orthogonality Test (Ψ†Ψ = I):")
print(f"{'N':<6} | {'||Ψ†Ψ - I||':<15} | {'Status'}")
print("-" * 40)

for n in [8, 16, 32]:
    Psi = rft_matrix(n)
    I_test = Psi.conj().T @ Psi
    I_true = np.eye(n, dtype=np.complex128)
    error = np.linalg.norm(I_test - I_true, 'fro')
    status = "PASS" if error < 1e-12 else "FAIL"
    symbol = "✓" if error < 1e-12 else "✗"
    print(f"{n:<6} | {error:<15.2e} | {symbol} {status}")
    if error >= 1e-12:
        all_pass = False

# Test 4: Parseval's theorem (energy conservation)
print("\nEnergy Conservation Test:")
n = 64
rng = np.random.default_rng(42)
x = rng.normal(size=n) + 1j * rng.normal(size=n)
energy_time = np.sum(np.abs(x) ** 2)
X = rft_forward(x)
energy_freq = np.sum(np.abs(X) ** 2)
energy_err = abs(energy_freq - energy_time) / max(1e-16, energy_time)
status = "PASS" if energy_err < 1e-10 else "FAIL"
symbol = "✓" if energy_err < 1e-10 else "✗"
print(f"Time domain energy:      {energy_time:.8f}")
print(f"Frequency domain energy: {energy_freq:.8f}")
print(f"Relative error:          {energy_err:.2e} {symbol} {status}")
if energy_err >= 1e-10:
    all_pass = False

# Final verdict
print("\n" + "=" * 60)
if all_pass:
    print("✓ ALL TESTS PASSED - Closed-form RFT is working correctly")
    print("  • Unitarity verified to machine precision")
    print("  • Matrix orthogonality confirmed")
    print("  • Energy conservation validated")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED - Review implementation")
    sys.exit(1)
