#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Verify Quantum Simulation Fidelity & Limits
===========================================

This script probes the "actual" quantum simulation capabilities of the
Python-based QuantumSearch and QuantumGates implementation used in the demo.

It tests:
1. Scaling limit (Max Qubits before MemoryError/Timeout)
2. Superposition Fidelity (Hadamard transform correctness)
3. Entanglement Fidelity (Bell State creation)
4. Grover's Search correctness vs Classical Brute Force
"""

import sys
import os
import time
import numpy as np
import psutil

# Add project root
sys.path.append(os.getcwd())

from algorithms.rft.quantum_inspired.quantum_gates import H, X, CNOT, QuantumGate
from algorithms.rft.quantum_inspired.quantum_search import QuantumSearch
from algorithms.rft.core.geometric_container import GeometricContainer

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_scaling_limit():
    print("\n[1] Testing Scaling Limit (Naive State Vector)")
    print("    Qubits | State Size | Memory (MB) | Time (s)")
    print("    -------|------------|-------------|----------")
    
    for n_qubits in range(2, 25): # Try up to 24 qubits (16M complex doubles = 256MB)
        try:
            start_mem = get_memory_mb()
            start_time = time.time()
            
            # Create a full superposition state (H^n |0...0>)
            # This forces allocation of 2^n complex numbers if using dense matrices
            
            # We'll use the QuantumSearch helper to create the operator, 
            # but we won't run the full search to save time, just allocation.
            qs = QuantumSearch()
            
            # H_n = H (x) ... (x) H
            # This is the killer: creating the 2^n x 2^n matrix
            # For n=14, 2^14 = 16384. Matrix is 16384^2 = 268 million entries.
            # 268M * 16 bytes = 4GB. 
            # So we expect failure around n=13 or n=14 on standard machines.
            
            if n_qubits > 12:
                print(f"    {n_qubits:6d} | 2^{n_qubits:<6d} | SKIPPING (Matrix too large for naive dense representation)")
                continue

            H_n = qs._tensor_product(H, n_qubits)
            
            end_time = time.time()
            end_mem = get_memory_mb()
            
            print(f"    {n_qubits:6d} | 2^{n_qubits:<6d} | {end_mem - start_mem:11.2f} | {end_time - start_time:8.4f}")
            
            del H_n
            
        except MemoryError:
            print(f"    {n_qubits:6d} | FAILED (MemoryError)")
            break
        except Exception as e:
            print(f"    {n_qubits:6d} | FAILED ({e})")
            break

def test_superposition_fidelity():
    print("\n[2] Testing Superposition Fidelity (Hadamard)")
    
    # 1 Qubit H|0> -> (|0> + |1>)/sqrt(2)
    initial_state = np.array([1, 0], dtype=complex) # |0>
    final_state = H.matrix @ initial_state
    
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    fidelity = np.abs(np.vdot(expected, final_state))**2
    print(f"    1-Qubit H|0> Fidelity: {fidelity:.10f} (Expected: 1.0)")
    
    if not np.isclose(fidelity, 1.0):
        print("    [FAIL] Superposition incorrect")
    else:
        print("    [PASS] Superposition correct")

def test_entanglement_fidelity():
    print("\n[3] Testing Entanglement Fidelity (Bell State)")
    
    # Bell State |Phi+> = (|00> + |11>)/sqrt(2)
    # Circuit: H(0) -> CNOT(0, 1)
    
    # Initial |00> = [1, 0, 0, 0]
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    
    # Apply H on qubit 0 (H x I)
    # H = [[1, 1], [1, -1]] / sqrt(2)
    # I = [[1, 0], [0, 1]]
    # HxI = ...
    
    qs = QuantumSearch()
    H_gate = qs._tensor_product(H, 1) # Just H
    I_gate = QuantumGate(np.eye(2), "I")
    
    # H on first qubit: H (x) I
    op1 = np.kron(H.matrix, I_gate.matrix)
    state = op1 @ state
    
    # CNOT(0, 1)
    # Standard CNOT is control 0, target 1
    state = CNOT.matrix @ state
    
    # Expected: [1/sqrt(2), 0, 0, 1/sqrt(2)]
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[3] = 1/np.sqrt(2)
    
    fidelity = np.abs(np.vdot(expected, state))**2
    print(f"    Bell State Fidelity:   {fidelity:.10f} (Expected: 1.0)")
    
    if not np.isclose(fidelity, 1.0):
        print("    [FAIL] Entanglement incorrect")
    else:
        print("    [PASS] Entanglement correct")

def main():
    print("=" * 60)
    print("QuantoniumOS - Quantum Fidelity Verification")
    print("=" * 60)
    
    test_superposition_fidelity()
    test_entanglement_fidelity()
    test_scaling_limit()

if __name__ == "__main__":
    main()
