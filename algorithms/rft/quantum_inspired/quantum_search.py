#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Quantum Search Implementation - QuantoniumOS
============================================

Grover's algorithm implementation for searching geometric containers.
"""

import numpy as np
import math
from typing import List, Optional
from ..core.geometric_container import GeometricContainer
from .quantum_gates import H, QuantumGate

class QuantumSearch:
    """
    Quantum Search implementation using Grover's Algorithm.
    """

    def __init__(self):
        pass

    def _tensor_product(self, gate: QuantumGate, n: int) -> np.ndarray:
        """Compute tensor product of a gate n times."""
        if n < 1:
            raise ValueError("n must be >= 1")
        result = gate.matrix
        for _ in range(n - 1):
            result = np.kron(result, gate.matrix)
        return result

    def search(self, containers: List[GeometricContainer], target_index: int) -> Optional[GeometricContainer]:
        """
        Search for a target container using Grover's algorithm simulation.
        
        Args:
            containers: List of GeometricContainers to search.
            target_index: The index of the target container (for Oracle construction).
            
        Returns:
            The found GeometricContainer or None if not found.
        """
        if not containers:
            return None
            
        N = len(containers)
        if target_index < 0 or target_index >= N:
            raise ValueError("Target index out of bounds")

        # 1. Determine number of qubits n
        if N == 0:
            return None
        n_qubits = math.ceil(math.log2(N))
        if n_qubits == 0:
            n_qubits = 1
        dim = 2 ** n_qubits
        
        # 2. Initialize state vector |s> = H^n |0>
        # Start with |0...0>
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        # Create H^n matrix
        H_n_matrix = self._tensor_product(H, n_qubits)
        
        # Apply H^n to get superposition
        state = H_n_matrix @ state
        
        # 3. Construct Oracle
        # Diagonal matrix with -1 at target_index, 1 elsewhere
        oracle_matrix = np.eye(dim, dtype=complex)
        oracle_matrix[target_index, target_index] = -1.0
        
        # 4. Construct Diffusion Operator
        # D = H^n (2|0><0| - I) H^n
        
        # Phase shift matrix (2|0><0| - I)
        # |0><0| is matrix with 1 at [0,0] and 0 elsewhere.
        # 2|0><0| - I has 1 at [0,0] and -1 elsewhere on diagonal.
        phase_shift_matrix = -np.eye(dim, dtype=complex)
        phase_shift_matrix[0, 0] = 1.0
        
        # Construct Diffusion matrix
        diffusion_matrix = H_n_matrix @ phase_shift_matrix @ H_n_matrix
        
        # 5. Run Simulation
        # Optimal number of iterations k approx floor((pi/4) * sqrt(dim))
        k = int(math.floor((math.pi / 4) * math.sqrt(dim)))
        if k == 0: k = 1
        
        for _ in range(k):
            # Apply Oracle
            state = oracle_matrix @ state
            # Apply Diffusion
            state = diffusion_matrix @ state
            
        # 6. Measure
        probabilities = np.abs(state) ** 2
        measured_index = np.argmax(probabilities)
        
        if measured_index < N:
            return containers[measured_index]
        else:
            return None
