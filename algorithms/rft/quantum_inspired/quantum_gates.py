#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Quantum Gates Implementation - QuantoniumOS
===========================================

Complete set of quantum gates implemented for vertex-based quantum computing.
Supports both standard qubit operations and RFT-enhanced transformations.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Callable
import math

# Try to import RFT components
try:
    from ..assembly.python_bindings import UnitaryRFT, RFT_FLAG_UNITARY
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class QuantumGate:
    """Base class for quantum gates with matrix representation."""

    def __init__(self, matrix: np.ndarray, name: str = "Gate"):
        """Initialize quantum gate with unitary matrix.

        Args:
            matrix: Unitary matrix representing the gate
            name: Human-readable name for the gate
        """
        self.matrix = np.array(matrix, dtype=complex)
        self.name = name
        self.size = matrix.shape[0]

        # Validate unitarity
        if not self._is_unitary():
            raise ValueError(f"Gate {name} is not unitary")

    def _is_unitary(self, tolerance: float = 1e-10) -> bool:
        """Check if the gate matrix is unitary."""
        identity = np.eye(self.size, dtype=complex)
        product = self.matrix @ self.matrix.conj().T
        return np.allclose(product, identity, atol=tolerance)

    def __repr__(self) -> str:
        return f"{self.name}({self.size}x{self.size})"

    def __matmul__(self, other: 'QuantumGate') -> 'QuantumGate':
        """Compose gates: self @ other"""
        if self.size != other.size:
            raise ValueError("Gate sizes must match for composition")
        return QuantumGate(self.matrix @ other.matrix, f"{self.name}@{other.name}")

class PauliGates:
    """Pauli gate implementations."""

    @staticmethod
    def X() -> QuantumGate:
        """Pauli-X (NOT) gate."""
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return QuantumGate(matrix, "Pauli-X")

    @staticmethod
    def Y() -> QuantumGate:
        """Pauli-Y gate."""
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return QuantumGate(matrix, "Pauli-Y")

    @staticmethod
    def Z() -> QuantumGate:
        """Pauli-Z gate."""
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        return QuantumGate(matrix, "Pauli-Z")

class RotationGates:
    """Rotation gate implementations."""

    @staticmethod
    def Rx(theta: float) -> QuantumGate:
        """Rotation around X-axis."""
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)
        matrix = np.array([
            [cos, -1j*sin],
            [-1j*sin, cos]
        ], dtype=complex)
        return QuantumGate(matrix, f"Rx({theta:.3f})")

    @staticmethod
    def Ry(theta: float) -> QuantumGate:
        """Rotation around Y-axis."""
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)
        matrix = np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=complex)
        return QuantumGate(matrix, f"Ry({theta:.3f})")

    @staticmethod
    def Rz(theta: float) -> QuantumGate:
        """Rotation around Z-axis."""
        phase = np.exp(1j * theta/2)
        matrix = np.array([
            [phase.conj(), 0],
            [0, phase]
        ], dtype=complex)
        return QuantumGate(matrix, f"Rz({theta:.3f})")

class PhaseGates:
    """Phase gate implementations."""

    @staticmethod
    def S() -> QuantumGate:
        """S gate (Z^0.5)."""
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        return QuantumGate(matrix, "S")

    @staticmethod
    def T() -> QuantumGate:
        """T gate (Z^0.25)."""
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
        return QuantumGate(matrix, "T")

    @staticmethod
    def P(phi: float) -> QuantumGate:
        """General phase gate."""
        matrix = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
        return QuantumGate(matrix, f"P({phi:.3f})")

class HadamardGates:
    """Hadamard and related gates."""

    @staticmethod
    def H() -> QuantumGate:
        """Hadamard gate."""
        factor = 1/np.sqrt(2)
        matrix = np.array([
            [factor, factor],
            [factor, -factor]
        ], dtype=complex)
        return QuantumGate(matrix, "Hadamard")

class ControlledGates:
    """Controlled gate implementations."""

    @staticmethod
    def CNOT() -> QuantumGate:
        """Controlled-NOT gate."""
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        return QuantumGate(matrix, "CNOT")

    @staticmethod
    def CZ() -> QuantumGate:
        """Controlled-Z gate."""
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        return QuantumGate(matrix, "CZ")

    @staticmethod
    def Toffoli() -> QuantumGate:
        """Toffoli (CCNOT) gate."""
        matrix = np.eye(8, dtype=complex)
        # Flip the |110⟩ ↔ |111⟩ elements
        matrix[6, 6] = 0  # |110⟩⟨110|
        matrix[6, 7] = 1  # |110⟩⟨111|
        matrix[7, 6] = 1  # |111⟩⟨110|
        matrix[7, 7] = 0  # |111⟩⟨111|
        return QuantumGate(matrix, "Toffoli")

class RFTGates:
    """RFT-enhanced quantum gates."""

    @staticmethod
    def rft_hadamard(size: int) -> QuantumGate:
        """RFT-enhanced Hadamard gate for larger systems."""
        if not RFT_AVAILABLE:
            raise RuntimeError("RFT not available for enhanced gates")

        try:
            rft = UnitaryRFT(size, RFT_FLAG_UNITARY)
            # Create RFT-based Hadamard-like transformation
            # This is a simplified version - full implementation would use RFT basis
            factor = 1/np.sqrt(size)
            matrix = np.full((size, size), factor, dtype=complex)
            return QuantumGate(matrix, f"RFT-H({size})")
        except Exception:
            # Fallback to standard Hadamard for 2x2
            if size == 2:
                return HadamardGates.H()
            else:
                raise RuntimeError("RFT-enhanced gates require compiled assembly")

    @staticmethod
    def rft_phase_gate(size: int, phi: float) -> QuantumGate:
        """RFT-enhanced phase gate."""
        if not RFT_AVAILABLE:
            raise RuntimeError("RFT not available for enhanced gates")

        try:
            # Create diagonal phase matrix using RFT parameterization
            phases = np.exp(1j * phi * np.arange(size) / size)
            matrix = np.diag(phases)
            return QuantumGate(matrix, f"RFT-P({size}, {phi:.3f})")
        except Exception:
            raise RuntimeError("RFT-enhanced gates require compiled assembly")

class QuantumGates:
    """Unified interface to all quantum gate types."""
    
    def __init__(self):
        """Initialize quantum gates collection."""
        pass
    
    # Pauli gates
    @staticmethod
    def X() -> QuantumGate:
        """Pauli-X (NOT) gate."""
        return PauliGates.X()
    
    @staticmethod
    def Y() -> QuantumGate:
        """Pauli-Y gate."""
        return PauliGates.Y()
    
    @staticmethod
    def Z() -> QuantumGate:
        """Pauli-Z gate."""
        return PauliGates.Z()
    
    # Rotation gates
    @staticmethod
    def Rx(theta: float) -> QuantumGate:
        """Rotation around X-axis."""
        return RotationGates.Rx(theta)
    
    @staticmethod
    def Ry(theta: float) -> QuantumGate:
        """Rotation around Y-axis."""
        return RotationGates.Ry(theta)
    
    @staticmethod
    def Rz(theta: float) -> QuantumGate:
        """Rotation around Z-axis."""
        return RotationGates.Rz(theta)
    
    # Phase gates
    @staticmethod
    def S() -> QuantumGate:
        """S gate (Z^0.5)."""
        return PhaseGates.S()
    
    @staticmethod
    def T() -> QuantumGate:
        """T gate (Z^0.25)."""
        return PhaseGates.T()
    
    @staticmethod
    def P(phi: float) -> QuantumGate:
        """General phase gate."""
        return PhaseGates.P(phi)
    
    # Hadamard gates
    @staticmethod
    def H() -> QuantumGate:
        """Hadamard gate."""
        return HadamardGates.H()
    
    # Controlled gates
    @staticmethod
    def CNOT() -> QuantumGate:
        """Controlled-NOT gate."""
        return ControlledGates.CNOT()
    
    @staticmethod
    def CZ() -> QuantumGate:
        """Controlled-Z gate."""
        return ControlledGates.CZ()
    
    @staticmethod
    def Toffoli() -> QuantumGate:
        """Toffoli (CCNOT) gate."""
        return ControlledGates.Toffoli()
    
    # RFT-enhanced gates
    @staticmethod
    def rft_hadamard(size: int) -> QuantumGate:
        """RFT-enhanced Hadamard gate."""
        return RFTGates.rft_hadamard(size)
    
    @staticmethod
    def rft_phase_gate(size: int, phi: float) -> QuantumGate:
        """RFT-enhanced phase gate."""
        return RFTGates.rft_phase_gate(size, phi)

# Convenience instances for common gates
I = QuantumGate(np.eye(2, dtype=complex), "Identity")
X = PauliGates.X()
Y = PauliGates.Y()
Z = PauliGates.Z()
H = HadamardGates.H()
S = PhaseGates.S()
T = PhaseGates.T()
CNOT = ControlledGates.CNOT()
CZ = ControlledGates.CZ()

# Export all gate classes and instances
__all__ = [
    'QuantumGate',
    'PauliGates', 'RotationGates', 'PhaseGates', 'HadamardGates', 'ControlledGates', 'RFTGates', 'QuantumGates',
    'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CNOT', 'CZ'
]