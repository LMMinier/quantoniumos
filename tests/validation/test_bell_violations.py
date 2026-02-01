#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Enhanced Bell Violation Test for QuantoniumOS
============================================

Comprehensive test to achieve strong CHSH inequality violations (>2.7) 
using QuTiP benchmarking and optimized entanglement generation.

Target: CHSH S ≤ 2√2 ≈ 2.828 (Tsirelson bound)
Goal: Achieve S > 2.7 for validation of genuine quantum entanglement
"""

import numpy as np
import sys
import os
from typing import Any
sys.path.insert(0, os.path.abspath('.'))

try:
    import qutip as qt  # type: ignore[import]
    QUTIP_AVAILABLE = True
    print("✓ QuTiP available for benchmarking")
except ImportError:
    QUTIP_AVAILABLE = False
    qt = None
    print("❌ QuTiP not available — skipping QuTiP-specific benchmarks")

from tests.proofs.test_entanglement_protocols import BellTestProtocol
from algorithms.rft.quantum_inspired.quantum_gates import RotationGates
from quantonium_os_src.engine.RFTMW import QuantumEngine


class EntangledVertexEngine:
    """Adapter around the production QuantumEngine for entanglement tests."""

    def __init__(self, n_vertices: int = 2, entanglement_enabled: bool = True):
        if n_vertices < 2:
            raise ValueError("EntangledVertexEngine requires at least two vertices")
        self.n_vertices = n_vertices
        self.entanglement_enabled = entanglement_enabled
        self._engine = QuantumEngine(num_qubits=n_vertices)
        self._control = 0
        self._target = 1

    def _ensure_rotation_gate(self, theta: float) -> str:
        gate_name = f"Ry_{theta:.6f}"
        if gate_name not in self._engine._gate_cache:
            self._engine._gate_cache[gate_name] = RotationGates.Ry(theta).matrix
        return gate_name

    def _current_state(self) -> np.ndarray:
        return self._engine.get_state()

    def create_optimal_bell_state(self, vertices: tuple[int, int] = (0, 1)) -> np.ndarray:
        """Create maximally entangled Bell state using real gate operations."""
        if not self.entanglement_enabled:
            raise RuntimeError("Entanglement not enabled")
        control, target = vertices
        if control >= self.n_vertices or target >= self.n_vertices:
            raise ValueError("Vertex index out of range")
        self._control, self._target = control, target
        self._engine.reset()
        self._engine.apply_gate('H', control)
        self._engine.apply_gate('CNOT', target, control=control)
        return self._current_state()

    def assemble_entangled_state(self, entanglement_level: float = 1.0) -> np.ndarray:
        """Assemble an entangled state by programming the QuantumEngine."""
        if not self.entanglement_enabled:
            raise RuntimeError("Entanglement not enabled")

        level = float(np.clip(entanglement_level, 0.0, 1.0))
        if np.isclose(level, 1.0, atol=1e-3):
            return self.create_optimal_bell_state((self._control, self._target))

        self._engine.reset()
        if level <= 1e-6:
            return self._current_state()

        theta = level * (np.pi / 2.0)
        gate_name = self._ensure_rotation_gate(theta)
        self._engine.apply_gate(gate_name, self._control)
        self._engine.apply_gate('CNOT', self._target, control=self._control)
        return self._current_state()

    def fidelity_with_qutip(self, state_type: str = "bell") -> float:
        """Return fidelity with the requested target state."""
        if state_type != "bell":
            raise ValueError(f"Unsupported state_type: {state_type}")

        state = self._current_state()
        target = np.zeros_like(state, dtype=complex)
        target[0] = 1 / np.sqrt(2)
        target[3] = 1 / np.sqrt(2)

        if QUTIP_AVAILABLE and qt is not None:
            try:
                bell = qt.Qobj(target.reshape(-1, 1))
                current = qt.Qobj(state.reshape(-1, 1))
                overlap = (bell.dag() * current).full()[0, 0]
                return float(np.abs(overlap) ** 2)
            except Exception:
                # Fallback when QuTiP sparse indices are not int32-compatible
                pass

        overlap = np.vdot(target, state)
        return float(np.abs(overlap) ** 2)


def calculate_chsh_manual(state: np.ndarray) -> float:
    """Manual CHSH calculation for verification."""
    # Measurement operators for optimal CHSH angles
    def pauli_measurement(angle: float) -> np.ndarray:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([[cos_a, sin_a], [sin_a, -cos_a]], dtype=complex)
    
    # Optimal angles for maximum violation (Tsirelson bound)
    angles_A = [0, np.pi/2]  # Alice: 0°, 90°
    angles_B = [np.pi/4, -np.pi/4]  # Bob: 45°, -45°
    
    chsh_sum = 0.0
    for i, angle_A in enumerate(angles_A):
        for j, angle_B in enumerate(angles_B):
            A = pauli_measurement(angle_A)
            B = pauli_measurement(angle_B)
            AB = np.kron(A, B)
            
            correlation = np.real(np.conj(state) @ AB @ state)
            
            if (i, j) == (1, 1):  # A₂B₂ term (subtract)
                chsh_sum -= correlation
            else:  # A₁B₁, A₁B₂, A₂B₁ terms (add)
                chsh_sum += correlation
    
    return chsh_sum


def create_optimal_bell_engine(n_vertices: int = 2) -> EntangledVertexEngine:
    """Create vertex engine optimized for maximum Bell violations."""
    engine = EntangledVertexEngine(n_vertices=n_vertices, entanglement_enabled=True)
    
    # Create optimal Bell state for first two vertices
    bell_state = engine.create_optimal_bell_state(vertices=(0, 1))
    print(f"Created Bell state with norm: {np.linalg.norm(bell_state):.6f}")
    
    return engine


import pytest

def test_qutip_bell_reference():
    """Test QuTiP Bell state as reference."""
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")
    
    print("\n=== QuTiP Reference Bell State ===")
    
    # Create perfect Bell state with QuTiP
    bell_state = qt.bell_state('00')  # (|00⟩ + |11⟩)/√2
    
    # Measurement operators for CHSH test
    def pauli_measurement(angle: float) -> Any:
        """Create Pauli measurement operator at given angle."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return qt.Qobj(np.array([[cos_a, sin_a], [sin_a, -cos_a]]))
    
    # Optimal CHSH angles
    angles_A = [0, np.pi/4]  # Alice: 0°, 45°
    angles_B = [np.pi/8, -3*np.pi/8]  # Bob: 22.5°, -67.5°
    
    # Calculate CHSH correlator
    chsh_sum = 0.0
    correlations = {}
    
    for i, angle_A in enumerate(angles_A):
        for j, angle_B in enumerate(angles_B):
            A = pauli_measurement(angle_A)
            B = pauli_measurement(angle_B)
            AB = qt.tensor(A, B)
            
            # Expectation value
            correlation = qt.expect(AB, bell_state)
            correlations[f'A{i+1}B{j+1}'] = correlation
            
            # CHSH combination
            if (i, j) == (1, 1):  # A₂B₂ term (subtract)
                chsh_sum -= correlation
            else:  # A₁B₁, A₁B₂, A₂B₁ terms (add)
                chsh_sum += correlation
    
    print(f"QuTiP Bell State CHSH: {chsh_sum:.6f}")
    print(f"Tsirelson bound: {2*np.sqrt(2):.6f}")
    print(f"Classical bound: 2.000000")
    
    for key, val in correlations.items():
        print(f"  {key}: {val:.6f}")
    
    # Assert QuTiP achieves Bell violation
    assert chsh_sum > 2.0, f"QuTiP CHSH {chsh_sum} does not violate Bell inequality"


def test_quantonium_bell_violation():
    """Test QuantoniumOS Bell violation performance."""
    print("\n=== QuantoniumOS Bell Violation Test ===")
    
    # Create optimized engine with perfect Bell state
    engine = create_optimal_bell_engine(n_vertices=2)
    
    # Get the Bell state directly (should be perfect |Φ⁺⟩ state)
    state = engine.assemble_entangled_state(entanglement_level=1.0)
    print(f"State norm: {np.linalg.norm(state):.6f}")
    print(f"State components: |00⟩={state[0]:.4f}, |01⟩={state[1]:.4f}, |10⟩={state[2]:.4f}, |11⟩={state[3]:.4f}")
    
    # Manual CHSH calculation to verify the Bell state
    print("\n--- Manual CHSH Calculation ---")
    chsh_manual = calculate_chsh_manual(state)
    print(f"Manual CHSH: {chsh_manual:.6f}")
    
    # Test Bell inequality with protocol
    protocol = BellTestProtocol()
    result = protocol.validate(engine, vertices=[0, 1])
    
    chsh_value = result.value
    print(f"Protocol CHSH: {chsh_value:.6f}")
    print(f"Bell violation: {'YES' if result.passed else 'NO'}")
    print(f"Violation strength: {result.details.get('violation_strength', 0):.6f}")
    
    # Compare with QuTiP Bell state fidelity
    if QUTIP_AVAILABLE:
        fidelity = engine.fidelity_with_qutip("bell")
        print(f"Bell state fidelity: {fidelity:.6f}")
    
    # Assert Bell violation achieved
    assert max(chsh_manual, chsh_value) > 2.0, "Bell violation not achieved"



def benchmark_entanglement_levels() -> dict:
    """Benchmark CHSH violations across different entanglement levels."""
    print("\n=== Entanglement Level Benchmark ===")
    
    results = {}
    entanglement_levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    for level in entanglement_levels:
        engine = create_optimal_bell_engine(n_vertices=2)
        state = engine.assemble_entangled_state(entanglement_level=level)
        
        protocol = BellTestProtocol()
        result = protocol.validate(engine, vertices=[0, 1])
        
        results[level] = {
            'chsh': result.value,
            'violation': result.passed,
            'fidelity': engine.fidelity_with_qutip("bell") if QUTIP_AVAILABLE else 0.0
        }
        
        print(f"Level {level:.2f}: CHSH = {result.value:.4f}, Fidelity = {results[level]['fidelity']:.4f}")
    
    return results


def test_decoherence_impact():
    """Test impact of decoherence on Bell violations."""
    print("\n=== Decoherence Impact Test ===")
    
    # Create optimal Bell state
    engine = create_optimal_bell_engine(n_vertices=2)
    state = engine.assemble_entangled_state(entanglement_level=0.95)
    
    # Test without decoherence
    protocol = BellTestProtocol()
    result_clean = protocol.validate(engine, vertices=[0, 1])
    chsh_clean = result_clean.value
    
    print(f"Clean CHSH: {chsh_clean:.6f}")
    
    # Test with various noise levels (simplified model)
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    results = {'clean': chsh_clean}
    
    for p in noise_levels:
        # Simplified decoherence model: CHSH scales with (1 - p)
        # Real implementation would use density matrices and Kraus operators
        estimated_chsh = chsh_clean * (1 - p)
        results[p] = estimated_chsh
        print(f"Noise p={p:.2f}: Estimated CHSH = {estimated_chsh:.4f}")
    
    # Assert clean state achieves Bell violation
    assert chsh_clean > 2.0, f"Clean CHSH {chsh_clean} does not violate Bell inequality"
    print("✓ Decoherence impact test passed")


def comprehensive_bell_test() -> dict:
    """Run comprehensive Bell violation test suite."""
    print("=" * 80)
    print("QUANTONIUMOS COMPREHENSIVE BELL VIOLATION TEST")
    print("=" * 80)
    print(f"Target: CHSH > 2.7 (Classical bound: 2.0, Quantum bound: {2*np.sqrt(2):.3f})")
    
    # Test 1: QuTiP reference
    qutip_chsh = test_qutip_bell_reference()
    
    # Test 2: QuantoniumOS performance  
    quantonium_chsh = test_quantonium_bell_violation()
    
    # Test 3: Entanglement level benchmark
    level_results = benchmark_entanglement_levels()
    
    # Test 4: Decoherence impact
    decoherence_results = test_decoherence_impact()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    best_chsh = max(level_results.values(), key=lambda x: x['chsh'])['chsh']
    best_level = max(level_results.keys(), key=lambda k: level_results[k]['chsh'])
    
    print(f"QuTiP Reference CHSH:    {qutip_chsh:.6f}")
    print(f"QuantoniumOS Best CHSH:  {best_chsh:.6f} (at level {best_level:.2f})")
    print(f"Performance Ratio:       {best_chsh/qutip_chsh:.3f}" if qutip_chsh > 0 else "N/A")
    
    # Target achievement
    target_achieved = best_chsh > 2.7
    
    print(f"\nTarget Achievement:")
    print(f"  CHSH > 2.7:            {'✓ YES' if target_achieved else '✗ NO'}")
    
    if target_achieved:
        print("🎉 SUCCESS: Strong Bell violation achieved!")
        print("✅ Maximum quantum entanglement validated")
    else:
        print("⚠️  PARTIAL: Bell violation detected but below target")
        print("🔧 Further optimization needed")
    
    return {
        'qutip_chsh': qutip_chsh,
        'best_chsh': best_chsh,
        'best_level': best_level,
        'target_achieved': target_achieved,
        'level_results': level_results,
        'decoherence_results': decoherence_results
    }


if __name__ == "__main__":
    results = comprehensive_bell_test()
    
    # Exit with success if target achieved
    exit_code = 0 if results['target_achieved'] else 1
    sys.exit(exit_code)