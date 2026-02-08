# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Continuous Compute Engine
=========================

A computation framework that operates on CONTINUOUS values (phase, amplitude)
and defers ALL binary decisions to the final readout.

This is NOT simulation of classical logic in wave domain (that's fake).
This IS a legitimate computational paradigm where:

1. Inputs: Continuous signals (audio, images, sensor data)
2. Processing: Linear algebra on complex amplitudes (no thresholding)
3. Output: Continuous values OR single final threshold if classification needed

APPLICATIONS:
- Signal classification (is this ECG normal/abnormal?)
- Anomaly detection (does this signal match the template?)
- Feature extraction (what are the dominant frequencies?)
- Compression (represent with fewer coefficients)
- Filtering (remove noise, extract components)

WHAT MAKES THIS "POST-BINARY":
- All intermediate computations stay in continuous ℂ^N
- No bit extraction until final answer
- Information is carried by amplitude AND phase
- Parallelism is natural (superposition)

PHYSICAL ANALOGS:
- Optical neural networks (light intensity = activation)
- Analog crossbar arrays (conductance = weight)
- Reservoir computing (dynamics = computation)
- Fourier optics (lens = FFT)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

PHI = (1 + np.sqrt(5)) / 2


class ReadoutMode(Enum):
    """How to extract final answer from continuous state."""
    CONTINUOUS = "continuous"      # Return raw continuous values
    ARGMAX = "argmax"              # Return index of max amplitude (classification)
    THRESHOLD = "threshold"        # Binary threshold on amplitude
    PHASE_QUANTIZE = "phase"       # Quantize phase to discrete levels


@dataclass
class ComputeState:
    """
    State of the continuous compute engine.
    
    The "wave function" carries all information as complex amplitudes.
    Phase and magnitude both encode information.
    """
    amplitudes: np.ndarray    # Complex amplitudes in current basis
    basis: str                # 'phi' (RFT), 'fourier' (FFT), or 'signal' (time domain)
    history: List[str]        # Operations applied (for debugging/visualization)
    
    @property
    def N(self) -> int:
        return len(self.amplitudes)
    
    @property 
    def energy(self) -> float:
        """Total energy (conserved under unitary ops)."""
        return float(np.sum(np.abs(self.amplitudes)**2))
    
    @property
    def phase_distribution(self) -> np.ndarray:
        """Phase of each component."""
        return np.angle(self.amplitudes)
    
    @property
    def magnitude_distribution(self) -> np.ndarray:
        """Magnitude of each component."""
        return np.abs(self.amplitudes)
    
    def __repr__(self):
        return f"ComputeState(N={self.N}, basis={self.basis}, energy={self.energy:.4f}, ops={len(self.history)})"


class ContinuousComputer:
    """
    Continuous computation engine.
    
    All operations preserve the continuous nature of the state.
    Binary decisions only happen at readout (if ever).
    
    This models how optical/analog neural networks actually work.
    """
    
    def __init__(self, N: int = 256):
        self.N = N
        self._build_transforms()
    
    def _build_transforms(self):
        """Build basis transformation matrices."""
        N = self.N
        n = np.arange(N)
        
        # φ-grid frequencies
        k = np.arange(N)
        f_phi = np.mod((k + 1) * PHI, 1.0)
        
        # Raw φ-basis
        Phi_raw = np.exp(2j * np.pi * np.outer(n, f_phi)) / np.sqrt(N)
        
        # Gram-orthonormalize to get unitary
        G = Phi_raw.conj().T @ Phi_raw
        G_inv_sqrt = np.linalg.inv(np.linalg.cholesky(G)).conj().T
        self.U_phi = Phi_raw @ G_inv_sqrt
        self.U_phi_H = self.U_phi.conj().T
        
        # Standard Fourier
        self.F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
        self.F_H = self.F.conj().T
    
    # =========================================================================
    # STATE CREATION
    # =========================================================================
    
    def from_signal(self, x: np.ndarray, basis: str = 'phi') -> ComputeState:
        """
        Create compute state from time-domain signal.
        
        Args:
            x: Real or complex signal of length N
            basis: 'phi' for RFT basis, 'fourier' for FFT, 'signal' for time domain
        """
        x = np.asarray(x, dtype=np.complex128)
        if len(x) != self.N:
            # Zero-pad or truncate
            if len(x) < self.N:
                x = np.pad(x, (0, self.N - len(x)))
            else:
                x = x[:self.N]
        
        if basis == 'phi':
            amplitudes = self.U_phi_H @ x
        elif basis == 'fourier':
            amplitudes = self.F_H @ x
        elif basis == 'signal':
            amplitudes = x.copy()
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
        return ComputeState(
            amplitudes=amplitudes,
            basis=basis,
            history=[f"encode({basis})"]
        )
    
    def from_class_index(self, class_idx: int, num_classes: int) -> ComputeState:
        """
        Create state representing a class label.
        
        Uses one-hot encoding in amplitude space.
        This is how classification targets are represented.
        """
        amplitudes = np.zeros(self.N, dtype=np.complex128)
        # Spread the class across multiple frequencies for robustness
        stride = self.N // num_classes
        start = class_idx * stride
        amplitudes[start:start + stride] = 1.0 / np.sqrt(stride)
        
        return ComputeState(
            amplitudes=amplitudes,
            basis='phi',
            history=[f"class({class_idx}/{num_classes})"]
        )
    
    # =========================================================================
    # CONTINUOUS OPERATIONS (no binary decisions!)
    # =========================================================================
    
    def linear_combine(self, states: List[ComputeState], 
                       weights: np.ndarray) -> ComputeState:
        """
        Linear combination of states: Σ w_i |ψ_i⟩
        
        This is the fundamental operation of neural networks
        (weighted sum of inputs).
        
        CONTINUOUS: weights can be any complex numbers.
        """
        if not all(s.basis == states[0].basis for s in states):
            raise ValueError("All states must be in same basis")
        
        weights = np.asarray(weights, dtype=np.complex128)
        result = sum(w * s.amplitudes for w, s in zip(weights, states))
        
        return ComputeState(
            amplitudes=result,
            basis=states[0].basis,
            history=states[0].history + [f"linear_combine({len(states)})"]
        )
    
    def multiply(self, state: ComputeState, 
                 diagonal: np.ndarray) -> ComputeState:
        """
        Pointwise multiply by diagonal matrix.
        
        In spectral basis: this is filtering/gating.
        In signal basis: this is amplitude modulation.
        
        CONTINUOUS: diagonal can be any complex values.
        """
        return ComputeState(
            amplitudes=state.amplitudes * diagonal,
            basis=state.basis,
            history=state.history + ["multiply"]
        )
    
    def convolve_spectral(self, state: ComputeState, 
                          kernel_spectrum: np.ndarray) -> ComputeState:
        """
        Convolution via spectral multiplication.
        
        If state is in Fourier basis, this implements circular convolution.
        
        CONTINUOUS: kernel_spectrum is complex filter response.
        """
        return ComputeState(
            amplitudes=state.amplitudes * kernel_spectrum,
            basis=state.basis,
            history=state.history + ["convolve"]
        )
    
    def unitary_transform(self, state: ComputeState, 
                          U: np.ndarray) -> ComputeState:
        """
        Apply unitary transformation.
        
        Preserves energy (||Uψ|| = ||ψ||).
        This includes rotations, Fourier transforms, etc.
        
        CONTINUOUS: U is complex unitary matrix.
        """
        return ComputeState(
            amplitudes=U @ state.amplitudes,
            basis=state.basis,  # Technically changes, but we track abstractly
            history=state.history + ["unitary"]
        )
    
    def change_basis(self, state: ComputeState, 
                     target_basis: str) -> ComputeState:
        """
        Change representation basis.
        
        CONTINUOUS: Just a unitary change of coordinates.
        """
        if state.basis == target_basis:
            return state
        
        # First go to signal domain
        if state.basis == 'phi':
            signal = self.U_phi @ state.amplitudes  
        elif state.basis == 'fourier':
            signal = self.F @ state.amplitudes
        else:
            signal = state.amplitudes
        
        # Then to target basis
        if target_basis == 'phi':
            amplitudes = self.U_phi_H @ signal
        elif target_basis == 'fourier':
            amplitudes = self.F_H @ signal
        else:
            amplitudes = signal
        
        return ComputeState(
            amplitudes=amplitudes,
            basis=target_basis,
            history=state.history + [f"basis→{target_basis}"]
        )
    
    def nonlinearity(self, state: ComputeState, 
                     mode: str = 'modReLU') -> ComputeState:
        """
        Apply nonlinear activation.
        
        Options (all stay continuous!):
        - 'modReLU': max(|z| - b, 0) * e^{iθ}  (magnitude ReLU, preserve phase)
        - 'cardioid': Re(z) + |z|  (complex → real, preserves some info)
        - 'softmax': softmax on magnitudes (for classification)
        - 'normalize': z / ||z||  (unit sphere projection)
        - 'tanh_mag': tanh(|z|) * e^{iθ}  (bounded magnitude)
        
        CONTINUOUS: No binary thresholding, just smooth nonlinearities.
        """
        z = state.amplitudes
        mag = np.abs(z)
        phase = np.angle(z)
        
        if mode == 'modReLU':
            # Magnitude ReLU with learnable bias (hardcoded here)
            bias = 0.1 * np.mean(mag)
            new_mag = np.maximum(mag - bias, 0)
            result = new_mag * np.exp(1j * phase)
        
        elif mode == 'cardioid':
            # Maps ℂ → ℝ₊ while keeping spectral structure
            result = (np.real(z) + mag).astype(np.complex128)
        
        elif mode == 'softmax':
            # Softmax on magnitudes (for classification readout prep)
            exp_mag = np.exp(mag - np.max(mag))  # Numerical stability
            result = (exp_mag / np.sum(exp_mag)).astype(np.complex128)
        
        elif mode == 'normalize':
            # Project to unit sphere
            norm = np.linalg.norm(z)
            result = z / (norm + 1e-10)
        
        elif mode == 'tanh_mag':
            # Bounded magnitude, preserve phase
            new_mag = np.tanh(mag)
            result = new_mag * np.exp(1j * phase)
        
        else:
            raise ValueError(f"Unknown nonlinearity: {mode}")
        
        return ComputeState(
            amplitudes=result,
            basis=state.basis,
            history=state.history + [f"nonlin({mode})"]
        )
    
    def correlation(self, state1: ComputeState, 
                    state2: ComputeState) -> complex:
        """
        Compute correlation (inner product) between states.
        
        Returns: ⟨ψ1|ψ2⟩ = ψ1† · ψ2
        
        CONTINUOUS: Returns complex number encoding both
        magnitude (similarity) and phase (alignment).
        """
        return np.vdot(state1.amplitudes, state2.amplitudes)
    
    def superpose(self, state1: ComputeState, state2: ComputeState,
                  alpha: complex = 0.5, beta: complex = 0.5) -> ComputeState:
        """
        Quantum-style superposition: α|ψ1⟩ + β|ψ2⟩
        
        CONTINUOUS: Weights can be any complex numbers.
        """
        if state1.basis != state2.basis:
            state2 = self.change_basis(state2, state1.basis)
        
        return ComputeState(
            amplitudes=alpha * state1.amplitudes + beta * state2.amplitudes,
            basis=state1.basis,
            history=state1.history + [f"superpose(α={alpha:.2f},β={beta:.2f})"]
        )
    
    # =========================================================================
    # READOUT (where binary extraction happens, if needed)
    # =========================================================================
    
    def readout(self, state: ComputeState, 
                mode: ReadoutMode = ReadoutMode.CONTINUOUS,
                **kwargs) -> Union[np.ndarray, int, np.ndarray]:
        """
        Extract final answer from continuous state.
        
        THIS IS THE ONLY PLACE WHERE BINARY/DISCRETE DECISIONS HAPPEN.
        
        Modes:
        - CONTINUOUS: Return raw complex amplitudes (no discretization)
        - ARGMAX: Return index of maximum magnitude (classification)
        - THRESHOLD: Binary threshold on magnitudes
        - PHASE_QUANTIZE: Quantize phases to discrete levels
        """
        if mode == ReadoutMode.CONTINUOUS:
            return state.amplitudes
        
        elif mode == ReadoutMode.ARGMAX:
            # Classification: which component has max energy?
            return int(np.argmax(np.abs(state.amplitudes)))
        
        elif mode == ReadoutMode.THRESHOLD:
            threshold = kwargs.get('threshold', 0.5)
            return (np.abs(state.amplitudes) > threshold).astype(int)
        
        elif mode == ReadoutMode.PHASE_QUANTIZE:
            levels = kwargs.get('levels', 4)  # e.g., QPSK = 4
            phases = np.angle(state.amplitudes)
            quantized = np.round(phases / (2 * np.pi / levels)) % levels
            return quantized.astype(int)
        
        else:
            raise ValueError(f"Unknown readout mode: {mode}")
    
    def to_signal(self, state: ComputeState) -> np.ndarray:
        """
        Convert back to time-domain signal.
        
        This IS a form of readout, but preserves continuous nature.
        """
        if state.basis == 'phi':
            return self.U_phi @ state.amplitudes
        elif state.basis == 'fourier':
            return self.F @ state.amplitudes
        else:
            return state.amplitudes


# =============================================================================
# CONTINUOUS NEURAL NETWORK
# =============================================================================

class ContinuousLayer:
    """A single layer in a continuous neural network."""
    
    def __init__(self, in_dim: int, out_dim: int, 
                 nonlinearity: str = 'modReLU'):
        self.W = (np.random.randn(out_dim, in_dim) + 
                  1j * np.random.randn(out_dim, in_dim)) / np.sqrt(in_dim)
        self.b = np.zeros(out_dim, dtype=np.complex128)
        self.nonlinearity = nonlinearity
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: Wx + b, then nonlinearity."""
        z = self.W @ x + self.b
        
        # Apply nonlinearity
        mag = np.abs(z)
        phase = np.angle(z)
        
        if self.nonlinearity == 'modReLU':
            bias = 0.1
            new_mag = np.maximum(mag - bias, 0)
            return new_mag * np.exp(1j * phase)
        elif self.nonlinearity == 'tanh_mag':
            return np.tanh(mag) * np.exp(1j * phase)
        elif self.nonlinearity == 'linear':
            return z
        else:
            return z


class ContinuousNeuralNet:
    """
    Neural network operating entirely in continuous domain.
    
    All layers work with complex amplitudes.
    Only the final readout (if classification) discretizes.
    
    This models how optical neural networks work:
    - Light intensity = activation magnitude
    - Interference = addition
    - Beam splitters = matrix multiply
    - Nonlinear crystals = activation functions
    """
    
    def __init__(self, layer_dims: List[int], 
                 hidden_nonlinearity: str = 'modReLU'):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            nonlin = hidden_nonlinearity if i < len(layer_dims) - 2 else 'linear'
            self.layers.append(ContinuousLayer(
                layer_dims[i], layer_dims[i+1], nonlin
            ))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass, staying continuous throughout."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def classify(self, x: np.ndarray) -> int:
        """Forward pass + argmax readout for classification."""
        out = self.forward(x)
        return int(np.argmax(np.abs(out)))
    
    def continuous_output(self, x: np.ndarray) -> np.ndarray:
        """Forward pass, return continuous output (no discretization)."""
        return self.forward(x)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_continuous_compute():
    """Show end-to-end continuous computation."""
    
    print("=" * 70)
    print("CONTINUOUS COMPUTATION DEMONSTRATION")
    print("=" * 70)
    
    N = 64
    cc = ContinuousComputer(N)
    
    # -------------------------------------------------------------------------
    print("\n1. SIGNAL PROCESSING PIPELINE (all continuous)")
    print("-" * 50)
    
    # Create test signal
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(N)
    
    # Encode to continuous state
    state = cc.from_signal(signal, basis='phi')
    print(f"   Input: {state}")
    
    # Apply spectral filter (lowpass)
    lowpass = np.exp(-np.arange(N) / 10)  # Exponential decay
    state = cc.multiply(state, lowpass)
    print(f"   After filter: energy {state.energy:.4f}")
    
    # Apply nonlinearity
    state = cc.nonlinearity(state, 'modReLU')
    print(f"   After modReLU: energy {state.energy:.4f}")
    
    # Readout (still continuous!)
    output = cc.to_signal(state)
    print(f"   Output signal energy: {np.sum(np.abs(output)**2):.4f}")
    print("   ✓ Entire pipeline stayed continuous. No bits extracted.")
    
    # -------------------------------------------------------------------------
    print("\n2. CLASSIFICATION (continuous until final argmax)")
    print("-" * 50)
    
    # Create a simple continuous neural network
    net = ContinuousNeuralNet([N, 32, 16, 4], hidden_nonlinearity='modReLU')
    
    # Use our state's amplitudes as input
    input_vec = cc.from_signal(signal, basis='fourier').amplitudes
    
    # Forward pass (all continuous)
    continuous_out = net.continuous_output(input_vec)
    print(f"   Output vector (continuous): shape={continuous_out.shape}")
    print(f"   Output magnitudes: {np.abs(continuous_out)}")
    
    # Only NOW do we discretize (classification)
    class_idx = net.classify(input_vec)
    print(f"   Predicted class: {class_idx}")
    print("   ✓ All hidden layers were continuous. Only final argmax is discrete.")
    
    # -------------------------------------------------------------------------
    print("\n3. SUPERPOSITION (quantum-like)")
    print("-" * 50)
    
    # Create two "class template" states
    template_0 = cc.from_class_index(0, num_classes=4)
    template_1 = cc.from_class_index(1, num_classes=4)
    
    # Superpose them (like quantum superposition)
    superposed = cc.superpose(template_0, template_1, 
                               alpha=0.6+0.1j, beta=0.4-0.2j)
    print(f"   Template 0: {template_0}")
    print(f"   Template 1: {template_1}")
    print(f"   Superposition: {superposed}")
    
    # Readout collapses to one class
    result = cc.readout(superposed, ReadoutMode.ARGMAX)
    print(f"   Argmax readout: class {result}")
    print("   ✓ State was in superposition until measurement.")
    
    # -------------------------------------------------------------------------
    print("\n4. CORRELATION-BASED MATCHING (all continuous)")
    print("-" * 50)
    
    # Create templates for matching
    templates = [cc.from_signal(np.sin(2 * np.pi * f * t)) 
                 for f in [3, 5, 7, 11]]
    
    # Test signal
    test = cc.from_signal(np.sin(2 * np.pi * 5.1 * t))  # Close to f=5
    
    # Compute correlations (continuous complex values)
    correlations = [cc.correlation(test, tmpl) for tmpl in templates]
    print(f"   Correlations with f=3,5,7,11 Hz templates:")
    for f, corr in zip([3, 5, 7, 11], correlations):
        print(f"     f={f} Hz: |⟨test|tmpl⟩| = {np.abs(corr):.4f}")
    
    # Best match
    best = np.argmax(np.abs(correlations))
    print(f"   Best match: {[3,5,7,11][best]} Hz")
    print("   ✓ All comparisons done via continuous inner products.")
    
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WHAT WE JUST DID")
    print("=" * 70)
    print("""
✓ Signal Processing: filter, nonlinearity, reconstruction - ALL CONTINUOUS
✓ Neural Network: forward pass through 4 layers - ALL CONTINUOUS
✓ Superposition: quantum-style state combination - CONTINUOUS
✓ Pattern Matching: correlation-based similarity - CONTINUOUS

The ONLY discrete operations were:
- Final classification (argmax on output magnitudes)
- Nothing else!

This IS "post-binary" in a meaningful sense:
- Information encoded in continuous amplitudes & phases
- All intermediate computation stays continuous
- Binary decisions deferred to final readout (if classification needed)
- For regression/signal output, we can skip discretization entirely

This models real analog/optical hardware:
- Photonic neural networks work exactly this way
- Light intensity = activation
- Interference = weighted sum
- Nonlinear materials = activation functions
""")


if __name__ == "__main__":
    demonstrate_continuous_compute()
