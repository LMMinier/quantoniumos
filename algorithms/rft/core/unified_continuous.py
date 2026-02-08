# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Unified Continuous Computation API
==================================

This module integrates:
1. RFT (Resonant Fourier Transform) - φ-basis encoding
2. ContinuousComputer - Wave-native operations
3. GoldenReservoir - Dynamics-based temporal computation

Together they form a complete continuous computation framework where:
- Signals enter via RFT encoding (φ-basis)
- Operations stay in continuous amplitude/phase space
- Temporal patterns use reservoir dynamics
- Binary decisions only at final readout (if classification)

ARCHITECTURE
============

   Input Signal
        │
        ▼
┌───────────────────┐
│   RFT ENCODE      │  Map to φ-basis coefficients
│   (φ-basis)       │  Theorem 8 advantage for golden quasi-periodic signals
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   CONTINUOUS      │  Linear algebra on complex amplitudes:
│   OPERATIONS      │  • Filtering (spectral multiply)
│                   │  • Superposition (weighted sum)
│                   │  • Correlation (inner products)
│                   │  • Neural network layers
└───────────────────┘
        │
        ├─── If temporal/sequential ───┐
        │                              ▼
        │                    ┌───────────────────┐
        │                    │   φ-RESERVOIR     │
        │                    │   DYNAMICS        │
        │                    │   (GoldenRC)      │
        │                    └───────────────────┘
        │                              │
        ▼◄─────────────────────────────┘
┌───────────────────┐
│   READOUT         │  Continuous output, OR
│                   │  Classification (argmax), OR
│                   │  Threshold (binary)
└───────────────────┘
        │
        ▼
   Final Output

USE CASES
=========

1. SIGNAL CLASSIFICATION (e.g., ECG normal/abnormal)
   signal → RFT → reservoir → softmax → argmax → class label

2. COMPRESSION
   signal → RFT → top-k selection → sparse encoding

3. FILTERING/DENOISING
   signal → RFT → spectral mask → inverse RFT → clean signal

4. FEATURE EXTRACTION
   signal → RFT → reservoir states → feature vectors

5. PATTERN MATCHING
   template → RFT → |; signal → RFT → ; correlation(template, signal) → similarity
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

# Import existing RFT
from .resonant_fourier_transform import (
    rft_forward, 
    rft_inverse,
    rft_basis_matrix,
    PHI
)

# Import continuous compute components
from .continuous_compute import ContinuousComputer, ComputeState, ReadoutMode
from .golden_reservoir import GoldenReservoir


class TaskType(Enum):
    """Type of computational task."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FILTERING = "filtering"
    COMPRESSION = "compression"
    FEATURE_EXTRACTION = "feature_extraction"


@dataclass
class PipelineConfig:
    """Configuration for unified pipeline."""
    N: int = 256                        # Signal/reservoir dimension
    use_rft: bool = True                # Use RFT (True) or FFT (False)
    use_reservoir: bool = False         # Include reservoir dynamics
    reservoir_nodes: int = 200          # Reservoir size (if used)
    spectral_radius: float = 0.95       # Reservoir spectral radius
    compression_ratio: float = 0.1      # Keep top k% of coefficients
    num_classes: int = 2                # For classification tasks


class UnifiedPipeline:
    """
    Unified continuous computation pipeline.
    
    Combines RFT encoding, continuous operations, and reservoir dynamics
    into a single coherent API.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize unified pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.N = self.config.N
        
        # Build basis
        self._build_basis()
        
        # Initialize continuous computer
        self.cc = ContinuousComputer(self.N)
        
        # Initialize reservoir if needed
        self.reservoir = None
        if self.config.use_reservoir:
            self.reservoir = GoldenReservoir(
                N_reservoir=self.config.reservoir_nodes,
                N_input=self.N,
                N_output=self.config.num_classes,
                spectral_radius=self.config.spectral_radius,
                use_phi_structure=self.config.use_rft
            )
    
    def _build_basis(self):
        """Build RFT or FFT basis matrices."""
        N = self.N
        
        if self.config.use_rft:
            # Use existing canonical RFT basis (Gram-normalized)
            self.U = rft_basis_matrix(N, N, use_gram_normalization=True)
            self.U_H = self.U.conj().T
        else:
            # Standard Fourier
            self.U = np.fft.fft(np.eye(N), axis=0, norm='ortho')
            self.U_H = self.U.conj().T
    
    # =========================================================================
    # ENCODING / DECODING
    # =========================================================================
    
    def encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode signal to spectral coefficients.
        
        Uses RFT (φ-basis) if config.use_rft, else FFT.
        
        Args:
            signal: Time-domain signal, shape (N,) or (N, channels)
        
        Returns:
            coefficients: Complex spectral coefficients
        """
        signal = np.asarray(signal, dtype=np.complex128)
        
        # Handle multi-channel
        if signal.ndim == 2:
            return np.stack([self.encode(signal[:, i]) 
                           for i in range(signal.shape[1])], axis=-1)
        
        # Adjust length
        if len(signal) != self.N:
            if len(signal) < self.N:
                signal = np.pad(signal, (0, self.N - len(signal)))
            else:
                signal = signal[:self.N]
        
        return self.U_H @ signal
    
    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Decode spectral coefficients to time-domain signal.
        
        Args:
            coefficients: Complex spectral coefficients
        
        Returns:
            signal: Time-domain signal
        """
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        
        if coefficients.ndim == 2:
            return np.stack([self.decode(coefficients[:, i])
                           for i in range(coefficients.shape[1])], axis=-1)
        
        return self.U @ coefficients
    
    # =========================================================================
    # CONTINUOUS OPERATIONS (in spectral domain)
    # =========================================================================
    
    def filter_spectral(self, coefficients: np.ndarray, 
                        mask: np.ndarray) -> np.ndarray:
        """
        Apply spectral filter.
        
        Args:
            coefficients: Input coefficients
            mask: Complex filter mask (same shape)
        
        Returns:
            filtered: Filtered coefficients
        """
        return coefficients * mask
    
    def lowpass_filter(self, signal: np.ndarray, 
                       cutoff_fraction: float = 0.25) -> np.ndarray:
        """
        Apply lowpass filter in spectral domain.
        
        Args:
            signal: Time-domain signal
            cutoff_fraction: Keep bottom fraction of frequencies
        
        Returns:
            filtered: Lowpass filtered signal
        """
        c = self.encode(signal)
        
        # Create smooth lowpass mask
        k = np.arange(self.N)
        cutoff = int(self.N * cutoff_fraction)
        mask = np.exp(-np.maximum(k - cutoff, 0)**2 / (cutoff * 0.5)**2)
        
        c_filtered = self.filter_spectral(c, mask)
        return np.real(self.decode(c_filtered))
    
    def compress(self, signal: np.ndarray, 
                 keep_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress signal by keeping top coefficients.
        
        Args:
            signal: Time-domain signal
            keep_ratio: Fraction of coefficients to keep (default: config)
        
        Returns:
            compressed: Sparse coefficient vector
            indices: Indices of kept coefficients
        """
        keep_ratio = keep_ratio or self.config.compression_ratio
        c = self.encode(signal)
        
        k = max(1, int(self.N * keep_ratio))
        magnitudes = np.abs(c)
        indices = np.argsort(magnitudes)[-k:]
        
        compressed = np.zeros_like(c)
        compressed[indices] = c[indices]
        
        return compressed, indices
    
    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Reconstruct signal from compressed coefficients."""
        return np.real(self.decode(compressed))
    
    def correlate(self, c1: np.ndarray, c2: np.ndarray) -> complex:
        """
        Compute correlation between two coefficient vectors.
        
        Returns complex inner product ⟨c1|c2⟩.
        """
        return np.vdot(c1, c2)
    
    def superpose(self, coefficients_list: List[np.ndarray],
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Linear superposition of multiple coefficient vectors.
        
        Args:
            coefficients_list: List of coefficient vectors
            weights: Complex weights (equal if None)
        
        Returns:
            superposed: Weighted sum of coefficients
        """
        if weights is None:
            weights = np.ones(len(coefficients_list)) / len(coefficients_list)
        
        weights = np.asarray(weights, dtype=np.complex128)
        return sum(w * c for w, c in zip(weights, coefficients_list))
    
    # =========================================================================
    # RESERVOIR INTEGRATION
    # =========================================================================
    
    def reservoir_transform(self, signal_sequence: np.ndarray,
                           washout: int = 50) -> np.ndarray:
        """
        Transform signal sequence through reservoir dynamics.
        
        Each time step:
        1. Encode signal slice to RFT
        2. Feed to reservoir (continuous dynamics)
        3. Collect reservoir states
        
        Args:
            signal_sequence: Shape (T, N) or (T,) for 1D signals
            washout: Initial steps to discard
        
        Returns:
            states: Reservoir states, shape (T - washout, N_reservoir)
        """
        if self.reservoir is None:
            raise ValueError("Reservoir not initialized. Set use_reservoir=True in config.")
        
        if signal_sequence.ndim == 1:
            # Assume each sample is a single value; need to encode differently
            # For 1D sequence, feed directly to reservoir
            signal_sequence = signal_sequence.reshape(-1, 1)
        
        T = len(signal_sequence)
        
        # If signal samples are longer than reservoir input, encode first
        if signal_sequence.shape[1] > 1:
            encoded = np.array([self.encode(signal_sequence[t]) 
                               for t in range(T)])
            # Take first N_reservoir magnitudes as input
            encoded = np.abs(encoded[:, :self.reservoir.N_in])
        else:
            encoded = signal_sequence
        
        states, _ = self.reservoir.run(encoded, washout=washout)
        return states
    
    def train_classifier(self, signals: List[np.ndarray], 
                        labels: np.ndarray,
                        washout: int = 50) -> float:
        """
        Train reservoir for classification task.
        
        Args:
            signals: List of signal sequences
            labels: Class labels for each signal
            washout: Reservoir warmup period
        
        Returns:
            training_error: NRMSE on training data
        """
        if self.reservoir is None:
            raise ValueError("Reservoir not initialized. Set use_reservoir=True in config.")
        
        # Encode signals and concatenate
        all_inputs = []
        all_targets = []
        
        for signal, label in zip(signals, labels):
            c = self.encode(signal)
            # Feed magnitude to reservoir
            all_inputs.append(np.abs(c))
            
            # One-hot encode label
            target = np.zeros(self.config.num_classes)
            target[int(label)] = 1.0
            all_targets.append(target)
        
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)
        
        return self.reservoir.train(all_inputs, all_targets, washout=washout)
    
    def classify(self, signal: np.ndarray) -> int:
        """
        Classify a signal using trained reservoir.
        
        Args:
            signal: Input signal
        
        Returns:
            class_index: Predicted class
        """
        if self.reservoir is None:
            raise ValueError("Reservoir not initialized.")
        
        c = self.encode(signal)
        prediction = self.reservoir.predict(np.abs(c).reshape(1, -1), washout=0)
        return int(np.argmax(np.abs(prediction)))
    
    # =========================================================================
    # END-TO-END PIPELINES
    # =========================================================================
    
    def process_signal(self, signal: np.ndarray,
                       task: TaskType = TaskType.FILTERING,
                       **kwargs) -> Union[np.ndarray, int, Tuple]:
        """
        End-to-end signal processing pipeline.
        
        Args:
            signal: Input signal
            task: Type of task to perform
            **kwargs: Task-specific parameters
        
        Returns:
            Result depends on task type
        """
        if task == TaskType.FILTERING:
            cutoff = kwargs.get('cutoff', 0.25)
            return self.lowpass_filter(signal, cutoff)
        
        elif task == TaskType.COMPRESSION:
            ratio = kwargs.get('ratio', self.config.compression_ratio)
            compressed, indices = self.compress(signal, ratio)
            reconstructed = self.decompress(compressed)
            return reconstructed, compressed, indices
        
        elif task == TaskType.FEATURE_EXTRACTION:
            c = self.encode(signal)
            # Return features: magnitude + phase
            return np.concatenate([np.abs(c), np.angle(c)])
        
        elif task == TaskType.CLASSIFICATION:
            if self.reservoir is None:
                # Simple classification via correlation with templates
                templates = kwargs.get('templates', [])
                if not templates:
                    raise ValueError("Need templates for classification without reservoir")
                
                c = self.encode(signal)
                correlations = [np.abs(self.correlate(c, self.encode(t))) 
                               for t in templates]
                return int(np.argmax(correlations))
            else:
                return self.classify(signal)
        
        elif task == TaskType.REGRESSION:
            if self.reservoir is None:
                raise ValueError("Regression requires reservoir")
            c = self.encode(signal)
            prediction = self.reservoir.predict(np.abs(c).reshape(1, -1), washout=0)
            return prediction.flatten()
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    # =========================================================================
    # ANALYSIS / DIAGNOSTICS
    # =========================================================================
    
    def compare_rft_fft(self, signal: np.ndarray) -> dict:
        """
        Compare RFT vs FFT representation for a signal.
        
        Returns metrics showing which basis is more efficient.
        """
        # RFT coefficients (canonical form)
        N = len(signal)
        c_rft = rft_forward(signal, T=N, use_gram_normalization=True)
        
        # FFT coefficients  
        c_fft = np.fft.fft(signal, norm='ortho')
        
        # Sparsity metrics (how many coeffs needed for 99% energy)
        def energy_concentration(c, threshold=0.99):
            magnitudes = np.abs(c)
            sorted_mags = np.sort(magnitudes)[::-1]
            cumsum = np.cumsum(sorted_mags**2) / np.sum(magnitudes**2)
            return np.searchsorted(cumsum, threshold) + 1
        
        k_rft = energy_concentration(c_rft)
        k_fft = energy_concentration(c_fft)
        
        # Entropy
        def spectral_entropy(c):
            p = np.abs(c)**2
            p = p / (np.sum(p) + 1e-10)
            return -np.sum(p * np.log(p + 1e-10))
        
        H_rft = spectral_entropy(c_rft)
        H_fft = spectral_entropy(c_fft)
        
        return {
            'rft_coeffs_99pct_energy': k_rft,
            'fft_coeffs_99pct_energy': k_fft,
            'rft_advantage': k_fft / k_rft,
            'rft_entropy': H_rft,
            'fft_entropy': H_fft,
            'entropy_ratio': H_rft / H_fft
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(use_rft: bool = True, 
                    use_reservoir: bool = False,
                    N: int = 256,
                    **kwargs) -> UnifiedPipeline:
    """
    Create a unified pipeline with sensible defaults.
    
    Args:
        use_rft: Use RFT (True) or FFT (False)
        use_reservoir: Include reservoir dynamics
        N: Signal dimension
        **kwargs: Additional config parameters
    
    Returns:
        Configured pipeline
    """
    config = PipelineConfig(
        N=N,
        use_rft=use_rft,
        use_reservoir=use_reservoir,
        **kwargs
    )
    return UnifiedPipeline(config)


def continuous_classify(signal: np.ndarray,
                        templates: List[np.ndarray],
                        use_rft: bool = True) -> Tuple[int, np.ndarray]:
    """
    Classify signal by correlation with templates.
    
    Simple, fast classification without training.
    100% continuous until final argmax.
    
    Args:
        signal: Signal to classify
        templates: List of template signals (one per class)
        use_rft: Use RFT (True) or FFT (False) encoding
    
    Returns:
        class_index: Predicted class
        correlations: Correlation with each template
    """
    pipe = create_pipeline(use_rft=use_rft, N=len(signal))
    
    c_signal = pipe.encode(signal)
    correlations = np.array([np.abs(pipe.correlate(c_signal, pipe.encode(t)))
                            for t in templates])
    
    return int(np.argmax(correlations)), correlations


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_unified():
    """Demonstrate the unified continuous computation API."""
    
    print("=" * 70)
    print("UNIFIED CONTINUOUS COMPUTATION PIPELINE")
    print("=" * 70)
    
    N = 128
    t = np.linspace(0, 1, N, endpoint=False)
    
    # -------------------------------------------------------------------------
    print("\n1. FILTERING (all continuous)")
    print("-" * 50)
    
    pipe = create_pipeline(use_rft=True, N=N)
    
    # Noisy signal
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(N)
    filtered = pipe.process_signal(signal, TaskType.FILTERING, cutoff=0.2)
    
    noise_before = np.var(signal - np.sin(2 * np.pi * 5 * t))
    noise_after = np.var(filtered - np.sin(2 * np.pi * 5 * t))
    print(f"   Noise variance: {noise_before:.4f} → {noise_after:.4f}")
    print(f"   Noise reduction: {(1 - noise_after/noise_before)*100:.1f}%")
    
    # -------------------------------------------------------------------------
    print("\n2. COMPRESSION (all continuous until reconstruction)")
    print("-" * 50)
    
    reconstructed, compressed, indices = pipe.process_signal(
        signal, TaskType.COMPRESSION, ratio=0.1
    )
    
    error = np.sqrt(np.mean((signal - reconstructed)**2)) / np.std(signal)
    print(f"   Kept {len(indices)}/{N} coefficients ({len(indices)/N*100:.1f}%)")
    print(f"   Reconstruction NRMSE: {error:.4f}")
    
    # -------------------------------------------------------------------------
    print("\n3. TEMPLATE CLASSIFICATION (continuous until argmax)")
    print("-" * 50)
    
    # Templates for different frequency classes
    templates = [
        np.sin(2 * np.pi * 3 * t),   # Class 0: 3 Hz
        np.sin(2 * np.pi * 7 * t),   # Class 1: 7 Hz
        np.sin(2 * np.pi * 12 * t),  # Class 2: 12 Hz
    ]
    
    # Test signals
    test_signals = [
        np.sin(2 * np.pi * 3.1 * t) + 0.1 * np.random.randn(N),  # ~3 Hz
        np.sin(2 * np.pi * 6.9 * t) + 0.1 * np.random.randn(N),  # ~7 Hz
        np.sin(2 * np.pi * 11.8 * t) + 0.1 * np.random.randn(N), # ~12 Hz
    ]
    
    print("   Signal  | True Class | Predicted | Correct?")
    print("   --------|------------|-----------|----------")
    for i, test in enumerate(test_signals):
        pred, corrs = continuous_classify(test, templates, use_rft=True)
        correct = "✓" if pred == i else "✗"
        print(f"   {i+1}       | {i}          | {pred}         | {correct}")
    
    # -------------------------------------------------------------------------
    print("\n4. RFT vs FFT COMPARISON")
    print("-" * 50)
    
    # Create golden quasi-periodic signal (where RFT excels)
    t_fine = np.linspace(0, 10, N, endpoint=False)
    golden_signal = np.sin(2 * np.pi * PHI * t_fine) + \
                   0.5 * np.sin(2 * np.pi * PHI**2 * t_fine) + \
                   0.25 * np.sin(2 * np.pi * PHI**3 * t_fine)
    
    comparison = pipe.compare_rft_fft(golden_signal)
    print(f"   Golden quasi-periodic signal:")
    print(f"     RFT coeffs for 99% energy: {comparison['rft_coeffs_99pct_energy']}")
    print(f"     FFT coeffs for 99% energy: {comparison['fft_coeffs_99pct_energy']}")
    print(f"     RFT advantage: {comparison['rft_advantage']:.2f}x more compact")
    
    # Regular harmonic signal (where FFT excels)
    harmonic_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    comparison2 = pipe.compare_rft_fft(harmonic_signal)
    print(f"\n   Regular harmonic signal:")
    print(f"     RFT coeffs for 99% energy: {comparison2['rft_coeffs_99pct_energy']}")
    print(f"     FFT coeffs for 99% energy: {comparison2['fft_coeffs_99pct_energy']}")
    print(f"     RFT advantage: {comparison2['rft_advantage']:.2f}x")
    
    # -------------------------------------------------------------------------
    print("\n5. RESERVOIR PIPELINE (temporal patterns)")
    print("-" * 50)
    
    pipe_rc = create_pipeline(
        use_rft=True, 
        use_reservoir=True, 
        N=64,
        reservoir_nodes=100,
        num_classes=2
    )
    
    # Generate simple temporal classification task
    # Class 0: increasing frequency chirp
    # Class 1: decreasing frequency chirp
    
    N_rc = 64
    t_rc = np.linspace(0, 1, N_rc)
    
    def chirp_up(duration=1.0):
        return np.sin(2 * np.pi * (5 + 10 * t_rc) * t_rc)
    
    def chirp_down(duration=1.0):
        return np.sin(2 * np.pi * (15 - 10 * t_rc) * t_rc)
    
    # Training data
    train_signals = ([chirp_up() + 0.1*np.random.randn(N_rc) for _ in range(20)] +
                    [chirp_down() + 0.1*np.random.randn(N_rc) for _ in range(20)])
    train_labels = np.array([0]*20 + [1]*20)
    
    # Train reservoir
    train_err = pipe_rc.train_classifier(train_signals, train_labels, washout=10)
    print(f"   Training NRMSE: {train_err:.4f}")
    
    # Test
    test_up = chirp_up() + 0.2*np.random.randn(N_rc)
    test_down = chirp_down() + 0.2*np.random.randn(N_rc)
    
    pred_up = pipe_rc.classify(test_up)
    pred_down = pipe_rc.classify(test_down)
    print(f"   Chirp-up test: predicted class {pred_up} (should be 0)")
    print(f"   Chirp-down test: predicted class {pred_down} (should be 1)")
    
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
We have a complete continuous computation stack:

1. RFT ENCODING: φ-basis with Theorem 8 advantage for golden signals
2. CONTINUOUS OPS: Filtering, compression, correlation - all in ℂ^N
3. RESERVOIR: Temporal patterns via φ-structured dynamics
4. READOUT: Continuous (regression) or discrete (classification)

The ONLY binary decision is the final argmax for classification.
For regression/filtering/compression, we can stay fully continuous.

This IS post-binary computation in a meaningful sense:
- Phase + amplitude encode information (not 0/1)
- Operations are linear algebra on complex vectors
- Dynamics compute via continuous evolution
- Binary only at the boundary (final output, if needed)
""")


if __name__ == "__main__":
    demonstrate_unified()
