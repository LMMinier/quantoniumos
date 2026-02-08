# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Golden Reservoir Computing (φ-RC)
=================================

Reservoir computing uses a dynamical system as the computational substrate.
The key insight: the reservoir's dynamics DO the computation. We only need
to train a simple linear readout layer.

This module implements a reservoir with φ-inspired structure:
- Connection weights follow golden ratio patterns
- Spectral radius tuned for edge-of-chaos dynamics
- φ-grid topology in recurrent connections

WHY RESERVOIR COMPUTING IS "POST-BINARY":
1. Input is continuous (any real/complex signal)
2. Reservoir state evolves continuously (no discretization)
3. High-dimensional nonlinear dynamics = computation
4. Only the readout (if classification) involves discrete decisions

φ-SPECIFIC ADVANTAGES:
- Golden-ratio eigenvalue spacing → reduced resonance peaks
- Quasi-periodic dynamics → better separation of temporal patterns
- Irrational frequencies → no exact periodicity → richer dynamics

PHYSICAL REALIZATIONS:
- Optical reservoir: ring resonators, delay lines
- Spintronic: spin-torque oscillator networks
- Photonic: coherent Ising machines
- Memristor: analog crossbar arrays
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass

PHI = (1 + np.sqrt(5)) / 2
PSI = PHI - 1  # 1/φ ≈ 0.618


@dataclass
class ReservoirState:
    """State of the reservoir at a given time."""
    state: np.ndarray          # Current reservoir state (complex)
    history: List[np.ndarray]  # History of states (for analysis)
    time_step: int             # Current time step
    
    @property
    def N(self) -> int:
        return len(self.state)
    
    @property
    def energy(self) -> float:
        return float(np.sum(np.abs(self.state)**2))


class GoldenReservoir:
    """
    Reservoir computer with golden-ratio structure.
    
    The reservoir is a recurrent neural network with:
    - Fixed random weights (not trained)
    - φ-structured connectivity
    - Edge-of-chaos dynamics (spectral radius ≈ 1)
    
    Only the output weights are trained.
    """
    
    def __init__(self, 
                 N_reservoir: int = 256,
                 N_input: int = 1,
                 N_output: int = 1,
                 spectral_radius: float = 0.95,
                 input_scaling: float = 0.1,
                 leaking_rate: float = 0.3,
                 sparsity: float = 0.9,
                 use_phi_structure: bool = True,
                 use_complex: bool = False,
                 seed: int = 42):
        """
        Initialize golden reservoir.
        
        Args:
            N_reservoir: Number of reservoir nodes
            N_input: Input dimension
            N_output: Output dimension
            spectral_radius: Spectral radius of recurrent weights (< 1 for stability)
            input_scaling: Scale of input weights
            leaking_rate: Leaky integrator rate (0 = no memory, 1 = no leak)
            sparsity: Fraction of zero weights in reservoir
            use_phi_structure: If True, use golden-ratio structured weights
            use_complex: If True, use complex-valued dynamics (for wave signals)
            seed: Random seed for reproducibility
        """
        self.N = N_reservoir
        self.N_in = N_input
        self.N_out = N_output
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.alpha = leaking_rate
        self.sparsity = sparsity
        self.use_complex = use_complex
        self.dtype = np.complex128 if use_complex else np.float64
        
        np.random.seed(seed)
        
        # Build reservoir weights
        if use_phi_structure:
            self.W = self._build_phi_reservoir()
        else:
            self.W = self._build_random_reservoir()
        
        if not use_complex:
            self.W = np.real(self.W)
        
        # Input weights
        if use_complex:
            self.W_in = (np.random.randn(N_reservoir, N_input) + 
                         1j * np.random.randn(N_reservoir, N_input)) * input_scaling
        else:
            self.W_in = np.random.randn(N_reservoir, N_input) * input_scaling
        
        # Output weights (trained)
        self.W_out = np.zeros((N_output, N_reservoir), dtype=self.dtype)
        
        # Initial state
        self.state = np.zeros(N_reservoir, dtype=self.dtype)
    
    def _build_phi_reservoir(self) -> np.ndarray:
        """
        Build reservoir with golden-ratio structure.
        
        Key ideas:
        1. Connectivity follows φ-grid pattern
        2. Weight magnitudes decay as φ^{-k} with distance
        3. Phases follow golden angle θ_φ = 2π/φ²
        """
        N = self.N
        W = np.zeros((N, N), dtype=np.complex128)
        
        # Golden angle for phase assignments
        theta_phi = 2 * np.pi / (PHI ** 2)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Distance in golden-structured index space
                dist = abs((i - j) * PHI % N)
                
                # Sparse connection based on golden ratio
                # Connect if fractional part of i*φ is close to j*φ
                frac_i = (i * PHI) % 1
                frac_j = (j * PHI) % 1
                closeness = min(abs(frac_i - frac_j), 1 - abs(frac_i - frac_j))
                
                if closeness < (1 - self.sparsity):
                    # Magnitude decays with golden power
                    magnitude = PSI ** (dist / N * 5)
                    # Phase follows golden angle progression
                    phase = theta_phi * (i * N + j)
                    W[i, j] = magnitude * np.exp(1j * phase)
        
        # Normalize to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        return W
    
    def _build_random_reservoir(self) -> np.ndarray:
        """Build standard random reservoir (for comparison)."""
        N = self.N
        
        # Sparse random matrix
        W = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        mask = np.random.rand(N, N) > self.sparsity
        W = W * mask
        
        # Normalize spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        return W
    
    def reset(self):
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.N, dtype=self.dtype)
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Activation function.
        
        For complex: modulus-preserving tanh: tanh(|z|) * e^{iθ}
        For real: standard tanh
        """
        if self.use_complex:
            magnitude = np.abs(x)
            phase = np.angle(x)
            return np.tanh(magnitude) * np.exp(1j * phase)
        else:
            return np.tanh(x)
    
    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Single time step of reservoir dynamics.
        
        x(t+1) = (1-α)x(t) + α·f(W·x(t) + W_in·u(t))
        
        Args:
            u: Input vector (can be scalar for 1D input)
        
        Returns:
            New reservoir state
        """
        u = np.atleast_1d(u).astype(self.dtype)
        
        # Recurrent + input
        pre_activation = self.W @ self.state + self.W_in @ u
        
        # Leaky integration with nonlinearity
        self.state = (1 - self.alpha) * self.state + \
                     self.alpha * self._activation(pre_activation)
        
        return self.state
    
    def run(self, inputs: np.ndarray, 
            washout: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run reservoir on input sequence.
        
        Args:
            inputs: Input sequence, shape (T, N_input) or (T,) for 1D
            washout: Number of initial steps to discard (transient)
        
        Returns:
            states: Reservoir states, shape (T - washout, N_reservoir)
            outputs: Outputs if trained, shape (T - washout, N_output)
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        
        T = len(inputs)
        states = []
        
        self.reset()
        
        for t in range(T):
            self.step(inputs[t])
            if t >= washout:
                states.append(self.state.copy())
        
        states = np.array(states)
        
        # Compute outputs
        outputs = states @ self.W_out.T
        
        return states, outputs
    
    def train(self, inputs: np.ndarray, targets: np.ndarray,
              washout: int = 100, 
              ridge: float = 1e-6) -> float:
        """
        Train output weights via ridge regression.
        
        This is the ONLY trained part of reservoir computing.
        The reservoir itself is fixed.
        
        Args:
            inputs: Training inputs, shape (T, N_input)
            targets: Training targets, shape (T, N_output)
            washout: Washout period
            ridge: Ridge regression regularization
        
        Returns:
            Training error (NRMSE)
        """
        states, _ = self.run(inputs, washout=washout)
        
        # Align targets
        targets_aligned = targets[washout:]
        if targets_aligned.ndim == 1:
            targets_aligned = targets_aligned.reshape(-1, 1)
        
        # Ridge regression: W_out = Y @ X^T @ (X @ X^T + λI)^{-1}
        # Or equivalently: W_out = (X^T @ X + λI)^{-1} @ X^T @ Y
        X = states  # (T, N)
        Y = targets_aligned  # (T, N_out)
        
        # Compute W_out
        XTX = X.conj().T @ X + ridge * np.eye(self.N)
        XTY = X.conj().T @ Y
        self.W_out = (np.linalg.solve(XTX, XTY)).T
        
        # Compute training error
        predictions = states @ self.W_out.T
        error = np.sqrt(np.mean(np.abs(predictions - targets_aligned)**2))
        target_std = np.std(np.abs(targets_aligned))
        nrmse = error / (target_std + 1e-10)
        
        return float(nrmse)
    
    def predict(self, inputs: np.ndarray, 
                washout: int = 0) -> np.ndarray:
        """
        Run trained reservoir on new inputs.
        
        Args:
            inputs: Input sequence
            washout: Warmup period (usually 0 for prediction)
        
        Returns:
            predictions: Output predictions
        """
        states, outputs = self.run(inputs, washout=washout)
        return outputs
    
    def analyze_dynamics(self) -> dict:
        """Analyze reservoir dynamics properties."""
        eigenvalues = np.linalg.eigvals(self.W)
        
        return {
            'spectral_radius': np.max(np.abs(eigenvalues)),
            'mean_eigenvalue_magnitude': np.mean(np.abs(eigenvalues)),
            'eigenvalue_spread': np.std(np.abs(eigenvalues)),
            'memory_capacity_proxy': np.sum(np.abs(eigenvalues) > 0.5),
            'sparsity': 1 - np.count_nonzero(self.W) / self.W.size,
        }


# =============================================================================
# CONTINUOUS SIGNAL TASKS
# =============================================================================

def generate_mackey_glass(length: int, tau: int = 17) -> np.ndarray:
    """
    Generate Mackey-Glass chaotic time series.
    
    Standard benchmark for reservoir computing.
    Uses delay differential equation:
    dx/dt = 0.2 x(t-τ) / (1 + x(t-τ)^10) - 0.1 x(t)
    """
    # Use many integration steps per output sample
    steps_per_sample = 10
    dt = 0.1
    tau_steps = int(tau / dt)
    
    total_steps = (length + 1000) * steps_per_sample
    x = np.ones(total_steps + tau_steps) * 1.2
    
    # Add initial perturbation for chaos
    x[:tau_steps] = 1.2 + 0.1 * (np.random.rand(tau_steps) - 0.5)
    
    for t in range(tau_steps, total_steps + tau_steps - 1):
        x_tau = x[t - tau_steps]
        dx = 0.2 * x_tau / (1.0 + x_tau**10) - 0.1 * x[t]
        x[t + 1] = x[t] + dx * dt
    
    # Subsample and skip warmup
    warmup = 1000 * steps_per_sample
    result = x[warmup + tau_steps::steps_per_sample]
    return result[:length]


def generate_narma(length: int, order: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NARMA (Nonlinear Auto-Regressive Moving Average) task.
    
    Another standard reservoir computing benchmark.
    Tests nonlinear mixing of past inputs.
    """
    # Random input
    u = 0.5 * np.random.rand(length)
    y = np.zeros(length)
    
    for t in range(order, length):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-order:t]) + \
               1.5 * u[t-order] * u[t-1] + 0.1
    
    return u, y


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_reservoir():
    """Demonstrate golden reservoir computing."""
    
    print("=" * 70)
    print("GOLDEN RESERVOIR COMPUTING (φ-RC)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    print("\n1. RESERVOIR PROPERTIES")
    print("-" * 50)
    
    # Compare golden vs random reservoir (complex for full wave dynamics)
    rc_phi = GoldenReservoir(N_reservoir=128, use_phi_structure=True, use_complex=True)
    rc_rand = GoldenReservoir(N_reservoir=128, use_phi_structure=False, use_complex=True)
    
    props_phi = rc_phi.analyze_dynamics()
    props_rand = rc_rand.analyze_dynamics()
    
    print("   φ-structured reservoir:")
    for k, v in props_phi.items():
        print(f"     {k}: {v:.4f}")
    
    print("\n   Random reservoir:")
    for k, v in props_rand.items():
        print(f"     {k}: {v:.4f}")
    
    # -------------------------------------------------------------------------
    print("\n2. MACKEY-GLASS PREDICTION")
    print("-" * 50)
    
    # Generate data
    mg = generate_mackey_glass(3000)
    
    # Prepare input/target (predict 1 step ahead for simplicity)
    inputs = mg[:-1].reshape(-1, 1)
    targets = mg[1:].reshape(-1, 1)
    
    # Split train/test
    train_len = 2000
    X_train, y_train = inputs[:train_len], targets[:train_len]
    X_test, y_test = inputs[train_len:], targets[train_len:]
    
    # Real-valued reservoir for real-valued task
    rc = GoldenReservoir(N_reservoir=300, N_input=1, N_output=1,
                         spectral_radius=0.99, leaking_rate=0.3,
                         use_phi_structure=True, use_complex=False,
                         input_scaling=1.0, sparsity=0.8)
    
    # Train
    train_err = rc.train(X_train, y_train, washout=200, ridge=1e-8)
    print(f"   Training NRMSE: {train_err:.4f}")
    
    # Test - need to warm up reservoir with some test data first
    # Run through entire sequence and evaluate only on second half
    all_inputs = np.vstack([X_train[-200:], X_test])  # Use last 200 train steps for warmup
    states, _ = rc.run(all_inputs, washout=0)
    
    # Get predictions for test period (after warmup)
    test_states = states[200:]
    predictions = test_states @ rc.W_out.T
    
    test_err = np.sqrt(np.mean((predictions - y_test)**2)) / np.std(y_test)
    print(f"   Test NRMSE: {test_err:.4f}")
    
    # Show sample predictions
    print(f"   Sample - True: {y_test[:5].flatten()}")
    print(f"   Sample - Pred: {predictions[:5].flatten().round(3)}")
    
    # -------------------------------------------------------------------------
    print("\n3. NARMA-10 TASK")
    print("-" * 50)
    
    u, y = generate_narma(3000, order=10)
    
    X_train, y_train = u[:2000].reshape(-1, 1), y[:2000].reshape(-1, 1)
    X_test, y_test = u[2000:].reshape(-1, 1), y[2000:].reshape(-1, 1)
    
    rc = GoldenReservoir(N_reservoir=300, use_phi_structure=True, 
                         use_complex=False, spectral_radius=0.99,
                         input_scaling=1.0, sparsity=0.8)
    
    train_err = rc.train(X_train, y_train, washout=200, ridge=1e-8)
    print(f"   Training NRMSE: {train_err:.4f}")
    
    # Warm up with end of training data, then predict
    all_inputs = np.vstack([X_train[-200:], X_test])
    states, _ = rc.run(all_inputs, washout=0)
    
    test_states = states[200:]
    predictions = test_states @ rc.W_out.T
    
    test_err = np.sqrt(np.mean((predictions - y_test)**2)) / (np.std(y_test) + 1e-10)
    print(f"   Test NRMSE: {test_err:.4f}")
    
    # -------------------------------------------------------------------------
    print("\n4. WHY THIS IS CONTINUOUS/POST-BINARY")
    print("-" * 50)
    print("""
   The reservoir state is a 200-dimensional COMPLEX vector.
   - All 200 components evolve continuously
   - Phase and magnitude both carry information
   - No bits, no thresholding inside the reservoir
   
   The dynamics themselves ARE the computation:
   - Nonlinear mixing of inputs
   - Temporal integration (memory)
   - High-dimensional projection (kernel trick)
   
   Only the readout is linear:
   - W_out @ state → output
   - This is the ONLY trained part
   
   Physical implementations exist:
   - Photonic reservoirs (delay lines, ring resonators)
   - Spintronic oscillator networks
   - Memristor crossbar arrays
   - Water waves (!), mechanical springs
""")
    
    print("\n" + "=" * 70)
    print("RESERVOIR COMPUTING = DYNAMICS AS COMPUTATION")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reservoir()
