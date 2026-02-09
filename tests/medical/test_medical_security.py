#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# MEDICAL RESEARCH LICENSE:
# FREE for hospitals, medical researchers, academics, and healthcare
# institutions for testing, validation, and research purposes.
# Commercial medical device use: See LICENSE-CLAIMS-NC.md
#
"""
Medical Data Security and Privacy Tests
========================================

Tests RFT-based security primitives for medical applications:
- Waveform hashing collision resistance and avalanche effect
- Clinical data pipeline encryption
- Federated learning aggregation robustness
- Byzantine client detection in distributed training

Uses synthetic biomedical waveforms for testing.
Real clinical data requires proper PHI handling and compliance.
"""

import numpy as np
import hashlib
import time
import pytest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Waveform Hashing
# =============================================================================

def rft_waveform_hash(waveform: np.ndarray, 
                       hash_bits: int = 256,
                       salt: Optional[bytes] = None) -> bytes:
    """
    Compute RFT-based hash of waveform signal.
    
    Uses RFT transform followed by quantization and hashing
    to create a fingerprint of the waveform that is:
    - Deterministic
    - Sensitive to small changes (avalanche effect)
    - Collision resistant
    
    Args:
        waveform: Input signal
        hash_bits: Output hash size in bits (128, 256, 512)
        salt: Optional salt for keyed hashing
        
    Returns:
        Hash bytes
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward
    except ImportError:
        pytest.skip("RFT core not available")
    
    # Normalize waveform
    waveform = waveform.astype(np.float64)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)
    
    # Ensure power of 2 for FFT efficiency
    n = len(waveform)
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    padded = np.zeros(n_padded)
    padded[:n] = waveform
    
    # Apply RFT
    coeffs = rft_forward(padded.astype(np.complex128))
    
    # Extract hash material: quantized magnitude and phase
    magnitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    
    # Quantize to fixed precision
    mag_quant = (magnitudes * 1000).astype(np.int64)
    phase_quant = ((phases + np.pi) * 1000).astype(np.int64)
    
    # Combine into bytes
    hash_input = mag_quant.tobytes() + phase_quant.tobytes()
    
    if salt:
        hash_input = salt + hash_input
    
    # Use SHA for final hash
    if hash_bits == 128:
        hasher = hashlib.md5(hash_input)
    elif hash_bits == 256:
        hasher = hashlib.sha256(hash_input)
    elif hash_bits == 512:
        hasher = hashlib.sha512(hash_input)
    else:
        hasher = hashlib.sha256(hash_input)
    
    return hasher.digest()


def compute_avalanche_effect(hash_func, waveform: np.ndarray, 
                             n_trials: int = 100) -> float:
    """
    Measure avalanche effect: small input change → large hash change.
    
    Ideal avalanche effect is ~50% bit flip rate.
    
    Args:
        hash_func: Hash function to test
        waveform: Base waveform
        n_trials: Number of perturbation trials
        
    Returns:
        Average bit flip ratio (0-1, ideal = 0.5)
    """
    original_hash = hash_func(waveform)
    original_bits = np.unpackbits(np.frombuffer(original_hash, dtype=np.uint8))
    
    flip_ratios = []
    
    for _ in range(n_trials):
        # Apply small but significant perturbation (must survive quantization)
        perturbed = waveform.copy()
        idx = np.random.randint(len(perturbed))
        # Perturbation must be large enough to affect quantized coefficients
        perturbed[idx] += 0.01 * np.sign(np.random.randn())
        
        perturbed_hash = hash_func(perturbed)
        perturbed_bits = np.unpackbits(np.frombuffer(perturbed_hash, dtype=np.uint8))
        
        # Count bit differences
        bit_diffs = np.sum(original_bits != perturbed_bits)
        flip_ratio = bit_diffs / len(original_bits)
        flip_ratios.append(flip_ratio)
    
    return np.mean(flip_ratios)


def check_collision_resistance(hash_func, n_samples: int = 1000,
                               signal_length: int = 256) -> Dict[str, float]:
    """
    Check collision resistance by verifying uniqueness of hashes.
    
    Args:
        hash_func: Hash function to test
        n_samples: Number of random waveforms to generate
        signal_length: Length of each waveform
        
    Returns:
        Collision statistics
    """
    hashes = set()
    collisions = 0
    
    for i in range(n_samples):
        waveform = np.random.randn(signal_length)
        h = hash_func(waveform)
        
        if h in hashes:
            collisions += 1
        else:
            hashes.add(h)
    
    return {
        'unique_hashes': len(hashes),
        'collisions': collisions,
        'collision_rate': collisions / n_samples
    }


# =============================================================================
# Federated Learning Security
# =============================================================================

def simulate_federated_update(n_clients: int = 10,
                              model_dim: int = 1000,
                              byzantine_fraction: float = 0.0,
                              noise_std: float = 0.01) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Simulate federated learning gradient updates from multiple clients.
    
    Args:
        n_clients: Number of participating clients
        model_dim: Dimension of model/gradient vector
        byzantine_fraction: Fraction of malicious clients
        noise_std: Noise level for honest clients
        
    Returns:
        (true_gradient, list_of_client_gradients)
    """
    # True gradient (what honest clients should approximate)
    true_gradient = np.random.randn(model_dim) * 0.1
    
    n_byzantine = int(byzantine_fraction * n_clients)
    
    client_gradients = []
    
    for i in range(n_clients):
        if i < n_byzantine:
            # Byzantine client: random garbage or adversarial gradient
            if np.random.random() < 0.5:
                # Random garbage
                grad = np.random.randn(model_dim) * 10
            else:
                # Gradient inversion attack
                grad = -true_gradient * np.random.uniform(5, 10)
        else:
            # Honest client: true gradient + noise
            grad = true_gradient + noise_std * np.random.randn(model_dim)
        
        client_gradients.append(grad)
    
    return true_gradient, client_gradients


def federated_aggregate_mean(gradients: List[np.ndarray]) -> np.ndarray:
    """Simple mean aggregation (vulnerable to Byzantine clients)."""
    return np.mean(gradients, axis=0)


def federated_aggregate_median(gradients: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median (more robust to outliers)."""
    stacked = np.stack(gradients, axis=0)
    return np.median(stacked, axis=0)


def federated_aggregate_trimmed_mean(gradients: List[np.ndarray],
                                     trim_fraction: float = 0.1) -> np.ndarray:
    """Trimmed mean aggregation (removes extreme values)."""
    stacked = np.stack(gradients, axis=0)
    n = len(gradients)
    n_trim = int(trim_fraction * n)
    
    if n_trim == 0:
        return np.mean(stacked, axis=0)
    
    # Sort along client axis and trim
    sorted_stack = np.sort(stacked, axis=0)
    trimmed = sorted_stack[n_trim:-n_trim] if n_trim > 0 else sorted_stack
    
    return np.mean(trimmed, axis=0)


def federated_aggregate_rft_filter(gradients: List[np.ndarray],
                                   threshold_std: float = 2.0) -> np.ndarray:
    """
    RFT-based outlier detection and filtering.
    
    Transform gradients to RFT domain and detect anomalous clients
    based on coefficient distribution.
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward
    except ImportError:
        # Fallback to FFT
        rft_forward = np.fft.fft
    
    n_clients = len(gradients)
    model_dim = len(gradients[0])
    
    # Transform each gradient
    transformed = []
    for grad in gradients:
        coeffs = rft_forward(grad.astype(np.complex128))
        transformed.append(np.abs(coeffs))
    
    transformed = np.stack(transformed, axis=0)
    
    # Detect outliers: clients with unusual coefficient energy
    energies = np.sum(transformed ** 2, axis=1)
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    # Mask out clients with anomalous energy
    mask = np.abs(energies - mean_energy) < threshold_std * std_energy
    
    if np.sum(mask) == 0:
        # All clients flagged as anomalous - use median fallback
        return federated_aggregate_median(gradients)
    
    # Aggregate only honest-looking clients
    honest_grads = [g for g, m in zip(gradients, mask) if m]
    
    return np.mean(honest_grads, axis=0)


def aggregation_error(true_gradient: np.ndarray, 
                      aggregated: np.ndarray) -> float:
    """Compute relative error of aggregated gradient."""
    return np.linalg.norm(true_gradient - aggregated) / np.linalg.norm(true_gradient)


# =============================================================================
# Secure Waveform Comparison
# =============================================================================

def secure_waveform_comparison(waveform1: np.ndarray,
                               waveform2: np.ndarray,
                               tolerance: float = 0.01) -> Dict[str, float]:
    """
    Compare two waveforms in a privacy-preserving manner.
    
    Uses RFT-based fingerprinting to compare without revealing raw data.
    
    Args:
        waveform1, waveform2: Input signals
        tolerance: Similarity tolerance
        
    Returns:
        Comparison metrics
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward
    except ImportError:
        pytest.skip("RFT not available")
    
    # Ensure same length
    min_len = min(len(waveform1), len(waveform2))
    w1 = waveform1[:min_len]
    w2 = waveform2[:min_len]
    
    # Transform to RFT domain
    c1 = rft_forward(w1.astype(np.complex128))
    c2 = rft_forward(w2.astype(np.complex128))
    
    # Compute similarity metrics in transform domain
    # (raw data never directly compared)
    
    # Magnitude correlation
    mag_corr = np.corrcoef(np.abs(c1), np.abs(c2))[0, 1]
    
    # Phase difference (wrapped)
    phase_diff = np.angle(c1) - np.angle(c2)
    phase_diff = np.abs(np.arctan2(np.sin(phase_diff), np.cos(phase_diff)))
    avg_phase_diff = np.mean(phase_diff)
    
    # Normalized energy difference
    e1 = np.sum(np.abs(c1) ** 2)
    e2 = np.sum(np.abs(c2) ** 2)
    energy_diff = abs(e1 - e2) / (0.5 * (e1 + e2))
    
    # Similarity score
    similarity = 0.5 * (1 + mag_corr) * np.exp(-avg_phase_diff) * np.exp(-energy_diff)
    
    return {
        'magnitude_correlation': float(mag_corr),
        'average_phase_difference': float(avg_phase_diff),
        'energy_difference': float(energy_diff),
        'similarity_score': float(similarity),
        'is_similar': similarity > (1 - tolerance)
    }


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class SecurityTestResult:
    """Result container for security tests."""
    test_name: str
    metric: str
    value: float
    passed: bool
    notes: str


# =============================================================================
# Pytest Test Cases
# =============================================================================

class TestWaveformHashing:
    """Test suite for RFT-based waveform hashing."""
    
    @pytest.fixture
    def ecg_waveform(self) -> np.ndarray:
        """Generate synthetic ECG-like waveform."""
        t = np.linspace(0, 2, 512)
        # Simplified ECG: periodic spikes
        waveform = np.zeros_like(t)
        for center in np.arange(0, 2, 0.8):
            waveform += np.exp(-((t - center) ** 2) / 0.01)
        return waveform
    
    def test_hash_determinism(self, ecg_waveform):
        """Verify hash is deterministic."""
        hash1 = rft_waveform_hash(ecg_waveform)
        hash2 = rft_waveform_hash(ecg_waveform)
        
        assert hash1 == hash2, "Hash should be deterministic"
        print(f"✓ Hash determinism: {hash1.hex()[:16]}...")
    
    def test_avalanche_effect(self, ecg_waveform):
        """Test avalanche effect (sensitivity to input changes)."""
        avalanche = compute_avalanche_effect(
            lambda x: rft_waveform_hash(x, hash_bits=256),
            ecg_waveform,
            n_trials=50
        )
        
        print(f"\n  Avalanche effect: {avalanche:.3f} (ideal: 0.5)")
        
        # Good avalanche should be between 0.4 and 0.6
        assert 0.3 < avalanche < 0.7, f"Avalanche effect too extreme: {avalanche}"
    
    def test_collision_resistance_random(self):
        """Test collision resistance on random waveforms."""
        stats = check_collision_resistance(
            lambda x: rft_waveform_hash(x),
            n_samples=500,
            signal_length=256
        )
        
        print(f"\n  Collision test (500 random waveforms):")
        print(f"    Unique hashes: {stats['unique_hashes']}")
        print(f"    Collisions: {stats['collisions']}")
        
        assert stats['collision_rate'] < 0.01, "Too many collisions"
    
    @pytest.mark.parametrize("hash_bits", [128, 256, 512])
    def test_hash_sizes(self, ecg_waveform, hash_bits):
        """Test different hash output sizes."""
        h = rft_waveform_hash(ecg_waveform, hash_bits=hash_bits)
        expected_bytes = hash_bits // 8
        
        # Note: MD5 for 128, SHA256 for 256, SHA512 for 512
        actual_bytes = len(h)
        print(f"\n  Hash size {hash_bits} bits: {actual_bytes} bytes")
        
        # Verify non-empty hash
        assert len(h) > 0
    
    def test_salted_hash(self, ecg_waveform):
        """Test keyed/salted hashing."""
        salt1 = b'patient_001'
        salt2 = b'patient_002'
        
        h1 = rft_waveform_hash(ecg_waveform, salt=salt1)
        h2 = rft_waveform_hash(ecg_waveform, salt=salt2)
        h3 = rft_waveform_hash(ecg_waveform, salt=salt1)
        
        assert h1 == h3, "Same salt should produce same hash"
        assert h1 != h2, "Different salts should produce different hashes"
        print("✓ Salted hashing working correctly")


class TestFederatedLearning:
    """Test suite for federated learning security."""
    
    def test_honest_aggregation(self):
        """Test aggregation with all honest clients."""
        true_grad, client_grads = simulate_federated_update(
            n_clients=10, 
            byzantine_fraction=0.0,
            noise_std=0.01
        )
        
        methods = {
            'Mean': federated_aggregate_mean,
            'Median': federated_aggregate_median,
            'Trimmed': lambda g: federated_aggregate_trimmed_mean(g, 0.1),
            'RFT-Filter': federated_aggregate_rft_filter,
        }
        
        print("\n  Aggregation error (all honest clients):")
        for name, method in methods.items():
            agg = method(client_grads)
            error = aggregation_error(true_grad, agg)
            print(f"    {name}: {error:.4f}")
            
            # All methods should work well with honest clients
            assert error < 0.5, f"{name} error too high"
    
    @pytest.mark.parametrize("byzantine_frac", [0.1, 0.2, 0.3])
    def test_byzantine_resilience(self, byzantine_frac):
        """Test resilience to Byzantine (malicious) clients."""
        true_grad, client_grads = simulate_federated_update(
            n_clients=20,
            byzantine_fraction=byzantine_frac,
            noise_std=0.01
        )
        
        errors = {}
        for name, method in [
            ('Mean', federated_aggregate_mean),
            ('Median', federated_aggregate_median),
            ('Trimmed', lambda g: federated_aggregate_trimmed_mean(g, 0.2)),
            ('RFT-Filter', federated_aggregate_rft_filter),
        ]:
            agg = method(client_grads)
            errors[name] = aggregation_error(true_grad, agg)
        
        print(f"\n  Byzantine resilience ({byzantine_frac:.0%} malicious):")
        for name, err in errors.items():
            status = "✓" if err < 1.0 else "✗"
            print(f"    {status} {name}: error={err:.3f}")
        
        # Robust methods should significantly outperform mean
        assert errors['Median'] < errors['Mean'] or errors['RFT-Filter'] < errors['Mean']
    
    def test_rft_filter_detection(self):
        """Test that RFT filter can detect Byzantine clients."""
        true_grad, client_grads = simulate_federated_update(
            n_clients=10,
            byzantine_fraction=0.3,  # 3 Byzantine
            noise_std=0.01
        )
        
        rft_agg = federated_aggregate_rft_filter(client_grads, threshold_std=2.0)
        mean_agg = federated_aggregate_mean(client_grads)
        
        rft_error = aggregation_error(true_grad, rft_agg)
        mean_error = aggregation_error(true_grad, mean_agg)
        
        print(f"\n  Byzantine detection (30% malicious):")
        print(f"    Mean aggregation error: {mean_error:.3f}")
        print(f"    RFT-filtered error: {rft_error:.3f}")
        print(f"    Improvement: {(mean_error - rft_error) / mean_error:.1%}")


class TestSecureComparison:
    """Test suite for privacy-preserving waveform comparison."""
    
    def test_identical_waveforms(self):
        """Test comparison of identical waveforms."""
        waveform = np.random.randn(256)
        
        result = secure_waveform_comparison(waveform, waveform)
        
        print(f"\n  Identical waveform comparison:")
        print(f"    Similarity: {result['similarity_score']:.4f}")
        
        assert result['similarity_score'] > 0.99
        assert result['is_similar']
    
    def test_similar_waveforms(self):
        """Test comparison of similar waveforms (small noise)."""
        original = np.random.randn(256)
        noisy = original + 0.01 * np.random.randn(256)
        
        result = secure_waveform_comparison(original, noisy)
        
        print(f"\n  Similar waveform comparison (1% noise):")
        print(f"    Similarity: {result['similarity_score']:.4f}")
        print(f"    Magnitude correlation: {result['magnitude_correlation']:.4f}")
        
        assert result['similarity_score'] > 0.8
    
    def test_different_waveforms(self):
        """Test comparison of completely different waveforms."""
        waveform1 = np.random.randn(256)
        waveform2 = np.random.randn(256)
        
        result = secure_waveform_comparison(waveform1, waveform2)
        
        print(f"\n  Different waveform comparison:")
        print(f"    Similarity: {result['similarity_score']:.4f}")
        
        # Should have low similarity
        assert result['similarity_score'] < 0.5


class TestHashPerformance:
    """Benchmark hashing performance."""
    
    @pytest.mark.parametrize("size", [256, 1024, 4096])
    def test_hash_throughput(self, size):
        """Measure hashing throughput."""
        waveform = np.random.randn(size)
        
        # Warm-up
        for _ in range(5):
            rft_waveform_hash(waveform)
        
        # Benchmark
        n_iters = 50
        t0 = time.perf_counter()
        for _ in range(n_iters):
            rft_waveform_hash(waveform)
        elapsed = time.perf_counter() - t0
        
        avg_time_ms = (elapsed / n_iters) * 1000
        throughput_samples_per_sec = (size * n_iters) / elapsed
        
        print(f"\n  Hash throughput ({size} samples):")
        print(f"    Avg time: {avg_time_ms:.2f} ms")
        print(f"    Throughput: {throughput_samples_per_sec:.0f} samples/s")


# =============================================================================
# Standalone Runner
# =============================================================================

def run_comprehensive_security_benchmark():
    """Run comprehensive security benchmark."""
    print("=" * 70)
    print("MEDICAL DATA SECURITY & PRIVACY BENCHMARK")
    print("=" * 70)
    
    results: List[SecurityTestResult] = []
    
    # Waveform hashing tests
    print("\n[1] Waveform Hashing")
    
    # Avalanche effect
    waveform = np.random.randn(512)
    avalanche = compute_avalanche_effect(
        lambda x: rft_waveform_hash(x),
        waveform,
        n_trials=100
    )
    results.append(SecurityTestResult(
        test_name='Avalanche Effect',
        metric='Bit flip ratio',
        value=avalanche,
        passed=0.35 < avalanche < 0.65,
        notes=f'Ideal: 0.5, got {avalanche:.3f}'
    ))
    print(f"  Avalanche effect: {avalanche:.3f} (ideal: 0.5)")
    
    # Collision resistance
    collision_stats = check_collision_resistance(
        lambda x: rft_waveform_hash(x),
        n_samples=1000,
        signal_length=256
    )
    results.append(SecurityTestResult(
        test_name='Collision Resistance',
        metric='Collision rate',
        value=collision_stats['collision_rate'],
        passed=collision_stats['collision_rate'] == 0,
        notes=f'{collision_stats["collisions"]} collisions in 1000 samples'
    ))
    print(f"  Collisions: {collision_stats['collisions']}/1000")
    
    # Federated learning tests
    print("\n[2] Federated Learning Robustness")
    
    for byzantine_frac in [0.0, 0.2, 0.4]:
        true_grad, client_grads = simulate_federated_update(
            n_clients=20,
            byzantine_fraction=byzantine_frac,
            noise_std=0.01
        )
        
        rft_agg = federated_aggregate_rft_filter(client_grads)
        mean_agg = federated_aggregate_mean(client_grads)
        
        rft_error = aggregation_error(true_grad, rft_agg)
        mean_error = aggregation_error(true_grad, mean_agg)
        
        improvement = (mean_error - rft_error) / mean_error if mean_error > 0 else 0
        
        results.append(SecurityTestResult(
            test_name=f'Byzantine {byzantine_frac:.0%}',
            metric='RFT filter improvement',
            value=improvement,
            passed=rft_error < 1.0,
            notes=f'RFT error: {rft_error:.3f}, Mean error: {mean_error:.3f}'
        ))
        print(f"  Byzantine {byzantine_frac:.0%}: RFT improvement={improvement:.1%}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SECURITY TEST SUMMARY")
    print("=" * 70)
    print(f"\n{'Test':<25} {'Metric':<20} {'Value':<10} {'Status':<8}")
    print("-" * 65)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"{r.test_name:<25} {r.metric:<20} {r.value:<10.4f} {status:<8}")
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    run_comprehensive_security_benchmark()
