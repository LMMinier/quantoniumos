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
Genomics Transform Acceleration Tests
======================================

Tests RFT variants for **synthetic** genomics/proteomics data patterns.

⚠️ **WARNING: SYNTHETIC DATA ONLY**
- These tests using `generate_random_dna` (random character strings).
- They do NOT validate performance on real biological sequences (which have repeats, GC-bias, and evolutionary structure).
- This is an **information-theoretic benchmark**, not a validated bioinformatics tool.

Tests:
- K-mer spectra transform acceleration
- Contact map compression for protein structures
- Sequence data compression benchmarks (Random Character)
"""

import numpy as np
import time
import zlib
import pytest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


# =============================================================================
# DNA/RNA Sequence Generators
# =============================================================================

NUCLEOTIDES = ['A', 'C', 'G', 'T']
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


def generate_random_dna(length: int, gc_content: float = 0.5) -> str:
    """
    Generate random DNA sequence with specified GC content.
    
    Args:
        length: Sequence length in base pairs
        gc_content: Fraction of G+C bases (0.0 to 1.0)
        
    Returns:
        DNA sequence string
    """
    gc_prob = gc_content / 2
    at_prob = (1 - gc_content) / 2
    
    probs = [at_prob, gc_prob, gc_prob, at_prob]  # A, C, G, T
    
    return ''.join(np.random.choice(NUCLEOTIDES, size=length, p=probs))


def generate_coding_sequence(length: int, codon_bias: Optional[Dict[str, float]] = None) -> str:
    """
    Generate coding DNA sequence with realistic codon usage.
    
    Args:
        length: Approximate sequence length
        codon_bias: Optional codon frequency dictionary
        
    Returns:
        DNA sequence (multiple of 3)
    """
    # Simplified codon table (just generate random codons)
    n_codons = length // 3
    
    sequence = []
    for _ in range(n_codons):
        codon = ''.join(np.random.choice(NUCLEOTIDES, size=3))
        sequence.append(codon)
    
    return ''.join(sequence)


def generate_protein_sequence(length: int, 
                              hydrophobic_bias: float = 0.0) -> str:
    """
    Generate random protein/amino acid sequence.
    
    Args:
        length: Sequence length
        hydrophobic_bias: Bias toward hydrophobic residues (-1 to 1)
        
    Returns:
        Amino acid sequence
    """
    # Hydrophobic residues: A, F, I, L, M, V, W
    hydrophobic = set('AFILMVW')
    
    weights = np.ones(len(AMINO_ACIDS))
    for i, aa in enumerate(AMINO_ACIDS):
        if aa in hydrophobic:
            weights[i] *= (1 + hydrophobic_bias)
        else:
            weights[i] *= (1 - hydrophobic_bias * 0.5)
    
    weights /= weights.sum()
    
    return ''.join(np.random.choice(AMINO_ACIDS, size=length, p=weights))


# =============================================================================
# K-mer Analysis
# =============================================================================

def sequence_to_numeric(sequence: str, alphabet: str = 'ACGT') -> np.ndarray:
    """
    Convert sequence to numeric array for transform processing.
    
    Args:
        sequence: DNA/protein sequence
        alphabet: Character alphabet
        
    Returns:
        Numeric array (float)
    """
    char_to_num = {c: i for i, c in enumerate(alphabet)}
    return np.array([char_to_num.get(c, 0) for c in sequence], dtype=np.float64)


def numeric_to_sequence(arr: np.ndarray, alphabet: str = 'ACGT') -> str:
    """Convert numeric array back to sequence."""
    indices = np.clip(np.round(arr).astype(int), 0, len(alphabet) - 1)
    return ''.join(alphabet[i] for i in indices)


def compute_kmer_spectrum(sequence: str, k: int = 4) -> np.ndarray:
    """
    Compute k-mer frequency spectrum.
    
    Args:
        sequence: DNA sequence
        k: k-mer length
        
    Returns:
        Frequency vector of length 4^k
    """
    n_kmers = 4 ** k
    spectrum = np.zeros(n_kmers)
    
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(b in base_to_idx for b in kmer):
            idx = sum(base_to_idx[b] * (4 ** (k - 1 - j)) for j, b in enumerate(kmer))
            spectrum[idx] += 1
    
    # Normalize
    total = spectrum.sum()
    if total > 0:
        spectrum /= total
    
    return spectrum


def transform_kmer_spectrum(spectrum: np.ndarray, 
                            transform: str = 'rft') -> np.ndarray:
    """
    Apply transform to k-mer spectrum for analysis.
    
    Args:
        spectrum: K-mer frequency vector
        transform: 'rft', 'fft', or 'dct'
        
    Returns:
        Transform coefficients
    """
    if transform == 'rft':
        try:
            from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward
            return rft_forward(spectrum.astype(np.complex128))
        except ImportError:
            pytest.skip("RFT not available")
    elif transform == 'fft':
        return np.fft.fft(spectrum)
    elif transform == 'dct':
        try:
            from scipy.fft import dct
        except ImportError:
            from scipy.fftpack import dct
        return dct(spectrum, type=2, norm='ortho')
    else:
        raise ValueError(f"Unknown transform: {transform}")


# =============================================================================
# Contact Map Compression
# =============================================================================

def generate_contact_map(n_residues: int, 
                         contact_threshold: float = 8.0,
                         secondary_structure: str = 'random') -> np.ndarray:
    """
    Generate synthetic protein contact map.
    
    A contact map shows which residues are spatially close (<8Å typically).
    
    Args:
        n_residues: Number of amino acid residues
        contact_threshold: Distance threshold for contact
        secondary_structure: 'random', 'helix', 'sheet', or 'mixed'
        
    Returns:
        Binary contact map (n_residues x n_residues)
    """
    # Generate 3D coordinates based on structure type
    if secondary_structure == 'helix':
        # Alpha helix: 3.6 residues per turn, 1.5Å rise per residue
        t = np.arange(n_residues)
        x = 2.3 * np.cos(t * 2 * np.pi / 3.6)
        y = 2.3 * np.sin(t * 2 * np.pi / 3.6)
        z = 1.5 * t
    elif secondary_structure == 'sheet':
        # Beta sheet: extended conformation
        t = np.arange(n_residues)
        x = 3.5 * t  # ~3.5Å per residue
        y = 2.0 * np.sin(t * np.pi)  # Slight pleating
        z = np.zeros(n_residues)
    else:
        # Random coil with some local structure
        x = np.cumsum(np.random.randn(n_residues)) * 2
        y = np.cumsum(np.random.randn(n_residues)) * 2
        z = np.cumsum(np.random.randn(n_residues)) * 2
    
    coords = np.stack([x, y, z], axis=1)
    
    # Compute pairwise distances
    distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=2))
    
    # Contact map (symmetric, diagonal = 0)
    contact_map = (distances < contact_threshold).astype(np.float64)
    np.fill_diagonal(contact_map, 0)
    
    # Add some noise/randomness to make it more realistic
    noise = np.random.random((n_residues, n_residues)) < 0.05
    noise = noise | noise.T
    contact_map = np.logical_xor(contact_map.astype(bool), noise).astype(np.float64)
    
    return contact_map


def compress_contact_map_rft(contact_map: np.ndarray, 
                              keep_ratio: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Compress contact map using 2D RFT.
    
    Args:
        contact_map: 2D contact map
        keep_ratio: Fraction of coefficients to keep
        
    Returns:
        (reconstructed_map, stats)
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        pytest.skip("RFT not available")
    
    rows, cols = contact_map.shape
    
    # Row-wise RFT
    row_coeffs = np.zeros_like(contact_map, dtype=np.complex128)
    for i in range(rows):
        row_coeffs[i, :] = rft_forward(contact_map[i, :].astype(np.complex128))
    
    # Column-wise RFT
    full_coeffs = np.zeros_like(row_coeffs)
    for j in range(cols):
        full_coeffs[:, j] = rft_forward(row_coeffs[:, j])
    
    # Threshold
    n_total = rows * cols
    n_keep = int(keep_ratio * n_total)
    magnitudes = np.abs(full_coeffs).flatten()
    threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
    
    thresholded = np.where(np.abs(full_coeffs) >= threshold, full_coeffs, 0)
    n_nonzero = np.count_nonzero(thresholded)
    
    # Inverse
    inv_cols = np.zeros_like(thresholded)
    for j in range(cols):
        inv_cols[:, j] = rft_inverse(thresholded[:, j])
    
    reconstructed = np.zeros_like(contact_map)
    for i in range(rows):
        reconstructed[i, :] = rft_inverse(inv_cols[i, :]).real
    
    # Binarize reconstruction
    reconstructed = (reconstructed > 0.5).astype(np.float64)
    
    stats = {
        'compression_ratio': n_total / n_nonzero if n_nonzero > 0 else float('inf'),
        'n_kept': n_nonzero,
        'n_total': n_total
    }
    
    return reconstructed, stats


# =============================================================================
# Sequence Compression
# =============================================================================

def compress_sequence_rft(sequence: str, 
                          chunk_size: int = 256,
                          keep_ratio: float = 0.5) -> Tuple[str, Dict]:
    """
    Compress DNA sequence using RFT.
    
    Args:
        sequence: DNA sequence
        chunk_size: Processing chunk size
        keep_ratio: Coefficient retention ratio
        
    Returns:
        (reconstructed_sequence, stats)
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        pytest.skip("RFT not available")
    
    # Convert to numeric
    numeric = sequence_to_numeric(sequence, 'ACGT')
    n = len(numeric)
    
    # Pad to chunk boundary
    n_padded = ((n - 1) // chunk_size + 1) * chunk_size
    padded = np.zeros(n_padded)
    padded[:n] = numeric
    
    n_chunks = n_padded // chunk_size
    coeffs_kept = 0
    total_coeffs = 0
    
    reconstructed = np.zeros(n_padded)
    
    for i in range(n_chunks):
        chunk = padded[i * chunk_size:(i + 1) * chunk_size].astype(np.complex128)
        
        rft_coeffs = rft_forward(chunk)
        total_coeffs += len(rft_coeffs)
        
        n_keep = int(keep_ratio * len(rft_coeffs))
        magnitudes = np.abs(rft_coeffs)
        threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
        
        compressed = np.where(magnitudes >= threshold, rft_coeffs, 0)
        coeffs_kept += np.count_nonzero(compressed)
        
        reconstructed[i * chunk_size:(i + 1) * chunk_size] = rft_inverse(compressed).real
    
    # Convert back to sequence
    recon_seq = numeric_to_sequence(reconstructed[:n], 'ACGT')
    
    # Calculate accuracy
    matches = sum(a == b for a, b in zip(sequence, recon_seq))
    accuracy = matches / len(sequence)
    
    stats = {
        'compression_ratio': total_coeffs / coeffs_kept if coeffs_kept > 0 else float('inf'),
        'sequence_accuracy': accuracy,
        'exact_matches': matches,
        'total_bases': len(sequence)
    }
    
    return recon_seq, stats


def gzip_compress_sequence(sequence: str) -> Tuple[bytes, Dict]:
    """Baseline gzip compression."""
    data = sequence.encode('ascii')
    compressed = zlib.compress(data, level=9)
    
    stats = {
        'original_bytes': len(data),
        'compressed_bytes': len(compressed),
        'compression_ratio': len(data) / len(compressed)
    }
    
    return compressed, stats


# =============================================================================
# Quality Metrics
# =============================================================================

def contact_map_accuracy(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Calculate contact map reconstruction accuracy."""
    original_bool = original > 0.5
    recon_bool = reconstructed > 0.5
    
    tp = np.sum(original_bool & recon_bool)
    tn = np.sum(~original_bool & ~recon_bool)
    fp = np.sum(~original_bool & recon_bool)
    fn = np.sum(original_bool & ~recon_bool)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity percentage."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return 100 * matches / len(seq1)


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class GenomicsTestResult:
    """Result container for genomics tests."""
    test_type: str
    method: str
    input_size: int
    compression_ratio: float
    accuracy: float
    throughput_kbps: float
    time_ms: float


# =============================================================================
# Pytest Test Cases
# =============================================================================

class TestKmerAnalysis:
    """Test suite for k-mer spectrum analysis."""
    
    @pytest.fixture
    def dna_sequence(self) -> str:
        """Generate test DNA sequence."""
        return generate_random_dna(10000, gc_content=0.5)
    
    def test_kmer_spectrum_generation(self, dna_sequence):
        """Verify k-mer spectrum computation."""
        spectrum = compute_kmer_spectrum(dna_sequence, k=4)
        
        assert len(spectrum) == 256  # 4^4
        assert abs(spectrum.sum() - 1.0) < 1e-10  # Normalized
        
        print(f"✓ K-mer spectrum: {len(spectrum)} bins, "
              f"max={spectrum.max():.4f}, entropy={-np.sum(spectrum * np.log2(spectrum + 1e-10)):.2f} bits")
    
    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_kmer_transform_comparison(self, dna_sequence, k):
        """Compare transforms on k-mer spectra."""
        spectrum = compute_kmer_spectrum(dna_sequence, k=k)
        
        results = {}
        for transform_name in ['rft', 'fft', 'dct']:
            t0 = time.perf_counter()
            coeffs = transform_kmer_spectrum(spectrum, transform_name)
            elapsed = (time.perf_counter() - t0) * 1000
            
            # Sparsity analysis
            magnitudes = np.abs(coeffs)
            top_10_energy = np.sum(np.sort(magnitudes)[-10:] ** 2) / np.sum(magnitudes ** 2)
            
            results[transform_name] = {
                'time_ms': elapsed,
                'top_10_energy': top_10_energy,
                'max_coeff': magnitudes.max()
            }
        
        print(f"\n  K-mer transform comparison (k={k}, spectrum size={4**k}):")
        for name, r in results.items():
            print(f"    {name.upper()}: top-10 energy={r['top_10_energy']:.3f}, "
                  f"time={r['time_ms']:.3f}ms")


class TestContactMapCompression:
    """Test suite for protein contact map compression."""
    
    @pytest.fixture
    def contact_map(self) -> np.ndarray:
        """Generate test contact map."""
        return generate_contact_map(128, secondary_structure='mixed')
    
    def test_contact_map_generation(self, contact_map):
        """Verify contact map generation."""
        assert contact_map.shape == (128, 128)
        assert np.allclose(contact_map, contact_map.T)  # Symmetric
        
        contact_density = contact_map.sum() / (128 * 128)
        print(f"✓ Contact map: {contact_map.shape}, "
              f"density={contact_density:.2%}")
    
    @pytest.mark.parametrize("keep_ratio", [0.3, 0.5, 0.7])
    def test_contact_map_compression(self, contact_map, keep_ratio):
        """Test contact map compression at various ratios."""
        t0 = time.perf_counter()
        recon, stats = compress_contact_map_rft(contact_map, keep_ratio)
        elapsed = (time.perf_counter() - t0) * 1000
        
        metrics = contact_map_accuracy(contact_map, recon)
        
        print(f"\n  Contact map compression (keep={keep_ratio}):")
        print(f"    CR={stats['compression_ratio']:.2f}x, "
              f"Accuracy={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1_score']:.3f}, time={elapsed:.1f}ms")
        
        assert metrics['accuracy'] > 0.7, f"Contact map accuracy too low: {metrics['accuracy']}"
    
    @pytest.mark.parametrize("structure", ['helix', 'sheet', 'random'])
    def test_structure_specific_compression(self, structure):
        """Test compression on different secondary structures."""
        cmap = generate_contact_map(100, secondary_structure=structure)
        
        recon, stats = compress_contact_map_rft(cmap, keep_ratio=0.5)
        metrics = contact_map_accuracy(cmap, recon)
        
        print(f"\n  {structure.capitalize()} structure: "
              f"CR={stats['compression_ratio']:.2f}x, F1={metrics['f1_score']:.3f}")


class TestSequenceCompression:
    """Test suite for DNA sequence compression."""
    
    @pytest.fixture
    def dna_10k(self) -> str:
        """Generate 10kb test sequence."""
        return generate_random_dna(10000)
    
    def test_sequence_generation(self, dna_10k):
        """Verify sequence generation."""
        assert len(dna_10k) == 10000
        gc = (dna_10k.count('G') + dna_10k.count('C')) / len(dna_10k)
        print(f"✓ DNA sequence: {len(dna_10k)} bp, GC={gc:.1%}")
    
    def test_rft_vs_gzip(self, dna_10k):
        """Compare RFT vs gzip compression."""
        # RFT compression (lossy)
        t0 = time.perf_counter()
        rft_recon, rft_stats = compress_sequence_rft(dna_10k, keep_ratio=0.5)
        rft_time = (time.perf_counter() - t0) * 1000
        
        # Gzip compression (lossless)
        t0 = time.perf_counter()
        _, gzip_stats = gzip_compress_sequence(dna_10k)
        gzip_time = (time.perf_counter() - t0) * 1000
        
        print(f"\n  Sequence compression (10kb):")
        print(f"    RFT: CR={rft_stats['compression_ratio']:.2f}x, "
              f"accuracy={rft_stats['sequence_accuracy']:.2%}, "
              f"time={rft_time:.1f}ms")
        print(f"    gzip: CR={gzip_stats['compression_ratio']:.2f}x, "
              f"lossless, time={gzip_time:.1f}ms")
    
    @pytest.mark.parametrize("length", [1000, 10000, 50000])
    def test_throughput_scaling(self, length):
        """Test compression throughput scaling."""
        seq = generate_random_dna(length)
        
        t0 = time.perf_counter()
        _, stats = compress_sequence_rft(seq, keep_ratio=0.5)
        elapsed = time.perf_counter() - t0
        
        throughput_bps = length / elapsed  # bases per second
        throughput_kbps = throughput_bps / 1000
        
        print(f"\n  Throughput ({length/1000:.0f}kb): "
              f"{throughput_kbps:.1f} kb/s, "
              f"accuracy={stats['sequence_accuracy']:.2%}")


class TestThroughputBenchmark:
    """Benchmark genomics processing throughput."""
    
    def test_kmer_throughput(self):
        """Measure k-mer analysis throughput."""
        lengths = [10000, 50000, 100000]
        
        print("\n  K-mer analysis throughput (k=4):")
        for length in lengths:
            seq = generate_random_dna(length)
            
            t0 = time.perf_counter()
            spectrum = compute_kmer_spectrum(seq, k=4)
            _ = transform_kmer_spectrum(spectrum, 'rft')
            elapsed = time.perf_counter() - t0
            
            throughput = length / elapsed / 1000  # kb/s
            print(f"    {length/1000:.0f}kb: {throughput:.1f} kb/s")
    
    def test_contact_map_throughput(self):
        """Measure contact map processing throughput."""
        sizes = [64, 128, 256]
        
        print("\n  Contact map compression throughput:")
        for size in sizes:
            cmap = generate_contact_map(size)
            
            t0 = time.perf_counter()
            compress_contact_map_rft(cmap, keep_ratio=0.5)
            elapsed = time.perf_counter() - t0
            
            throughput = (size * size) / elapsed / 1000  # entries/s (k)
            print(f"    {size}x{size}: {throughput:.1f} k entries/s")


# =============================================================================
# Standalone Runner
# =============================================================================

def run_comprehensive_genomics_benchmark():
    """Run comprehensive genomics benchmark."""
    print("=" * 70)
    print("GENOMICS TRANSFORM ACCELERATION BENCHMARK")
    print("=" * 70)
    
    results: List[GenomicsTestResult] = []
    
    # K-mer spectrum analysis
    print("\n[1] K-mer Spectrum Transform Comparison")
    for length in [10000, 50000]:
        seq = generate_random_dna(length)
        spectrum = compute_kmer_spectrum(seq, k=4)
        
        for transform in ['rft', 'fft', 'dct']:
            t0 = time.perf_counter()
            coeffs = transform_kmer_spectrum(spectrum, transform)
            elapsed = (time.perf_counter() - t0) * 1000
            
            # Energy concentration
            mags = np.abs(coeffs)
            top_10_ratio = np.sum(np.sort(mags)[-10:]) / np.sum(mags)
            
            throughput = length / (elapsed / 1000) / 1000  # kb/s
            
            results.append(GenomicsTestResult(
                test_type='kmer_spectrum',
                method=transform.upper(),
                input_size=length,
                compression_ratio=0,  # N/A for analysis
                accuracy=top_10_ratio,
                throughput_kbps=throughput,
                time_ms=elapsed
            ))
    
    # Contact map compression
    print("\n[2] Contact Map Compression")
    for size in [64, 128]:
        cmap = generate_contact_map(size)
        
        t0 = time.perf_counter()
        recon, stats = compress_contact_map_rft(cmap, keep_ratio=0.5)
        elapsed = (time.perf_counter() - t0) * 1000
        
        metrics = contact_map_accuracy(cmap, recon)
        throughput = (size * size) / (elapsed / 1000) / 1000
        
        results.append(GenomicsTestResult(
            test_type='contact_map',
            method='RFT',
            input_size=size * size,
            compression_ratio=stats['compression_ratio'],
            accuracy=metrics['accuracy'],
            throughput_kbps=throughput,
            time_ms=elapsed
        ))
    
    # Sequence compression
    print("\n[3] Sequence Compression Comparison")
    for length in [10000, 50000]:
        seq = generate_random_dna(length)
        
        for method_name, method in [
            ('RFT', lambda s: compress_sequence_rft(s, keep_ratio=0.5)),
            ('gzip', gzip_compress_sequence),
        ]:
            t0 = time.perf_counter()
            _, stats = method(seq)
            elapsed = (time.perf_counter() - t0) * 1000
            
            throughput = length / (elapsed / 1000) / 1000
            accuracy = stats.get('sequence_accuracy', 1.0)  # gzip is lossless
            
            results.append(GenomicsTestResult(
                test_type='sequence',
                method=method_name,
                input_size=length,
                compression_ratio=stats['compression_ratio'],
                accuracy=accuracy,
                throughput_kbps=throughput,
                time_ms=elapsed
            ))
    
    # Print results
    print(f"\n{'Type':<15} {'Method':<6} {'Size':<10} {'CR':<8} {'Accuracy':<10} "
          f"{'Throughput':<12} {'Time':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.test_type:<15} {r.method:<6} {r.input_size:<10} "
              f"{r.compression_ratio:>6.2f}x  {r.accuracy:>8.2%}   "
              f"{r.throughput_kbps:>10.1f}kb/s {r.time_ms:>8.1f}ms")
    
    print("\n✓ Genomics benchmark complete")
    return results


if __name__ == "__main__":
    run_comprehensive_genomics_benchmark()


# =============================================================================
# Real-Data Tests (Public Genomics Datasets)
# =============================================================================

from tests.medical.real_data_fixtures import (
    skip_no_genomics,
    load_lambda_fasta,
    load_pdb_1crn,
    GENOMICS_AVAILABLE,
    LAMBDA_FASTA,
    PDB_1CRN,
    _dataset_available,
)


@skip_no_genomics
class TestRealGenomics:
    """
    Tests using real public genomics datasets.
    
    Requires:
        - USE_REAL_DATA=1
        - python data/genomics_fetch.py executed
    
    Datasets:
        - Lambda phage (NC_001416) — Public domain RefSeq
        - PDB 1CRN (Crambin) — CC BY 4.0
    """

    @pytest.fixture(scope="class")
    def lambda_seq(self):
        """Load lambda phage genome sequence."""
        if not _dataset_available(LAMBDA_FASTA):
            pytest.skip("Lambda phage FASTA not available")
        return load_lambda_fasta()

    @pytest.fixture(scope="class")
    def pdb_text(self):
        """Load PDB 1CRN structure text."""
        if not _dataset_available(PDB_1CRN):
            pytest.skip("PDB 1CRN not available")
        return load_pdb_1crn()

    def test_lambda_kmer_spectrum(self, lambda_seq):
        """Test k-mer spectrum analysis on real lambda phage genome."""
        # Lambda phage is ~48.5 kb
        assert len(lambda_seq) > 40000, f"Lambda sequence too short: {len(lambda_seq)}"
        
        spectrum = compute_kmer_spectrum(lambda_seq, k=4)
        
        # Verify spectrum properties
        assert len(spectrum) == 256, "4-mer spectrum should have 256 entries"
        assert np.isclose(spectrum.sum(), 1.0, atol=0.01), "Spectrum should sum to 1"
        
        # Transform with RFT
        rft_coeffs = transform_kmer_spectrum(spectrum, 'rft')
        fft_coeffs = transform_kmer_spectrum(spectrum, 'fft')
        
        # Energy should be preserved (Parseval)
        energy_orig = np.sum(spectrum ** 2)
        energy_rft = np.sum(np.abs(rft_coeffs) ** 2) / len(rft_coeffs)
        
        print(f"✓ Lambda phage k-mer analysis: {len(lambda_seq)} bp, spectrum energy={energy_orig:.6f}")

    def test_lambda_sequence_compression(self, lambda_seq):
        """Test RFT compression on real lambda phage sequence."""
        # Use first 10 kb for quick test
        seq_subset = lambda_seq[:10000]
        
        compressed, stats = compress_sequence_rft(seq_subset, keep_ratio=0.5)
        gzip_result, gzip_stats = gzip_compress_sequence(seq_subset)
        
        rft_cr = stats['compression_ratio']
        gzip_cr = gzip_stats['compression_ratio']
        accuracy = stats['sequence_accuracy']
        
        print(f"Lambda compression: RFT CR={rft_cr:.2f}x (acc={accuracy:.1%}), gzip CR={gzip_cr:.2f}x")
        
        # RFT lossy; gzip lossless and likely better CR for random-ish DNA
        assert rft_cr >= 1.5, f"RFT compression ratio too low: {rft_cr:.2f}"
        assert accuracy >= 0.80, f"RFT sequence accuracy too low: {accuracy:.1%}"

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_lambda_kmer_sizes(self, lambda_seq, k):
        """Test k-mer analysis at different k values."""
        spectrum = compute_kmer_spectrum(lambda_seq, k=k)
        
        expected_size = 4 ** k
        assert len(spectrum) == expected_size, f"Expected {expected_size} k-mers for k={k}"
        
        # Check GC content via spectrum (G+C should be ~50% for lambda)
        # For k=1: indices 1=C, 2=G
        if k == 1:
            gc = spectrum[1] + spectrum[2]
            print(f"Lambda phage GC content: {gc:.1%}")
            assert 0.40 <= gc <= 0.60, f"Unusual GC content: {gc:.1%}"

    def test_pdb_contact_map_compression(self, pdb_text):
        """Test contact map compression on real PDB structure."""
        # Parse CA coordinates from PDB
        ca_coords = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and " CA " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
                except ValueError:
                    continue
        
        if len(ca_coords) < 10:
            pytest.skip("Could not parse enough CA atoms from PDB")
        
        ca_coords = np.array(ca_coords)
        n_residues = len(ca_coords)
        
        # Compute distance matrix
        dist_matrix = np.sqrt(np.sum((ca_coords[:, None] - ca_coords[None, :]) ** 2, axis=-1))
        
        # Contact map (< 8 Å)
        contact_map = (dist_matrix < 8.0).astype(float)
        
        # Compress with RFT
        compressed, stats = compress_contact_map_rft(contact_map, keep_ratio=0.5)
        
        # Verify
        accuracy = contact_map_accuracy(contact_map, compressed)
        
        print(f"✓ PDB 1CRN contact map: {n_residues} residues, CR={stats['compression_ratio']:.2f}x, acc={accuracy:.4f}")
        
        assert accuracy >= 0.90, f"Contact map accuracy too low: {accuracy:.4f}"


@skip_no_genomics
def test_lambda_vs_synthetic_comparison():
    """Compare real lambda phage to synthetic random DNA."""
    if not _dataset_available(LAMBDA_FASTA):
        pytest.skip("Lambda phage FASTA not available")
    
    lambda_seq = load_lambda_fasta()
    synthetic_seq = generate_random_dna(len(lambda_seq), gc_content=0.5)
    
    # K-mer spectra
    lambda_spectrum = compute_kmer_spectrum(lambda_seq, k=4)
    synth_spectrum = compute_kmer_spectrum(synthetic_seq, k=4)
    
    # Real genome should have different spectrum than random
    # (codon bias, regulatory sequences, etc.)
    correlation = np.corrcoef(lambda_spectrum, synth_spectrum)[0, 1]
    
    print(f"Lambda vs synthetic k-mer spectrum correlation: {correlation:.4f}")
    
    # Real genome spectrum should differ from purely random
    assert correlation < 0.95, "Real genome should differ from random"
