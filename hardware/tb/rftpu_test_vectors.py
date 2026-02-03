#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
# (research/education only). Commercial rights require a separate license.
# Patent Application: USPTO #19/169,399
"""
RFTPU Hardware Test Vector Generator
=====================================
Generates known-good test vectors from Python reference implementation
for validating the SystemVerilog RFTPU accelerator.

Output formats:
- SystemVerilog include files with test vectors
- Hex memory initialization files
- JSON test metadata
"""

import sys
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Use CANONICAL Gram-normalized RFT (January 2026)
# Per README.md: Canonical RFT = Φ̃ = Φ (ΦᴴΦ)^{-1/2} where Φ[k,n] = exp(j 2π frac((k+1)φ) n) / √N
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix, PHI

# Legacy operator-based variant kept for comparison only
from algorithms.rft.variants.operator_variants import generate_rft_golden as generate_rft_golden_legacy

# =============================================================================
# HARDWARE PARAMETERS (must match rftpu_pkg)
# =============================================================================
SAMPLE_WIDTH = 16      # Q1.15 fixed-point
BLOCK_SAMPLES = 8      # Samples per block
DIGEST_WIDTH = 256     # SIS digest bits
TILE_DIM = 8           # 8x8 grid
TILE_COUNT = 64

# Fixed-point configuration
FRAC_BITS = 15         # Q1.15 format for samples
INT_BITS = 1
TOTAL_BITS = 16

# Reference RFT parameters (matching hardware)
RFT_BETA = 1.0
RFT_SIGMA = 1.0


@dataclass
class RFTTestVector:
    """Single RFT transform test vector"""
    test_id: int
    name: str
    input_real: List[int]      # Fixed-point Q1.15
    input_imag: List[int]
    expected_real: List[int]   # Fixed-point Q1.15
    expected_imag: List[int]
    energy: float              # For verification
    round_trip_error: float


@dataclass
class SISTestVector:
    """Single SIS lattice test vector"""
    test_id: int
    name: str
    input_samples: List[int]   # 8 x 16-bit samples
    expected_digest: str       # 256-bit hex string
    expected_lat: List[int]    # 8 x 16-bit lattice values


@dataclass 
class TileTestVector:
    """Full tile operation test"""
    test_id: int
    name: str
    tile_x: int
    tile_y: int
    mode: int
    blocks: List[List[int]]    # List of sample blocks
    expected_digests: List[str]
    cascade_enable: bool
    h3_enable: bool


def float_to_q15(value: float) -> int:
    """Convert float to Q1.15 fixed-point (16-bit signed)"""
    # Clamp to [-1, 1)
    value = max(-1.0, min(0.999969482421875, value))
    fixed = int(value * (1 << FRAC_BITS))
    
    # Ensure 16-bit range
    if fixed < -32768:
        fixed = -32768
    elif fixed > 32767:
        fixed = 32767
        
    return fixed & 0xFFFF


def q15_to_float(fixed: int) -> float:
    """Convert Q1.15 fixed-point to float"""
    if fixed >= 0x8000:  # Negative
        fixed = fixed - 0x10000
    return fixed / (1 << FRAC_BITS)


def complex_to_q15_pair(c: complex) -> Tuple[int, int]:
    """Convert complex to (real_q15, imag_q15)"""
    return float_to_q15(c.real), float_to_q15(c.imag)


# =============================================================================
# HARDWARE RFT KERNEL (matching phi_rft_core)
# =============================================================================

# Cache the kernel matrix
_KERNEL_CACHE = {}

def get_rft_kernel(n: int = BLOCK_SAMPLES) -> np.ndarray:
    """
    Get CANONICAL RFT kernel matrix.
    
    CANONICAL DEFINITION (January 2026):
        Φ̃ = Φ (ΦᴴΦ)^{-1/2}
    where:
        Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
    
    This is the Gram-normalized φ-grid exponential basis that provides:
    - Exact unitarity: Φ̃ᴴ Φ̃ = I
    - +15-20 dB PSNR on golden quasi-periodic signals
    
    See: THEOREMS_RFT_IRONCLAD.md, algorithms/rft/README_RFT.md
    """
    if n not in _KERNEL_CACHE:
        # Use canonical Gram-normalized basis
        _KERNEL_CACHE[n] = rft_basis_matrix(n, n, use_gram_normalization=True)
    return _KERNEL_CACHE[n]


def get_rft_kernel_legacy(n: int = BLOCK_SAMPLES) -> np.ndarray:
    """
    Get LEGACY operator-based RFT kernel (for comparison only).
    
    DEPRECATED: This uses the eigenbasis of resonance operator K = T(R_φ * d).
    The canonical definition is now the Gram-normalized φ-grid basis.
    
    Kept for backwards compatibility and comparison studies.
    """
    return generate_rft_golden_legacy(n)


def generate_rft_kernel_coefficients(n: int = BLOCK_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """Generate kernel coefficients matching hardware implementation"""
    # Use CANONICAL Gram-normalized RFT
    kernel = get_rft_kernel(n)
    return kernel.real, kernel.imag


def python_rft_forward(samples: np.ndarray) -> np.ndarray:
    """Apply forward canonical RFT: y = Ψ @ x"""
    kernel = get_rft_kernel(len(samples))
    return kernel @ samples


def python_rft_inverse(coeffs: np.ndarray) -> np.ndarray:
    """Apply inverse canonical RFT: x = Ψ^H @ y (unitary inverse)"""
    kernel = get_rft_kernel(len(coeffs))
    return kernel.conj().T @ coeffs


def python_rft_block(samples: np.ndarray, mode: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference RFT block computation matching hardware.
    
    Args:
        samples: 8 complex samples (Q1.15 will be converted)
        mode: Kernel variant (0 = standard/golden)
        
    Returns:
        (rft_real, rft_imag): 8 RFT coefficients as separate real/imag
    """
    x = np.asarray(samples, dtype=np.complex128)
    if len(x) != BLOCK_SAMPLES:
        raise ValueError(f"Expected {BLOCK_SAMPLES} samples, got {len(x)}")
    
    # Apply canonical RFT: y = Ψ @ x (matrix-vector multiply)
    y = python_rft_forward(x)
    
    return y.real, y.imag


def python_sis_digest(rft_real: np.ndarray, rft_imag: np.ndarray) -> Tuple[bytes, List[int]]:
    """
    Reference SIS digest computation matching hardware.
    
    Args:
        rft_real: 8 real coefficients
        rft_imag: 8 imaginary coefficients
        
    Returns:
        (digest, lat_values): 256-bit digest and intermediate lattice values
    """
    # Quantize to 16-bit
    q_real = np.clip((rft_real * 32767).astype(int), -32768, 32767) & 0xFFFF
    q_imag = np.clip((rft_imag * 32767).astype(int), -32768, 32767) & 0xFFFF
    
    # S-values: 8-bit from sum
    s_vals = [(int(q_real[i]) + int(q_imag[i])) & 0xFF for i in range(8)]
    
    # Lattice values: 16-bit sliding window sums
    lat = [
        (s_vals[0] + s_vals[1] + s_vals[2] + s_vals[3]) & 0xFFFF,
        (s_vals[1] + s_vals[2] + s_vals[3] + s_vals[4]) & 0xFFFF,
        (s_vals[2] + s_vals[3] + s_vals[4] + s_vals[5]) & 0xFFFF,
        (s_vals[3] + s_vals[4] + s_vals[5] + s_vals[6]) & 0xFFFF,
        (s_vals[4] + s_vals[5] + s_vals[6] + s_vals[7]) & 0xFFFF,
        (s_vals[5] + s_vals[6] + s_vals[7] + s_vals[0]) & 0xFFFF,
        (s_vals[6] + s_vals[7] + s_vals[0] + s_vals[1]) & 0xFFFF,
        (s_vals[7] + s_vals[0] + s_vals[1] + s_vals[2]) & 0xFFFF,
    ]
    
    # Build 256-bit digest
    # sis_lo = {lat[3], lat[2], lat[1], lat[0], lat[7], lat[6], lat[5], lat[4]}
    # sis_hi = same pattern
    sis_lo = struct.pack('>8H', lat[3], lat[2], lat[1], lat[0], lat[7], lat[6], lat[5], lat[4])
    sis_hi = sis_lo  # Same pattern repeated
    digest = sis_hi + sis_lo
    
    return digest, lat


def compute_energy(rft_real: np.ndarray, rft_imag: np.ndarray) -> int:
    """Compute total energy (matching hardware)"""
    total = 0
    for i in range(len(rft_real)):
        mag_real = abs(int(rft_real[i] * 32767))
        mag_imag = abs(int(rft_imag[i] * 32767))
        amplitude = mag_real + mag_imag
        amp_scaled = amplitude >> 16
        total += amp_scaled * amp_scaled
    return total


# =============================================================================
# TEST VECTOR GENERATORS
# =============================================================================

def generate_rft_test_vectors() -> List[RFTTestVector]:
    """Generate comprehensive RFT transform test vectors"""
    vectors = []
    
    test_cases = [
        # Basic signals
        ("dc_signal", np.ones(BLOCK_SAMPLES, dtype=complex) / np.sqrt(BLOCK_SAMPLES)),
        ("impulse", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)),
        ("alternating", np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=complex) / np.sqrt(BLOCK_SAMPLES)),
        
        # Pure frequencies
        ("freq_k1", np.exp(2j * np.pi * 1 * np.arange(BLOCK_SAMPLES) / BLOCK_SAMPLES)),
        ("freq_k2", np.exp(2j * np.pi * 2 * np.arange(BLOCK_SAMPLES) / BLOCK_SAMPLES)),
        ("freq_k3", np.exp(2j * np.pi * 3 * np.arange(BLOCK_SAMPLES) / BLOCK_SAMPLES)),
        
        # Mixed signals
        ("mixed_12", 0.7 * np.exp(2j * np.pi * 1 * np.arange(BLOCK_SAMPLES) / BLOCK_SAMPLES) +
                    0.3 * np.exp(2j * np.pi * 2 * np.arange(BLOCK_SAMPLES) / BLOCK_SAMPLES)),
        
        # Ramp and chirp
        ("ramp", (np.arange(BLOCK_SAMPLES) - 3.5) / 4.0 + 0j),
        ("chirp", np.exp(1j * np.pi * np.arange(BLOCK_SAMPLES)**2 / BLOCK_SAMPLES)),
        
        # Edge cases
        ("zeros", np.zeros(BLOCK_SAMPLES, dtype=complex)),
        ("max_positive", np.ones(BLOCK_SAMPLES, dtype=complex) * 0.999),
        ("max_negative", -np.ones(BLOCK_SAMPLES, dtype=complex) * 0.999),
        
        # Random (deterministic seed)
        ("random_1", None),  # Will be filled
        ("random_2", None),
        ("random_3", None),
    ]
    
    # Generate random vectors with fixed seed for reproducibility
    rng = np.random.default_rng(0xDEADBEEF)
    for i, (name, _) in enumerate(test_cases):
        if name.startswith("random"):
            real = rng.uniform(-0.9, 0.9, BLOCK_SAMPLES)
            imag = rng.uniform(-0.9, 0.9, BLOCK_SAMPLES)
            test_cases[i] = (name, real + 1j * imag)
    
    for test_id, (name, samples) in enumerate(test_cases):
        # Normalize
        norm = np.linalg.norm(samples)
        if norm > 1e-10:
            samples = samples / norm * 0.9  # Keep within fixed-point range
        
        # Compute reference RFT
        rft_real, rft_imag = python_rft_block(samples)
        
        # Round-trip check
        y = rft_real + 1j * rft_imag
        x_rec = python_rft_inverse(y)
        rt_error = np.linalg.norm(samples - x_rec) / max(1e-16, norm)
        
        # Energy
        energy = compute_energy(rft_real, rft_imag)
        
        # Convert to fixed-point
        input_real = [float_to_q15(s.real) for s in samples]
        input_imag = [float_to_q15(s.imag) for s in samples]
        expected_real = [float_to_q15(r) for r in rft_real]
        expected_imag = [float_to_q15(r) for r in rft_imag]
        
        vectors.append(RFTTestVector(
            test_id=test_id,
            name=name,
            input_real=input_real,
            input_imag=input_imag,
            expected_real=expected_real,
            expected_imag=expected_imag,
            energy=float(np.sum(np.abs(rft_real + 1j * rft_imag)**2)),
            round_trip_error=float(rt_error)
        ))
    
    return vectors


def generate_sis_test_vectors() -> List[SISTestVector]:
    """Generate SIS lattice digest test vectors"""
    vectors = []
    
    # Use RFT vectors as input to SIS
    rft_vectors = generate_rft_test_vectors()
    
    for i, rft_vec in enumerate(rft_vectors[:8]):  # First 8 RFT tests
        # Get RFT output as float
        rft_real = np.array([q15_to_float(x) for x in rft_vec.expected_real])
        rft_imag = np.array([q15_to_float(x) for x in rft_vec.expected_imag])
        
        # Compute SIS digest
        digest, lat = python_sis_digest(rft_real, rft_imag)
        
        vectors.append(SISTestVector(
            test_id=i,
            name=f"sis_{rft_vec.name}",
            input_samples=rft_vec.input_real,  # Use real part of input
            expected_digest=digest.hex().upper(),
            expected_lat=lat
        ))
    
    return vectors


def generate_tile_test_vectors() -> List[TileTestVector]:
    """Generate full tile operation test vectors"""
    vectors = []
    
    rng = np.random.default_rng(0xCAFEBABE)
    
    # Test 1: Single block processing
    samples = rng.uniform(-0.5, 0.5, BLOCK_SAMPLES)
    blocks = [[float_to_q15(s) for s in samples]]
    
    rft_real, rft_imag = python_rft_block(samples + 0j)
    digest, _ = python_sis_digest(rft_real, rft_imag)
    
    vectors.append(TileTestVector(
        test_id=0,
        name="single_block",
        tile_x=0,
        tile_y=0,
        mode=1,
        blocks=blocks,
        expected_digests=[digest.hex().upper()],
        cascade_enable=False,
        h3_enable=False
    ))
    
    # Test 2: Multi-block processing
    multi_blocks = []
    multi_digests = []
    for b in range(4):
        samples = rng.uniform(-0.5, 0.5, BLOCK_SAMPLES)
        multi_blocks.append([float_to_q15(s) for s in samples])
        rft_real, rft_imag = python_rft_block(samples + 0j)
        digest, _ = python_sis_digest(rft_real, rft_imag)
        multi_digests.append(digest.hex().upper())
    
    vectors.append(TileTestVector(
        test_id=1,
        name="multi_block",
        tile_x=3,
        tile_y=5,
        mode=1,
        blocks=multi_blocks,
        expected_digests=multi_digests,
        cascade_enable=False,
        h3_enable=False
    ))
    
    return vectors


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def write_sv_include(vectors: List[RFTTestVector], output_path: Path):
    """Write SystemVerilog include file with test vectors"""
    with open(output_path, 'w') as f:
        f.write("// Auto-generated RFT test vectors\n")
        f.write("// DO NOT EDIT - Generated by rpu_test_vectors.py\n\n")
        
        f.write(f"localparam int NUM_RFT_TESTS = {len(vectors)};\n")
        f.write(f"localparam int BLOCK_SIZE = {BLOCK_SAMPLES};\n\n")
        
        # Input arrays
        f.write("// Test inputs (Q1.15 fixed-point)\n")
        f.write("logic signed [15:0] rft_test_input_real [NUM_RFT_TESTS][BLOCK_SIZE] = '{\n")
        for i, v in enumerate(vectors):
            f.write(f"  // Test {i}: {v.name}\n")
            vals = ", ".join(f"16'sh{x:04X}" for x in v.input_real)
            f.write(f"  '{{{vals}}}")
            f.write(",\n" if i < len(vectors) - 1 else "\n")
        f.write("};\n\n")
        
        f.write("logic signed [15:0] rft_test_input_imag [NUM_RFT_TESTS][BLOCK_SIZE] = '{\n")
        for i, v in enumerate(vectors):
            vals = ", ".join(f"16'sh{x:04X}" for x in v.input_imag)
            f.write(f"  '{{{vals}}}")
            f.write(",\n" if i < len(vectors) - 1 else "\n")
        f.write("};\n\n")
        
        # Expected outputs
        f.write("// Expected outputs (Q1.15 fixed-point)\n")
        f.write("logic signed [15:0] rft_test_expected_real [NUM_RFT_TESTS][BLOCK_SIZE] = '{\n")
        for i, v in enumerate(vectors):
            vals = ", ".join(f"16'sh{x:04X}" for x in v.expected_real)
            f.write(f"  '{{{vals}}}")
            f.write(",\n" if i < len(vectors) - 1 else "\n")
        f.write("};\n\n")
        
        f.write("logic signed [15:0] rft_test_expected_imag [NUM_RFT_TESTS][BLOCK_SIZE] = '{\n")
        for i, v in enumerate(vectors):
            vals = ", ".join(f"16'sh{x:04X}" for x in v.expected_imag)
            f.write(f"  '{{{vals}}}")
            f.write(",\n" if i < len(vectors) - 1 else "\n")
        f.write("};\n\n")
        
        # Test names for reporting
        f.write("string rft_test_names [NUM_RFT_TESTS] = '{\n")
        for i, v in enumerate(vectors):
            f.write(f'  "{v.name}"')
            f.write(",\n" if i < len(vectors) - 1 else "\n")
        f.write("};\n")


def write_json_metadata(rft_vectors: List[RFTTestVector], 
                        sis_vectors: List[SISTestVector],
                        tile_vectors: List[TileTestVector],
                        output_path: Path):
    """Write JSON metadata for test analysis"""
    metadata = {
        "generator": "rpu_test_vectors.py",
        "parameters": {
            "sample_width": SAMPLE_WIDTH,
            "block_samples": BLOCK_SAMPLES,
            "digest_width": DIGEST_WIDTH,
            "rft_beta": RFT_BETA,
            "rft_sigma": RFT_SIGMA,
            "phi": PHI
        },
        "rft_tests": [asdict(v) for v in rft_vectors],
        "sis_tests": [asdict(v) for v in sis_vectors],
        "tile_tests": [asdict(v) for v in tile_vectors],
        "summary": {
            "num_rft_tests": len(rft_vectors),
            "num_sis_tests": len(sis_vectors),
            "num_tile_tests": len(tile_vectors),
            "max_round_trip_error": max(v.round_trip_error for v in rft_vectors)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def write_hex_memory(vectors: List[RFTTestVector], output_path: Path):
    """Write Verilog $readmemh compatible file"""
    with open(output_path, 'w') as f:
        f.write("// RFT Test Vectors - Memory initialization\n")
        f.write("// Format: input_real[0-7], input_imag[0-7], expected_real[0-7], expected_imag[0-7]\n\n")
        
        for v in vectors:
            f.write(f"// Test {v.test_id}: {v.name}\n")
            # All 32 values on separate lines
            for x in v.input_real:
                f.write(f"{x:04X}\n")
            for x in v.input_imag:
                f.write(f"{x:04X}\n")
            for x in v.expected_real:
                f.write(f"{x:04X}\n")
            for x in v.expected_imag:
                f.write(f"{x:04X}\n")
            f.write("\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RPU Hardware Test Vector Generator")
    print("=" * 70)
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Generate vectors
    print("\n[1/3] Generating RFT test vectors...")
    rft_vectors = generate_rft_test_vectors()
    print(f"  Generated {len(rft_vectors)} RFT test vectors")
    
    print("\n[2/3] Generating SIS test vectors...")
    sis_vectors = generate_sis_test_vectors()
    print(f"  Generated {len(sis_vectors)} SIS test vectors")
    
    print("\n[3/3] Generating tile test vectors...")
    tile_vectors = generate_tile_test_vectors()
    print(f"  Generated {len(tile_vectors)} tile test vectors")
    
    # Write outputs
    print("\nWriting output files...")
    
    sv_path = output_dir / "rft_test_vectors.svh"
    write_sv_include(rft_vectors, sv_path)
    print(f"  ✓ {sv_path}")
    
    json_path = output_dir / "test_vectors_metadata.json"
    write_json_metadata(rft_vectors, sis_vectors, tile_vectors, json_path)
    print(f"  ✓ {json_path}")
    
    hex_path = output_dir / "rft_test_vectors.hex"
    write_hex_memory(rft_vectors, hex_path)
    print(f"  ✓ {hex_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Max round-trip error: {max(v.round_trip_error for v in rft_vectors):.2e}")
    print(f"  Total test vectors: {len(rft_vectors) + len(sis_vectors) + len(tile_vectors)}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
