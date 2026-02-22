#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate random bitstreams from EnhancedRFTCryptoV2 for NIST Statistical Test Suite.

NIST STS requires binary files of specific sizes. This script generates:
1. CTR-mode encryption output (pseudorandom stream)
2. Raw cipher output from random plaintexts

Output: data/artifacts/nist_sts/

To run NIST STS after generating:
1. Download: https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software
2. Build: cd sts-2.1.2 && make
3. Run: ./assess <bitstream_length>
"""

import os
import sys
import struct
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atomic_io import atomic_write_json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2


def get_git_commit():
    """Get current git commit SHA."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def generate_ctr_stream(cipher: EnhancedRFTCryptoV2, num_blocks: int) -> bytes:
    """Generate CTR-mode pseudorandom stream."""
    output = bytearray()
    nonce = b'\x00' * 8  # 64-bit nonce
    
    for counter in range(num_blocks):
        # CTR block = nonce || counter
        ctr_block = nonce + struct.pack('>Q', counter)
        # Encrypt counter block
        ct = cipher._feistel_encrypt(ctr_block)
        output.extend(ct)
    
    return bytes(output)


def generate_random_encryption(cipher: EnhancedRFTCryptoV2, num_blocks: int, seed: int) -> bytes:
    """Generate ciphertext from random plaintexts."""
    np.random.seed(seed)
    output = bytearray()
    
    for _ in range(num_blocks):
        pt = bytes(np.random.randint(0, 256, 16, dtype=np.uint8))
        ct = cipher._feistel_encrypt(pt)
        output.extend(ct)
    
    return bytes(output)


def write_binary_file(data: bytes, filepath: Path):
    """Write raw binary file for NIST STS."""
    with open(filepath, 'wb') as f:
        f.write(data)


def analyze_basic_stats(data: bytes) -> dict:
    """Basic statistical analysis before NIST STS."""
    bits = ''.join(format(byte, '08b') for byte in data)
    total_bits = len(bits)
    ones = bits.count('1')
    zeros = bits.count('0')
    
    # Frequency (monobit) test
    proportion = ones / total_bits
    expected = 0.5
    deviation = abs(proportion - expected)
    
    # Runs test (count runs of consecutive bits)
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs += 1
    
    expected_runs = (2 * ones * zeros) / total_bits + 1
    
    return {
        "total_bits": total_bits,
        "ones": ones,
        "zeros": zeros,
        "proportion_ones": proportion,
        "deviation_from_half": deviation,
        "frequency_assessment": "LIKELY PASS" if deviation < 0.01 else "LIKELY FAIL",
        "runs": runs,
        "expected_runs": expected_runs,
        "runs_ratio": runs / expected_runs if expected_runs > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate NIST STS bitstreams")
    parser.add_argument("--output", default="data/artifacts/nist_sts",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--megabits", type=int, default=10, 
                        help="Size in megabits (NIST recommends 1-100)")
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate number of blocks needed
    bits_needed = args.megabits * 1_000_000
    bytes_needed = bits_needed // 8
    blocks_needed = bytes_needed // 16  # 16 bytes per block
    
    print(f"Generating NIST STS bitstreams to {args.output}/")
    print(f"Target size: {args.megabits} Mbit ({bytes_needed:,} bytes, {blocks_needed:,} blocks)")
    
    # Generate key
    key = bytes(range(32))  # Deterministic key for reproducibility
    cipher = EnhancedRFTCryptoV2(key)
    
    # 1. CTR mode stream
    print("\n[1/2] Generating CTR-mode stream...")
    ctr_data = generate_ctr_stream(cipher, blocks_needed)
    ctr_path = out_dir / "ctr_stream.bin"
    write_binary_file(ctr_data, ctr_path)
    ctr_stats = analyze_basic_stats(ctr_data)
    print(f"  Written: {ctr_path} ({len(ctr_data):,} bytes)")
    print(f"  Proportion of 1s: {ctr_stats['proportion_ones']:.6f} (ideal: 0.5)")
    print(f"  Frequency assessment: {ctr_stats['frequency_assessment']}")
    
    # 2. Random encryption stream
    print("\n[2/2] Generating random-plaintext stream...")
    rnd_data = generate_random_encryption(cipher, blocks_needed, args.seed)
    rnd_path = out_dir / "random_encryption.bin"
    write_binary_file(rnd_data, rnd_path)
    rnd_stats = analyze_basic_stats(rnd_data)
    print(f"  Written: {rnd_path} ({len(rnd_data):,} bytes)")
    print(f"  Proportion of 1s: {rnd_stats['proportion_ones']:.6f} (ideal: 0.5)")
    print(f"  Frequency assessment: {rnd_stats['frequency_assessment']}")
    
    # Write manifest
    manifest = {
        "commit_sha": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "key_hex": key.hex(),
        "megabits": args.megabits,
        "blocks": blocks_needed,
        "cipher": "EnhancedRFTCryptoV2",
        "files": {
            "ctr_stream.bin": {
                "mode": "CTR",
                "bytes": len(ctr_data),
                "stats": ctr_stats,
            },
            "random_encryption.bin": {
                "mode": "random plaintext encryption",
                "bytes": len(rnd_data),
                "stats": rnd_stats,
            }
        }
    }
    
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest, indent=2)
    
    print(f"\n✓ Artifacts written to {args.output}/")
    print(f"  - ctr_stream.bin")
    print(f"  - random_encryption.bin")
    print(f"  - manifest.json")
    
    print("\n" + "="*60)
    print("NEXT STEPS: Run NIST Statistical Test Suite")
    print("="*60)
    print("""
1. Download NIST STS:
   wget https://csrc.nist.gov/CSRC/media/Projects/Random-Bit-Generation/documents/sts-2_1_2.zip
   unzip sts-2_1_2.zip

2. Build:
   cd sts-2.1.2
   make -f makefile

3. Run on CTR stream:
   ./assess {bits}
   # Select option 0 (input file)
   # Enter: {ctr_path}
   # Select all tests (option 1)
   
4. Check results in:
   experiments/AlgorithmTesting/finalAnalysisReport.txt

Alternative: Use Python dieharder wrapper:
   pip install sp800_22_tests
   python -c "from sp800_22_tests import *; run_all_tests('{ctr_path}')"
""".format(bits=bits_needed, ctr_path=ctr_path.absolute()))


if __name__ == "__main__":
    main()
