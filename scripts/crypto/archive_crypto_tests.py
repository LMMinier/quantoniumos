#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate cryptographic test artifacts for EnhancedRFTCryptoV2.

Uses the ACTUAL implementation from algorithms.rft.crypto.enhanced_cipher.

Outputs:
- Known-Answer Test (KAT) vectors
- Avalanche effect measurements  
- Basic differential analysis

⚠️ WARNING: This does NOT constitute a security analysis.
These are minimum sanity checks, not cryptographic validation.
"""

# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from atomic_io import atomic_write_json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the ACTUAL cipher implementation
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


def encrypt_block(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt using the ACTUAL EnhancedRFTCryptoV2."""
    # Pad key to 32 bytes if needed
    if len(key) < 32:
        key = key + b'\x00' * (32 - len(key))
    cipher = EnhancedRFTCryptoV2(key[:32])
    
    # Pad plaintext to 16 bytes
    if len(plaintext) < 16:
        plaintext = plaintext + b'\x00' * (16 - len(plaintext))
    
    return cipher._feistel_encrypt(plaintext[:16])


def decrypt_block(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt using the ACTUAL EnhancedRFTCryptoV2."""
    if len(key) < 32:
        key = key + b'\x00' * (32 - len(key))
    cipher = EnhancedRFTCryptoV2(key[:32])
    return cipher._feistel_decrypt(ciphertext[:16])


def hamming_distance(a: bytes, b: bytes) -> int:
    """Count differing bits between two byte strings."""
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))


def generate_kat_vectors() -> List[Dict]:
    """Generate Known-Answer Test vectors."""
    vectors = []
    
    test_cases = [
        ("All-zero", bytes(32), bytes(16)),
        ("All-one key", bytes([0xFF] * 32), bytes(16)),
        ("All-one plaintext", bytes(32), bytes([0xFF] * 16)),
        ("Counter pattern key", bytes(range(32)), bytes(16)),
        ("Counter pattern plaintext", bytes(32), bytes(range(16))),
        ("Alternating bits", bytes([0xAA] * 32), bytes([0x55] * 16)),
    ]
    
    for name, key, plaintext in test_cases:
        ciphertext = encrypt_block(plaintext, key)
        # Verify decryption
        decrypted = decrypt_block(ciphertext, key)
        vectors.append({
            "name": name,
            "key_hex": key.hex(),
            "plaintext_hex": plaintext.hex(),
            "ciphertext_hex": ciphertext.hex(),
            "decrypt_matches": decrypted == plaintext,
        })
    
    return vectors


def test_avalanche_effect(num_trials: int = 1000, seed: int = 42) -> Dict:
    """
    Test avalanche effect: single bit flip should change ~50% of output bits.
    """
    np.random.seed(seed)
    
    bit_changes = []
    
    for _ in range(num_trials):
        # Random key and plaintext
        key = bytes(np.random.randint(0, 256, 32, dtype=np.uint8))
        plaintext = bytes(np.random.randint(0, 256, 16, dtype=np.uint8))
        
        # Original ciphertext
        c1 = encrypt_block(plaintext, key)
        
        # Flip one random bit in plaintext
        bit_pos = np.random.randint(0, 128)
        byte_pos = bit_pos // 8
        bit_in_byte = bit_pos % 8
        
        modified = bytearray(plaintext)
        modified[byte_pos] ^= (1 << bit_in_byte)
        
        # Modified ciphertext
        c2 = encrypt_block(bytes(modified), key)
        
        # Count bit changes
        changes = hamming_distance(c1, c2)
        bit_changes.append(changes)
    
    bit_changes = np.array(bit_changes)
    
    mean_changed = float(np.mean(bit_changes))
    pct_of_ideal = mean_changed / 64.0 * 100
    
    # PASS if within 90-110% of ideal (45-55% bit changes for 128-bit output)
    is_pass = 90 <= pct_of_ideal <= 110
    
    return {
        "num_trials": num_trials,
        "seed": seed,
        "total_output_bits": 128,
        "mean_bits_changed": mean_changed,
        "std_bits_changed": float(np.std(bit_changes)),
        "min_bits_changed": int(np.min(bit_changes)),
        "max_bits_changed": int(np.max(bit_changes)),
        "expected_for_ideal": 64.0,
        "percentage_of_ideal": pct_of_ideal,
        "assessment": "PASS" if is_pass else "FAIL",
        "note": "Ideal avalanche = 50% bits change = 64 bits for 128-bit output"
    }


def test_basic_differential(num_trials: int = 1000, seed: int = 42) -> Dict:
    """
    Basic differential analysis: check output difference distribution.
    This is NOT a proper differential cryptanalysis, just a sanity check.
    """
    np.random.seed(seed)
    
    # Test with fixed input difference (single bit)
    fixed_diff = bytes([0x01] + [0x00] * 15)  # Flip first bit
    
    output_diffs = []
    
    for _ in range(num_trials):
        key = bytes(np.random.randint(0, 256, 32, dtype=np.uint8))
        p1 = bytes(np.random.randint(0, 256, 16, dtype=np.uint8))
        p2 = bytes(a ^ b for a, b in zip(p1, fixed_diff))
        
        c1 = encrypt_block(p1, key)
        c2 = encrypt_block(p2, key)
        
        # Output XOR difference
        diff = bytes(a ^ b for a, b in zip(c1, c2))
        output_diffs.append(diff.hex())
    
    # Check if output differences are uniformly distributed
    unique_diffs = len(set(output_diffs))
    
    return {
        "num_trials": num_trials,
        "seed": seed,
        "input_difference_hex": fixed_diff.hex(),
        "unique_output_differences": unique_diffs,
        "uniqueness_ratio": unique_diffs / num_trials,
        "assessment": "PASS (diverse)" if unique_diffs > num_trials * 0.9 else "FAIL (biased)",
        "note": "This is a sanity check, NOT a security proof"
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate crypto test artifacts")
    parser.add_argument("--output", default="data/artifacts/crypto_tests",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = get_git_commit()
    
    print(f"Generating crypto test artifacts to {args.output}/")
    print(f"Commit: {commit_sha}")
    print(f"Seed: {args.seed}")
    print(f"Using: ACTUAL EnhancedRFTCryptoV2 implementation")
    
    # Manifest
    manifest = {
        "commit_sha": commit_sha,
        "timestamp": timestamp,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "cipher": "EnhancedRFTCryptoV2 (ACTUAL implementation)",
        "implementation_path": "algorithms/rft/crypto/enhanced_cipher.py",
        "block_size_bits": 128,
        "key_size_bits": 256,
        "num_rounds": 48,
        "warning": "These are sanity checks, NOT security validation"
    }
    
    # 1. KAT vectors
    print("\n[1/3] Generating KAT vectors...")
    kat_vectors = generate_kat_vectors()
    all_decrypt_ok = all(v['decrypt_matches'] for v in kat_vectors)
    for v in kat_vectors:
        status = "✓" if v['decrypt_matches'] else "✗"
        print(f"  {status} {v['name']}: {v['ciphertext_hex'][:16]}...")
    
    atomic_write_json(os.path.join(args.output, "kat_vectors.json"), kat_vectors, indent=2)
    
    # 2. Avalanche test
    print("\n[2/3] Testing avalanche effect...")
    avalanche = test_avalanche_effect(num_trials=1000, seed=args.seed)
    print(f"  Mean bits changed: {avalanche['mean_bits_changed']:.1f} / 128")
    print(f"  Percentage of ideal: {avalanche['percentage_of_ideal']:.1f}%")
    print(f"  Assessment: {avalanche['assessment']}")
    
    atomic_write_json(os.path.join(args.output, "avalanche_test.json"), avalanche, indent=2)
    
    # 3. Basic differential
    print("\n[3/3] Basic differential test...")
    differential = test_basic_differential(num_trials=1000, seed=args.seed)
    print(f"  Unique output differences: {differential['unique_output_differences']} / 1000")
    print(f"  Assessment: {differential['assessment']}")
    
    with open(os.path.join(args.output, "differential_test.json"), 'w') as f:
        json.dump(differential, f, indent=2)
    
    # Write manifest
    atomic_write_json(os.path.join(args.output, "manifest.json"), manifest, indent=2)
    
    print(f"\n✓ Artifacts written to {args.output}/")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    atomic_write_json(os.path.join(args.output, "differential_test.json"), differential, indent=2)
    print(f"  Differential diversity: {differential['assessment']}")


if __name__ == "__main__":
    main()
