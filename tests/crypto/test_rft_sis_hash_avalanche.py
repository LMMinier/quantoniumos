#!/usr/bin/env python3
"""Test RFT-SIS Hash Avalanche Effect"""
from algorithms.rft.crypto.rft_sis_hash import RFTSISHash
import numpy as np

def hamming_distance(h1: bytes, h2: bytes) -> int:
    """Count bit differences."""
    return sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(h1, h2))

def test_avalanche():
    """Test if 1-bit input change affects ~50% of output bits."""
    hasher = RFTSISHash()
    
    # Test vectors
    test_cases = [
        b"hello",
        b"Hello",  # 1-bit change
        b"test message",
        b"test messag",  # 1 char removed
        b"\x00" * 32,
        b"\x00" * 31 + b"\x01",  # 1-bit flip at end
    ]
    
    results = []
    for i in range(0, len(test_cases), 2):
        h1 = hasher.hash(test_cases[i])
        h2 = hasher.hash(test_cases[i+1])
        
        bits_changed = hamming_distance(h1, h2)
        total_bits = len(h1) * 8
        percentage = 100 * bits_changed / total_bits
        
        print(f"Input: {test_cases[i]!r}")
        print(f"   vs: {test_cases[i+1]!r}")
        print(f"  Bits changed: {bits_changed}/{total_bits} ({percentage:.1f}%)")
        print(f"  Hash1: {h1.hex()[:32]}...")
        print(f"  Hash2: {h2.hex()[:32]}...")
        print()
        
        results.append(percentage)
    
    avg = np.mean(results)
    print(f"Average avalanche: {avg:.1f}% (ideal: 50%)")
    
    if 45 <= avg <= 55:
        print("✓ Good avalanche effect")
    else:
        print("✗ Poor avalanche - not cryptographically strong")

if __name__ == "__main__":
    test_avalanche()
