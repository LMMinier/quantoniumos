#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Cryptographic Primitives - QuantoniumOS (RESEARCH ONLY)
=======================================================

Core cryptographic building blocks using RFT-enhanced algorithms.
Provides authenticated encryption, key derivation, and cryptographic hashing.

**CRITICAL DISCLAIMER:**
This module is for ACADEMIC RESEARCH ONLY. The cryptographic constructions
have NOT been formally analyzed, peer-reviewed, or certified by any standards
body. Do NOT use for production security applications. The term "quantum"
refers to mathematical inspiration, NOT proven post-quantum security.
"""

import os
import hashlib
import hmac
import secrets
from typing import Union, Tuple, List, Optional, Callable
import numpy as np

# Try to import RFT components
try:
    from ..assembly.python_bindings import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class HMACSHA256:
    """HMAC-SHA256 implementation with RFT enhancement."""

    @staticmethod
    def compute(key: bytes, message: bytes) -> bytes:
        """Compute HMAC-SHA256 of message using key."""
        return hmac.new(key, message, hashlib.sha256).digest()

    @staticmethod
    def verify(key: bytes, message: bytes, mac: bytes) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = HMACSHA256.compute(key, message)
        return hmac.compare_digest(expected, mac)

class RFTEnhancedHMAC:
    """RFT-enhanced HMAC for experimental cryptographic research.
    
    WARNING: NOT CERTIFIED. Research use only. Security properties unverified.
    """

    @staticmethod
    def compute(key: bytes, message: bytes) -> bytes:
        """Compute RFT-enhanced HMAC."""
        if not RFT_AVAILABLE:
            # Fallback to standard HMAC
            return HMACSHA256.compute(key, message)

        try:
            # Use RFT to enhance the key scheduling
            key_size = 64  # RFT transform size
            rft = UnitaryRFT(key_size, RFT_FLAG_QUANTUM_SAFE)

            # Convert key to complex array for RFT
            key_array = np.frombuffer(key, dtype=np.uint8).astype(np.float64)
            key_array = key_array[:key_size]  # Truncate or pad as needed
            if len(key_array) < key_size:
                key_array = np.pad(key_array, (0, key_size - len(key_array)))

            # Apply RFT transform to key
            key_complex = key_array + 0j
            rft_key = rft.forward(key_complex)

            # Use transformed key for HMAC
            key_bytes = rft_key.real.tobytes()[:32]  # Take first 32 bytes
            return hmac.new(key_bytes, message, hashlib.sha256).digest()

        except Exception:
            # Fallback to standard HMAC if RFT fails
            return HMACSHA256.compute(key, message)

class FeistelNetwork:
    """Feistel network implementation for block ciphers."""

    def __init__(self, rounds: int = 16, block_size: int = 16):
        """Initialize Feistel network.

        Args:
            rounds: Number of encryption rounds
            block_size: Block size in bytes (must be even)
        """
        if block_size % 2 != 0:
            raise ValueError("Block size must be even for Feistel network")

        self.rounds = rounds
        self.block_size = block_size
        self.half_block = block_size // 2

    def _round_function(self, right: bytes, round_key: bytes) -> bytes:
        """Round function for Feistel network."""
        # Simple round function: HMAC of right half with round key
        mac = HMACSHA256.compute(round_key, right)
        return mac[:self.half_block]

    def _generate_round_keys(self, key: bytes) -> List[bytes]:
        """Generate round keys from master key."""
        round_keys = []
        for i in range(self.rounds):
            # Derive round key using HMAC
            round_key = HMACSHA256.compute(key, i.to_bytes(4, 'big'))
            round_keys.append(round_key[:self.half_block])
        return round_keys

    def encrypt_block(self, block: bytes, key: bytes) -> bytes:
        """Encrypt a single block."""
        if len(block) != self.block_size:
            raise ValueError(f"Block size must be {self.block_size} bytes")

        round_keys = self._generate_round_keys(key)
        left = block[:self.half_block]
        right = block[self.half_block:]

        for round_key in round_keys:
            new_left = right
            f_output = self._round_function(right, round_key)
            new_right = bytes(a ^ b for a, b in zip(left, f_output))
            left, right = new_left, new_right

        return left + right

    def decrypt_block(self, block: bytes, key: bytes) -> bytes:
        """Decrypt a single block."""
        if len(block) != self.block_size:
            raise ValueError(f"Block size must be {self.block_size} bytes")

        round_keys = self._generate_round_keys(key)
        round_keys.reverse()  # Use keys in reverse order for decryption

        left = block[:self.half_block]
        right = block[self.half_block:]

        for round_key in round_keys:
            new_right = left
            f_output = self._round_function(left, round_key)
            new_left = bytes(a ^ b for a, b in zip(right, f_output))
            left, right = new_left, new_right

        return left + right

class RFTEnhancedFeistel:
    """RFT-enhanced Feistel network for experimental cryptographic research.
    
    WARNING: NOT CERTIFIED. Research use only. Security properties unverified.
    """

    def __init__(self, rounds: int = 48, block_size: int = 16):
        """Initialize RFT-enhanced Feistel network."""
        self.rounds = rounds
        self.block_size = block_size
        self.half_block = block_size // 2

    def _rft_round_function(self, right: bytes, round_key: bytes) -> bytes:
        """RFT-enhanced round function."""
        if not RFT_AVAILABLE:
            # Fallback to standard HMAC
            mac = HMACSHA256.compute(round_key, right)
            return mac[:self.half_block]

        try:
            # Use RFT for round function
            rft_size = 32  # Small RFT for speed
            rft = UnitaryRFT(rft_size, RFT_FLAG_QUANTUM_SAFE)

            # Combine right half and round key
            combined = right + round_key
            data_array = np.frombuffer(combined, dtype=np.uint8).astype(np.float64)
            data_array = data_array[:rft_size]
            if len(data_array) < rft_size:
                data_array = np.pad(data_array, (0, rft_size - len(data_array)))

            # Apply RFT transform
            data_complex = data_array + 0j
            rft_result = rft.forward(data_complex)

            # Convert back to bytes
            result_bytes = rft_result.real.tobytes()
            return result_bytes[:self.half_block]

        except Exception:
            # Fallback to standard HMAC
            mac = HMACSHA256.compute(round_key, right)
            return mac[:self.half_block]

    def _generate_rft_round_keys(self, key: bytes) -> List[bytes]:
        """Generate RFT-enhanced round keys."""
        round_keys = []
        for i in range(self.rounds):
            if RFT_AVAILABLE:
                try:
                    # Use RFT to enhance key derivation
                    rft = UnitaryRFT(64, RFT_FLAG_QUANTUM_SAFE)
                    key_data = np.frombuffer(key + i.to_bytes(4, 'big'), dtype=np.uint8).astype(np.float64)
                    key_data = key_data[:64]
                    if len(key_data) < 64:
                        key_data = np.pad(key_data, (0, 64 - len(key_data)))

                    rft_key = rft.forward(key_data + 0j)
                    round_key = rft_key.real.tobytes()[:self.half_block]
                except Exception:
                    # Fallback key derivation
                    round_key = HMACSHA256.compute(key, i.to_bytes(4, 'big'))[:self.half_block]
            else:
                # Standard key derivation
                round_key = HMACSHA256.compute(key, i.to_bytes(4, 'big'))[:self.half_block]

            round_keys.append(round_key)
        return round_keys

    def encrypt_block(self, block: bytes, key: bytes) -> bytes:
        """Encrypt a single block with RFT enhancement."""
        if len(block) != self.block_size:
            raise ValueError(f"Block size must be {self.block_size} bytes")

        round_keys = self._generate_rft_round_keys(key)
        left = block[:self.half_block]
        right = block[self.half_block:]

        for round_key in round_keys:
            new_left = right
            f_output = self._rft_round_function(right, round_key)
            new_right = bytes(a ^ b for a, b in zip(left, f_output))
            left, right = new_left, new_right

        return left + right

    def decrypt_block(self, block: bytes, key: bytes) -> bytes:
        """Decrypt a single block with RFT enhancement."""
        if len(block) != self.block_size:
            raise ValueError(f"Block size must be {self.block_size} bytes")

        round_keys = self._generate_rft_round_keys(key)
        round_keys.reverse()

        left = block[:self.half_block]
        right = block[self.half_block:]

        for round_key in round_keys:
            new_right = left
            f_output = self._rft_round_function(left, round_key)
            new_left = bytes(a ^ b for a, b in zip(right, f_output))
            left, right = new_left, new_right

        return left + right

class KeyDerivation:
    """Key derivation functions."""

    @staticmethod
    def hkdf(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
        """HKDF key derivation function."""
        # Simplified HKDF implementation
        if len(salt) == 0:
            salt = b'\x00' * 32

        # Extract
        prk = HMACSHA256.compute(salt, key)

        # Expand
        okm = b''
        t = b''
        for i in range((length + 31) // 32):
            t = HMACSHA256.compute(prk, t + info + bytes([i + 1]))
            okm += t

        return okm[:length]

    @staticmethod
    def pbkdf2(password: bytes, salt: bytes, iterations: int, length: int) -> bytes:
        """PBKDF2 key derivation function."""
        def _pbkdf2_f(password: bytes, salt: bytes, iterations: int, block_num: int) -> bytes:
            u = hmac.new(password, salt + block_num.to_bytes(4, 'big'), hashlib.sha256).digest()
            result = u
            for _ in range(iterations - 1):
                u = hmac.new(password, u, hashlib.sha256).digest()
                result = bytes(a ^ b for a, b in zip(result, u))
            return result

        derived_key = b''
        blocks_needed = (length + 31) // 32

        for i in range(1, blocks_needed + 1):
            derived_key += _pbkdf2_f(password, salt, iterations, i)

        return derived_key[:length]

class RandomGenerator:
    """Cryptographically secure random number generation."""

    @staticmethod
    def generate_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)

    @staticmethod
    def generate_int(min_value: int, max_value: int) -> int:
        """Generate cryptographically secure random integer."""
        return secrets.randbelow(max_value - min_value + 1) + min_value

# Export all cryptographic primitives
__all__ = [
    'HMACSHA256', 'RFTEnhancedHMAC',
    'FeistelNetwork', 'RFTEnhancedFeistel',
    'KeyDerivation', 'RandomGenerator'
]