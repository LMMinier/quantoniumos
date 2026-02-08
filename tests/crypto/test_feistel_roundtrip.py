#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Feistel 48-Round Roundtrip Gate
================================

Automated unit test proving encrypt → decrypt == identity for
the EnhancedRFTCryptoV2 cipher.

If this test fails, the cipher is broken and must not be used.

Tests:
1. Single block roundtrip (16 bytes)
2. Multi-block roundtrip (message > 1 block)
3. AEAD roundtrip with associated data
4. Different keys produce different ciphertexts
5. Bit-flip in ciphertext fails authentication
6. Empty and edge-case payloads
"""
import secrets

import numpy as np
import pytest

from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2


@pytest.fixture
def cipher():
    """Fresh cipher instance with random key."""
    return EnhancedRFTCryptoV2(master_key=secrets.token_bytes(32))


@pytest.fixture
def fixed_cipher():
    """Deterministic cipher for reproducible tests."""
    return EnhancedRFTCryptoV2(master_key=b"\x42" * 32)


# ---------------------------------------------------------------------------
# Core roundtrip: encrypt(decrypt(x)) == x
# ---------------------------------------------------------------------------

class TestFeistelRoundtrip:
    """Prove encrypt → decrypt == identity."""

    def test_single_block_roundtrip(self, fixed_cipher):
        """Exactly one 128-bit block (16 bytes)."""
        pt = b"EXACTLY16BYTES!!"
        assert len(pt) == 16
        ct = fixed_cipher.encrypt_aead(pt)
        recovered = fixed_cipher.decrypt_aead(ct)
        assert recovered == pt, "Single-block roundtrip failed"

    def test_multi_block_roundtrip(self, fixed_cipher):
        """Multiple blocks — padding must be handled correctly."""
        pt = b"This message is longer than one block and exercises padding."
        ct = fixed_cipher.encrypt_aead(pt)
        recovered = fixed_cipher.decrypt_aead(ct)
        assert recovered == pt, "Multi-block roundtrip failed"

    @pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 31, 32, 48, 100, 255, 256, 1024])
    def test_roundtrip_various_sizes(self, fixed_cipher, size):
        """Roundtrip for every boundary-adjacent payload size."""
        pt = secrets.token_bytes(size) if size > 0 else b""
        ct = fixed_cipher.encrypt_aead(pt)
        recovered = fixed_cipher.decrypt_aead(ct)
        assert recovered == pt, f"Roundtrip failed at size={size}"

    def test_roundtrip_with_associated_data(self, fixed_cipher):
        """AEAD: associated data must be authenticated but not encrypted."""
        pt = b"secret payload"
        ad = b"authenticated header"
        ct = fixed_cipher.encrypt_aead(pt, associated_data=ad)
        recovered = fixed_cipher.decrypt_aead(ct, associated_data=ad)
        assert recovered == pt, "AEAD roundtrip with associated data failed"

    def test_random_key_random_payload(self):
        """100 random key+payload pairs must all roundtrip."""
        for _ in range(100):
            key = secrets.token_bytes(32)
            cipher = EnhancedRFTCryptoV2(master_key=key)
            pt = secrets.token_bytes(secrets.randbelow(200) + 1)
            ct = cipher.encrypt_aead(pt)
            assert cipher.decrypt_aead(ct) == pt


# ---------------------------------------------------------------------------
# Integrity: ciphertext modification must be detected
# ---------------------------------------------------------------------------

class TestFeistelIntegrity:
    """Prove tamper detection works."""

    def test_bitflip_detected(self, fixed_cipher):
        """Single bit flip in ciphertext must fail decryption."""
        pt = b"integrity check payload!!"
        ct = fixed_cipher.encrypt_aead(pt)
        tampered = bytearray(ct)
        # Flip a bit in the middle of the ciphertext body
        mid = len(tampered) // 2
        tampered[mid] ^= 0x01
        tampered = bytes(tampered)
        with pytest.raises(Exception):
            fixed_cipher.decrypt_aead(tampered)

    def test_wrong_associated_data_fails(self, fixed_cipher):
        """Wrong associated data must fail AEAD verification."""
        pt = b"secret"
        ct = fixed_cipher.encrypt_aead(pt, associated_data=b"correct")
        with pytest.raises(Exception):
            fixed_cipher.decrypt_aead(ct, associated_data=b"wrong")


# ---------------------------------------------------------------------------
# Key separation: different keys → different ciphertexts
# ---------------------------------------------------------------------------

class TestFeistelKeySeparation:
    """Prove different keys produce different outputs."""

    def test_different_keys_different_ciphertext(self):
        """Same plaintext under different keys must produce different ciphertext."""
        pt = b"key separation test block"
        c1 = EnhancedRFTCryptoV2(master_key=b"\x01" * 32)
        c2 = EnhancedRFTCryptoV2(master_key=b"\x02" * 32)
        ct1 = c1.encrypt_aead(pt)
        ct2 = c2.encrypt_aead(pt)
        assert ct1 != ct2, "Different keys produced identical ciphertext"

    def test_avalanche_single_block(self, fixed_cipher):
        """One-bit change in plaintext must flip ~50% of ciphertext bits."""
        pt1 = b"\x00" * 16
        pt2 = b"\x01" + b"\x00" * 15  # flip one bit
        ct1 = fixed_cipher._feistel_encrypt(pt1)
        ct2 = fixed_cipher._feistel_encrypt(pt2)
        diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(ct1, ct2))
        total_bits = len(ct1) * 8
        avalanche = diff_bits / total_bits
        assert 0.35 < avalanche < 0.65, (
            f"Avalanche {avalanche*100:.1f}% outside [35%, 65%]"
        )


# ---------------------------------------------------------------------------
# Internal Feistel block: raw encrypt/decrypt without AEAD wrapper
# ---------------------------------------------------------------------------

class TestFeistelBlockLevel:
    """Prove the raw _feistel_encrypt/_feistel_decrypt pair is an involution."""

    def test_raw_block_roundtrip(self, fixed_cipher):
        """Raw block encrypt+decrypt must be identity."""
        pt = secrets.token_bytes(16)
        ct = fixed_cipher._feistel_encrypt(pt)
        recovered = fixed_cipher._feistel_decrypt(ct)
        assert recovered == pt, "Raw block roundtrip failed"

    def test_raw_block_100_random(self):
        """100 random blocks under 100 random keys."""
        for _ in range(100):
            key = secrets.token_bytes(32)
            cipher = EnhancedRFTCryptoV2(master_key=key)
            pt = secrets.token_bytes(16)
            ct = cipher._feistel_encrypt(pt)
            recovered = cipher._feistel_decrypt(ct)
            assert recovered == pt, "Random block roundtrip failed"

    def test_encrypt_is_not_identity(self, fixed_cipher):
        """Encryption must actually change the plaintext."""
        pt = b"\xAA" * 16
        ct = fixed_cipher._feistel_encrypt(pt)
        assert ct != pt, "Encryption produced plaintext unchanged"

    def test_decrypt_is_not_encrypt(self, fixed_cipher):
        """encrypt(x) != decrypt(x) for non-trivial input."""
        pt = secrets.token_bytes(16)
        ct = fixed_cipher._feistel_encrypt(pt)
        dt = fixed_cipher._feistel_decrypt(pt)
        assert ct != dt, "encrypt and decrypt produced same output"
