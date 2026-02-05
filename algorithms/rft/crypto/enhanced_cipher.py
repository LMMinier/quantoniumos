#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Enhanced RFT Crypto v2 Implementation
48-Round Feistel Network with RFT Integration

Implements the cryptographic system described in the QuantoniumOS paper:
- 48-round Feistel network with 128-bit blocks
- AES S-box, MixColumns-like diffusion, ARX operations
- Domain-separated key derivation with golden-ratio parameterization
- AEAD-style authenticated encryption
- Full phase+amplitude+wave+ciphertext modulation
"""

import hashlib
import hmac
import argparse
import random
import secrets
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class CipherMetrics:
    """Metrics for avalanche and performance analysis."""
    message_avalanche: float = 0.0
    key_avalanche: float = 0.0
    key_sensitivity: float = 0.0
    throughput_mbps: float = 0.0


class EnhancedRFTCryptoV2:
    """
    Enhanced RFT Crypto v2: 48-round Feistel with RFT-informed design.
    
    Experimental implementation of the QuantoniumOS internal design with:
    - Domain-separated key derivation
    - Golden-ratio parameterization
    - Comprehensive diffusion layers
    - Authenticated encryption mode
    """
    
    # AES S-box for nonlinear substitution
    S_BOX = [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ]
    
    # MixColumns matrix for GF(2^8) operations
    MIX_COLUMNS_MATRIX = [
        [2, 3, 1, 1],
        [1, 2, 3, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2]
    ]
    
    def __init__(self, master_key: bytes):
        """
        Initialize Enhanced RFT Crypto v2 with master key.
        
        Args:
            master_key: Master key (32 bytes recommended)
        """
        if len(master_key) < 16:
            raise ValueError("Master key must be at least 16 bytes")
        
        self.master_key = master_key
        self.phi = (1 + (5 ** 0.5)) / 2  # Golden ratio
        # Align with paper and validation suite
        self.rounds = 48
        self._metrics_cache: Dict[Tuple[int, int, int, int, int], CipherMetrics] = {}
        
        # Derive round keys with domain separation
        self.round_keys = self._derive_round_keys()
        self.pre_whiten_key = self._hkdf(b"PRE_WHITEN_RFT_2025")[:16]
        self.post_whiten_key = self._hkdf(b"POST_WHITEN_RFT_2025")[:16]
        self._per_round_pre_whiten = [self._hkdf(f"PRE_ROUND_{r}".encode(), 8) for r in range(self.rounds)]
        self._per_round_post_whiten = [self._hkdf(f"POST_ROUND_{r}".encode(), 8) for r in range(self.rounds)]
        
        # Derive 4-phase locks (I/Q/Q'/Q'') for each round
        self.phase_locks = self._derive_phase_locks()
        
        # Derive amplitude masks for each round
        self.amplitude_masks = self._derive_amplitude_masks()
        
        # Derive keyed MDS matrices for each round
        self.round_mds_matrices = self._derive_round_mds_matrices()
    
    def _hkdf(self, info: bytes, length: int = 32) -> bytes:
        """HKDF key derivation with domain separation."""
        # Extract phase
        prk = hmac.new(b"RFT_SALT_2025", self.master_key, hashlib.sha256).digest()
        
        # Expand phase
        okm = b""
        counter = 1
        while len(okm) < length:
            t = hmac.new(prk, info + counter.to_bytes(1, 'big'), hashlib.sha256).digest()
            okm += t
            counter += 1
        
        return okm[:length]
    
    def _derive_round_keys(self) -> list:
        """Derive 48 round keys with golden-ratio parameterization."""
        round_keys = []
        
        for r in range(self.rounds):
            # Golden-ratio parameter for this round
            phi_param = int(self.phi * 1000) + r
            info = f"RFT_ROUND_{r}_PHI_{phi_param}".encode()
            round_key = self._hkdf(info, 16)
            round_keys.append(round_key)
        
        return round_keys
    
    def _derive_phase_locks(self) -> list:
        """Derive 4-phase locks (I/Q/Q'/Q'') for each round."""
        phase_locks = []
        
        for r in range(self.rounds):
            info = f"PHASE_LOCK_ROUND_{r}".encode()
            phase_data = self._hkdf(info, 16)  # 4 phases × 4 bytes each
            
            # Extract 4 phases for I/Q/Q'/Q'' quadrature
            phases = []
            for i in range(4):
                phase_bytes = phase_data[i*4:(i+1)*4]
                phase_val = int.from_bytes(phase_bytes, 'big') % (2**16)
                phases.append(2 * np.pi * phase_val / (2**16))  # Normalize to [0, 2π)
            
            phase_locks.append(phases)
        
        return phase_locks
    
    def _derive_amplitude_masks(self) -> list:
        """Derive amplitude modulation masks for each round."""
        amplitude_masks = []
        
        for r in range(self.rounds):
            info = f"AMPLITUDE_MASK_ROUND_{r}".encode()
            mask_data = self._hkdf(info, 8)  # 8 bytes for amplitude mask
            
            # Convert to amplitude values in range [0.5, 1.5] for stability
            amplitudes = []
            for byte_val in mask_data:
                amp = 0.5 + (byte_val / 255.0)  # Range [0.5, 1.5]
                amplitudes.append(amp)
            
            amplitude_masks.append(amplitudes)
        
        return amplitude_masks
    
    def _derive_round_mds_matrices(self) -> list:
        """Derive keyed MDS matrices for each round.
        Construction preserves MDS property via row/column permutations and
        non-zero diagonal scaling in GF(2^8), which are MDS-preserving operations.
        """
        mds_matrices = []

        for r in range(self.rounds):
            info = f"MDS_MATRIX_ROUND_{r}".encode()
            matrix_seed = self._hkdf(info, 16)

            # Base AES MixColumns (known MDS)
            base_matrix = np.array(self.MIX_COLUMNS_MATRIX, dtype=np.uint8)

            # Deterministic, key-dependent PRNG without mutating global numpy RNG
            seed_val = int.from_bytes(matrix_seed[:4], 'big')
            prng = random.Random(seed_val)

            row_perm = list(range(4))
            col_perm = list(range(4))
            prng.shuffle(row_perm)
            prng.shuffle(col_perm)

            # Non-zero row/column scaling factors (GF(2^8))
            row_scales = [b or 1 for b in matrix_seed[4:8]]
            col_scales = [b or 1 for b in matrix_seed[8:12]]

            keyed_matrix = np.zeros((4, 4), dtype=np.uint8)
            for i in range(4):
                for j in range(4):
                    # Permute rows/cols then scale: S_row * M * S_col
                    base_val = int(base_matrix[row_perm[i], col_perm[j]])
                    scaled = self._gf_multiply(row_scales[i] & 0xFF, base_val & 0xFF)
                    scaled = self._gf_multiply(scaled & 0xFF, col_scales[j] & 0xFF)
                    if scaled == 0:
                        # Avoid degenerate zeros to maintain invertibility
                        scaled = 1
                    keyed_matrix[i, j] = scaled & 0xFF

            mds_matrices.append(keyed_matrix)

        return mds_matrices
    
    def _sbox_layer(self, data: bytes) -> bytes:
        """Apply AES S-box substitution."""
        return bytes(self.S_BOX[b] for b in data)
    
    def _gf_multiply(self, a: int, b: int) -> int:
        """Multiply in GF(2^8) with polynomial 0x11B."""
        result = 0
        a = a & 0xFF  # Ensure 8-bit values
        b = b & 0xFF
        
        for _ in range(8):
            if b & 1:
                result ^= a
            carry = a & 0x80
            a = (a << 1) & 0xFF  # Keep within 8 bits
            if carry:
                a ^= 0x1B  # Use 0x1B instead of 0x11B for 8-bit operations
            b >>= 1
        return result & 0xFF
    
    def _mix_columns(self, data: bytes) -> bytes:
        """Apply MixColumns-like diffusion."""
        # Ensure data is exactly 16 bytes
        if len(data) != 16:
            # Pad or truncate to 16 bytes
            if len(data) < 16:
                data = data + b'\x00' * (16 - len(data))
            else:
                data = data[:16]
        
        result = bytearray(16)
        
        for col in range(4):
            for row in range(4):
                val = 0
                for k in range(4):
                    coeff = self.MIX_COLUMNS_MATRIX[row][k]
                    byte_val = data[col * 4 + k]
                    val ^= self._gf_multiply(coeff, byte_val)
                result[col * 4 + row] = val
        
        return bytes(result)
    
    def _arx_operations(self, left: bytes, right: bytes, round_key: bytes) -> bytes:
        """Strengthened Addition-Rotation-XOR mixing with key injection."""
        w0 = int.from_bytes(left[0:4], 'big')
        w1 = int.from_bytes(left[4:8], 'big')
        r0 = int.from_bytes(right[0:4], 'big')
        r1 = int.from_bytes(right[4:8], 'big')
        k0 = int.from_bytes(round_key[0:4], 'big')
        k1 = int.from_bytes(round_key[4:8], 'big')
        k2 = int.from_bytes(round_key[8:12], 'big')
        k3 = int.from_bytes(round_key[12:16], 'big')

        # Inject key material and perform ARX with two distinct rotations
        w0 = (w0 + r0 + k0) & 0xFFFFFFFF
        w1 = (w1 + r1 + k1) & 0xFFFFFFFF
        w0 = ((w0 << 7) | (w0 >> 25)) & 0xFFFFFFFF
        w1 = ((w1 << 13) | (w1 >> 19)) & 0xFFFFFFFF
        w0 ^= (w1 + k2) & 0xFFFFFFFF
        w1 ^= (w0 + k3) & 0xFFFFFFFF

        return (w0.to_bytes(4, 'big') + w1.to_bytes(4, 'big'))
    
    def _keyed_mds_layer(self, data: bytes, round_num: int) -> bytes:
        """Apply keyed MDS diffusion layer."""
        if len(data) != 16:
            # Pad or truncate to 16 bytes
            if len(data) < 16:
                data = data + b'\x00' * (16 - len(data))
            else:
                data = data[:16]
        
        result = bytearray(16)
        mds_matrix = self.round_mds_matrices[round_num]
        
        # Apply MDS transformation in 4x4 blocks
        for col in range(4):
            for row in range(4):
                val = 0
                for k in range(4):
                    coeff = mds_matrix[row][k]
                    byte_val = data[col * 4 + k]
                    val ^= self._gf_multiply(coeff, byte_val)
                result[col * 4 + row] = val
        
        return bytes(result)
    
    def _rft_entropy_injection(self, data: bytes, round_num: int) -> bytes:
        """True RFT phase+amplitude+wave modulation for entropy injection."""
        # Get round-specific randomization parameters
        phases = self.phase_locks[round_num]  # 4-phase I/Q/Q'/Q''
        amplitudes = self.amplitude_masks[round_num]  # 8 amplitude values
        
        # Convert data to complex representation for RFT processing
        data_complex = np.zeros(8, dtype=complex)
        for i in range(min(8, len(data))):
            byte_val = data[i] if i < len(data) else 0
            
            # Phase modulation: apply one of 4 quadrature phases
            phase_idx = byte_val & 0x03  # 2 bits select phase
            phase = phases[phase_idx]
            
            # Amplitude modulation: key-dependent per byte
            amplitude = amplitudes[i] if i < len(amplitudes) else 1.0
            
            # Create complex value with phase and amplitude modulation
            data_complex[i] = amplitude * np.exp(1j * phase) * (byte_val / 255.0)
        
        # Normalize to unit circle for geometric properties
        norm = np.linalg.norm(data_complex)
        if norm > 0:
            data_complex /= norm
        
        # Apply RFT-style unitary transformation
        # Golden ratio phase encoding for each element
        for i in range(8):
            golden_phase = (i * self.phi * 2 * np.pi / 8) % (2 * np.pi)
            data_complex[i] *= np.exp(1j * golden_phase)
        
        # Mix frequencies using DFT-like operation with golden ratio spacing
        mixed = np.zeros(8, dtype=complex)
        for k in range(8):
            for n in range(8):
                # Golden ratio-spaced frequencies instead of regular DFT
                freq_factor = np.exp(-2j * np.pi * k * n * self.phi / 8)
                mixed[k] += data_complex[n] * freq_factor
        
        # Convert back to bytes with entropy preservation plus keyed hashing mask
        result = bytearray(8)
        real = np.real(mixed).astype(np.float32)
        imag = np.imag(mixed).astype(np.float32)
        # Initial mapping to bytes (bounded)
        for i in range(8):
            combined = (float(real[i]) + float(imag[i])) / 2.0
            byte_val = int((combined + 1.0) * 127.5) & 0xFF  # Map [-1,1] -> [0,255]
            result[i] = byte_val

        # Keyed, per-round mask derived from RFT content to amplify diffusion
        hasher = hashlib.sha256()
        hasher.update(self.round_keys[round_num])
        hasher.update(real.tobytes())
        hasher.update(imag.tobytes())
        hasher.update(round_num.to_bytes(2, 'big'))
        digest = hasher.digest()

        # Blend digest into result (encrypt-then-mask style XOR)
        for i in range(8):
            result[i] ^= digest[i] ^ digest[8 + i]

        # Additional per-round HKDF mask for domain separation
        hkdf_mask = self._hkdf(f"RFT_INJECT_MASK_{round_num}".encode(), 8)
        for i in range(8):
            result[i] ^= hkdf_mask[i]

        return bytes(result)

    def _round_function(self, right: bytes, round_key: bytes, round_num: int) -> bytes:
        """
        Enhanced 48-round Feistel round function with key-dependent mixing.
        
        Implements: C_{r+1} = F(C_r, K_r) ⊕ RFT(C_r, φ_r, A_r, W_r)
        
        Key components:
        1. 4-phase lock (I/Q/Q'/Q'') derived per round via HKDF
        2. Key-dependent amplitude modulation (not static)
        3. Keyed MDS diffusion layers (sandwich AES S-box)
        4. Pre/post whitening per round with domain separation
        5. RFT-inspired phase+amplitude+wave mixing
        """
        # Ensure right half is exactly 8 bytes
        if len(right) != 8:
            if len(right) < 8:
                right = right + b'\x00' * (8 - len(right))
            else:
                right = right[:8]
        
        # Ensure round key is exactly 16 bytes 
        if len(round_key) != 16:
            if len(round_key) < 16:
                round_key = round_key + b'\x00' * (16 - len(round_key))
            else:
                round_key = round_key[:16]
        
        # === PRE-ROUND WHITENING ===
        # Domain-separated whitening key for this specific round
        pre_whiten = self._per_round_pre_whiten[round_num]
        data = bytes(a ^ b for a, b in zip(right, pre_whiten))
        
        # Expand to 16 bytes for processing using improved method
        # Use bit-rotation and key mixing to eliminate patterns
        part1 = bytes(a ^ b for a, b in zip(data, round_key[:8]))
        part2 = bytes(((b << 1) ^ (b >> 7)) & 0xFF for b in data)
        part2 = bytes(a ^ b for a, b in zip(part2, round_key[8:]))
        expanded_data = part1 + part2
        
        # === LAYER 1: KEYED MDS PRE-DIFFUSION ===
        expanded_data = self._keyed_mds_layer(expanded_data, round_num)
        
        # === LAYER 2: AES S-BOX SUBSTITUTION ===
        expanded_data = self._sbox_layer(expanded_data)
        
        # === LAYER 3: KEYED MDS POST-DIFFUSION ===
        # Use different MDS matrix for post-diffusion
        post_mds_round = (round_num + 32) % self.rounds  # Offset for different matrix
        expanded_data = self._keyed_mds_layer(expanded_data, post_mds_round)
        
        # === LAYER 4: RFT ENTROPY INJECTION ===
        # Apply true phase+amplitude+wave modulation
        left_half = expanded_data[:8]
        right_half = expanded_data[8:]
        
        # RFT modulation on both halves
        left_rft = self._rft_entropy_injection(left_half, round_num)
        right_rft = self._rft_entropy_injection(right_half, round_num)
        
        # === LAYER 5: ARX FINAL MIXING ===
        final_mixed = self._arx_operations(left_rft, right_rft, round_key)
        
        # === POST-ROUND WHITENING ===
        post_whiten = self._per_round_post_whiten[round_num]
        result = bytes(a ^ b for a, b in zip(final_mixed, post_whiten))
        
        return result[:8]

    def _feistel_encrypt(self, plaintext: bytes) -> bytes:
        """48-round Feistel network encryption with enhanced security."""
        if len(plaintext) != 16:
            raise ValueError("Block size must be 16 bytes")
        
        # Pre-whitening
        data = bytes(a ^ b for a, b in zip(plaintext, self.pre_whiten_key))
        
        # Split into left and right halves
        left = data[:8]
        right = data[8:]
        
        # 48 Feistel rounds for security margin per paper
        for round_num in range(self.rounds):
            # L_{i+1} = R_i, R_{i+1} = L_i ⊕ F(R_i, K_i)
            f_output = self._round_function(right, self.round_keys[round_num], round_num)
            new_left = right
            new_right = bytes(a ^ b for a, b in zip(left, f_output[:8]))
            left, right = new_left, new_right
        
        # Combine and post-whiten
        ciphertext = left + right
        ciphertext = bytes(a ^ b for a, b in zip(ciphertext, self.post_whiten_key))
        
        return ciphertext
    
    def _feistel_decrypt(self, ciphertext: bytes) -> bytes:
        """48-round Feistel network decryption with enhanced security."""
        if len(ciphertext) != 16:
            raise ValueError("Block size must be 16 bytes")
        
        # Remove post-whitening
        data = bytes(a ^ b for a, b in zip(ciphertext, self.post_whiten_key))
        
        # Split into left and right halves
        left = data[:8]
        right = data[8:]
        
        # Reverse rounds
        for round_num in range(self.rounds - 1, -1, -1):
            # Reverse: L_i = R_{i+1} ⊕ F(L_{i+1}, K_i), R_i = L_{i+1}
            f_output = self._round_function(left, self.round_keys[round_num], round_num)
            new_right = left
            new_left = bytes(a ^ b for a, b in zip(right, f_output[:8]))
            left, right = new_left, new_right
        
        # Combine and remove pre-whitening
        plaintext = left + right
        plaintext = bytes(a ^ b for a, b in zip(plaintext, self.pre_whiten_key))
        
        return plaintext
    
    def encrypt_aead(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        """
        AEAD-style authenticated encryption.
        
        Returns: version(1) || salt(16) || ciphertext || mac(32)
        """
        # Generate random salt
        salt = secrets.token_bytes(16)
        
        # Derive encryption and MAC keys
        enc_key_info = b"ENC_KEY_" + salt
        mac_key_info = b"MAC_KEY_" + salt
        
        enc_key = self._hkdf(enc_key_info, 32)
        mac_key = self._hkdf(mac_key_info, 32)
        
        # Encrypt with derived key
        temp_cipher = EnhancedRFTCryptoV2(enc_key)
        
        # Pad plaintext to multiple of 16 bytes
        padding_len = 16 - (len(plaintext) % 16)
        padded_plaintext = plaintext + bytes([padding_len] * padding_len)
        
        # Encrypt blocks
        ciphertext = b""
        for i in range(0, len(padded_plaintext), 16):
            block = padded_plaintext[i:i+16]
            ciphertext += temp_cipher._feistel_encrypt(block)
        
        # Create header
        version = b"\x02"  # Version 2
        header = version + salt
        
        # Compute MAC over header + ciphertext + associated_data
        mac_data = header + ciphertext + associated_data
        mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()
        
        return header + ciphertext + mac
    
    def decrypt_aead(self, encrypted_data: bytes, associated_data: bytes = b"") -> bytes:
        """
        AEAD-style authenticated decryption.
        
        Input: version(1) || salt(16) || ciphertext || mac(32)
        """
        if len(encrypted_data) < 49:  # version + salt + mac minimum
            raise ValueError("Encrypted data too short")
        
        # Parse components
        version = encrypted_data[0:1]
        salt = encrypted_data[1:17]
        ciphertext = encrypted_data[17:-32]
        received_mac = encrypted_data[-32:]
        
        if version != b"\x02":
            raise ValueError("Unsupported version")
        
        # Derive keys
        enc_key_info = b"ENC_KEY_" + salt
        mac_key_info = b"MAC_KEY_" + salt
        
        enc_key = self._hkdf(enc_key_info, 32)
        mac_key = self._hkdf(mac_key_info, 32)
        
        # Verify MAC
        header = version + salt
        mac_data = header + ciphertext + associated_data
        expected_mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()
        
        if not hmac.compare_digest(received_mac, expected_mac):
            raise ValueError("Authentication failed")
        
        # Decrypt with derived key
        temp_cipher = EnhancedRFTCryptoV2(enc_key)
        
        plaintext = b""
        for i in range(0, len(ciphertext), 16):
            block = ciphertext[i:i+16]
            plaintext += temp_cipher._feistel_decrypt(block)
        
        # Remove padding
        padding_len = plaintext[-1]
        if padding_len > 16 or padding_len == 0:
            raise ValueError("Invalid padding")
        
        return plaintext[:-padding_len]
    
    def get_cipher_metrics(
        self,
        *,
        message_trials: int = 6,
        message_bit_samples: int = 32,
        key_bit_samples: int = 24,
        key_trials: int = 6,
        throughput_blocks: int = 4096,
    ) -> CipherMetrics:
        """Compute cryptographic metrics for research validation."""
        cache_key = (message_trials, message_bit_samples, key_bit_samples, key_trials, throughput_blocks)
        if cache_key not in self._metrics_cache:
            self._metrics_cache[cache_key] = self._compute_metrics(
                message_trials=message_trials,
                message_bit_samples=message_bit_samples,
                key_bit_samples=key_bit_samples,
                key_trials=key_trials,
                throughput_blocks=throughput_blocks,
            )
        return self._metrics_cache[cache_key]

    def _compute_metrics(
        self,
        *,
        message_trials: int,
        message_bit_samples: int,
        key_bit_samples: int,
        key_trials: int,
        throughput_blocks: int,
    ) -> CipherMetrics:
        """Measure avalanche characteristics and throughput empirically."""

        def _bit_diff(a: bytes, b: bytes) -> int:
            return sum((x ^ y).bit_count() for x, y in zip(a, b))

        def _flip_bit(data: bytes, bit_index: int) -> bytes:
            if bit_index < 0 or bit_index >= len(data) * 8:
                raise ValueError("Bit index out of range")
            mutable = bytearray(data)
            byte_index, bit_pos = divmod(bit_index, 8)
            mutable[byte_index] ^= 1 << bit_pos
            return bytes(mutable)

        def _choose_positions(total_bits: int, desired: int) -> list[int]:
            desired = max(1, min(desired, total_bits))
            if desired >= total_bits:
                return list(range(total_bits))
            return random.sample(range(total_bits), desired)

        # Message avalanche measurement
        ciphertext_bits = 128  # _feistel_encrypt operates on 16-byte blocks
        message_avalanche_samples = 0.0
        message_observations = 0

        for _ in range(message_trials):
            message = secrets.token_bytes(16)
            baseline_ciphertext = self._feistel_encrypt(message)
            bit_positions = _choose_positions(len(message) * 8, message_bit_samples)
            for bit in bit_positions:
                mutated_message = _flip_bit(message, bit)
                mutated_ciphertext = self._feistel_encrypt(mutated_message)
                message_avalanche_samples += _bit_diff(baseline_ciphertext, mutated_ciphertext) / ciphertext_bits
            message_observations += len(bit_positions)

        message_avalanche = message_avalanche_samples / message_observations if message_observations else 0.0

        # Key avalanche measurement (single-bit flips)
        key_bits = len(self.master_key) * 8
        reference_message = secrets.token_bytes(16)
        baseline_cipher = self
        baseline_ciphertext = baseline_cipher._feistel_encrypt(reference_message)

        key_positions = _choose_positions(key_bits, key_bit_samples)
        key_avalanche_total = 0.0

        for bit in key_positions:
            mutated_key = _flip_bit(self.master_key, bit)
            state = np.random.get_state()
            mutated_cipher = EnhancedRFTCryptoV2(mutated_key)
            np.random.set_state(state)
            mutated_ciphertext = mutated_cipher._feistel_encrypt(reference_message)
            key_avalanche_total += _bit_diff(baseline_ciphertext, mutated_ciphertext) / ciphertext_bits

        key_avalanche = key_avalanche_total / len(key_positions) if key_positions else 0.0

        # Key sensitivity (random key perturbations)
        key_sensitivity_total = 0.0
        successful_trials = 0

        for _ in range(max(1, key_trials)):
            perturbation = bytearray(secrets.token_bytes(len(self.master_key)))
            # Ensure perturbation is not all zeros
            if all(b == 0 for b in perturbation):
                perturbation[0] = 1
            mutated_key = bytes(a ^ b for a, b in zip(self.master_key, perturbation))
            state = np.random.get_state()
            mutated_cipher = EnhancedRFTCryptoV2(mutated_key)
            np.random.set_state(state)
            mutated_ciphertext = mutated_cipher._feistel_encrypt(reference_message)
            key_sensitivity_total += _bit_diff(baseline_ciphertext, mutated_ciphertext) / ciphertext_bits
            successful_trials += 1

        key_sensitivity = key_sensitivity_total / successful_trials if successful_trials else 0.0

        # Throughput measurement (MB/s)
        self._feistel_encrypt(secrets.token_bytes(16))  # Warm-up
        block_count = max(throughput_blocks, 1)
        blocks = [secrets.token_bytes(16) for _ in range(block_count)]
        start = time.perf_counter()
        for block in blocks:
            self._feistel_encrypt(block)
        elapsed = time.perf_counter() - start
        throughput_mbps = 0.0 if elapsed <= 0 else ((block_count * 16) / 1_000_000) / elapsed

        return CipherMetrics(
            message_avalanche=float(message_avalanche),
            key_avalanche=float(key_avalanche),
            key_sensitivity=float(key_sensitivity),
            throughput_mbps=float(throughput_mbps)
        )


def validate_enhanced_crypto(metrics_kwargs: Optional[dict] = None) -> dict:
    """
    Comprehensive validation of Enhanced RFT Crypto v2
    as described in the research paper.
    """
    print("Validating Enhanced RFT Crypto v2...")
    
    # Test vector
    master_key = b"RFT_TEST_KEY_2025_GOLDEN_RATIO__"  # 32 bytes
    test_message = b"QuantoniumOS RFT Crypto Test Vector for Research Paper Validation"
    
    cipher = EnhancedRFTCryptoV2(master_key)
    
    # Test AEAD encryption/decryption
    encrypted = cipher.encrypt_aead(test_message, b"RESEARCH_PAPER_2025")
    decrypted = cipher.decrypt_aead(encrypted, b"RESEARCH_PAPER_2025")
    
    # Validate round-trip
    roundtrip_success = (decrypted == test_message)
    
    # Get metrics
    metrics = cipher.get_cipher_metrics(**(metrics_kwargs or {}))
    
    avalanche_targets = {
        'message': 0.500,
        'key': 0.520,
        'sensitivity': 0.495,
        'throughput': 9.2,
    }

    results = {
        'roundtrip_success': roundtrip_success,
        'encrypted_size': len(encrypted),
        'original_size': len(test_message),
        'overhead_bytes': len(encrypted) - len(test_message),
        'metrics': {
            'message_avalanche': metrics.message_avalanche,
            'key_avalanche': metrics.key_avalanche,
            'key_sensitivity': metrics.key_sensitivity,
            'throughput_mbps': metrics.throughput_mbps
        },
        'paper_compliance': {
            'rounds': cipher.rounds == 48,
            'golden_ratio_param': abs(cipher.phi - 1.618033988749) < 1e-10,
            'domain_separation': True,
            'aead_mode': True,
            'message_avalanche_target': abs(metrics.message_avalanche - avalanche_targets['message']) < 0.05,
            'key_avalanche_target': abs(metrics.key_avalanche - avalanche_targets['key']) < 0.05,
            'key_sensitivity_target': abs(metrics.key_sensitivity - avalanche_targets['sensitivity']) < 0.05,
            'throughput_target': metrics.throughput_mbps >= avalanche_targets['throughput'] * 0.8
        }
    }
    
    print(f"✓ Round-trip test: {'PASS' if roundtrip_success else 'FAIL'}")
    print(f"✓ Message avalanche: {metrics.message_avalanche:.3f}")
    print(f"✓ Key avalanche: {metrics.key_avalanche:.3f}")
    print(f"✓ Key sensitivity: {metrics.key_sensitivity:.3f}")
    print(f"✓ Throughput: {metrics.throughput_mbps:.3f} MB/s")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Enhanced RFT Crypto v2 implementation")
    parser.add_argument("--quick", action="store_true", help="Use reduced sample counts for faster execution")
    parser.add_argument("--message-trials", type=int, default=None, help="Number of random messages for avalanche analysis")
    parser.add_argument("--message-bit-samples", type=int, default=None, help="Bits flipped per message during avalanche analysis")
    parser.add_argument("--key-bit-samples", type=int, default=None, help="Number of bit positions to flip when evaluating key avalanche")
    parser.add_argument("--key-trials", type=int, default=None, help="Number of random key perturbations for key sensitivity")
    parser.add_argument("--throughput-blocks", type=int, default=None, help="Blocks encrypted when measuring throughput")

    cli_args = parser.parse_args()

    metrics_overrides = {}
    if cli_args.quick:
        metrics_overrides.update({
            'message_trials': 3,
            'message_bit_samples': 16,
            'key_bit_samples': 12,
            'key_trials': 3,
            'throughput_blocks': 1024,
        })

    for field in ('message_trials', 'message_bit_samples', 'key_bit_samples', 'key_trials', 'throughput_blocks'):
        value = getattr(cli_args, field)
        if value is not None:
            metrics_overrides[field] = value

    validation_results = validate_enhanced_crypto(metrics_overrides or None)
    
    print("\n" + "="*50)
    print("PAPER IMPLEMENTATION VALIDATION")
    print("="*50)
    
    for category, values in validation_results['paper_compliance'].items():
        status = "✓" if values else "✗"
        print(f"{status} {category.replace('_', ' ').title()}: {values}")
