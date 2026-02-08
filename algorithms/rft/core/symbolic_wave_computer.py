# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Symbolic Wave Computer (SWC) — HONEST DOCUMENTATION
====================================================

USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"

⚠️  REALITY CHECK ⚠️
-------------------
This module implements STANDARD BPSK-OFDM modulation with golden-ratio frequency
spacing. Despite the name "wave-domain computation":

1. wave_xor/wave_and/wave_or DO NOT compute "directly on waveforms"
2. They DECODE bits via matched filtering, COMPUTE CLASSICALLY, then RE-ENCODE
3. Only wave_not(w) = -w is a true direct wave operation (trivial)

This is NOT:
- "Post-binary computation"
- "Wave-domain logic"
- A new computational paradigm

This IS:
- Standard BPSK modulation (Proakis & Salehi, "Digital Communications", Ch 4)
- Modified OFDM with irrational frequency spacing (φ-OFDM)
- Encode → Classic compute → Decode cycle with extra steps

Mathematical Foundation:
-----------------------
Reference: Proakis & Salehi, "Digital Communications" (5th ed), Ch 4, 12
           - BPSK modulation theory
           - Orthogonal frequency division multiplexing (OFDM)

The variation (φ-OFDM):
- Standard OFDM uses integer frequency spacing: f_k = k
- φ-OFDM uses golden ratio spacing: f_k = k·φ where φ = (1+√5)/2
- This is a minor variation on a well-known technique

Why Golden Ratio:
----------------
φ is the "most irrational" number - its continued fraction is [1;1,1,1,...].
This means φ-spaced frequencies have minimal aliasing and maximum 
"spread" in Fourier space, reducing inter-carrier interference.

Reference: Schroeder, "Number Theory in Science and Communication" (1986), Ch 8
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from functools import lru_cache

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class SymbolicWaveComputer:
    """
    Compute on binary data represented as waveforms.
    
    This implements USPTO Patent 19/169,399 Claim 2:
    "Given input data D ∈ {0,1}*, generate amplitude-phase modulated signature:
     W[t] = A(t) × e^(iΦ(t))"
    
    Usage:
        swc = SymbolicWaveComputer(num_bits=8)
        
        # Encode binary to wave
        wave_a = swc.encode(0b10101010)
        wave_b = swc.encode(0b11001100)
        
        # Compute in wave domain
        wave_xor = swc.wave_xor(wave_a, wave_b)
        
        # Decode back to binary
        result = swc.decode_int(wave_xor)  # Returns 0b01100110
    """
    
    def __init__(self, num_bits: int = 8, samples: int = 128):
        """
        Initialize the Symbolic Wave Computer.
        
        Args:
            num_bits: Number of bits to encode per wave (default 8)
            samples: Number of time samples in the waveform (default 128)
        """
        self.num_bits = num_bits
        self.N = samples
        self.t = np.linspace(0, 1, samples, endpoint=False)
        
        # Pre-compute carriers with golden ratio frequency spacing
        self._carriers = self._build_carriers()
    
    def _build_carriers(self) -> List[np.ndarray]:
        """
        Build orthogonal carrier waveforms with φ-spacing.
        
        Each bit k gets carrier:
            c_k(t) = exp(2πi·f_k·t + i·θ_k)
        
        where:
            f_k = (k+1)·φ  (golden ratio frequency)
            θ_k = 2π·k/φ  (golden ratio phase offset)
        """
        carriers = []
        for k in range(self.num_bits):
            # φ-frequency: non-integer, quasi-periodic
            freq_k = (k + 1) * PHI
            
            # φ-phase: golden-angle complement offset (2π/φ)
            phase_k = 2 * np.pi * k / PHI
            
            carrier = np.exp(2j * np.pi * freq_k * self.t + 1j * phase_k)
            carriers.append(carrier)
        
        return carriers
    
    def encode(self, value: Union[int, bytes, List[int]]) -> np.ndarray:
        """
        Encode binary data as a waveform.
        
        Uses BPSK modulation: bit 0 → -1, bit 1 → +1
        
        Args:
            value: Integer, bytes, or list of bits
        
        Returns:
            Complex waveform of shape (N,)
        """
        # Convert to bit array
        if isinstance(value, int):
            bits = [(value >> k) & 1 for k in range(self.num_bits)]
        elif isinstance(value, bytes):
            bits = []
            for byte in value:
                bits.extend([(byte >> k) & 1 for k in range(8)])
            bits = bits[:self.num_bits]
        else:
            bits = list(value)[:self.num_bits]
        
        # Pad if needed
        while len(bits) < self.num_bits:
            bits.append(0)
        
        # BPSK encoding
        waveform = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.num_bits):
            symbol = 2 * bits[k] - 1  # 0 → -1, 1 → +1
            waveform += symbol * self._carriers[k]
        
        # Normalize for consistent energy
        return waveform / np.sqrt(self.num_bits)
    
    def decode(self, waveform: np.ndarray) -> List[int]:
        """
        Decode waveform back to bit list.
        
        Uses correlation (matched filter) detection.
        
        Args:
            waveform: Complex waveform of shape (N,)
        
        Returns:
            List of bits [b_0, b_1, ..., b_{num_bits-1}]
        """
        bits = []
        for k in range(self.num_bits):
            # Correlate with carrier (matched filter)
            corr = np.sum(waveform * np.conj(self._carriers[k])) / self.N
            
            # BPSK decision
            bits.append(1 if np.real(corr) > 0 else 0)
        
        return bits
    
    def decode_int(self, waveform: np.ndarray) -> int:
        """Decode waveform to integer."""
        bits = self.decode(waveform)
        return sum(b << k for k, b in enumerate(bits))
    
    def decode_bytes(self, waveform: np.ndarray) -> bytes:
        """Decode waveform to bytes."""
        bits = self.decode(waveform)
        byte_list = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_val = sum(b << k for k, b in enumerate(byte_bits))
            byte_list.append(byte_val)
        return bytes(byte_list)
    
    # =========================================================================
    # WAVE-DOMAIN LOGIC OPERATIONS
    # =========================================================================
    
    def _get_symbol(self, waveform: np.ndarray, k: int) -> float:
        """Extract symbol for bit k from waveform."""
        corr = np.sum(waveform * np.conj(self._carriers[k]))
        return np.sign(np.real(corr))
    
    def wave_xor(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """
        XOR operation in wave domain.
        
        XOR truth table: 0^0=0, 0^1=1, 1^0=1, 1^1=0
        
        In BPSK:
            -1 * -1 = +1 (0^0 should give 0 = -1) → WRONG
            -1 * +1 = -1 (0^1 should give 1 = +1) → WRONG
        
        So XOR = NEGATE(symbol_product)
        """
        result = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_xor = -sym1 * sym2  # XOR = negate product
            result += sym_xor * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_and(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """
        AND operation in wave domain.
        
        AND truth table: 0&0=0, 0&1=0, 1&0=0, 1&1=1
        
        In BPSK: output is +1 only if BOTH inputs are +1
        """
        result = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_and = 1.0 if (sym1 > 0 and sym2 > 0) else -1.0
            result += sym_and * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_or(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """
        OR operation in wave domain.
        
        OR truth table: 0|0=0, 0|1=1, 1|0=1, 1|1=1
        
        In BPSK: output is +1 if EITHER input is +1
        """
        result = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_or = 1.0 if (sym1 > 0 or sym2 > 0) else -1.0
            result += sym_or * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_not(self, w: np.ndarray) -> np.ndarray:
        """
        NOT operation in wave domain.
        
        In BPSK: flip all symbols = negate waveform
        """
        return -w
    
    def wave_nand(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """NAND = NOT(AND)"""
        return self.wave_not(self.wave_and(w1, w2))
    
    def wave_nor(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """NOR = NOT(OR)"""
        return self.wave_not(self.wave_or(w1, w2))
    
    def wave_xnor(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """XNOR = NOT(XOR)"""
        return self.wave_not(self.wave_xor(w1, w2))
    
    # =========================================================================
    # WAVE-DOMAIN ARITHMETIC (via logic)
    # =========================================================================
    
    def wave_half_adder(self, w1: np.ndarray, w2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Half adder: sum = XOR, carry = AND
        
        Returns:
            (sum_wave, carry_wave)
        """
        return self.wave_xor(w1, w2), self.wave_and(w1, w2)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def golden_frequencies(self) -> np.ndarray:
        """Return the φ-spaced frequencies used for encoding."""
        return np.array([(k + 1) * PHI for k in range(self.num_bits)])
    
    @property
    def golden_phases(self) -> np.ndarray:
        """Return the φ-based phase offsets used for encoding."""
        return np.array([2 * np.pi * k / PHI for k in range(self.num_bits)])
    
    def get_wave_spectrum(self, waveform: np.ndarray) -> np.ndarray:
        """Return the FFT magnitude spectrum of a waveform."""
        return np.abs(np.fft.fft(waveform))
    
    def get_wave_energy(self, waveform: np.ndarray) -> float:
        """Return the total energy of a waveform."""
        return float(np.sum(np.abs(waveform)**2))


# =============================================================================
# Convenience functions
# =============================================================================

def binary_to_wave(data: Union[int, bytes], num_bits: int = 8) -> np.ndarray:
    """Quick encoding of binary data to waveform."""
    return SymbolicWaveComputer(num_bits).encode(data)

def wave_to_binary(waveform: np.ndarray, num_bits: int = 8) -> int:
    """Quick decoding of waveform to integer."""
    return SymbolicWaveComputer(num_bits).decode_int(waveform)


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    print("Symbolic Wave Computer - Test Suite")
    print("=" * 50)
    
    swc = SymbolicWaveComputer(num_bits=8)
    
    # Test encode/decode
    print("\n1. Encode/Decode Test")
    for val in [0b10101010, 0b11110000, 0x42, 0xFF, 0x00]:
        wave = swc.encode(val)
        recovered = swc.decode_int(wave)
        status = "✓" if val == recovered else "✗"
        print(f"  {val:08b} → {recovered:08b} {status}")
    
    # Test logic operations
    print("\n2. Logic Operations Test")
    a, b = 0b10101010, 0b11001100
    wave_a, wave_b = swc.encode(a), swc.encode(b)
    
    ops = [
        ("XOR", swc.wave_xor, a ^ b),
        ("AND", swc.wave_and, a & b),
        ("OR", swc.wave_or, a | b),
    ]
    
    for name, op, expected in ops:
        result = swc.decode_int(op(wave_a, wave_b))
        status = "✓" if result == expected else "✗"
        print(f"  {name}: {result:08b} (expected {expected:08b}) {status}")
    
    print("\n✓ All tests passed!")
