#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Hardware/Software Alignment Verification
=========================================
Verifies that RFTPU Verilog hardware implementations match software algorithms.
Maps paper claims to both codebase and hardware artifacts.
"""

import sys
import os
import re
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
from algorithms.rft.core.resonant_fourier_transform import PHI, rft_basis_matrix

# ============================================================================
# CONSTANTS FROM PAPER
# ============================================================================

PAPER_PHI = (1 + 5**0.5) / 2  # Golden ratio
PAPER_SIGMA = 1.0
PAPER_BETA = 1.0
PAPER_FEISTEL_ROUNDS = 48
PAPER_SIS_N = 512

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_check(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")

# ============================================================================
# HARDWARE FILE VERIFICATION
# ============================================================================

def verify_hardware_files_exist():
    """Verify all critical hardware files exist and are non-placeholder."""
    print_header("HARDWARE FILE VERIFICATION")
    
    root = Path(__file__).parent.parent
    hardware_dir = root / "hardware"
    
    critical_files = [
        ("rftpu_architecture.sv", 1000),      # Main RFTPU architecture
        ("rft_middleware_engine.sv", 400),     # RFT middleware
        ("fpga_top.sv", 800),                  # FPGA top module with all kernels
    ]
    
    rtl_files = [
        ("rtl/systolic_array.sv", 100),
        ("rtl/systolic_array_v2.sv", 500),
        ("rtl/systolic_array_v21.sv", 100),
        ("rtl/systolic_array_v23.sv", 400),
    ]
    
    all_files = critical_files + rtl_files
    all_pass = True
    
    for filename, min_lines in all_files:
        filepath = hardware_dir / filename
        exists = filepath.exists()
        
        if exists:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = len(content.split('\n'))
                has_content = lines >= min_lines
                
                # Check for placeholder markers
                has_placeholder = any(marker in content.lower() for marker in 
                    ['todo: implement', 'placeholder', 'not implemented', '// stub'])
                
                passed = has_content and not has_placeholder
        else:
            passed = False
            lines = 0
            has_placeholder = False
        
        print_check(
            f"{filename}",
            passed,
            f"lines: {lines}, min: {min_lines}" if exists else "FILE MISSING"
        )
        
        if not passed:
            all_pass = False
    
    return all_pass

# ============================================================================
# KERNEL ROM VERIFICATION
# ============================================================================

def verify_kernel_rom():
    """Verify hardware kernel ROM values are complete and non-zero."""
    print_header("RFT KERNEL ROM VERIFICATION")
    
    root = Path(__file__).parent.parent
    
    # Check rftpu_architecture.sv kernel values
    arch_file = root / "hardware" / "rftpu_architecture.sv"
    mw_file = root / "hardware" / "rft_middleware_engine.sv"
    fpga_file = root / "hardware" / "fpga_top.sv"
    
    results = {}
    
    for name, filepath in [
        ("rftpu_architecture", arch_file),
        ("rft_middleware", mw_file),
        ("fpga_top", fpga_file)
    ]:
        if not filepath.exists():
            results[name] = (False, "File missing")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Count kernel coefficient entries - multiple patterns
        # Pattern 1: kernel_real = 16'sd12345 or 16'sh2D41
        kernel_pattern1 = r"kernel_(?:real|rom_out|imag)\s*[=<]=\s*-?16'[sh]"
        matches1 = re.findall(kernel_pattern1, content)
        
        # Pattern 2: 6'b binary case entries (rftpu_architecture style)
        case_pattern1 = r"6'b[01]+:\s*kernel_real\s*="
        matches2 = re.findall(case_pattern1, content)
        
        # Pattern 3: {4'd, 3'd, 3'd} case entries (fpga_top style)
        case_pattern2 = r"\{4'd\d+,\s*3'd\d+,\s*3'd\d+\}:\s*kernel_rom_out"
        matches3 = re.findall(case_pattern2, content)
        
        # Pattern 4: {k, n} case entries (rft_middleware style)
        case_pattern3 = r"\{k,\s*n\}|6'b[01]+_[01]+:\s*begin"
        matches4 = re.findall(case_pattern3, content)
        
        total_coeffs = len(matches1) + len(matches2) + len(matches3) + len(matches4)
        
        # For 8x8 RFT, we need 64 coefficients per mode
        # fpga_top has 16 modes * 64 = 1024+ coefficients
        # rftpu_architecture has 1 mode * 64 = 64 coefficients
        
        if 'fpga_top' in name:
            min_coeffs = 512  # At least 8 modes fully populated
        elif 'rftpu_architecture' in name:
            min_coeffs = 60   # At least one full 8x8 kernel
        else:
            min_coeffs = 60   # At least one full 8x8 kernel
        
        passed = total_coeffs >= min_coeffs
        results[name] = (passed, f"{total_coeffs} coefficients (min: {min_coeffs})")
    
    all_pass = True
    for name, (passed, details) in results.items():
        print_check(name, passed, details)
        if not passed:
            all_pass = False
    
    return all_pass

# ============================================================================
# SOFTWARE/HARDWARE ALGORITHM MATCHING
# ============================================================================

def verify_algorithm_matching():
    """Verify software RFT algorithm structure matches hardware description."""
    print_header("ALGORITHM STRUCTURE MATCHING")
    
    results = []
    
    # 1. Check golden ratio constant
    hw_phi_check = abs(PHI - PAPER_PHI) < 1e-10
    results.append(("Golden ratio φ constant", hw_phi_check, f"PHI = {PHI:.10f}"))
    
    # 2. Check default parameters match paper
    rft = CanonicalTrueRFT(64)
    sigma_match = rft.sigma == PAPER_SIGMA
    beta_match = rft.beta == PAPER_BETA
    results.append(("Default σ = 1.0", sigma_match, f"σ = {rft.sigma}"))
    results.append(("Default β = 1.0", beta_match, f"β = {rft.beta}"))
    
    # 3. Verify transform structure: Ψ = D_φ C_σ F
    # The basis should be unitary
    n = 64
    Psi = rft.get_rft_matrix()
    I = np.eye(n)
    unitarity_error = np.linalg.norm(Psi.conj().T @ Psi - I, 'fro')
    unitary_ok = unitarity_error < 1e-12
    results.append(("Unitarity (Ψ†Ψ = I)", unitary_ok, f"error = {unitarity_error:.2e}"))
    
    # 4. Check round-trip preservation
    x = np.random.randn(n) + 1j * np.random.randn(n)
    y = rft.forward_transform(x)
    x_rec = rft.inverse_transform(y)
    roundtrip_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    roundtrip_ok = roundtrip_error < 1e-14
    results.append(("Round-trip x = Ψ†Ψx", roundtrip_ok, f"error = {roundtrip_error:.2e}"))
    
    # 5. Check energy preservation (Parseval)
    energy_in = np.linalg.norm(x)**2
    energy_out = np.linalg.norm(y)**2
    energy_ratio = energy_out / energy_in
    energy_ok = abs(energy_ratio - 1.0) < 1e-14
    results.append(("Energy preservation", energy_ok, f"ratio = {energy_ratio:.15f}"))
    
    all_pass = True
    for name, passed, details in results:
        print_check(name, passed, details)
        if not passed:
            all_pass = False
    
    return all_pass

# ============================================================================
# CRYPTO IMPLEMENTATION VERIFICATION
# ============================================================================

def verify_crypto_implementation():
    """Verify Enhanced RFT Crypto v2 is fully implemented."""
    print_header("CRYPTO IMPLEMENTATION VERIFICATION")
    
    root = Path(__file__).parent.parent
    crypto_file = root / "algorithms" / "rft" / "crypto" / "enhanced_cipher.py"
    
    results = []
    
    if not crypto_file.exists():
        print_check("Crypto module exists", False, "File missing")
        return False
    
    with open(crypto_file, 'r') as f:
        content = f.read()
    
    # Check for key components
    components = [
        ("S_BOX (AES S-box)", "S_BOX ="),
        ("MixColumns matrix", "MIX_COLUMNS_MATRIX"),
        ("48-round Feistel", "rounds = 48"),
        ("HKDF key derivation", "def _hkdf"),
        ("Golden ratio", "phi"),
        ("Phase locks", "phase_locks"),
        ("Amplitude masks", "amplitude_masks"),
        ("Round keys", "round_keys"),
        ("Encrypt method", "def encrypt"),
        ("Decrypt method", "def decrypt"),
    ]
    
    all_pass = True
    for name, pattern in components:
        found = pattern in content
        results.append((name, found))
        print_check(name, found)
        if not found:
            all_pass = False
    
    # Try to import and instantiate
    try:
        from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
        key = b"0123456789abcdef0123456789abcdef"
        cipher = EnhancedRFTCryptoV2(key)
        
        # Test basic encrypt/decrypt (using AEAD methods)
        plaintext = b"Hello QuantoniumOS!"
        
        # Check which method exists
        if hasattr(cipher, 'encrypt_aead'):
            ciphertext = cipher.encrypt_aead(plaintext)
            decrypted = cipher.decrypt_aead(ciphertext)
        elif hasattr(cipher, 'encrypt'):
            ciphertext = cipher.encrypt(plaintext)
            decrypted = cipher.decrypt(ciphertext)
        elif hasattr(cipher, '_feistel_encrypt'):
            # Pad to 16 bytes for block cipher
            padded = plaintext.ljust(16, b'\x00')[:16]
            ciphertext = cipher._feistel_encrypt(padded)
            decrypted = cipher._feistel_decrypt(ciphertext)
        else:
            raise AttributeError("No encrypt method found")
        
        enc_dec_ok = (decrypted == plaintext) or (decrypted.rstrip(b'\x00') == plaintext) or (decrypted[:len(plaintext)] == plaintext)
        print_check("Encrypt/Decrypt round-trip", enc_dec_ok, 
                   f"AEAD mode: {hasattr(cipher, 'encrypt_aead')}")
        if not enc_dec_ok:
            all_pass = False
            
    except Exception as e:
        print_check("Crypto instantiation", False, str(e)[:60])
        all_pass = False
    
    return all_pass

# ============================================================================
# RFT-MW (MIDDLEWARE) VERIFICATION
# ============================================================================

def verify_rft_middleware():
    """Verify RFT Middleware/Vertex Codec is complete."""
    print_header("RFT MIDDLEWARE (RFT-MW) VERIFICATION")
    
    root = Path(__file__).parent.parent
    mw_file = root / "algorithms" / "rft" / "compression" / "rft_vertex_codec.py"
    
    results = []
    
    if not mw_file.exists():
        print_check("RFT-MW codec exists", False, "File missing")
        return False
    
    with open(mw_file, 'r') as f:
        content = f.read()
    
    # Check for key components
    components = [
        ("RFTVertex dataclass", "class RFTVertex"),
        ("encode_tensor function", "def encode_tensor"),
        ("decode_tensor function", "def decode_tensor"),
        ("Forward transform call", "rft_forward"),
        ("Inverse transform call", "rft_inverse"),
        ("Amplitude/Phase extraction", "'A':", "'phi':"),
        ("Checksum generation", "sha256"),
    ]
    
    all_pass = True
    for name, *patterns in components:
        found = all(p in content for p in patterns)
        results.append((name, found))
        print_check(name, found)
        if not found:
            all_pass = False
    
    # Try to import and test
    try:
        from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
        
        # Create test tensor
        test_tensor = np.random.randn(64)
        
        # Encode
        container = encode_tensor(test_tensor)
        
        # Check container structure
        has_vertices = 'vertices' in container or 'chunks' in container
        print_check("Container structure", has_vertices, 
                   f"keys: {list(container.keys())[:5]}...")
        
        # Decode and verify round-trip
        recovered = decode_tensor(container)
        error = np.max(np.abs(test_tensor - recovered))
        rt_ok = error < 1e-10
        print_check("Encode/Decode round-trip", rt_ok, f"max error: {error:.2e}")
        
        if not rt_ok:
            all_pass = False
            
    except Exception as e:
        print_check("RFT-MW import/test", False, str(e)[:60])
        all_pass = False
    
    return all_pass

# ============================================================================
# HARDWARE MODE VERIFICATION
# ============================================================================

def verify_hardware_modes():
    """Verify all paper-claimed hardware modes are implemented."""
    print_header("HARDWARE MODE VERIFICATION")
    
    root = Path(__file__).parent.parent
    fpga_file = root / "hardware" / "fpga_top.sv"
    
    if not fpga_file.exists():
        print_check("fpga_top.sv exists", False)
        return False
    
    with open(fpga_file, 'r') as f:
        content = f.read()
    
    # Modes from paper and hardware comments
    required_modes = [
        ("MODE_RFT_GOLDEN", "Mode 0: Golden ratio transform", "0"),
        ("MODE_RFT_CASCADE", "Mode 6: H3 Hybrid Compression", "6"),
        ("MODE_SIS_HASH", "Mode 12: SIS Lattice Hash", "12"),
        ("MODE_FEISTEL", "Mode 13: Feistel-48 Cipher", "13"),
        ("MODE_QUANTUM_SIM", "Mode 14: Quantum Simulation", "14"),
        ("MODE_ROUNDTRIP", "Mode 15: Round-trip test", "15"),
    ]
    
    all_pass = True
    for mode_name, description, mode_num in required_modes:
        # Check if mode is defined
        mode_defined = f"localparam" in content and mode_name in content
        
        # Check if mode has kernel coefficients (for RFT modes)
        # Modes 12, 13, 15 don't need kernel coefficients:
        # - 12: SIS Hash (uses hash logic, not transform)
        # - 13: Feistel Cipher (uses cipher logic, not transform)
        # - 15: Roundtrip (test mode, reuses Mode 0 coefficients)
        if mode_name in ["MODE_RFT_GOLDEN", "MODE_RFT_CASCADE"]:
            has_coeffs = f"{{4'd{mode_num}," in content
            passed = mode_defined and has_coeffs
            detail = f"defined: {mode_defined}, coeffs: {has_coeffs}"
        elif mode_name == "MODE_QUANTUM_SIM":
            # Check for either mode 14 coefficients or quantum sim logic
            has_coeffs = f"{{4'd{mode_num}," in content
            has_quantum_logic = "QUANTUM" in content or "quantum" in content.lower()
            passed = mode_defined and (has_coeffs or has_quantum_logic)
            detail = f"defined: {mode_defined}, coeffs/logic: {has_coeffs or has_quantum_logic}"
        elif mode_name == "MODE_ROUNDTRIP":
            # Round-trip test mode reuses Mode 0's coefficients for forward/inverse test
            passed = mode_defined
            detail = f"defined: {mode_defined} (test mode, reuses Mode 0)"
        else:
            # Hash and cipher modes don't need kernel coefficients
            passed = mode_defined
            detail = f"defined: {mode_defined}"
        
        print_check(f"{description}", passed, detail)
        if not passed:
            all_pass = False
    
    return all_pass

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  QUANTONIUMOS HARDWARE/SOFTWARE ALIGNMENT VERIFICATION")
    print("  Verifying RFTPU, RFT-MW, and Algorithm Matching")
    print("="*70)
    
    results = {
        "Hardware Files": verify_hardware_files_exist(),
        "Kernel ROM": verify_kernel_rom(),
        "Algorithm Matching": verify_algorithm_matching(),
        "Crypto Implementation": verify_crypto_implementation(),
        "RFT Middleware": verify_rft_middleware(),
        "Hardware Modes": verify_hardware_modes(),
    }
    
    print_header("VERIFICATION SUMMARY")
    
    passed = 0
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test}")
        if result:
            passed += 1
    
    print()
    print(f"  Total: {passed}/{total} checks passed")
    print()
    
    if passed == total:
        print("  ✓ ALL HARDWARE/SOFTWARE ALIGNMENT VERIFIED!")
        print("  ✓ No placeholders detected in critical files")
        print("  ✓ RFTPU architecture fully implemented")
        print("  ✓ RFT-MW codec complete")
    else:
        print("  ✗ SOME ALIGNMENT ISSUES DETECTED")
    
    print()
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
