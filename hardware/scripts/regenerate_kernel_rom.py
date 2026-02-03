#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Regenerate Kernel ROM Coefficients from CANONICAL RFT
=====================================================

This script generates Q1.15 fixed-point kernel coefficients from the
CANONICAL Gram-normalized RFT basis for use in hardware RTL.

CANONICAL RFT DEFINITION (January 2026):
    Φ̃ = Φ (ΦᴴΦ)^{-1/2}
where:
    Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N

Usage:
    python regenerate_kernel_rom.py --output fpga_top_webfpga.v
    python regenerate_kernel_rom.py --format verilog > kernel_rom.vh
    python regenerate_kernel_rom.py --format json > kernel_rom.json
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Use CANONICAL Gram-normalized RFT
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix, PHI

# Hardware parameters
N = 8  # Transform size for hardware
Q15_SCALE = 32768  # Q1.15 scale factor


def float_to_q15(value: float) -> int:
    """Convert float to Q1.15 fixed-point (signed 16-bit)."""
    scaled = int(round(value * Q15_SCALE))
    return max(-32768, min(32767, scaled))


def generate_canonical_kernel(n: int = N) -> np.ndarray:
    """
    Generate CANONICAL RFT kernel matrix.
    
    Returns Gram-normalized φ-grid exponential basis Φ̃.
    """
    return rft_basis_matrix(n, n, use_gram_normalization=True)


def kernel_to_q15(kernel: np.ndarray) -> np.ndarray:
    """Convert float kernel to Q1.15 fixed-point."""
    q15_kernel = np.zeros_like(kernel, dtype=np.int16)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            # Use real part for hardware (complex kernels need separate handling)
            q15_kernel[i, j] = float_to_q15(kernel[i, j].real)
    return q15_kernel


def verify_unitarity(kernel: np.ndarray, tolerance: float = 1e-10) -> tuple:
    """Verify that kernel is unitary."""
    n = kernel.shape[0]
    identity = np.eye(n)
    product = kernel.conj().T @ kernel
    error = np.linalg.norm(product - identity, 'fro')
    return error, error < tolerance


def generate_verilog_rom(kernel: np.ndarray, mode: int = 0, mode_name: str = "RFT-GOLDEN") -> str:
    """Generate Verilog case statements for kernel ROM."""
    n = kernel.shape[0]
    q15_kernel = kernel_to_q15(kernel)
    
    lines = [
        f"            // MODE {mode}: {mode_name} (canonical Gram-normalized basis)",
        f"            // Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"            // Unitarity error: {verify_unitarity(kernel)[0]:.2e}",
    ]
    
    for k in range(n):
        for j in range(n):
            val = q15_kernel[k, j]
            sign = "-" if val < 0 else ""
            abs_val = abs(val)
            lines.append(f"            {{2'd{mode}, 3'd{k}, 3'd{j}}}: kernel_rom_out = {sign}16'sd{abs_val};")
    
    return "\n".join(lines)


def generate_json_rom(kernel: np.ndarray, mode_name: str = "RFT-GOLDEN") -> dict:
    """Generate JSON representation of kernel."""
    n = kernel.shape[0]
    q15_kernel = kernel_to_q15(kernel)
    error, is_unitary = verify_unitarity(kernel)
    
    return {
        "mode_name": mode_name,
        "size": n,
        "unitarity_error": float(error),
        "is_unitary": is_unitary,
        "generated": datetime.now().isoformat(),
        "coefficients": q15_kernel.tolist(),
        "float_coefficients": kernel.real.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate Kernel ROM from Canonical RFT")
    parser.add_argument("--format", choices=["verilog", "json", "both"], default="both",
                        help="Output format")
    parser.add_argument("--output", type=str, help="Output file (stdout if not specified)")
    parser.add_argument("--size", type=int, default=N, help="Transform size")
    parser.add_argument("--verify", action="store_true", help="Verify unitarity only")
    args = parser.parse_args()
    
    print(f"Generating CANONICAL RFT kernel (N={args.size})...", file=sys.stderr)
    
    # Generate canonical kernel
    kernel = generate_canonical_kernel(args.size)
    
    # Verify unitarity
    error, is_unitary = verify_unitarity(kernel)
    print(f"Unitarity error: {error:.2e} {'✓' if is_unitary else '✗'}", file=sys.stderr)
    
    if args.verify:
        print(f"\nCanonical RFT Kernel Verification:")
        print(f"  Size: {args.size}x{args.size}")
        print(f"  Unitarity error: {error:.2e}")
        print(f"  Is unitary: {is_unitary}")
        print(f"\nFirst row (Q1.15):")
        q15_kernel = kernel_to_q15(kernel)
        print(f"  {list(q15_kernel[0, :])}")
        return
    
    output_lines = []
    
    if args.format in ["verilog", "both"]:
        output_lines.append("// CANONICAL RFT Kernel ROM (Gram-normalized φ-grid basis)")
        output_lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"// Definition: Φ̃ = Φ (ΦᴴΦ)^{{-1/2}} where Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N")
        output_lines.append(f"// Unitarity error: {error:.2e}")
        output_lines.append("")
        output_lines.append(generate_verilog_rom(kernel, mode=0, mode_name="RFT-GOLDEN"))
    
    if args.format in ["json", "both"]:
        if args.format == "both":
            output_lines.append("\n// JSON representation:")
            output_lines.append("/*")
        json_data = generate_json_rom(kernel, mode_name="RFT-GOLDEN")
        output_lines.append(json.dumps(json_data, indent=2))
        if args.format == "both":
            output_lines.append("*/")
    
    output = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
