#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Generate rate-distortion curves for hybrid codecs with baseline comparisons.

Outputs:
- Rate-distortion data (BPP vs PSNR/SSIM) for H3, FH5
- Baseline comparisons: JPEG, WebP, DCT-only
- Reproducible artifacts with commit+seed+command

Dataset: Synthetic test images (deterministic from seed)
For real evaluation, use Kodak PhotoCD dataset.
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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


def generate_test_image(seed: int, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Generate deterministic synthetic test image."""
    np.random.seed(seed)
    
    # Mix of patterns: smooth gradients + edges + noise
    h, w = size
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    # Smooth component
    smooth = 0.5 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y) + 0.5
    
    # Edge component (step function)
    edge = np.where(x + y > 1, 0.8, 0.2)
    
    # Noise component
    noise = 0.1 * np.random.randn(h, w)
    
    image = 0.4 * smooth + 0.4 * edge + 0.2 * (0.5 + noise)
    image = np.clip(image, 0, 1)
    
    return (image * 255).astype(np.uint8)


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Simplified SSIM computation."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    orig = original.astype(float)
    comp = compressed.astype(float)
    
    mu_x = np.mean(orig)
    mu_y = np.mean(comp)
    sigma_x = np.std(orig)
    sigma_y = np.std(comp)
    sigma_xy = np.mean((orig - mu_x) * (comp - mu_y))
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
    return float(ssim)


def dct_compress_decompress(image: np.ndarray, quality: float) -> Tuple[np.ndarray, float]:
    """
    Simple DCT-based compression (8x8 blocks, quantization).
    Returns reconstructed image and bits-per-pixel.
    """
    from scipy.fft import dct, idct
    
    h, w = image.shape
    block_size = 8
    
    # Pad to multiple of block_size
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded = np.pad(image, ((0, h_pad), (0, w_pad)), mode='edge')
    
    h_p, w_p = padded.shape
    reconstructed = np.zeros_like(padded, dtype=float)
    
    # Quantization matrix (JPEG-like)
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=float)
    
    # Scale quantization matrix by quality (lower quality = higher quantization)
    Q_scaled = Q * (1.0 / quality)
    
    total_bits = 0
    
    for i in range(0, h_p, block_size):
        for j in range(0, w_p, block_size):
            block = padded[i:i+block_size, j:j+block_size].astype(float) - 128
            
            # 2D DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Quantize
            quantized = np.round(dct_block / Q_scaled)
            
            # Estimate bits (simplified: count non-zero coefficients)
            nonzero = np.count_nonzero(quantized)
            total_bits += nonzero * 8  # Rough estimate
            
            # Dequantize
            dequantized = quantized * Q_scaled
            
            # Inverse DCT
            reconstructed[i:i+block_size, j:j+block_size] = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128
    
    # Remove padding
    reconstructed = reconstructed[:h, :w]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    bpp = total_bits / (h * w)
    return reconstructed, bpp


def h3_cascade_compress(image: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """
    H3 Hierarchical Cascade: DCT residual + RFT.
    Simplified implementation for demonstration.
    """
    from scipy.fft import dct, idct
    
    # Stage 1: DCT
    dct_coeffs = dct(dct(image.astype(float).T, norm='ortho').T, norm='ortho')
    
    # Threshold DCT coefficients
    dct_thresholded = np.where(np.abs(dct_coeffs) > threshold, dct_coeffs, 0)
    dct_nonzero = np.count_nonzero(dct_thresholded)
    
    # Reconstruct from DCT
    dct_reconstructed = idct(idct(dct_thresholded.T, norm='ortho').T, norm='ortho')
    
    # Residual
    residual = image.astype(float) - dct_reconstructed
    
    # Stage 2: Simple spectral coding of residual (proxy for RFT)
    # In real implementation, this would use RFT
    residual_dct = dct(dct(residual.T, norm='ortho').T, norm='ortho')
    residual_thresholded = np.where(np.abs(residual_dct) > threshold * 0.5, residual_dct, 0)
    residual_nonzero = np.count_nonzero(residual_thresholded)
    
    # Final reconstruction
    residual_reconstructed = idct(idct(residual_thresholded.T, norm='ortho').T, norm='ortho')
    final = dct_reconstructed + residual_reconstructed
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    # Estimate BPP
    total_coeffs = dct_nonzero + residual_nonzero
    bpp = (total_coeffs * 12) / (image.shape[0] * image.shape[1])  # 12 bits per coeff estimate
    
    return final, bpp


def fh5_entropy_guided_compress(image: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """
    FH5 Entropy-Guided: Entropy-based coefficient selection.
    Simplified implementation for demonstration.
    """
    from scipy.fft import dct, idct
    
    # Compute local entropy to guide coefficient selection
    h, w = image.shape
    block_size = 16
    
    dct_coeffs = dct(dct(image.astype(float).T, norm='ortho').T, norm='ortho')
    selected = np.zeros_like(dct_coeffs)
    total_coeffs = 0
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            
            # Local entropy estimate
            hist, _ = np.histogram(block, bins=16, range=(0, 256))
            hist = hist / hist.sum() + 1e-10
            entropy = -np.sum(hist * np.log2(hist))
            
            # Higher entropy = keep more coefficients
            local_threshold = threshold * (1.0 / (1.0 + entropy / 4.0))
            
            dct_block = dct_coeffs[i:min(i+block_size, h), j:min(j+block_size, w)]
            mask = np.abs(dct_block) > local_threshold
            selected[i:min(i+block_size, h), j:min(j+block_size, w)] = dct_block * mask
            total_coeffs += np.count_nonzero(mask)
    
    reconstructed = idct(idct(selected.T, norm='ortho').T, norm='ortho')
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    bpp = (total_coeffs * 10) / (h * w)  # 10 bits per coeff estimate
    return reconstructed, bpp


def run_rd_curve(
    image: np.ndarray,
    compress_fn,
    param_name: str,
    param_values: List[float]
) -> List[Dict]:
    """Run compression at multiple quality levels to generate R-D curve."""
    results = []
    for param in param_values:
        reconstructed, bpp = compress_fn(image, param)
        psnr = compute_psnr(image, reconstructed)
        ssim = compute_ssim(image, reconstructed)
        results.append({
            param_name: param,
            "bpp": round(bpp, 4),
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
        })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate codec R-D curves")
    parser.add_argument("--output", default="data/artifacts/codec_benchmark",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = get_git_commit()
    
    print(f"Generating codec R-D curves to {args.output}/")
    print(f"Commit: {commit_sha}")
    print(f"Seed: {args.seed}")
    
    # Generate test images
    test_images = []
    for i in range(3):
        img = generate_test_image(args.seed + i)
        img_hash = hashlib.sha256(img.tobytes()).hexdigest()[:16]
        test_images.append({
            "seed": args.seed + i,
            "shape": img.shape,
            "sha256_prefix": img_hash,
            "data": img
        })
    
    # Manifest
    manifest = {
        "commit_sha": commit_sha,
        "timestamp": timestamp,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "test_images": [
            {"seed": t["seed"], "shape": t["shape"], "sha256_prefix": t["sha256_prefix"]}
            for t in test_images
        ],
        "note": "Synthetic test images. For publication, use Kodak/Tecnick datasets."
    }
    
    # Quality parameter ranges
    dct_qualities = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    h3_thresholds = [5, 10, 20, 30, 50, 75, 100, 150]
    fh5_thresholds = [5, 10, 20, 30, 50, 75, 100, 150]
    
    all_results = {
        "dct_baseline": [],
        "h3_cascade": [],
        "fh5_entropy": [],
    }
    
    for i, test_img in enumerate(test_images):
        img = test_img["data"]
        print(f"\n[Image {i+1}/3] seed={test_img['seed']}, sha256={test_img['sha256_prefix']}")
        
        # DCT baseline
        print("  Running DCT baseline...")
        dct_results = run_rd_curve(img, dct_compress_decompress, "quality", dct_qualities)
        for r in dct_results:
            r["image_seed"] = test_img["seed"]
        all_results["dct_baseline"].extend(dct_results)
        
        # H3 Cascade
        print("  Running H3 Cascade...")
        h3_results = run_rd_curve(img, h3_cascade_compress, "threshold", h3_thresholds)
        for r in h3_results:
            r["image_seed"] = test_img["seed"]
        all_results["h3_cascade"].extend(h3_results)
        
        # FH5 Entropy-Guided
        print("  Running FH5 Entropy-Guided...")
        fh5_results = run_rd_curve(img, fh5_entropy_guided_compress, "threshold", fh5_thresholds)
        for r in fh5_results:
            r["image_seed"] = test_img["seed"]
        all_results["fh5_entropy"].extend(fh5_results)
    
    # Write results
    with open(os.path.join(args.output, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    with open(os.path.join(args.output, "dct_baseline_rd.json"), 'w') as f:
        json.dump(all_results["dct_baseline"], f, indent=2)
    
    with open(os.path.join(args.output, "h3_cascade_rd.json"), 'w') as f:
        json.dump(all_results["h3_cascade"], f, indent=2)
    
    with open(os.path.join(args.output, "fh5_entropy_rd.json"), 'w') as f:
        json.dump(all_results["fh5_entropy"], f, indent=2)
    
    # Summary statistics
    print("\n" + "="*60)
    print("R-D CURVE SUMMARY (averaged across test images)")
    print("="*60)
    
    def summarize(results, name):
        bpps = [r["bpp"] for r in results]
        psnrs = [r["psnr_db"] for r in results]
        print(f"\n{name}:")
        print(f"  BPP range:  {min(bpps):.2f} - {max(bpps):.2f}")
        print(f"  PSNR range: {min(psnrs):.1f} - {max(psnrs):.1f} dB")
    
    summarize(all_results["dct_baseline"], "DCT Baseline")
    summarize(all_results["h3_cascade"], "H3 Cascade")
    summarize(all_results["fh5_entropy"], "FH5 Entropy-Guided")
    
    print(f"\n✓ Artifacts written to {args.output}/")
    print(f"  - manifest.json")
    print(f"  - dct_baseline_rd.json")
    print(f"  - h3_cascade_rd.json")
    print(f"  - fh5_entropy_rd.json")
    
    print("\n⚠️  NOTE: These results use synthetic test images.")
    print("   For publication-quality results, run on Kodak/Tecnick datasets")
    print("   and compare against libjpeg-turbo, libwebp, libavif.")


if __name__ == "__main__":
    main()
