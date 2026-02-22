#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
RFT vs SOTA Compression Benchmark (JPEG XL, AVIF) - Experimental
===============================================================

Purpose:
  Provide a reproducible harness to compare experimental Φ-RFT-based codecs
  against established image codecs (JPEG XL, AVIF) on small public test sets.

Status:
  - This script is **[EXPERIMENTAL]**.
  - Only produces comparative metrics; NO claims are "[VERIFIED]" until
    integrated into automated CI tests.

Metrics Collected:
  - File size (bytes)
  - Compression ratio (original_size / compressed_size)
  - Decode reconstruction quality: PSNR, SSIM (RGB)
  - Encode + decode wall-clock times (seconds)

Dependencies:
  - External tools: `cjxl`, `djxl`, `avifenc`, `avifdec` (install via apt).
  - Python: numpy, pillow, scikit-image.

Datasets (default):
  - A small set of PNG test images placed under `tests/data/images/`.
    (User must supply; script will skip missing files.)

Output:
  - JSON summary: `benchmark_results_rft_vs_sota.json`
  - Markdown table: `benchmark_results_rft_vs_sota.md`

Usage:
  python tests/benchmarks/rft_sota_comparison.py --images tests/data/images \
      --rft-mode vertex --quality 75

Future Extensions:
  - Add hybrid codec integration
  - Add Llama.cpp quantization benchmarks (model size & perplexity)
  - Add perceptual metrics (LPIPS) if dependencies allowed
"""

import argparse
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    raise SystemExit("Missing Python deps. Install pillow, numpy, scikit-image.") from e

# Ensure repository root is on sys.path for local package imports
import sys
from pathlib import Path as _P

# Attempt to locate project root containing 'algorithms'
_here = _P(__file__).resolve()
_candidates = [_here.parents[i] for i in range(1, min(6, len(_here.parents)))]
_added = False
for cand in _candidates:
    if (cand / "algorithms" / "rft" / "core" / "resonant_fourier_transform.py").exists():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            _added = True
        break

if not _added:
    # Fallback: add immediate parent of tests/benchmarks (likely repo root)
    _fallback = _here.parents[2]
    if str(_fallback) not in sys.path:
        sys.path.insert(0, str(_fallback))

try:
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
except ModuleNotFoundError as e:
    raise SystemExit(
        "Import failure: cannot locate 'algorithms.rft.core.resonant_fourier_transform'. "
        "Ensure you run from repository root and that __init__.py files exist. "
        f"Checked candidates: {[str(c) for c in _candidates]}"
    ) from e


# --- Utility Functions -----------------------------------------------------

def run_cmd(cmd: List[str]) -> Tuple[str, str, int]:
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    elapsed = time.time() - start
    return out, err, proc.returncode, elapsed


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255.0 ** 2) / mse)


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # skimage expects (H,W,3); data already in that shape
    return ssim(a, b, channel_axis=2, data_range=255)


# --- RFT Codec Implementation (Real Φ-RFT Transform) ----------------------

def rft_encode(img_array: np.ndarray, mode: str, keep_fraction: float = 0.05) -> Dict[str, Any]:
    """RFT-based sparsifying encoder using real Φ-RFT transform.

    Steps per channel:
      1. Apply 2D forward Φ-RFT by composing separable row/col 1D transforms.
      2. Compute magnitude, keep top-K coefficients (K = keep_fraction * N).
      3. Store (indices, complex values) with complex64 quantization.

    This uses the canonical Φ-RFT implementation from
    algorithms.rft.core.resonant_fourier_transform (rft_forward/rft_inverse).
    """
    h, w, c = img_array.shape
    artifact = {"shape": (h, w, c), "channels": []}
    for ch in range(c):
        plane = img_array[..., ch].astype(np.float32)
        # Apply separable Φ-RFT: rows then cols
        tmp = np.vstack([rft_forward(row) for row in plane])  # row-wise
        freq = np.column_stack([rft_forward(col) for col in tmp.T])  # col-wise
        flat = freq.flatten()
        K = max(1, int(keep_fraction * flat.size))
        idx = np.argpartition(np.abs(flat), -K)[-K:]
        kept_vals = flat[idx].astype(np.complex64)
        artifact["channels"].append({"indices": idx.astype(np.int32), "values": kept_vals})
    artifact["mode"] = mode
    artifact["keep_fraction"] = keep_fraction
    return artifact


def rft_decode(artifact: Dict[str, Any]) -> np.ndarray:
    h, w, c = artifact["shape"]
    recon = np.zeros((h, w, c), dtype=np.float32)
    for ch, chdata in enumerate(artifact["channels"]):
        idx = chdata["indices"]
        vals = chdata["values"].astype(np.complex128)
        freq = np.zeros(h * w, dtype=np.complex128)
        freq[idx] = vals
        freq2d = freq.reshape(h, w)
        # Inverse separable Φ-RFT: cols then rows (inverse order of forward)
        tmp = np.column_stack([rft_inverse(col) for col in freq2d.T])
        plane = np.vstack([rft_inverse(row) for row in tmp])
        # Clip to byte range
        recon[..., ch] = np.clip(np.real(plane), 0, 255)
    return recon.astype(np.uint8)


# --- Benchmark Procedures --------------------------------------------------

def benchmark_image(path: Path, args) -> Dict[str, Any]:
    record: Dict[str, Any] = {"image": str(path.name)}
    orig = load_image(path)
    orig_size = path.stat().st_size
    record["original_size_bytes"] = orig_size

    # RFT (experimental real transform backed)
    t0 = time.time()
    rft_artifact = rft_encode(orig, args.rft_mode, keep_fraction=args.keep_fraction)
    
    # Serialize to get actual byte size
    # Simple format: 
    #   Header: Magic(4) + H(2) + W(2) + C(1) + KeepFrac(4)
    #   Per Channel: Count(4) + Indices(4*K) + Values(8*K for complex64)
    # This is still naive (no entropy coding, 32-bit indices) but better than python object guess
    
    serialized_data = bytearray(b'RFT1')
    h, w, c = rft_artifact["shape"]
    serialized_data.extend(struct.pack('>HHBf', h, w, c, args.keep_fraction))
    
    for ch_data in rft_artifact["channels"]:
        indices = ch_data["indices"].astype(np.uint32)
        values = ch_data["values"].astype(np.complex64)
        k = len(indices)
        serialized_data.extend(struct.pack('>I', k))
        serialized_data.extend(indices.tobytes())
        serialized_data.extend(values.tobytes())
        
    rft_size = len(serialized_data)
    enc_time = time.time() - t0

    t1 = time.time()
    rft_recon = rft_decode(rft_artifact)
    dec_time = time.time() - t1
    record.update({
        "rft_mode": args.rft_mode,
        "rft_keep_fraction": args.keep_fraction,
        "rft_size_bytes": rft_size,
        "rft_cr": round(orig_size / rft_size, 3) if rft_size else None,
        "rft_encode_s": round(enc_time, 4),
        "rft_decode_s": round(dec_time, 4),
        "rft_psnr": round(compute_psnr(orig, rft_recon), 3),
        "rft_ssim": round(compute_ssim(orig, rft_recon), 4),
    })

    # JPEG XL
    if shutil_which("cjxl") and shutil_which("djxl"):
        jxl_out = path.with_suffix(".jxl")
        out, err, rc, elapsed_enc = run_cmd(["cjxl", str(path), str(jxl_out), f"-q", str(args.quality)])
        if rc == 0 and jxl_out.exists():
            jxl_size = jxl_out.stat().st_size
            # Decode
            recon_path = path.with_suffix(".jxl.png")
            out2, err2, rc2, elapsed_dec = run_cmd(["djxl", str(jxl_out), str(recon_path)])
            if rc2 == 0 and recon_path.exists():
                jxl_img = load_image(recon_path)
                record.update({
                    "jxl_size_bytes": jxl_size,
                    "jxl_cr": round(orig_size / jxl_size, 3),
                    "jxl_encode_s": round(elapsed_enc, 4),
                    "jxl_decode_s": round(elapsed_dec, 4),
                    "jxl_psnr": round(compute_psnr(orig, jxl_img), 3),
                    "jxl_ssim": round(compute_ssim(orig, jxl_img), 4),
                })
            # cleanup
            if recon_path.exists():
                os.remove(recon_path)
            os.remove(jxl_out)

    # AVIF
    if shutil_which("avifenc") and shutil_which("avifdec"):
        avif_out = path.with_suffix(".avif")
        out, err, rc, elapsed_enc = run_cmd(["avifenc", str(path), str(avif_out), "--min", str(args.quality), "--max", str(args.quality)])
        if rc == 0 and avif_out.exists():
            avif_size = avif_out.stat().st_size
            recon_path = path.with_suffix(".avif.png")
            out2, err2, rc2, elapsed_dec = run_cmd(["avifdec", str(avif_out), str(recon_path)])
            if rc2 == 0 and recon_path.exists():
                avif_img = load_image(recon_path)
                record.update({
                    "avif_size_bytes": avif_size,
                    "avif_cr": round(orig_size / avif_size, 3),
                    "avif_encode_s": round(elapsed_enc, 4),
                    "avif_decode_s": round(elapsed_dec, 4),
                    "avif_psnr": round(compute_psnr(orig, avif_img), 3),
                    "avif_ssim": round(compute_ssim(orig, avif_img), 4),
                })
            if recon_path.exists():
                os.remove(recon_path)
            os.remove(avif_out)

    return record


def shutil_which(cmd: str) -> str:
    from shutil import which
    return which(cmd)


def write_outputs(results: List[Dict[str, Any]], args):
    # JSON
    json_path = Path("benchmark_results_rft_vs_sota.json")
    with json_path.open("w") as f:
        json.dump({"results": results, "quality": args.quality}, f, indent=2)

    # Markdown table (basic)
    md_path = Path("benchmark_results_rft_vs_sota.md")
    headers = ["image", "rft_keep_fraction", "rft_cr", "jxl_cr", "avif_cr", "rft_psnr", "jxl_psnr", "avif_psnr", "rft_ssim", "jxl_ssim", "avif_ssim"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in results:
        row = [str(r.get(h, "")) for h in headers]
        lines.append("|" + "|".join(row) + "|")
    with md_path.open("w") as f:
        f.write("# Experimental Compression Benchmark\n\n")
        f.write("Quality parameter: {}\n\n".format(args.quality))
        f.write("NOTE: RFT figures use the canonical Φ-RFT transform (resonant_fourier_transform.py). ")
        f.write("Results reflect real transform performance with top-K coefficient retention.\n\n")
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True, help="Directory of test images (PNG/JPG)")
    parser.add_argument("--rft-mode", type=str, default="vertex", choices=["vertex", "hybrid"], help="Experimental RFT codec mode")
    parser.add_argument("--quality", type=int, default=75, help="Quality setting for baseline codecs")
    parser.add_argument("--keep-fraction", type=float, default=0.05, help="Fraction of largest RFT coefficients to retain per channel")
    parser.add_argument("--iterations", type=int, default=1, help="Repeat each image benchmark N times for timing average")
    args = parser.parse_args()

    img_dir = Path(args.images)
    if not img_dir.exists():
        raise SystemExit(f"Image directory {img_dir} does not exist.")

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        print("No images found; nothing to benchmark.")
        return

    results = []
    for img in images:
        try:
            agg: Dict[str, Any] = {}
            reps = []
            for _ in range(args.iterations):
                rec = benchmark_image(img, args)
                reps.append(rec)
            # Average numeric fields across iterations
            rep0 = reps[0]
            for k, v in rep0.items():
                if isinstance(v, (int, float)) and all(isinstance(r.get(k), (int, float)) for r in reps):
                    agg[k] = round(sum(r[k] for r in reps) / len(reps), 4)
                else:
                    agg[k] = v
            agg["iterations"] = args.iterations
            results.append(agg)
        except Exception as e:
            print(f"Error processing {img}: {e}")

    write_outputs(results, args)


if __name__ == "__main__":
    main()
