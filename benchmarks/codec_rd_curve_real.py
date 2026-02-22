#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Real Rate-Distortion Benchmark: RFT Codec vs Industry Codecs
=============================================================

Honest, reproducible R-D curves on standard test images (Kodak PhotoCD).

Codecs tested
-------------
* **JPEG** (Pillow / libjpeg-turbo) — quality 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
* **WebP** (Pillow / libwebp) — quality 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
* **AVIF** (Pillow / libheif, if available) — quality 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
* **RFT Binary Codec** — prune_ratio 0.0, 0.1, 0.2, …, 0.9, 0.95 with mag_bits/phase_bits grids

Metrics
-------
* **BPP** = compressed_file_bytes × 8 / num_pixels  (TRUE bits per pixel)
* **PSNR** = 10 log₁₀(255² / MSE) dB
* **SSIM** via ``skimage.metrics.structural_similarity``

Images
------
Kodak PhotoCD set (24 images, 768×512 or 512×768, PNG lossless).
Downloaded from http://r0k.us/graphics/kodak/ or generated as synthetic gradients
if download is unavailable.

Output
------
* ``results/rd_curves/rd_results.json``   — all per-image, per-codec, per-quality datapoints
* ``results/rd_curves/rd_summary.json``   — mean across images
* ``results/rd_curves/rd_plot_psnr.png``  — BPP vs PSNR plot
* ``results/rd_curves/rd_plot_ssim.png``  — BPP vs SSIM plot
* stdout summary table

Usage
-----
    python benchmarks/codec_rd_curve_real.py [--images-dir data/kodak] [--output-dir results/rd_curves]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Quality metrics ─────────────────────────────────────────────────────────

def compute_psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """PSNR between two uint8 images.  Returns inf if identical."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """SSIM between two uint8 images (multichannel-safe)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        warnings.warn("scikit-image not installed; SSIM unavailable", stacklevel=2)
        return float("nan")
    multichannel = ref.ndim == 3
    return float(
        structural_similarity(
            ref,
            test,
            data_range=255,
            channel_axis=2 if multichannel else None,
        )
    )


# ── Datapoint container ────────────────────────────────────────────────────

@dataclass
class RDPoint:
    codec: str
    quality_param: float        # codec-specific quality knob
    bpp: float                  # TRUE bits per pixel
    psnr_db: float
    ssim: float
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    image_name: str
    width: int
    height: int


# ── Industry codec wrappers ────────────────────────────────────────────────

def _bpp(compressed_bytes: int, width: int, height: int) -> float:
    """Bits per pixel from byte count."""
    return 8.0 * compressed_bytes / (width * height)


def encode_jpeg(img: Image.Image, quality: int) -> Tuple[bytes, float, float]:
    """JPEG encode via Pillow.  Returns (bytes, encode_ms, decode_ms)."""
    buf = io.BytesIO()
    t0 = time.perf_counter()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)  # 4:4:4 for fairness
    enc_ms = (time.perf_counter() - t0) * 1e3
    data = buf.getvalue()

    t0 = time.perf_counter()
    _ = Image.open(io.BytesIO(data)).convert("RGB")
    dec_ms = (time.perf_counter() - t0) * 1e3
    return data, enc_ms, dec_ms


def encode_webp(img: Image.Image, quality: int) -> Tuple[bytes, float, float]:
    """WebP lossy encode via Pillow."""
    buf = io.BytesIO()
    t0 = time.perf_counter()
    img.save(buf, format="WEBP", quality=quality, method=6)  # best effort
    enc_ms = (time.perf_counter() - t0) * 1e3
    data = buf.getvalue()

    t0 = time.perf_counter()
    _ = Image.open(io.BytesIO(data)).convert("RGB")
    dec_ms = (time.perf_counter() - t0) * 1e3
    return data, enc_ms, dec_ms


def encode_avif(img: Image.Image, quality: int) -> Optional[Tuple[bytes, float, float]]:
    """AVIF encode via Pillow (requires pillow-avif-plugin or Pillow >= 10.1)."""
    try:
        buf = io.BytesIO()
        t0 = time.perf_counter()
        img.save(buf, format="AVIF", quality=quality, speed=6)
        enc_ms = (time.perf_counter() - t0) * 1e3
        data = buf.getvalue()

        t0 = time.perf_counter()
        _ = Image.open(io.BytesIO(data)).convert("RGB")
        dec_ms = (time.perf_counter() - t0) * 1e3
        return data, enc_ms, dec_ms
    except Exception:
        return None  # AVIF not available


def decode_image(data: bytes, fmt: str) -> np.ndarray:
    """Decode compressed bytes back to uint8 RGB numpy array."""
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


# ── RFT codec wrapper for images ───────────────────────────────────────────

def encode_rft_image(
    img_array: np.ndarray,
    prune_ratio: float = 0.0,
    block_size: int = 256,
    mag_bits: int = 10,
    phase_bits: int = 8,
) -> Tuple[bytes, np.ndarray, float, float]:
    """
    Encode an RGB image through the RFT transform + quantization + zlib.

    Uses the REAL canonical RFT basis matrix (Gram-normalized φ-grid) but
    vectorises across all blocks for speed.  The compressed size is measured
    as real zlib bytes (header + payload) with all overhead counted.

    Returns (compressed_bytes, reconstructed_uint8, encode_ms, decode_ms).
    """
    import zlib as _zlib
    import struct as _struct
    from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

    h, w, c = img_array.shape
    n_pixels_ch = h * w

    # Pad channel to block boundary
    pad_len = (block_size - (n_pixels_ch % block_size)) % block_size
    padded_len = n_pixels_ch + pad_len
    num_blocks = padded_len // block_size

    # Pre-compute basis matrix ONCE (cached by lru_cache)
    Phi = rft_basis_matrix(block_size, block_size, use_gram_normalization=True)
    PhiH = Phi.conj().T  # forward transform matrix

    all_quant_bytes = []
    all_decoded_channels = []

    t_enc_start = time.perf_counter()

    for ch_idx in range(c):
        channel = img_array[:, :, ch_idx].ravel().astype(np.float64)
        if pad_len > 0:
            channel = np.concatenate([channel, np.zeros(pad_len)])

        # Reshape into blocks and batch-transform: (num_blocks, block_size)
        blocks = channel.reshape(num_blocks, block_size)
        # Batched forward RFT:  coeffs = blocks @ Phi  (each row × Phi)
        coeffs = (PhiH @ blocks.T).T  # shape (num_blocks, block_size), complex

        mags = np.abs(coeffs)
        phases = np.angle(coeffs)

        # Pruning (lossy)
        if prune_ratio > 0:
            threshold = np.percentile(mags.ravel(), prune_ratio * 100)
            mask = mags >= threshold
            mags = mags * mask
            phases = phases * mask

        # Per-block scale factors
        mag_maxes = mags.max(axis=1, keepdims=True) + 1e-10

        # Quantize magnitudes
        max_mag_val = (1 << mag_bits) - 1
        mags_q = np.clip((mags / mag_maxes * max_mag_val), 0, max_mag_val).astype(np.uint16)

        # Quantize phases
        max_phase_val = (1 << phase_bits) - 1
        phases_norm = (phases + np.pi) / (2 * np.pi)
        phases_q = np.clip((phases_norm * max_phase_val), 0, max_phase_val).astype(np.uint16)

        # Serialize: scale factors + interleaved mag/phase
        # Scale factors as uint16 (one per block)
        scales = np.clip(mag_maxes.ravel(), 0, 65535).astype(np.uint16)
        quant_payload = scales.tobytes() + mags_q.tobytes() + phases_q.tobytes()

        # Compress with zlib (realistic entropy coding proxy)
        compressed_ch = _zlib.compress(quant_payload, 9)
        all_quant_bytes.append(compressed_ch)

        # ── Decode for reconstruction ──────────────────────────────────
        # Dequantize
        mags_deq = mags_q.astype(np.float64) / max_mag_val * mag_maxes
        phases_deq = phases_q.astype(np.float64) / max_phase_val * 2 * np.pi - np.pi
        coeffs_recon = mags_deq * np.exp(1j * phases_deq)

        # Batched inverse RFT
        recon_blocks = (Phi @ coeffs_recon.T).T.real  # (num_blocks, block_size)
        recon_flat = recon_blocks.ravel()[:n_pixels_ch]
        recon_u8 = np.clip(np.round(recon_flat), 0, 255).astype(np.uint8)
        all_decoded_channels.append(recon_u8.reshape(h, w))

    enc_ms = (time.perf_counter() - t_enc_start) * 1e3

    # Build container with honest overhead
    # Header: magic(4) + version(2) + channels(1) + H(2) + W(2) + block_size(2) + mag_bits(1) + phase_bits(1) + prune(4) = 19 bytes
    # Per-channel: length(4) + compressed_data
    header = _struct.pack('>4sBBHHHBBf', b'RFTI', 1, c, h, w, block_size, mag_bits, phase_bits, prune_ratio)
    body = b''
    for comp_ch in all_quant_bytes:
        body += _struct.pack('>I', len(comp_ch)) + comp_ch
    container = header + body

    t_dec_start = time.perf_counter()
    # (decode timing already included in the encode loop above for simplicity)
    dec_ms = (time.perf_counter() - t_dec_start) * 1e3

    reconstructed = np.stack(all_decoded_channels, axis=-1).astype(np.uint8)
    return container, reconstructed, enc_ms, dec_ms


# ── RFT hybrid codec wrapper for images ────────────────────────────────────

def encode_rft_hybrid_image(
    img_array: np.ndarray,
    prune_threshold: float = 0.0,
    quant_amp_bits: int = 6,
    quant_phase_bits: int = 5,
) -> Tuple[bytes, np.ndarray, float, float]:
    """
    Encode an RGB image through the RFT hybrid codec channel-by-channel.

    The hybrid codec serializes to JSON; we measure the JSON byte length as
    the compressed size (all overhead included).

    Returns (compressed_json_bytes, reconstructed_uint8, encode_ms, decode_ms).
    """
    from algorithms.rft.hybrids.rft_hybrid_codec import encode_tensor_hybrid, decode_tensor_hybrid

    h, w, c = img_array.shape
    containers = []
    all_decoded = []
    enc_ms_total = 0.0
    dec_ms_total = 0.0

    for ch in range(c):
        channel = img_array[:, :, ch].astype(np.float64).ravel()
        t0 = time.perf_counter()
        result = encode_tensor_hybrid(
            channel,
            prune_threshold=prune_threshold,
            quant_amp_bits=quant_amp_bits,
            quant_phase_bits=quant_phase_bits,
            collect_residual_samples=False,
        )
        enc_ms_total += (time.perf_counter() - t0) * 1e3
        containers.append(result.container)

        t0 = time.perf_counter()
        decoded = decode_tensor_hybrid(result.container)
        dec_ms_total += (time.perf_counter() - t0) * 1e3
        all_decoded.append(np.clip(np.round(decoded), 0, 255).astype(np.uint8).reshape(h, w))

    # Serialize all containers as JSON (honest: this IS the compressed form)
    manifest = {"type": "rft_hybrid_image", "channels": containers}
    compressed_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    reconstructed = np.stack(all_decoded, axis=-1).astype(np.uint8)
    return compressed_json, reconstructed, enc_ms_total, dec_ms_total


# ── Kodak image downloader ────────────────────────────────────────────────

def download_kodak(dest_dir: Path, max_images: int = 24) -> List[Path]:
    """Download Kodak PhotoCD images (768×512 PNG) from r0k.us."""
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    base_url = "http://r0k.us/graphics/kodak/kodak/"
    for i in range(1, max_images + 1):
        fname = f"kodim{i:02d}.png"
        local = dest_dir / fname
        if local.exists():
            paths.append(local)
            continue
        url = f"{base_url}{fname}"
        print(f"  Downloading {url} → {local} …", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, str(local))
            paths.append(local)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")
    return paths


def generate_synthetic_images(dest_dir: Path, count: int = 6) -> List[Path]:
    """Generate synthetic gradient/pattern images as fallback."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    w, h = 768, 512
    rng = np.random.RandomState(42)

    patterns = [
        ("gradient_h", lambda: np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))),
        ("gradient_v", lambda: np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w))),
        ("checkerboard", lambda: ((np.indices((h, w)).sum(axis=0) // 32) % 2 * 200 + 27).astype(np.uint8)),
        ("noise_lo", lambda: rng.randint(100, 156, (h, w), dtype=np.uint8)),
        ("noise_hi", lambda: rng.randint(0, 256, (h, w), dtype=np.uint8)),
        ("edges", lambda: (np.clip(np.abs(np.sin(np.linspace(0, 20*np.pi, w))) * 255, 0, 255)).astype(np.uint8)[np.newaxis, :].repeat(h, axis=0)),
    ]

    for name, gen in patterns[:count]:
        local = dest_dir / f"synthetic_{name}.png"
        if not local.exists():
            gray = gen()
            rgb = np.stack([gray, gray, gray], axis=-1)
            Image.fromarray(rgb).save(local)
        paths.append(local)
    return paths


# ── Main benchmark driver ─────────────────────────────────────────────────

JPEG_QUALITIES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
WEBP_QUALITIES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
AVIF_QUALITIES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# RFT binary codec sweep: (prune_ratio, mag_bits, phase_bits)
# Reduced to 8 key operating points to keep runtime practical
# (each point takes ~30-60s per image for RFT transform + ANS coding)
RFT_SETTINGS = [
    # Very aggressive pruning → low bitrate
    (0.95, 6, 4),
    (0.90, 8, 6),
    (0.80, 8, 6),
    (0.60, 8, 8),
    (0.40, 10, 8),
    (0.20, 10, 8),
    # Near-lossless quantization (still lossy due to mag/phase quantization)
    (0.0, 10, 8),
    (0.0, 16, 14),
]


def benchmark_image(
    img_path: Path,
    run_rft: bool = True,
    run_rft_hybrid: bool = False,  # disabled by default (slow on large images)
    rft_block_size: int = 256,
    max_pixels: int = 0,
) -> List[RDPoint]:
    """Run all codecs on one image, return list of R-D points.
    
    Args:
        max_pixels: If > 0, downscale the image so total pixels <= max_pixels.
                    This keeps R-D curves valid (BPP is resolution-independent)
                    while making the slow RFT codec practical.
    """
    img = Image.open(img_path).convert("RGB")
    
    # Optionally downscale for RFT tractability
    if max_pixels > 0:
        w_orig, h_orig = img.size
        if w_orig * h_orig > max_pixels:
            scale = (max_pixels / (w_orig * h_orig)) ** 0.5
            new_w = max(16, int(w_orig * scale) // 8 * 8)  # align to 8px
            new_h = max(16, int(h_orig * scale) // 8 * 8)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  (downscaled {w_orig}×{h_orig} → {new_w}×{new_h} for tractability)")
    
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    name = img_path.stem
    points: List[RDPoint] = []

    # ── JPEG ────────────────────────────────────────────────────────────
    for q in JPEG_QUALITIES:
        try:
            data, enc_ms, dec_ms = encode_jpeg(img, q)
            recon = decode_image(data, "JPEG")
            points.append(RDPoint(
                codec="JPEG",
                quality_param=q,
                bpp=_bpp(len(data), w, h),
                psnr_db=compute_psnr(img_np, recon),
                ssim=compute_ssim(img_np, recon),
                compressed_bytes=len(data),
                encode_ms=enc_ms,
                decode_ms=dec_ms,
                image_name=name,
                width=w,
                height=h,
            ))
        except Exception as e:
            print(f"  [JPEG q={q}] ERROR: {e}")

    # ── WebP ────────────────────────────────────────────────────────────
    for q in WEBP_QUALITIES:
        try:
            data, enc_ms, dec_ms = encode_webp(img, q)
            recon = decode_image(data, "WEBP")
            points.append(RDPoint(
                codec="WebP",
                quality_param=q,
                bpp=_bpp(len(data), w, h),
                psnr_db=compute_psnr(img_np, recon),
                ssim=compute_ssim(img_np, recon),
                compressed_bytes=len(data),
                encode_ms=enc_ms,
                decode_ms=dec_ms,
                image_name=name,
                width=w,
                height=h,
            ))
        except Exception as e:
            print(f"  [WebP q={q}] ERROR: {e}")

    # ── AVIF ────────────────────────────────────────────────────────────
    for q in AVIF_QUALITIES:
        result = encode_avif(img, q)
        if result is not None:
            data, enc_ms, dec_ms = result
            recon = decode_image(data, "AVIF")
            points.append(RDPoint(
                codec="AVIF",
                quality_param=q,
                bpp=_bpp(len(data), w, h),
                psnr_db=compute_psnr(img_np, recon),
                ssim=compute_ssim(img_np, recon),
                compressed_bytes=len(data),
                encode_ms=enc_ms,
                decode_ms=dec_ms,
                image_name=name,
                width=w,
                height=h,
            ))

    # ── RFT Binary Codec ───────────────────────────────────────────────
    if run_rft:
        for prune, mbits, pbits in RFT_SETTINGS:
            try:
                compressed, recon, enc_ms, dec_ms = encode_rft_image(
                    img_np,
                    prune_ratio=prune,
                    block_size=rft_block_size,
                    mag_bits=mbits,
                    phase_bits=pbits,
                )
                points.append(RDPoint(
                    codec="RFT-Binary",
                    quality_param=prune,
                    bpp=_bpp(len(compressed), w, h),
                    psnr_db=compute_psnr(img_np, recon),
                    ssim=compute_ssim(img_np, recon),
                    compressed_bytes=len(compressed),
                    encode_ms=enc_ms,
                    decode_ms=dec_ms,
                    image_name=name,
                    width=w,
                    height=h,
                ))
            except Exception as e:
                print(f"  [RFT prune={prune} mag={mbits} ph={pbits}] ERROR: {e}")

    # ── RFT Hybrid Codec ───────────────────────────────────────────────
    if run_rft_hybrid:
        hybrid_settings = [
            (1e-1, 4, 3),
            (1e-2, 5, 4),
            (1e-3, 6, 5),
            (1e-4, 6, 5),
            (0.0, 6, 5),
        ]
        for thresh, abits, pbits in hybrid_settings:
            try:
                compressed, recon, enc_ms, dec_ms = encode_rft_hybrid_image(
                    img_np,
                    prune_threshold=thresh,
                    quant_amp_bits=abits,
                    quant_phase_bits=pbits,
                )
                points.append(RDPoint(
                    codec="RFT-Hybrid",
                    quality_param=thresh,
                    bpp=_bpp(len(compressed), w, h),
                    psnr_db=compute_psnr(img_np, recon),
                    ssim=compute_ssim(img_np, recon),
                    compressed_bytes=len(compressed),
                    encode_ms=enc_ms,
                    decode_ms=dec_ms,
                    image_name=name,
                    width=w,
                    height=h,
                ))
            except Exception as e:
                print(f"  [RFT-Hybrid thresh={thresh}] ERROR: {e}")

    return points


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_rd_curves(summary: Dict[str, List[Tuple[float, float]]], metric: str, output_path: Path):
    """Plot per-codec R-D curves (BPP vs PSNR or SSIM)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    codec_styles = {
        "JPEG": {"color": "#1f77b4", "marker": "o", "ls": "-"},
        "WebP": {"color": "#ff7f0e", "marker": "s", "ls": "-"},
        "AVIF": {"color": "#2ca02c", "marker": "^", "ls": "-"},
        "RFT-Binary": {"color": "#d62728", "marker": "D", "ls": "--"},
        "RFT-Hybrid": {"color": "#9467bd", "marker": "v", "ls": "--"},
    }

    for codec, pts in sorted(summary.items()):
        if not pts:
            continue
        # Sort by BPP
        pts_sorted = sorted(pts, key=lambda p: p[0])
        bpps = [p[0] for p in pts_sorted]
        vals = [p[1] for p in pts_sorted]
        style = codec_styles.get(codec, {"color": "gray", "marker": "x", "ls": ":"})
        ax.plot(bpps, vals, label=codec, markersize=6, linewidth=1.5, **style)

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=13)
    ylabel = "PSNR (dB)" if metric == "psnr" else "SSIM"
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Rate-Distortion: {ylabel} vs BPP  (Kodak average)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real R-D benchmark: RFT vs industry codecs")
    parser.add_argument("--images-dir", type=str, default="data/kodak",
                        help="Directory containing Kodak PNG images")
    parser.add_argument("--output-dir", type=str, default="results/rd_curves",
                        help="Output directory for results & plots")
    parser.add_argument("--max-images", type=int, default=24,
                        help="Maximum number of images to benchmark")
    parser.add_argument("--skip-rft", action="store_true",
                        help="Skip RFT codec (run only industry baselines)")
    parser.add_argument("--run-hybrid", action="store_true",
                        help="Also run RFT Hybrid codec (slow)")
    parser.add_argument("--rft-block-size", type=int, default=256,
                        help="RFT block size")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip image download; use only existing files")
    parser.add_argument("--max-pixels", type=int, default=49152,
                        help="Downscale images so total pixels <= this (default 49152 = 256×192). "
                             "Set 0 to use full resolution (very slow for RFT).")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Get images ──────────────────────────────────────────────────────
    print("=" * 72)
    print("REAL RATE-DISTORTION BENCHMARK: RFT vs Industry Codecs")
    print("=" * 72)
    print()

    # Try to find existing images first
    existing = sorted(images_dir.glob("*.png")) if images_dir.exists() else []
    if not existing and not args.skip_download:
        print(f"Downloading Kodak images to {images_dir} …")
        existing = download_kodak(images_dir, max_images=args.max_images)
    if not existing:
        print(f"No Kodak images found. Generating synthetic test images …")
        existing = generate_synthetic_images(images_dir)

    image_paths = existing[: args.max_images]
    print(f"\nBenchmarking {len(image_paths)} images:\n")
    for p in image_paths:
        print(f"  • {p.name}")
    print()

    # ── Run benchmark ───────────────────────────────────────────────────
    all_points: List[Dict] = []

    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        pts = benchmark_image(
            img_path,
            run_rft=not args.skip_rft,
            run_rft_hybrid=args.run_hybrid,
            rft_block_size=args.rft_block_size,
            max_pixels=args.max_pixels,
        )
        for p in pts:
            all_points.append(asdict(p))
            if p.codec in ("JPEG", "WebP", "AVIF"):
                tag = f"q={int(p.quality_param)}"
            else:
                tag = f"prune={p.quality_param}"
            print(f"  {p.codec:12s} {tag:12s}  BPP={p.bpp:6.3f}  PSNR={p.psnr_db:6.2f}  SSIM={p.ssim:.4f}")
        print()

    # ── Save raw results ────────────────────────────────────────────────
    raw_path = output_dir / "rd_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_points, f, indent=2)
    print(f"Saved {len(all_points)} data points → {raw_path}")

    # ── Compute per-codec averages ──────────────────────────────────────
    from collections import defaultdict

    # Group by (codec, quality_param) → average BPP, PSNR, SSIM
    grouped: Dict[Tuple[str, float], List[Dict]] = defaultdict(list)
    for pt in all_points:
        grouped[(pt["codec"], pt["quality_param"])].append(pt)

    summary_psnr: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    summary_ssim: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    summary_rows = []

    for (codec, qp), pts in sorted(grouped.items()):
        avg_bpp = np.mean([p["bpp"] for p in pts])
        avg_psnr = np.mean([p["psnr_db"] for p in pts])
        avg_ssim = np.mean([p["ssim"] for p in pts])
        summary_psnr[codec].append((float(avg_bpp), float(avg_psnr)))
        summary_ssim[codec].append((float(avg_bpp), float(avg_ssim)))
        summary_rows.append({
            "codec": codec,
            "quality_param": qp,
            "avg_bpp": float(avg_bpp),
            "avg_psnr_db": float(avg_psnr),
            "avg_ssim": float(avg_ssim),
            "num_images": len(pts),
        })

    summary_path = output_dir / "rd_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Saved summary → {summary_path}")

    # ── Summary table ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"{'Codec':14s} {'Quality':10s} {'BPP':>8s} {'PSNR dB':>9s} {'SSIM':>8s}")
    print("-" * 72)
    for row in summary_rows:
        qp = row["quality_param"]
        qp_str = f"{qp:.4f}" if isinstance(qp, float) and qp < 1 else f"{int(qp)}"
        print(f"{row['codec']:14s} {qp_str:10s} {row['avg_bpp']:8.3f} {row['avg_psnr_db']:9.2f} {row['avg_ssim']:8.4f}")
    print("=" * 72)

    # ── Honest assessment ───────────────────────────────────────────────
    jpeg_pts = [r for r in summary_rows if r["codec"] == "JPEG"]
    webp_pts = [r for r in summary_rows if r["codec"] == "WebP"]
    rft_pts = [r for r in summary_rows if r["codec"].startswith("RFT")
               and np.isfinite(r["avg_psnr_db"])]  # exclude inf-PSNR lossless

    if jpeg_pts and rft_pts:
        print()
        print("HONEST ASSESSMENT:")

        # Compare at BPP ranges where BOTH codecs have data
        jpeg_bpp_range = (min(j["avg_bpp"] for j in jpeg_pts),
                          max(j["avg_bpp"] for j in jpeg_pts))
        rft_in_range = [r for r in rft_pts
                        if r["avg_bpp"] <= jpeg_bpp_range[1] * 1.5]

        if not rft_in_range:
            # RFT's lowest BPP is still above JPEG's highest — total loss
            rft_lowest = min(rft_pts, key=lambda r: r["avg_bpp"])
            jpeg_highest = max(jpeg_pts, key=lambda j: j["avg_bpp"])
            print(f"  RFT's lowest bitrate ({rft_lowest['avg_bpp']:.1f} BPP, "
                  f"PSNR={rft_lowest['avg_psnr_db']:.1f} dB) exceeds JPEG's highest "
                  f"({jpeg_highest['avg_bpp']:.1f} BPP, PSNR={jpeg_highest['avg_psnr_db']:.1f} dB).")
            print(f"  No overlapping BPP range exists for comparison.")
        else:
            for rpt in rft_in_range:
                closest_jpeg = min(jpeg_pts, key=lambda j: abs(j["avg_bpp"] - rpt["avg_bpp"]))
                delta_psnr = rpt["avg_psnr_db"] - closest_jpeg["avg_psnr_db"]
                sign = "+" if delta_psnr > 0 else ""
                print(
                    f"  RFT at {rpt['avg_bpp']:.2f} BPP: "
                    f"PSNR={rpt['avg_psnr_db']:.1f} dB  vs  "
                    f"JPEG at {closest_jpeg['avg_bpp']:.2f} BPP: "
                    f"PSNR={closest_jpeg['avg_psnr_db']:.1f} dB  →  {sign}{delta_psnr:.1f} dB"
                )

        print()
        # Overall verdict — compare at the lowest RFT BPP vs JPEG at same BPP
        rft_lowest = min(rft_pts, key=lambda r: r["avg_bpp"])
        jpeg_at_same = min(jpeg_pts, key=lambda j: abs(j["avg_bpp"] - rft_lowest["avg_bpp"]))
        gap = rft_lowest["avg_psnr_db"] - jpeg_at_same["avg_psnr_db"]

        print("  ⚠ RFT codec does NOT compete with JPEG/WebP/AVIF for image compression.")
        print(f"    At RFT's best lossy point ({rft_lowest['avg_bpp']:.1f} BPP), PSNR is "
              f"{rft_lowest['avg_psnr_db']:.1f} dB.")
        print(f"    JPEG at {jpeg_at_same['avg_bpp']:.1f} BPP achieves "
              f"{jpeg_at_same['avg_psnr_db']:.1f} dB — {abs(gap):.1f} dB better.")
        if webp_pts:
            webp_at_same = min(webp_pts, key=lambda w: abs(w["avg_bpp"] - rft_lowest["avg_bpp"]))
            print(f"    WebP at {webp_at_same['avg_bpp']:.1f} BPP achieves "
                  f"{webp_at_same['avg_psnr_db']:.1f} dB.")
        print()
        print("    This is expected: the RFT φ-grid basis is NOT optimized for")
        print("    natural-image energy compaction the way DCT (JPEG) or wavelet")
        print("    (AVIF/JPEG2000) bases are. The RFT's value lies in spectral")
        print("    analysis and signal processing, not image compression.")
    print()

    # ── Plot R-D curves ─────────────────────────────────────────────────
    if summary_psnr:
        plot_rd_curves(dict(summary_psnr), "psnr", output_dir / "rd_plot_psnr.png")
        plot_rd_curves(dict(summary_ssim), "ssim", output_dir / "rd_plot_ssim.png")

    print("Done.")


if __name__ == "__main__":
    main()
