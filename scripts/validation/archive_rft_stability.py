#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
scripts/archive_rft_stability.py

Archives reproducible RFT round-trip stability metrics with provenance.

Outputs:
  <output>/
    manifest.json
    results.json
    stdout.txt

This script is intentionally robust to different RFT APIs:
- class ResonantFourierTransform with forward/inverse or __call__
- function resonant_fourier_transform / inverse_resonant_fourier_transform
- fallback to canonical_true_rft module if available
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def _git_sha() -> str:
    try:
        return _run(["git", "rev-parse", "HEAD"])
    except Exception:
        return "UNKNOWN"


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _env_info() -> Dict[str, str]:
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    return info


def _command_string() -> str:
    # Best-effort exact invocation
    return " ".join([sys.executable] + sys.argv)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_forward_inverse() -> Tuple[Callable[..., Any], Optional[Callable[..., Any]], str]:
    """
    Returns: (forward, inverse_or_None, api_desc)
    """
    # 1) Try module algorithms.rft.core.resonant_fourier_transform
    try:
        import importlib

        m = importlib.import_module("algorithms.rft.core.resonant_fourier_transform")

        # Class API
        if hasattr(m, "ResonantFourierTransform"):
            cls = getattr(m, "ResonantFourierTransform")
            obj = cls()  # type: ignore
            forward = getattr(obj, "forward", None) or getattr(obj, "__call__", None)
            inv = getattr(obj, "inverse", None) or getattr(obj, "inv", None)
            if callable(forward):
                return forward, inv if callable(inv) else None, "resonant_fourier_transform:ResonantFourierTransform"

        # Function API (common names)
        for fwd_name in ("resonant_fourier_transform", "rft_forward", "forward_rft", "rft"):
            if hasattr(m, fwd_name) and callable(getattr(m, fwd_name)):
                forward = getattr(m, fwd_name)
                inv = None
                for inv_name in ("inverse_resonant_fourier_transform", "rft_inverse", "inverse_rft", "irft"):
                    if hasattr(m, inv_name) and callable(getattr(m, inv_name)):
                        inv = getattr(m, inv_name)
                        break
                return forward, inv, f"resonant_fourier_transform:{fwd_name}"

        # Heuristic: pick any callable containing both 'rft' and 'transform'
        for name in dir(m):
            if "rft" in name.lower() and "transform" in name.lower():
                obj = getattr(m, name)
                if callable(obj):
                    return obj, None, f"resonant_fourier_transform:{name}"

    except Exception:
        pass

    # 2) Fallback: canonical_true_rft module
    try:
        import importlib

        m = importlib.import_module("algorithms.rft.core.canonical_true_rft")

        # Class API
        for cls_name in ("CanonicalTrueRFT", "TrueRFT", "PhiRFT", "CanonicalRFT"):
            if hasattr(m, cls_name):
                cls = getattr(m, cls_name)
                obj = cls()  # type: ignore
                forward = getattr(obj, "forward", None) or getattr(obj, "__call__", None)
                inv = getattr(obj, "inverse", None) or getattr(obj, "inv", None)
                if callable(forward):
                    return forward, inv if callable(inv) else None, f"canonical_true_rft:{cls_name}"

        # Function API
        for fwd_name in ("forward", "canonical_true_rft", "phi_rft_forward"):
            if hasattr(m, fwd_name) and callable(getattr(m, fwd_name)):
                forward = getattr(m, fwd_name)
                inv = None
                for inv_name in ("inverse", "inverse_canonical_true_rft", "phi_rft_inverse"):
                    if hasattr(m, inv_name) and callable(getattr(m, inv_name)):
                        inv = getattr(m, inv_name)
                        break
                return forward, inv, f"canonical_true_rft:{fwd_name}"

    except Exception:
        pass

    raise ImportError(
        "Could not locate a usable RFT forward() in "
        "algorithms.rft.core.resonant_fourier_transform or algorithms.rft.core.canonical_true_rft"
    )


def _round_trip_metrics(
    forward: Callable[..., Any],
    inverse: Optional[Callable[..., Any]],
    x: np.ndarray,
    N: int,
) -> Dict[str, float]:
    """
    Test RFT round-trip using square-kernel mode with gram normalization.
    
    The default rft_forward() uses waveform mode (T=N*16) which is NOT invertible
    via simple rft_inverse(). For proper round-trip testing, we must use:
    - use_gram_normalization=True OR frame_correct=True
    
    This uses the rft_basis_matrix directly to test true invertibility.
    """
    metrics: Dict[str, float] = {}
    metrics["x_l2"] = float(np.linalg.norm(x))
    
    # Try square-kernel mode with gram normalization
    try:
        # Method 1: Use forward with use_gram_normalization=True
        y = forward(x, use_gram_normalization=True)
        y = np.asarray(y)
        metrics["y_l2"] = float(np.linalg.norm(y))
        metrics["energy_ratio_y_over_x"] = float((metrics["y_l2"] / metrics["x_l2"]) if metrics["x_l2"] != 0 else np.nan)
        
        if inverse is not None:
            # Inverse with same settings
            try:
                xr = inverse(y, N, use_gram_normalization=True)
            except TypeError:
                xr = inverse(y, N)
            xr = np.asarray(xr)
            
            diff = xr - x
            denom = np.linalg.norm(x) if np.linalg.norm(x) != 0 else 1.0
            metrics["has_inverse"] = 1.0
            metrics["rel_l2_recon_error"] = float(np.linalg.norm(diff) / denom)
            metrics["max_abs_recon_error"] = float(np.max(np.abs(diff)))
            metrics["api_mode"] = "gram_normalized"
        else:
            metrics["has_inverse"] = 0.0
        
        return metrics
    except TypeError:
        pass
    
    # Method 2: Use direct matrix computation for true invertibility test
    try:
        from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
        
        # Build square basis matrix with gram normalization
        Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
        
        # Forward: y = Phi^H @ x
        y = Phi.conj().T @ x
        y = np.asarray(y)
        
        metrics["y_l2"] = float(np.linalg.norm(y))
        metrics["energy_ratio_y_over_x"] = float((metrics["y_l2"] / metrics["x_l2"]) if metrics["x_l2"] != 0 else np.nan)
        
        # Inverse: xr = Phi @ y (for unitary Phi)
        xr = Phi @ y
        xr = np.asarray(xr)
        
        diff = xr - x
        denom = np.linalg.norm(x) if np.linalg.norm(x) != 0 else 1.0
        metrics["has_inverse"] = 1.0
        metrics["rel_l2_recon_error"] = float(np.linalg.norm(diff) / denom)
        metrics["max_abs_recon_error"] = float(np.max(np.abs(diff)))
        metrics["api_mode"] = "direct_matrix_gram"
        
        return metrics
    except Exception as e:
        metrics["error"] = str(e)
    
    # Fallback: just measure forward energy (no inverse test)
    try:
        y = forward(x)
        y = np.asarray(y)
        metrics["y_l2"] = float(np.linalg.norm(y))
        metrics["energy_ratio_y_over_x"] = float((metrics["y_l2"] / metrics["x_l2"]) if metrics["x_l2"] != 0 else np.nan)
        metrics["has_inverse"] = 0.0
        metrics["api_mode"] = "forward_only"
    except Exception as e:
        metrics["error"] = str(e)
        metrics["has_inverse"] = 0.0
    
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output directory, e.g. data/artifacts/rft_stability")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic vectors")
    ap.add_argument("--sizes", type=str, default="64,128,256,512,1024", help="Comma-separated N sizes")
    ap.add_argument("--trials", type=int, default=10, help="Trials per size")
    args = ap.parse_args()

    out_dir = Path(args.output)
    _ensure_dir(out_dir)

    sha = _git_sha()
    ts = _now_utc_iso()
    cmd = _command_string()
    env = _env_info()

    # Capture stdout to file as well
    stdout_path = out_dir / "stdout.txt"
    def log(msg: str) -> None:
        print(msg)
        with stdout_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"Generating RFT stability artifacts to {out_dir}/")
    log(f"Commit: {sha}")
    log(f"Seed: {args.seed}")

    forward, inverse, api_desc = _resolve_forward_inverse()
    log(f"API: {api_desc}")
    log(f"Inverse available: {'YES' if inverse is not None else 'NO'}")

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    results: Dict[str, Any] = {
        "commit_sha": sha,
        "timestamp_utc": ts,
        "command": cmd,
        "seed": args.seed,
        "api": api_desc,
        "env": env,
        "dataset": {
            "type": "synthetic_complex_vectors",
            "trials_per_size": args.trials,
            "sizes": sizes,
        },
        "by_size": {},
    }

    for N in sizes:
        errs = []
        maxabs = []
        eratio = []
        has_inv = None

        for _ in range(args.trials):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            x = x.astype(np.complex128, copy=False)
            m = _round_trip_metrics(forward, inverse, x, N)
            eratio.append(m.get("energy_ratio_y_over_x", float("nan")))
            has_inv = bool(m.get("has_inverse", 0.0))
            if has_inv:
                errs.append(m["rel_l2_recon_error"])
                maxabs.append(m["max_abs_recon_error"])

        entry: Dict[str, Any] = {
            "N": N,
            "trials": args.trials,
            "energy_ratio_y_over_x": {
                "mean": float(np.nanmean(eratio)),
                "std": float(np.nanstd(eratio)),
            },
            "has_inverse": bool(has_inv),
        }
        if has_inv:
            entry["rel_l2_recon_error"] = {
                "mean": float(np.mean(errs)),
                "std": float(np.std(errs)),
                "max": float(np.max(errs)),
            }
            entry["max_abs_recon_error"] = {
                "mean": float(np.mean(maxabs)),
                "max": float(np.max(maxabs)),
            }

        results["by_size"][str(N)] = entry
        log(f"N={N} has_inverse={has_inv} energy_ratio_mean={entry['energy_ratio_y_over_x']['mean']:.6g}"
            + (f" rel_l2_err_mean={entry['rel_l2_recon_error']['mean']:.6g}" if has_inv else ""))

    manifest = {
        "timestamp_utc": ts,
        "commit_sha": sha,
        "command": cmd,
        "seed": args.seed,
        "env": env,
        "dataset": results["dataset"],
        "api": api_desc,
        "outputs": ["manifest.json", "results.json", "stdout.txt"],
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    log("✓ Artifacts written:")
    log(f"  - {out_dir / 'manifest.json'}")
    log(f"  - {out_dir / 'results.json'}")
    log(f"  - {out_dir / 'stdout.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
