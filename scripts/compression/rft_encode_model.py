# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

#!/usr/bin/env python3
"""CLI to encode model weights into RFT vertex containers on disk (streaming-friendly)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency, validated via requirements
    snapshot_download = None  # type: ignore

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - ensures helpful error if missing at runtime
    safe_open = None  # type: ignore

try:
    import transformers  # noqa: F401
    from transformers import AutoModel
except Exception:  # pragma: no cover - transformers optional for local checkpoints
    AutoModel = None  # type: ignore

# Ensure repo root on path for src imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.rft_vertex_codec import decode_tensor, encode_tensor, enable_assembly_rft  # noqa: E402


PathLike = Path | str


def _sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def _tensor_original_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _find_model_files(base: Path) -> List[Tuple[Path, str]]:
    """Return a prioritized list of model weight files under ``base``."""
    if base.is_file():
        suffix = base.suffix.lower()
        if suffix == ".safetensors":
            return [(base, "safetensors")]
        if suffix in {".bin", ".pt"}:
            return [(base, "bin")]
        raise ValueError(f"Unsupported checkpoint file: {base}")

    safetensors_files = sorted(p for p in base.glob("*.safetensors"))
    if safetensors_files:
        return [(p, "safetensors") for p in safetensors_files]

    bin_files = sorted(p for p in base.glob("*.bin"))
    if bin_files:
        return [(p, "bin") for p in bin_files]

    pt_files = sorted(p for p in base.glob("*.pt"))
    if pt_files:
        return [(p, "bin") for p in pt_files]

    raise FileNotFoundError(f"No weight files (.safetensors/.bin/.pt) found in {base}")


def _resolve_model_sources(model_id: str, hf_auto_class: str | None) -> Tuple[List[Tuple[Path, str]], str]:
    path = Path(model_id)
    if path.exists():
        files = _find_model_files(path if path.is_dir() else path)
        return files, "local"

    # Remote model; prefer safetensors snapshots
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required to download models by ID")

    allow_patterns = ["*.safetensors", "*.bin", "*.pt"]
    local_dir = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.onnx", "*.msgpack", "*.h5"],
        )
    )
    files = _find_model_files(local_dir)

    # If only .bin were found and AutoModel is available, allow fallback to AutoModel load
    if not files and AutoModel is not None:
        auto_class = AutoModel
        if hf_auto_class and hasattr(transformers, hf_auto_class):
            auto_class = getattr(transformers, hf_auto_class)
        model = auto_class.from_pretrained(model_id)
        model.to("cpu")
        model.eval()
        tmp_path = local_dir / "state_dict.pt"
        torch.save(model.state_dict(), tmp_path)
        files = [(tmp_path, "bin")]

    if not files:
        raise FileNotFoundError(f"Unable to locate weights for {model_id}")

    return files, "huggingface"


def encode_state_dict_to_disk(
    weights: Iterable[Tuple[str, torch.Tensor]],
    output_dir: Path,
    chunk_size: int,
    sample_tensors: int,
    log_every: int,
    source_file: str,
    tolerance: float,
    prune_threshold: Optional[float],
    quant_bits_amplitude: Optional[int],
    quant_bits_phase: Optional[int],
    ans_precision: Optional[int],
) -> Dict:
    chunks_dir = output_dir / "chunks"
    raw_dir = output_dir / "raw"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "type": "rft_vertex_manifest",
        "version": 1,
        "chunk_size": chunk_size,
        "streamed": True,
        "source_file": source_file,
        "tensors": [],
        "metrics": {
            "original_size_bytes": 0,
            "encoded_size_bytes": 0,
            "rft_tensor_count": 0,
            "raw_tensor_count": 0,
            "max_sample_error": None,
            "lossy_tensor_count": 0,
            "lossy_chunk_count": 0,
        },
    }

    codec_summary: Dict[str, Any] = {}
    if prune_threshold is not None:
        codec_summary["prune_threshold"] = prune_threshold
    if quant_bits_amplitude is not None:
        codec_summary["quant_bits_amplitude"] = quant_bits_amplitude
    if quant_bits_phase is not None:
        codec_summary["quant_bits_phase"] = quant_bits_phase
    if ans_precision is not None:
        codec_summary["ans_precision"] = ans_precision
    if codec_summary:
        manifest["codec"] = codec_summary

    original_total = 0
    encoded_total = 0
    max_sample_error = 0.0
    backend_seen: set[str] = set()
    lossy_tensor_count = 0
    lossy_chunk_count = 0

    float_sampled = 0
    start_time = time.perf_counter()

    for idx, (name, tensor) in enumerate(weights, start=1):
        tensor = tensor.detach().cpu()
        orig_size = _tensor_original_size(tensor)
        original_total += orig_size

        entry = {
            "tensor_name": name,
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "original_size_bytes": orig_size,
            "source": source_file,
        }

        np_tensor = tensor.numpy()
        container = encode_tensor(
            np_tensor,
            chunk_size=chunk_size,
            tolerance=tolerance,
            prune_threshold=prune_threshold,
            quant_bits_amplitude=quant_bits_amplitude,
            quant_bits_phase=quant_bits_phase,
            ans_precision=ans_precision,
        )

        if container["type"] == "rft_vertex_tensor_container":
            safe_name = f"{_sanitize_name(name)}.json"
            chunk_path = chunks_dir / safe_name
            _write_json(chunk_path, container)
            size_bytes = chunk_path.stat().st_size
            encoded_total += size_bytes
            entry.update({
                "scheme": "rft",
                "file": f"chunks/{safe_name}",
                "encoded_size_bytes": size_bytes,
            })

            codec_info = container.get("codec", {})
            if codec_info:
                entry["codec"] = codec_info
            if codec_info.get("mode") == "lossy":
                entry["lossy"] = True
                lossy_tensor_count += 1
                lossy_chunk_count += sum(
                    1
                    for chunk in container.get("chunks", [])
                    if chunk.get("codec", {}).get("mode", "lossless") != "lossless"
                )
            else:
                entry["lossy"] = False

            backend_value = container.get("backend")
            if backend_value:
                entry["backend"] = backend_value
                backend_seen.add(backend_value)
            if container.get("tolerance") is not None:
                entry["tolerance"] = container.get("tolerance")
            if container.get("quantized_checksum"):
                entry["quantized_checksum"] = container.get("quantized_checksum")
            manifest["metrics"]["rft_tensor_count"] += 1

            if float_sampled < sample_tensors:
                decoded = decode_tensor(container, verify_checksum=False)
                np_dec = decoded.astype(np.float64)
                np_src = np_tensor.astype(np.float64)
                err = float(np.max(np.abs(np_dec - np_src)))
                max_sample_error = max(max_sample_error, err)
                float_sampled += 1
        elif container["type"] == "rft_raw_tensor_container":
            safe_name = f"{_sanitize_name(name)}.json"
            raw_path = raw_dir / safe_name
            _write_json(raw_path, container)
            size_bytes = raw_path.stat().st_size
            encoded_total += size_bytes
            entry.update({
                "scheme": "raw_base64",
                "file": f"raw/{safe_name}",
                "encoded_size_bytes": size_bytes,
            })
            manifest["metrics"]["raw_tensor_count"] += 1
            print(f"Bypassed non-floating tensor: {name} ({entry['dtype']})")
        else:
            raise ValueError(f"Unsupported container type {container['type']} for tensor {name}")

        manifest["tensors"].append(entry)

        if log_every and idx % log_every == 0:
            elapsed = time.perf_counter() - start_time
            print(f"Processed {idx} tensors in {elapsed:.2f}s - running encoded size {encoded_total/1024**2:.2f} MB")

    metrics = manifest["metrics"]
    metrics["original_size_bytes"] = original_total
    metrics["encoded_size_bytes"] = encoded_total
    metrics["max_sample_error"] = max_sample_error if float_sampled > 0 else None
    metrics["processing_seconds"] = time.perf_counter() - start_time
    metrics["lossy_tensor_count"] = lossy_tensor_count
    metrics["lossy_chunk_count"] = lossy_chunk_count
    if backend_seen:
        manifest["backend"] = sorted(backend_seen)
    return manifest


def _iter_safetensors(file_path: Path) -> Iterator[Tuple[str, torch.Tensor]]:
    if safe_open is None:
        raise RuntimeError("safetensors is required to read .safetensors files")
    with safe_open(file_path, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            yield key, sf.get_tensor(key)


def _iter_torch_bin(file_path: Path) -> Iterator[Tuple[str, torch.Tensor]]:
    state = torch.load(file_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {file_path}")
    for item in state.items():
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a model checkpoint into RFT vertex JSON chunks.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model ID, directory, or checkpoint path")
    parser.add_argument("--output-dir", default="encoded_model", help="Directory to write chunks and manifest")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Maximum chunk size for RFT encoding")
    parser.add_argument("--sample-tensors", type=int, default=3, help="Number of tensors to sample for roundtrip error")
    parser.add_argument("--log-every", type=int, default=25, help="Progress logging frequency (0 to disable)")
    parser.add_argument("--hf-auto-class", default=None, help="Optional transformers auto class (e.g. AutoModelForCausalLM)")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance used for quantized checksum fallbacks")
    parser.add_argument("--use-assembly", action="store_true", help="Enable assembly UnitaryRFT backend when available")
    parser.add_argument("--prune-threshold", type=float, default=None, help="Amplitude threshold for coefficient pruning")
    parser.add_argument("--quant-bits-amplitude", type=int, default=None, help="Bit depth for amplitude quantization")
    parser.add_argument("--quant-bits-phase", type=int, default=None, help="Bit depth for phase quantization")
    parser.add_argument(
        "--ans-precision",
        type=int,
        default=None,
        help="Precision (in bits) for rANS entropy coding; 0 selects the default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assembly_on = enable_assembly_rft(args.use_assembly)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files, framework = _resolve_model_sources(args.model_id, args.hf_auto_class)

    aggregate_manifest: Dict[str, Any] = {
        "type": "rft_vertex_manifest_bundle",
        "version": 1,
        "model_name": args.model_id,
        "framework": framework,
        "chunk_size": args.chunk_size,
        "manifests": [],
    }
    if args.use_assembly:
        aggregate_manifest["assembly_requested"] = True

    aggregate_codec: Dict[str, Any] = {}
    if args.prune_threshold is not None:
        aggregate_codec["prune_threshold"] = args.prune_threshold
    if args.quant_bits_amplitude is not None:
        aggregate_codec["quant_bits_amplitude"] = args.quant_bits_amplitude
    if args.quant_bits_phase is not None:
        aggregate_codec["quant_bits_phase"] = args.quant_bits_phase
    if args.ans_precision is not None:
        aggregate_codec["ans_precision"] = args.ans_precision
    if aggregate_codec:
        aggregate_manifest["codec"] = aggregate_codec

    total_original = 0
    total_encoded = 0
    global_max_error = 0.0
    sampled_any = False
    total_lossy_tensors = 0
    total_lossy_chunks = 0

    for file_idx, (file_path, fmt) in enumerate(files, start=1):
        print(f"Processing source [{file_idx}/{len(files)}]: {file_path} ({fmt})")
        if fmt == "safetensors":
            weights_iter = _iter_safetensors(file_path)
        else:
            if file_path.suffix.endswith(".bin"):
                print("Warning: loading .bin file into memory; consider safetensors for large checkpoints.")
            weights_iter = _iter_torch_bin(file_path)

        manifest = encode_state_dict_to_disk(
            weights=weights_iter,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            sample_tensors=args.sample_tensors,
            log_every=args.log_every,
            source_file=str(file_path),
            tolerance=args.tolerance,
            prune_threshold=args.prune_threshold,
            quant_bits_amplitude=args.quant_bits_amplitude,
            quant_bits_phase=args.quant_bits_phase,
            ans_precision=args.ans_precision,
        )

        aggregate_manifest["manifests"].append(manifest)
        total_original += manifest["metrics"]["original_size_bytes"]
        total_encoded += manifest["metrics"]["encoded_size_bytes"]
        total_lossy_tensors += manifest["metrics"].get("lossy_tensor_count", 0) or 0
        total_lossy_chunks += manifest["metrics"].get("lossy_chunk_count", 0) or 0
        if manifest["metrics"]["max_sample_error"] is not None:
            sampled_any = True
            global_max_error = max(global_max_error, manifest["metrics"]["max_sample_error"])

    aggregate_manifest["metrics"] = {
        "original_size_bytes": total_original,
        "encoded_size_bytes": total_encoded,
        "max_sample_error": global_max_error if global_max_error else None,
        "lossy_tensor_count": total_lossy_tensors,
        "lossy_chunk_count": total_lossy_chunks,
    }

    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, aggregate_manifest)

    original_mb = total_original / 1024**2 if total_original else 0.0
    encoded_mb = total_encoded / 1024**2 if total_encoded else 0.0
    print(f"Manifest written to {manifest_path}")
    if args.use_assembly:
        status = "enabled" if assembly_on else "requested but unavailable"
        print(f"Assembly backend: {status}")
    print(f"Original size: {original_mb:.2f} MB | Encoded size: {encoded_mb:.2f} MB")
    if total_lossy_tensors:
        print(f"Lossy tensors encoded: {total_lossy_tensors} across {total_lossy_chunks} chunks")
    if sampled_any:
        print(f"Max sampled reconstruction error: {global_max_error:.3e}")
    elif args.sample_tensors:
        print("Max sampled reconstruction error: (sampling skipped)")


if __name__ == "__main__":
    main()
