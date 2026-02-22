# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

#!/usr/bin/env python3
"""CLI to decode RFT vertex containers back into a PyTorch checkpoint."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

try:
    from safetensors.torch import save_file as safetensors_save
except Exception:  # pragma: no cover - optional dependency
    safetensors_save = None  # type: ignore

# Ensure repo root on path for src imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.rft_vertex_codec import decode_tensor, enable_assembly_rft, is_assembly_enabled  # noqa: E402


def _load_manifest(input_dir: Path) -> Dict:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {input_dir}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_container(input_dir: Path, relative_path: str) -> Dict:
    file_path = input_dir / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Chunk file missing: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_raw_tensor(input_dir: Path, relative_path: str) -> torch.Tensor:
    file_path = input_dir / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Raw tensor file missing: {file_path}")
    return torch.load(file_path, map_location="cpu")


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    if dtype_str.startswith("torch."):
        attr = dtype_str.split(".", 1)[1]
        if hasattr(torch, attr):
            return getattr(torch, attr)
    if hasattr(torch, dtype_str):
        return getattr(torch, dtype_str)
    try:
        np_dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise ValueError(f"Unrecognized dtype string '{dtype_str}'") from exc

    numpy_to_torch = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
    }
    if np_dtype in numpy_to_torch:
        return numpy_to_torch[np_dtype]
    raise ValueError(f"Unsupported dtype mapping for '{dtype_str}'")


def _iter_manifest_entries(manifest: Dict) -> Iterator[Dict]:
    if manifest.get("type") == "rft_vertex_manifest_bundle":
        for sub in manifest.get("manifests", []):
            yield from sub.get("tensors", [])
    else:
        yield from manifest.get("tensors", [])


def decode_manifest(input_dir: Path, verify: bool, tolerance: float | None) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    manifest = _load_manifest(input_dir)
    tensors: Dict[str, torch.Tensor] = {}
    secondary_accepts: List[str] = []
    assembly_required: List[str] = []

    for entry in _iter_manifest_entries(manifest):
        name = entry["tensor_name"]
        scheme = entry.get("scheme", "rft")
        if scheme == "rft":
            container = _load_container(input_dir, entry["file"])
            tensor_array, status = decode_tensor(
                container,
                verify_checksum=verify,
                atol=tolerance,
                tensor_name=name,
                return_status=True,
            )
            dtype = _dtype_from_string(entry.get("dtype", "torch.float32"))
            tensor = torch.from_numpy(tensor_array).to(dtype=dtype).contiguous()
            tensors[name] = tensor
            if status.get("used_secondary_checksum"):
                secondary_accepts.append(name)
            if container.get("backend") == "assembly":
                assembly_required.append(name)
        elif scheme == "raw":
            tensors[name] = _load_raw_tensor(input_dir, entry["file"])
        elif scheme == "raw_base64":
            container = _load_container(input_dir, entry["file"])
            data_bytes = base64.b64decode(container["data"])
            np_dtype = np.dtype(container["dtype"])
            np_array = np.frombuffer(data_bytes, dtype=np_dtype).copy()
            np_array = np_array.reshape(tuple(container["original_shape"]))
            torch_dtype = _dtype_from_string(container["dtype"])
            tensor = torch.from_numpy(np_array).to(dtype=torch_dtype).contiguous()
            tensors[name] = tensor
        else:
            raise ValueError(f"Unknown encoding scheme '{scheme}' for tensor {name}")

    return tensors, secondary_accepts, assembly_required


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode RFT vertex JSON chunks to a PyTorch checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory containing manifest.json and chunk files")
    parser.add_argument("--output-file", default="pytorch_model.bin", help="Output file for reconstructed weights (.bin or .safetensors)")
    parser.add_argument("--verify", action="store_true", help="Enable checksum verification during decoding")
    parser.add_argument("--verify-tolerance", type=float, default=None, help="Override tolerance for secondary checksum verification")
    parser.add_argument("--use-assembly", action="store_true", help="Enable assembly UnitaryRFT backend when available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    assembly_on = enable_assembly_rft(args.use_assembly)
    tensors, secondary_accepts, assembly_required = decode_manifest(
        input_dir,
        verify=args.verify,
        tolerance=args.verify_tolerance,
    )

    output_path = Path(args.output_file)
    if output_path.suffix == ".safetensors":
        if safetensors_save is None:
            raise RuntimeError("safetensors is required to write .safetensors files")
        safetensors_save(tensors, str(output_path))
    else:
        torch.save(tensors, output_path)

    print(f"Decoded state_dict written to {output_path}")
    if secondary_accepts:
        joined = ", ".join(secondary_accepts)
        print(f"Secondary quantized checksum accepted for tensors: {joined}")
    if assembly_required:
        joined = ", ".join(assembly_required)
        status = "enabled" if assembly_on and is_assembly_enabled() else "disabled"
        print(f"Assembly backend tensors: {joined} (assembly {status})")


if __name__ == "__main__":
    main()
