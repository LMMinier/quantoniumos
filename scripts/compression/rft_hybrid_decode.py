#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Hybrid RFT Decode CLI
=========================

Reconstructs a state_dict from a hybrid RFT compressed directory created by
`rft_hybrid_compress.py`.

Outputs either a PyTorch .bin (torch.save(dict)) or a .safetensors if extension
is .safetensors and safetensors is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    from safetensors.torch import save_file as safetensors_save
except Exception:  # pragma: no cover
    safetensors_save = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.rft_hybrid_codec import decode_tensor_hybrid  # noqa: E402
from src.core.hybrid_residual_predictor import TinyResidualPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Decode hybrid RFT compressed model')
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-file', default='pytorch_model.bin')
    p.add_argument('--no-predictor', action='store_true', help='Ignore residual predictor even if present')
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    manifest_path = input_dir / 'manifest_hybrid.json'
    if not manifest_path.exists():
        raise FileNotFoundError('manifest_hybrid.json not found (did you run hybrid compress?)')
    manifest = _load_json(manifest_path)

    predictor = None
    predictor_ref = manifest.get('predictor')
    if predictor_ref and not args.no_predictor:
        pred_path = input_dir / predictor_ref
        if not pred_path.exists():
            raise FileNotFoundError(f'Predictor file missing: {pred_path}')
        predictor_obj = _load_json(pred_path)
        predictor = TinyResidualPredictor.deserialize(predictor_obj)

    state_dict: Dict[str, torch.Tensor] = {}
    for entry in manifest.get('tensors', []):
        tensor_path = input_dir / entry['file']
        container = _load_json(tensor_path)
        tensor_np = decode_tensor_hybrid(container, predictor=predictor)
        state_dict[entry['tensor_name']] = torch.from_numpy(tensor_np)
        print(f"Decoded tensor {entry['tensor_name']}")

    out_path = Path(args.output_file)
    if out_path.suffix == '.safetensors':
        if safetensors_save is None:
            raise RuntimeError('safetensors not installed')
        safetensors_save(state_dict, str(out_path))
    else:
        torch.save(state_dict, out_path)
    print(f"Hybrid decoded checkpoint written to {out_path}")


if __name__ == '__main__':
    main()
