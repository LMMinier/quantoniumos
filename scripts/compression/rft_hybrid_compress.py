#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Hybrid RFT Compression CLI
================================

Encodes a model checkpoint into hybrid RFT containers (quantized + residual
predictor training) producing:
  * manifest_hybrid.json (bundle of tensor entries referencing per-tensor JSON files)
  * tensors/<tensor_name>.json (hybrid container)
  * predictors/<predictor_name>.json (trained residual predictor)

Key Steps:
 1. Load model weights (local path or HF snapshot if huggingface_hub installed).
 2. For each floating tensor, perform hybrid encoding (quant/prune + banding).
 3. Collect residual training samples across tensors.
 4. Train a single global predictor (unless disabled) and write it.
 5. Update each tensor container with predictor reference.

Usage Example:
  python tools/rft_hybrid_compress.py \
      --model-id gpt2 \
      --out encoded_models/gpt2_hybrid \
      --prune-threshold 1e-4 \
      --quant-amp-bits 6 --quant-phase-bits 5 \
      --train-residual --residual-hidden 32 --residual-epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None  # type: ignore

try:
    import transformers  # noqa: F401
    from transformers import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.rft_hybrid_codec import encode_tensor_hybrid  # noqa: E402
from src.core.hybrid_residual_predictor import train_residual_predictor, TinyResidualPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers from existing encode script (simplified)
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    return name.replace('/', '__').replace('.', '_')


def _find_model_files(base: Path) -> List[Tuple[Path, str]]:
    if base.is_file():
        suffix = base.suffix.lower()
        if suffix == '.safetensors':
            return [(base, 'safetensors')]
        if suffix in {'.bin', '.pt'}:
            return [(base, 'bin')]
        raise ValueError(f'Unsupported checkpoint file: {base}')
    safetensors_files = sorted(p for p in base.glob('*.safetensors'))
    if safetensors_files:
        return [(p, 'safetensors') for p in safetensors_files]
    bin_files = sorted(p for p in base.glob('*.bin'))
    if bin_files:
        return [(p, 'bin') for p in bin_files]
    pt_files = sorted(p for p in base.glob('*.pt'))
    if pt_files:
        return [(p, 'bin') for p in pt_files]
    raise FileNotFoundError(f'No weight files found in {base}')


def _resolve_model_sources(model_id: str, hf_auto_class: str | None) -> Tuple[List[Tuple[Path, str]], str]:
    path = Path(model_id)
    if path.exists():
        return _find_model_files(path if path.is_dir() else path), 'local'
    if snapshot_download is None:
        raise RuntimeError('huggingface_hub required for remote model id')
    allow_patterns = ['*.safetensors', '*.bin', '*.pt']
    local_dir = Path(snapshot_download(repo_id=model_id, allow_patterns=allow_patterns))
    return _find_model_files(local_dir), 'huggingface'


def _iter_safetensors(file_path: Path) -> Iterator[Tuple[str, torch.Tensor]]:
    if safe_open is None:
        raise RuntimeError('safetensors required to read .safetensors')
    with safe_open(file_path, framework='pt', device='cpu') as sf:
        for key in sf.keys():
            yield key, sf.get_tensor(key)


def _iter_torch_bin(file_path: Path) -> Iterator[Tuple[str, torch.Tensor]]:
    state = torch.load(file_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise ValueError(f'Unsupported checkpoint format: {file_path}')
    for item in state.items():
        yield item


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Hybrid RFT model compressor')
    p.add_argument('--model-id', required=True)
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--hf-auto-class', default=None)
    p.add_argument('--prune-threshold', type=float, default=1e-4)
    p.add_argument('--quant-amp-bits', type=int, default=6)
    p.add_argument('--quant-phase-bits', type=int, default=5)
    p.add_argument('--min-band', type=int, default=64)
    p.add_argument('--band-growth', type=float, default=2.0)
    p.add_argument('--train-residual', action='store_true')
    p.add_argument('--residual-hidden', type=int, default=32)
    p.add_argument('--residual-epochs', type=int, default=5)
    p.add_argument('--residual-steps-cap', type=int, default=None)
    p.add_argument('--max-tensors', type=int, default=None, help='Limit number of tensors (debugging)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main compression flow
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    tensors_dir = out_dir / 'tensors'
    predictors_dir = out_dir / 'predictors'
    tensors_dir.mkdir(parents=True, exist_ok=True)
    predictors_dir.mkdir(parents=True, exist_ok=True)

    files, framework = _resolve_model_sources(args.model_id, args.hf_auto_class)

    manifest: Dict = {
        'type': 'rft_hybrid_manifest_bundle',
        'version': 1,
        'model_name': args.model_id,
        'framework': framework,
        'codec': {
            'prune_threshold': args.prune_threshold,
            'quant_amp_bits': args.quant_amp_bits,
            'quant_phase_bits': args.quant_phase_bits,
        },
        'tensors': [],
        'predictor': None,
        'metrics': {
            'total_tensors': 0,
            'encoded_tensors': 0,
            'kept_coeff_total': 0,
            'coeff_total': 0,
            'avg_sparsity': 0.0,
        },
    }

    residual_samples_all = []

    tensor_count = 0
    encoded_count = 0
    for file_idx, (file_path, fmt) in enumerate(files, start=1):
        if args.max_tensors and encoded_count >= args.max_tensors:
            break
        if fmt == 'safetensors':
            weight_iter = _iter_safetensors(file_path)
        else:
            weight_iter = _iter_torch_bin(file_path)
        for name, tensor in weight_iter:
            tensor_count += 1
            if args.max_tensors and encoded_count >= args.max_tensors:
                break
            if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
                continue
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            result = encode_tensor_hybrid(
                arr,
                prune_threshold=args.prune_threshold,
                quant_amp_bits=args.quant_amp_bits,
                quant_phase_bits=args.quant_phase_bits,
                band_growth=args.band_growth,
                min_band=args.min_band,
                tensor_name=name,
                collect_residual_samples=args.train_residual,
            )
            safe_name = _sanitize_name(name) + '.json'
            with (tensors_dir / safe_name).open('w', encoding='utf-8') as fh:
                json.dump(result.container, fh)
            entry = {
                'tensor_name': name,
                'file': f'tensors/{safe_name}',
                'kept_coeff': result.container['kept_coeff'],
                'total_coeff': result.container['total_coeff'],
                'sparsity': result.container['sparsity'],
                'bitrate_coeff': result.container['bitrate_coeff'],
            }
            manifest['tensors'].append(entry)
            encoded_count += 1
            residual_samples_all.extend(result.residual_samples)
            print(f"Encoded tensor {name} kept={entry['kept_coeff']} sparsity={entry['sparsity']:.4f}")
        # end per file

    if args.train_residual and residual_samples_all:
        print(f"Training residual predictor on {len(residual_samples_all)} tensor sample groups ...")
        bands = 1
        for _, _, band_ids, _, _, _ in residual_samples_all:
            if band_ids.size:
                bands = max(bands, int(band_ids.max()) + 1)
        predictor_payload = train_residual_predictor(
            residual_samples_all,
            bands=bands,
            hidden_dim=args.residual_hidden,
            epochs=args.residual_epochs,
            steps_cap=args.residual_steps_cap,
        )
        predictor_name = 'global_residual_predictor.json'
        with (predictors_dir / predictor_name).open('w', encoding='utf-8') as fh:
            json.dump(predictor_payload, fh)
        manifest['predictor'] = f'predictors/{predictor_name}'
        # Update tensor containers with predictor ref
        for entry in manifest['tensors']:
            tensor_path = out_dir / entry['file']
            with tensor_path.open('r', encoding='utf-8') as fh:
                cont = json.load(fh)
            cont['codec']['residual_predictor_ref'] = manifest['predictor']
            with tensor_path.open('w', encoding='utf-8') as fh:
                json.dump(cont, fh)
        print('Residual predictor stored and referenced in all tensor containers.')

    # Metrics
    if manifest['tensors']:
        sparsities = [t['sparsity'] for t in manifest['tensors']]
        manifest['metrics']['avg_sparsity'] = float(sum(sparsities) / len(sparsities))
        manifest['metrics']['kept_coeff_total'] = int(sum(t['kept_coeff'] for t in manifest['tensors']))
        manifest['metrics']['coeff_total'] = int(sum(t['total_coeff'] for t in manifest['tensors']))
    manifest['metrics']['total_tensors'] = tensor_count
    manifest['metrics']['encoded_tensors'] = encoded_count

    with (out_dir / 'manifest_hybrid.json').open('w', encoding='utf-8') as fh:
        json.dump(manifest, fh)
    print(f"Hybrid manifest written: {out_dir / 'manifest_hybrid.json'}")


if __name__ == '__main__':
    main()
