#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Shim for legacy imports.
Provides a CLI-compatible main() that runs basic avalanche tests
by delegating to EnhancedRFTCryptoV2 metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path

import sys

# Ensure core is importable relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / 'algorithms' / 'rft' / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

try:
    from enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2  # type: ignore
except Exception as e:
    print(f"Failed to import EnhancedRFTCryptoV2: {e}")
    sys.exit(1)


def run_avalanche(samples: int = 3, message_bits: int = 16, key_bits: int = 12) -> dict:
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    metrics = cipher.get_cipher_metrics(
        message_trials=samples,
        message_bit_samples=message_bits,
        key_bit_samples=key_bits,
        key_trials=samples,
        throughput_blocks=1024,
    )
    return {
        'message_avalanche': metrics.message_avalanche,
        'key_avalanche': metrics.key_avalanche,
        'key_sensitivity': metrics.key_sensitivity,
        'throughput_mbps': metrics.throughput_mbps,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Avalanche analyzer shim")
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--message-bits', type=int, default=16)
    parser.add_argument('--key-bits', type=int, default=12)
    parser.add_argument('--out', type=str, default='results/avalanche_quick.json')
    args = parser.parse_args(argv)

    results = run_avalanche(args.samples, args.message_bits, args.key_bits)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
