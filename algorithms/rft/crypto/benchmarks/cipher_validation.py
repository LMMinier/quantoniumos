# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Deterministic validation of the Enhanced RFT Crypto and waveform hash."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
from algorithms.rft.quantum_inspired.geometric_waveform_hash import GeometricWaveformHash


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_METRICS_FILE = RESULTS_DIR / "paper_metrics_verification.json"


@dataclass
class TestResult:
    description: str
    status: str
    details: str


def _format_status(success: bool) -> str:
    return "✅ PASS" if success else "❌ FAIL"


def validate_block_cipher() -> List[TestResult]:
    print("🔐 Validating deterministic Feistel block behaviour...")
    test_results: List[TestResult] = []

    key = bytes.fromhex("544553545f4d41535445525f4b45595f33325f42595445535f5f")
    plaintext = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    expected_ciphertext = bytes.fromhex("99f9289bdbee6e99ae3ac3776983c6da")

    cipher = EnhancedRFTCryptoV2(key)
    ciphertext = cipher._feistel_encrypt(plaintext)
    success_enc = ciphertext == expected_ciphertext
    test_results.append(TestResult(
        description="Feistel encrypt matches reference vector",
        status=_format_status(success_enc),
        details=f"expected={expected_ciphertext.hex()} actual={ciphertext.hex()}"
    ))
    print(f"  {_format_status(success_enc)} Feistel encryption vector")

    decrypted = cipher._feistel_decrypt(ciphertext)
    success_dec = decrypted == plaintext
    test_results.append(TestResult(
        description="Feistel decrypt restores original block",
        status=_format_status(success_dec),
        details=f"expected={plaintext.hex()} actual={decrypted.hex()}"
    ))
    print(f"  {_format_status(success_dec)} Feistel decryption round-trip")

    metrics = cipher.get_cipher_metrics(message_trials=2, key_trials=2, throughput_blocks=128)
    thresholds = {
        "message_avalanche": 0.35,
        "key_avalanche": 0.35,
        "key_sensitivity": 0.30,
    }
    for metric_name, threshold in thresholds.items():
        value = getattr(metrics, metric_name)
        success = value >= threshold
        test_results.append(TestResult(
            description=f"{metric_name.replace('_', ' ').title()} >= {threshold}",
            status=_format_status(success),
            details=f"value={value:.3f}"
        ))
        print(f"  {_format_status(success)} {metric_name.replace('_', ' ')} = {value:.3f}")

    return test_results


def validate_waveform_hash() -> List[TestResult]:
    print("🧮 Validating geometric waveform hash determinism...")
    hasher = GeometricWaveformHash()
    results: List[TestResult] = []

    message = b"quantonium"
    expected_digest = bytes.fromhex("8933c4d8a4da0668f731a1f364f6a766a9af3a18228d22b8319f06eb2f105a58")
    digest = hasher.hash(message)
    success_det = digest == expected_digest
    results.append(TestResult(
        description="Waveform hash matches reference digest",
        status=_format_status(success_det),
        details=f"expected={expected_digest.hex()} actual={digest.hex()}"
    ))
    print(f"  {_format_status(success_det)} hash reference digest")

    digest2 = hasher.hash(message)
    success_repeat = digest2 == digest
    results.append(TestResult(
        description="Waveform hash deterministic on repeated calls",
        status=_format_status(success_repeat),
        details=f"digest={digest.hex()}"
    ))
    print(f"  {_format_status(success_repeat)} hash repeatability")

    different_digest = hasher.hash(b"quantonium+")
    success_diff = different_digest != digest
    results.append(TestResult(
        description="Waveform hash differentiates nearby inputs",
        status=_format_status(success_diff),
        details=f"alt_digest={different_digest.hex()}"
    ))
    print(f"  {_format_status(success_diff)} hash input sensitivity")

    return results


def main() -> None:
    results = {
        "block_cipher": [asdict(item) for item in validate_block_cipher()],
        "waveform_hash": [asdict(item) for item in validate_waveform_hash()],
    }

    total_failures = sum(1 for section in results.values() for item in section if not item["status"].startswith("✅"))

    with PAPER_METRICS_FILE.open("w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\n📄 Report written to {PAPER_METRICS_FILE}")
    if total_failures:
        print(f"⚠️ {total_failures} crypto checks failed.")
        raise SystemExit(1)
    print("🎉 All crypto checks passed.")


if __name__ == "__main__":
    main()
