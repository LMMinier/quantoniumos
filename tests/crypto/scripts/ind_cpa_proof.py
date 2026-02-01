#!/usr/bin/env python3
"""IND-CPA game harness (attack demonstration, not a proof).

This file previously existed in repo history as a claimed "IND-CPA proof".
The current implementation is intentionally conservative:

- It runs a standard IND-CPA experiment against the current
  `EnhancedRFTCryptoV2.encrypt_aead` construction.
- It implements a simple *polynomial-time distinguisher* that exploits
  per-message ECB-style block encryption (repeated plaintext blocks map to
  repeated ciphertext blocks within the same message).

If this distinguisher wins with non-negligible advantage, the scheme is NOT
IND-CPA as implemented.

Usage:
  python tests/crypto/scripts/ind_cpa_proof.py --trials 2000

Notes:
- This script provides *evidence of insecurity* under IND-CPA. It does not
  provide a reduction or a security proof.
- If you want IND-CPA / IND-CCA2, the usual fix is to use a standard AEAD
  (AES-GCM / ChaCha20-Poly1305) or switch the construction to a proven mode
  (e.g., CTR + MAC with unique nonce/counter per block).
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass

from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2


_BLOCK = 16
_HEADER = 1 + 16
_TAG = 32


@dataclass(frozen=True)
class Result:
    trials: int
    success_rate: float
    advantage: float


def _parse_ciphertext_blocks(ct: bytes) -> list[bytes]:
    if len(ct) < _HEADER + _TAG:
        raise ValueError("Ciphertext too short")
    body = ct[_HEADER:-_TAG]
    if len(body) % _BLOCK != 0:
        raise ValueError("Ciphertext body not block-aligned")
    return [body[i : i + _BLOCK] for i in range(0, len(body), _BLOCK)]


def repeated_block_distinguisher(challenge_ct: bytes) -> int:
    """Return guess bit b' for IND-CPA game.

    Distinguisher:
    - If ciphertext block 0 == block 1 then guess message with repeated blocks.
    """
    blocks = _parse_ciphertext_blocks(challenge_ct)
    if len(blocks) < 2:
        return secrets.randbelow(2)
    return 0 if blocks[0] == blocks[1] else 1


def run_ind_cpa_game(*, trials: int, associated_data: bytes = b"ind-cpa") -> Result:
    """Run an IND-CPA game and estimate adversary advantage."""

    # Fresh master key for the oracle.
    master_key = secrets.token_bytes(32)
    oracle = EnhancedRFTCryptoV2(master_key)

    correct = 0

    # Choose fixed (m0, m1) that differ only by intra-message repetition.
    # m0: [A, A]  (two identical 16-byte blocks)
    # m1: [A, B]  (two different 16-byte blocks)
    block_a = b"A" * _BLOCK
    block_b = b"B" * _BLOCK
    m0 = block_a + block_a
    m1 = block_a + block_b

    for _ in range(trials):
        b = secrets.randbelow(2)
        mb = m0 if b == 0 else m1
        ct = oracle.encrypt_aead(mb, associated_data)
        b_guess = repeated_block_distinguisher(ct)
        if b_guess == b:
            correct += 1

    success_rate = correct / trials
    advantage = abs(success_rate - 0.5)
    return Result(trials=trials, success_rate=success_rate, advantage=advantage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()

    res = run_ind_cpa_game(trials=args.trials)
    print("IND-CPA game (distinguisher = repeated-block test)")
    print(f"trials         : {res.trials}")
    print(f"success_rate   : {res.success_rate:.4f}")
    print(f"advantage      : {res.advantage:.4f}")

    if res.advantage > 0.05:
        print("assessment     : NOT IND-CPA (attack succeeds with clear advantage)")
    else:
        print("assessment     : inconclusive at this sample size")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
