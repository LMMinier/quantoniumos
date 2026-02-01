#!/usr/bin/env python3
"""Post-quantum security notes (non-claiming).

This script existed in older history as a "post-quantum analysis" and made
strong claims. The current repository position is explicit:

- No post-quantum security claim is made for the custom constructions.
- No reduction to SIS/LWE (or other standard hardness assumptions) is claimed.

What this script does now:
- Reports the *generic* Grover impact on brute-force key search.
- Reminds where the formal "what you can and cannot claim" theorems live.

Usage:
  python tests/crypto/scripts/post_quantum_analysis.py
"""

from __future__ import annotations

import math


def grover_effective_security_bits(classical_key_bits: int) -> int:
    # Grover yields quadratic speedup for unstructured search.
    return math.ceil(classical_key_bits / 2)


def main() -> int:
    print("Post-quantum notes (non-claiming)")
    print("- No IND-CPA/IND-CCA claims for custom schemes")
    print("- No reduction to SIS/LWE claimed for structured RFT-derived matrices")
    print()

    for classical_bits in (128, 192, 256, 384):
        pq_bits = grover_effective_security_bits(classical_bits)
        print(f"Grover impact: {classical_bits}-bit key -> ~{pq_bits}-bit security")

    print()
    print("Formal reduction boundaries and proof statements:")
    print("- docs/proofs/THEOREMS_RFT_IRONCLAD.md (Theorem 7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
