"""Quick SIS security estimator (heuristic).

Outputs JSON with BKZ block size and Core-SVP costs using the Chen–Nguyen
delta(b) formula. Intended for reproducibility of Appendix B / Theorem 7.5.

Model assumptions:
- Lattice dimension is m (rows of A)
- det(Λ) = q^n for SIS
- Target norm bound is β
- Cost model: classical sieving ~ 2^(0.292 b), quantum sieving ~ 2^(0.2655 b)
- Chen–Nguyen delta(b) approximation; not calibrated for very large b

This script is not a replacement for lattice-estimator; it is a minimal,
transparent reproduction of the calculations reported in the docs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


def delta_from_b(b: int) -> float:
    """Chen–Nguyen root-Hermite factor approximation.

    For b < 50, return LLL baseline (heuristic guard).
    """
    if b < 50:
        return 1.0219
    b_float = float(b)
    return ((math.pi * b_float) ** (1.0 / b_float) * b_float / (2.0 * math.pi * math.e)) ** (1.0 / (2.0 * (b_float - 1.0)))


def required_block_size(n: int, m: int, q: int, beta: float) -> tuple[int, float, float]:
    """Binary search for the smallest BKZ block size b such that
    delta(b)^(m-1) * q^(n/m) <= beta.

    Returns (b, delta(b), output_len). Search upper bound is 10_000; if
    the condition is not met within that range, returns the upper bound
    (out-of-domain / upper-bound style).
    """
    det_root = q ** (n / m)  # q^(n/m)
    low, high = 50, 10_000
    while high - low > 1:
        mid = (low + high) // 2
        delta = delta_from_b(mid)
        output_len = (delta ** (m - 1)) * det_root
        if output_len > beta:
            low = mid
        else:
            high = mid
    b = high
    delta = delta_from_b(b)
    output_len = (delta ** (m - 1)) * det_root
    return b, delta, output_len


def estimate(n: int, m: int, q: int, beta: float) -> dict:
    det_root = q ** (n / m)
    delta_needed = (beta / det_root) ** (1 / (m - 1))
    b, delta, output_len = required_block_size(n, m, q, beta)
    cost_classical_bits = 0.292 * b
    cost_quantum_bits = 0.2655 * b
    return {
        "n": n,
        "m": m,
        "q": q,
        "beta": beta,
        "det_root": det_root,
        "delta_needed": delta_needed,
        "delta_bkz": delta,
        "output_len": output_len,
        "bkz_block": b,
        "cost_bits_classical": cost_classical_bits,
        "cost_bits_quantum": cost_quantum_bits,
        "model": {
            "delta": "Chen–Nguyen heuristic",
            "cost": "Core-SVP sieving 0.292*b (classical), 0.2655*b (quantum)",
            "notes": "Not calibrated for very large b; treat as heuristic upper-bound style"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic SIS security estimator")
    parser.add_argument("--n", type=int, default=512, help="SIS dimension n")
    parser.add_argument("--m", type=int, default=1024, help="SIS width m")
    parser.add_argument("--q", type=int, default=3329, help="Modulus q")
    parser.add_argument("--beta", type=float, default=100.0, help="Norm bound beta")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    result = estimate(args.n, args.m, args.q, args.beta)
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
