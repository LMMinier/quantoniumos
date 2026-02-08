#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms.rft.core.theorem8_bootstrap_verification import (
    BootstrapResult,
    Theorem8BootstrapResult,
    bootstrap_ci,
)
from algorithms.rft.core.transform_theorems import k99


@dataclass(frozen=True)
class EnergyDump:
    energies: np.ndarray  # shape (num_cases, N)


def _k99_from_energies(energies: np.ndarray) -> int:
    total = float(np.sum(energies))
    if not np.isfinite(total) or total <= 0.0:
        return int(energies.size)
    order = np.argsort(energies)[::-1]
    cumulative = 0.0
    for idx, k in enumerate(order, start=1):
        cumulative += float(energies[k])
        if cumulative / total >= 0.99:
            return idx
    return int(energies.size)


def read_energy_dump_csv(path: Path, *, num_cases: int, n: int) -> EnergyDump:
    energies = np.zeros((num_cases, n), dtype=np.float64)

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "k", "energy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            case_id = int(row["case_id"])
            k = int(row["k"])
            energy = float(row["energy"])
            if 0 <= case_id < num_cases and 0 <= k < n:
                energies[case_id, k] = energy

    return EnergyDump(energies=energies)


def _parse_memh_words(path: Path) -> list[int]:
    words: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        token = line.split()[0]
        if token.lower().startswith("0x"):
            token = token[2:]
        words.append(int(token, 16) & 0xFFFF)
    return words


def _u16_to_s16(word: int) -> int:
    return word - 0x10000 if (word & 0x8000) else word


def load_inputs_from_memh(path: Path, *, num_cases: int, n: int) -> np.ndarray:
    words = _parse_memh_words(path)
    words_per_case = 2 * n
    expected = num_cases * words_per_case
    if len(words) < expected:
        raise ValueError(f"memh too short: got {len(words)} words, expected {expected}")

    x = np.empty((num_cases, n), dtype=np.complex128)
    scale = float((1 << 15) - 1)

    for case_id in range(num_cases):
        base = case_id * words_per_case
        re = np.array([_u16_to_s16(words[base + i]) for i in range(n)], dtype=np.float64) / scale
        im = np.array([_u16_to_s16(words[base + n + i]) for i in range(n)], dtype=np.float64) / scale
        x[case_id, :] = re + 1j * im

    return x


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Option A verifier: read RTL energy dump -> compute K99 + bootstrap stats",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("hardware/tb/theorem8_hw_energies.csv"),
        help="Energy dump CSV from tb_theorem8_energy_dump.sv",
    )
    ap.add_argument(
        "--memh",
        type=Path,
        default=Path("hardware/tb/theorem8_vectors_N8_q15.memh"),
        help="Input vectors memh used by the simulation",
    )
    ap.add_argument("--num-cases", type=int, default=100)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument(
        "--fail-on",
        choices=["warn", "error"],
        default="error",
        help="Exit nonzero on failure (error) or always zero (warn)",
    )
    args = ap.parse_args()

    dump = read_energy_dump_csv(args.csv, num_cases=args.num_cases, n=args.n)
    k99_rft = np.array([_k99_from_energies(dump.energies[i]) for i in range(args.num_cases)])

    # FFT baseline computed on host from the exact quantized inputs.
    x = load_inputs_from_memh(args.memh, num_cases=args.num_cases, n=args.n)
    # For the unitary DFT matrix F with exp(-i2πnk/N)/√N, coefficients are F^H x.
    # Under NumPy conventions, this equals √N * ifft(x).
    y_fft = np.fft.ifft(x, axis=1) * np.sqrt(args.n)
    k99_fft = np.array([k99(y_fft[i]) for i in range(args.num_cases)])

    improvement = k99_fft - k99_rft
    rng = np.random.default_rng(args.seed)
    improvement_ci: BootstrapResult = bootstrap_ci(
        improvement.astype(np.float64),
        n_bootstrap=args.bootstrap,
        ci_level=args.ci_level,
        rng=rng,
    )

    diff_std = float(np.std(improvement, ddof=1))
    cohens_d = float(np.mean(improvement) / diff_std) if diff_std > 0 else 0.0
    rft_wins = int(np.sum(k99_rft < k99_fft))

    result = Theorem8BootstrapResult(
        mean_improvement=float(np.mean(improvement)),
        improvement_ci=improvement_ci,
        mean_k99_rft=float(np.mean(k99_rft)),
        mean_k99_fft=float(np.mean(k99_fft)),
        cohens_d=cohens_d,
        rft_win_rate=rft_wins / int(args.num_cases),
        rft_wins=rft_wins,
        total_samples=int(args.num_cases),
        mean_k99_random=None,
        random_improvement=None,
        N=int(args.n),
        M=int(args.num_cases),
    )

    mean_rft = float(np.mean(k99_rft))
    p50_rft = float(np.percentile(k99_rft, 50))
    p90_rft = float(np.percentile(k99_rft, 90))

    print("=== Theorem 8 Option A (hardware RFT vs host FFT) ===")
    print(f"cases={args.num_cases} N={args.n}")
    print(f"K99_RFT mean={mean_rft:.3f} p50={p50_rft:.1f} p90={p90_rft:.1f}")
    print(f"K99_FFT mean={float(np.mean(k99_fft)):.3f}")
    print("\n=== Bootstrap (paired improvement Δ = K99(FFT) - K99(RFT)) ===")
    print(f"Δ_mean={result.mean_improvement:.6f}")
    print(f"CI{int(args.ci_level*100)}=({result.improvement_ci.ci_lower:.6f}, {result.improvement_ci.ci_upper:.6f})")
    print(f"cohens_d={result.cohens_d:.3f} win_rate={result.rft_win_rate:.3f}")
    print(f"theorem_holds={result.theorem_holds}")

    if not result.theorem_holds and args.fail_on == "error":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
