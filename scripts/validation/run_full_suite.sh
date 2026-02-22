#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   ./scripts/run_full_suite.sh

if [ ! -d ".venv" ]; then
  echo "[!] .venv not found – creating..."
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -e .

echo "[*] Building native module..."
cd src/rftmw_native
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j"$(nproc)"
cd ../../..

echo "[*] Running competitor benchmarks..."
cd experiments/competitors

python benchmark_transforms_vs_fft.py \
  --sizes 256,1024 \
  --runs 3 \
  --output-dir ../../results/competitors

python benchmark_compression_vs_codecs.py \
  --datasets ascii random \
  --runs 2 \
  --output-dir ../../results/competitors

python benchmark_crypto_throughput.py \
  --sizes 1024,4096 \
  --runs 3 \
  --output-dir ../../results/competitors

echo "[*] Done. See results in results/competitors/"
