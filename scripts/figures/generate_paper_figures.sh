#!/bin/bash
# Master script to regenerate all validation data and MATLAB figures
# Run this before compiling the LaTeX paper

set -e

echo "=============================================="
echo " QuantoniumOS: Full Validation Pipeline"
echo "=============================================="

# Create output directories
mkdir -p figures/latex_data
mkdir -p figures

echo ""
echo "[1/5] Running Irrevocable Truths (Unitarity Verification)..."
python scripts/irrevocable_truths.py

echo ""
echo "[2/5] Running Variant Claims Verification..."
python scripts/verify_variant_claims.py

echo ""
echo "[3/5] Running Rate-Distortion Analysis..."
python scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv

echo ""
echo "[4/5] Running Wave Computer Benchmark..."
python tests/benchmarks/rft_wave_computer_demo.py --export figures/latex_data/wave_computer.csv

echo ""
echo "[5/5] Running Scaling Laws Analysis..."
python scripts/verify_scaling_laws.py

echo ""
echo "=============================================="
echo " Validation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Open MATLAB and navigate to figures/latex_data/"
echo "  2. Run: plot_rate_distortion.m"
echo "  3. Run: plot_wave_computer.m"
echo "  4. Compile LaTeX: cd docs/research && pdflatex THE_PHI_RFT_FRAMEWORK_PAPER.tex"
echo ""
echo "CSV exports:"
echo "  - figures/latex_data/rate_distortion.csv"
echo "  - figures/latex_data/wave_computer.csv"
echo ""
