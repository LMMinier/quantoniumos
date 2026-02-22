#!/bin/bash
set -e

# 1. Install Dependencies
echo "Installing Python dependencies..."
pip install pandas matplotlib scipy numpy > /dev/null 2>&1

# 2. Set Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Generate Data
echo "Generating Unitarity Data for 7 Variants..."
python3 scripts/irrevocable_truths.py --export figures/latex_data/unitarity_errors.csv

echo "Generating Rate-Distortion Data..."
python3 scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv

echo "Generating Wave Computer Data..."
python3 tests/benchmarks/rft_wave_computer_demo.py --export figures/latex_data/wave_computer.csv

# 4. Render Figures
echo "Rendering PDF Figures..."
python3 scripts/generate_pdf_figures_for_latex.py

echo "---------------------------------------------------"
echo "✅ Figures Generated Successfully:"
echo "   - figures/rft_unitarity_errors.pdf (7 Variants Test)"
echo "   - figures/rft_rate_distortion_matlab.pdf"
echo "   - figures/rft_wave_computer_matlab.pdf"
