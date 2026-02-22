# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import os
import sys
import subprocess

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def main():
    # Install deps
    run_command("pip install pandas matplotlib scipy numpy")
    
    # Set python path
    os.environ["PYTHONPATH"] = os.getcwd()
    
    # Generate Data
    run_command("python3 scripts/irrevocable_truths.py --export figures/latex_data/unitarity_errors.csv")
    run_command("python3 scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv")
    run_command("python3 tests/benchmarks/rft_wave_computer_demo.py --export figures/latex_data/wave_computer.csv")
    
    # Generate Figures
    run_command("python3 scripts/generate_pdf_figures_for_latex.py")
    
    print("Figures generated successfully.")

if __name__ == "__main__":
    main()