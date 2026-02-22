# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
import subprocess
import sys
import os
from datetime import datetime

# --- Configuration ---
VALIDATION_DIR = "/workspaces/quantoniumos/validation"
CRYPTO_DIR = os.path.join(VALIDATION_DIR, "crypto_benchmarks")
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
REPORT_FILE = os.path.join(RESULTS_DIR, f"MASTER_VALIDATION_REPORT_{int(datetime.now().timestamp())}.md")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of tests to run, in logical order
# (Path, Description)
TEST_SUITE = [
    (os.path.join(VALIDATION_DIR, "test_rft_invariants.py"), "Core RFT Mathematical Invariants"),
    (os.path.join(VALIDATION_DIR, "test_c_rft.py"), "RFT C Kernel vs. Python Implementation"),
    (os.path.join(VALIDATION_DIR, "test_rft_vertex_codec_roundtrip.py"), "Symbolic Vertex Codec Round-trip"),
    (os.path.join(VALIDATION_DIR, "test_rft_hybrid_codec_e2e.py"), "Hybrid (Lossy) Codec End-to-End"),
    (os.path.join(CRYPTO_DIR, "cipher_validation.py"), "Cryptographic Validation Vectors"),
]

def run_test(script_path, description):
    """Runs a single pytest or python script and returns the result."""
    if not os.path.exists(script_path):
        return {
            "description": description,
            "script": os.path.basename(script_path),
            "status": "MISSING",
            "output": f"Test script not found at {script_path}",
        }

    print(f"▶️  Running: {description}...")
    
    # For python scripts in crypto_benchmarks, run them directly. For others, use pytest.
    if "crypto_benchmarks" in script_path or "test_c_rft.py" in script_path:
        command = [sys.executable, script_path]
    else:
        command = [sys.executable, "-m", "pytest", script_path]

    cwd = os.path.dirname(script_path)

    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(VALIDATION_DIR), "src")
    existing_path = env.get("PYTHONPATH", "")
    if existing_path:
        if src_path not in existing_path.split(":"):
            env["PYTHONPATH"] = f"{src_path}:{existing_path}"
    else:
        env["PYTHONPATH"] = src_path

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout
            cwd=cwd,
            env=env,
        )

        if process.returncode == 0:
            status = "✅ PASS"
            print(f"✅ PASS: {description}")
        else:
            status = "❌ FAIL"
            print(f"❌ FAIL: {description}")

        return {
            "description": description,
            "script": os.path.basename(script_path),
            "status": status,
            "output": process.stdout + process.stderr,
        }
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return {
            "description": description,
            "script": os.path.basename(script_path),
            "status": "⏰ TIMEOUT",
            "output": "The test took longer than 5 minutes and was terminated.",
        }
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return {
            "description": description,
            "script": os.path.basename(script_path),
            "status": "💥 ERROR",
            "output": str(e),
        }

def generate_report(results):
    """Generates a markdown report from the test results."""
    print(f"\n📝 Generating report at {REPORT_FILE}...")
    
    summary = "## 🚀 QuantoniumOS Master Validation Report\n\n"
    summary += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    summary += "### Summary\n\n| Status | Test Description | Script |\n"
    summary += "|:------:|:-----------------|:-------|\n"
    
    for result in results:
        summary += f"| {result['status']} | {result['description']} | `{result['script']}` |\n"
        
    details = "\n### Detailed Logs\n\n"
    for result in results:
        details += f"<details>\n<summary><strong>{result['status']} - {result['description']}</strong></summary>\n\n"
        details += f"**Script:** `{result['script']}`\n\n"
        details += "```\n"
        details += result['output'].strip() + "\n"
        details += "```\n\n"
        details += "</details>\n\n"
        
    with open(REPORT_FILE, "w") as f:
        f.write(summary + details)
        
    print(f"🎉 Report generation complete.")

def main():
    """Main function to run all tests and generate a report."""
    all_results = []
    for script, description in TEST_SUITE:
        result = run_test(script, description)
        all_results.append(result)
        
    generate_report(all_results)
    print(f"\nAll tests complete. See the full report at: {REPORT_FILE}")

if __name__ == "__main__":
    main()
