#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos

import os
import platform
import subprocess
import sys
from pathlib import Path

# Define project structure
ROOT_DIR = Path(__file__).parent
MAKEFILE_DIR = ROOT_DIR / "algorithms" / "rft" / "kernels"
OUTPUT_DIR = MAKEFILE_DIR / "compiled"

# Define library names
if platform.system() == "Windows":
    LIB_NAME = "libquantum_symbolic.dll"
else:
    LIB_NAME = "libquantum_symbolic.so"

FINAL_LIB_PATH = OUTPUT_DIR / LIB_NAME

def has_command(command):
    """Check if a command exists on the system's PATH."""
    try:
        subprocess.check_output([command, "--version"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def run_command(command, cwd):
    """Runs a command in a given directory and streams its output."""
    print(f"[{cwd}]$ {command}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def build_with_make():
    """Builds the library using the Makefile."""
    if not (MAKEFILE_DIR / "Makefile").exists():
        print(f"Error: Makefile not found in {MAKEFILE_DIR}")
        sys.exit(1)
    
    run_command("make clean", cwd=MAKEFILE_DIR)
    run_command("make all", cwd=MAKEFILE_DIR)

def build_windows():
    """Builds the DLL on Windows."""
    print("--- Building for Windows ---")
    
    # Prefer `make` if available (e.g., from Git Bash, MSYS2, etc.)
    if has_command("make"):
        print("Found 'make' on PATH. Attempting to build with Makefile...")
        try:
            build_with_make()
            return
        except subprocess.CalledProcessError as e:
            print(f"Makefile build failed with error: {e}. Falling back to MSBuild if possible.")
            # Fall through to MSBuild if make fails
    
    # Attempt to find MSBuild.exe if make is not available or fails
    vswhere_path = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere_path.exists():
        print("Error: vswhere.exe not found. Is Visual Studio installed?")
        print("Neither 'make' nor Visual Studio Build Tools could be found.")
        sys.exit(1)

    try:
        install_path_bytes = subprocess.check_output([str(vswhere_path), "-latest", "-property", "installationPath"])
        vs_install_path = Path(install_path_bytes.decode('utf-8').strip())
        msbuild_path = vs_install_path / "MSBuild/Current/Bin/MSBuild.exe"

        if not msbuild_path.exists():
             # Fallback for older VS versions
            msbuild_path = vs_install_path / "MSBuild/15.0/Bin/MSBuild.exe"

        if not msbuild_path.exists():
            print("Error: MSBuild.exe not found in Visual Studio installation path.")
            sys.exit(1)
            
    except subprocess.CalledProcessError:
        print("Error: Failed to query Visual Studio installation path via vswhere.")
        sys.exit(1)

    # This part is speculative as we haven't confirmed a .sln file.
    # If a .sln file were added, this would be the logic.
    solution_file = MAKEFILE_DIR / "QuantoniumOS_kernels.sln" # Hypothetical name
    if not solution_file.exists():
        print(f"Error: Visual Studio Solution file not found at {solution_file}")
        print("The Makefile build failed and no .sln file is available for fallback.")
        sys.exit(1)

    print(f"Found MSBuild at: {msbuild_path}")
    print(f"Building solution: {solution_file}")

    run_command(f'"{msbuild_path}" {solution_file} /t:Clean', cwd=MAKEFILE_DIR)
    run_command(f'"{msbuild_path}" {solution_file} /p:Configuration=Release /p:Platform=x64', cwd=MAKEFILE_DIR)


def main():
    """Main build function."""
    print(f"Starting build process for {platform.system()}...")
    
    if not MAKEFILE_DIR.exists():
        print(f"Error: Kernel source directory not found at {MAKEFILE_DIR}")
        sys.exit(1)

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        if platform.system() == "Linux":
            build_with_make()
        elif platform.system() == "Windows":
            build_windows()
        else:
            print(f"Unsupported OS: {platform.system()}")
            sys.exit(1)
        
        print("\n--- Build Successful ---")
        print(f"Native kernel located at: {FINAL_LIB_PATH.resolve()}")

    except subprocess.CalledProcessError as e:
        print(f"\n--- Build Failed ---")
        print(f"An error occurred during the build process: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
