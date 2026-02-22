# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
Isolated launcher for RFT Visualizer to prevent QApplication conflicts
"""
import sys
import os
import subprocess
import time

def launch_rft_visualizer():
    """Launch RFT visualizer in completely isolated process"""
    try:
        # Get the absolute path to the RFT visualizer
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rft_path = os.path.join(script_dir, "rft_visualizer.py")
        
        # Use subprocess with complete isolation
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
        
        # Launch with CREATE_NEW_CONSOLE to completely isolate the process
        if sys.platform == "win32":
            # On Windows, use CREATE_NEW_PROCESS_GROUP for isolation
            process = subprocess.Popen(
                [sys.executable, rft_path],
                cwd=script_dir,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # On Unix-like systems
            process = subprocess.Popen(
                [sys.executable, rft_path],
                cwd=script_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        
        print(f"≡ƒîè RFT Visualizer launched with PID: {process.pid}")
        return True
        
    except Exception as e:
        print(f"Γ¥î Failed to launch RFT Visualizer: {e}")
        return False

if __name__ == "__main__":
    launch_rft_visualizer()

