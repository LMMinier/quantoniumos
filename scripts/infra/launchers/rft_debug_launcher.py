# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
RFT Visualizer Debug Launcher with crash logging
This will help us see exactly what's going wrong
"""
import os
import sys
import traceback
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'rft_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("≡ƒÜÇ Starting RFT Visualizer Debug Launcher")
        
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        logger.info(f"Script dir: {script_dir}")
        logger.info(f"Root dir: {root_dir}")
        
        os.chdir(root_dir)
        logger.info(f"Changed to directory: {os.getcwd()}")
        
        # Add the root directory to Python path
        sys.path.insert(0, root_dir)
        sys.path.insert(0, os.path.join(root_dir, 'apps'))
        logger.info(f"Python path: {sys.path[:3]}...")
        
        # Check if the visualizer file exists
        viz_path = os.path.join(root_dir, 'apps', 'rft_visualizer.py')
        logger.info(f"Checking visualizer at: {viz_path}")
        logger.info(f"File exists: {os.path.exists(viz_path)}")
        
        # Try importing PyQt5 first
        logger.info("Testing PyQt5 import...")
        from PyQt5.QtWidgets import QApplication
        logger.info("Γ£à PyQt5 imported successfully")
        
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            logger.info("Creating new QApplication")
            app = QApplication(sys.argv)
        else:
            logger.info("Using existing QApplication")
        
        # Import the RFT visualizer module
        logger.info("Importing RFT visualizer...")
        import rft_visualizer
        logger.info("Γ£à RFT visualizer imported successfully")
        
        # Run it directly
        logger.info("Starting RFT visualizer main...")
        rft_visualizer.main()
        logger.info("Γ£à RFT visualizer completed")
        
    except Exception as e:
        logger.error(f"Γ¥î Error in debug launcher: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Also write to a simple text file for easy viewing
        error_file = os.path.join(log_dir, 'rft_crash.txt')
        with open(error_file, 'w') as f:
            f.write(f"RFT Visualizer Crash Report - {datetime.now()}\n")
            f.write(f"Error: {e}\n")
            f.write(f"Type: {type(e).__name__}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        print(f"≡ƒÆ╛ Crash report saved to: {error_file}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

