# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
RFT Scientific Validation Visualizer - QuantoniumOS
===================================================
This is a bridge module that loads the actual implementation from launch_rft_validation.py
for compatibility with the rest of the system.
"""

import os
import sys

# Import the actual implementation from the launcher file
try:
    from launch_rft_validation import RFTValidatorApp, main
    
    # Re-export the main classes
    __all__ = ['RFTValidatorApp']
    
    if __name__ == "__main__":
        # If this file is run directly, start the launcher
        main()
except ImportError as e:
    print(f"Error importing RFT Scientific Validator: {e}")
    sys.exit(1)

