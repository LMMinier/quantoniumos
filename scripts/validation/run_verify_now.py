#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import and run
from scripts import verify_ascii_bottleneck
verify_ascii_bottleneck.run_ascii_bottleneck_test()
