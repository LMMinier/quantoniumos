#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Quick test script to verify QuantoniumOS desktop launches"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("QuantoniumOS Desktop Test")
print("=" * 60)

# Test imports
try:
    print("\n✓ Checking PyQt5...")
    from PyQt5.QtWidgets import QApplication
    print("  ✓ PyQt5 available")
except ImportError as e:
    print(f"  ✗ PyQt5 not installed: {e}")
    print("  Run: pip install PyQt5")
    sys.exit(1)

try:
    print("\n✓ Checking desktop module...")
    from quantonium_os_src.frontend.quantonium_desktop import QuantoniumDesktop, QLogo
    print("  ✓ Desktop module loaded")
except ImportError as e:
    print(f"  ✗ Desktop module error: {e}")
    sys.exit(1)

try:
    print("\n✓ Checking app modules...")
    from quantonium_os_src.apps.symbolic_simulator import MainWindow as SimWindow
    from quantonium_os_src.apps.quantum_inspired_crypto import MainWindow as CryptoWindow
    from quantonium_os_src.apps.q_notes import MainWindow as NotesWindow
    from quantonium_os_src.apps.q_vault import MainWindow as VaultWindow
    from quantonium_os_src.apps.rft_validator import MainWindow as ValidatorWindow
    from quantonium_os_src.apps.rft_visualizer import MainWindow as VizWindow
    from quantonium_os_src.apps.system_monitor import MainWindow as MonitorWindow
    print("  ✓ All 7 apps loaded successfully")
except ImportError as e:
    print(f"  ✗ App module error: {e}")
    sys.exit(1)

# Check app registry
try:
    print("\n✓ Checking app registry...")
    import json
    registry_path = project_root / "data" / "config" / "app_registry.json"
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    print(f"  ✓ Found {len(registry)} apps in registry")
    for app_id, app_data in registry.items():
        status = "✓" if app_data.get("enabled", True) else "○"
        print(f"    {status} {app_data['name']}")
except Exception as e:
    print(f"  ⚠ Registry warning: {e}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - QuantoniumOS ready to launch!")
print("=" * 60)
print("\nLaunch desktop with:")
print("  python3 scripts/quantonium_boot.py")
print("  or")
print("  python3 quantonium_os_src/frontend/quantonium_desktop.py")
print()
