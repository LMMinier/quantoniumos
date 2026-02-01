#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
COMPREHENSIVE QUANTONIUMOS ROUTE & DEPENDENCY CHECKER
=====================================================
Checks all routes, dependencies, and connectors to diagnose Quantum Simulator issues
"""

import sys
import os
import importlib.util
import subprocess
from pathlib import Path

def check_python_environment():
    print("=" * 60)
    print("PYTHON ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path entries:")
    for i, path in enumerate(sys.path):
        print(f"  [{i}] {path}")

def check_pyqt5_installation():
    print("\n" + "=" * 60)
    print("PYQT5 INSTALLATION CHECK")
    print("=" * 60)
    
    try:
        import PyQt5
        print(f"✅ PyQt5 installed: {PyQt5.__file__}")
        
        from PyQt5 import QtCore, QtWidgets, QtGui
        print(f"✅ QtCore version: {QtCore.PYQT_VERSION_STR}")
        print(f"✅ Qt version: {QtCore.QT_VERSION_STR}")
        
        # Test QApplication creation
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            test_app = QApplication([])
            print("✅ QApplication can be created")
            test_app.quit()
        else:
            print("✅ QApplication already exists")
            
    except Exception as e:
        print(f"❌ PyQt5 issue: {e}")

def check_matplotlib_installation():
    print("\n" + "=" * 60)
    print("MATPLOTLIB INSTALLATION CHECK") 
    print("=" * 60)
    
    try:
        import matplotlib
        print(f"✅ Matplotlib installed: {matplotlib.__version__}")
        
        import matplotlib.pyplot as plt
        print("✅ pyplot available")
        
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        print("✅ Qt5Agg backend available")
        
    except Exception as e:
        print(f"❌ Matplotlib issue: {e}")

def check_numpy_scipy():
    print("\n" + "=" * 60)
    print("NUMPY/SCIPY CHECK")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        
    except Exception as e:
        print(f"❌ NumPy/SciPy issue: {e}")

def check_assembly_bindings():
    print("\n" + "=" * 60)
    print("ASSEMBLY BINDINGS CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    binding_paths = [
        base_dir / "src" / "assembly" / "python_bindings",
        base_dir / "src" / "ASSEMBLY" / "python_bindings"
    ]
    
    for binding_path in binding_paths:
        if binding_path.exists():
            print(f"✅ Found bindings directory: {binding_path}")
            
            # Check for key files
            key_files = ["unitary_rft.py", "quantum_symbolic_engine.py"]
            for key_file in key_files:
                file_path = binding_path / key_file
                if file_path.exists():
                    print(f"  ✅ {key_file} found")
                else:
                    print(f"  ❌ {key_file} MISSING")
            
            # Try to import
            sys.path.insert(0, str(binding_path))
            try:
                from unitary_rft import UnitaryRFT
                print("  ✅ UnitaryRFT import successful")
            except Exception as e:
                print(f"  ❌ UnitaryRFT import failed: {e}")
                
        else:
            print(f"❌ Bindings directory not found: {binding_path}")

def check_simulator_file_structure():
    print("\n" + "=" * 60)
    print("QUANTUM SIMULATOR FILE STRUCTURE CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    simulator_path = base_dir / "src" / "apps" / "symbolic_simulator.py"
    
    print(f"Checking: {simulator_path}")
    
    if simulator_path.exists():
        print("✅ symbolic_simulator.py exists")
        
        # Check file size
        size = simulator_path.stat().st_size
        print(f"✅ File size: {size} bytes")
        
        # Check if it's readable
        try:
            with open(simulator_path, 'r') as f:
                first_lines = f.readlines()[:10]
            print("✅ File is readable")
            print("First few lines:")
            for i, line in enumerate(first_lines):
                print(f"  {i+1}: {line.strip()}")
        except Exception as e:
            print(f"❌ File read error: {e}")
            
    else:
        print(f"❌ symbolic_simulator.py NOT FOUND at {simulator_path}")

def check_simulator_imports():
    print("\n" + "=" * 60)
    print("QUANTUM SIMULATOR IMPORT CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    simulator_path = base_dir / "src" / "apps" / "symbolic_simulator.py"
    
    if not simulator_path.exists():
        print("❌ Cannot check imports - file doesn't exist")
        return
        
    try:
        # Add necessary paths
        apps_dir = str(base_dir / "src" / "apps")
        if apps_dir not in sys.path:
            sys.path.insert(0, apps_dir)
            
        # Try to load the module
        spec = importlib.util.spec_from_file_location("symbolic_simulator", str(simulator_path))
        if spec is None:
            print("❌ Cannot create module spec")
            return
            
        print("✅ Module spec created")
        
        simulator_module = importlib.util.module_from_spec(spec)
        print("✅ Module object created")
        
        # Execute the module to load it
        spec.loader.exec_module(simulator_module)
        print("✅ Module executed successfully")
        
        # Check for main class
        if hasattr(simulator_module, 'RFTQuantumSimulator'):
            print("✅ RFTQuantumSimulator class found")
            
            # Try to create instance (with QApp)
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
                print("✅ Created QApplication for testing")
            
            simulator = simulator_module.RFTQuantumSimulator()
            print("✅ RFTQuantumSimulator instance created successfully")
            
        else:
            print("❌ RFTQuantumSimulator class NOT found")
            available = [attr for attr in dir(simulator_module) if not attr.startswith('_')]
            print(f"Available attributes: {available}")
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()

def check_desktop_launcher():
    print("\n" + "=" * 60)
    print("DESKTOP LAUNCHER CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    desktop_path = base_dir / "src" / "frontend" / "quantonium_desktop.py"
    
    if desktop_path.exists():
        print(f"✅ Desktop file found: {desktop_path}")
        
        # Check the launch_symbolic_simulator method
        try:
            with open(desktop_path, 'r') as f:
                content = f.read()
                
            if 'launch_symbolic_simulator' in content:
                print("✅ launch_symbolic_simulator method found")
                
                # Extract the method
                lines = content.split('\n')
                in_method = False
                method_lines = []
                
                for line in lines:
                    if 'def launch_symbolic_simulator' in line:
                        in_method = True
                        method_lines.append(line)
                    elif in_method:
                        if line.strip().startswith('def ') and 'launch_symbolic_simulator' not in line:
                            break
                        method_lines.append(line)
                        
                print("Method implementation:")
                for line in method_lines[:20]:  # First 20 lines
                    print(f"  {line}")
                    
            else:
                print("❌ launch_symbolic_simulator method NOT found")
                
        except Exception as e:
            print(f"❌ Error reading desktop file: {e}")
    else:
        print(f"❌ Desktop file NOT found: {desktop_path}")

def check_subprocess_execution():
    print("\n" + "=" * 60)
    print("SUBPROCESS EXECUTION TEST")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    simulator_path = base_dir / "src" / "apps" / "symbolic_simulator.py"
    
    if not simulator_path.exists():
        print("❌ Cannot test subprocess - simulator file missing")
        return
        
    try:
        print("Testing subprocess execution...")
        
        # Test if Python can execute the file
        result = subprocess.run(
            [sys.executable, str(simulator_path)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ Subprocess execution successful")
        else:
            print("❌ Subprocess execution failed")
            
    except subprocess.TimeoutExpired:
        print("⚠️ Subprocess timed out (this might be normal for GUI apps)")
    except Exception as e:
        print(f"❌ Subprocess test error: {e}")

def main():
    print("🔍 QUANTONIUMOS COMPREHENSIVE SYSTEM CHECK")
    print("Diagnosing Quantum Simulator connectivity issues...")
    
    check_python_environment()
    check_pyqt5_installation()
    check_matplotlib_installation()
    check_numpy_scipy()
    check_assembly_bindings()
    check_simulator_file_structure()
    check_simulator_imports()
    check_desktop_launcher()
    check_subprocess_execution()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("Check the output above for any ❌ errors that need to be fixed.")
    print("All components should show ✅ for the Quantum Simulator to work properly.")

if __name__ == '__main__':
    main()
