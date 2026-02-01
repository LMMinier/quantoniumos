# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QuantoniumOS Import Management System
====================================
Centralized import management with proper path resolution.
Handles all cross-module imports and dependencies.
"""

import sys
import importlib
import warnings
from pathlib import Path
from typing import Optional, Any, Dict, List

from .paths import paths, setup_python_path


class QuantoniumImports:
    """
    Centralized import management for QuantoniumOS.
    Handles path setup and safe importing of all modules.
    """
    
    def __init__(self):
        self._imported_modules: Dict[str, Any] = {}
        self._setup_complete = False
        
    def setup_paths(self) -> None:
        """Setup all required paths for imports"""
        if self._setup_complete:
            return
            
        setup_python_path()
        self._setup_complete = True
        
    def safe_import(self, module_name: str, package: Optional[str] = None) -> Optional[Any]:
        """
        Safely import a module with error handling
        
        Args:
            module_name: Name of module to import
            package: Package name for relative imports
            
        Returns:
            Imported module or None if failed
        """
        if not self._setup_complete:
            self.setup_paths()
            
        cache_key = f"{package}.{module_name}" if package else module_name
        
        if cache_key in self._imported_modules:
            return self._imported_modules[cache_key]
            
        try:
            if package:
                module = importlib.import_module(module_name, package)
            else:
                module = importlib.import_module(module_name)
                
            self._imported_modules[cache_key] = module
            return module
            
        except ImportError as e:
            warnings.warn(f"Failed to import {module_name}: {e}", ImportWarning)
            return None
            
    def import_kernel(self) -> Optional[Any]:
        """
        Import RFT kernel with multiple fallback strategies
        
        Returns:
            RFT kernel module or None if failed
        """
        if 'kernel' in self._imported_modules:
            return self._imported_modules['kernel']
            
        # Try multiple import strategies
        strategies = [
            # Strategy 1: Direct import from bindings
            lambda: self.safe_import('unitary_rft'),
            
            # Strategy 2: Import with explicit path setup
            lambda: self._import_kernel_with_path(),
            
            # Strategy 3: Import from ASSEMBLY
            lambda: self._import_kernel_from_assembly(),
        ]
        
        for strategy in strategies:
            try:
                kernel = strategy()
                if kernel:
                    self._imported_modules['kernel'] = kernel
                    print(f"✓ RFT Kernel loaded successfully")
                    return kernel
            except Exception as e:
                continue
                
        warnings.warn("Could not load RFT kernel with any strategy", ImportWarning)
        return None
        
    def _import_kernel_with_path(self) -> Optional[Any]:
        """Import kernel with explicit path setup"""
        bindings_path = str(paths.python_bindings)
        if bindings_path not in sys.path:
            sys.path.insert(0, bindings_path)
        return self.safe_import('unitary_rft')
        
    def _import_kernel_from_assembly(self) -> Optional[Any]:
        """Import kernel from ASSEMBLY directory"""
        assembly_path = str(paths.assembly)
        if assembly_path not in sys.path:
            sys.path.insert(0, assembly_path)
        return self.safe_import('python_bindings.unitary_rft')
        
    def import_core_module(self, module_name: str) -> Optional[Any]:
        """
        Import a core quantum module
        
        Args:
            module_name: Name of core module to import
            
        Returns:
            Core module or None if failed
        """
        full_name = f"core.{module_name}" if not module_name.startswith('core.') else module_name
        return self.safe_import(full_name)
        
    def import_app(self, app_name: str) -> Optional[Any]:
        """
        Import an application module
        
        Args:
            app_name: Name of app to import
            
        Returns:
            App module or None if failed
        """
        # Try as direct module first
        app_module = self.safe_import(f"apps.{app_name}")
        if app_module:
            return app_module
            
        # Try as file name
        app_module = self.safe_import(f"{app_name}")
        if app_module:
            return app_module
            
        return None
        
    def import_frontend_component(self, component_name: str) -> Optional[Any]:
        """
        Import a frontend component
        
        Args:
            component_name: Name of frontend component
            
        Returns:
            Frontend component or None if failed
        """
        return self.safe_import(f"frontend.{component_name}")
        
    def get_import_status(self) -> Dict[str, bool]:
        """
        Get status of key imports
        
        Returns:
            Dictionary of import statuses
        """
        status = {}
        
        # Test kernel import
        kernel = self.import_kernel()
        status['kernel'] = kernel is not None
        
        # Test core modules
        core_modules = ['topological_quantum_kernel', 'working_quantum_kernel']
        for module in core_modules:
            mod = self.import_core_module(module)
            status[f'core.{module}'] = mod is not None
            
        # Test key apps
        key_apps = ['quantum_inspired_crypto', 'q_notes', 'symbolic_simulator']
        for app in key_apps:
            mod = self.import_app(app)
            status[f'app.{app}'] = mod is not None
            
        return status
        
    def reload_module(self, module_name: str) -> Optional[Any]:
        """
        Reload a module (useful for development)
        
        Args:
            module_name: Name of module to reload
            
        Returns:
            Reloaded module or None if failed
        """
        if module_name in self._imported_modules:
            try:
                module = self._imported_modules[module_name]
                reloaded = importlib.reload(module)
                self._imported_modules[module_name] = reloaded
                return reloaded
            except Exception as e:
                warnings.warn(f"Failed to reload {module_name}: {e}", ImportWarning)
                
        return self.safe_import(module_name)
        
    def clear_cache(self) -> None:
        """Clear import cache"""
        self._imported_modules.clear()
        
    def validate_imports(self) -> bool:
        """
        Validate that all critical imports work
        
        Returns:
            True if all critical imports successful
        """
        status = self.get_import_status()
        critical_imports = ['kernel']
        
        failed = [name for name in critical_imports if not status.get(name, False)]
        
        if failed:
            print(f"✗ Critical imports failed: {failed}")
            return False
        else:
            print("✓ All critical imports successful")
            return True


# Global instance
imports = QuantoniumImports()


def setup_quantonium_imports() -> bool:
    """
    Setup all QuantoniumOS imports
    
    Returns:
        True if setup successful
    """
    imports.setup_paths()
    return imports.validate_imports()


def get_kernel():
    """Get RFT kernel (convenience function)"""
    return imports.import_kernel()


def get_core_module(name: str):
    """Get core module (convenience function)"""
    return imports.import_core_module(name)


def get_app(name: str):
    """Get app module (convenience function)"""
    return imports.import_app(name)


if __name__ == "__main__":
    # Test import system
    print("Testing QuantoniumOS Import System:")
    print("=" * 40)
    
    # Setup imports
    setup_success = setup_quantonium_imports()
    print(f"Setup successful: {setup_success}")
    
    # Show import status
    print("\nImport Status:")
    status = imports.get_import_status()
    for name, success in status.items():
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}")
        
    # Show paths
    print(f"\nPython Path Entries:")
    for i, path in enumerate(sys.path[:10]):  # Show first 10
        print(f"  {i}: {path}")
