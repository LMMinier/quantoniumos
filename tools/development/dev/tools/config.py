# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QuantoniumOS Configuration Management
====================================
Centralized configuration system for all QuantoniumOS components.
Handles app registry, build settings, and runtime configuration.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from .paths import paths


@dataclass
class AppConfig:
    """Configuration for a single application"""
    name: str
    module: str
    class_name: str
    icon: str
    description: str
    enabled: bool = True
    args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = {}


@dataclass 
class BuildConfig:
    """Build system configuration"""
    kernel_source_dir: str
    kernel_build_dir: str
    python_bindings_dir: str
    target_name: str
    compiler: str
    compiler_flags: List[str]
    
    def __post_init__(self):
        if isinstance(self.compiler_flags, str):
            self.compiler_flags = self.compiler_flags.split()


class QuantoniumConfig:
    """
    Centralized configuration manager for QuantoniumOS
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or paths.project_root
        self._app_registry: Optional[Dict[str, AppConfig]] = None
        self._build_config: Optional[BuildConfig] = None
        self._runtime_config: Dict[str, Any] = {}
        
    def get_app_registry(self) -> Dict[str, AppConfig]:
        """
        Get application registry
        
        Returns:
            Dictionary of app configurations
        """
        if self._app_registry is None:
            self._load_app_registry()
        return self._app_registry
        
    def _load_app_registry(self) -> None:
        """Load app registry from file or create default"""
        registry_file = self.config_dir / "app_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                self._app_registry = {
                    name: AppConfig(**config) 
                    for name, config in data.get('apps', {}).items()
                }
            except Exception as e:
                print(f"Warning: Could not load app registry: {e}")
                self._app_registry = self._create_default_app_registry()
        else:
            self._app_registry = self._create_default_app_registry()
            self.save_app_registry()
            
    def _create_default_app_registry(self) -> Dict[str, AppConfig]:
        """Create default app registry based on discovered apps"""
        default_apps = {
            "quantum_inspired_crypto": AppConfig(
                name="Quantum Crypto",
                module="apps.quantum_inspired_crypto",
                class_name="QuantumCrypto",
                icon="quantum_inspired_crypto.svg",
                description="Quantum cryptography and QKD protocols"
            ),
            "q_notes": AppConfig(
                name="Q Notes", 
                module="apps.q_notes",
                class_name="QNotes",
                icon="q_notes.svg",
                description="Quantum-enhanced note taking"
            ),
            "q_vault": AppConfig(
                name="Q Vault",
                module="apps.q_vault", 
                class_name="QVault",
                icon="q_vault.svg",
                description="Secure quantum storage"
            ),
            "symbolic_simulator": AppConfig(
                name="Quantum Simulator",
                module="apps.symbolic_simulator",
                class_name="QuantumSimulator", 
                icon="symbolic_simulator.svg",
                description="Quantum circuit simulation"
            ),
            "rft_validator": AppConfig(
                name="RFT Validator",
                module="apps.rft_validation_suite",
                class_name="RFTValidationSuite",
                icon="rft_validator.svg", 
                description="RFT mathematical validation"
            ),
            "rft_visualizer": AppConfig(
                name="RFT Visualizer",
                module="apps.rft_visualizer",
                class_name="RFTVisualizer",
                icon="rft_visualizer.svg",
                description="RFT data visualization"
            ),
            "system_monitor": AppConfig(
                name="System Monitor",
                module="apps.qshll_system_monitor",
                class_name="QSystemMonitor", 
                icon="system_monitor.svg",
                description="System performance monitoring"
            )
        }
        return default_apps
        
    def save_app_registry(self) -> None:
        """Save app registry to file"""
        if self._app_registry is None:
            return
            
        registry_file = self.config_dir / "app_registry.json"
        
        data = {
            "apps": {
                name: asdict(config) 
                for name, config in self._app_registry.items()
            }
        }
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save app registry: {e}")
            
    def get_build_config(self) -> BuildConfig:
        """
        Get build configuration
        
        Returns:
            Build configuration object
        """
        if self._build_config is None:
            self._load_build_config()
        return self._build_config
        
    def _load_build_config(self) -> None:
        """Load build configuration from file or create default"""
        config_file = self.config_dir / "build_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                self._build_config = BuildConfig(**data.get('build', {}))
            except Exception as e:
                print(f"Warning: Could not load build config: {e}")
                self._build_config = self._create_default_build_config()
        else:
            self._build_config = self._create_default_build_config()
            self.save_build_config()
            
    def _create_default_build_config(self) -> BuildConfig:
        """Create default build configuration"""
        return BuildConfig(
            kernel_source_dir=str(paths.kernel),
            kernel_build_dir=str(paths.build),
            python_bindings_dir=str(paths.python_bindings),
            target_name="librftkernel",
            compiler="gcc",
            compiler_flags=["-O3", "-fPIC", "-shared"]
        )
        
    def save_build_config(self) -> None:
        """Save build configuration to file"""
        if self._build_config is None:
            return
            
        config_file = self.config_dir / "build_config.json"
        
        data = {
            "build": asdict(self._build_config)
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save build config: {e}")
            
    def get_app_config(self, app_name: str) -> Optional[AppConfig]:
        """
        Get configuration for specific app
        
        Args:
            app_name: Name of the application
            
        Returns:
            App configuration or None if not found
        """
        registry = self.get_app_registry()
        return registry.get(app_name)
        
    def get_enabled_apps(self) -> Dict[str, AppConfig]:
        """
        Get only enabled applications
        
        Returns:
            Dictionary of enabled app configurations
        """
        registry = self.get_app_registry()
        return {
            name: config 
            for name, config in registry.items() 
            if config.enabled
        }
        
    def set_runtime_setting(self, key: str, value: Any) -> None:
        """Set runtime configuration setting"""
        self._runtime_config[key] = value
        
    def get_runtime_setting(self, key: str, default: Any = None) -> Any:
        """Get runtime configuration setting"""
        return self._runtime_config.get(key, default)
        
    def get_theme_path(self, theme_name: str = "styles.qss") -> Path:
        """
        Get path to theme file
        
        Args:
            theme_name: Name of theme file
            
        Returns:
            Path to theme file
        """
        # Try project root first (legacy)
        legacy_path = paths.project_root / theme_name
        if legacy_path.exists():
            return legacy_path
            
        # Try UI directory
        ui_path = paths.ui / theme_name
        if ui_path.exists():
            return ui_path
            
        # Default fallback
        return paths.ui / "styles.qss"
        
    def get_icon_path(self, icon_name: str) -> Path:
        """
        Get path to icon file
        
        Args:
            icon_name: Name of icon file
            
        Returns:
            Path to icon file
        """
        return paths.icons / icon_name
        
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return any issues
        
        Returns:
            List of validation errors/warnings
        """
        issues = []
        
        # Check app registry
        registry = self.get_app_registry()
        for name, config in registry.items():
            # Check if module exists
            module_path = paths.apps / f"{config.module.split('.')[-1]}.py"
            if not module_path.exists():
                issues.append(f"App {name}: Module file not found at {module_path}")
                
            # Check if icon exists  
            icon_path = self.get_icon_path(config.icon)
            if not icon_path.exists():
                issues.append(f"App {name}: Icon file not found at {icon_path}")
                
        # Check build config
        build_config = self.get_build_config()
        
        if not Path(build_config.kernel_source_dir).exists():
            issues.append(f"Build: Kernel source directory not found: {build_config.kernel_source_dir}")
            
        # Check theme
        theme_path = self.get_theme_path()
        if not theme_path.exists():
            issues.append(f"Theme file not found: {theme_path}")
            
        return issues
        
    def create_config_files(self) -> None:
        """Create all configuration files with current settings"""
        self.save_app_registry()
        self.save_build_config()
        
        # Create paths config for reference
        paths_config = {
            "paths": paths.get_all_paths(),
            "note": "This file is auto-generated for reference only"
        }
        
        try:
            with open(self.config_dir / "paths_reference.json", 'w') as f:
                json.dump(paths_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save paths reference: {e}")


# Global instance
config = QuantoniumConfig()


def get_app_registry() -> Dict[str, AppConfig]:
    """Get application registry (convenience function)"""
    return config.get_app_registry()


def get_app_config(app_name: str) -> Optional[AppConfig]:
    """Get app configuration (convenience function)"""
    return config.get_app_config(app_name)


def get_build_config() -> BuildConfig:
    """Get build configuration (convenience function)"""
    return config.get_build_config()


if __name__ == "__main__":
    # Test configuration system
    print("Testing QuantoniumOS Configuration:")
    print("=" * 40)
    
    # Show app registry
    print("Registered Apps:")
    registry = get_app_registry()
    for name, app_config in registry.items():
        status = "enabled" if app_config.enabled else "disabled"
        print(f"  ✓ {name}: {app_config.name} ({status})")
        
    # Show build config
    print(f"\nBuild Configuration:")
    build_cfg = get_build_config()
    print(f"  Compiler: {build_cfg.compiler}")
    print(f"  Target: {build_cfg.target_name}")
    print(f"  Source: {build_cfg.kernel_source_dir}")
    
    # Validate config
    issues = config.validate_config()
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print(f"\n✓ Configuration is valid")
        
    # Create config files
    config.create_config_files()
    print(f"\n✓ Configuration files created")
