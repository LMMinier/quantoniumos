# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
QuantoniumOS App Launcher Base
==============================
Base class for app launchers
"""

import os
import sys
import subprocess
from typing import Optional

# Try to import PyQt5 for the GUI
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
    from PyQt5.QtGui import QIcon, QFont
    from PyQt5.QtCore import Qt, QSize
    
    # Try to import qtawesome for icons
    try:
        import qtawesome as qta
        HAS_QTAWESOME = True
    except ImportError:
        HAS_QTAWESOME = False
    
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    print("PyQt5 not found. Please install it with: pip install PyQt5")
    if "--no-gui" not in sys.argv:
        sys.exit(1)

class AppLauncherBase:
    """Base class for app launchers"""
    
    def __init__(self, app_name: str, app_icon: str = "fa5s.cube"):
        """Initialize the app launcher"""
        self.app_name = app_name
        self.app_icon = app_icon
        
        # Try to get the app directory
        try:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))
            self.os_dir = os.path.dirname(self.app_dir)
        except:
            self.app_dir = os.getcwd()
            self.os_dir = os.path.dirname(self.app_dir)
    
    def launch_gui(self, app_class):
        """Launch the GUI app"""
        if not HAS_PYQT:
            print(f"Error: PyQt5 not available. Cannot launch {self.app_name} GUI.")
            return False
        
        try:
            app = QApplication(sys.argv)
            app.setStyle("Fusion")  # Use Fusion style for better cross-platform appearance
            
            window = app_class(self.app_name, self.app_icon)
            window.show()
            
            sys.exit(app.exec_())
        except Exception as e:
            print(f"Error launching {self.app_name} GUI: {e}")
            return False
    
    def launch_terminal(self, terminal_class):
        """Launch the terminal app"""
        try:
            terminal = terminal_class(self.app_name)
            terminal.start()
            return True
        except Exception as e:
            print(f"Error launching {self.app_name} terminal: {e}")
            return False
    
    def get_module_path(self, module_name: str) -> Optional[str]:
        """Get the path to a module"""
        module_paths = [
            os.path.join(self.os_dir, module_name + ".py"),
            os.path.join(self.os_dir, "core", module_name + ".py"),
            os.path.join(self.os_dir, "engines", module_name + ".py")
        ]
        
        for path in module_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def import_module(self, module_name: str):
        """Import a module by name"""
        module_path = self.get_module_path(module_name)
        
        if module_path is None:
            print(f"Error: Module {module_name} not found")
            return None
        
        try:
            # Add the module directory to the path
            module_dir = os.path.dirname(module_path)
            if module_dir not in sys.path:
                sys.path.append(module_dir)
            
            # Import the module
            module_name_only = os.path.basename(module_path)[:-3]  # Remove .py
            
            # Try direct import first
            try:
                return __import__(module_name_only)
            except ImportError:
                # Try to load as a file
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name_only, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            print(f"Error importing module {module_name}: {e}")
            return None

class AppWindow(QMainWindow):
    """Base window for apps"""
    
    def __init__(self, app_name: str, app_icon: str = "fa5s.cube"):
        """Initialize the app window"""
        super().__init__()
        
        # Set the window title and size
        self.setWindowTitle(app_name)
        self.setGeometry(100, 100, 800, 600)
        
        # Set the window icon
        if HAS_QTAWESOME:
            icon = qta.icon(app_icon, color='white')
            self.setWindowIcon(icon)
        
        # Set the window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
                font-size: 16px;
            }
        """)
        
        # Create the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create the layout
        self.layout = QVBoxLayout(self.central_widget)
        
        # Add a welcome label
        welcome_label = QLabel(f"Welcome to {app_name}")
        welcome_label.setFont(QFont("Arial", 20))
        welcome_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(welcome_label)

class AppTerminal:
    """Base class for terminal apps"""
    
    def __init__(self, app_name: str):
        """Initialize the terminal app"""
        self.app_name = app_name
        self.running = True
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help  - Show this help message")
        print("  exit  - Exit the application\n")
        
        # Main loop
        while self.running:
            command = input(f"{self.app_name}> ").strip()
            self.process_command(command)
    
    def process_command(self, command: str):
        """Process a terminal command"""
        parts = command.split()
        
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  help  - Show this help message")
            print("  exit  - Exit the application\n")
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")

