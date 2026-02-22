# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
QuantoniumOS Q-Vault
=================
Secure quantum-encrypted password vault
"""

import os
import sys
import json
import hashlib
import base64
import random
import string
import datetime
from typing import List, Dict, Optional

# Import the base launcher
try:
    from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
except ImportError:
    # Try to find the launcher_base module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
    except ImportError:
        print("Error: launcher_base.py not found")
        sys.exit(1)

# Try to import PyQt5 for the GUI
if HAS_PYQT:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QTextEdit, QListWidget, QTabWidget, QFrame, QListWidgetItem,
                              QMessageBox, QMenu, QDialog, QFormLayout, QInputDialog)
    from PyQt5.QtGui import QIcon, QFont
    from PyQt5.QtCore import Qt, QSize, QTimer

class SimpleEncryption:
    """Simple encryption for demonstration"""
    
    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive a key from a password"""
        # Use PBKDF2 for key derivation
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return key
    
    @staticmethod
    def encrypt(data: str, password: str) -> str:
        """Encrypt data with a password"""
        # Generate a random salt
        salt = os.urandom(16)
        
        # Derive a key from the password
        key = SimpleEncryption.derive_key(password, salt)
        
        # XOR encryption (for demonstration only)
        data_bytes = data.encode()
        encrypted = bytearray()
        
        for i, b in enumerate(data_bytes):
            key_byte = key[i % len(key)]
            encrypted.append(b ^ key_byte)
        
        # Combine salt and encrypted data
        result = salt + bytes(encrypted)
        
        # Base64 encode the result
        return base64.b64encode(result).decode()
    
    @staticmethod
    def decrypt(encrypted_data: str, password: str) -> str:
        """Decrypt data with a password"""
        try:
            # Base64 decode the data
            data = base64.b64decode(encrypted_data)
            
            # Extract the salt
            salt = data[:16]
            encrypted = data[16:]
            
            # Derive the key
            key = SimpleEncryption.derive_key(password, salt)
            
            # XOR decryption
            decrypted = bytearray()
            
            for i, b in enumerate(encrypted):
                key_byte = key[i % len(key)]
                decrypted.append(b ^ key_byte)
            
            # Return the decrypted data
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            return ""

class QVaultApp(AppWindow):
    """Q-Vault window"""
    
    def __init__(self, app_name: str, app_icon: str):
        """Initialize the Q-Vault window"""
        super().__init__(app_name, app_icon)
        
        # Initialize the vault
        self.vault = None
        self.master_password = None
        
        # Create the UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Clear the layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Show the login screen if not logged in
        if self.vault is None:
            self.create_login_ui()
        else:
            self.create_vault_ui()
    
    def create_login_ui(self):
        """Create the login UI"""
        # Create the login frame
        login_frame = QFrame()
        login_frame.setFrameShape(QFrame.StyledPanel)
        login_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        login_layout = QVBoxLayout(login_frame)
        
        # Add the logo
        logo_label = QLabel("Q-VAULT")
        logo_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        logo_label.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(logo_label)
        
        # Add the subtitle
        subtitle_label = QLabel("Secure Password Vault")
        subtitle_label.setStyleSheet("color: white; font-size: 16px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(subtitle_label)
        
        # Add some space
        login_layout.addSpacing(20)
        
        # Add the password field
        password_layout = QHBoxLayout()
        password_label = QLabel("Master Password:")
        password_label.setStyleSheet("color: white;")
        password_layout.addWidget(password_label)
        
        self.password_field = QLineEdit()
        self.password_field.setEchoMode(QLineEdit.Password)
        self.password_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        password_layout.addWidget(self.password_field)
        
        login_layout.addLayout(password_layout)
        
        # Add the login button
        login_button = QPushButton("Unlock Vault")
        login_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 80, 60, 200);
                color: white;
                border: 1px solid rgba(100, 200, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 120, 80, 200);
                border: 1px solid rgba(150, 255, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 60, 40, 200);
                border: 1px solid rgba(100, 200, 100, 200);
            }
        """)
        login_button.clicked.connect(self.login)
        login_layout.addWidget(login_button)
        
        # Add the new vault button
        new_vault_button = QPushButton("Create New Vault")
        new_vault_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid rgba(100, 100, 200, 200);
            }
        """)
        new_vault_button.clicked.connect(self.create_new_vault)
        login_layout.addWidget(new_vault_button)
        
        # Add some space
        login_layout.addSpacing(20)
        
        # Add the status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(self.status_label)
        
        # Add the frame to the layout
        self.layout.addWidget(login_frame, alignment=Qt.AlignCenter)
    
    def create_vault_ui(self):
        """Create the vault UI"""
        # Create the main layout
        main_layout = QHBoxLayout()
        self.layout.addLayout(main_layout)
        
        # Create the password list panel
        list_panel = QFrame()
        list_panel.setFrameShape(QFrame.StyledPanel)
        list_panel.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        list_panel.setFixedWidth(250)
        list_layout = QVBoxLayout(list_panel)
        
        # Add the list header
        header_label = QLabel("Passwords")
        header_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        list_layout.addWidget(header_label)
        
        # Add the password list
        self.password_list = QListWidget()
        self.password_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
            QListWidget::item {
                border-bottom: 1px solid rgba(60, 60, 80, 150);
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: rgba(80, 80, 120, 150);
                border: 1px solid rgba(150, 150, 255, 150);
            }
        """)
        self.password_list.itemClicked.connect(self.password_selected)
        list_layout.addWidget(self.password_list)
        
        # Add the new password button
        new_password_button = QPushButton("Add Password")
        new_password_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 80, 60, 200);
                color: white;
                border: 1px solid rgba(100, 200, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 120, 80, 200);
                border: 1px solid rgba(150, 255, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 60, 40, 200);
                border: 1px solid rgba(100, 200, 100, 200);
            }
        """)
        new_password_button.clicked.connect(self.add_password)
        list_layout.addWidget(new_password_button)
        
        # Add the logout button
        logout_button = QPushButton("Lock Vault")
        logout_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(80, 60, 60, 200);
                color: white;
                border: 1px solid rgba(200, 100, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(120, 80, 80, 200);
                border: 1px solid rgba(255, 150, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(60, 40, 40, 200);
                border: 1px solid rgba(200, 100, 100, 200);
            }
        """)
        logout_button.clicked.connect(self.logout)
        list_layout.addWidget(logout_button)
        
        main_layout.addWidget(list_panel)
        
        # Create the password details panel
        details_panel = QFrame()
        details_panel.setFrameShape(QFrame.StyledPanel)
        details_panel.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        details_layout = QVBoxLayout(details_panel)
        
        # Add the details form
        form_layout = QFormLayout()
        
        # Add the title field
        self.title_label = QLabel("Title:")
        self.title_label.setStyleSheet("color: white; font-weight: bold;")
        self.title_field = QLineEdit()
        self.title_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        form_layout.addRow(self.title_label, self.title_field)
        
        # Add the username field
        self.username_label = QLabel("Username:")
        self.username_label.setStyleSheet("color: white; font-weight: bold;")
        self.username_field = QLineEdit()
        self.username_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        form_layout.addRow(self.username_label, self.username_field)
        
        # Add the password field
        self.password_label = QLabel("Password:")
        self.password_label.setStyleSheet("color: white; font-weight: bold;")
        password_layout = QHBoxLayout()
        self.password_field = QLineEdit()
        self.password_field.setEchoMode(QLineEdit.Password)
        self.password_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        password_layout.addWidget(self.password_field)
        
        self.show_password_button = QPushButton("Show")
        self.show_password_button.setFixedWidth(60)
        self.show_password_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 5px;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid rgba(100, 100, 200, 200);
            }
        """)
        self.show_password_button.pressed.connect(lambda: self.password_field.setEchoMode(QLineEdit.Normal))
        self.show_password_button.released.connect(lambda: self.password_field.setEchoMode(QLineEdit.Password))
        password_layout.addWidget(self.show_password_button)
        
        form_layout.addRow(self.password_label, password_layout)
        
        # Add the URL field
        self.url_label = QLabel("URL:")
        self.url_label.setStyleSheet("color: white; font-weight: bold;")
        self.url_field = QLineEdit()
        self.url_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        form_layout.addRow(self.url_label, self.url_field)
        
        # Add the notes field
        self.notes_label = QLabel("Notes:")
        self.notes_label.setStyleSheet("color: white; font-weight: bold;")
        self.notes_field = QTextEdit()
        self.notes_field.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
        """)
        form_layout.addRow(self.notes_label, self.notes_field)
        
        details_layout.addLayout(form_layout)
        
        # Add the buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Changes")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 80, 60, 200);
                color: white;
                border: 1px solid rgba(100, 200, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 120, 80, 200);
                border: 1px solid rgba(150, 255, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 60, 40, 200);
                border: 1px solid rgba(100, 200, 100, 200);
            }
        """)
        self.save_button.clicked.connect(self.save_password)
        button_layout.addWidget(self.save_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(80, 60, 60, 200);
                color: white;
                border: 1px solid rgba(200, 100, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(120, 80, 80, 200);
                border: 1px solid rgba(255, 150, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(60, 40, 40, 200);
                border: 1px solid rgba(200, 100, 100, 200);
            }
        """)
        self.delete_button.clicked.connect(self.delete_password)
        button_layout.addWidget(self.delete_button)
        
        self.generate_button = QPushButton("Generate Password")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid rgba(100, 100, 200, 200);
            }
        """)
        self.generate_button.clicked.connect(self.generate_password)
        button_layout.addWidget(self.generate_button)
        
        details_layout.addLayout(button_layout)
        
        main_layout.addWidget(details_panel)
        
        # Populate the password list
        self.populate_password_list()
        
        # Select the first password
        if self.vault and self.vault["passwords"]:
            self.password_list.setCurrentRow(0)
            self.password_selected(self.password_list.item(0))
        else:
            self.clear_password_fields()
    
    def login(self):
        """Log in to the vault"""
        password = self.password_field.text()
        
        if not password:
            self.status_label.setText("Please enter a password")
            return
        
        # Try to open the vault
        vault_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        vault_file = os.path.join(vault_dir, "vault.dat")
        
        if os.path.exists(vault_file):
            try:
                with open(vault_file, "r") as f:
                    encrypted_data = f.read()
                
                # Try to decrypt
                decrypted_data = SimpleEncryption.decrypt(encrypted_data, password)
                
                if not decrypted_data:
                    self.status_label.setText("Invalid password")
                    return
                
                # Parse the JSON
                self.vault = json.loads(decrypted_data)
                self.master_password = password
                
                # Show the vault UI
                self.create_vault_ui()
            except Exception as e:
                self.status_label.setText(f"Error opening vault: {e}")
        else:
            self.status_label.setText("No vault found. Create a new one.")
    
    def create_new_vault(self):
        """Create a new vault"""
        password = self.password_field.text()
        
        if not password:
            self.status_label.setText("Please enter a password")
            return
        
        # Create a new vault
        self.vault = {
            "created": datetime.datetime.now().isoformat(),
            "passwords": []
        }
        self.master_password = password
        
        # Save the vault
        self.save_vault()
        
        # Show the vault UI
        self.create_vault_ui()
    
    def logout(self):
        """Log out of the vault"""
        self.vault = None
        self.master_password = None
        
        # Show the login UI
        self.create_login_ui()
    
    def save_vault(self):
        """Save the vault to disk"""
        if not self.vault or not self.master_password:
            return
        
        try:
            # Convert the vault to JSON
            json_data = json.dumps(self.vault)
            
            # Encrypt the data
            encrypted_data = SimpleEncryption.encrypt(json_data, self.master_password)
            
            # Save to file
            vault_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(vault_dir):
                os.makedirs(vault_dir)
            
            vault_file = os.path.join(vault_dir, "vault.dat")
            
            with open(vault_file, "w") as f:
                f.write(encrypted_data)
        except Exception as e:
            print(f"Error saving vault: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save vault: {e}")
    
    def populate_password_list(self):
        """Populate the password list"""
        if not self.vault:
            return
        
        self.password_list.clear()
        
        for password in self.vault["passwords"]:
            item = QListWidgetItem(password["title"])
            item.setData(Qt.UserRole, password["id"])
            self.password_list.addItem(item)
    
    def password_selected(self, item):
        """Handle password selection"""
        if not item:
            return
        
        password_id = item.data(Qt.UserRole)
        
        # Find the password
        password = next((p for p in self.vault["passwords"] if p["id"] == password_id), None)
        
        if password:
            # Update the fields
            self.title_field.setText(password["title"])
            self.username_field.setText(password.get("username", ""))
            self.password_field.setText(password.get("password", ""))
            self.url_field.setText(password.get("url", ""))
            self.notes_field.setText(password.get("notes", ""))
    
    def clear_password_fields(self):
        """Clear the password fields"""
        self.title_field.clear()
        self.username_field.clear()
        self.password_field.clear()
        self.url_field.clear()
        self.notes_field.clear()
    
    def add_password(self):
        """Add a new password"""
        if not self.vault:
            return
        
        # Generate a unique ID
        password_id = f"pass_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the password
        password = {
            "id": password_id,
            "title": "New Password",
            "username": "",
            "password": "",
            "url": "",
            "notes": "",
            "created": datetime.datetime.now().isoformat(),
            "modified": datetime.datetime.now().isoformat()
        }
        
        # Add to the vault
        self.vault["passwords"].append(password)
        
        # Save the vault
        self.save_vault()
        
        # Update the UI
        self.populate_password_list()
        
        # Select the new password
        for i in range(self.password_list.count()):
            if self.password_list.item(i).data(Qt.UserRole) == password_id:
                self.password_list.setCurrentRow(i)
                self.password_selected(self.password_list.item(i))
                break
    
    def save_password(self):
        """Save the current password"""
        if not self.vault:
            return
        
        # Get the current password
        if not self.password_list.currentItem():
            return
        
        password_id = self.password_list.currentItem().data(Qt.UserRole)
        
        # Find the password
        password = next((p for p in self.vault["passwords"] if p["id"] == password_id), None)
        
        if password:
            # Update the password
            password["title"] = self.title_field.text()
            password["username"] = self.username_field.text()
            password["password"] = self.password_field.text()
            password["url"] = self.url_field.text()
            password["notes"] = self.notes_field.toPlainText()
            password["modified"] = datetime.datetime.now().isoformat()
            
            # Update the list
            self.password_list.currentItem().setText(password["title"])
            
            # Save the vault
            self.save_vault()
    
    def delete_password(self):
        """Delete the current password"""
        if not self.vault:
            return
        
        # Get the current password
        if not self.password_list.currentItem():
            return
        
        password_id = self.password_list.currentItem().data(Qt.UserRole)
        
        # Find the password
        password = next((p for p in self.vault["passwords"] if p["id"] == password_id), None)
        
        if password:
            # Confirm deletion
            result = QMessageBox.question(
                self,
                "Delete Password",
                f"Are you sure you want to delete '{password['title']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if result == QMessageBox.Yes:
                # Remove the password
                self.vault["passwords"].remove(password)
                
                # Save the vault
                self.save_vault()
                
                # Update the UI
                self.populate_password_list()
                
                # Select the first password or clear the fields
                if self.password_list.count() > 0:
                    self.password_list.setCurrentRow(0)
                    self.password_selected(self.password_list.item(0))
                else:
                    self.clear_password_fields()
    
    def generate_password(self):
        """Generate a random password"""
        # Ask for password length
        length, ok = QInputDialog.getInt(
            self,
            "Generate Password",
            "Password length:",
            16,  # Default
            8,   # Min
            64,  # Max
            1    # Step
        )
        
        if ok:
            # Generate the password
            chars = string.ascii_letters + string.digits + string.punctuation
            password = ''.join(random.choice(chars) for _ in range(length))
            
            # Set the password
            self.password_field.setText(password)

class QVaultTerminal(AppTerminal):
    """Q-Vault terminal"""
    
    def __init__(self, app_name: str):
        """Initialize the Q-Vault terminal"""
        super().__init__(app_name)
        
        # Initialize the vault
        self.vault = None
        self.master_password = None
        self.current_password = None
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        # Show the login prompt
        self.show_login_prompt()
    
    def show_login_prompt(self):
        """Show the login prompt"""
        print("Please log in to your vault")
        print("  1. Unlock existing vault")
        print("  2. Create new vault")
        print("  3. Exit")
        
        choice = input("Choice: ").strip()
        
        if choice == "1":
            self.login()
        elif choice == "2":
            self.create_new_vault()
        elif choice == "3":
            print("Exiting...")
            self.running = False
        else:
            print("Invalid choice")
            self.show_login_prompt()
    
    def login(self):
        """Log in to the vault"""
        password = input("Master password: ").strip()
        
        if not password:
            print("Password cannot be empty")
            self.show_login_prompt()
            return
        
        # Try to open the vault
        vault_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        vault_file = os.path.join(vault_dir, "vault.dat")
        
        if os.path.exists(vault_file):
            try:
                with open(vault_file, "r") as f:
                    encrypted_data = f.read()
                
                # Try to decrypt
                decrypted_data = SimpleEncryption.decrypt(encrypted_data, password)
                
                if not decrypted_data:
                    print("Invalid password")
                    self.show_login_prompt()
                    return
                
                # Parse the JSON
                self.vault = json.loads(decrypted_data)
                self.master_password = password
                
                # Show the main menu
                self.show_main_menu()
            except Exception as e:
                print(f"Error opening vault: {e}")
                self.show_login_prompt()
        else:
            print("No vault found. Create a new one.")
            self.show_login_prompt()
    
    def create_new_vault(self):
        """Create a new vault"""
        password = input("Master password: ").strip()
        
        if not password:
            print("Password cannot be empty")
            self.show_login_prompt()
            return
        
        # Confirm password
        confirm = input("Confirm password: ").strip()
        
        if password != confirm:
            print("Passwords do not match")
            self.show_login_prompt()
            return
        
        # Create a new vault
        self.vault = {
            "created": datetime.datetime.now().isoformat(),
            "passwords": []
        }
        self.master_password = password
        
        # Save the vault
        self.save_vault()
        
        print("Vault created successfully")
        
        # Show the main menu
        self.show_main_menu()
    
    def save_vault(self):
        """Save the vault to disk"""
        if not self.vault or not self.master_password:
            return
        
        try:
            # Convert the vault to JSON
            json_data = json.dumps(self.vault)
            
            # Encrypt the data
            encrypted_data = SimpleEncryption.encrypt(json_data, self.master_password)
            
            # Save to file
            vault_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(vault_dir):
                os.makedirs(vault_dir)
            
            vault_file = os.path.join(vault_dir, "vault.dat")
            
            with open(vault_file, "w") as f:
                f.write(encrypted_data)
        except Exception as e:
            print(f"Error saving vault: {e}")
    
    def show_main_menu(self):
        """Show the main menu"""
        print("\n" + "=" * 60)
        print("Q-Vault Main Menu")
        print("=" * 60)
        
        print("Available commands:")
        print("  list          - List all passwords")
        print("  view [number] - View a password")
        print("  add           - Add a new password")
        print("  edit [number] - Edit a password")
        print("  delete [number] - Delete a password")
        print("  generate      - Generate a random password")
        print("  lock          - Lock the vault")
        print("  exit          - Exit the application\n")
        
        # Main loop
        while self.running and self.vault is not None:
            command = input(f"{self.app_name}> ").strip()
            self.process_command(command)
    
    def process_command(self, command: str):
        """Process a terminal command"""
        parts = command.split()
        
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  list          - List all passwords")
            print("  view [number] - View a password")
            print("  add           - Add a new password")
            print("  edit [number] - Edit a password")
            print("  delete [number] - Delete a password")
            print("  generate      - Generate a random password")
            print("  lock          - Lock the vault")
            print("  exit          - Exit the application\n")
        
        elif cmd == "list":
            self.list_passwords()
        
        elif cmd == "view":
            if not args:
                print("Error: Missing password number")
                print("Usage: view [number]")
                return
            
            try:
                password_index = int(args[0]) - 1
                if password_index < 0 or password_index >= len(self.vault["passwords"]):
                    print(f"Error: Password number must be between 1 and {len(self.vault['passwords'])}")
                    return
                
                self.view_password(password_index)
            except ValueError:
                print("Error: Invalid password number")
        
        elif cmd == "add":
            self.add_password()
        
        elif cmd == "edit":
            if not args:
                print("Error: Missing password number")
                print("Usage: edit [number]")
                return
            
            try:
                password_index = int(args[0]) - 1
                if password_index < 0 or password_index >= len(self.vault["passwords"]):
                    print(f"Error: Password number must be between 1 and {len(self.vault['passwords'])}")
                    return
                
                self.edit_password(password_index)
            except ValueError:
                print("Error: Invalid password number")
        
        elif cmd == "delete":
            if not args:
                print("Error: Missing password number")
                print("Usage: delete [number]")
                return
            
            try:
                password_index = int(args[0]) - 1
                if password_index < 0 or password_index >= len(self.vault["passwords"]):
                    print(f"Error: Password number must be between 1 and {len(self.vault['passwords'])}")
                    return
                
                self.delete_password(password_index)
            except ValueError:
                print("Error: Invalid password number")
        
        elif cmd == "generate":
            self.generate_password()
        
        elif cmd == "lock":
            print("Locking vault...")
            self.vault = None
            self.master_password = None
            self.current_password = None
            self.show_login_prompt()
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def list_passwords(self):
        """List all passwords"""
        if not self.vault:
            return
        
        print("\nPasswords:")
        for i, password in enumerate(self.vault["passwords"]):
            print(f"  {i+1}. {password['title']}")
        print("")
    
    def view_password(self, index):
        """View a password"""
        if not self.vault:
            return
        
        password = self.vault["passwords"][index]
        
        print("\n" + "=" * 60)
        print(f"Title: {password['title']}")
        print(f"Username: {password.get('username', '')}")
        print(f"Password: {password.get('password', '')}")
        print(f"URL: {password.get('url', '')}")
        print(f"Notes: {password.get('notes', '')}")
        
        created = datetime.datetime.fromisoformat(password.get('created', datetime.datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')
        modified = datetime.datetime.fromisoformat(password.get('modified', datetime.datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Created: {created}")
        print(f"Modified: {modified}")
        print("=" * 60 + "\n")
    
    def add_password(self):
        """Add a new password"""
        if not self.vault:
            return
        
        print("\nAdding a new password")
        title = input("Title: ").strip()
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        url = input("URL: ").strip()
        notes = input("Notes: ").strip()
        
        if not title:
            print("Error: Title is required")
            return
        
        # Generate a unique ID
        password_id = f"pass_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the password
        new_password = {
            "id": password_id,
            "title": title,
            "username": username,
            "password": password,
            "url": url,
            "notes": notes,
            "created": datetime.datetime.now().isoformat(),
            "modified": datetime.datetime.now().isoformat()
        }
        
        # Add to the vault
        self.vault["passwords"].append(new_password)
        
        # Save the vault
        self.save_vault()
        
        print("\nPassword added successfully!\n")
    
    def edit_password(self, index):
        """Edit a password"""
        if not self.vault:
            return
        
        password = self.vault["passwords"][index]
        
        print("\nEditing password")
        print(f"Current title: {password['title']}")
        title = input("New title (leave empty to keep current): ").strip()
        if title:
            password["title"] = title
        
        print(f"Current username: {password.get('username', '')}")
        username = input("New username (leave empty to keep current): ").strip()
        if username:
            password["username"] = username
        
        print(f"Current password: {password.get('password', '')}")
        new_password = input("New password (leave empty to keep current): ").strip()
        if new_password:
            password["password"] = new_password
        
        print(f"Current URL: {password.get('url', '')}")
        url = input("New URL (leave empty to keep current): ").strip()
        if url:
            password["url"] = url
        
        print(f"Current notes: {password.get('notes', '')}")
        notes = input("New notes (leave empty to keep current): ").strip()
        if notes:
            password["notes"] = notes
        
        # Update the modified timestamp
        password["modified"] = datetime.datetime.now().isoformat()
        
        # Save the vault
        self.save_vault()
        
        print("\nPassword updated successfully!\n")
    
    def delete_password(self, index):
        """Delete a password"""
        if not self.vault:
            return
        
        password = self.vault["passwords"][index]
        
        print(f"\nAre you sure you want to delete '{password['title']}'? (y/n)")
        confirmation = input().strip().lower()
        
        if confirmation == "y":
            # Remove the password
            del self.vault["passwords"][index]
            
            # Save the vault
            self.save_vault()
            
            print("\nPassword deleted successfully!\n")
        else:
            print("\nDeletion cancelled\n")
    
    def generate_password(self):
        """Generate a random password"""
        print("\nGenerating a random password")
        
        try:
            length = int(input("Password length (8-64): ").strip())
            
            if length < 8 or length > 64:
                print("Error: Length must be between 8 and 64")
                return
            
            # Generate the password
            chars = string.ascii_letters + string.digits + string.punctuation
            password = ''.join(random.choice(chars) for _ in range(length))
            
            # Print the password
            print(f"\nGenerated password: {password}\n")
        except ValueError:
            print("Error: Invalid length")

def main():
    """Main function"""
    # Create the app launcher
    launcher = AppLauncherBase("Q-Vault", "fa5s.shield-alt")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        launcher.launch_terminal(QVaultTerminal)
    else:
        launcher.launch_gui(QVaultApp)

if __name__ == "__main__":
    main()

