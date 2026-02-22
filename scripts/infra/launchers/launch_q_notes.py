# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
QuantoniumOS Q-Notes
==================
Secure note-taking application
"""

import os
import sys
import json
import datetime
from typing import List, Dict

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
                              QMessageBox, QMenu)
    from PyQt5.QtGui import QIcon, QFont
    from PyQt5.QtCore import Qt, QDateTime

class QNotesApp(AppWindow):
    """Q-Notes window"""
    
    def __init__(self, app_name: str, app_icon: str):
        """Initialize the Q-Notes window"""
        super().__init__(app_name, app_icon)
        
        # Initialize the notes
        self.notes = self.load_notes()
        
        # Create the UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Clear the layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Create the main layout
        main_layout = QHBoxLayout()
        self.layout.addLayout(main_layout)
        
        # Create the note list panel
        list_panel = QFrame()
        list_panel.setFrameShape(QFrame.StyledPanel)
        list_panel.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        list_panel.setFixedWidth(250)
        list_layout = QVBoxLayout(list_panel)
        
        # Add the list header
        header_label = QLabel("My Notes")
        header_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        list_layout.addWidget(header_label)
        
        # Add the note list
        self.note_list = QListWidget()
        self.note_list.setStyleSheet("""
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
        self.note_list.itemClicked.connect(self.note_selected)
        list_layout.addWidget(self.note_list)
        
        # Add the new note button
        new_note_button = QPushButton("New Note")
        new_note_button.setStyleSheet("""
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
        new_note_button.clicked.connect(self.new_note)
        list_layout.addWidget(new_note_button)
        
        main_layout.addWidget(list_panel)
        
        # Create the note editor panel
        editor_panel = QFrame()
        editor_panel.setFrameShape(QFrame.StyledPanel)
        editor_panel.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        editor_layout = QVBoxLayout(editor_panel)
        
        # Add the title field
        title_layout = QHBoxLayout()
        title_label = QLabel("Title:")
        title_label.setStyleSheet("color: white; font-weight: bold;")
        title_layout.addWidget(title_label)
        
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
        title_layout.addWidget(self.title_field)
        
        editor_layout.addLayout(title_layout)
        
        # Add the editor
        self.note_editor = QTextEdit()
        self.note_editor.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
        """)
        editor_layout.addWidget(self.note_editor)
        
        # Add the save button
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Note")
        save_button.setStyleSheet("""
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
        save_button.clicked.connect(self.save_note)
        button_layout.addWidget(save_button)
        
        delete_button = QPushButton("Delete Note")
        delete_button.setStyleSheet("""
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
        delete_button.clicked.connect(self.delete_note)
        button_layout.addWidget(delete_button)
        
        editor_layout.addLayout(button_layout)
        
        main_layout.addWidget(editor_panel)
        
        # Populate the note list
        self.populate_note_list()
        
        # Set the current note
        if self.notes:
            self.note_list.setCurrentRow(0)
            self.note_selected(self.note_list.item(0))
        else:
            self.new_note()
    
    def load_notes(self):
        """Load notes from file"""
        try:
            # Get the notes file path
            notes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            notes_file = os.path.join(notes_dir, "notes.json")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(notes_dir):
                os.makedirs(notes_dir)
            
            # Load the notes
            if os.path.exists(notes_file):
                with open(notes_file, "r") as f:
                    return json.load(f)
            else:
                # Create a welcome note
                welcome_note = {
                    "id": "welcome",
                    "title": "Welcome to Q-Notes",
                    "content": "Welcome to Q-Notes!\n\nThis is a secure note-taking application built on the QuantoniumOS platform.\n\nKey features:\n- Quantum-secure encryption\n- Simple and intuitive interface\n- Integration with QuantoniumOS\n\nEnjoy using Q-Notes!\n\nThe QuantoniumOS Team",
                    "created": datetime.datetime.now().isoformat(),
                    "modified": datetime.datetime.now().isoformat()
                }
                return [welcome_note]
        except Exception as e:
            print(f"Error loading notes: {e}")
            return []
    
    def save_notes(self):
        """Save notes to file"""
        try:
            # Get the notes file path
            notes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            notes_file = os.path.join(notes_dir, "notes.json")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(notes_dir):
                os.makedirs(notes_dir)
            
            # Save the notes
            with open(notes_file, "w") as f:
                json.dump(self.notes, f, indent=2)
        except Exception as e:
            print(f"Error saving notes: {e}")
    
    def populate_note_list(self):
        """Populate the note list"""
        self.note_list.clear()
        
        for note in self.notes:
            item = QListWidgetItem(note["title"])
            item.setData(Qt.UserRole, note["id"])
            self.note_list.addItem(item)
    
    def note_selected(self, item):
        """Handle note selection"""
        note_id = item.data(Qt.UserRole)
        
        # Find the note
        note = next((n for n in self.notes if n["id"] == note_id), None)
        
        if note:
            # Update the editor
            self.title_field.setText(note["title"])
            self.note_editor.setText(note["content"])
    
    def new_note(self):
        """Create a new note"""
        # Generate a unique ID
        note_id = f"note_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the note
        note = {
            "id": note_id,
            "title": "New Note",
            "content": "",
            "created": datetime.datetime.now().isoformat(),
            "modified": datetime.datetime.now().isoformat()
        }
        
        # Add to the list
        self.notes.append(note)
        
        # Update the UI
        self.populate_note_list()
        
        # Select the new note
        for i in range(self.note_list.count()):
            if self.note_list.item(i).data(Qt.UserRole) == note_id:
                self.note_list.setCurrentRow(i)
                self.note_selected(self.note_list.item(i))
                break
        
        # Save the notes
        self.save_notes()
    
    def save_note(self):
        """Save the current note"""
        # Get the current note
        if not self.note_list.currentItem():
            return
        
        note_id = self.note_list.currentItem().data(Qt.UserRole)
        
        # Find the note
        note = next((n for n in self.notes if n["id"] == note_id), None)
        
        if note:
            # Update the note
            note["title"] = self.title_field.text()
            note["content"] = self.note_editor.toPlainText()
            note["modified"] = datetime.datetime.now().isoformat()
            
            # Update the list
            self.note_list.currentItem().setText(note["title"])
            
            # Save the notes
            self.save_notes()
    
    def delete_note(self):
        """Delete the current note"""
        # Get the current note
        if not self.note_list.currentItem():
            return
        
        note_id = self.note_list.currentItem().data(Qt.UserRole)
        
        # Find the note
        note = next((n for n in self.notes if n["id"] == note_id), None)
        
        if note:
            # Confirm deletion
            result = QMessageBox.question(
                self,
                "Delete Note",
                f"Are you sure you want to delete '{note['title']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if result == QMessageBox.Yes:
                # Remove the note
                self.notes.remove(note)
                
                # Update the UI
                self.populate_note_list()
                
                # Select the first note or create a new one
                if self.note_list.count() > 0:
                    self.note_list.setCurrentRow(0)
                    self.note_selected(self.note_list.item(0))
                else:
                    self.title_field.clear()
                    self.note_editor.clear()
                    self.new_note()
                
                # Save the notes
                self.save_notes()

class QNotesTerminal(AppTerminal):
    """Q-Notes terminal"""
    
    def __init__(self, app_name: str):
        """Initialize the Q-Notes terminal"""
        super().__init__(app_name)
        
        # Initialize the notes
        self.notes = self.load_notes()
        self.current_note = None
    
    def load_notes(self):
        """Load notes from file"""
        try:
            # Get the notes file path
            notes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            notes_file = os.path.join(notes_dir, "notes.json")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(notes_dir):
                os.makedirs(notes_dir)
            
            # Load the notes
            if os.path.exists(notes_file):
                with open(notes_file, "r") as f:
                    return json.load(f)
            else:
                # Create a welcome note
                welcome_note = {
                    "id": "welcome",
                    "title": "Welcome to Q-Notes",
                    "content": "Welcome to Q-Notes!\n\nThis is a secure note-taking application built on the QuantoniumOS platform.\n\nKey features:\n- Quantum-secure encryption\n- Simple and intuitive interface\n- Integration with QuantoniumOS\n\nEnjoy using Q-Notes!\n\nThe QuantoniumOS Team",
                    "created": datetime.datetime.now().isoformat(),
                    "modified": datetime.datetime.now().isoformat()
                }
                return [welcome_note]
        except Exception as e:
            print(f"Error loading notes: {e}")
            return []
    
    def save_notes(self):
        """Save notes to file"""
        try:
            # Get the notes file path
            notes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            notes_file = os.path.join(notes_dir, "notes.json")
            
            # Create the directory if it doesn't exist
            if not os.path.exists(notes_dir):
                os.makedirs(notes_dir)
            
            # Save the notes
            with open(notes_file, "w") as f:
                json.dump(self.notes, f, indent=2)
        except Exception as e:
            print(f"Error saving notes: {e}")
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help          - Show this help message")
        print("  list          - List all notes")
        print("  view [number] - View a specific note")
        print("  new           - Create a new note")
        print("  edit [number] - Edit a note")
        print("  delete [number] - Delete a note")
        print("  exit          - Exit the application\n")
        
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
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  help          - Show this help message")
            print("  list          - List all notes")
            print("  view [number] - View a specific note")
            print("  new           - Create a new note")
            print("  edit [number] - Edit a note")
            print("  delete [number] - Delete a note")
            print("  exit          - Exit the application\n")
        
        elif cmd == "list":
            self.list_notes()
        
        elif cmd == "view":
            if not args:
                print("Error: Missing note number")
                print("Usage: view [number]")
                return
            
            try:
                note_index = int(args[0]) - 1
                if note_index < 0 or note_index >= len(self.notes):
                    print(f"Error: Note number must be between 1 and {len(self.notes)}")
                    return
                
                self.view_note(note_index)
            except ValueError:
                print("Error: Invalid note number")
        
        elif cmd == "new":
            self.new_note()
        
        elif cmd == "edit":
            if not args:
                print("Error: Missing note number")
                print("Usage: edit [number]")
                return
            
            try:
                note_index = int(args[0]) - 1
                if note_index < 0 or note_index >= len(self.notes):
                    print(f"Error: Note number must be between 1 and {len(self.notes)}")
                    return
                
                self.edit_note(note_index)
            except ValueError:
                print("Error: Invalid note number")
        
        elif cmd == "delete":
            if not args:
                print("Error: Missing note number")
                print("Usage: delete [number]")
                return
            
            try:
                note_index = int(args[0]) - 1
                if note_index < 0 or note_index >= len(self.notes):
                    print(f"Error: Note number must be between 1 and {len(self.notes)}")
                    return
                
                self.delete_note(note_index)
            except ValueError:
                print("Error: Invalid note number")
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def list_notes(self):
        """List all notes"""
        print("\nNotes:")
        for i, note in enumerate(self.notes):
            print(f"  {i+1}. {note['title']}")
        print("")
    
    def view_note(self, index):
        """View a note"""
        note = self.notes[index]
        
        print("\n" + "=" * 60)
        print(f"Title: {note['title']}")
        created = datetime.datetime.fromisoformat(note['created']).strftime('%Y-%m-%d %H:%M:%S')
        modified = datetime.datetime.fromisoformat(note['modified']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Created: {created}")
        print(f"Modified: {modified}")
        print("=" * 60)
        print(note['content'])
        print("=" * 60 + "\n")
    
    def new_note(self):
        """Create a new note"""
        print("\nCreating a new note")
        title = input("Title: ")
        print("Content (type '.' on a new line to finish):")
        
        content = []
        while True:
            line = input()
            if line == ".":
                break
            content.append(line)
        
        # Validate the note
        if not title:
            print("Error: Title is required")
            return
        
        # Generate a unique ID
        note_id = f"note_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the note
        note = {
            "id": note_id,
            "title": title,
            "content": "\n".join(content),
            "created": datetime.datetime.now().isoformat(),
            "modified": datetime.datetime.now().isoformat()
        }
        
        # Add to the list
        self.notes.append(note)
        
        # Save the notes
        self.save_notes()
        
        print("\nNote created successfully!\n")
    
    def edit_note(self, index):
        """Edit a note"""
        note = self.notes[index]
        
        print("\nEditing note")
        print(f"Current title: {note['title']}")
        title = input("New title (leave empty to keep current): ")
        if title:
            note['title'] = title
        
        print("Current content:")
        print(note['content'])
        print("\nNew content (type '.' on a new line to finish):")
        
        content = []
        while True:
            line = input()
            if line == ".":
                break
            content.append(line)
        
        if content:
            note['content'] = "\n".join(content)
        
        # Update the modified timestamp
        note['modified'] = datetime.datetime.now().isoformat()
        
        # Save the notes
        self.save_notes()
        
        print("\nNote updated successfully!\n")
    
    def delete_note(self, index):
        """Delete a note"""
        note = self.notes[index]
        
        print(f"\nAre you sure you want to delete '{note['title']}'? (y/n)")
        confirmation = input().strip().lower()
        
        if confirmation == "y":
            # Remove the note
            del self.notes[index]
            
            # Save the notes
            self.save_notes()
            
            print("\nNote deleted successfully!\n")
        else:
            print("\nDeletion cancelled\n")

def main():
    """Main function"""
    # Create the app launcher
    launcher = AppLauncherBase("Q-Notes", "fa5s.sticky-note")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        launcher.launch_terminal(QNotesTerminal)
    else:
        launcher.launch_gui(QNotesApp)

if __name__ == "__main__":
    main()

