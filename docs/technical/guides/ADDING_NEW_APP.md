# How to Add a New Application

This guide walks you through the process of creating a new application for the QuantoniumOS desktop environment.

## Prerequisites

- Basic knowledge of Python and PyQt5
- Understanding of the QuantoniumOS architecture
- Development environment set up (see [Quick Start](../../onboarding/QUICK_START.md))

## Step 1: Create Application Directory

All applications live in `os/apps/`. Create a new directory for your application:

```bash
mkdir -p os/apps/my_new_app
cd os/apps/my_new_app
```

## Step 2: Create Application Files

Create the following files in your application directory:

```
my_new_app/
├── __init__.py
├── main.py              # Main application entry point
├── ui/                  # UI components
│   ├── main_window.py
│   └── widgets.py
├── core/                # Business logic
│   └── processor.py
└── resources/           # Icons, images, etc.
    └── icon.svg
```

## Step 3: Implement the Main Application Class

Create `main.py` with the standard QuantoniumOS application pattern:

```python
#!/usr/bin/env python3
"""
My New App - Description of what your app does
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# Try to import RFT engine (graceful fallback if not available)
try:
    from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False
    print("⚠️  RFT engine not available - using Python fallback")


class MyNewApp(QMainWindow):
    """
    Main application class for My New App.
    
    This class follows the standard QuantoniumOS application pattern:
    - RFT integration with graceful fallback
    - Golden ratio UI proportions
    - Standard theme integration
    - Proper cleanup on exit
    """
    
    def __init__(self):
        super().__init__()
        
        # Golden ratio constant
        self.phi = 1.618033988749895
        
        # Initialize components
        self.rft_engine = None
        self.init_rft()
        self.init_ui()
        self.load_styles()
        
    def init_rft(self):
        """Initialize RFT engine with error handling"""
        if RFT_AVAILABLE:
            try:
                self.rft_engine = UnitaryRFT(
                    size=256,
                    flags=RFT_FLAG_QUANTUM_SAFE
                )
                print("✅ RFT engine initialized")
            except Exception as e:
                print(f"⚠️  RFT initialization failed: {e}")
                self.rft_engine = None
        else:
            print("ℹ️  Running in fallback mode (no RFT)")
            
    def init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("My New App — QuantoniumOS")
        
        # Calculate dimensions using golden ratio
        width = int(800)
        height = int(width / self.phi)
        self.setGeometry(100, 100, width, height)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add your widgets here
        title_label = QLabel("My New App")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Status label
        self.status_label = QLabel()
        self.update_status()
        layout.addWidget(self.status_label)
        
    def load_styles(self):
        """Load application styles"""
        # Use the standard QuantoniumOS color scheme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #3498db;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #5dade2;
            }
        """)
        
    def update_status(self):
        """Update status label with RFT availability"""
        if self.rft_engine:
            status_text = "🟢 RFT Engine: Active"
        else:
            status_text = "🟡 RFT Engine: Fallback Mode"
        self.status_label.setText(status_text)
        self.status_label.setAlignment(Qt.AlignCenter)
        
    def process_with_rft(self, data):
        """
        Example method showing how to use RFT with fallback.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed result
        """
        if self.rft_engine:
            try:
                # Use RFT engine for processing
                result = self.rft_engine.process_quantum_field(data)
                return result
            except Exception as e:
                print(f"⚠️  RFT processing failed: {e}")
                return self.fallback_processing(data)
        else:
            # Use fallback processing
            return self.fallback_processing(data)
            
    def fallback_processing(self, data):
        """Fallback processing when RFT is not available"""
        # Implement your fallback logic here
        return data
        
    def closeEvent(self, event):
        """Handle application close with proper cleanup"""
        if self.rft_engine:
            try:
                # Cleanup RFT engine resources
                # (Add any specific cleanup needed)
                print("🧹 Cleaning up RFT resources")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
        
        event.accept()


def main():
    """Application entry point"""
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MyNewApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
```

## Step 4: Create an Icon

Create an SVG icon for your application at `os/apps/my_new_app/resources/icon.svg`.

You can use a simple example icon:

```svg
<?xml version="1.0" encoding="UTF-8"?>
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="28" fill="#3498db"/>
  <text x="32" y="40" font-size="24" fill="white" text-anchor="middle" font-family="Arial">MN</text>
</svg>
```

## Step 5: Register with Desktop Manager

Add your application to the desktop launcher by editing `os/frontend/quantonium_desktop.py`.

Find the `APPLICATIONS` list and add your app:

```python
APPLICATIONS = [
    # ... existing apps ...
    {
        'name': 'My New App',
        'description': 'Description of what it does',
        'icon': 'os/apps/my_new_app/resources/icon.svg',
        'path': 'os/apps/my_new_app/main.py',
        'category': 'tools'  # or 'system', 'developer', etc.
    }
]
```

## Step 6: Test Your Application

### Standalone Testing

First, test your application standalone:

```bash
python os/apps/my_new_app/main.py
```

### Desktop Integration Testing

Then test it through the desktop manager:

```bash
python quantonium_boot.py
# Click on your app icon in the desktop
```

## Step 7: Add Tests

Create tests for your application:

```bash
mkdir -p tests/apps/my_new_app
touch tests/apps/my_new_app/test_main.py
```

Example test file:

```python
import pytest
from quantonium_os_src.apps.my_new_app.main import MyNewApp
from PyQt5.QtWidgets import QApplication

@pytest.fixture
def app():
    """Create QApplication for testing"""
    import sys
    return QApplication(sys.argv)

def test_app_creation(app):
    """Test that the app can be created"""
    window = MyNewApp()
    assert window is not None
    assert window.windowTitle() == "My New App — QuantoniumOS"

def test_rft_fallback(app):
    """Test that app works without RFT"""
    window = MyNewApp()
    result = window.process_with_rft([1, 2, 3])
    assert result is not None
```

Run your tests:

```bash
pytest tests/apps/my_new_app/
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
try:
    result = self.rft_engine.process(data)
except Exception as e:
    self.log_error(f"Processing failed: {e}")
    self.show_error_dialog("Operation failed", str(e))
    return None
```

### 2. User Feedback

Provide clear feedback to users:

```python
def long_running_task(self):
    # Show progress
    progress = QProgressDialog("Processing...", "Cancel", 0, 100, self)
    progress.show()
    
    try:
        for i in range(100):
            # Do work
            progress.setValue(i)
            QApplication.processEvents()  # Keep UI responsive
    finally:
        progress.close()
```

### 3. Settings Persistence

Save user preferences:

```python
from PyQt5.QtCore import QSettings

def save_settings(self):
    settings = QSettings("QuantoniumOS", "MyNewApp")
    settings.setValue("window_geometry", self.saveGeometry())
    settings.setValue("user_preference", self.preference_value)

def load_settings(self):
    settings = QSettings("QuantoniumOS", "MyNewApp")
    geometry = settings.value("window_geometry")
    if geometry:
        self.restoreGeometry(geometry)
```

### 4. Documentation

Document your code thoroughly:

```python
def complex_function(self, param1, param2):
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ValueError: Description of when this is raised
        RuntimeError: Description of when this is raised
    """
```

## Troubleshooting

### App doesn't appear in desktop

1. Check that the path in `APPLICATIONS` is correct
2. Verify the icon SVG file exists
3. Check console for error messages

### RFT engine not initializing

1. Verify assembly kernels are compiled: `make -C src/assembly all`
2. Check Python bindings are installed
3. Look for import errors in console

### UI not responding

1. Ensure long operations are in separate threads
2. Call `QApplication.processEvents()` in loops
3. Use `QThread` for background tasks

## Next Steps

- Add your app to the documentation
- Create a pull request with your changes
- Share your app with the community!

## Examples

Check out these existing applications for reference:

- **Simple App**: `os/apps/q_notes/` - Basic text editor
- **Complex App**: `os/apps/symbolic_simulator/` - Full-featured simulator
- **RFT Integration**: `os/apps/visualizers/` - Heavy RFT usage
