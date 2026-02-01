# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Scientific Minimal Design with SVG Icons
======================================================
PhD-level UI/UX for quantum computing research platform
"""

import sys
import os
import math
import subprocess
import importlib.util
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsProxyWidget,
                            QPushButton, QLabel, QDesktopWidget, QGraphicsRectItem, QVBoxLayout,
                            QGraphicsLineItem, QGraphicsObject)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QBrush, QCursor, 
                        QPainterPath)
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem


class GoldenSpiralLoader(QGraphicsObject):
    """Radial spiral intro that mirrors the legacy solid-line animation."""

    finished = pyqtSignal()

    def __init__(self, diameter, parent=None):
        super().__init__(parent)

        self.spiral_count = 8
        self.points_per_arm = 60
        self.arm_delay_ms = 80
        self.fade_in_duration = 1500
        self.hold_duration = 1000
        self.fade_out_duration = 1000
        self.total_spiral_duration = (
            self.fade_in_duration + self.hold_duration + self.fade_out_duration
        )

        self.max_radius = min(220.0, max(120.0, diameter * 0.33))
        self.stroke_width = max(2.0, diameter * 0.007)

        self.elapsed_ms = 0
        self._fade_out_anim = None
        self._fade_out_started = False
        self._finish_emitted = False

        self.spiral_paths = self._build_spiral_paths()

        self.timer_interval = 50
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(self.timer_interval)

        self.setOpacity(0.0)
        self._fade_in_anim = QPropertyAnimation(self, b"opacity")
        self._fade_in_anim.setDuration(600)
        self._fade_in_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.start()

    def boundingRect(self):
        diameter = (self.max_radius + self.stroke_width) * 2.0
        return QRectF(-diameter / 2.0, -diameter / 2.0, diameter, diameter)

    def fade_out(self, delay_ms=0):
        if self._fade_out_started:
            return

        self._fade_out_started = True
        if delay_ms > 0:
            QTimer.singleShot(delay_ms, self._begin_fade_out)
        else:
            self._begin_fade_out()

    def _begin_fade_out(self):
        if self._fade_out_anim is not None:
            return

        self._fade_out_anim = QPropertyAnimation(self, b"opacity")
        self._fade_out_anim.setDuration(700)
        self._fade_out_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._fade_out_anim.setStartValue(self.opacity())
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.finished.connect(self._handle_fade_finished)
        self._fade_out_anim.start()

    def _handle_fade_finished(self):
        self.timer.stop()
        self.hide()
        if not self._finish_emitted:
            self._finish_emitted = True
            self.finished.emit()

    def update_animation(self):
        self.elapsed_ms += self.timer_interval

        if not self._fade_out_started and self.elapsed_ms > self.total_spiral_duration + 500:
            self.fade_out()

        self.update()

    def _build_spiral_paths(self):
        paths = []
        phi = 1.618033988749895
        for arm in range(self.spiral_count):
            path = QPainterPath()
            for idx in range(self.points_per_arm):
                t = idx / max(1, self.points_per_arm - 1)
                angle = arm * (2 * math.pi / self.spiral_count) + t * math.pi * 4
                radius = (phi ** (t * 0.2)) * t * self.max_radius
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                if idx == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            paths.append(path)
        return paths

    def _arm_alpha(self, arm_index):
        delay = arm_index * self.arm_delay_ms
        adjusted = self.elapsed_ms - delay
        if adjusted <= 0:
            return 0.0

        if adjusted < self.fade_in_duration:
            return adjusted / self.fade_in_duration

        if adjusted < self.fade_in_duration + self.hold_duration:
            pulse = 0.5 + 0.3 * math.sin(self.elapsed_ms * 0.003 + arm_index * 0.5)
            return pulse

        if adjusted < self.total_spiral_duration:
            fade_progress = (
                (adjusted - self.fade_in_duration - self.hold_duration)
                / self.fade_out_duration
            )
            base = max(0.0, 1.0 - fade_progress)
            pulse = 0.3 + 0.2 * math.sin(self.elapsed_ms * 0.003 + arm_index * 0.5)
            return base * pulse

        return 0.0

    def paint(self, painter, option, widget=None):
        if self.opacity() <= 0.0:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        base_color = QColor("#2471a3")

        for arm_index, path in enumerate(self.spiral_paths):
            alpha = max(0.0, min(1.0, self._arm_alpha(arm_index)))
            if alpha <= 0.0:
                continue

            color = QColor(base_color)
            color.setAlphaF(alpha)
            pen = QPen(color, self.stroke_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        painter.restore()

class QuantoniumDesktop(QMainWindow):
    """
    Scientific Desktop Manager for Quantum Computation Environment
    with mathematically precise Golden Ratio proportions
    """
    
    def __init__(self, show_immediately=True):
        super().__init__()
        
        # Mathematical constants for design precision (Golden Ratio)
        self.phi = 1.618033988749895  # ╧å (Golden Ratio)
        self.phi_sq = self.phi * self.phi  # ╧å┬▓
        self.phi_inv = 1 / self.phi  # 1/╧å
        self.base_unit = 16  # Base unit for scaling (multiply by ╧å powers for harmony)
        
        # Setup UI
        self.setWindowTitle("QuantoniumOS")
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Start transparent for cross-fade
        self.setWindowOpacity(0.0)
        
        # Only show immediately if requested
        if show_immediately:
            self.showFullScreen()
            self.setWindowOpacity(1.0)  # If showing immediately, make it visible
        
        # Graphics View setup with absolute precision
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.TextAntialiasing)
        
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)
        
        # Configure view for minimal design
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.view.setRenderHint(QPainter.Antialiasing)
        
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        self.scene.setSceneRect(0, 0, screen.width(), screen.height())

        self.intro_complete = False
        self.status_hint = None

        # Initialize scientific interface
        self.is_arch_expanded = False
        self.app_items = []
        self.arch_background = None
        self.panel_trigger = self.create_panel_trigger()
        self.quantum_logo_items = self.create_quantum_logo()
        self.quantum_logo = self.quantum_logo_items["outer"]
        self.system_time = self.create_system_time()
        self.system_status = self.create_system_status()
        self.spiral_loader = self.create_spiral_loader()
        
        # Initialize timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()
        
        self.load_scientific_apps()
    
    def create_panel_trigger(self):
        """Create quantum logo as expandable arch trigger"""
        screen = QDesktopWidget().screenGeometry()
        
        # Create circular trigger zone around the quantum logo
        trigger = QGraphicsEllipseItem(screen.width()//2 - 60, screen.height()//2 - 60, 120, 120)
        trigger.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        trigger.setPen(QPen(QColor(0, 0, 0, 0), 0))    # No border
        
        # Add interaction
        trigger.setAcceptedMouseButtons(Qt.LeftButton)
        trigger.mousePressEvent = self.toggle_arch
        trigger.setCursor(QCursor(Qt.PointingHandCursor))
        trigger.setEnabled(False)
        
        self.scene.addItem(trigger)
        return trigger
        
    def create_quantum_logo(self):
        """Create mathematically precise Q logo using Golden Ratio proportions"""
        screen = QDesktopWidget().screenGeometry()
        
        # Center coordinates
        center_x = screen.width() / 2
        center_y = screen.height() / 2
        
        # Q Logo dimensions scaled by Golden Ratio
        outer_radius = self.base_unit * self.phi_sq  # ╧å┬▓ scaling for outer circle
        inner_radius = outer_radius * self.phi_inv   # 1/╧å scaling for inner circle
        stroke_width = self.base_unit * 0.15         # Precise stroke scaling
        
        # Create outer circle
        outer_circle = QGraphicsEllipseItem(
            center_x - outer_radius, 
            center_y - outer_radius,
            outer_radius * 2, 
            outer_radius * 2
        )
        
        # Create inner circle (precise proportion of outer)
        inner_circle = QGraphicsEllipseItem(
            center_x - inner_radius, 
            center_y - inner_radius,
            inner_radius * 2, 
            inner_radius * 2
        )
        
        # Create diagonal line (mathematically placed)
        line_length = outer_radius * self.phi_inv  # 1/╧å scaling
        line_angle = math.radians(45)  # 45 degrees in radians
        
        # Calculate line endpoints with precise geometric harmony
        line_dx = math.cos(line_angle) * line_length
        line_dy = math.sin(line_angle) * line_length
        
        # Apply Golden Ratio offset for the line
        offset = inner_radius * self.phi_inv * 0.5
        
        # Q dash line - positioned to create "Q" from "O"
        dash_start_x = center_x + offset
        dash_start_y = center_y + offset
        dash_end_x = dash_start_x + line_dx
        dash_end_y = dash_start_y + line_dy
        
        # Styling with minimal, precise aesthetics
        pen = QPen(QColor("#3498db"), stroke_width)
        pen.setCapStyle(Qt.RoundCap)
        
        outer_circle.setPen(pen)
        outer_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent fill
        
        inner_circle.setPen(pen)
        inner_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent fill
        
        # Create Q dash line
        q_dash = QGraphicsLineItem(dash_start_x, dash_start_y, dash_end_x, dash_end_y)
        q_dash.setPen(pen)
        
        # Add to scene with perfect layering
        self.scene.addItem(outer_circle)
        self.scene.addItem(inner_circle)
        self.scene.addItem(q_dash)

        for item in (outer_circle, inner_circle, q_dash):
            item.setOpacity(0.0)
        
        # Return for reference
        return {
            "outer": outer_circle,
            "inner": inner_circle,
            "dash": q_dash,
        }
    
    def create_system_time(self):
        """Create mathematically precise time display using Golden Ratio"""
        screen = QDesktopWidget().screenGeometry()
        
        time_text = QGraphicsTextItem()
        time_text.setDefaultTextColor(QColor("#34495e"))
        
        # Font size based on golden ratio proportions
        time_font_size = int(self.base_unit * self.phi_inv)  # 1/╧å scaling
        font = QFont("SF Mono", time_font_size, QFont.Normal)
        if not font.exactMatch():
            font = QFont("Consolas", time_font_size, QFont.Normal)
        
        time_text.setFont(font)
        
        # Position with golden ratio margins
        margin_x = self.base_unit * self.phi_sq  # ╧å┬▓ margin from edge
        margin_y = self.base_unit * self.phi_inv  # 1/╧å margin from top
        time_text.setPos(screen.width() - margin_x - 120, margin_y)
        time_text.setOpacity(0.0)
        
        self.scene.addItem(time_text)
        return time_text
        
    def create_system_status(self):
        """Create mathematically precise system status using Golden Ratio"""
        screen = QDesktopWidget().screenGeometry()
        center_x = screen.width() / 2
        center_y = screen.height() / 2
        
        status_text = QGraphicsTextItem("QUANTONIUMOS")
        status_text.setDefaultTextColor(QColor("#3498db"))
        
        # Font size based on golden ratio
        status_font_size = int(self.base_unit * self.phi_inv * 0.8)  # Smaller than main logo
        font = QFont("SF Pro Display", status_font_size, QFont.Medium)
        if not font.exactMatch():
            font = QFont("Segoe UI", status_font_size, QFont.Normal)
        font.setLetterSpacing(QFont.AbsoluteSpacing, self.base_unit * 0.08)
        
        status_text.setFont(font)
        
        # Position at the bottom of the screen, near the footer
        status_bounds = status_text.boundingRect()
        status_x = center_x - (status_bounds.width() / 2)  # Center horizontally
        status_y = screen.height() - status_bounds.height() - (self.base_unit * 2.5)  # Near bottom but above taskbar
        status_text.setPos(status_x, status_y)
        
        # Add subtle interaction hint with precise positioning
        hint_text = QGraphicsTextItem("Click Q to show apps")
        hint_text.setDefaultTextColor(QColor(52, 73, 94, 120))  # Semi-transparent
        
        hint_font_size = int(self.base_unit * self.phi_inv * 0.6)  # Even smaller
        hint_font = QFont("SF Pro Display", hint_font_size, QFont.Normal)
        if not hint_font.exactMatch():
            hint_font = QFont("Segoe UI", hint_font_size, QFont.Normal)
        
        hint_text.setFont(hint_font)
        
        # Position hint below the Q logo in the center
        hint_bounds = hint_text.boundingRect()
        hint_x = center_x - (hint_bounds.width() / 2)
        hint_y = center_y + (self.base_unit * self.phi)
        
        self.scene.addItem(status_text)
        self.scene.addItem(hint_text)

        status_text.setOpacity(0.0)
        hint_text.setOpacity(0.0)
        self.status_hint = hint_text

        return status_text

    def create_spiral_loader(self):
        """Create and schedule the golden spiral boot loader animation."""
        screen = QDesktopWidget().screenGeometry()
        diameter = min(screen.width(), screen.height()) * 0.55
        loader = GoldenSpiralLoader(diameter)
        loader.setPos(screen.width() / 2, screen.height() / 2)
        loader.setZValue(25)
        loader.finished.connect(self.on_intro_finished)
        self.scene.addItem(loader)
        return loader

    def on_intro_finished(self):
        """Reveal the main interface once the intro animation completes."""
        if self.intro_complete:
            return

        self.intro_complete = True

        if self.panel_trigger:
            self.panel_trigger.setEnabled(True)

        if self.quantum_logo_items:
            for item in self.quantum_logo_items.values():
                item.setOpacity(1.0)

        if self.system_time is not None:
            self.system_time.setOpacity(1.0)

        if self.system_status is not None:
            self.system_status.setOpacity(1.0)

        if self.status_hint is not None:
            self.status_hint.setOpacity(0.6)
    
    def update_time(self):
        current_time = datetime.now().strftime("%H:%M")
        self.system_time.setPlainText(current_time)
    
    def load_scientific_apps(self):
        """Load applications with scientific metadata"""
        # Get the project root directory (quantoniumos/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.apps = [
            {
                "name": "RFT Validation Suite", 
                "path": os.path.join(project_root, "src", "apps", "rft_validation_suite.py"), 
                "category": "ANALYSIS",
                "description": "Mathematical validation framework",
                "icon": "rft_validator.svg"
            },
            {
                "name": "AI Chat", 
                "path": os.path.join(project_root, "src", "apps", "qshll_chatbox.py"), 
                "category": "AI", 
                "description": "Quantum-enhanced AI assistant",
                "icon": "ai_chat.svg"
            },
            {
                "name": "Quantum Simulator", 
                "path": os.path.join(project_root, "src", "apps", "symbolic_simulator.py"), 
                "category": "SIMULATION",
                "description": "Quantum circuit modeling",
                "icon": "symbolic_simulator.svg"
            },
            {
                "name": "Quantum Cryptography", 
                "path": os.path.join(project_root, "src", "apps", "quantum_inspired_crypto.py"), 
                "category": "SECURITY",
                "description": "Cryptographic protocols",
                "icon": "quantum_inspired_crypto.svg"
            },
            {
                "name": "System Monitor", 
                "path": os.path.join(project_root, "src", "apps", "qshll_system_monitor.py"), 
                "category": "SYSTEM",
                "description": "Resource monitoring",
                "icon": "system_monitor.svg"
            },
            {
                "name": "Q-Notes", 
                "path": os.path.join(project_root, "src", "apps", "q_notes.py"), 
                "category": "RESEARCH",
                "description": "Research documentation",
                "icon": "q_notes.svg"
            },
            {
                "name": "Q-Vault", 
                "path": os.path.join(project_root, "src", "apps", "q_vault.py"), 
                "category": "DATA",
                "description": "Secure data storage",
                "icon": "q_vault.svg"
            },
            # === QuantSoundDesign ===
            {
                "name": "QuantSoundDesign",
                "path": os.path.join(project_root, "src", "apps", "quantsounddesign", "gui.py"),
                "category": "STUDIO",
                "description": "Φ-RFT Sound Design Studio",
                "icon": "quantsounddesign.svg"
            }
        ]
    
    def toggle_arch(self, event):
        """Toggle expandable arch display"""
        print(f"Toggle arch called. Current state: expanded={self.is_arch_expanded}")
        if not self.is_arch_expanded:
            print("Expanding arch...")
            self.expand_arch()
        else:
            print("Collapsing arch...")
            self.collapse_arch()
    
    def expand_arch(self):
        """Expand applications in arch formation with SVG icons"""
        print("DEBUG: expand_arch function called")
        self.is_arch_expanded = True
        
        # Clear existing items
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        
        screen = QDesktopWidget().screenGeometry()
        center_x = screen.width() // 2
        center_y = screen.height() // 2
        
        # Dimensions to match the provided image
        arch_radius = 280  # Adjusted radius to match image
        button_size = 80   # Square app buttons
        font_size = 10     # Better readable font size
        
        # Create outer circle background
        arch_bg = QGraphicsEllipseItem(center_x - arch_radius, center_y - arch_radius, 
                                      arch_radius * 2, arch_radius * 2)
        arch_bg.setBrush(QBrush(QColor(255, 255, 255, 10)))  # Very subtle overlay
        arch_bg.setPen(QPen(QColor(52, 152, 219, 30), 1))    # Subtle border
        
        # Create inner circle for aesthetics (matching image)
        inner_radius = arch_radius * 0.4
        inner_circle = QGraphicsEllipseItem(center_x - inner_radius, center_y - inner_radius,
                                           inner_radius * 2, inner_radius * 2)
        inner_circle.setBrush(QBrush(QColor(255, 255, 255, 5)))  # Very subtle fill
        inner_circle.setPen(QPen(QColor(52, 152, 219, 20), 1))   # Very subtle border
        
        self.scene.addItem(arch_bg)
        self.scene.addItem(inner_circle)
        self.app_items.append(arch_bg)
        self.app_items.append(inner_circle)
        self.arch_background = arch_bg
        
        # Position apps in circular layout around Q
        num_apps = len(self.apps)
        
        # Using a partial circle for apps (top half plus sides)
        angle_step = math.pi * 1.5 / (num_apps - 1) if num_apps > 1 else 0
        start_angle = math.pi * 1.75  # Start from top-left (315 degrees)
        
        for i, app in enumerate(self.apps):
            # Calculate position on the circle
            angle = start_angle - i * angle_step
            x = center_x + arch_radius * math.cos(angle)
            y = center_y + arch_radius * math.sin(angle)
            
            # Get app name and information
            app_name = app["name"]
            app_path = app["path"]
            app_icon = app.get("icon", None)
            display_name = app_name.split()[-1] if len(app_name.split()) > 1 else app_name
            
            # Create app button with background
            app_rect = QGraphicsRectItem(x - button_size/2, y - button_size/2, button_size, button_size)
            app_rect.setBrush(QBrush(QColor(255, 255, 255, 128)))
            app_rect.setPen(QPen(QColor(52, 152, 219, 51), 1))
            app_rect.setAcceptedMouseButtons(Qt.LeftButton)
            app_rect.setCursor(QCursor(Qt.PointingHandCursor))
            app_rect.mousePressEvent = lambda event, path=app_path, name=app_name: self.launch_app({"path": path, "name": name})
            
            self.scene.addItem(app_rect)
            self.app_items.append(app_rect)
            
            # Add SVG icon if available
            if app_icon:
                icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui", "icons", app_icon)
                if os.path.exists(icon_path):
                    # Create SVG item
                    svg_item = QGraphicsSvgItem(icon_path)
                    icon_size = button_size * 0.7  # Slightly smaller than button
                    
                    # Calculate scale factor to fit icon within the desired size
                    svg_bounds = svg_item.boundingRect()
                    scale_factor = min(icon_size / svg_bounds.width(), icon_size / svg_bounds.height())
                    svg_item.setScale(scale_factor)
                    
                    # Center the icon on the button
                    svg_x = x - (svg_bounds.width() * scale_factor / 2)
                    svg_y = y - (svg_bounds.height() * scale_factor / 2)
                    svg_item.setPos(svg_x, svg_y)
                    
                    self.scene.addItem(svg_item)
                    self.app_items.append(svg_item)
                else:
                    print(f"Icon not found: {icon_path}")
            
            # Add text label below icon
            label = QGraphicsTextItem(display_name)
            label.setDefaultTextColor(QColor("#2c3e50"))
            
            # Font settings
            font = QFont("Segoe UI", font_size)
            font.setWeight(QFont.Medium)
            label.setFont(font)
            
            # Center text relative to icon position
            label_bounds = label.boundingRect()
            label_x = x - (label_bounds.width() / 2)
            
            # Position labels consistently below icons for clarity
            label_y = y + (button_size / 2) + 5
            label.setPos(label_x, label_y)
            
            self.scene.addItem(label)
            self.app_items.append(label)
        
        print(f"Expanded arch with {len(self.apps)} applications - SVG version")
        
    def collapse_arch(self):
        """Collapse arch formation"""
        self.is_arch_expanded = False
        
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        self.arch_background = None
        
        print("Collapsed application arch")
    
    def launch_app(self, app_data):
        """Launch application within QuantoniumOS environment"""
        app_name = app_data["name"]
        app_path = app_data["path"]
        
        print(f"Launching: {app_name}")
        self.collapse_arch()
        
        try:
            # Update status with mathematical precision
            self.system_status.setPlainText(f"LAUNCHING {app_name.upper()}")
            
            # Get the project root directory for imports
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # Add the apps directory to Python path for imports
            apps_dir = os.path.join(project_root, "src", "apps")
            if apps_dir not in sys.path:
                sys.path.insert(0, apps_dir)
            
            # Launch app based on type
            if "q_notes" in app_path.lower():
                self.launch_q_notes()
            elif "q_vault" in app_path.lower():
                self.launch_q_vault()
            elif "symbolic_simulator" in app_path.lower():
                self.launch_symbolic_simulator()
            elif "quantum_inspired_crypto" in app_path.lower():
                self.launch_quantum_inspired_crypto()
            elif "qshll_system_monitor" in app_path.lower():
                self.launch_system_monitor()
            elif "qshll_chatbox" in app_path.lower():
                self.launch_ai_chat()
            elif "rft_validation" in app_path.lower():
                self.launch_rft_validator()
            # QuantSoundDesign
            elif "quantsounddesign" in app_path.lower():
                self.launch_quantsounddesign()
            else:
                # Fallback to subprocess for unknown apps
                self.launch_app_subprocess(app_data)
                return
            
            print(f"Successfully launched {app_name} within QuantoniumOS")
            
            # Reset status after delay using golden ratio timing (╧å seconds)
            reset_delay = int(self.phi * 1000)  # ╧å seconds in milliseconds
            QTimer.singleShot(reset_delay, lambda: self.system_status.setPlainText("QUANTONIUMOS"))
            
        except Exception as e:
            print(f"Error launching {app_name}: {e}")
            import traceback
            traceback.print_exc()
            self.system_status.setPlainText("LAUNCH ERROR")
            QTimer.singleShot(int(self.phi_sq * 1000), lambda: self.system_status.setPlainText("QUANTONIUMOS"))
    
    def launch_app_subprocess(self, app_data):
        """Fallback subprocess launcher for apps that can't be integrated"""
        import subprocess
        app_name = app_data["name"]
        app_path = app_data["path"]
        
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            process = subprocess.Popen(
                [sys.executable, app_path], 
                cwd=project_root,
                stdout=subprocess.DEVNULL if "qshll_chatbox" in app_path.lower() else subprocess.PIPE,
                stderr=subprocess.DEVNULL if "qshll_chatbox" in app_path.lower() else subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' and "qshll_chatbox" in app_path.lower() else 0
            )
            print(f"Successfully launched {app_name} (PID: {process.pid})")
        except Exception as e:
            print(f"Error in subprocess launch for {app_name}: {e}")
    
    def launch_q_notes(self):
        """Launch Q-Notes within the OS environment"""
        try:
            # Import the Q-Notes module
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            q_notes_path = os.path.join(project_root, "src", "apps", "q_notes.py")
            
            spec = importlib.util.spec_from_file_location("q_notes", q_notes_path)
            q_notes_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(q_notes_module)
            
            # Create the Q-Notes window without creating a new QApplication
            self.q_notes_window = q_notes_module.QNotes()
            self.q_notes_window.show()
            
        except Exception as e:
            print(f"Error launching Q-Notes: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_q_vault(self):
        """Launch Q-Vault within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            q_vault_path = os.path.join(project_root, "src", "apps", "q_vault.py")
            
            spec = importlib.util.spec_from_file_location("q_vault", q_vault_path)
            q_vault_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(q_vault_module)
            
            self.q_vault_window = q_vault_module.QVault()
            self.q_vault_window.show()
            
        except Exception as e:
            print(f"Error launching Q-Vault: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_symbolic_simulator(self):
        """Launch Quantum Simulator within the OS environment"""
        try:
            print("≡ƒÜÇ Launching Quantum Simulator...")
            
            # Use direct import method since subprocess fails on Windows
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            simulator_path = os.path.join(project_root, "src", "apps", "symbolic_simulator.py")
            
            print(f"≡ƒôü Loading simulator from: {simulator_path}")
            
            # Add apps directory to path
            apps_dir = os.path.join(project_root, "src", "apps")
            if apps_dir not in sys.path:
                sys.path.insert(0, apps_dir)
            
            # Check if file exists
            if not os.path.exists(simulator_path):
                print(f"Γ¥î Simulator file not found: {simulator_path}")
                return
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("symbolic_simulator", simulator_path)
            simulator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(simulator_module)
            
            print("Γ£à Simulator module loaded successfully")
            
            # Create and show the simulator window
            self.symbolic_simulator_window = simulator_module.RFTQuantumSimulator()
            
            # Force window to be visible and on top
            self.symbolic_simulator_window.show()
            self.symbolic_simulator_window.raise_()
            self.symbolic_simulator_window.activateWindow()
            
            # Make it stay on top temporarily
            from PyQt5.QtCore import Qt
            self.symbolic_simulator_window.setWindowFlags(
                self.symbolic_simulator_window.windowFlags() | Qt.WindowStaysOnTopHint
            )
            self.symbolic_simulator_window.show()  # Show again after flag change
            
            # Move to center of screen to make sure it's visible
            screen = self.symbolic_simulator_window.screen().geometry()
            window = self.symbolic_simulator_window.geometry()
            x = (screen.width() - window.width()) // 2
            y = (screen.height() - window.height()) // 2
            self.symbolic_simulator_window.move(x, y)
            
            print("Γ£à Quantum Simulator launched successfully!")
            print(f"≡ƒûÑ∩╕Å Window position: {x}, {y}")
            print("≡ƒöì Window should be visible on top of all other windows!")
            
        except Exception as e:
            print(f"Γ¥î Error launching Quantum Simulator: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_quantum_inspired_crypto(self):
        """Launch Quantum Cryptography within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            crypto_path = os.path.join(project_root, "src", "apps", "quantum_inspired_crypto.py")
            
            spec = importlib.util.spec_from_file_location("quantum_inspired_crypto", crypto_path)
            crypto_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(crypto_module)
            
            # Find the main class (it's called QuantumCrypto)
            if hasattr(crypto_module, 'QuantumCrypto'):
                self.quantum_inspired_crypto_window = crypto_module.QuantumCrypto()
                self.quantum_inspired_crypto_window.show()
            else:
                print("Unknown quantum crypto class structure")
                return
                
            self.quantum_inspired_crypto_window.show()
            
        except Exception as e:
            print(f"Error launching Quantum Cryptography: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_system_monitor(self):
        """Launch System Monitor within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            monitor_path = os.path.join(project_root, "src", "apps", "qshll_system_monitor.py")
            
            spec = importlib.util.spec_from_file_location("qshll_system_monitor", monitor_path)
            monitor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(monitor_module)
            
            # Find the main class
            if hasattr(monitor_module, 'SystemMonitor'):
                self.system_monitor_window = monitor_module.SystemMonitor()
                self.system_monitor_window.show()
            else:
                print("Unknown system monitor class structure")
                return
            
        except Exception as e:
            print(f"Error launching System Monitor: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_ai_chat(self):
        """Launch AI Chat within the OS environment with quantum AI integration"""
        try:
            print("≡ƒñû Launching AI Chat with Quantum AI System...")
            
            # Get project root and set up paths
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            chatbox_path = os.path.join(project_root, "src", "apps", "qshll_chatbox.py")
            
            print(f"≡ƒôü Loading chatbox from: {chatbox_path}")
            
            # Add project root to Python path for quantum AI imports
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Check if file exists
            if not os.path.exists(chatbox_path):
                print(f"Γ¥î Chatbox file not found: {chatbox_path}")
                return
            
            # Import the chatbox module
            spec = importlib.util.spec_from_file_location("qshll_chatbox", chatbox_path)
            chat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chat_module)
            
            print("Γ£à Chatbox module loaded successfully")
            
            # Create and show the chatbox window
            self.ai_chat_window = chat_module.Chatbox()
            
            # Force window to be visible and on top
            self.ai_chat_window.show()
            self.ai_chat_window.raise_()
            self.ai_chat_window.activateWindow()
            
            print("Γ£à AI Chat launched successfully with Quantum AI integration!")
            
        except Exception as e:
            print(f"Γ¥î Error launching AI Chat: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_rft_validator(self):
        """Launch RFT Validator within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            validator_path = os.path.join(project_root, "src", "apps", "rft_validation_suite.py")
            
            spec = importlib.util.spec_from_file_location("rft_validation_suite", validator_path)
            validator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validator_module)
            
            # Find the main class
            if hasattr(validator_module, 'RFTValidationSuite'):
                self.rft_validator_window = validator_module.RFTValidationSuite()
                self.rft_validator_window.show()
            else:
                print("Unknown RFT validator class structure")
                return
            
        except Exception as e:
            print(f"Error launching RFT Validator: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_ai_chat(self):
        """Launch AI Chat within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            chat_path = os.path.join(project_root, "src", "apps", "qshll_chatbox.py")
            
            spec = importlib.util.spec_from_file_location("qshll_chatbox", chat_path)
            chat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chat_module)
            
            # The class is called 'Chatbox'
            if hasattr(chat_module, 'Chatbox'):
                self.ai_chat_window = chat_module.Chatbox()
                self.ai_chat_window.show()
            else:
                print("Unknown AI chat class structure")
                return
            
        except Exception as e:
            print(f"Error launching AI Chat: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_rft_validator(self):
        """Launch RFT Validator within the OS environment"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            rft_path = os.path.join(project_root, "src", "apps", "rft_validation_suite.py")
            
            spec = importlib.util.spec_from_file_location("rft_validation_suite", rft_path)
            rft_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rft_module)
            
            # Find the main class (it's called RFTValidationSuite)
            if hasattr(rft_module, 'RFTValidationSuite'):
                self.rft_validator_window = rft_module.RFTValidationSuite()
                self.rft_validator_window.show()
            else:
                print("Unknown RFT validator class structure")
                return
                
            self.rft_validator_window.show()
            
        except Exception as e:
            print(f"Error launching RFT Validator: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_quantsounddesign(self):
        """Launch QuantSoundDesign - Φ-RFT Sound Design Studio"""
        try:
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.apps.quantsounddesign.gui import QuantSoundDesign
            self.quantsounddesign_window = QuantSoundDesign()
            self.quantsounddesign_window.show()
        except Exception as e:
            print(f"Error launching QuantSoundDesign: {e}")
            import traceback
            traceback.print_exc()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
        super().keyPressEvent(event)
    
    def start_fadein(self):
        """Begin cross-fade transition - fade in the desktop"""
        self.showFullScreen()  # Show the window
        
        self.fadein_timer = QTimer()
        self.fadein_timer.timeout.connect(self.fadein_step)
        self.fadein_opacity = 0.0
        self.fadein_timer.start(16)  # ~60fps
    
    def fadein_step(self):
        """Gradually fade in the desktop"""
        self.fadein_opacity += 0.02  # Fade in over ~1.25 seconds
        
        if self.fadein_opacity >= 1.0:
            self.fadein_opacity = 1.0
            self.fadein_timer.stop()
        
        self.setWindowOpacity(self.fadein_opacity)

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("SF Pro Display", 10)
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    desktop = QuantoniumDesktop()
    desktop.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

