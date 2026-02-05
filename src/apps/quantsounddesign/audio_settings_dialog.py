#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Audio Settings Dialog & Performance Meters

EPIC 1: User-facing audio configuration with:
- Device selection (input/output)
- Sample rate and buffer size configuration
- Latency mode presets with real-time latency display
- Performance monitoring (XRuns, CPU load)
- Persistent settings that survive restarts

This is the in-development audio settings UI.
"""

from typing import Optional, Callable
from dataclasses import dataclass

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
        QPushButton, QComboBox, QGroupBox, QCheckBox, QSlider,
        QFrame, QWidget, QSpacerItem, QSizePolicy, QProgressBar,
        QMessageBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QFont
    HAS_QT = True
except ImportError:
    HAS_QT = False

# Import audio backend components
try:
    from .audio_backend import (
        AudioSettings, AudioDeviceInfo, LatencyMode, SampleRate,
        PerformanceStats, get_audio_devices, get_audio_backend,
        get_audio_settings, set_audio_settings, save_audio_config,
        get_performance_stats, BUFFER_SIZES
    )
except ImportError:
    from audio_backend import (
        AudioSettings, AudioDeviceInfo, LatencyMode, SampleRate,
        PerformanceStats, get_audio_devices, get_audio_backend,
        get_audio_settings, set_audio_settings, save_audio_config,
        get_performance_stats, BUFFER_SIZES
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DIALOG_STYLE = """
QDialog {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

QGroupBox {
    background-color: #222222;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #00aaff;
}

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e0e0e0;
    min-width: 200px;
}

QComboBox:hover {
    border-color: #00aaff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QLabel {
    color: #c0c0c0;
}

QPushButton {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 8px 16px;
    color: #e0e0e0;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #00aaff;
    border-color: #00aaff;
    color: white;
}

QPushButton:pressed {
    background-color: #0088cc;
}

QPushButton#applyButton {
    background-color: #00aa55;
    border-color: #00aa55;
}

QPushButton#applyButton:hover {
    background-color: #00cc66;
}

QCheckBox {
    color: #c0c0c0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    background-color: #2a2a2a;
}

QCheckBox::indicator:checked {
    background-color: #00aaff;
    border-color: #00aaff;
}

QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #00aaff;
    border-radius: 2px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO METER WIDGET - For status bar
# ═══════════════════════════════════════════════════════════════════════════════

class AudioMeterWidget(QWidget):
    """
    Compact audio performance meter for the status bar.
    
    Shows:
    - CPU load as a colored bar
    - XRun count (underruns/overruns)
    - Click to open audio settings
    """
    
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 24)
        self.setToolTip("Audio Performance - Click to open settings")
        
        # Performance data
        self.cpu_load = 0.0
        self.underruns = 0
        self.overruns = 0
        self.buffer_size = 256
        self.sample_rate = 48000
        
        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(100)  # Update 10x per second
    
    def _update_stats(self):
        """Update from global performance stats"""
        stats = get_performance_stats()
        if stats:
            self.cpu_load = stats.cpu_load_percent
            self.underruns = stats.underruns
            self.overruns = stats.overruns
        
        settings = get_audio_settings()
        if settings:
            self.buffer_size = settings.buffer_size
            self.sample_rate = settings.sample_rate
        
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#151515"))
        
        # CPU meter bar
        bar_x = 4
        bar_y = 4
        bar_width = 60
        bar_height = 16
        
        # Background bar
        painter.fillRect(bar_x, bar_y, bar_width, bar_height, QColor("#2a2a2a"))
        
        # CPU load bar with color gradient based on load
        load_width = int((self.cpu_load / 100.0) * bar_width)
        if self.cpu_load < 50:
            color = QColor("#00aa55")  # Green
        elif self.cpu_load < 75:
            color = QColor("#ffaa00")  # Yellow
        else:
            color = QColor("#ff4444")  # Red
        
        painter.fillRect(bar_x, bar_y, load_width, bar_height, color)
        
        # CPU text
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(bar_x, bar_y, bar_width, bar_height,
                        Qt.AlignCenter, f"{self.cpu_load:.0f}%")
        
        # XRun indicator
        xrun_x = bar_x + bar_width + 6
        xrun_count = self.underruns + self.overruns
        
        if xrun_count > 0:
            painter.setPen(QColor("#ff4444"))
            painter.drawText(xrun_x, bar_y, 40, bar_height,
                           Qt.AlignLeft | Qt.AlignVCenter, f"X:{xrun_count}")
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(xrun_x, bar_y, 40, bar_height,
                           Qt.AlignLeft | Qt.AlignVCenter, "X:0")
        
        # Buffer size indicator
        buf_x = xrun_x + 40
        latency_ms = (self.buffer_size / self.sample_rate) * 1000
        painter.setPen(QColor("#888888"))
        painter.drawText(buf_x, bar_y, 60, bar_height,
                        Qt.AlignLeft | Qt.AlignVCenter, f"{latency_ms:.1f}ms")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class AudioMeterWidgetExpanded(QWidget):
    """
    Expanded audio meter for mixer or dedicated performance panel.
    
    Shows detailed breakdown of:
    - Per-buffer CPU usage
    - Peak CPU usage
    - XRun history
    - Device info
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(200)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("Audio Performance")
        title.setStyleSheet("color: #00aaff; font-weight: bold;")
        layout.addWidget(title)
        
        # CPU meter
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setTextVisible(True)
        cpu_layout.addWidget(self.cpu_bar)
        layout.addLayout(cpu_layout)
        
        # Stats labels
        self.stats_label = QLabel("Avg: --  Peak: --")
        self.stats_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self.stats_label)
        
        # XRuns
        self.xrun_label = QLabel("XRuns: 0 underruns, 0 overruns")
        self.xrun_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.xrun_label)
        
        # Device info
        self.device_label = QLabel("Device: --")
        self.device_label.setStyleSheet("color: #555555; font-size: 10px;")
        layout.addWidget(self.device_label)
        
        # Reset button
        reset_btn = QPushButton("Reset Stats")
        reset_btn.clicked.connect(self._reset_stats)
        layout.addWidget(reset_btn)
    
    def _update_stats(self):
        stats = get_performance_stats()
        settings = get_audio_settings()
        
        if stats:
            self.cpu_bar.setValue(int(stats.cpu_load_percent))
            
            # Color based on load
            if stats.cpu_load_percent < 50:
                self.cpu_bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: #00aa55; }")
            elif stats.cpu_load_percent < 75:
                self.cpu_bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: #ffaa00; }")
            else:
                self.cpu_bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: #ff4444; }")
            
            self.stats_label.setText(
                f"Avg: {stats.callback_time_us:.0f}µs  "
                f"Peak: {stats.max_callback_time_us:.0f}µs  "
                f"Callbacks: {stats.callbacks_total}")
            
            xrun_color = "#ff4444" if (stats.underruns + stats.overruns) > 0 else "#888888"
            self.xrun_label.setStyleSheet(f"color: {xrun_color};")
            self.xrun_label.setText(
                f"XRuns: {stats.underruns} underruns, {stats.overruns} overruns")
        
        if settings:
            latency_ms = settings.get_latency_ms()
            self.device_label.setText(
                f"{settings.sample_rate}Hz / {settings.buffer_size} samples "
                f"({latency_ms:.1f}ms)")
    
    def _reset_stats(self):
        stats = get_performance_stats()
        if stats:
            stats.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════════════════════

class AudioSettingsDialog(QDialog):
    """
    In-development audio settings dialog.
    
    Features:
    - Input/output device selection with refresh
    - Sample rate configuration
    - Latency mode presets with real-time latency calculation
    - Manual buffer size override
    - Performance monitoring toggle
    - Apply without closing, or OK to apply and close
    - Settings persist to disk
    """
    
    settings_changed = pyqtSignal(object)  # Emits new AudioSettings
    
    def __init__(self, parent=None, current_settings: AudioSettings = None):
        super().__init__(parent)
        self.setWindowTitle("Audio Settings")
        self.setMinimumWidth(450)
        self.setStyleSheet(DIALOG_STYLE)
        
        # Store current settings
        self.original_settings = current_settings or get_audio_settings()
        self.current_settings = AudioSettings.from_dict(self.original_settings.to_dict())
        
        # Device lists
        self.input_devices, self.output_devices = get_audio_devices()
        
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # ─── Output Device ───
        output_group = QGroupBox("Output Device")
        output_layout = QVBoxLayout(output_group)
        
        self.output_combo = QComboBox()
        self.output_combo.addItem("System Default", None)
        for dev in self.output_devices:
            label = f"{dev.name}"
            if dev.is_default_output:
                label += " (default)"
            self.output_combo.addItem(label, dev.id)
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        output_layout.addWidget(self.output_combo)
        
        layout.addWidget(output_group)
        
        # ─── Input Device ───
        input_group = QGroupBox("Input Device")
        input_layout = QVBoxLayout(input_group)
        
        self.input_combo = QComboBox()
        self.input_combo.addItem("System Default", None)
        for dev in self.input_devices:
            label = f"{dev.name}"
            if dev.is_default_input:
                label += " (default)"
            self.input_combo.addItem(label, dev.id)
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        input_layout.addWidget(self.input_combo)
        
        layout.addWidget(input_group)
        
        # ─── Sample Rate ───
        rate_group = QGroupBox("Sample Rate")
        rate_layout = QHBoxLayout(rate_group)
        
        self.rate_combo = QComboBox()
        for sr in SampleRate:
            self.rate_combo.addItem(f"{sr.value} Hz", sr.value)
        self.rate_combo.currentIndexChanged.connect(self._on_rate_changed)
        rate_layout.addWidget(self.rate_combo)
        
        layout.addWidget(rate_group)
        
        # ─── Latency Mode ───
        latency_group = QGroupBox("Latency Mode")
        latency_layout = QVBoxLayout(latency_group)
        
        self.latency_combo = QComboBox()
        latency_descriptions = {
            LatencyMode.ULTRA_LOW: "Ultra Low (64 samples) - Minimum latency, high CPU",
            LatencyMode.LOW: "Low (128 samples) - Good for live playing",
            LatencyMode.BALANCED: "Balanced (256 samples) - Recommended default",
            LatencyMode.SAFE: "Safe (512 samples) - For complex projects",
            LatencyMode.HIGH: "High (1024 samples) - Maximum stability",
        }
        for mode, desc in latency_descriptions.items():
            self.latency_combo.addItem(desc, mode)
        self.latency_combo.currentIndexChanged.connect(self._on_latency_changed)
        latency_layout.addWidget(self.latency_combo)
        
        # Latency display
        self.latency_label = QLabel("One-way: -- ms | Round-trip: -- ms")
        self.latency_label.setStyleSheet("color: #00aaff; margin-top: 6px;")
        latency_layout.addWidget(self.latency_label)
        
        layout.addWidget(latency_group)
        
        # ─── Performance Options ───
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout(perf_group)
        
        self.perf_check = QCheckBox("Enable performance monitoring")
        self.perf_check.setChecked(True)
        perf_layout.addWidget(self.perf_check)
        
        self.low_latency_check = QCheckBox("Request low latency from driver")
        self.low_latency_check.setChecked(True)
        perf_layout.addWidget(self.low_latency_check)
        
        # Live performance meter
        self.perf_meter = AudioMeterWidgetExpanded()
        perf_layout.addWidget(self.perf_meter)
        
        layout.addWidget(perf_group)
        
        # ─── Buttons ───
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._refresh_devices)
        btn_layout.addWidget(refresh_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("applyButton")
        apply_btn.clicked.connect(self._apply_settings)
        btn_layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._ok_clicked)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _load_current_settings(self):
        """Load current settings into UI"""
        settings = self.current_settings
        
        # Output device
        idx = 0
        for i in range(self.output_combo.count()):
            if self.output_combo.itemData(i) == settings.output_device:
                idx = i
                break
        self.output_combo.setCurrentIndex(idx)
        
        # Input device
        idx = 0
        for i in range(self.input_combo.count()):
            if self.input_combo.itemData(i) == settings.input_device:
                idx = i
                break
        self.input_combo.setCurrentIndex(idx)
        
        # Sample rate
        for i in range(self.rate_combo.count()):
            if self.rate_combo.itemData(i) == settings.sample_rate:
                self.rate_combo.setCurrentIndex(i)
                break
        
        # Latency mode
        for i in range(self.latency_combo.count()):
            if self.latency_combo.itemData(i) == settings.latency_mode:
                self.latency_combo.setCurrentIndex(i)
                break
        
        # Performance options
        self.perf_check.setChecked(settings.enable_performance_monitoring)
        self.low_latency_check.setChecked(settings.use_low_latency)
        
        self._update_latency_display()
    
    def _on_output_changed(self, index):
        self.current_settings.output_device = self.output_combo.itemData(index)
    
    def _on_input_changed(self, index):
        self.current_settings.input_device = self.input_combo.itemData(index)
    
    def _on_rate_changed(self, index):
        self.current_settings.sample_rate = self.rate_combo.itemData(index)
        self._update_latency_display()
    
    def _on_latency_changed(self, index):
        mode = self.latency_combo.itemData(index)
        self.current_settings.latency_mode = mode
        self.current_settings.buffer_size = BUFFER_SIZES[mode]
        self._update_latency_display()
    
    def _update_latency_display(self):
        """Update the latency display label"""
        settings = self.current_settings
        one_way = settings.get_latency_ms()
        roundtrip = settings.get_roundtrip_latency_ms()
        self.latency_label.setText(
            f"One-way: {one_way:.2f} ms | Round-trip: {roundtrip:.2f} ms | "
            f"Buffer: {settings.buffer_size} samples")
    
    def _refresh_devices(self):
        """Refresh device lists"""
        self.input_devices, self.output_devices = get_audio_devices()
        
        # Rebuild combos
        self.output_combo.clear()
        self.output_combo.addItem("System Default", None)
        for dev in self.output_devices:
            label = f"{dev.name}"
            if dev.is_default_output:
                label += " (default)"
            self.output_combo.addItem(label, dev.id)
        
        self.input_combo.clear()
        self.input_combo.addItem("System Default", None)
        for dev in self.input_devices:
            label = f"{dev.name}"
            if dev.is_default_input:
                label += " (default)"
            self.input_combo.addItem(label, dev.id)
        
        self._load_current_settings()
    
    def _apply_settings(self):
        """Apply settings without closing"""
        # Update settings from UI
        self.current_settings.enable_performance_monitoring = self.perf_check.isChecked()
        self.current_settings.use_low_latency = self.low_latency_check.isChecked()
        
        # Apply to audio backend
        success = set_audio_settings(self.current_settings)
        
        if success:
            # Save to disk
            save_audio_config(self.current_settings)
            self.settings_changed.emit(self.current_settings)
            
            # Update original settings reference
            self.original_settings = AudioSettings.from_dict(
                self.current_settings.to_dict())
        else:
            QMessageBox.warning(self, "Audio Error",
                "Failed to apply audio settings. "
                "Check if the selected device is available.")
    
    def _ok_clicked(self):
        """Apply and close"""
        self._apply_settings()
        self.accept()
    
    def reject(self):
        """Cancel - restore original settings if changed"""
        # Check if settings were modified
        current_dict = self.current_settings.to_dict()
        original_dict = self.original_settings.to_dict()
        
        if current_dict != original_dict:
            # Settings were applied, ask if user wants to revert
            reply = QMessageBox.question(
                self, "Revert Changes?",
                "Audio settings were changed. Revert to previous settings?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                set_audio_settings(self.original_settings)
                save_audio_config(self.original_settings)
        
        super().reject()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def show_audio_settings_dialog(parent=None) -> Optional[AudioSettings]:
    """
    Show the audio settings dialog and return new settings if accepted.
    
    Returns:
        AudioSettings if user clicked OK/Apply, None if cancelled
    """
    dialog = AudioSettingsDialog(parent)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.current_settings
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'AudioSettingsDialog',
    'AudioMeterWidget',
    'AudioMeterWidgetExpanded',
    'show_audio_settings_dialog',
]
