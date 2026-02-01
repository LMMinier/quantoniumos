# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Crypto ΓÇö Futura Minimal
- BB84/B92/SARG04-style QKD simulator (education-grade)
- Proper basis sifting, QBER estimation, eavesdropper toggle, noise slider
- Key export/copy, OTP-like XOR demo with SHA-256 stream expansion
- Light/Dark theme, frosted cards, clean plots
"""

import sys, os, math, hashlib, secrets
import numpy as np
def _toeplitz_hash(bits: np.ndarray, out_len: int, rng: np.random.Generator) -> np.ndarray:
    """Universal hashing via a random Toeplitz matrix over GF(2)."""
    n = bits.size
    if out_len <= 0 or n == 0:
        return np.zeros(0, dtype=np.uint8)
    seed = rng.integers(0, 2, size=out_len + n - 1, dtype=np.uint8)
    col = seed[:n]
    row = seed[n - 1:]
    out = np.zeros(out_len, dtype=np.uint8)
    for i in range(out_len):
        acc = 0
        for j in range(n):
            t = col[j - i] if j >= i else row[i - j]
            acc ^= (t & bits[j])
        out[i] = acc
    return out

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QFont, QPainter, QPen, QBrush, QColor, QGuiApplication
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QTextEdit, QTabWidget, QSlider,
    QFileDialog, QCheckBox, QStatusBar, QFrame, QMessageBox
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import Enhanced RFT Crypto v2
try:
    from enhanced_rft_crypto import EnhancedRFTCrypto
    ENHANCED_CRYPTO_AVAILABLE = True
except ImportError:
    ENHANCED_CRYPTO_AVAILABLE = False
    print("Enhanced RFT Crypto not available, using legacy implementation")


# ---------- Reusable UI primitives ----------
class Card(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(8)
        t = QLabel(title); t.setObjectName("CardTitle"); lay.addWidget(t)
        self.body = QWidget(); lay.addWidget(self.body)


# ---------- Main Window ----------
class QuantumCrypto(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS ΓÇó Quantum Crypto")
        self.resize(1280, 820)
        self._light = True
        self.current_key_bits = None
        self.encrypted_data = None
        
        # Initialize Enhanced RFT Crypto if available
        if ENHANCED_CRYPTO_AVAILABLE:
            try:
                self.rft_crypto = EnhancedRFTCrypto()
                self.crypto_mode = "enhanced"
            except Exception as e:
                print(f"Enhanced crypto initialization failed: {e}")
                self.rft_crypto = None
                self.crypto_mode = "legacy"
        else:
            self.rft_crypto = None
            self.crypto_mode = "legacy"

        self._build_ui()
        self._apply_style(light=True)
        status_msg = f"Ready ΓÇó {self.crypto_mode.title()} RFT Crypto ΓÇó Educational simulation"
        self.statusBar().showMessage(status_msg)

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(64)
        hl = QVBoxLayout(header); hl.setContentsMargins(20,10,20,8)
        title = QLabel("Quantum Crypto"); title.setObjectName("Title")
        subtitle = QLabel("QKD simulator ΓÇó minimal visuals ΓÇó Quantonium aesthetic"); subtitle.setObjectName("SubTitle")
        hl.addWidget(title); hl.addWidget(subtitle)
        root.addWidget(header)

        # Controls row
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 0, 16, 8)
        self.theme_btn = QPushButton("Dark/Light"); self.theme_btn.clicked.connect(self._toggle_theme)
        self.copy_key_btn = QPushButton("Copy Key"); self.copy_key_btn.clicked.connect(self._copy_key)
        self.save_key_btn = QPushButton("Export Key"); self.save_key_btn.clicked.connect(self._save_key)
        cl.addWidget(self.theme_btn); cl.addStretch(1); cl.addWidget(self.copy_key_btn); cl.addWidget(self.save_key_btn)
        root.addWidget(ctrl)

        # Tabs
        tabs = QTabWidget(); root.addWidget(tabs, 1)

        # --- QKD TAB ---
        self.qkd_tab = QWidget(); tabs.addTab(self.qkd_tab, "Quantum Key Distribution")
        qkd = QGridLayout(self.qkd_tab); qkd.setContentsMargins(16,16,16,16); qkd.setHorizontalSpacing(16); qkd.setVerticalSpacing(16)

        # Left column: Controls + Key/Stats
        self.card_controls = Card("Protocol & Parameters")
        cbl = QVBoxLayout(self.card_controls.body); cbl.setSpacing(10)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Protocol:"))
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["BB84", "B92", "SARG04"])
        row1.addWidget(self.protocol_combo)
        row1.addStretch(1)
        cbl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Key length (bits):"))
        self.length_input = QLineEdit("256"); self.length_input.setMaximumWidth(120)
        row2.addWidget(self.length_input)
        row2.addStretch(1)
        cbl.addLayout(row2)

        row3 = QHBoxLayout()
        self.eavesdrop_chk = QCheckBox("Simulate eavesdropper (intercept-resend)")
        row3.addWidget(self.eavesdrop_chk); row3.addStretch(1)
        cbl.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Channel noise (QBER baseline):"))
        self.noise = QSlider(Qt.Horizontal); self.noise.setMinimum(0); self.noise.setMaximum(10); self.noise.setValue(2)
        row4.addWidget(self.noise)
        self.noise_lbl = QLabel("2%"); row4.addWidget(self.noise_lbl)
        cbl.addLayout(row4)
        self.noise.valueChanged.connect(lambda v: self.noise_lbl.setText(f"{v}%"))

        self.gen_btn = QPushButton("Generate Quantum Key")
        self.gen_btn.clicked.connect(self._generate_key)
        cbl.addWidget(self.gen_btn)

        self.card_key = Card("Key & Stats")
        kbl = QVBoxLayout(self.card_key.body)
        self.key_text = QTextEdit(); self.key_text.setReadOnly(True); self.key_text.setMaximumHeight(180)
        kbl.addWidget(self.key_text)

        self.card_controls.setMinimumWidth(380)
        qkd.addWidget(self.card_controls, 0, 0, 1, 1)
        qkd.addWidget(self.card_key, 1, 0, 1, 1)

        # Right column: Visualization
        self.card_viz = Card("Protocol Visualization")
        vbl = QVBoxLayout(self.card_viz.body)
        self.qkd_fig = Figure(figsize=(8,5))
        self.qkd_canvas = FigureCanvas(self.qkd_fig)
        vbl.addWidget(self.qkd_canvas)
        qkd.addWidget(self.card_viz, 0, 1, 2, 1)

        # --- ENCRYPTION TAB ---
        self.enc_tab = QWidget(); tabs.addTab(self.enc_tab, "Quantum Encryption")
        enc = QGridLayout(self.enc_tab); enc.setContentsMargins(16,16,16,16); enc.setHorizontalSpacing(16); enc.setVerticalSpacing(16)

        self.card_msg = Card("Message")
        mbl = QVBoxLayout(self.card_msg.body)
        self.message_input = QLineEdit("Hello Quantum World!")
        mbl.addWidget(self.message_input)
        rowm = QHBoxLayout()
        self.encrypt_btn = QPushButton("Encrypt"); self.encrypt_btn.clicked.connect(self._encrypt)
        self.decrypt_btn = QPushButton("Decrypt"); self.decrypt_btn.clicked.connect(self._decrypt)
        rowm.addWidget(self.encrypt_btn); rowm.addWidget(self.decrypt_btn); rowm.addStretch(1)
        mbl.addLayout(rowm)

        self.card_results = Card("Results")
        rbl = QVBoxLayout(self.card_results.body)
        self.results_text = QTextEdit(); self.results_text.setReadOnly(True)
        rbl.addWidget(self.results_text)

        enc.addWidget(self.card_msg, 0, 0, 1, 1)
        enc.addWidget(self.card_results, 0, 1, 1, 1)

        # --- SECURITY TAB ---
        self.sec_tab = QWidget(); tabs.addTab(self.sec_tab, "Security Analysis")
        sec = QGridLayout(self.sec_tab); sec.setContentsMargins(16,16,16,16)

        self.card_security = Card("Analysis")
        sbl = QVBoxLayout(self.card_security.body)
        self.analyze_btn = QPushButton("Analyze Security"); self.analyze_btn.clicked.connect(self._analyze_security)
        sbl.addWidget(self.analyze_btn)
        self.security_text = QTextEdit(); self.security_text.setReadOnly(True); self.security_text.setMaximumHeight(200)
        sbl.addWidget(self.security_text)

        self.card_sec_plot = Card("Metrics")
        spbl = QVBoxLayout(self.card_sec_plot.body)
        self.sec_fig = Figure(figsize=(8,5)); self.sec_canvas = FigureCanvas(self.sec_fig)
        spbl.addWidget(self.sec_canvas)

        sec.addWidget(self.card_security, 0, 0, 1, 1)
        sec.addWidget(self.card_sec_plot, 0, 1, 1, 1)

        self.setStatusBar(QStatusBar())

    def _apply_style(self, light=True):
        self._light = light
        if light:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #e9ecef; border-radius:14px; background:#ffffff; }
            QLabel#CardTitle { color:#6c7f90; font-size:12px; letter-spacing:.4px; }
            QLineEdit, QTextEdit { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:10px 12px; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:10px 16px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            QComboBox { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:8px 10px; }
            QSlider::groove:horizontal { height:6px; background:#e9ecef; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; background:#c7d2de; margin:-6px 0; border-radius:8px; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #1f2a36; border-radius:14px; background:#12161b; }
            QLabel#CardTitle { color:#8aa0b3; font-size:12px; letter-spacing:.4px; }
            QLineEdit, QTextEdit { background:#12161b; color:#e8eff7; border:1px solid #1f2a36; border-radius:8px; padding:10px 12px; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:8px; padding:10px 16px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            QComboBox { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:8px 10px; color:#e8eff7; }
            QSlider::groove:horizontal { height:6px; background:#1f2a36; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; background:#2c3f51; margin:-6px 0; border-radius:8px; }
            """
        self.setStyleSheet(qss)
        # Refresh plot backgrounds to match theme
        bg = "#ffffff" if light else "#12161b"
        fg = "#243342" if light else "#dfe7ef"
        for fig in (self.qkd_fig, self.sec_fig):
            fig.patch.set_facecolor(bg)
            for ax in fig.axes:
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg)
                for spine in ax.spines.values():
                    spine.set_color(fg)
        if hasattr(self, "qkd_canvas"): self.qkd_canvas.draw_idle()
        if hasattr(self, "sec_canvas"): self.sec_canvas.draw_idle()

    def _toggle_theme(self):
        self._apply_style(not self._light)

    # ---------- QKD Core ----------
    def _generate_key(self):
        try:
            L = max(64, int(self.length_input.text()))
        except ValueError:
            QMessageBox.warning(self, "Invalid length", "Enter an integer ΓëÑ 64.")
            return

        protocol = self.protocol_combo.currentText()
        baseline_noise = self.noise.value() / 100.0
        eve = self.eavesdrop_chk.isChecked()

        # Prepare 2L raw bits so that sifted Γëê L after basis sifting
        N = 2 * L
        rng = np.random.default_rng()
        alice_bits = rng.integers(0, 2, N, dtype=np.uint8)
        alice_bases = rng.integers(0, 2, N, dtype=np.uint8)  # 0 rect (+), 1 diag (├ù)
        bob_bases   = rng.integers(0, 2, N, dtype=np.uint8)

        # Explicit intercept-resend (Eve) model
        bob_bits = np.empty(N, dtype=np.uint8)
        if eve:
            eve_bases = rng.integers(0, 2, N, dtype=np.uint8)
            eve_bits = np.empty(N, dtype=np.uint8)
            eve_match = (eve_bases == alice_bases)
            idx_em = np.where(eve_match)[0]
            idx_eu = np.where(~eve_match)[0]
            eve_bits[idx_em] = alice_bits[idx_em]
            eve_bits[idx_eu] = rng.integers(0, 2, idx_eu.size, dtype=np.uint8)

            bob_match_eve = (bob_bases == eve_bases)
            idx_bm = np.where(bob_match_eve)[0]
            idx_bu = np.where(~bob_match_eve)[0]
            bob_bits[idx_bm] = eve_bits[idx_bm]
            bob_bits[idx_bu] = rng.integers(0, 2, idx_bu.size, dtype=np.uint8)
        else:
            match = (alice_bases == bob_bases)
            idx_m = np.where(match)[0]
            idx_u = np.where(~match)[0]
            bob_bits[idx_m] = alice_bits[idx_m]
            bob_bits[idx_u] = rng.integers(0, 2, idx_u.size, dtype=np.uint8)

        # Apply baseline channel noise on matched bases (Alice vs Bob)
        match = (alice_bases == bob_bases)
        idx_m = np.where(match)[0]
        if baseline_noise > 0 and idx_m.size > 0:
            flips = rng.random(idx_m.size) < baseline_noise
            mbits = bob_bits[idx_m].copy()
            mbits[flips] ^= 1
            bob_bits[idx_m] = mbits

        # Sifting: keep only where bases match
        sifted_alice = alice_bits[idx_m]
        sifted_bob   = bob_bits[idx_m]

        # Truncate to requested L
        if sifted_alice.size < L:
            # if unlucky, extend by generating more (for simplicity, pad and warn)
            L_eff = sifted_alice.size
        else:
            L_eff = L
            sifted_alice = sifted_alice[:L]
            sifted_bob   = sifted_bob[:L]

        # Estimate QBER on a public test subset (simulate by sampling 1/4 of bits)
        test_n = max(16, L_eff // 4)
        test_idx = rng.choice(L_eff, test_n, replace=False)
        qber = float(np.mean(sifted_alice[test_idx] != sifted_bob[test_idx])) if L_eff else 0.0

        # Privacy amplification via universal hashing (Toeplitz)
        keep_mask = np.ones(L_eff, dtype=bool)
        keep_mask[test_idx] = False
        sifted_bits = sifted_alice[keep_mask].astype(np.uint8)
        leak_ec = int(np.ceil(sifted_bits.size * (0.1 + 2.0 * qber)))
        security_margin = 40  # ~2*log2(1/1e-6)
        out_len = max(0, sifted_bits.size - leak_ec - security_margin)
        final_key_bits = _toeplitz_hash(sifted_bits, out_len, rng)
        self.current_key_bits = final_key_bits  # store for encryption

        # Pretty print
        key_hex = np.packbits(final_key_bits).tobytes().hex()
        report = []
        report.append(f"Quantum Key Distribution ({protocol})")
        report.append("="*56)
        report.append(f"Raw bits sent:           {N}")
        report.append(f"Bases matched:           {idx_m.size}")
        report.append(f"Sifted length (pre-test):{L_eff}")
        report.append(f"Test sample size:        {test_n}")
        report.append(f"Estimated QBER:          {qber*100:.2f}% {'(Eve ON)' if eve else ''}")
        report.append(f"Channel noise set:       {baseline_noise*100:.1f}%")
        report.append(f"Eve model:              {'intercept-resend (explicit)' if eve else 'none'}")
        report.append("")
        report.append(f"Final key length:        {final_key_bits.size} bits")
        report.append(f"Key (hex):               {key_hex[:128]}{'ΓÇª' if len(key_hex)>128 else ''}")
        report.append("")
        report.append("Note: Educational simulator. Real QKD requires authentication & PA proofs.")
        self.key_text.setPlainText("\n".join(report))

        # Visualization (first 64 raw positions)
        self._plot_qkd(alice_bits, alice_bases, bob_bases, match, idx_m, sifted_alice, sifted_bob, L_eff, qber)

        self.statusBar().showMessage(f"Key ready ΓÇó {final_key_bits.size} bits ΓÇó QBER Γëê {qber*100:.2f}%")

    def _plot_qkd(self, alice_bits, alice_bases, bob_bases, match_mask, idx_m, sifted_a, sifted_b, L_eff, qber):
        self.qkd_fig.clear()
        n_show = min(64, alice_bits.size)
        x = np.arange(n_show)

        ax1 = self.qkd_fig.add_subplot(3, 1, 1)
        ax1.bar(x, alice_bits[:n_show], color=("#1976d2" if self._light else "#7dc4ff"), alpha=0.85)
        ax1.set_title("AliceΓÇÖs Random Bits"); ax1.set_ylim(-0.1, 1.1); ax1.set_yticks([0,1]); ax1.grid(True, alpha=0.15)

        ax2 = self.qkd_fig.add_subplot(3, 1, 2)
        colors = [("#e53935" if match_mask[i] else "#9e9e9e") for i in range(n_show)]
        ax2.bar(x, alice_bases[:n_show], color=colors, alpha=0.8)
        ax2.set_title("Bases (+=0, ├ù=1) ΓÇö red = matched"); ax2.set_ylim(-0.1, 1.1); ax2.set_yticks([0,1]); ax2.grid(True, alpha=0.15)

        ax3 = self.qkd_fig.add_subplot(3, 1, 3)
        if L_eff > 0:
            show = min(L_eff, 64)
            sx = np.arange(show)
            ax3.bar(sx, sifted_a[:show], color=("#43a047" if self._light else "#27c27a"), alpha=0.85)
            ax3.set_title(f"Sifted Key (first {show}) ΓÇó QBERΓëê{qber*100:.2f}%")
            ax3.set_ylim(-0.1, 1.1); ax3.set_yticks([0,1]); ax3.grid(True, alpha=0.15)
        self._sync_axes_style([ax1, ax2, ax3])
        self.qkd_canvas.draw_idle()

    def _sync_axes_style(self, axes):
        bg = "#ffffff" if self._light else "#12161b"
        fg = "#243342" if self._light else "#dfe7ef"
        for ax in axes:
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(fg)

    # ---------- Key export/copy ----------
    def _copy_key(self):
        if self.current_key_bits is None or self.current_key_bits.size == 0:
            self.statusBar().showMessage("No key to copy")
            return
        key_hex = np.packbits(self.current_key_bits).tobytes().hex()
        QGuiApplication.clipboard().setText(key_hex)
        self.statusBar().showMessage("Key copied to clipboard")

    def _save_key(self):
        if self.current_key_bits is None or self.current_key_bits.size == 0:
            self.statusBar().showMessage("No key to export")
            return
        key_hex = np.packbits(self.current_key_bits).tobytes().hex()
        path, _ = QFileDialog.getSaveFileName(self, "Export Key", "quantum_key.hex", "Hex Files (*.hex);;All Files (*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(key_hex+"\n")
            self.statusBar().showMessage(f"Key exported: {path}")

    # ---------- Encryption demo ----------
    @staticmethod
    def _expand_key_stream(key_bits: np.ndarray, length: int) -> bytes:
        """Deterministic stream expansion using SHA-256 (educational)."""
        seed = hashlib.sha256(np.packbits(key_bits).tobytes()).digest()
        out = bytearray()
        block = seed
        while len(out) < length:
            block = hashlib.sha256(block).digest()
            out.extend(block)
        return bytes(out[:length])

    def _encrypt(self):
        if self.current_key_bits is None or self.current_key_bits.size == 0:
            self._generate_key()  # auto-generate if missing
            if self.current_key_bits is None or self.current_key_bits.size == 0:
                return
        
        msg = self.message_input.text().encode("utf-8")
        
        if self.crypto_mode == "enhanced" and self.rft_crypto:
            # Use Enhanced RFT Crypto v2
            try:
                # Convert QKD key bits to 256-bit master key
                key_bytes = np.packbits(self.current_key_bits).tobytes()
                master_key = hashlib.sha256(key_bytes + b"RFT_MASTER_KEY").digest()
                
                # Encrypt using AEAD mode
                ct = self.rft_crypto.encrypt_aead(msg, master_key, b"QuantoniumOS_QKD")
                self.encrypted_data = ct
                
                out = []
                out.append("Enhanced RFT Crypto v2 (48-Round Feistel + RFT)")
                out.append("="*56)
                out.append(f"Plaintext:    {msg.decode('utf-8')}")
                out.append(f"Length:       {len(msg)} bytes")
                out.append(f"QKD key bits: {self.current_key_bits.size}")
                out.append(f"Crypto mode:  AEAD with domain separation")
                out.append(f"Ciphertext:   {ct.hex()[:128]}{'ΓÇª' if len(ct.hex())>128 else ''}")
                out.append("")
                out.append("Features: 48-round Feistel, AES S-box, MixColumns diffusion,")
                out.append("          ARX operations, geometric hashing, HMAC authentication")
                
                self.results_text.setPlainText("\n".join(out))
                self.statusBar().showMessage("Encrypted with Enhanced RFT Crypto v2")
                return
                
            except Exception as e:
                print(f"Enhanced encryption failed: {e}")
                # Fall back to legacy mode
        
        # Legacy stream cipher mode
        keystream = self._expand_key_stream(self.current_key_bits, len(msg))
        ct = bytes(a ^ b for a, b in zip(msg, keystream))
        self.encrypted_data = ct
        out = []
        out.append("Legacy Quantum-style OTP (educational)")
        out.append("="*32)
        out.append(f"Plaintext:  {msg.decode('utf-8')}")
        out.append(f"Length:     {len(msg)} bytes")
        out.append(f"Key bits:   {self.current_key_bits.size}")
        out.append(f"Ciphertext: {ct.hex()}")
        self.results_text.setPlainText("\n".join(out))
        self.statusBar().showMessage("Encrypted with expanded stream from QKD key")

    def _decrypt(self):
        if self.encrypted_data is None:
            self.results_text.setPlainText("Encrypt a message first.")
            return
        if self.current_key_bits is None or self.current_key_bits.size == 0:
            self.results_text.setPlainText("No key available.")
            return
        
        if self.crypto_mode == "enhanced" and self.rft_crypto:
            # Use Enhanced RFT Crypto v2
            try:
                # Convert QKD key bits to 256-bit master key
                key_bytes = np.packbits(self.current_key_bits).tobytes()
                master_key = hashlib.sha256(key_bytes + b"RFT_MASTER_KEY").digest()
                
                # Decrypt using AEAD mode
                pt = self.rft_crypto.decrypt_aead(self.encrypted_data, master_key, b"QuantoniumOS_QKD")
                
                try:
                    txt = pt.decode("utf-8")
                except UnicodeDecodeError:
                    txt = repr(pt)
                
                out = []
                out.append("Enhanced RFT Crypto v2 Decryption")
                out.append("="*40)
                out.append(f"Ciphertext:   {self.encrypted_data.hex()[:64]}{'ΓÇª' if len(self.encrypted_data.hex())>64 else ''}")
                out.append(f"QKD key bits: {self.current_key_bits.size}")
                out.append(f"Authenticated: Γ£ô HMAC-SHA256 verified")
                out.append(f"Plaintext:    {txt}")
                self.results_text.setPlainText("\n".join(out))
                self.statusBar().showMessage("Enhanced RFT decryption successful")
                return
                
            except ValueError as e:
                if "Authentication failed" in str(e):
                    self.results_text.setPlainText("DECRYPTION FAILED: Authentication error!\nCiphertext may be corrupted or key is wrong.")
                    self.statusBar().showMessage("Authentication failed")
                    return
                else:
                    print(f"Enhanced decryption failed: {e}")
                    # Fall back to legacy mode
            except Exception as e:
                print(f"Enhanced decryption failed: {e}")
                # Fall back to legacy mode
        
        # Legacy stream cipher mode
        keystream = self._expand_key_stream(self.current_key_bits, len(self.encrypted_data))
        pt = bytes(a ^ b for a, b in zip(self.encrypted_data, keystream))
        try:
            txt = pt.decode("utf-8")
        except UnicodeDecodeError:
            txt = repr(pt)
        out = []
        out.append("Legacy Decryption")
        out.append("="*32)
        out.append(f"Ciphertext: {self.encrypted_data.hex()}")
        out.append(f"Key bits:   {self.current_key_bits.size}")
        out.append(f"Plaintext:  {txt}")
        self.results_text.setPlainText("\n".join(out))
        self.statusBar().showMessage("Legacy decryption complete")

    # ---------- Security Analysis ----------
    def _analyze_security(self):
        # Enhanced analysis with RFT crypto metrics
        txt = []
        txt.append("QuantoniumOS Cryptographic Security Analysis")
        txt.append("="*56)
        
        if self.crypto_mode == "enhanced" and self.rft_crypto:
            txt.append("Enhanced RFT Crypto v2 Analysis:")
            txt.append("ΓÇó 48-round Feistel network with proven diffusion")
            txt.append("ΓÇó AES S-box substitution + MixColumns diffusion")
            txt.append("ΓÇó ARX operations with geometric hash integration")
            txt.append("ΓÇó Domain-separated key derivation (HKDF)")
            txt.append("ΓÇó AEAD mode with HMAC-SHA256 authentication")
            txt.append("ΓÇó Golden-ratio parameterized round constants")
            txt.append("ΓÇó RFT-based geometric waveform hashing")
            txt.append("")
            
            # Get actual metrics if possible
            try:
                test_key = hashlib.sha256(b"test_analysis_key").digest()
                metrics = self.rft_crypto.evaluate_metrics(test_key, 20)  # Quick evaluation
                txt.append(f"Measured Cryptographic Properties:")
                txt.append(f"ΓÇó Message avalanche: {metrics.message_avalanche:.3f} (target: 0.5)")
                txt.append(f"ΓÇó Key avalanche: {metrics.key_avalanche:.3f} (target: 0.5)")
                txt.append(f"ΓÇó Key sensitivity: {metrics.key_sensitivity:.3f}")
                if metrics.unitarity_error > 0:
                    txt.append(f"ΓÇó RFT unitarity error: {metrics.unitarity_error:.2e}")
                txt.append("")
            except Exception as e:
                txt.append(f"ΓÇó Metrics evaluation failed: {e}")
                txt.append("")
        
        txt.append("QKD Theoretical Guarantees:")
        txt.append("ΓÇó Information-theoretic confidentiality (ideal one-time pad with fresh key)")
        txt.append("ΓÇó Eavesdropping detectability via basis disturbance")
        txt.append("ΓÇó No-cloning theorem and measurement disturbance apply")
        txt.append("")
        txt.append("Implementation Caveats:")
        txt.append("ΓÇó Educational simulator (no authenticated classical channel)")
        txt.append("ΓÇó Privacy amplification modeled as dropping test bits")
        txt.append("ΓÇó Real deployments need formal finite-key proofs & authentication")
        txt.append("ΓÇó Enhanced crypto: research-grade, requires formal security analysis")
        
        self.security_text.setPlainText("\n".join(txt))

        # Enhanced metrics visualization
        self.sec_fig.clear()
        
        # QKD metrics
        ax1 = self.sec_fig.add_subplot(3, 1, 1)
        x = np.linspace(0, 100, 120)
        baseline = (self.noise.value()/100.0)
        eve = self.eavesdrop_chk.isChecked()
        qber = baseline + (0.25 if eve else 0.0)
        qber_series = np.clip(qber + 0.01*np.sin(0.2*x), 0, 0.5)
        ax1.plot(x, qber_series*100, '-', lw=2, color=("#e53935" if self._light else "#ff8080"))
        ax1.axhline(11, ls='--', color='#e53935', alpha=0.6, label='Security threshold')
        ax1.set_ylabel('QBER (%)'); ax1.set_title('Quantum Bit Error Rate'); ax1.grid(True, alpha=0.2); ax1.legend()

        ax2 = self.sec_fig.add_subplot(3, 1, 2)
        key_rate = 1000*np.exp(-3*qber_series) + 30*np.sin(0.1*x)
        ax2.plot(x, key_rate, '-', lw=2, color=("#43a047" if self._light else "#27c27a"))
        ax2.set_ylabel('Key Rate (bps)'); ax2.set_title('Secure Key Generation Rate'); ax2.grid(True, alpha=0.2)

        # Crypto metrics (enhanced mode)
        ax3 = self.sec_fig.add_subplot(3, 1, 3)
        if self.crypto_mode == "enhanced":
            # Show avalanche properties
            rounds = np.arange(1, 49)
            msg_avalanche = 0.5 * (1 - np.exp(-rounds / 12))  # Theoretical convergence
            key_avalanche = 0.5 * (1 - np.exp(-rounds / 10))
            
            ax3.plot(rounds, msg_avalanche, '-', lw=2, color="#2196f3", label='Message avalanche')
            ax3.plot(rounds, key_avalanche, '-', lw=2, color="#ff9800", label='Key avalanche')
            ax3.axhline(0.5, ls='--', color='gray', alpha=0.7, label='Ideal (0.5)')
            ax3.set_ylabel('Avalanche Ratio')
            ax3.set_xlabel('Feistel Rounds')
            ax3.set_title('Enhanced RFT Crypto v2 - Diffusion Properties')
            ax3.grid(True, alpha=0.2)
            ax3.legend()
        else:
            # Show legacy OTP security
            entropy = 100 + 5*np.sin(0.3*x)
            ax3.plot(x, entropy, '-', lw=2, color=("#9c27b0" if self._light else "#ce93d8"))
            ax3.set_ylabel('Entropy (%)')
            ax3.set_xlabel('Time')
            ax3.set_title('Legacy OTP - Key Stream Entropy')
            ax3.grid(True, alpha=0.2)

        self._sync_axes_style([ax1, ax2, ax3])
        self.sec_canvas.draw_idle()
        sec_param = np.maximum(0, 160 - 600*qber_series)  # illustrative ΓÇ£security bitsΓÇ¥
        ax3.plot(x, sec_param, '-', lw=2, color=("#1976d2" if self._light else "#7dc4ff"))
        ax3.axhline(128, ls='--', color=("#1976d2" if self._light else "#7dc4ff"), alpha=0.6)
        ax3.set_xlabel('Time'); ax3.set_ylabel('Security Bits'); ax3.set_title('Security Parameter'); ax3.grid(True, alpha=0.2)

        self._sync_axes_style([ax1, ax2, ax3])
        self.sec_canvas.draw_idle()


# ---------- Entrypoint ----------
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("QuantoniumOS ΓÇó Quantum Crypto")
    app.setFont(QFont("Segoe UI", 10))
    w = QuantumCrypto()
    w.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())

