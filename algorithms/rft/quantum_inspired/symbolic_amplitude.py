#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Symbolic Amplitude Implementation - QuantoniumOS
================================================

Wrapper for complex amplitudes or wave segments.
"""

import numpy as np
from typing import Union

class SymbolicAmplitude:
    """Wrapper for complex amplitudes or wave segments."""
    def __init__(self, value: Union[complex, np.ndarray]):
        self.value = value
    
    def __repr__(self):
        return f"SymbolicAmplitude({self.value})"
        
    def add(self, other: 'SymbolicAmplitude') -> 'SymbolicAmplitude':
        return SymbolicAmplitude(self.value + other.value)
        
    def multiply(self, other: 'SymbolicAmplitude') -> 'SymbolicAmplitude':
        return SymbolicAmplitude(self.value * other.value)
