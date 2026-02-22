# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""QuantoniumOS Engine Package - RFT Middleware and core computation engines"""
from .RFTMW import MiddlewareTransformEngine
from .rftmw_memory import RFTMWMemoryLayer

__all__ = ['MiddlewareTransformEngine', 'RFTMWMemoryLayer']
