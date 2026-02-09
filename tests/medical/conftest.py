# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import os
import pytest

from algorithms.rft.variants.operator_variants import list_operator_variants
import algorithms.rft.kernels.operator_eigenbasis_rft as rftk


# -----------------------------------------------------------------------------
# Real-Data Environment Configuration
# -----------------------------------------------------------------------------

def pytest_addoption(parser):
    """Add --real-data CLI option to enable real-data tests."""
    parser.addoption(
        "--real-data",
        action="store_true",
        default=False,
        help="Enable real-data tests (requires datasets to be downloaded).",
    )


def pytest_configure(config):
    """Register custom markers and propagate CLI flag to environment."""
    config.addinivalue_line("markers", "realdata: mark test as requiring real datasets")
    config.addinivalue_line("markers", "mitbih: mark test as requiring MIT-BIH data")
    config.addinivalue_line("markers", "sleepedf: mark test as requiring Sleep-EDF data")
    config.addinivalue_line("markers", "fastmri: mark test as requiring FastMRI data")
    config.addinivalue_line("markers", "genomics: mark test as requiring genomics data")
    
    # If --real-data passed, set environment variable so fixtures detect it
    if config.getoption("real_data"):
        os.environ["USE_REAL_DATA"] = "1"


# -----------------------------------------------------------------------------
# RFT Variant Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rft_variants():
    """List of all registered RFT operator variants."""
    return list_operator_variants()


@pytest.fixture(params=list_operator_variants(), scope="module")
def rft_variant(request):
    """Parameterize tests over all RFT variants/hybrids."""
    return request.param


@pytest.fixture(autouse=True)
def _apply_rft_variant(rft_variant):
    """Autouse fixture to set default RFT variant for all tests in module."""
    # Set module-level default so existing calls pick up the variant
    previous = rftk.DEFAULT_VARIANT
    rftk.DEFAULT_VARIANT = rft_variant
    try:
        yield
    finally:
        # Restore prior default after each test
        rftk.DEFAULT_VARIANT = previous
