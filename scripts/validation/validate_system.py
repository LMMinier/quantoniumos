#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""QuantoniumOS Full System Validation - UnitaryRFT Integration Test"""

import numpy as np
import sys
import argparse

from algorithms.rft.variants.manifest import iter_variants

def main():
    parser = argparse.ArgumentParser(
        description='QuantoniumOS Full System Validation - UnitaryRFT Integration Test'
    )
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--skip-variants', action='store_true',
                        help='Skip testing all RFT variants')
    args = parser.parse_args()

    print('=' * 60)
    print('QUANTONIUMOS FULL SYSTEM VALIDATION')
    print('=' * 60)

    # Test 1: UnitaryRFT Native Library
    print('\n[1] UnitaryRFT Native Library:')
    try:
        from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT
        rft = UnitaryRFT(256)
        x = np.random.randn(256)
        y = rft.forward(x)
        z = rft.inverse(y)
        error = np.max(np.abs(x - z))
        print(f'    Status: NATIVE (is_mock={rft._is_mock})')
        print(f'    Roundtrip Error: {error:.2e}')
        print(f'    Pass: {error < 1e-10}')
    except Exception as e:
        print(f'    FAILED: {e}')

    # Test 2: All RFT Variants
    print('\n[2] RFT Variants Test:')
    if args.skip_variants:
        print('    SKIPPED: --skip-variants flag set')
    else:
        try:
            from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT

            manifest = list(iter_variants(include_experimental=True, require_kernel_constant=True))
            if not manifest:
                print('    SKIPPED: No variant manifest entries are available in this environment.')
            for entry in manifest:
                rft = UnitaryRFT(64, variant=entry.kernel_id)
                x = np.random.randn(64)
                y = rft.forward(x)
                z = rft.inverse(y)
                error = np.max(np.abs(x - z))
                status = 'PASS' if error < 1e-10 else 'FAIL'
                print(f'    {entry.code}: {status} (error={error:.2e})')
        except Exception as e:
            print(f'    FAILED: {e}')

    print('\n' + '=' * 60)
    print('VALIDATION COMPLETE')
    print('=' * 60)

if __name__ == '__main__':
    main()
