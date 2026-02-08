#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
import numpy as np
from algorithms.rft.core.resonant_fourier_transform import rft_matrix_canonical as rft_matrix

# Fixed-point format: Q1.15 signed
SCALE = 2**15 - 1  # 32767
N = 8

# Choose parameters consistent with your stack
BETA = 1.0
SIGMA = 1.0

Psi = rft_matrix(N, beta=BETA, sigma=SIGMA)

# Normalize to max |value| <= 1.0 (already unitary with ortho FFT, but ensure safety)
max_abs = np.max(np.abs(Psi))
if max_abs == 0:
    max_abs = 1.0

lines = []
for k in range(N):
    for n in range(N):
        val = Psi[n, k]  # column-k basis evaluated at n
        r = np.real(val) / max_abs
        i = np.imag(val) / max_abs
        # clamp
        r = float(max(-1.0, min(1.0, r)))
        i = float(max(-1.0, min(1.0, i)))
        # quantize
        rq = int(np.round(r * SCALE))
        iq = int(np.round(i * SCALE))
        # wrap to 16-bit signed
        def to_u16(x):
            x = (x + (1<<16)) % (1<<16)
            return x
        rq_u16 = to_u16(rq)
        iq_u16 = to_u16(iq)
        lines.append((k, n, rq_u16, iq_u16))

# Emit TL-V case table entries
print("// Closed-form Φ-RFT 8x8 kernel (Q1.15), k = row, n = column")
print("// Replace $kernel_real/$kernel_imag case arms with the following")
print()
print("// $kernel_real[15:0]")
for k,n,rq,iq in lines:
    print(f"// k={k}, n={n}")
    print(f"({{\$k_index, \$n_index}} == 6'b{(k>>2)&1}{(k>>1)&1}{k&1}_"\
          f"{(n>>2)&1}{(n>>1)&1}{n&1}) ? 16'h{rq:04x} :")
print("16'h0000;")
print()
print("// $kernel_imag[15:0]")
for k,n,rq,iq in lines:
    print(f"// k={k}, n={n}")
    print(f"({{\$k_index, \$n_index}} == 6'b{(k>>2)&1}{(k>>1)&1}{k&1}_"\
          f"{(n>>2)&1}{(n>>1)&1}{n&1}) ? 16'h{iq:04x} :")
print("16'h0000;")
