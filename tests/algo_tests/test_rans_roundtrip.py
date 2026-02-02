# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import random
import pytest
from algorithms.rft.compression import ans as ans_module

@pytest.mark.parametrize("alphabet", [2, 16, 256])
@pytest.mark.parametrize("precision", [8, 12])
def test_rans_roundtrip_small(alphabet, precision):
    random.seed(42)
    data = [random.randrange(alphabet) for _ in range(1000)]

    encoded, freq_data = ans_module.ans_encode(data, precision=precision)
    decoded = ans_module.ans_decode(encoded, freq_data, num_symbols=len(data))

    assert decoded == data, f"Roundtrip failed for alphabet={alphabet}, precision={precision}"
