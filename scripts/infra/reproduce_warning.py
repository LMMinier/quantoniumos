# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
import numpy as np
import warnings

def reproduce_warning():
    flat = np.array([1e300, -1e300, np.nan, np.inf])
    tolerance = 1e-10
    scale = 1.0 / tolerance
    clip_min, clip_max = float(np.iinfo(np.int64).min), float(np.iinfo(np.int64).max)
    
    print(f"Scale: {scale}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Proposed fix
        with np.errstate(invalid='ignore', over='ignore'):
            scaled = flat * scale
            # nan_to_num handles NaNs and Infs
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=clip_max, neginf=clip_min)
            # Clip just in case (though nan_to_num should handle it)
            # Actually nan_to_num returns float, so we can cast
            quantized_int = np.rint(scaled).astype(np.int64)
            
        print(f"Quantized: {quantized_int}")
        
        if len(w) > 0:
            print(f"Caught {len(w)} warnings:")
            for warning in w:
                print(f"- {warning.category.__name__}: {warning.message}")
        else:
            print("No warnings caught!")

if __name__ == "__main__":
    reproduce_warning()
