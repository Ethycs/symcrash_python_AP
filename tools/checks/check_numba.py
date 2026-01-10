#!/usr/bin/env python3
try:
    import numba
    print(f"Numba version: {numba.__version__}")
except ImportError:
    print("Numba not installed")
