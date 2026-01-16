#!/usr/bin/env python3
"""
Symcrash Preset Editor Launcher
===============================

Run the multi-level InstrumentIR preset editor.

Usage:
    pixi run python tools/preset_editor.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from symcrash_preset_editor import main

if __name__ == "__main__":
    main()
