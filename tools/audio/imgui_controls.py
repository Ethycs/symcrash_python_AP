#!/usr/bin/env python3
"""
Symcrash ImGui Controls Launcher
================================

Run the ImGui-based control interface for the synthesizer.

Usage:
    python tools/imgui_controls.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from symcrash_imgui_controls import main

if __name__ == "__main__":
    main()
