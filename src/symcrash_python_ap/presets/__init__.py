"""
Backward compatibility re-exports for presets.

DEPRECATED: Import from symcrash_python_ap.library.preset_data instead.
"""

from ..library.preset_data import (
    list_presets,
    load_preset,
    get_all_presets,
    get_preset_info,
    PRESET_DIR,
)

__all__ = [
    'list_presets',
    'load_preset',
    'get_all_presets',
    'get_preset_info',
    'PRESET_DIR',
]
