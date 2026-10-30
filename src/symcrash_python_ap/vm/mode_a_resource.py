"""Validated Mode A tables prepared before playback; lookups never build data."""
from functools import lru_cache
import hashlib
import logging
import mmap
from numbers import Integral
from pathlib import Path
import struct
from threading import Lock

import numpy as np

from .mode_a_recipe import (NOTE_FREQUENCIES_HZ, NOTE_PHASE_INCREMENTS,
    PULSE_AMPLITUDES, LEGACY_FREQUENCY_HZ, LEGACY_PHASE_INCREMENT)

MAGIC = b'SYMLUT01'
VERSION = 1
HEADER = struct.Struct('<8sII32s32s')
SLOTS = 4
PHASES = 64
TABLE_BYTES = 64 * SLOTS * PHASES * 2
NOTE_TABLES = 5 * 128
TABLE_COUNT = NOTE_TABLES + 3
WAVE_NAMES = ('sine', 'saw', 'square', 'triangle', 'pulse')
_active = None
_prepare_lock = Lock()


def _wave_id(value):
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or not 0 <= value < 2**32):
        raise ValueError('Mode A waveform ID must be an unsigned 32-bit integer')
    return int(value)


def note_address(wave_id, note):
    """Use the shared pitch law's clamped MIDI address; keep unknown-ID fallback."""
    from ..primitives import _pitch_cent_index
    wave = _wave_id(wave_id)
    canonical_note = _pitch_cent_index(note, 0) // 100
    return wave, canonical_note, (wave if wave < 5 else 0) * 128 + canonical_note


def legacy_address(wave_id):
    wave = _wave_id(wave_id)
    return NOTE_TABLES + (wave if wave < 3 else 0)


def supported_table_keys():
    return tuple(range(TABLE_COUNT))


def _validated_keys(keys):
    keys = tuple(keys)
    if (not keys or len(keys) > TABLE_COUNT
            or any(type(key) is not int or not 0 <= key < TABLE_COUNT for key in keys)
            or any(a >= b for a, b in zip(keys, keys[1:]))):
        raise ValueError('Mode A resource keys must be sorted unique table IDs')
    return keys


@lru_cache(maxsize=1)
def recipe_digest():
    from .hotpath_jit import _get_sine_lut_np
    digest = hashlib.sha256(b'ModeA-v1/Fs48000/N64/K4/P64/headroom1/RMS16384-floor/'
        b'five-H32-waveforms/legacy-C4-H16/SineCore-RNE/FWPT-RNE')
    for values, dtype in [(_get_sine_lut_np(), '<i4'), (NOTE_FREQUENCIES_HZ, '<f8'),
            (NOTE_PHASE_INCREMENTS, '<u4'), (PULSE_AMPLITUDES, '<f8')]:
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    digest.update(struct.pack('<dI', LEGACY_FREQUENCY_HZ, LEGACY_PHASE_INCREMENT))
    return digest.digest()


def default_resource_path():
    from ..paths import PRESET_BUILD_DIR
    return PRESET_BUILD_DIR / 'resources' / f'mode_a_lut_{recipe_digest().hex()[:16]}.bin'


class PreparedModeAAtlas:
    """Read-only backing bytes for all native and legacy Mode A lookups."""

    def __init__(self, path, *, expected_keys=None):
        keys = _validated_keys(supported_table_keys() if expected_keys is None else expected_keys)
        data_start = HEADER.size + 4 * len(keys)
        expected_size = data_start + len(keys) * TABLE_BYTES
        with Path(path).open('rb') as stream:
            if stream.seek(0, 2) != expected_size:
                raise ValueError('Mode A resource size or table coverage mismatch')
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic, version, count, recipe, checksum = HEADER.unpack_from(mapping)
            if (magic != MAGIC or version != VERSION or count != len(keys)
                    or recipe != recipe_digest()):
                raise ValueError('Unsupported Mode A resource header or arithmetic recipe')
            if mapping[HEADER.size:data_start] != np.asarray(keys, dtype='<u4').tobytes():
                raise ValueError('Mode A resource table coverage mismatch')
            with memoryview(mapping) as view:
                actual = hashlib.sha256(view[HEADER.size:]).digest()
            if checksum != actual:
                raise ValueError('Mode A resource checksum mismatch')
        except BaseException:
            mapping.close()
            raise
        self.path = Path(path)
        self.keys = keys
        self._mapping = mapping
        self._offsets = {key: data_start + i * TABLE_BYTES for i, key in enumerate(keys)}

    def table(self, key, *, wave_id=None):
        from .coeff_lut_streaming import StreamingCoeffLUT
        if type(key) is not int or key not in self._offsets:
            raise ValueError('Table is absent from the prepared Mode A resource')
        if key < NOTE_TABLES:
            wave, note = divmod(key, 128)
            frequency = NOTE_FREQUENCIES_HZ[note]
            name = f'{WAVE_NAMES[wave]}_{note}'
        else:
            wave = key - NOTE_TABLES
            frequency, name = LEGACY_FREQUENCY_HZ, WAVE_NAMES[wave]
        data = np.ndarray((64, SLOTS, PHASES), dtype='<i2',
                          buffer=self._mapping, offset=self._offsets[key])
        return StreamingCoeffLUT(wave if wave_id is None else _wave_id(wave_id),
            name, data, K=SLOTS, P=PHASES, target_freq_hz=frequency)


def prepared_mode_a_atlas():
    """Playback accessor; no file access, compilation, or waveform generation."""
    if _active is None:
        raise RuntimeError('Prepare Mode A resources before rendering (prepare_mode_a_resources)')
    return _active


def prepare_mode_a_resources(path=None, *, build_if_missing=True):
    """Atomically adopt a verified image at startup/allocation, outside playback."""
    global _active
    if path is None and _active is not None:
        return _active
    with _prepare_lock:
        if path is None and _active is not None:
            return _active
        target = default_resource_path() if path is None else Path(path)
        try:
            candidate = PreparedModeAAtlas(target)
        except FileNotFoundError:
            if not build_if_missing:
                raise
            from .mode_a_resource_builder import build_mode_a_resource
            logging.getLogger(__name__).warning(
                'Preparing Mode A table resource (%s); first startup uses about 21 MB of disk space', target)
            try:
                build_mode_a_resource(target)
            except PermissionError:
                if not target.is_file():
                    raise
            candidate = PreparedModeAAtlas(target)
        _active = candidate
        from .coeff_lut_streaming import clear_mode_a_views
        clear_mode_a_views()
        return candidate
