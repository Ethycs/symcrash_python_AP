"""Prepared exact-pitch Mode B resources; no transforms in the lookup path.

Call prepare_basis_resources at host initialization, before audio callbacks.
The default atlas covers every output of the current integer MIDI/cents law.
Its disposable disk image is memory mapped read-only, with bounded view caching
in harmonic_bank. Mapping does not guarantee that pages remain resident in RAM.
"""
from functools import lru_cache
import hashlib
import logging
import mmap
from pathlib import Path
import struct
from threading import Lock

import numpy as np

from .carrier_spectrum import harmonic_limit

MAGIC = b'SYMBAS01'
VERSION = 1
HEADER = struct.Struct('<8sII32s32s')
DC_BYTES = 64 * 4 * 2
HARMONIC_BYTES = 2 * DC_BYTES
MAX_PITCH_ENTRIES = 65536
_active = None
_prepare_lock = Lock()


@lru_cache(maxsize=1)
def supported_pitch_increments() -> tuple[int, ...]:
    """Complete image of the frozen integer-cent pitch law, MIDI 0..127."""
    from ..pitch_rom import MIDI_CENT_PHASE_Q32
    return MIDI_CENT_PHASE_Q32


@lru_cache(maxsize=1)
def recipe_digest() -> bytes:
    """Version the arithmetic recipe and the frozen sine/pitch inputs.

    Bump the recipe version if SineCore or the FWPT arithmetic changes. The
    independent reference comparisons in test_basis_resource guard this rule.
    """
    from .hotpath_jit import _get_sine_lut_np
    digest = hashlib.sha256(b'ModeB-basis-v1/Fs48000/N64/K4/stride1/H32/headroom2/SineCore-RNE/FWPT-RNE')
    digest.update(_get_sine_lut_np().astype('<i4').tobytes())
    digest.update(np.asarray(supported_pitch_increments(), dtype='<u4').tobytes())
    return digest.digest()


def default_resource_path() -> Path:
    from ..paths import PRESET_BUILD_DIR
    return PRESET_BUILD_DIR / 'resources' / f'mode_b_basis_{recipe_digest().hex()[:16]}.bin'


def _validated_keys(keys) -> tuple[int, ...]:
    keys = tuple(keys)
    if (not keys or len(keys) > MAX_PITCH_ENTRIES or
            any(type(p) is not int or not 0 <= p < 2**32 for p in keys) or
            any(a >= b for a, b in zip(keys, keys[1:]))):
        raise ValueError('Basis resource pitches must be sorted, unique unsigned Q32 integers')
    return keys


class PreparedBasisAtlas:
    """Validated immutable backing storage; retained views keep the map alive."""

    def __init__(self, path: Path, *, expected_keys=None):
        keys = _validated_keys(supported_pitch_increments() if expected_keys is None else expected_keys)
        data_start = HEADER.size + 4 * len(keys)
        expected_size = data_start + DC_BYTES + sum(harmonic_limit(p) * HARMONIC_BYTES for p in keys)
        with Path(path).open('rb') as stream:
            if stream.seek(0, 2) != expected_size:
                raise ValueError('Basis resource size or pitch coverage mismatch')
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic, version, count, recipe, checksum = HEADER.unpack_from(mapping)
            if (magic != MAGIC or version != VERSION or count != len(keys) or recipe != recipe_digest()):
                raise ValueError('Unsupported basis resource header or arithmetic recipe')
            encoded_keys = np.asarray(keys, dtype='<u4').tobytes()
            if mapping[HEADER.size:data_start] != encoded_keys:
                raise ValueError('Basis resource pitch coverage mismatch')
            # Digest the mapping without copying the ~857 MiB production image.
            with memoryview(mapping) as view:
                actual = hashlib.sha256(view[HEADER.size:]).digest()
            if actual != checksum:
                raise ValueError('Basis resource checksum mismatch')
        except BaseException:
            mapping.close()
            raise
        self._mapping = mapping
        self.path = Path(path)
        self.keys = keys
        self.dc = np.ndarray((64, 4), dtype='<i2', buffer=mapping, offset=data_start)
        offset = data_start + DC_BYTES
        self._offsets = {}
        for pitch in keys:
            count = harmonic_limit(pitch)
            self._offsets[pitch] = (offset, count)
            offset += count * HARMONIC_BYTES

    def basis(self, note: int, phase_inc: int, harmonics: int):
        from .harmonic_bank import HarmonicBasisFrames
        if type(phase_inc) is not int or not 0 <= phase_inc < 2**32:
            raise ValueError('Basis phase increment must be unsigned Q32')
        if type(harmonics) is not int or not 0 <= harmonics <= 32:
            raise ValueError('Basis harmonic count must be in 0..32')
        try:
            offset, capacity = self._offsets[phase_inc]
        except KeyError:
            raise ValueError(f'Pitch increment {phase_inc} is absent from the prepared basis resource') from None
        if harmonics > capacity:
            raise ValueError('Basis harmonic count exceeds the prepared pitch bandlimit')
        data = np.ndarray((2, capacity, 64, 4), dtype='<i2', buffer=self._mapping, offset=offset)
        return HarmonicBasisFrames(self.dc, data[0, :harmonics], data[1, :harmonics],
                                   harmonics, note, phase_inc * 48000 / 2**32)


def prepared_basis_atlas() -> PreparedBasisAtlas:
    """Playback-only accessor: never opens files, builds, or runs a transform."""
    if _active is None:
        raise RuntimeError('Prepare Mode B resources before rendering (prepare_basis_resources)')
    return _active


def prepare_basis_resources(path=None, *, build_if_missing=True) -> PreparedBasisAtlas:
    """Initialize outside playback, atomically adopting a fully verified atlas.

    A missing default image is built once. Corrupt/incompatible images fail
    explicitly; use tools/prepare_mode_b_basis.py --force to replace them.
    An explicit path is always revalidated, and failed adoption preserves the
    currently active atlas. No resource bytes are part of musical checkpoints.
    """
    global _active
    if path is None and _active is not None:
        return _active
    with _prepare_lock:
        if path is None and _active is not None:
            return _active
        target = default_resource_path() if path is None else Path(path)
        try:
            candidate = PreparedBasisAtlas(target)
        except FileNotFoundError:
            if not build_if_missing:
                raise
            from .basis_resource_builder import build_basis_resource
            logging.getLogger(__name__).warning('Preparing Mode B basis resource (%s); first startup requires about 288 MiB of disk space', target)
            try:
                build_basis_resource(target)
            except PermissionError:
                # Another process may have published and mapped the same image
                # while we built it (Windows prevents replacing mapped files).
                if not target.is_file():
                    raise
            candidate = PreparedBasisAtlas(target)
        _active = candidate
        from .harmonic_bank import get_basis_for_pitch
        get_basis_for_pitch.cache_clear()
        return candidate
