"""Prepared fixed-anchor Mode B resources; no transforms in the lookup path.

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
from .basis_anchors import (ANCHORS, capacity_for_anchor, frequency_directory,
                           interpolation_grid, frequency_stencil, weights_for_frequency,
                           compiled_descriptor_arrays)

MAGIC = b'SYMBAS03'
VERSION = 3
HEADER = struct.Struct('<8sII32s32s')
DC_BYTES = 64 * 4 * 2
HARMONIC_BYTES = 2 * DC_BYTES
MAX_PITCH_ENTRIES = 65536
_active = None
_prepare_lock = Lock()


@lru_cache(maxsize=1)
def supported_pitch_increments() -> tuple[int, ...]:
    """Fixed semitone anchors; accepted target pitches retain integer-cent Q32."""
    return ANCHORS


def resource_size(keys=None):
    keys = supported_pitch_increments() if keys is None else keys
    return HEADER.size + 4*len(keys) + 12*len(frequency_directory(keys)) + DC_BYTES + sum(
        capacity_for_anchor(p)*HARMONIC_BYTES for p in keys)


@lru_cache(maxsize=1)
def recipe_digest() -> bytes:
    """Version the arithmetic recipe and the frozen sine/pitch inputs.

    Bump the recipe version if SineCore or the FWPT arithmetic changes. The
    independent reference comparisons in test_basis_resource guard this rule.
    """
    from .hotpath_jit import _get_sine_lut_np
    digest = hashlib.sha256(b'ModeB-basis-v3/Fs48000/N64/K4/stride1/H32/headroom2/'
        b'SineCore-RNE/FWPT-RNE/shared-harmonic-frequency-anchors/'
        b'cluster64/outer-span/Lagrange4-Q30-ratio-product-RNE/unity-largest-tie-first')
    digest.update(_get_sine_lut_np().astype('<i4').tobytes())
    digest.update(np.asarray(supported_pitch_increments(), dtype='<u4').tobytes())
    from ..pitch_rom import MIDI_CENT_PHASE_Q32
    digest.update(np.asarray(MIDI_CENT_PHASE_Q32, dtype='<u4').tobytes())
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
        directory = frequency_directory(keys)
        directory_bytes = np.asarray(directory,dtype='<u4').tobytes()
        expected_size = resource_size(keys)
        from ..budgets.hardware_resources import COEFFICIENT_STORE_BYTES
        if expected_size > COEFFICIENT_STORE_BYTES:
            raise ValueError('Mode B resource exceeds the documented coefficient storage budget')
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
            if mapping[data_start:data_start+len(directory_bytes)] != directory_bytes:
                raise ValueError('Basis resource frequency directory mismatch')
            # Validate the complete backing bytes before adoption.
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
        self.frequency_directory = np.ndarray((len(directory),3),dtype='<u4',buffer=mapping,offset=data_start)
        grid = interpolation_grid(directory)
        self._grid_indices = grid
        self._grid_array = np.frombuffer(np.asarray(grid,dtype=np.int32).tobytes(),dtype=np.int32)
        self._grid_frequencies = tuple(directory[i][0] for i in grid)
        self._grid_frequency_array = np.frombuffer(
            np.asarray(self._grid_frequencies,dtype=np.uint32).tobytes(),dtype=np.uint32)
        self._frequency_offsets = {frequency:(cosine,sine) for frequency,cosine,sine in directory}
        # Compile the fixed-width port during preparation, before any lookup.
        # This retains the Python formula, Q30 resolution and every RNE step.
        if keys == ANCHORS:
            compiled_descriptor_arrays(ANCHORS[0],0,self.frequency_directory,
                                       self._grid_array,self._grid_frequency_array)
        self.dc = np.ndarray((64, 4), dtype='<i2', buffer=mapping, offset=data_start+len(directory_bytes))
        offset = data_start + len(directory_bytes) + DC_BYTES
        self.coefficients = np.ndarray(((expected_size-offset)//2,),dtype='<i2',buffer=mapping,offset=offset)
        self._offsets = {}
        for pitch in keys:
            count = capacity_for_anchor(pitch)
            self._offsets[pitch] = (offset, count)
            offset += count * HARMONIC_BYTES
        if keys == ANCHORS:
            self._prepare_interpolated_kernel()

    def _prepare_interpolated_kernel(self):
        """Resolve the new audio-kernel signature during host preparation.

        Zero harmonics touch no basis entries or musical state. Array
        mutability and layout match actual descriptors, weights and VM output.
        """
        from . import hotpath_jit as hot
        offsets = np.frombuffer(bytes(0),dtype=np.int32).reshape(0,4,2)
        weights = np.frombuffer(bytes(0),dtype=np.int32).reshape(0,4)
        harmonics = np.empty(0,dtype=np.int64)
        phases = np.zeros(4,dtype=np.int64)
        output = np.empty(64*4,dtype=np.int16)
        hot.mode_b_interpolated_coefficients(hot._get_sine_lut_np(),
            harmonics,harmonics,0,phases,self.dc,self.coefficients,offsets,weights,output)

    def basis(self, note: int, phase_inc: int, harmonics: int):
        from .harmonic_bank import HarmonicBasisFrames
        if type(phase_inc) is not int or not 0 <= phase_inc < 2**32:
            raise ValueError('Basis phase increment must be unsigned Q32')
        if type(harmonics) is not int or not 0 <= harmonics <= 32:
            raise ValueError('Basis harmonic count must be in 0..32')
        if harmonics > harmonic_limit(phase_inc):
            raise ValueError('Basis harmonic count exceeds the prepared pitch bandlimit')
        if phase_inc not in self._offsets:
            if self.keys != ANCHORS:
                raise ValueError('Pitch increment is absent from the diagnostic basis resource')
            from .basis_anchors import InterpolatedBasis, pitch_index
            from .hotpath_jit import jit_enabled
            pitch_index(phase_inc)
            if jit_enabled():
                offsets,weights = compiled_descriptor_arrays(phase_inc,harmonics,
                    self.frequency_directory,self._grid_array,self._grid_frequency_array)
            else:
                offsets = np.empty((harmonics,4,2),dtype=np.int32)
                weights = np.zeros((harmonics,4),dtype=np.int32)
                for h in range(harmonics):
                    frequency=phase_inc*(h+1)
                    exact=self._frequency_offsets.get(frequency)
                    if exact is not None:
                        offsets[h]=exact
                        weights[h,0]=1<<30
                    else:
                        stencil=frequency_stencil(frequency,self._grid_frequencies)
                        nodes=tuple(self._grid_frequencies[i] for i in stencil)
                        weights[h]=weights_for_frequency(frequency,nodes)
                        offsets[h]=[self.frequency_directory[self._grid_indices[i],1:] for i in stencil]
            offsets=np.frombuffer(offsets.tobytes(),dtype=np.int32).reshape(offsets.shape)
            weights=np.frombuffer(weights.tobytes(),dtype=np.int32).reshape(weights.shape)
            return InterpolatedBasis(self.dc,self.coefficients,offsets,weights,harmonics,
                note, phase_inc*48000/2**32)
        data = self._anchor(phase_inc, harmonics)
        return HarmonicBasisFrames(self.dc, data[0], data[1],
                                   harmonics, note, phase_inc * 48000 / 2**32)

    def _anchor(self, phase_inc, harmonics):
        offset, capacity = self._offsets[phase_inc]
        data = np.ndarray((2, capacity, 64, 4), dtype='<i2', buffer=self._mapping, offset=offset)
        return data[:, :harmonics]


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
            logging.getLogger(__name__).warning('Preparing fixed-anchor Mode B basis resource (%s)', target)
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
