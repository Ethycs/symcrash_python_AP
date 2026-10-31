"""Offline generation for fixed-anchor Mode B resources. Never called by lookup."""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from numba import njit

from ..primitives import Q15_ONE
from . import hotpath_jit as hot
from .basis_resource import (HEADER, MAGIC, VERSION, _validated_keys, recipe_digest,
                            supported_pitch_increments, resource_size)
from .basis_anchors import capacity_for_anchor, frequency_directory


@njit(cache=True)
def _generate_pitch(lut, phase_inc, count):
    """Same SineCore/64-point lifting arithmetic as the scalar reference."""
    result = np.zeros((2, count, 64, 4), dtype=np.int16)
    cosine = np.zeros(64, dtype=np.int16)
    sine = np.zeros(64, dtype=np.int16)
    for h in range(1, count + 1):
        for k in range(4):
            for n in range(64):
                phase = ((n + k) * phase_inc * h) & 0xffffffff
                sine[n] = hot._sine_q15_k(lut, phase)
                cosine[n] = hot._sine_q15_k(lut, (phase + 0x40000000) & 0xffffffff)
            hot.fwpt_64(cosine, 2, result[0, h-1, :, k])
            hot.fwpt_64(sine, 2, result[1, h-1, :, k])
    return result


def build_basis_resource(path, *, pitch_increments=None, progress=None):
    """Deterministic image, streamed to a sibling temporary file then replaced.

    pitch_increments allows small diagnostic resources; host adoption requires
    the complete default domain. Failure never replaces the previous resource.
    """
    keys = _validated_keys(supported_pitch_increments() if pitch_increments is None else pitch_increments)
    from ..budgets.hardware_resources import COEFFICIENT_STORE_BYTES
    if resource_size(keys) > COEFFICIENT_STORE_BYTES:
        raise ValueError('Mode B resource exceeds the documented coefficient storage budget')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lut = hot._get_sine_lut_np()
    dc_coeffs = np.zeros(64, dtype=np.int16)
    hot.fwpt_64(np.full(64, Q15_ONE, dtype=np.int16), 2, dc_coeffs)
    dc = np.repeat(dc_coeffs[:, None], 4, axis=1)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(bytes(HEADER.size))
            digest = hashlib.sha256()
            directory = np.asarray(frequency_directory(keys),dtype='<u4').tobytes()
            for payload in (np.asarray(keys, dtype='<u4').tobytes(), directory, dc.astype('<i2').tobytes()):
                stream.write(payload)
                digest.update(payload)
            for index, pitch in enumerate(keys):
                payload = _generate_pitch(lut, pitch, capacity_for_anchor(pitch)).astype('<i2').tobytes()
                stream.write(payload)
                digest.update(payload)
                if progress is not None and (index % 512 == 0 or index + 1 == len(keys)):
                    progress(index + 1, len(keys))
            stream.seek(0)
            stream.write(HEADER.pack(MAGIC, VERSION, len(keys), recipe_digest(), digest.digest()))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
