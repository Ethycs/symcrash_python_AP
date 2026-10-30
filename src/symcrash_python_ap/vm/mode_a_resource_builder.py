"""Offline generation of the original Mode A native and legacy table recipes."""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from numba import njit

from ..primitives import Q15_ONE, Q15_NEG_ONE, Q15_HALF
from . import hotpath_jit as hot
from .mode_a_recipe import NOTE_PHASE_INCREMENTS, PULSE_AMPLITUDES, LEGACY_PHASE_INCREMENT
from .mode_a_resource import (HEADER, MAGIC, VERSION, SLOTS, PHASES, NOTE_TABLES,
    _validated_keys, recipe_digest, supported_table_keys)


@njit(cache=True)
def _wave_sample(lut, phase, wave, legacy, pulse):
    if wave == 0:
        return hot._sine_q15_k(lut, phase)
    value = np.int64(0)
    stop = 17 if legacy else 33
    for h in range(1, stop):
        if wave in (2, 3) and h % 2 == 0:
            continue
        sine = hot._sine_q15_k(lut, (phase * h) & 0xffffffff)
        if wave == 1:
            sign = 1 if legacy or h % 2 else -1
            value += (sign * sine) // h
        elif wave == 2:
            value += sine // h
        elif wave == 3:
            sign = 1 if h % 4 == 1 else -1
            value += (sign * sine) // (h * h)
        elif abs(pulse[h - 1]) > 0.01:
            value += int((pulse[h - 1] * sine) / h)
    if wave == 1:
        value = value // 2 if legacy else value * 2 // 3
    elif wave == 2:
        value = value if legacy else value * 3 // 4
    elif wave == 3:
        return max(-Q15_ONE, min(Q15_ONE, value * 13 // 16))
    else:
        value = value * 2 // 3
    return max(Q15_NEG_ONE, min(Q15_ONE, value))


@njit(cache=True)
def _generate_table(lut, phase_inc, wave, legacy, pulse):
    result = np.empty((64, SLOTS, PHASES), dtype=np.int16)
    block = np.empty(64, dtype=np.int16)
    coeffs = np.empty(64, dtype=np.int16)
    for p in range(PHASES):
        phase_p = p << 26
        for k in range(SLOTS):
            phase = (phase_p + k * phase_inc) & 0xffffffff
            for n in range(64):
                sample = _wave_sample(lut, phase, wave, legacy, pulse)
                block[n] = max(Q15_NEG_ONE, min(Q15_ONE, sample * Q15_HALF // Q15_ONE))
                phase = (phase + phase_inc) & 0xffffffff
            hot.fwpt_64(block, 1, coeffs)
            result[:, k, p] = coeffs
    return result


def build_mode_a_resource(path, *, table_keys=None, progress=None):
    """Stream a deterministic image to a sibling temporary, then replace it.

    Small diagnostic subsets are explicit; host adoption needs the full image.
    Interrupted generation leaves any previous destination intact.
    """
    keys = _validated_keys(supported_table_keys() if table_keys is None else table_keys)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lut, pulse = hot._get_sine_lut_np(), np.asarray(PULSE_AMPLITUDES, dtype=np.float64)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(bytes(HEADER.size))
            directory = np.asarray(keys, dtype='<u4').tobytes()
            stream.write(directory)
            digest = hashlib.sha256(directory)
            for index, key in enumerate(keys):
                if key < NOTE_TABLES:
                    wave, note = divmod(key, 128)
                    phase_inc, legacy = NOTE_PHASE_INCREMENTS[note], False
                else:
                    wave, phase_inc, legacy = key - NOTE_TABLES, LEGACY_PHASE_INCREMENT, True
                payload = _generate_table(lut, phase_inc, wave, legacy, pulse).astype('<i2').tobytes()
                stream.write(payload)
                digest.update(payload)
                if progress is not None and (index % 16 == 0 or index + 1 == len(keys)):
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
