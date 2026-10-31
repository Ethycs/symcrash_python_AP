"""Offline generation of exact-pitch Mode A tables and legacy C4 recipes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from numba import njit

from ..primitives import Q15_ONE, Q15_NEG_ONE, Q15_HALF
from . import hotpath_jit as hot
from ..pitch_rom import MIDI_CENT_PHASE_Q32
from .mode_a_recipe import PULSE_AMPLITUDES, LEGACY_PHASE_INCREMENT
from .mode_a_resource import (HEADER, MAGIC, VERSION, SLOTS, PHASES, NOTE_TABLES,
    PITCH_COUNT, _validated_keys, recipe_digest, supported_table_keys)


@njit(cache=True)
def _wave_sample(lut, phase, wave, legacy, pulse, harmonic_count):
    if wave == 0:
        return hot._sine_q15_k(lut, phase)
    value = np.int64(0)
    stop = 17 if legacy else harmonic_count + 1
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


@njit(cache=True, nogil=True)
def _generate_table(lut, phase_inc, wave, legacy, pulse):
    result = np.empty((64, SLOTS, PHASES), dtype=np.int16)
    block = np.empty(64, dtype=np.int16)
    coeffs = np.empty(64, dtype=np.int16)
    harmonic_count = min(32, (9 * (1 << 31)) // (10 * phase_inc))
    for p in range(PHASES):
        phase_p = p << 26
        for k in range(SLOTS):
            phase = (phase_p + k * phase_inc) & 0xffffffff
            for n in range(64):
                sample = _wave_sample(lut, phase, wave, legacy, pulse, harmonic_count)
                block[n] = max(Q15_NEG_ONE, min(Q15_ONE, sample * Q15_HALF // Q15_ONE))
                phase = (phase + phase_inc) & 0xffffffff
            hot.fwpt_64(block, 1, coeffs)
            result[:, k, p] = coeffs
    return result


def build_mode_a_resource(path, *, table_keys=None, progress=None, workers=None):
    """Stream a deterministic image to a sibling temporary, then replace it.

    Small diagnostic subsets are explicit; host adoption needs the full image.
    Interrupted generation leaves any previous destination intact.
    """
    keys = _validated_keys(supported_table_keys() if table_keys is None else table_keys)
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if type(workers) is not int or workers < 1:
        raise ValueError('Mode A builder workers must be a positive integer')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lut, pulse = hot._get_sine_lut_np(), np.asarray(PULSE_AMPLITUDES, dtype=np.float64)
    def generate(key):
        if key < NOTE_TABLES:
            wave, cent = divmod(key, PITCH_COUNT)
            phase_inc, legacy = MIDI_CENT_PHASE_Q32[cent], False
        else:
            wave, phase_inc, legacy = key - NOTE_TABLES, LEGACY_PHASE_INCREMENT, True
        return _generate_table(lut, phase_inc, wave, legacy, pulse).astype('<i2').tobytes()
    # Compile before workers start; batches bound pending results to 8 MiB.
    first = generate(keys[0])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(bytes(HEADER.size))
            directory = np.asarray(keys, dtype='<u4').tobytes()
            stream.write(directory)
            digest = hashlib.sha256(directory)
            stream.write(first)
            digest.update(first)
            if progress is not None:
                progress(1, len(keys))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(1, len(keys), 256):
                    batch = keys[start:start+256]
                    for payload in pool.map(generate, batch):
                        stream.write(payload)
                        digest.update(payload)
                    if progress is not None:
                        progress(start + len(batch), len(keys))
            stream.seek(0)
            stream.write(HEADER.pack(MAGIC, VERSION, len(keys), recipe_digest(), digest.digest()))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
