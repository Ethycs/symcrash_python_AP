"""Compare MIDI-note and octave Mode A storage; never adopt a candidate.

Read an existing validated exact-cent image. Lossless anchor compression
reuses the separately tested codec; interpolation BETWEEN anchors is lossy.
All candidate renderers retain N64/K4/P64, int16 coefficients, existing
headroom and exact Q32 phase stepping. Carrier-only measurements are not
full-VM, E3, hardware-decoder, or perceptual qualification.

Run: pixi run python -m tools.audit_mode_a_anchors --output build/mode_a_anchors.json
"""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np
from numba import njit

from symcrash_python_ap.budgets.hardware_resources import (
    COEFFICIENT_STORE_BYTES, COEFFICIENT_WORKING_BYTES)
from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
from symcrash_python_ap.vm import basis_resource, hotpath_jit as hot
from symcrash_python_ap.vm import mode_a_resource as mode_a
from tools.audit_mode_a_lossless import encode_block, decode_block, DIRECTORY_ENTRY


ANCHORS = {'midi': tuple(range(128)), 'octave': (*range(0, 128, 12), 127)}


def bracket(cent, scheme):
    """Endpoint-safe bracketing; no wrapping MIDI127 or rounding pitch."""
    if type(cent) is not int or not 0 <= cent <= 12700:
        raise ValueError('Pitch must be an absolute integer cent in 0..12700')
    anchors = tuple(n * 100 for n in ANCHORS[scheme])
    i = bisect_left(anchors, cent)
    if anchors[i] == cent:
        return cent, cent
    return anchors[i - 1], anchors[i]


def interpolate(lower, upper, numerator, denominator):
    """Round the complete convex interpolation to even, including negative ties."""
    if not 0 <= numerator <= denominator or denominator <= 0:
        raise ValueError('Invalid interpolation fraction')
    a, b = lower.astype(np.int64), upper.astype(np.int64)
    weighted = a * (denominator - numerator) + b * numerator
    quotient, remainder = np.divmod(weighted, denominator)
    quotient += ((2 * remainder > denominator) |
                 ((2 * remainder == denominator) & ((quotient & 1) != 0)))
    return quotient.astype(np.int16)


def candidate_table(atlas, wave, cent, scheme):
    low, high = bracket(cent, scheme)
    lower = atlas.table(wave * mode_a.PITCH_COUNT + low).data
    if low == high:
        return lower
    upper = atlas.table(wave * mode_a.PITCH_COUNT + high).data
    inc = MIDI_CENT_PHASE_Q32
    return interpolate(lower, upper, inc[cent] - inc[low], inc[high] - inc[low])


@njit(cache=True)
def render_carrier(table, increment, blocks, start_phase=0):
    """Existing linear phase lookup and polyphase decoder, without VM operators."""
    coeff = np.empty(256, dtype=np.int16)
    output = np.empty(blocks * 64, dtype=np.int16)
    phase = np.int64(start_phase)
    for block in range(blocks):
        index = phase >> 26
        frac = phase & ((1 << 26) - 1)
        for leaf in range(64):
            for lane in range(4):
                a = np.int64(table[leaf, lane, index])
                b = np.int64(table[leaf, lane, (index + 1) & 63])
                coeff[leaf * 4 + lane] = hot._rshift_rne(
                    (a << 26) + (b - a) * frac, 26)
        hot.idwpt_polyphase(coeff, output[block * 64:(block + 1) * 64], 2)
        phase = (phase + 64 * increment) & 0xffffffff
    return output


def groups(kind):
    """Each group is independently encoded; flags can represent its key stride."""
    for wave in range(5):
        if kind == 'midi_octave_offsets':
            # Same pitch class over octaves: predictor difference is exact,
            # not an assumption that coefficients are scale invariant.
            note_groups = [tuple(range(pc, 128, 12)) for pc in range(12)]
        else:
            notes = ANCHORS['octave' if kind == 'octave_anchors' else 'midi']
            note_groups = [notes[i:i + 16] for i in range(0, len(notes), 16)]
        for notes in note_groups:
            yield tuple(wave * mode_a.PITCH_COUNT + note * 100 for note in notes)
    yield tuple(range(mode_a.NOTE_TABLES, mode_a.TABLE_COUNT))


def storage_audit(atlas):
    result = {}
    for kind in ('midi_anchors', 'midi_octave_offsets', 'octave_anchors'):
        payload_bytes = count = group_count = largest_group = 0
        durations = []
        original_digest, restored_digest = hashlib.sha256(), hashlib.sha256()
        largest_payload = 0
        for keys in groups(kind):
            source = np.stack([atlas.table(key).data for key in keys])
            payload = encode_block(source)
            tick = time.perf_counter_ns()
            restored = decode_block(payload, len(keys))
            durations.append((time.perf_counter_ns() - tick) / 1000)
            if not np.array_equal(source, restored):
                raise AssertionError(f'Lossless anchor mismatch: {kind}, {keys}')
            original_digest.update(source.astype('<i2', copy=False).tobytes())
            restored_digest.update(restored.astype('<i2', copy=False).tobytes())
            payload_bytes += len(payload)
            largest_payload = max(largest_payload, len(payload))
            count += len(keys)
            group_count += 1
            largest_group = max(largest_group, len(keys))
        image_bytes = mode_a.HEADER.size + DIRECTORY_ENTRY.size * group_count + payload_bytes
        result[kind] = dict(tables=count, groups=group_count,
            raw_image_bytes=mode_a.HEADER.size + count * (4 + mode_a.TABLE_BYTES),
            compressed_payload_bytes=payload_bytes, candidate_image_bytes=image_bytes,
            combined_with_mode_b_bytes=image_bytes + basis_resource.resource_size(),
            coefficient_storage_fits=image_bytes + basis_resource.resource_size() <= COEFFICIENT_STORE_BYTES,
            decoded_coefficients=count * 64 * 4 * 64, every_anchor_bit_exact=True,
            source_sha256=original_digest.hexdigest(), decoded_sha256=restored_digest.hexdigest(),
            largest_decoded_group_bytes=largest_group * mode_a.TABLE_BYTES,
            # Output alone is a lower bound; codec also uses widened arrays/zlib state.
            decoded_output_alone_fits_working_partition=largest_group * mode_a.TABLE_BYTES <= COEFFICIENT_WORKING_BYTES,
            largest_compressed_group_bytes=largest_payload,
            host_decode_us=dict(mean=float(np.mean(durations)), p95=float(np.percentile(durations, 95)),
                                max=max(durations), all_observations=durations))
        print(json.dumps({'storage': kind, **{k: result[kind][k] for k in
              ('tables', 'candidate_image_bytes', 'combined_with_mode_b_bytes', 'coefficient_storage_fits')}}), flush=True)
    return result


def spectral_metrics(samples, frequency, *, sine):
    samples = samples.astype(np.float64)
    differences = np.diff(samples)
    boundary = np.arange(len(differences)) % 64 == 63
    within = float(np.sqrt(np.mean(differences[~boundary] ** 2)))
    ratio = float(np.sqrt(np.mean(differences[boundary] ** 2))) / max(within, 1e-30)
    result = dict(boundary_difference_rms_ratio=ratio, boundary_below_3=ratio < 3)
    if sine:
        window = np.hanning(len(samples))
        # Eightfold zero-padding reduces FFT bin-estimator error at low notes.
        size = 1 << (len(samples) * 8 - 1).bit_length()
        spectrum = np.abs(np.fft.rfft(samples * window, n=size))
        peak = int(np.argmax(spectrum[1:]) + 1)
        shift = 0.0
        if 0 < peak < len(spectrum) - 1:
            a, b, c = spectrum[peak - 1:peak + 2]
            if 2 * b != a + c:
                shift = 0.5 * (c - a) / (2 * b - a - c)
        measured = float((peak + shift) * 48000 / size)
        phase = 2 * np.pi * frequency * np.arange(len(samples)) / 48000
        basis = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(samples))))
        fitted = basis @ np.linalg.lstsq(basis, samples, rcond=None)[0]
        residual_peak = np.abs(np.fft.rfft((samples - fitted) * window))[1:].max()
        tone_peak = np.abs(np.fft.rfft(fitted * window))[1:].max()
        spur = float(20 * np.log10(max(residual_peak, 1e-30) / max(tone_peak, 1e-30)))
        result.update(measured_peak_hz=measured, pitch_error_percent=100 * abs(measured / frequency - 1),
            pitch_below_0_1_percent=abs(measured / frequency - 1) < .001,
            residual_peak_dbc=spur, residual_below_minus_60_dbc=spur < -60)
    return result


def quality_audit(atlas):
    increments = np.asarray(MIDI_CENT_PHASE_Q32, dtype=np.int64)
    caps = np.minimum(32, (9 * 2**31) // (10 * increments))
    boundaries = np.flatnonzero(np.diff(caps)) + 1
    cents = sorted(set(range(50, 12700, 100)) |
                   {int(c + d) for c in boundaries for d in (-1, 0, 1) if 0 <= c + d <= 12700})
    coefficient_rows = []
    for wave in range(5):
        for cent in cents:
            reference = atlas.table(wave * mode_a.PITCH_COUNT + cent).data.astype(np.int64)
            for scheme in ANCHORS:
                candidate = candidate_table(atlas, wave, cent, scheme).astype(np.int64)
                difference = candidate - reference
                coefficient_rows.append(dict(wave=wave, cent=cent, scheme=scheme,
                    max_error=int(np.abs(difference).max()),
                    rms_error=float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))))
    print(f'Compared {len(coefficient_rows)} candidate coefficient tables, including all harmonic-cap transitions.', flush=True)
    audio_rows = []
    audio_cents = (50, 2450, 4850, 6050, 6500, 6900, 6950, 8100, 8150, 9650, 10800, 10850, 12050, 12650)
    render_carrier(np.zeros((64, 4, 64), dtype=np.int16), 1000, 1)
    for wave in range(5):
        for cent in audio_cents:
            increment = int(increments[cent])
            frequency = increment * 48000 / 2**32
            # At least 32 cycles and one second, capped by the supported lowest pitch.
            blocks = max(750, int(np.ceil(32 * 48000 / frequency / 64))) + 8
            reference_table = atlas.table(wave * mode_a.PITCH_COUNT + cent).data
            reference = render_carrier(reference_table, increment, blocks)[512:]
            reference_metrics = spectral_metrics(reference, frequency, sine=wave == 0)
            for scheme in (*ANCHORS, 'octave_reuse_uncorrected'):
                if scheme == 'octave_reuse_uncorrected':
                    # Test the tempting one-octave pitch-class reuse explicitly;
                    # this does not remap leaves or claim scale invariance.
                    candidate = candidate_table(atlas, wave, 6000 + cent % 1200, 'midi')
                else:
                    candidate = candidate_table(atlas, wave, cent, scheme)
                rendered = render_carrier(candidate, increment, blocks)[512:]
                delta = rendered.astype(np.float64) - reference.astype(np.float64)
                rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
                error_rms = float(np.sqrt(np.mean(delta ** 2)))
                metrics = spectral_metrics(rendered, frequency, sine=wave == 0)
                audio_rows.append(dict(wave=wave, cent=cent, scheme=scheme, samples=len(rendered),
                    expected_hz=frequency, exact_samples=bool(np.array_equal(rendered, reference)),
                    pcm_max_error=int(np.abs(delta).max()), rms_error=error_rms,
                    error_db_relative_to_reference=None if error_rms == 0 else
                        float(20 * np.log10(error_rms / max(rms, 1e-30))),
                    reference=reference_metrics, candidate=metrics))
        print(f'Carrier audio comparisons complete: {mode_a.WAVE_NAMES[wave]}', flush=True)
    return dict(coefficient_pitches=cents, coefficient_rows=coefficient_rows,
                audio_pitches=audio_cents, audio_rows=audio_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource', type=Path, default=mode_a.default_resource_path())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atlas = mode_a.PreparedModeAAtlas(args.resource)
    header = mode_a.HEADER.unpack_from(atlas._mapping)
    report = dict(scope=__doc__, source=dict(image_bytes=args.resource.stat().st_size,
        recipe_sha256=header[3].hex(), directory_data_sha256=header[4].hex()),
        inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), Path(__file__).with_name('audit_mode_a_lossless.py'),
                 Path(hot.__file__), Path(mode_a.__file__))},
        geometry=dict(N=64, K=4, P=64, coefficient_bits=16, phase_bits=32),
        limits=dict(shared_coefficient_bytes=COEFFICIENT_STORE_BYTES,
            mode_b_bytes=basis_resource.resource_size(), working_bytes=COEFFICIENT_WORKING_BYTES),
        metadata='Candidate size model: 80-byte header, 16-byte group descriptors with first key/count/anchor-layout flag/offset/length. Layout flags distinguish semitone, pitch-class octaves, octave anchors including MIDI127, and legacy; not an adopted file format.',
        codec=f'Phase/pitch difference, signed zigzag, byte planes, DEFLATE6 (zlib {zlib.ZLIB_RUNTIME_VERSION})',
        storage=storage_audit(atlas), quality=quality_audit(atlas), adopted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}; production resources and playback unchanged.', flush=True)


if __name__ == '__main__':
    main()
