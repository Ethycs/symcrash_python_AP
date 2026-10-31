"""Measure lossless block encodings of an existing, validated Mode A atlas.

Diagnostic only: no live resource adoption, waveform generation or precision
change. Every decoded coefficient must match before its size is counted.
The complete scan includes every pitch/wave and the three legacy tables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np

from symcrash_python_ap.budgets.hardware_resources import COEFFICIENT_STORE_BYTES, EXTERNAL_RAM_BYTES
from symcrash_python_ap.vm import basis_resource, mode_a_resource as mode_a

VALUES_PER_TABLE = 64 * mode_a.SLOTS * mode_a.PHASES
MAX_BLOCK_TABLES = 32
DIRECTORY_ENTRY = struct.Struct('<IHHII')  # first key, count, flags, offset, bytes


def _shape(count):
    if type(count) is not int or not 1 <= count <= MAX_BLOCK_TABLES:
        raise ValueError('Block count must be 1..32 tables')
    return (count, 64, mode_a.SLOTS, mode_a.PHASES)


def encode_block(values, *, predicted=True):
    if values.dtype != np.dtype('int16') or values.shape != _shape(len(values)):
        raise ValueError('Expected int16 [table,64,4,64] input')
    if not predicted:
        return zlib.compress(values.astype('<i2', copy=False).tobytes(), level=6)
    wide = values.astype(np.int32)
    phase_delta = wide.copy()
    phase_delta[..., 1:] -= wide[..., :-1]
    residual = phase_delta.copy()
    residual[1:] -= phase_delta[:-1]
    # First differences fit 17 bits; mixed differences fit signed 18 bits.
    # Zigzag is reversible, including negative extrema. No low bits are lost.
    codes = ((residual << 1) ^ (residual >> 31)).astype('<u4')
    planes = codes.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    return zlib.compress(planes, level=6)


def decode_block(payload, count, *, predicted=True):
    shape = _shape(count)
    expected_bytes = count * VALUES_PER_TABLE * (4 if predicted else 2)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload, expected_bytes + 1)
    if (len(decoded) != expected_bytes or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('Compressed block extent mismatch')
    if not predicted:
        return np.frombuffer(decoded, dtype='<i2').reshape(shape)
    codes = np.frombuffer(decoded, dtype=np.uint8).reshape(4, -1).T.copy()
    codes = codes.view('<u4').reshape(shape)
    if np.any(codes > 262140):
        raise ValueError('Mixed residual exceeds its proven 18-bit range')
    residual = (codes >> 1).astype(np.int64) ^ -(codes & 1).astype(np.int64)
    restored = residual.cumsum(axis=0).cumsum(axis=-1)
    if np.any(restored < -32768) or np.any(restored > 32767):
        raise ValueError('Decoded coefficient is outside int16')
    return restored.astype(np.int16)


def audit(path, *, block_tables=16, quick=False, progress=None):
    _shape(block_tables)
    path = Path(path)
    atlas = mode_a.PreparedModeAAtlas(path)
    data_start = mode_a.HEADER.size + len(atlas.keys) * 4
    source = np.ndarray((len(atlas.keys), 64, mode_a.SLOTS, mode_a.PHASES),
                        dtype='<i2', buffer=atlas._mapping, offset=data_start)
    with path.open('rb') as stream:
        _, version, count, recipe, data_digest = mode_a.HEADER.unpack(stream.read(mode_a.HEADER.size))
    rows = []
    for wave, name in enumerate((*mode_a.WAVE_NAMES, 'legacy')):
        low = wave * mode_a.PITCH_COUNT if wave < 5 else mode_a.NOTE_TABLES
        high = low + mode_a.PITCH_COUNT if wave < 5 else mode_a.TABLE_COUNT
        starts = list(range(low, high, block_tables))
        if quick:
            starts = sorted({starts[0], starts[len(starts)//2], starts[-1]})
        row = dict(wave=name, tables=0, blocks=0, source_bytes=0, source_sha256='', candidates={})
        source_digest = hashlib.sha256()
        for predicted in (False, True):
            row['candidates']['pitch_phase_delta' if predicted else 'raw_deflate'] = dict(
                payload_bytes=0, largest_block_bytes=0, encode_seconds=0.0, decode_seconds=0.0,
                largest_decode_seconds=0.0, roundtrip_sha256='')
        decoded_digests = {key: hashlib.sha256() for key in row['candidates']}
        for start in starts:
            values = source[start:min(start+block_tables, high)]
            raw = values.tobytes()
            source_digest.update(raw)
            row['tables'] += len(values)
            row['blocks'] += 1
            row['source_bytes'] += len(raw)
            for predicted in (False, True):
                key = 'pitch_phase_delta' if predicted else 'raw_deflate'
                measured = row['candidates'][key]
                tick = time.perf_counter()
                payload = encode_block(values, predicted=predicted)
                measured['encode_seconds'] += time.perf_counter() - tick
                tick = time.perf_counter()
                restored = decode_block(payload, len(values), predicted=predicted)
                elapsed = time.perf_counter() - tick
                measured['decode_seconds'] += elapsed
                measured['largest_decode_seconds'] = max(measured['largest_decode_seconds'], elapsed)
                if not np.array_equal(restored, values):
                    raise AssertionError(f'Lossless mismatch at table {start}, {key}')
                decoded_digests[key].update(restored.astype('<i2', copy=False).tobytes())
                measured['payload_bytes'] += len(payload)
                measured['largest_block_bytes'] = max(measured['largest_block_bytes'], len(payload))
        row['source_sha256'] = source_digest.hexdigest()
        for key, measured in row['candidates'].items():
            measured['roundtrip_sha256'] = decoded_digests[key].hexdigest()
            assert measured['roundtrip_sha256'] == row['source_sha256']
        rows.append(row)
        if progress:
            progress(row)
    blocks = sum(row['blocks'] for row in rows)
    totals = {}
    for key in rows[0]['candidates']:
        payload = sum(row['candidates'][key]['payload_bytes'] for row in rows)
        # Explicit candidate accounting, not a published/adopted file format.
        image = mode_a.HEADER.size + DIRECTORY_ENTRY.size * blocks + payload
        combined = image + basis_resource.resource_size()
        totals[key] = dict(payload_bytes=payload, directory_bytes=DIRECTORY_ENTRY.size*blocks,
            candidate_image_bytes=image, combined_with_mode_b_bytes=combined,
            coefficient_budget_passed=(not quick and combined <= COEFFICIENT_STORE_BYTES),
            external_capacity_passed=(not quick and combined <= EXTERNAL_RAM_BYTES))
    return dict(scope='Lossless storage experiment; no live adoption or hardware decoder/timing qualification',
        full_domain=not quick, block_tables=block_tables, table_count=sum(r['tables'] for r in rows),
        coefficient_count=sum(r['source_bytes']//2 for r in rows), zlib_version=zlib.ZLIB_RUNTIME_VERSION,
        source=dict(path=path.name, bytes=path.stat().st_size, version=version, tables=count,
                    recipe_sha256=recipe.hex(), directory_data_sha256=data_digest.hex()),
        limits=dict(coefficients=COEFFICIENT_STORE_BYTES, external_ram=EXTERNAL_RAM_BYTES,
                    mode_b_bytes=basis_resource.resource_size()),
        candidate_metadata='80-byte header plus 16-byte independently addressable block descriptors',
        block_raw_bytes=block_tables*mode_a.TABLE_BYTES,
        rows=rows, totals=totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource', type=Path, default=mode_a.default_resource_path())
    parser.add_argument('--block-tables', type=int, default=16)
    parser.add_argument('--quick', action='store_true', help='First/middle/last block per wave; never a full budget pass')
    parser.add_argument('--output', type=Path, help='Write the detailed report as JSON')
    args = parser.parse_args()
    def progress(row):
        print(json.dumps(dict(wave=row['wave'], tables=row['tables'],
              payload_bytes={key:value['payload_bytes'] for key,value in row['candidates'].items()})), flush=True)
    result = audit(args.resource, block_tables=args.block_tables, quick=args.quick, progress=progress)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(full_domain=result['full_domain'], totals=result['totals']), indent=2), flush=True)
    print(result['scope'], flush=True)


if __name__ == '__main__':
    main()
