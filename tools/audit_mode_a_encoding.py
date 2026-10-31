"""Measure the older Mode A storage proposal against an existing validated image.

This is a coefficient-domain diagnostic, not an audio-quality or hardware pass.
It never builds resources or changes the live encoding. The default scan covers
five native waves at all 128 nominal MIDI notes, not every detuned table.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from symcrash_python_ap.vm.mode_a_resource import (
    HEADER, PHASES, PITCH_COUNT, SLOTS, TABLE_BYTES, WAVE_NAMES,
    PreparedModeAAtlas, default_resource_path, note_address,
)


def audit(path):
    atlas = PreparedModeAAtlas(path)
    with Path(path).open('rb') as stream:
        _, version, table_count, recipe, checksum = HEADER.unpack(stream.read(HEADER.size))
    rows = []
    half = PHASES // 2
    for wave, name in enumerate(WAVE_NAMES):
        minimum, maximum, worst = 65536, -65536, None
        for note in range(128):
            # Widen before subtraction, addition or negation: int16 can hide
            # exactly the failures this diagnostic is intended to expose.
            data = atlas.table(note_address(wave, note)[2]).data.astype(np.int64)
            decoded = (data >> 8).astype(np.int8).astype(np.int64) << 8
            delta = data - decoded
            minimum = min(minimum, int(delta.min()))
            maximum = max(maximum, int(delta.max()))
            asymmetry = data[:, :, :half] + data[:, :, half:]
            magnitude = int(np.abs(asymmetry).max())
            if worst is None or magnitude > worst['absolute_error']:
                location = np.unravel_index(np.abs(asymmetry).argmax(), asymmetry.shape)
                leaf, lane, phase = (int(value) for value in location)
                worst = dict(absolute_error=magnitude, note=note,
                    first_location=[leaf, lane, phase],
                    first=int(data[leaf, lane, phase]),
                    second=int(data[leaf, lane, phase + half]))
        rows.append(dict(wave=name, floor_codec_error_min=minimum,
            floor_codec_error_max=maximum, half_phase_worst=worst))

    # Exhaustive scalar-width check, separate from the measured table subset.
    values = np.arange(-32768, 32768, dtype=np.int64)
    floor_delta = values - ((values >> 8) << 8)
    single_half_int8 = TABLE_BYTES // 4
    return dict(
        scope='Coefficient errors only; no codec adoption, audio or realtime qualification',
        source=dict(version=version, table_count=table_count,
            image_bytes=Path(path).stat().st_size, recipe_sha256=recipe.hex(),
            directory_data_sha256=checksum.hex()),
        coverage=dict(waves=len(WAVE_NAMES), nominal_notes_per_wave=128,
            tables=len(WAVE_NAMES) * 128, leaves=64, lanes=SLOTS, phases=PHASES,
            detuned_tables_scanned=False),
        floor_codec_int16_domain=dict(error_definition='source - decoded',
            error_min=int(floor_delta.min()), error_max=int(floor_delta.max())),
        cases=rows,
        storage=dict(bytes_per_full_int16_table=TABLE_BYTES,
            bytes_per_half_int8_table=single_half_int8,
            four_wave_half_int8_bytes=4 * single_half_int8,
            five_wave_complete_cent_full_int8_bytes=len(WAVE_NAMES) * PITCH_COUNT * TABLE_BYTES // 2,
            five_wave_complete_cent_half_int8_bytes=len(WAVE_NAMES) * PITCH_COUNT * single_half_int8,
            caveat='Native payload only; half-phase assumes unproven symmetry and excludes metadata'),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource', type=Path, default=default_resource_path(),
        help='Existing full Mode A image; missing or invalid images fail without building')
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.resource), indent=2))


if __name__ == '__main__':
    main()
