"""
Click Analysis - Trust but Verify
=================================

Analyze where clicks actually occur in the chromatic scale tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.oscillator import (
    WPOState, BLOCK_SIZE, SAMPLE_RATE, Q15_ONE, Q15_SCALE,
    TreeArchetype, midi_to_frequency
)

# Analysis parameters
SLEW_THRESHOLD = 4000  # Sample-to-sample delta that indicates a potential click
BLOCKS_PER_SEC = SAMPLE_RATE // BLOCK_SIZE


def find_clicks(samples: list, threshold: int = SLEW_THRESHOLD) -> list:
    """Find sample indices where slew rate exceeds threshold."""
    clicks = []
    for i in range(1, len(samples)):
        delta = abs(samples[i] - samples[i - 1])
        if delta > threshold:
            clicks.append((i, delta, samples[i - 1], samples[i]))
    return clicks


def analyze_chromatic_scale():
    """Analyze the phase-continuous chromatic scale for clicks."""
    print("=" * 70)
    print("CHROMATIC SCALE CLICK ANALYSIS")
    print("=" * 70)

    osc = WPOState(voice_id=0)
    osc.configure(note=48, waveform="saw")

    samples_per_note = int(0.25 * SAMPLE_RATE)  # 12000 samples
    all_samples = []
    note_boundaries = []  # Track where notes change

    for note in range(48, 73):  # C3 to C5
        if note > 48:
            # Record the boundary BEFORE changing pitch
            note_boundaries.append((len(all_samples), note - 1, note))
            osc.set_pitch(note)

        note_samples = []
        while len(note_samples) < samples_per_note:
            block = osc.render_block(BLOCK_SIZE)
            note_samples.extend(block)
        all_samples.extend(note_samples[:samples_per_note])

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Samples per note: {samples_per_note}")
    print(f"Blocks per note: {samples_per_note / BLOCK_SIZE:.1f}")
    print(f"Note boundaries: {len(note_boundaries)}")

    # Find all clicks
    clicks = find_clicks(all_samples, threshold=SLEW_THRESHOLD)

    print(f"\n--- HIGH SLEW EVENTS (delta > {SLEW_THRESHOLD}) ---")
    print(f"Total events: {len(clicks)}")

    # Analyze clicks near note boundaries
    boundary_clicks = []
    interior_clicks = []

    for click in clicks:
        sample_idx, delta, prev, curr = click

        # Is this click near a note boundary?
        near_boundary = False
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:  # Within one block
                boundary_clicks.append((click, boundary_idx, from_note, to_note))
                near_boundary = True
                break

        if not near_boundary:
            interior_clicks.append(click)

    print(f"\n--- BOUNDARY CLICKS (within 64 samples of note change) ---")
    print(f"Count: {len(boundary_clicks)}")
    for click, boundary, from_note, to_note in boundary_clicks[:10]:
        sample_idx, delta, prev, curr = click
        offset = sample_idx - boundary
        print(f"  Sample {sample_idx} (offset {offset:+d} from note {from_note}->{to_note}): "
              f"delta={delta}, {prev}->{curr}")

    print(f"\n--- INTERIOR CLICKS (not near note boundaries) ---")
    print(f"Count: {len(interior_clicks)}")
    if interior_clicks:
        # These are likely saw wave discontinuities (expected for bandlimited synthesis)
        for click in interior_clicks[:5]:
            sample_idx, delta, prev, curr = click
            # Figure out which note this is in
            note_idx = sample_idx // samples_per_note
            note = 48 + note_idx
            pos_in_note = sample_idx % samples_per_note
            print(f"  Sample {sample_idx} (note {note}, pos {pos_in_note}): "
                  f"delta={delta}, {prev}->{curr}")

    # Analyze note boundary positions
    print(f"\n--- NOTE BOUNDARY POSITIONS ---")
    for i, (boundary_idx, from_note, to_note) in enumerate(note_boundaries[:5]):
        block_idx = boundary_idx // BLOCK_SIZE
        pos_in_block = boundary_idx % BLOCK_SIZE
        print(f"  Note {from_note}->{to_note}: sample {boundary_idx}, "
              f"block {block_idx}, position in block: {pos_in_block}")

    return all_samples, clicks, note_boundaries


def analyze_with_phase_reset():
    """Analyze chromatic scale WITH phase reset (should click)."""
    print("\n" + "=" * 70)
    print("CHROMATIC SCALE WITH PHASE RESET (EXPECTED TO CLICK)")
    print("=" * 70)

    samples_per_note = int(0.25 * SAMPLE_RATE)
    all_samples = []
    note_boundaries = []

    for note in range(48, 73):
        if note > 48:
            note_boundaries.append((len(all_samples), note - 1, note))

        # NEW oscillator each note = phase reset
        osc = WPOState(voice_id=0)
        osc.configure(note=note, waveform="saw", reset_phase=True)

        note_samples = []
        while len(note_samples) < samples_per_note:
            block = osc.render_block(BLOCK_SIZE)
            note_samples.extend(block)
        all_samples.extend(note_samples[:samples_per_note])

    clicks = find_clicks(all_samples, threshold=SLEW_THRESHOLD)

    boundary_clicks = []
    for click in clicks:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks.append((click, boundary_idx, from_note, to_note))
                break

    print(f"\nTotal clicks: {len(clicks)}")
    print(f"Boundary clicks: {len(boundary_clicks)}")

    for click, boundary, from_note, to_note in boundary_clicks[:5]:
        sample_idx, delta, prev, curr = click
        offset = sample_idx - boundary
        print(f"  Sample {sample_idx} (offset {offset:+d} from note {from_note}->{to_note}): "
              f"delta={delta}, {prev}->{curr}")


def analyze_with_glide():
    """Analyze chromatic scale WITH glide (pitch interpolation)."""
    print("\n" + "=" * 70)
    print("CHROMATIC SCALE WITH GLIDE (PITCH INTERPOLATION)")
    print("=" * 70)

    osc = WPOState(voice_id=0)
    osc.configure(note=48, waveform="saw")

    samples_per_note = int(0.25 * SAMPLE_RATE)
    glide_blocks = 8  # ~10ms glide time
    all_samples = []
    note_boundaries = []

    for note in range(48, 73):
        if note > 48:
            note_boundaries.append((len(all_samples), note - 1, note))
            # Use glide instead of instant pitch change
            osc.glide_to(note, glide_blocks=glide_blocks)

        note_samples = []
        while len(note_samples) < samples_per_note:
            block = osc.render_block(BLOCK_SIZE)
            note_samples.extend(block)
        all_samples.extend(note_samples[:samples_per_note])

    clicks = find_clicks(all_samples, threshold=SLEW_THRESHOLD)

    boundary_clicks = []
    for click in clicks:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks.append((click, boundary_idx, from_note, to_note))
                break

    print(f"\nTotal high-slew events: {len(clicks)}")
    print(f"Boundary clicks: {len(boundary_clicks)}")

    for click, boundary, from_note, to_note in boundary_clicks[:5]:
        sample_idx, delta, prev, curr = click
        offset = sample_idx - boundary
        print(f"  Sample {sample_idx} (offset {offset:+d} from note {from_note}->{to_note}): "
              f"delta={delta}, {prev}->{curr}")


def analyze_with_regularity_operator():
    """Analyze chromatic scale WITH Regularity Operator R applied."""
    print("\n" + "=" * 70)
    print("CHROMATIC SCALE WITH REGULARITY OPERATOR R")
    print("=" * 70)

    from symcrash_python_ap.wavelet_engine import RegularityOperator, R_SLEW_MAX as SLEW_MAX

    osc = WPOState(voice_id=0)
    osc.configure(note=48, waveform="saw")

    samples_per_note = int(0.25 * SAMPLE_RATE)  # 12000 samples
    all_samples_raw = []
    all_samples_regularized = []
    note_boundaries = []

    R = RegularityOperator()

    for note in range(48, 73):  # C3 to C5
        if note > 48:
            note_boundaries.append((len(all_samples_raw), note - 1, note))
            osc.set_pitch(note)

        note_samples = []
        while len(note_samples) < samples_per_note:
            block = osc.render_block(BLOCK_SIZE)
            note_samples.extend(block)

        raw_note = note_samples[:samples_per_note]
        all_samples_raw.extend(raw_note)

        # Apply R to each note segment
        # Note: in practice R runs per-block, but we'll do per-note for clarity
        regularized = R.finalize(raw_note)
        all_samples_regularized.extend(regularized)

    print(f"\nTotal samples: {len(all_samples_raw)}")
    print(f"Note boundaries: {len(note_boundaries)}")

    # Find clicks in raw audio
    clicks_raw = find_clicks(all_samples_raw, threshold=SLEW_THRESHOLD)

    # Find clicks in regularized audio
    clicks_regularized = find_clicks(all_samples_regularized, threshold=SLEW_THRESHOLD)

    print(f"\n--- RAW AUDIO ---")
    print(f"High slew events: {len(clicks_raw)}")

    # Count boundary clicks in raw
    boundary_clicks_raw = []
    for click in clicks_raw:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks_raw.append((click, boundary_idx, from_note, to_note))
                break
    print(f"Boundary clicks: {len(boundary_clicks_raw)}")

    print(f"\n--- AFTER REGULARITY OPERATOR R ---")
    print(f"High slew events: {len(clicks_regularized)}")

    # Count boundary clicks in regularized
    boundary_clicks_reg = []
    for click in clicks_regularized:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks_reg.append((click, boundary_idx, from_note, to_note))
                break
    print(f"Boundary clicks: {len(boundary_clicks_reg)}")

    # Compute max slew in both
    max_slew_raw = 0
    prev = 0
    for s in all_samples_raw:
        delta = abs(s - prev)
        if delta > max_slew_raw:
            max_slew_raw = delta
        prev = s

    max_slew_reg = 0
    prev = 0
    for s in all_samples_regularized:
        delta = abs(s - prev)
        if delta > max_slew_reg:
            max_slew_reg = delta
        prev = s

    print(f"\n--- SLEW COMPARISON ---")
    print(f"Max slew (raw):        {max_slew_raw}")
    print(f"Max slew (regularized): {max_slew_reg}")
    print(f"SLEW_MAX threshold:    {SLEW_MAX}")
    print(f"Slew within bounds:    {max_slew_reg <= SLEW_MAX}")

    # Reduction percentage
    if len(clicks_raw) > 0:
        reduction = 100 * (1 - len(clicks_regularized) / len(clicks_raw))
        print(f"\nClick reduction: {reduction:.1f}%")

    if len(boundary_clicks_raw) > 0:
        boundary_reduction = 100 * (1 - len(boundary_clicks_reg) / len(boundary_clicks_raw))
        print(f"Boundary click reduction: {boundary_reduction:.1f}%")

    return all_samples_raw, all_samples_regularized, note_boundaries


def analyze_block_by_block_with_R():
    """Analyze with R applied block-by-block (realistic usage)."""
    print("\n" + "=" * 70)
    print("BLOCK-BY-BLOCK WITH REGULARITY OPERATOR R")
    print("=" * 70)

    from symcrash_python_ap.wavelet_engine import RegularityOperator, R_SLEW_MAX as SLEW_MAX

    osc = WPOState(voice_id=0)
    osc.configure(note=48, waveform="saw")

    samples_per_note = int(0.25 * SAMPLE_RATE)  # 12000 samples
    blocks_per_note = samples_per_note // BLOCK_SIZE

    all_samples_raw = []
    all_samples_regularized = []
    note_boundaries = []

    R = RegularityOperator()

    for note in range(48, 73):  # C3 to C5
        if note > 48:
            note_boundaries.append((len(all_samples_raw), note - 1, note))
            osc.set_pitch(note)

        for _ in range(blocks_per_note):
            block = osc.render_block(BLOCK_SIZE)
            all_samples_raw.extend(block)

            # Apply R per-block (realistic usage)
            regularized_block = R.finalize(block)
            all_samples_regularized.extend(regularized_block)

    print(f"\nTotal samples: {len(all_samples_raw)}")
    print(f"Total blocks: {len(all_samples_raw) // BLOCK_SIZE}")
    print(f"Note boundaries: {len(note_boundaries)}")

    # Find clicks
    clicks_raw = find_clicks(all_samples_raw, threshold=SLEW_THRESHOLD)
    clicks_regularized = find_clicks(all_samples_regularized, threshold=SLEW_THRESHOLD)

    # Count boundary clicks
    boundary_clicks_raw = []
    for click in clicks_raw:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks_raw.append((click, boundary_idx))
                break

    boundary_clicks_reg = []
    for click in clicks_regularized:
        sample_idx = click[0]
        for boundary_idx, from_note, to_note in note_boundaries:
            if abs(sample_idx - boundary_idx) < 64:
                boundary_clicks_reg.append((click, boundary_idx))
                break

    print(f"\n--- RAW AUDIO ---")
    print(f"High slew events: {len(clicks_raw)}")
    print(f"Boundary clicks: {len(boundary_clicks_raw)}")

    print(f"\n--- AFTER REGULARITY OPERATOR R (block-by-block) ---")
    print(f"High slew events: {len(clicks_regularized)}")
    print(f"Boundary clicks: {len(boundary_clicks_reg)}")

    # Max slew comparison
    max_slew_raw = max(abs(all_samples_raw[i] - all_samples_raw[i-1])
                       for i in range(1, len(all_samples_raw)))
    max_slew_reg = max(abs(all_samples_regularized[i] - all_samples_regularized[i-1])
                       for i in range(1, len(all_samples_regularized)))

    print(f"\n--- SLEW COMPARISON ---")
    print(f"Max slew (raw):        {max_slew_raw}")
    print(f"Max slew (regularized): {max_slew_reg}")
    print(f"SLEW_MAX threshold:    {SLEW_MAX}")
    print(f"Slew within bounds:    {max_slew_reg <= SLEW_MAX}")

    # Show improvement
    if len(clicks_raw) > 0:
        reduction = 100 * (1 - len(clicks_regularized) / len(clicks_raw))
        print(f"\nHigh-slew event reduction: {reduction:.1f}%")

    return all_samples_raw, all_samples_regularized


if __name__ == "__main__":
    analyze_chromatic_scale()
    analyze_with_phase_reset()
    analyze_with_glide()
    analyze_with_regularity_operator()
    analyze_block_by_block_with_R()
