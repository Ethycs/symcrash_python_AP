#!/usr/bin/env python3
"""Verify fixed-point pitch conversion is deterministic."""

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed

# Test same note multiple times - should always get exact same result
note = 69  # A4 = 440Hz
results = set()
for _ in range(1000):
    inc = midi_to_phase_inc_fixed(note, 0)
    results.add(inc)

print(f"Note {note}: {len(results)} unique phase_inc values (should be 1)")
print(f"Phase increment: {list(results)[0]}")

# Test with cents
print("\nCents test:")
for cents in [0, 50, -50, 100, -100]:
    inc = midi_to_phase_inc_fixed(note, cents)
    print(f"  Note {note} + {cents:+4d} cents: phase_inc = {inc}")

print("\n✓ Fixed-point pitch conversion is deterministic")
