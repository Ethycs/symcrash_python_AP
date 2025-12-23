"""Fast test of all 12 presets with short durations."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

from audio_generation_tests.wtf_presets import (
    preset_topology_fold,
    preset_catastrophe_snap,
    preset_perfect_morph,
    preset_living_chord,
    preset_polyphonic_pressure,
    preset_intentional_aliasing,
    preset_vhs_ghost_choir,
    preset_pressure_field,
    preset_imminent_failure,
    preset_saw_that_learned,
    preset_bass_with_memory,
    preset_single_knob_apocalypse,
)
from audio_io import write_audio

presets = [
    ("01_topology_fold", preset_topology_fold),
    ("02_catastrophe_snap", preset_catastrophe_snap),
    ("03_perfect_morph", preset_perfect_morph),
    ("04_living_chord", preset_living_chord),
    ("05_polyphonic_pressure", preset_polyphonic_pressure),
    ("06_intentional_aliasing", preset_intentional_aliasing),
    ("07_vhs_ghost_choir", preset_vhs_ghost_choir),
    ("08_pressure_field", preset_pressure_field),
    ("09_imminent_failure", preset_imminent_failure),
    ("10_saw_that_learned", preset_saw_that_learned),
    ("11_bass_with_memory", preset_bass_with_memory),
    ("12_single_knob_apocalypse", preset_single_knob_apocalypse),
]

print("=" * 60)
print("GENERATING ALL 12 WTF PRESETS (1 second each)")
print("=" * 60)

for i, (name, func) in enumerate(presets, 1):
    print(f"{i:2}. {name}...", end=" ", flush=True)
    try:
        L, R = func(duration=1.0)
        write_audio(name, L, R, subdirectory="wtf_presets", timestamp=False)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

print("=" * 60)
print("ALL PRESETS COMPLETE")
print("=" * 60)
