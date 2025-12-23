"""Quick test of first 3 presets with short durations."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

from audio_generation_tests.wtf_presets import (
    preset_topology_fold,
    preset_catastrophe_snap,
    preset_perfect_morph,
)
from audio_io import write_audio

print("Testing first 3 presets (1 second each)...")

print("1. Topology fold...")
L, R = preset_topology_fold(duration=1.0)
write_audio("test_01_topology", L, R, subdirectory="wtf_test", timestamp=False)
print(f"   Done: {len(L)} samples")

print("2. Catastrophe snap...")
L, R = preset_catastrophe_snap(duration=1.0)
write_audio("test_02_catastrophe", L, R, subdirectory="wtf_test", timestamp=False)
print(f"   Done: {len(L)} samples")

print("3. Perfect morph...")
L, R = preset_perfect_morph(duration=1.0)
write_audio("test_03_morph", L, R, subdirectory="wtf_test", timestamp=False)
print(f"   Done: {len(L)} samples")

print("\nAll 3 test presets complete!")
