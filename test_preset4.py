"""Test preset 4 (living chord) with short duration."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

from audio_generation_tests.wtf_presets import preset_living_chord
from audio_io import write_audio

print("Testing preset 4 (living chord) with 12 voices, 0.5 seconds...")
L, R = preset_living_chord(duration=0.5)
write_audio("test_04_chord", L, R, subdirectory="wtf_test", timestamp=False)
print(f"Done: {len(L)} samples")
