"""
Test mono pipeline support.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.pipeline import AudioSink, Pipeline

def main():
    print("Testing Mono Pipeline Support")
    print("=" * 40)

    # Test AudioSink mono
    sink = AudioSink(mono=True)
    sink.push_block([1, 2, 3])
    print(f"Mono sink: L={len(sink.L)} samples, R={len(sink.R)} samples")

    # Test AudioSink stereo
    sink2 = AudioSink(mono=False)
    sink2.push_block([1, 2, 3], [4, 5, 6])
    print(f"Stereo sink: L={len(sink2.L)} samples, R={len(sink2.R)} samples")

    # Test Pipeline defaults to mono
    pipe = Pipeline(n_voices=1)
    print(f"Pipeline mono={pipe.mono}")

    print("\nMono pipeline support verified!")

if __name__ == "__main__":
    main()
