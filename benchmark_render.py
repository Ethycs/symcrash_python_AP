"""Quick benchmark for render pipeline."""
import time
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

stage = UnifiedOutputStage()
stage.voice_on(0, note=60)
frame = VoiceFrame(voice_id=0, note=60, gain_q15=16384)

# Warmup
for _ in range(10):
    stage.render_voice(frame)

# Benchmark
n_blocks = 1000
start = time.perf_counter()
for _ in range(n_blocks):
    stage.render_voice(frame)
elapsed = time.perf_counter() - start

print(f"Benchmarking stage.render_voice() for {n_blocks} blocks...")
print(f"  Total: {elapsed*1000:.2f}ms, Per block: {elapsed/n_blocks*1000:.3f}ms")
print()
print(f"Real-time requirement: 1.451ms per block")
print(f"Actual render time: {elapsed/n_blocks*1000:.3f}ms per block")
ratio = (elapsed/n_blocks*1000) / 1.451
if ratio < 1:
    print(f"=> Rendering is {1/ratio:.1f}x FASTER than real-time!")
else:
    print(f"=> Rendering is {ratio:.1f}x SLOWER than real-time!")
