"""
Pipeline Orchestration
======================

Complete audio synthesis pipeline:
Symbol Engine -> Wavelet Engine -> Material Engine -> Safety Engine -> Output

Supports WAV export for listening tests.
"""

import sys
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/home/claude/synth_pipeline')

from typing import List, Tuple, Optional
import struct
import wave
import numpy as np

from fixed import Q15_ONE, Q15_HALF, sat16, mul_q15_sat
from frames import ParamFrame, VoiceFrame
from symbol_engine import SymbolEngine
from wavelet_engine import WaveletEngine
from material_engine import MaterialEngine
from safety_engine import SafetyEngine


class AudioSink:
    """Accumulates audio blocks and exports to WAV."""
    
    def __init__(self):
        self.L: List[int] = []
        self.R: List[int] = []
        self.block_count = 0
    
    def push_block(self, L: List[int], R: List[int]):
        """Add a stereo block."""
        self.L.extend(L)
        self.R.extend(R)
        self.block_count += 1
    
    def write_wav(self, path: str, fs: int = 48000):
        """Export to 16-bit stereo WAV."""
        n_samples = len(self.L)
        
        with wave.open(path, 'w') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(fs)
            
            # Interleave L/R
            data = []
            for i in range(n_samples):
                data.append(struct.pack('<h', self.L[i]))
                data.append(struct.pack('<h', self.R[i]))
            
            wav.writeframes(b''.join(data))
        
        print(f"Wrote {path}: {n_samples} samples, {n_samples/fs:.2f}s")
    
    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get as numpy arrays (float -1..1)."""
        L = np.array(self.L, dtype=np.float32) / 32768.0
        R = np.array(self.R, dtype=np.float32) / 32768.0
        return L, R
    
    def clear(self):
        """Reset buffers."""
        self.L = []
        self.R = []
        self.block_count = 0


class Pipeline:
    """
    Complete synthesis pipeline.
    """
    
    def __init__(self, n_voices: int = 4, coeff_len: int = 256, 
                 block_size: int = 64, sample_rate: int = 48000):
        self.n_voices = n_voices
        self.coeff_len = coeff_len
        self.B = block_size
        self.fs = sample_rate
        
        # Engines
        self.symbol = SymbolEngine(n_voices, coeff_len)
        self.wavelet = WaveletEngine(block_size)
        self.material = MaterialEngine(block_size)
        self.safety = SafetyEngine(block_size)
        
        # Output
        self.sink = AudioSink()
        
        # Default parameters
        self.params = ParamFrame(
            K_mode=1,
            drive_q15=0x2000,
            tone_q15=0x4000,
            bias_q15=0,
            pressure_q15=0,
            cut_q15=0,
            morph_q15=0,
            gate_q15=Q15_ONE
        )
        
        # Current material program
        self.material_id = 0  # Clean/transparent default
        self.cat_id = 0
    
    def set_operator_program(self, op_ids: List[int]):
        """Set the operator program for all voices."""
        self.symbol.ops.set_program(op_ids)
    
    def inject_note(self, voice_id: int, frequency: float = 440.0, 
                    amplitude: int = 0x4000):
        """
        Inject a note into a voice.
        
        The wavelet synthesis treats each leaf's 8 coefficients as an amplitude
        envelope that modulates a carrier at the leaf's center frequency.
        
        To create a tone at a specific frequency, we need to:
        1. Find the leaf whose frequency band contains the target frequency
        2. Set that leaf's coefficients to encode the amplitude envelope
        """
        coeffs = [0] * self.coeff_len
        
        # Frequency parameters
        nyquist = self.fs / 2  # 24000 Hz at 48kHz sample rate
        
        # For wavelet packet trees with 32 leaves at depth 5:
        # - Leaf 0 covers [0, 1/32] of Nyquist = 0-750 Hz
        # - Leaf 1 covers [1/32, 2/32] = 750-1500 Hz
        # - etc.
        
        band_width = nyquist / 32  # ~750 Hz per leaf
        
        # Generate harmonics
        for harmonic in range(1, 16):
            freq = frequency * harmonic
            if freq >= nyquist * 0.95:  # Stay below Nyquist
                break
            
            # Find which leaf contains this frequency
            leaf_idx = int(freq / band_width)
            leaf_idx = max(0, min(31, leaf_idx))
            
            # Harmonic amplitude (natural 1/n rolloff)
            harm_amp = amplitude // harmonic
            
            # For a sustained tone, we want constant amplitude envelope
            # The 8 coefficients represent the envelope over the block
            # For a steady tone, make them all the same (with slight variation for richness)
            node_start = leaf_idx * 8
            
            for i in range(8):
                # Create slight amplitude modulation for naturalness
                # but keep it mostly constant for sustained tone
                mod = 1.0 + 0.1 * np.sin(2 * np.pi * i / 8 + harmonic * 0.5)
                coeff_val = int(harm_amp * mod)
                
                # Add to existing (allows multiple harmonics in same band)
                coeffs[node_start + i] = sat16(coeffs[node_start + i] + coeff_val)
        
        self.symbol.inject_note(voice_id, coeffs)
    
    def step(self) -> Tuple[List[int], List[int]]:
        """Process one block through the full pipeline."""
        # 1) Symbol Engine: update coefficients, emit VoiceFrames
        vframes = self.symbol.step_block(self.params)
        
        # 2) Wavelet Engine: render each voice to audio
        voice_blocks = []
        for vf in vframes:
            audio = self.wavelet.render_voice(vf)
            voice_blocks.append((vf, audio))
        
        # 3) Mix voices to stereo
        L, R = self.wavelet.mix_block(voice_blocks)
        
        # 4) Material Engine: apply saturation/catastrophe/etc
        Lm, Rm = self.material.step_block(L, R, self.params, 
                                          self.material_id, self.cat_id)
        
        # 5) Safety Engine: limit and protect
        Ls, Rs = self.safety.step_block(Lm, Rm, self.params)
        
        # 6) Store output
        self.sink.push_block(Ls, Rs)
        
        return Ls, Rs
    
    def run(self, duration_sec: float = 1.0):
        """Run pipeline for specified duration."""
        n_blocks = int(duration_sec * self.fs / self.B)
        
        for _ in range(n_blocks):
            self.step()
        
        return self.sink
    
    def export_wav(self, path: str):
        """Export accumulated audio to WAV."""
        self.sink.write_wav(path, self.fs)


# =============================================================================
# PRESET PROGRAMS
# =============================================================================

PROGRAMS = {
    "identity": [0],
    "tilt_warm": [1, 3, 11],  # Tilt -> IntraMix -> Normalize
    "evolving": [1, 3, 7, 4, 11],  # Tilt -> Mix -> LowRank -> DiffuseUp -> Norm
    "chaotic": [1, 3, 6, 10, 7, 11],  # Add shear + rotate
    "catastrophe": [15, 3, 20, 7, 11],  # Pressure -> Mix -> FoldPressure -> LowRank
    "sturmian": [16, 3, 17, 11],  # Gate -> Mix -> QuasiRotate -> Norm
}


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_basic():
    """Basic demo: generate a sustained test tone."""
    print("=" * 60)
    print("BASIC PIPELINE DEMO")
    print("=" * 60)
    
    pipe = Pipeline(n_voices=2, block_size=64)
    
    # Set operator program
    pipe.set_operator_program(PROGRAMS["tilt_warm"])
    
    # Set parameters
    pipe.params.drive_q15 = 0x3000
    pipe.params.tone_q15 = 0x5000
    
    # Run for 2 seconds with periodic note re-injection
    n_blocks = int(2.0 * pipe.fs / pipe.B)
    inject_interval = 16  # Re-inject every 16 blocks (~21ms)
    
    for i in range(n_blocks):
        # Periodically re-inject to sustain the sound
        if i % inject_interval == 0:
            pipe.inject_note(0, frequency=220.0, amplitude=0x3000)
            pipe.inject_note(1, frequency=330.0, amplitude=0x2800)
        
        pipe.step()
    
    # Export
    pipe.export_wav("/home/claude/synth_pipeline/demo_basic.wav")
    
    return pipe


def demo_evolving():
    """Demo with evolving parameters."""
    print("=" * 60)
    print("EVOLVING PARAMETERS DEMO")
    print("=" * 60)
    
    pipe = Pipeline(n_voices=3, block_size=64)
    pipe.set_operator_program(PROGRAMS["evolving"])
    
    # Inject notes (chord)
    pipe.inject_note(0, frequency=130.0, amplitude=0x3000)  # C3
    pipe.inject_note(1, frequency=165.0, amplitude=0x2800)  # E3
    pipe.inject_note(2, frequency=196.0, amplitude=0x2800)  # G3
    
    # Evolve parameters over time
    n_blocks = int(4.0 * pipe.fs / pipe.B)
    
    for i in range(n_blocks):
        # Ramp drive
        t = i / n_blocks
        pipe.params.drive_q15 = int(0x1000 + t * 0x4000)
        pipe.params.pressure_q15 = int(t * 0x3000)
        pipe.params.morph_q15 = int(t * Q15_ONE)
        
        pipe.step()
    
    pipe.export_wav("/home/claude/synth_pipeline/demo_evolving.wav")
    
    return pipe


def demo_catastrophe():
    """
    Demo catastrophe material processing (fold and cusp bifurcations).
    
    Shows the characteristic "snap" behavior as pressure crosses the
    bifurcation threshold - the signature of catastrophe theory.
    """
    print("=" * 60)
    print("CATASTROPHE DEMO")
    print("=" * 60)
    
    pipe = Pipeline(n_voices=1, block_size=64)
    pipe.set_operator_program(PROGRAMS["tilt_warm"])
    
    # Use low frequency for clearer catastrophe visualization
    base_freq = 80.0
    inject_interval = 12
    
    # Part 1: Fold catastrophe (A2) with pressure ramp
    print("  Part 1: Fold catastrophe - pressure ramp")
    pipe.material_id = 4  # Fold
    
    n_blocks = int(3.0 * pipe.fs / pipe.B)
    
    for i in range(n_blocks):
        # Slowly ramp pressure from 0 to max
        t = i / n_blocks
        pipe.params.pressure_q15 = int(t * 0x7000)
        pipe.params.drive_q15 = 0x4000
        
        if i % inject_interval == 0:
            pipe.inject_note(0, frequency=base_freq, amplitude=0x5000)
        
        pipe.step()
    
    # Part 2: Cusp catastrophe (A3) with pressure + bias sweep
    print("  Part 2: Cusp catastrophe - hysteresis loop")
    pipe.material_id = 5  # Cusp
    
    # Reset material state
    pipe.material.cat_L.s = 0
    pipe.material.cat_R.s = 0
    
    n_blocks = int(4.0 * pipe.fs / pipe.B)
    
    for i in range(n_blocks):
        t = i / n_blocks
        
        # Sweep pressure up then down (hysteresis)
        if t < 0.5:
            pressure = int(t * 2 * 0x6000)
        else:
            pressure = int((1 - (t - 0.5) * 2) * 0x6000)
        
        # Bias oscillates slowly
        bias = int(0x2000 * np.sin(2 * np.pi * t * 2))
        
        pipe.params.pressure_q15 = pressure
        pipe.params.bias_q15 = bias
        pipe.params.drive_q15 = 0x3000
        
        if i % inject_interval == 0:
            pipe.inject_note(0, frequency=base_freq * 1.5, amplitude=0x4000)
        
        pipe.step()
    
    # Part 3: Rapid pressure oscillation (fold snapping)
    print("  Part 3: Fold snap oscillation")
    pipe.material_id = 4
    pipe.material.cat_L.s = 0
    pipe.material.cat_R.s = 0
    
    n_blocks = int(3.0 * pipe.fs / pipe.B)
    
    for i in range(n_blocks):
        t = i / n_blocks
        
        # Pressure oscillates around the bifurcation point
        # This creates repeated "snaps"
        pressure = int(0x4000 + 0x3000 * np.sin(2 * np.pi * t * 4))
        
        pipe.params.pressure_q15 = pressure
        pipe.params.drive_q15 = 0x5000
        
        if i % inject_interval == 0:
            pipe.inject_note(0, frequency=base_freq, amplitude=0x4000)
        
        pipe.step()
    
    pipe.export_wav("/home/claude/synth_pipeline/demo_catastrophe.wav")
    print(f"  Total duration: 10 seconds")
    
    return pipe


def demo_all_materials():
    """Demo cycling through all material types."""
    print("=" * 60)
    print("ALL MATERIALS DEMO")
    print("=" * 60)
    
    pipe = Pipeline(n_voices=2, block_size=64)
    pipe.set_operator_program(PROGRAMS["tilt_warm"])
    
    materials = [
        (0, "clean"),
        (1, "tape"),
        (2, "rank"),
        (4, "fold"),
        (5, "cusp"),
    ]
    
    segment_sec = 1.0
    inject_interval = 16
    
    for mat_id, mat_name in materials:
        print(f"  Material: {mat_name}")
        
        pipe.material_id = mat_id
        pipe.params.drive_q15 = 0x3000
        pipe.params.pressure_q15 = 0x2000
        
        n_blocks = int(segment_sec * pipe.fs / pipe.B)
        
        for i in range(n_blocks):
            if i % inject_interval == 0:
                pipe.inject_note(0, frequency=150.0, amplitude=0x3000)
                pipe.inject_note(1, frequency=225.0, amplitude=0x2800)
            pipe.step()
    
    pipe.export_wav("/home/claude/synth_pipeline/demo_materials.wav")
    
    return pipe


def demo_tree_types():
    """Demo all 5 wavelet packet tree types."""
    print("=" * 60)
    print("TREE TYPES DEMO")
    print("=" * 60)
    
    from wavelet_packet_trees import PacketTreeSet
    tree_set = PacketTreeSet()
    
    pipe = Pipeline(n_voices=2, block_size=64)
    pipe.set_operator_program(PROGRAMS["evolving"])
    
    segment_sec = 1.5
    inject_interval = 16
    
    for tree_id in range(5):
        tree_name = tree_set.trees[tree_id].name
        print(f"  Tree {tree_id}: {tree_name}")
        
        # Set tree for voices
        pipe.symbol.voice_trees[0] = tree_id
        pipe.symbol.voice_trees[1] = tree_id
        
        pipe.params.drive_q15 = 0x2800
        pipe.params.tone_q15 = 0x5000
        
        n_blocks = int(segment_sec * pipe.fs / pipe.B)
        
        for i in range(n_blocks):
            if i % inject_interval == 0:
                pipe.inject_note(0, frequency=110.0, amplitude=0x3000)
                pipe.inject_note(1, frequency=165.0, amplitude=0x2800)
            pipe.step()
    
    pipe.export_wav("/home/claude/synth_pipeline/demo_trees.wav")
    
    return pipe


def demo_tree_transition():
    """Demo smooth tree transitions."""
    print("=" * 60)
    print("TREE TRANSITION DEMO")
    print("=" * 60)
    
    pipe = Pipeline(n_voices=2, block_size=64)
    pipe.set_operator_program(PROGRAMS["tilt_warm"])
    
    # Start with Tree 0 (Balanced)
    pipe.symbol.voice_trees[0] = 0
    pipe.symbol.voice_trees[1] = 0
    
    n_blocks = int(4.0 * pipe.fs / pipe.B)
    transition_block = n_blocks // 4
    inject_interval = 16
    
    for i in range(n_blocks):
        # Trigger tree transitions at intervals
        if i == transition_block:
            print("  Transition: Balanced -> Harmonic")
            pipe.symbol.voice_trees[0] = 1
            pipe.symbol.voice_trees[1] = 1
        
        elif i == 2 * transition_block:
            print("  Transition: Harmonic -> Texture")
            pipe.symbol.voice_trees[0] = 2
            pipe.symbol.voice_trees[1] = 2
        
        elif i == 3 * transition_block:
            print("  Transition: Texture -> Drone")
            pipe.symbol.voice_trees[0] = 4
            pipe.symbol.voice_trees[1] = 4
        
        # Sustained note injection
        if i % inject_interval == 0:
            pipe.inject_note(0, frequency=130.0, amplitude=0x3000)
            pipe.inject_note(1, frequency=196.0, amplitude=0x2800)
        
        pipe.step()
    
    pipe.export_wav("/home/claude/synth_pipeline/demo_transitions.wav")
    
    return pipe


if __name__ == "__main__":
    # Run all demos
    demo_basic()
    demo_evolving()
    demo_catastrophe()
    demo_all_materials()
    demo_tree_types()
    demo_tree_transition()
    
    print("\n" + "=" * 60)
    print("All demos complete. WAV files in /home/claude/synth_pipeline/")
    print("=" * 60)
