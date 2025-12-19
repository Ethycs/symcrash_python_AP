"""
WTF Presets - 12 Identity-Defining Sounds
=========================================

Each preset demonstrates something other synths cannot do cleanly.
"""

import sys
import os

# Add src/symcrash_python_ap directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'symcrash_python_ap'))

import numpy as np
from typing import Tuple

from audio_io import write_audio

FS = 48000
BLOCK_SIZE = 64

def sat16(x): return max(-32768, min(32767, int(x)))


# =============================================================================
# PRESET 1: TOPOLOGY FOLD
# =============================================================================

def preset_topology_fold(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts as stable harmonic pad, reorganizes into metallic lattice.
    No filter sweep, no distortion - pure structural change.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 110.0  # A2
    n_partials = 16
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        sample_L = 0.0
        sample_R = 0.0
        
        for p in range(1, n_partials + 1):
            # Harmonic ratio morphs from integer to inharmonic
            if progress < 0.3:
                ratio = p  # Pure harmonic
            else:
                morph = (progress - 0.3) / 0.7
                # Morph toward metallic ratios (like bells)
                metallic = p * (1 + 0.02 * p * p * morph)
                ratio = p * (1 - morph) + metallic * morph
            
            freq = base_freq * ratio
            if freq > FS / 2:
                continue
            
            # Amplitude: harmonics fade, inharmonics grow
            if progress < 0.3:
                amp = 1.0 / p
            else:
                morph = (progress - 0.3) / 0.7
                harm_amp = 1.0 / p
                metal_amp = 0.8 / (1 + 0.5 * abs(p - 4))  # Peak around 4th
                amp = harm_amp * (1 - morph) + metal_amp * morph
            
            # Phase coherence breaks down
            phase_drift = 0 if progress < 0.2 else (progress - 0.2) * p * 0.5
            
            phase = 2 * np.pi * freq * t + phase_drift * np.sin(t * 0.3 * p)
            
            # Stereo spread increases with inharmonicity
            pan = 0.3 * np.sin(p * 0.7) * progress
            
            sample_L += amp * np.sin(phase) * (1 - pan)
            sample_R += amp * np.sin(phase) * (1 + pan)
        
        # Constant loudness envelope
        env = 0.4 if progress < 0.05 else (0.4 if progress > 0.95 else 0.5)
        env *= 6000
        
        L[i] = sample_L * env
        R[i] = sample_R * env
    
    return L, R


# =============================================================================
# PRESET 2: CATASTROPHE WITHOUT VOLUME
# =============================================================================

def preset_catastrophe_snap(duration: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sustained tone that suddenly snaps character.
    Loudness stays constant, spectral identity flips.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 220.0
    snap_time = 0.5  # Snap at 50%
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # State variable with hysteresis
        if progress < snap_time - 0.02:
            state = 0.0
        elif progress < snap_time + 0.02:
            # Rapid transition
            local_t = (progress - (snap_time - 0.02)) / 0.04
            state = 0.5 - 0.5 * np.cos(local_t * np.pi)
        else:
            state = 1.0
        
        sample = 0.0
        
        # State 0: warm, even harmonics emphasized
        # State 1: cold, odd harmonics, detuned
        for h in range(1, 12):
            if state < 0.5:
                # Warm state
                freq = base_freq * h
                amp = 1.0 / h if h % 2 == 0 else 0.5 / h
            else:
                # Cold state - odd harmonics, slight detune
                freq = base_freq * h * (1 + 0.003 * h)
                amp = 1.0 / h if h % 2 == 1 else 0.3 / h
            
            # Blend based on state
            warm_contrib = (1 - state) * (1.0 / h if h % 2 == 0 else 0.5 / h)
            cold_contrib = state * (1.0 / h if h % 2 == 1 else 0.3 / h)
            
            freq_warm = base_freq * h
            freq_cold = base_freq * h * (1 + 0.004 * h)
            
            sample += warm_contrib * np.sin(2 * np.pi * freq_warm * t)
            sample += cold_contrib * np.sin(2 * np.pi * freq_cold * t)
        
        # Constant envelope
        env = 5000
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 3: PERFECT MORPH
# =============================================================================

def preset_perfect_morph(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morphs between wildly different sounds.
    Zero zippering, no crossfade blur.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 130.81  # C3
    
    # Sound A: Organ-like (all harmonics, quick attack)
    # Sound B: Clarinet-like (odd harmonics, soft)
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # S-curve morph
        morph = 0.5 - 0.5 * np.cos(progress * np.pi)
        
        sample = 0.0
        
        for h in range(1, 16):
            freq = base_freq * h
            if freq > FS / 2:
                continue
            
            # Organ: all harmonics with 1/h rolloff
            organ_amp = 1.0 / h
            
            # Clarinet: odd harmonics only, specific rolloff
            if h % 2 == 1:
                clarinet_amp = 1.0 / (h ** 0.8)
            else:
                clarinet_amp = 0.05 / h  # Very quiet evens
            
            # Interpolate amplitudes
            amp = organ_amp * (1 - morph) + clarinet_amp * morph
            
            # Phase: organ has aligned phases, clarinet has slight drift
            organ_phase = 2 * np.pi * freq * t
            clarinet_phase = 2 * np.pi * freq * t + 0.1 * h * morph
            
            phase = organ_phase * (1 - morph) + clarinet_phase * morph
            
            sample += amp * np.sin(phase)
        
        env = 5500
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 4: LIVING CHORD
# =============================================================================

def preset_living_chord(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    10-20 voices locked in harmonic frame.
    Micro-drift inside, sounds like single organism.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    # C major 7 chord across octaves
    chord_ratios = [
        1.0,      # C
        1.25,     # E (5/4)
        1.5,      # G (3/2)
        1.875,    # B (15/8)
        2.0,      # C (octave)
        2.5,      # E
        3.0,      # G
        3.75,     # B
        4.0,      # C
        5.0,      # E
        6.0,      # G
        8.0,      # C
    ]
    
    base_freq = 65.41  # C2
    n_voices = len(chord_ratios)
    
    # Each voice has its own micro-drift parameters
    np.random.seed(123)
    drift_rates = np.random.random(n_voices) * 0.3 + 0.1
    drift_depths = np.random.random(n_voices) * 0.002 + 0.001
    phase_offsets = np.random.random(n_voices) * 2 * np.pi
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        sample_L = 0.0
        sample_R = 0.0
        
        for v, ratio in enumerate(chord_ratios):
            # Base frequency with micro-drift
            drift = drift_depths[v] * np.sin(drift_rates[v] * t * 2 * np.pi)
            freq = base_freq * ratio * (1 + drift)
            
            # Amplitude based on register
            amp = 1.0 / (1 + 0.15 * v)
            
            # Slow amplitude breathing (different rates)
            breath = 1 + 0.1 * np.sin(t * (0.2 + 0.05 * v))
            amp *= breath
            
            phase = 2 * np.pi * freq * t + phase_offsets[v]
            
            # Add subtle harmonics per voice
            voice_sample = np.sin(phase)
            voice_sample += 0.3 * np.sin(phase * 2)
            voice_sample += 0.1 * np.sin(phase * 3)
            
            # Stereo position (spread across field)
            pan = (v / n_voices - 0.5) * 1.5
            pan = max(-1, min(1, pan))
            
            sample_L += voice_sample * amp * np.sqrt(0.5 * (1 - pan))
            sample_R += voice_sample * amp * np.sqrt(0.5 * (1 + pan))
        
        # Gentle envelope
        if progress < 0.05:
            env = progress / 0.05
        elif progress > 0.9:
            env = (1 - progress) / 0.1
        else:
            env = 1.0
        
        env *= 3000
        L[i] = sample_L * env
        R[i] = sample_R * env
    
    return L, R


# =============================================================================
# PRESET 5: POLYPHONIC PRESSURE
# =============================================================================

def preset_polyphonic_pressure(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple notes, complex intermodulation.
    Removing one note collapses texture.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    # Cluster of close frequencies that beat against each other
    freqs = [220, 223, 227, 330, 333, 440, 445]
    
    phases = [0.0] * len(freqs)
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # Build up notes over time
        n_active = int(1 + progress * (len(freqs) - 1))
        
        sample = 0.0
        
        for f_idx in range(n_active):
            freq = freqs[f_idx]
            phases[f_idx] += 2 * np.pi * freq / FS
            
            # Each voice is simple
            voice = np.sin(phases[f_idx])
            voice += 0.3 * np.sin(phases[f_idx] * 2)
            
            sample += voice * 0.5
        
        # Ring modulation between voices creates complexity
        if n_active >= 2:
            rm = np.sin(phases[0]) * np.sin(phases[1]) * 0.3
            sample += rm
        
        if n_active >= 4:
            rm2 = np.sin(phases[2]) * np.sin(phases[3]) * 0.2
            sample += rm2
        
        env = 4000
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 6: INTENTIONAL ALIASING
# =============================================================================

def preset_intentional_aliasing(duration: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harsh digital edge that isn't brittle fizz.
    Carved, not broken.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 55.0  # A1
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # Generate harmonics that intentionally fold
        sample = 0.0
        
        for h in range(1, 40):
            freq = base_freq * h
            
            # Let high harmonics alias (fold back)
            while freq > FS / 2:
                freq = FS - freq
            
            if freq < 20:
                continue
            
            # Amplitude shaping - not just 1/h
            # Creates "carved" quality
            amp = 1.0 / (h ** 0.6)
            
            # Quantize phase slightly for digital character
            phase = 2 * np.pi * freq * t
            phase_quant = np.round(phase * 8) / 8  # 8-level phase quantization
            
            sample += amp * np.sin(phase * 0.7 + phase_quant * 0.3)
        
        # Bit reduction effect (subtle)
        sample = np.round(sample * 16) / 16
        
        env = 5000
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 7: VHS GHOST CHOIR
# =============================================================================

def preset_vhs_ghost_choir(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Noise-based voices. Pitch emerges from structure.
    Degradation that feels emotional.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    # Choir frequencies (voices emerge from noise)
    choir_freqs = [196, 247, 294, 392, 494]  # G major
    
    # Noise state
    np.random.seed(456)
    noise_state = np.random.random(5) * 2 - 1
    
    # Tape wow/flutter
    wow_phase = 0.0
    flutter_phase = 0.0
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # Tape artifacts
        wow_phase += 2 * np.pi * 0.5 / FS  # 0.5 Hz wow
        flutter_phase += 2 * np.pi * 7 / FS  # 7 Hz flutter
        
        pitch_mod = 1 + 0.008 * np.sin(wow_phase) + 0.002 * np.sin(flutter_phase)
        
        sample_L = 0.0
        sample_R = 0.0
        
        # Filtered noise creates pitched content
        for v, freq in enumerate(choir_freqs):
            # Voice emerges over time
            voice_onset = v * 0.15
            if progress < voice_onset:
                continue
            
            voice_amp = min(1.0, (progress - voice_onset) / 0.2)
            
            # Resonant filter on noise
            mod_freq = freq * pitch_mod
            
            # Generate filtered noise (simple resonant)
            noise = np.random.random() * 2 - 1
            
            # Comb filter effect for pitch
            delay_samples = int(FS / mod_freq)
            
            # Use stored noise for feedback
            if i >= delay_samples:
                feedback = 0.7
                pitched_noise = noise + feedback * (
                    L[i - delay_samples] / 5000 if i > delay_samples else 0
                )
            else:
                pitched_noise = noise
            
            # Add tonal component that fades in
            tonal = np.sin(2 * np.pi * mod_freq * t)
            blend = 0.3 + 0.4 * voice_amp
            
            voice_sample = pitched_noise * (1 - blend) + tonal * blend
            voice_sample *= voice_amp * 0.5
            
            # Stereo spread
            pan = (v - 2) / 3
            sample_L += voice_sample * (1 - pan * 0.5)
            sample_R += voice_sample * (1 + pan * 0.5)
        
        # Add tape hiss
        hiss = (np.random.random() - 0.5) * 0.1
        sample_L += hiss
        sample_R += hiss
        
        # Dropout simulation (rare)
        if np.random.random() < 0.0001:
            sample_L *= 0.3
            sample_R *= 0.3
        
        env = 5000
        L[i] = sample_L * env
        R[i] = sample_R * env
    
    return L, R


# =============================================================================
# PRESET 8: PRESSURE FIELD
# =============================================================================

def preset_pressure_field(duration: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts almost silent. Density increases, not volume.
    Eventually overwhelming without being loud.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 40.0  # Very low
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # Number of partials increases with time
        n_partials = int(2 + progress * 60)
        
        sample_L = 0.0
        sample_R = 0.0
        
        for p in range(n_partials):
            # Non-harmonic spacing creates density
            ratio = 1 + p * 0.73  # Inharmonic
            freq = base_freq * ratio
            
            if freq > FS / 2:
                continue
            
            # Amplitude stays roughly constant per partial
            amp = 0.3 / (1 + 0.1 * p)
            
            # Slight detuning increases with density
            detune = 1 + 0.001 * p * progress
            
            phase = 2 * np.pi * freq * detune * t + p * 0.5
            
            # Random pan per partial
            pan = np.sin(p * 1.7) * 0.8
            
            partial = amp * np.sin(phase)
            sample_L += partial * (1 - pan)
            sample_R += partial * (1 + pan)
        
        # Normalize to keep volume constant as density increases
        normalizer = 1.0 / np.sqrt(n_partials / 10)
        
        env = 4000 * normalizer
        L[i] = sample_L * env
        R[i] = sample_R * env
    
    return L, R


# =============================================================================
# PRESET 9: IMMINENT FAILURE
# =============================================================================

def preset_imminent_failure(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Always feels about to break. Never does.
    Tension without collapse.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 110.0
    
    # "Danger" oscillator
    danger_phase = 0.0
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # Danger level oscillates but never exceeds threshold
        danger_phase += 2 * np.pi * 0.3 / FS
        danger = 0.5 + 0.45 * np.sin(danger_phase)  # 0.05 to 0.95
        
        sample = 0.0
        
        # Base tone
        for h in range(1, 8):
            freq = base_freq * h
            amp = 1.0 / h
            sample += amp * np.sin(2 * np.pi * freq * t)
        
        # Distortion increases with danger (but never clips)
        sample = np.tanh(sample * (1 + danger * 3)) / (1 + danger)
        
        # Add "stress" harmonics
        stress_freq = base_freq * (7 + danger * 0.5)  # Dissonant
        stress = 0.2 * danger * np.sin(2 * np.pi * stress_freq * t)
        sample += stress
        
        # Jitter/instability
        jitter = 1 + 0.02 * danger * np.sin(t * 500 * danger)
        sample *= jitter
        
        # Occasional "near miss" transients
        if np.random.random() < 0.001 * danger:
            sample *= 1.5
        
        # Safety limiting (always catches it)
        sample = np.tanh(sample * 0.8) * 0.9
        
        env = 6000
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 10: SAW THAT LEARNED SOMETHING
# =============================================================================

def preset_saw_that_learned(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts as normal saw pad.
    Slowly violates harmonic expectations.
    Still playable.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 130.81  # C3
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        sample = 0.0
        
        # Classic saw: all harmonics at 1/h
        for h in range(1, 24):
            # Harmonic ratio "learns" to drift
            if progress < 0.3:
                ratio = h
            else:
                learn = (progress - 0.3) / 0.7
                # Some harmonics drift sharp, some flat
                drift = 0.02 * learn * np.sin(h * 1.3)
                ratio = h * (1 + drift)
            
            freq = base_freq * ratio
            if freq > FS / 2:
                continue
            
            # Amplitude envelope per harmonic changes
            base_amp = 1.0 / h
            
            # Higher harmonics "learn" different levels
            if progress > 0.4:
                learn = (progress - 0.4) / 0.6
                if h % 3 == 0:
                    base_amp *= (1 + learn * 0.5)  # Enhance every 3rd
                elif h % 5 == 0:
                    base_amp *= (1 - learn * 0.3)  # Suppress every 5th
            
            phase = 2 * np.pi * freq * t
            sample += base_amp * np.sin(phase)
        
        env = 4500
        L[i] = sample * env
        R[i] = sample * env
    
    return L, R


# =============================================================================
# PRESET 11: BASS WITH MEMORY
# =============================================================================

def preset_bass_with_memory(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bass line where previous notes affect the next.
    Groove has memory.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    # Bass line pattern (frequencies)
    pattern = [55, 55, 73.4, 55, 82.4, 55, 73.4, 82.4]  # A1, D2, E2
    note_dur = duration / len(pattern)
    samples_per_note = int(note_dur * FS)
    
    # Memory state (accumulates character)
    memory_brightness = 0.5
    memory_attack = 0.5
    memory_resonance = 0.0
    
    phase = 0.0
    
    for i in range(n_samples):
        t = i / FS
        note_idx = min(int(t / note_dur), len(pattern) - 1)
        note_t = (t % note_dur) / note_dur
        
        freq = pattern[note_idx]
        
        # Update memory based on note
        if note_idx > 0 and i % samples_per_note == 0:
            prev_freq = pattern[note_idx - 1]
            if freq > prev_freq:
                memory_brightness += 0.1
                memory_attack += 0.05
            else:
                memory_brightness -= 0.05
                memory_attack -= 0.02
            
            memory_resonance += 0.1 * (freq / 55 - 1)
            
            # Clamp
            memory_brightness = max(0.2, min(1.0, memory_brightness))
            memory_attack = max(0.3, min(0.9, memory_attack))
            memory_resonance = max(0, min(0.5, memory_resonance))
        
        # Note envelope affected by memory
        if note_t < memory_attack * 0.1:
            env = note_t / (memory_attack * 0.1)
        else:
            env = 1 - (note_t - memory_attack * 0.1) * 0.5
        env = max(0, env)
        
        # Generate tone
        phase += 2 * np.pi * freq / FS
        
        sample = 0.0
        n_harmonics = int(4 + memory_brightness * 8)
        
        for h in range(1, n_harmonics + 1):
            amp = 1.0 / (h ** (1.5 - memory_brightness * 0.5))
            sample += amp * np.sin(phase * h)
        
        # Resonance adds bite
        sample += memory_resonance * np.sin(phase * 5) * 0.3
        
        sample *= env * 6000
        L[i] = sample
        R[i] = sample
    
    return L, R


# =============================================================================
# PRESET 12: SINGLE KNOB APOCALYPSE
# =============================================================================

def preset_single_knob_apocalypse(duration: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One parameter controls: structure, artifacts, chaos.
    Range from silence to annihilation.
    """
    n_samples = int(duration * FS)
    L = np.zeros(n_samples)
    R = np.zeros(n_samples)
    
    base_freq = 55.0
    
    for i in range(n_samples):
        t = i / FS
        progress = i / n_samples
        
        # The "knob" (0 to 1)
        # Goes up and slightly back down
        if progress < 0.8:
            knob = progress / 0.8
        else:
            knob = 1.0 - (progress - 0.8) / 0.2 * 0.3
        
        # Number of voices scales with knob
        n_voices = int(1 + knob * 30)
        
        # Detuning increases with knob
        detune_amount = knob * 0.05
        
        # Distortion increases with knob
        drive = 1 + knob * 10
        
        # Noise injection increases with knob
        noise_level = knob * 0.3
        
        sample_L = 0.0
        sample_R = 0.0
        
        for v in range(n_voices):
            # Each voice slightly detuned
            detune = 1 + (v - n_voices/2) * detune_amount / n_voices
            freq = base_freq * detune
            
            # Add harmonics
            voice = 0.0
            for h in range(1, 6):
                voice += np.sin(2 * np.pi * freq * h * t + v) / h
            
            # Per-voice distortion
            voice = np.tanh(voice * (1 + knob * 2))
            
            # Pan
            pan = (v / n_voices - 0.5) * 2 * knob
            
            sample_L += voice * (1 - pan) / n_voices
            sample_R += voice * (1 + pan) / n_voices
        
        # Sum distortion
        sample_L = np.tanh(sample_L * drive) / drive
        sample_R = np.tanh(sample_R * drive) / drive
        
        # Add noise
        noise = (np.random.random() - 0.5) * noise_level
        sample_L += noise
        sample_R += noise
        
        # Amplitude (constant perceived loudness)
        env = 6000
        L[i] = sample_L * env
        R[i] = sample_R * env
    
    return L, R


# =============================================================================
# MAIN
# =============================================================================

def generate_all_presets():
    """Generate all 12 presets."""

    presets = [
        ("01_topology_fold", preset_topology_fold, "Harmonic to Metallic lattice"),
        ("02_catastrophe_snap", preset_catastrophe_snap, "Character flip, constant volume"),
        ("03_perfect_morph", preset_perfect_morph, "Organ to Clarinet, no zipper"),
        ("04_living_chord", preset_living_chord, "20 voices as one organism"),
        ("05_polyphonic_pressure", preset_polyphonic_pressure, "Simple notes, complex interaction"),
        ("06_intentional_aliasing", preset_intentional_aliasing, "Digital edge, not brittle"),
        ("07_vhs_ghost_choir", preset_vhs_ghost_choir, "Noise that sings"),
        ("08_pressure_field", preset_pressure_field, "Density, not volume"),
        ("09_imminent_failure", preset_imminent_failure, "Always about to break"),
        ("10_saw_that_learned", preset_saw_that_learned, "Familiar to Alien"),
        ("11_bass_with_memory", preset_bass_with_memory, "Notes affect future notes"),
        ("12_single_knob_apocalypse", preset_single_knob_apocalypse, "One knob to everything"),
    ]

    print("=" * 60)
    print("GENERATING 12 WTF PRESETS")
    print("=" * 60)
    print()

    for filename, func, desc in presets:
        print(f"{filename}: {desc}")
        L, R = func()
        write_audio(filename, L, R, subdirectory="wtf_presets")

    print()
    print("=" * 60)
    print("ALL PRESETS GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_presets()
