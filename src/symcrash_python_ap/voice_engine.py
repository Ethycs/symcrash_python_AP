"""
Voice Engine
============

Owns all musical time and note lifecycle. Emits VoiceFrames per block.

The Voice Engine is the ONLY layer that knows about:
- Note lifecycle (on/off, legato, retrigger)
- Musical timing (tempo, phase, "every N blocks")
- ADSR envelopes (amplitude, tree, brightness)
- Tree trajectory over note lifecycle

Downstream engines (Symbol, Wavelet, Safety) receive VoiceFrames
and treat them as stateless inputs - no musical time awareness.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from fixed import (
    sat16, mul_q15_sat, slew_q15,
    Q15_ONE, Q15_HALF, Q15_SHIFT
)


# =============================================================================
# VOICE LIFECYCLE STATES
# =============================================================================

class VoicePhase(IntEnum):
    """Voice lifecycle phases."""
    FREE = 0        # Not allocated
    ATTACK = 1      # Note-on, rising envelope
    DECAY = 2       # Past attack peak, falling to sustain
    SUSTAIN = 3     # Holding at sustain level
    RELEASE = 4     # Note-off, falling to zero


# =============================================================================
# ADSR ENVELOPE
# =============================================================================

@dataclass
class ADSRConfig:
    """ADSR envelope configuration."""
    attack_time: float = 0.01    # Seconds
    decay_time: float = 0.1      # Seconds
    sustain_level: float = 0.7   # 0-1 (relative to peak)
    release_time: float = 0.3    # Seconds

    # Curve shapes: -1 = log (fast start), 0 = linear, +1 = exp (slow start)
    attack_curve: float = 0.0
    decay_curve: float = 0.0
    release_curve: float = 0.0

    # Peak level (usually 1.0)
    peak_level: float = 1.0


@dataclass
class ADSRState:
    """ADSR envelope runtime state."""
    phase: VoicePhase = VoicePhase.FREE
    phase_time: float = 0.0      # Time within current phase
    current_level: float = 0.0   # Current envelope value (0-1)
    release_start_level: float = 0.0  # Level when release started


class ADSR:
    """
    Amplitude ADSR envelope generator.

    Call step() once per block to advance the envelope.
    """

    def __init__(self, config: ADSRConfig, block_size: int = 64, sample_rate: float = 48000.0):
        self.config = config
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.block_time = block_size / sample_rate

        self.state = ADSRState()

    def note_on(self, velocity: float = 1.0):
        """Trigger note-on. Starts attack phase."""
        self.state.phase = VoicePhase.ATTACK
        self.state.phase_time = 0.0
        # Don't reset current_level - allows retriggering without click
        self.config.peak_level = velocity

    def note_off(self):
        """Trigger note-off. Starts release phase."""
        if self.state.phase != VoicePhase.FREE:
            self.state.release_start_level = self.state.current_level
            self.state.phase = VoicePhase.RELEASE
            self.state.phase_time = 0.0

    def is_active(self) -> bool:
        """Check if voice is still sounding."""
        return self.state.phase != VoicePhase.FREE

    def step(self) -> float:
        """
        Advance envelope by one block. Returns current level (0-1).
        """
        cfg = self.config
        state = self.state

        if state.phase == VoicePhase.FREE:
            state.current_level = 0.0
            return 0.0

        # Advance time
        state.phase_time += self.block_time

        if state.phase == VoicePhase.ATTACK:
            if cfg.attack_time <= 0:
                # Instant attack
                state.current_level = cfg.peak_level
                state.phase = VoicePhase.DECAY
                state.phase_time = 0.0
            else:
                progress = min(1.0, state.phase_time / cfg.attack_time)
                progress = self._apply_curve(progress, cfg.attack_curve)
                state.current_level = progress * cfg.peak_level

                if state.phase_time >= cfg.attack_time:
                    state.phase = VoicePhase.DECAY
                    state.phase_time = 0.0

        elif state.phase == VoicePhase.DECAY:
            if cfg.decay_time <= 0:
                # Instant decay
                state.current_level = cfg.sustain_level * cfg.peak_level
                state.phase = VoicePhase.SUSTAIN
                state.phase_time = 0.0
            else:
                progress = min(1.0, state.phase_time / cfg.decay_time)
                progress = self._apply_curve(progress, cfg.decay_curve)
                sustain_target = cfg.sustain_level * cfg.peak_level
                state.current_level = cfg.peak_level * (1 - progress) + sustain_target * progress

                if state.phase_time >= cfg.decay_time:
                    state.phase = VoicePhase.SUSTAIN
                    state.phase_time = 0.0

        elif state.phase == VoicePhase.SUSTAIN:
            state.current_level = cfg.sustain_level * cfg.peak_level

        elif state.phase == VoicePhase.RELEASE:
            if cfg.release_time <= 0:
                # Instant release
                state.current_level = 0.0
                state.phase = VoicePhase.FREE
            else:
                progress = min(1.0, state.phase_time / cfg.release_time)
                progress = self._apply_curve(progress, cfg.release_curve)
                state.current_level = state.release_start_level * (1 - progress)

                if state.phase_time >= cfg.release_time:
                    state.current_level = 0.0
                    state.phase = VoicePhase.FREE

        return state.current_level

    def _apply_curve(self, t: float, curve: float) -> float:
        """Apply curve shaping to linear 0-1 progress."""
        if curve == 0:
            return t
        elif curve > 0:
            # Exponential (slow start, fast end)
            return t ** (1 + curve)
        else:
            # Logarithmic (fast start, slow end)
            return 1 - (1 - t) ** (1 - curve)

    def get_phase(self) -> VoicePhase:
        """Get current phase."""
        return self.state.phase


# =============================================================================
# TREE TRAJECTORY (Variant B: two trees + blend)
# =============================================================================

@dataclass
class TreeTrajectoryConfig:
    """
    Tree trajectory configuration for note lifecycle.

    Uses Variant B: two trees + blend for cleanest separation.
    Wavelet engine receives (tree_a, tree_b, blend) and handles interpolation.
    """
    # Tree betas for each phase
    attack_beta: float = 0.25    # Bright attack (treble-heavy)
    sustain_beta: float = 0.75   # Warm sustain (bass-heavy)
    release_beta: float = 0.5    # Neutral release

    # Transition timing (follows ADSR phases)
    # Attack: 0 -> attack_beta
    # Decay: attack_beta -> sustain_beta
    # Sustain: hold sustain_beta
    # Release: sustain_beta -> release_beta


@dataclass
class TreeTrajectoryState:
    """Tree trajectory runtime state."""
    tree_a: float = 0.5      # Current "from" tree
    tree_b: float = 0.5      # Current "to" tree
    blend: float = 0.0       # 0 = pure A, 1 = pure B
    phase: VoicePhase = VoicePhase.FREE


class TreeTrajectory:
    """
    Manages tree evolution over note lifecycle.

    Outputs (tree_a, tree_b, blend) for Variant B contract.
    Wavelet engine handles the actual coefficient blending.
    """

    def __init__(self, config: TreeTrajectoryConfig, block_size: int = 64, sample_rate: float = 48000.0):
        self.config = config
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.block_time = block_size / sample_rate

        self.state = TreeTrajectoryState()

        # Transition parameters (sync with ADSR timing)
        self.transition_time = 0.0
        self.transition_duration = 0.1

    def sync_with_adsr(self, adsr: ADSR):
        """Sync tree trajectory with ADSR phase changes."""
        new_phase = adsr.get_phase()
        old_phase = self.state.phase

        if new_phase != old_phase:
            self._on_phase_change(old_phase, new_phase, adsr)
            self.state.phase = new_phase

    def _on_phase_change(self, old_phase: VoicePhase, new_phase: VoicePhase, adsr: ADSR):
        """Handle phase transition - set up tree blend."""
        cfg = self.config

        if new_phase == VoicePhase.ATTACK:
            # Start: blend from neutral to attack tree
            self.state.tree_a = 0.5  # Start neutral
            self.state.tree_b = cfg.attack_beta
            self.state.blend = 0.0
            self.transition_duration = adsr.config.attack_time
            self.transition_time = 0.0

        elif new_phase == VoicePhase.DECAY:
            # Attack done: blend from attack to sustain tree
            self.state.tree_a = cfg.attack_beta
            self.state.tree_b = cfg.sustain_beta
            self.state.blend = 0.0
            self.transition_duration = adsr.config.decay_time
            self.transition_time = 0.0

        elif new_phase == VoicePhase.SUSTAIN:
            # Decay done: hold sustain tree
            self.state.tree_a = cfg.sustain_beta
            self.state.tree_b = cfg.sustain_beta
            self.state.blend = 0.0

        elif new_phase == VoicePhase.RELEASE:
            # Note off: blend from sustain to release tree
            self.state.tree_a = cfg.sustain_beta
            self.state.tree_b = cfg.release_beta
            self.state.blend = 0.0
            self.transition_duration = adsr.config.release_time
            self.transition_time = 0.0

        elif new_phase == VoicePhase.FREE:
            # Voice done
            self.state.tree_a = 0.5
            self.state.tree_b = 0.5
            self.state.blend = 0.0

    def step(self) -> Tuple[float, float, float]:
        """
        Advance tree trajectory by one block.
        Returns (tree_a, tree_b, blend).
        """
        if self.state.phase == VoicePhase.FREE:
            return (0.5, 0.5, 0.0)

        # Advance blend toward 1.0 during transitions
        if self.transition_duration > 0 and self.state.blend < 1.0:
            self.transition_time += self.block_time
            self.state.blend = min(1.0, self.transition_time / self.transition_duration)

        return (self.state.tree_a, self.state.tree_b, self.state.blend)


# =============================================================================
# VOICE STATE
# =============================================================================

@dataclass
class VoiceConfig:
    """Per-voice configuration."""
    # Amplitude envelope
    amp_adsr: ADSRConfig = field(default_factory=ADSRConfig)

    # Tree trajectory
    tree_traj: TreeTrajectoryConfig = field(default_factory=TreeTrajectoryConfig)

    # Optional: brightness envelope (same structure as amp)
    brightness_adsr: Optional[ADSRConfig] = None

    # Optional: density envelope
    density_adsr: Optional[ADSRConfig] = None


@dataclass
class Voice:
    """Single voice with all envelope generators."""
    voice_id: int
    config: VoiceConfig

    # Envelope generators
    amp: ADSR = None
    tree: TreeTrajectory = None
    brightness: Optional[ADSR] = None
    density: Optional[ADSR] = None

    # Note parameters
    note: int = 60           # MIDI note number
    velocity: float = 1.0    # 0-1
    pan: float = 0.0         # -1 to +1

    # Coefficient source (set by external code)
    base_coeffs: List[int] = field(default_factory=list)

    def __post_init__(self):
        block_size = 64
        sample_rate = 48000.0

        self.amp = ADSR(self.config.amp_adsr, block_size, sample_rate)
        self.tree = TreeTrajectory(self.config.tree_traj, block_size, sample_rate)

        if self.config.brightness_adsr:
            self.brightness = ADSR(self.config.brightness_adsr, block_size, sample_rate)
        if self.config.density_adsr:
            self.density = ADSR(self.config.density_adsr, block_size, sample_rate)


# =============================================================================
# VOICE FRAME OUTPUT (Updated contract)
# =============================================================================

@dataclass
class VoiceFrameV2:
    """
    Per-voice control record for downstream engines.

    This is what Voice Engine emits per block per voice.
    Symbol/Wavelet engines treat this as stateless input.
    """
    voice_id: int = 0

    # Amplitude (from ADSR)
    gain_q15: int = 0        # Q1.15 amplitude envelope output

    # Tree control - Variant B (cleanest separation)
    tree_a: float = 0.5      # First tree beta
    tree_b: float = 0.5      # Second tree beta
    tree_blend: float = 0.0  # 0 = pure A, 1 = pure B

    # Optional musical envelopes
    tilt_env_q15: int = Q15_HALF    # Brightness/tilt
    density_env_q15: int = Q15_HALF # Coefficient density

    # Panning
    pan_q15: int = 0         # -32768 to +32767

    # Material selection
    material_id: int = 0

    # Base coefficients (from symbol engine or preset)
    coeff_q15: List[int] = field(default_factory=list)

    # Voice state (for Safety Engine de-click)
    is_starting: bool = False
    is_stopping: bool = False
    is_active: bool = False


# =============================================================================
# VOICE ENGINE
# =============================================================================

class VoiceEngine:
    """
    Owns all musical time and emits VoiceFrames per block.

    This is the ONLY layer that knows about note lifecycle and timing.
    Downstream engines (Symbol, Wavelet, Safety) are stateless w.r.t. musical time.
    """

    def __init__(self, max_voices: int = 12, block_size: int = 64, sample_rate: float = 48000.0):
        self.max_voices = max_voices
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.block_time = block_size / sample_rate

        # Voice pool
        self.voices: Dict[int, Voice] = {}

        # Block counter (for timing)
        self.block_count: int = 0

        # Default voice config (can be overridden per voice)
        self.default_config = VoiceConfig()

    def allocate_voice(self, voice_id: Optional[int] = None,
                       config: Optional[VoiceConfig] = None) -> Optional[int]:
        """
        Allocate a voice slot. Returns voice_id or None if pool full.
        """
        if voice_id is None:
            # Find free slot
            for i in range(self.max_voices):
                if i not in self.voices or not self.voices[i].amp.is_active():
                    voice_id = i
                    break
            if voice_id is None:
                return None  # Pool exhausted

        cfg = config or self.default_config
        self.voices[voice_id] = Voice(voice_id=voice_id, config=cfg)

        return voice_id

    def note_on(self, voice_id: int, note: int = 60, velocity: float = 1.0,
                pan: float = 0.0, base_coeffs: Optional[List[int]] = None):
        """
        Trigger note-on for a voice.
        """
        if voice_id not in self.voices:
            self.allocate_voice(voice_id)

        voice = self.voices[voice_id]
        voice.note = note
        voice.velocity = velocity
        voice.pan = pan
        if base_coeffs:
            voice.base_coeffs = base_coeffs

        # Trigger all envelopes
        voice.amp.note_on(velocity)
        if voice.brightness:
            voice.brightness.note_on(velocity)
        if voice.density:
            voice.density.note_on(velocity)

    def note_off(self, voice_id: int):
        """
        Trigger note-off for a voice.
        """
        if voice_id not in self.voices:
            return

        voice = self.voices[voice_id]
        voice.amp.note_off()
        if voice.brightness:
            voice.brightness.note_off()
        if voice.density:
            voice.density.note_off()

    def is_voice_active(self, voice_id: int) -> bool:
        """Check if voice is still sounding."""
        if voice_id not in self.voices:
            return False
        return self.voices[voice_id].amp.is_active()

    def step_voice(self, voice_id: int) -> Optional[VoiceFrameV2]:
        """
        Step a single voice by one block. Returns VoiceFrameV2 or None if inactive.
        """
        if voice_id not in self.voices:
            return None

        voice = self.voices[voice_id]

        if not voice.amp.is_active():
            return None

        # Track phase for is_starting/is_stopping
        old_phase = voice.amp.get_phase()

        # Step amplitude envelope
        amp_level = voice.amp.step()
        gain_q15 = int(amp_level * Q15_ONE)
        gain_q15 = max(0, min(Q15_ONE, gain_q15))

        # Sync and step tree trajectory
        voice.tree.sync_with_adsr(voice.amp)
        tree_a, tree_b, tree_blend = voice.tree.step()

        # Step optional envelopes
        tilt_q15 = Q15_HALF
        density_q15 = Q15_HALF

        if voice.brightness:
            voice.brightness.sync_with_adsr(voice.amp) if hasattr(voice.brightness, 'sync_with_adsr') else None
            tilt_level = voice.brightness.step()
            tilt_q15 = int(tilt_level * Q15_ONE)

        if voice.density:
            density_level = voice.density.step()
            density_q15 = int(density_level * Q15_ONE)

        # Detect phase transitions for de-click signaling
        new_phase = voice.amp.get_phase()
        is_starting = (old_phase == VoicePhase.FREE and new_phase != VoicePhase.FREE)
        is_stopping = (new_phase == VoicePhase.FREE and old_phase != VoicePhase.FREE)

        # Pan to Q1.15
        pan_q15 = int(voice.pan * Q15_ONE)
        pan_q15 = max(-Q15_ONE, min(Q15_ONE - 1, pan_q15))

        return VoiceFrameV2(
            voice_id=voice_id,
            gain_q15=gain_q15,
            tree_a=tree_a,
            tree_b=tree_b,
            tree_blend=tree_blend,
            tilt_env_q15=tilt_q15,
            density_env_q15=density_q15,
            pan_q15=pan_q15,
            material_id=0,
            coeff_q15=list(voice.base_coeffs),
            is_starting=is_starting,
            is_stopping=is_stopping,
            is_active=voice.amp.is_active()
        )

    def step_all(self) -> List[VoiceFrameV2]:
        """
        Step all active voices by one block. Returns list of VoiceFrameV2.
        """
        self.block_count += 1
        frames = []

        for voice_id in list(self.voices.keys()):
            frame = self.step_voice(voice_id)
            if frame is not None:
                frames.append(frame)

        return frames

    def get_block_count(self) -> int:
        """Get current block count (for timing)."""
        return self.block_count

    def get_active_voice_count(self) -> int:
        """Get number of active voices."""
        return sum(1 for v in self.voices.values() if v.amp.is_active())


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def make_piano_config() -> VoiceConfig:
    """Piano-like voice configuration."""
    return VoiceConfig(
        amp_adsr=ADSRConfig(
            attack_time=0.005,
            decay_time=0.3,
            sustain_level=0.4,
            release_time=0.5,
            attack_curve=-0.5,  # Fast attack
            decay_curve=0.3,
        ),
        tree_traj=TreeTrajectoryConfig(
            attack_beta=0.2,    # Bright attack (lots of highs)
            sustain_beta=0.8,   # Warm sustain (more bass)
            release_beta=0.5,   # Neutral release
        ),
    )


def make_pad_config() -> VoiceConfig:
    """Pad/string-like voice configuration."""
    return VoiceConfig(
        amp_adsr=ADSRConfig(
            attack_time=0.3,
            decay_time=0.5,
            sustain_level=0.8,
            release_time=1.0,
            attack_curve=0.5,   # Slow attack
        ),
        tree_traj=TreeTrajectoryConfig(
            attack_beta=0.5,    # Neutral start
            sustain_beta=0.6,   # Slightly warm
            release_beta=0.7,   # Warmer release
        ),
    )


def make_pluck_config() -> VoiceConfig:
    """Pluck/percussion-like voice configuration."""
    return VoiceConfig(
        amp_adsr=ADSRConfig(
            attack_time=0.001,
            decay_time=0.2,
            sustain_level=0.0,  # No sustain
            release_time=0.1,
            attack_curve=-1.0,  # Instant attack
            decay_curve=0.5,    # Exponential decay
        ),
        tree_traj=TreeTrajectoryConfig(
            attack_beta=0.1,    # Very bright attack
            sustain_beta=0.9,   # Dark (but no sustain)
            release_beta=0.5,
        ),
    )


def make_organ_config() -> VoiceConfig:
    """Organ-like voice configuration (instant on/off)."""
    return VoiceConfig(
        amp_adsr=ADSRConfig(
            attack_time=0.002,
            decay_time=0.0,
            sustain_level=1.0,
            release_time=0.05,
        ),
        tree_traj=TreeTrajectoryConfig(
            attack_beta=0.5,
            sustain_beta=0.5,
            release_beta=0.5,
        ),
    )


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_voice_engine():
    """Demonstrate voice engine functionality."""
    print("=" * 60)
    print("VOICE ENGINE DEMO")
    print("=" * 60)

    # Create engine with piano config
    engine = VoiceEngine(max_voices=4)
    engine.default_config = make_piano_config()

    # Allocate and trigger a note
    voice_id = engine.allocate_voice()
    print(f"Allocated voice {voice_id}")

    # Note on
    engine.note_on(voice_id, note=60, velocity=0.8, pan=-0.3)
    print("Note ON: C4, velocity=0.8, pan=-0.3")

    # Simulate 100 blocks (~133ms at 64 samples @ 48kHz)
    print("\nSimulating note lifecycle:")
    print("-" * 40)

    for block in range(100):
        frames = engine.step_all()

        if frames:
            f = frames[0]
            phase = engine.voices[voice_id].amp.get_phase().name
            gain_db = 20 * np.log10(max(f.gain_q15 / Q15_ONE, 1e-6))

            # Print every 10 blocks
            if block % 10 == 0:
                print(f"Block {block:3d}: phase={phase:8s} gain={gain_db:+5.1f}dB "
                      f"tree=({f.tree_a:.2f}->{f.tree_b:.2f} @{f.tree_blend:.2f})")

        # Note off at block 50
        if block == 50:
            engine.note_off(voice_id)
            print("\n>>> Note OFF <<<\n")

    print("-" * 40)
    print(f"Final active voices: {engine.get_active_voice_count()}")

    # Demo with different configs
    print("\n" + "=" * 60)
    print("PRESET COMPARISON")
    print("=" * 60)

    presets = [
        ("Piano", make_piano_config()),
        ("Pad", make_pad_config()),
        ("Pluck", make_pluck_config()),
        ("Organ", make_organ_config()),
    ]

    for name, config in presets:
        engine = VoiceEngine()
        engine.default_config = config
        engine.allocate_voice(0)
        engine.note_on(0, note=60, velocity=1.0)

        # Step through attack
        attack_blocks = int(config.amp_adsr.attack_time * 48000 / 64) + 5
        for _ in range(attack_blocks):
            engine.step_all()

        frame = engine.step_voice(0)
        if frame:
            print(f"{name:8s}: attack_beta={config.tree_traj.attack_beta:.2f} "
                  f"sustain_beta={config.tree_traj.sustain_beta:.2f} "
                  f"attack_time={config.amp_adsr.attack_time*1000:.0f}ms")


if __name__ == "__main__":
    demo_voice_engine()
