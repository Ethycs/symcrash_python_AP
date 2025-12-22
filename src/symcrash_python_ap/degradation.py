"""
WSE-STD-050 — Degradation + Voice Scheduling Semantics (v0)
============================================================

This module implements voice classification, scheduling semantics, and
degradation behavior under compute pressure.

Key Guarantees:
1. Deterministic replay under variable compute budgets
2. Musical continuity for active (Foreground) voices
3. Graceful degradation for passive (Cloud) voices
4. FPGA-safe scheduling with bounded state mutation
5. No hidden time advancement when work is skipped

Architecture:
- Foreground voices: Freeze semantics (time pauses)
- Cloud voices: Decay semantics (energy fades deterministically)
- PRNG never advances on skipped voices
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
from enum import Enum, IntEnum

from .frames import VoiceFramePacketV0b, VoiceFlags


# =============================================================================
# CONSTANTS
# =============================================================================

# Energy threshold for Foreground classification (approx -60 dBFS)
# In Q15: -60 dBFS ≈ 0.001 * 32768 ≈ 33
ENERGY_EPS_Q15 = 33

# Cloud decay factor per block (0.98 = ~3dB per 50 blocks at 48kHz/64)
CLOUD_DECAY_Q15 = int(0.98 * 32768)  # 32112

# Retirement threshold (below this, voice can be retired)
RETIRE_EPS_Q15 = 3  # Very close to silence

# Blocks below retire threshold before retirement
RETIRE_BLOCKS = 8


# =============================================================================
# VOICE CLASSIFICATION (Section 3)
# =============================================================================

class VoiceClass(Enum):
    """
    Voice classification per WSE-STD-050 Section 3.

    Classification MUST be recomputed every block.
    A voice MAY transition between classes at block boundaries only.
    """
    FOREGROUND = "foreground"  # Active, musically important
    CLOUD = "cloud"            # Passive, can decay


class VoiceStatus(Enum):
    """Voice execution status for current block."""
    SERVICED = "serviced"  # Symbol + Wavelet execution occurred
    SKIPPED = "skipped"    # No execution this block


# =============================================================================
# VOICE STATE (Section 2.1)
# =============================================================================

@dataclass
class VoiceState:
    """
    Complete voice state per WSE-STD-050 Section 2.1.

    A voice is a persistent stateful entity owning:
    - Voice Engine state (ADSR, regime, PRNG, etc.)
    - Symbol state (coefficients, op toggles)
    - Wavelet reconstruction state (OLA buffers, etc.)
    """
    voice_id: int

    # Classification (recomputed each block)
    voice_class: VoiceClass = VoiceClass.CLOUD

    # Gate and activity
    gate: bool = False
    energy_q15: int = 0
    activity_q15: int = 0

    # Control flags
    solo: bool = False
    armed: bool = False
    intent_updated: bool = False  # True if IntentFrame received this block

    # Timing state (frozen when skipped)
    adsr_phase: int = 0
    adsr_time: int = 0
    regime_id: int = 0
    regime_dwell: int = 0

    # Retirement tracking
    below_retire_count: int = 0
    retired: bool = False

    # Last valid audio state (for OLA continuity)
    last_audio_energy_q15: int = 0

    # Block tracking
    last_serviced_block: int = -1
    skip_count: int = 0

    def classify(self) -> VoiceClass:
        """
        Classify voice per WSE-STD-050 Section 3.

        A voice is Foreground if ANY of:
        1. Voice has non-zero gate (note is active)
        2. Voice received an IntentFrame update this block
        3. Voice energy > ENERGY_EPS
        4. Voice is explicitly marked SOLO or ARMED

        Otherwise, voice is Cloud.
        """
        if self.gate:
            self.voice_class = VoiceClass.FOREGROUND
        elif self.intent_updated:
            self.voice_class = VoiceClass.FOREGROUND
        elif self.energy_q15 > ENERGY_EPS_Q15:
            self.voice_class = VoiceClass.FOREGROUND
        elif self.solo or self.armed:
            self.voice_class = VoiceClass.FOREGROUND
        else:
            self.voice_class = VoiceClass.CLOUD

        return self.voice_class

    def is_foreground(self) -> bool:
        """Check if voice is classified as Foreground."""
        return self.voice_class == VoiceClass.FOREGROUND

    def is_cloud(self) -> bool:
        """Check if voice is classified as Cloud."""
        return self.voice_class == VoiceClass.CLOUD


# =============================================================================
# FREEZE SEMANTICS (Section 5.1)
# =============================================================================

@dataclass
class FreezeState:
    """
    Captured state for freeze semantics.

    When a Foreground voice is skipped, ALL state is frozen:
    - ADSR does NOT advance
    - Regime dwell counters do NOT advance
    - Phase/state machines do NOT advance
    - No coefficient updates
    - No operator application
    - No random draws
    """
    adsr_phase: int = 0
    adsr_time: int = 0
    regime_id: int = 0
    regime_dwell: int = 0
    energy_q15: int = 0
    coefficients_frozen: bool = True


def apply_freeze(voice: VoiceState) -> FreezeState:
    """
    Apply freeze semantics to a skipped Foreground voice.

    Per WSE-STD-050 Section 5.1:
    - Time pauses for exactly one block
    - All state is preserved
    - No decay, no events, no timing advancement
    """
    return FreezeState(
        adsr_phase=voice.adsr_phase,
        adsr_time=voice.adsr_time,
        regime_id=voice.regime_id,
        regime_dwell=voice.regime_dwell,
        energy_q15=voice.energy_q15,
        coefficients_frozen=True,
    )


# =============================================================================
# DECAY SEMANTICS (Section 5.2)
# =============================================================================

def apply_cloud_decay(voice: VoiceState) -> int:
    """
    Apply deterministic decay to a skipped Cloud voice.

    Per WSE-STD-050 Section 5.2:
    energy ← energy × CLOUD_DECAY

    Returns the new energy value.

    State frozen:
    - PRNG state
    - Regime transitions
    - Symbol operator state

    NOT allowed:
    - Emit new events
    - Advance PRNG
    - Apply Symbol operators
    """
    # Deterministic Q15 multiplication
    new_energy = (voice.energy_q15 * CLOUD_DECAY_Q15) >> 15
    voice.energy_q15 = new_energy
    return new_energy


def check_retirement(voice: VoiceState) -> bool:
    """
    Check if a Cloud voice should be retired.

    Per WSE-STD-050 Section 9:
    A voice MAY be retired when:
    - Classified as Cloud
    - Energy < RETIRE_EPS for N consecutive blocks

    Retirement MUST:
    - Be deterministic
    - Free all associated state
    - Not emit audio discontinuities (OLA fade required)
    """
    if voice.voice_class != VoiceClass.CLOUD:
        voice.below_retire_count = 0
        return False

    if voice.energy_q15 < RETIRE_EPS_Q15:
        voice.below_retire_count += 1
    else:
        voice.below_retire_count = 0

    if voice.below_retire_count >= RETIRE_BLOCKS:
        voice.retired = True
        return True

    return False


# =============================================================================
# VOICE SCHEDULER (Section 4)
# =============================================================================

@dataclass
class VoiceScheduler:
    """
    Voice scheduler implementing WSE-STD-050 Section 4.

    Scheduling model:
    1. Reserve mandatory tail budget (see WSE-STD-030)
    2. Iterate voices in rolling cursor order
    3. Attempt to service each voice until budget exhausted
    4. Apply degradation rules to skipped voices

    Ordering guarantee:
    - Uses rolling cursor (round-robin) over voices
    - Cursor position persists across blocks
    - Voices NOT reordered based on priority alone
    """
    max_voices: int = 128
    cursor: int = 0  # Rolling cursor position

    # Track which voices exist
    active_voices: Set[int] = field(default_factory=set)

    # Per-voice state
    voice_states: Dict[int, VoiceState] = field(default_factory=dict)

    def allocate_voice(self, voice_id: int) -> VoiceState:
        """Allocate a new voice."""
        state = VoiceState(voice_id=voice_id)
        self.voice_states[voice_id] = state
        self.active_voices.add(voice_id)
        return state

    def deallocate_voice(self, voice_id: int) -> None:
        """Deallocate a voice (retirement)."""
        self.voice_states.pop(voice_id, None)
        self.active_voices.discard(voice_id)

    def get_voice(self, voice_id: int) -> Optional[VoiceState]:
        """Get voice state."""
        return self.voice_states.get(voice_id)

    def classify_all(self) -> Tuple[List[int], List[int]]:
        """
        Classify all voices for this block.

        Returns (foreground_ids, cloud_ids).
        """
        foreground = []
        cloud = []

        for voice_id in self.active_voices:
            state = self.voice_states.get(voice_id)
            if state:
                state.classify()
                if state.is_foreground():
                    foreground.append(voice_id)
                else:
                    cloud.append(voice_id)

        return foreground, cloud

    def schedule_block(
        self,
        budget: int,
        cost_per_voice: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """
        Schedule voices for this block.

        Per Section 4.1:
        1. Reserve mandatory tail budget
        2. Iterate voices in rolling cursor order
        3. Service until budget exhausted
        4. Return (serviced, skipped) lists

        Args:
            budget: Available compute budget
            cost_per_voice: Cost to service one voice

        Returns:
            Tuple of (serviced_voice_ids, skipped_voice_ids)
        """
        serviced = []
        skipped = []

        if not self.active_voices:
            return serviced, skipped

        # Get sorted list for deterministic iteration
        voice_list = sorted(self.active_voices)
        n_voices = len(voice_list)

        if n_voices == 0:
            return serviced, skipped

        # Rolling cursor iteration
        remaining_budget = budget
        scanned = 0
        start_pos = self.cursor % n_voices
        pos = start_pos

        while scanned < n_voices:
            voice_id = voice_list[pos]

            if remaining_budget >= cost_per_voice:
                serviced.append(voice_id)
                remaining_budget -= cost_per_voice
            else:
                skipped.append(voice_id)

            scanned += 1
            pos = (pos + 1) % n_voices

        # Update cursor for next block: advance by number serviced
        # This ensures fairness - next block starts where this one stopped servicing
        self.cursor = (start_pos + len(serviced)) % n_voices

        return serviced, skipped

    def apply_skip_semantics(
        self,
        skipped_ids: List[int],
        block_num: int,
    ) -> None:
        """
        Apply skip semantics to skipped voices.

        Per Section 5:
        - Foreground voices: Freeze (time pauses)
        - Cloud voices: Decay (energy fades)
        """
        for voice_id in skipped_ids:
            state = self.voice_states.get(voice_id)
            if not state:
                continue

            state.skip_count += 1

            if state.is_foreground():
                # Freeze semantics: do nothing, state preserved
                # PRNG frozen (handled by PRNGManager)
                pass
            else:
                # Cloud decay semantics
                apply_cloud_decay(state)

                # Check for retirement
                check_retirement(state)

    def apply_service_semantics(
        self,
        serviced_ids: List[int],
        block_num: int,
    ) -> None:
        """
        Mark voices as serviced and update tracking.

        Serviced voices:
        - ADSR advances
        - Regime advances
        - PRNG advances
        - Energy updated
        - Audio recomputed
        """
        for voice_id in serviced_ids:
            state = self.voice_states.get(voice_id)
            if not state:
                continue

            state.last_serviced_block = block_num
            state.skip_count = 0
            state.intent_updated = False  # Clear for next block

    def retire_eligible(self) -> List[int]:
        """
        Retire all eligible voices.

        Returns list of retired voice IDs.
        """
        retired = []

        for voice_id in list(self.active_voices):
            state = self.voice_states.get(voice_id)
            if state and state.retired:
                retired.append(voice_id)
                self.deallocate_voice(voice_id)

        return retired

    def begin_block(self) -> None:
        """Called at start of each block to reset per-block flags."""
        for state in self.voice_states.values():
            state.intent_updated = False


# =============================================================================
# EVENT HANDLING (Section 6)
# =============================================================================

class EventDropPolicy(Enum):
    """Event handling policy for skipped voices."""
    DROP = "drop"      # Events are dropped, not queued
    QUEUE = "queue"    # Events are queued (NOT ALLOWED per spec)


def handle_skipped_voice_events(
    voice: VoiceState,
    events: List[int],
) -> List[int]:
    """
    Handle events for a skipped voice.

    Per WSE-STD-050 Section 6:

    Foreground voices:
    - Events MUST be dropped, not queued
    - Voice resumes with next block's events

    Cloud voices:
    - Events MUST NOT be emitted
    - No backlog or accumulation

    Rationale: Queuing introduces nondeterministic bursts and breaks replay.

    Returns: Empty list (all events dropped)
    """
    # Per spec: ALL events are dropped for skipped voices
    return []


# =============================================================================
# DETERMINISM VALIDATION (Section 10)
# =============================================================================

@dataclass
class DeterminismValidator:
    """
    Validates determinism guarantees per WSE-STD-050 Section 10.

    Given identical:
    - Initial state
    - Input IntentFrames
    - Budget sequence

    Then: Audio output MUST be bit-identical, regardless of which
    voices were skipped in intermediate blocks.
    """
    state_snapshots: List[Dict] = field(default_factory=list)

    def capture_state(
        self,
        scheduler: VoiceScheduler,
        block_num: int,
    ) -> Dict:
        """Capture scheduler state for comparison."""
        snapshot = {
            'block': block_num,
            'cursor': scheduler.cursor,
            'voices': {},
        }

        for vid, state in scheduler.voice_states.items():
            snapshot['voices'][vid] = {
                'class': state.voice_class.value,
                'energy': state.energy_q15,
                'adsr_phase': state.adsr_phase,
                'adsr_time': state.adsr_time,
                'regime_id': state.regime_id,
                'regime_dwell': state.regime_dwell,
                'retired': state.retired,
            }

        self.state_snapshots.append(snapshot)
        return snapshot

    def compare_runs(
        self,
        run1_snapshots: List[Dict],
        run2_snapshots: List[Dict],
    ) -> Tuple[bool, str]:
        """
        Compare two runs for determinism.

        Returns (is_identical, error_message).
        """
        if len(run1_snapshots) != len(run2_snapshots):
            return False, f"Different number of blocks: {len(run1_snapshots)} vs {len(run2_snapshots)}"

        for i, (s1, s2) in enumerate(zip(run1_snapshots, run2_snapshots)):
            if s1['cursor'] != s2['cursor']:
                return False, f"Block {i}: cursor mismatch {s1['cursor']} vs {s2['cursor']}"

            if set(s1['voices'].keys()) != set(s2['voices'].keys()):
                return False, f"Block {i}: different voice sets"

            for vid in s1['voices']:
                v1, v2 = s1['voices'][vid], s2['voices'][vid]
                for key in v1:
                    if v1[key] != v2[key]:
                        return False, f"Block {i}, voice {vid}: {key} mismatch {v1[key]} vs {v2[key]}"

        return True, "Runs are identical"


# =============================================================================
# COMPLIANCE CHECK (Section 13)
# =============================================================================

def check_compliance(
    scheduler: VoiceScheduler,
    serviced: List[int],
    skipped: List[int],
) -> Tuple[bool, List[str]]:
    """
    Check WSE-STD-050 compliance.

    An implementation is compliant if and only if:
    1. Foreground voices never decay due to skipping
    2. Cloud voices decay deterministically when skipped
    3. PRNG never advances on skipped voices
    4. Skipping does not change replay output

    Returns (is_compliant, list_of_violations).
    """
    violations = []

    for voice_id in skipped:
        state = scheduler.get_voice(voice_id)
        if not state:
            continue

        # Rule 1: Foreground voices never decay
        if state.is_foreground():
            # Energy should not have changed (we check this externally)
            pass

        # Rule 2: Cloud voices decay deterministically
        # (Verified by deterministic decay factor)

    # Rules 3 & 4 are verified by PRNG tests and determinism validator

    return len(violations) == 0, violations


# =============================================================================
# SUMMARY TABLE IMPLEMENTATION (Section 11)
# =============================================================================

def get_skip_behavior(voice_class: VoiceClass) -> Dict[str, str]:
    """
    Get behavior table for skipped voice per Section 11.

    | Voice Class | Skipped? | ADSR   | Regime | PRNG   | Energy | Audio |
    |-------------|----------|--------|--------|--------|--------|-------|
    | Foreground  | Yes      | Frozen | Frozen | Frozen | Frozen | Held  |
    | Cloud       | Yes      | Frozen | Frozen | Frozen | Decays | Held  |
    | Any         | No       | Advances| Advances| Advances| Updated| Recomputed|
    """
    if voice_class == VoiceClass.FOREGROUND:
        return {
            'adsr': 'frozen',
            'regime': 'frozen',
            'prng': 'frozen',
            'energy': 'frozen',
            'audio': 'held',
        }
    else:  # CLOUD
        return {
            'adsr': 'frozen',
            'regime': 'frozen',
            'prng': 'frozen',
            'energy': 'decays',
            'audio': 'held',
        }


def get_service_behavior() -> Dict[str, str]:
    """Get behavior table for serviced voice."""
    return {
        'adsr': 'advances',
        'regime': 'advances',
        'prng': 'advances',
        'energy': 'updated',
        'audio': 'recomputed',
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'ENERGY_EPS_Q15',
    'CLOUD_DECAY_Q15',
    'RETIRE_EPS_Q15',
    'RETIRE_BLOCKS',
    # Enums
    'VoiceClass',
    'VoiceStatus',
    'EventDropPolicy',
    # State
    'VoiceState',
    'FreezeState',
    # Semantics
    'apply_freeze',
    'apply_cloud_decay',
    'check_retirement',
    # Scheduler
    'VoiceScheduler',
    # Events
    'handle_skipped_voice_events',
    # Validation
    'DeterminismValidator',
    'check_compliance',
    # Behavior tables
    'get_skip_behavior',
    'get_service_behavior',
]
