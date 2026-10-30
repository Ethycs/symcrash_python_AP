"""
Golden Trace SUBTRACTIVE v0 - ABI Regression Artifact
======================================================

This file defines a deterministic VoiceFrame sequence using Mode B (Harmonic Bank)
oscillator for subtractive synthesis, and the expected output hashes for the
WSE-STD-VM-HYBRID v0 VM implementation.

Purpose:
- Freeze the SUBTRACTIVE synthesis ABI boundary
- Verify Mode B (Harmonic Bank) generates consistent output
- Catch changes that affect waveform generation, PWM, harmonic weights

Usage:
    python -m pytest tests/artifacts/test_golden_trace_subtractive.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python tests/artifacts/golden_trace_subtractive_v0.py --regenerate
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess

# =============================================================================
# VERSION METADATA
# =============================================================================

GOLDEN_TRACE_VERSION = "subtractive-v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
OSC_MODE = "MODE_B (Harmonic Bank)"

VERSION_INFO = {
    "golden_trace": GOLDEN_TRACE_VERSION,
    "vm_std": VM_STD_VERSION,
    "osc_mode": OSC_MODE,
    "table_size": 1024,
    "q_bits": 15,
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

VM_VERSION = VM_STD_VERSION
TRACE_NAME = "subtractive_saw_sweep"
TRACE_DURATION_BLOCKS = 48  # ~64ms * 48 = ~3 seconds
SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def generate_golden_sequence() -> List['VoiceFrame']:
    """
    Generate the deterministic frame sequence for SUBTRACTIVE golden trace.

    This represents a "sawtooth filter sweep" sound:
    - Attack: Quick ramp up with full brightness
    - Sustain: Sweep through waveform types (saw -> square -> pulse)
    - Release: Fade out with PWM modulation
    """
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.primitives import Q15_ONE
    from symcrash_python_ap.vm.state import OSC_MODE_B

    frames = []

    # Attack phase (blocks 0-7): Ramp up gain with sawtooth
    for i in range(8):
        gain = (i + 1) * 4096
        frames.append(VoiceFrame(
            note=69,  # A4 = 440 Hz
            coeff_osc_mode=OSC_MODE_B,  # Mode B: Harmonic Bank
            gain_q15=min(32767, gain),
            H_request=16,
            waveform_type=3,  # SAW
            pwm_enabled=False,
            pwm_width_q15=16384,
        ))

    # Sustain phase (blocks 8-39): Waveform morphing
    for i in range(32):
        # Morph through waveforms: saw(0-10) -> square(11-20) -> pulse(21-31)
        if i < 11:
            waveform = 3  # SAW
            pwm_enabled = False
            pwm_width = 16384
        elif i < 21:
            waveform = 1  # SQUARE
            pwm_enabled = False
            pwm_width = 16384
        else:
            waveform = 2  # PULSE
            pwm_enabled = True
            # PWM sweep from 50% to 10%
            pwm_width = 16384 - (i - 21) * 1200

        frames.append(VoiceFrame(
            note=69,
            coeff_osc_mode=OSC_MODE_B,
            gain_q15=Q15_ONE,
            H_request=16,
            waveform_type=waveform,
            pwm_enabled=pwm_enabled,
            pwm_width_q15=max(3276, pwm_width),  # Min 10% duty
        ))

    # Release phase (blocks 40-47): Fade out
    for i in range(8):
        gain = 32767 - (i + 1) * 4096
        frames.append(VoiceFrame(
            note=69,
            coeff_osc_mode=OSC_MODE_B,
            gain_q15=max(0, gain),
            H_request=16,
            waveform_type=5,  # TRIANGLE for softer release
            pwm_enabled=False,
            pwm_width_q15=16384,
        ))

    return frames


# =============================================================================
# FROZEN HASHES (Generated 2026-01-21; PWM blocks 29-39 regenerated 2026-06-09:
# VoiceFrame.partial_amps_q15 default became empty, so the PWM segment now
# honors H_request=16 instead of being clamped to 8 by the old default list.
# PWM blocks 30-39 regenerated 2026-06-11: compute_pwm_weights /
# compute_pwm_harmonics no longer emit the pulse DC term (2d - 1); the
# oscillator is AC-coupled, so the duty-dependent DC offset (which became LF
# rumble under width LFOs, e.g. analog_strings_v0) was removed by design.
# Block 29 is the 50%-duty pulse (dc was already 0) and is unchanged; all
# non-PWM blocks are byte-identical. Regenerated via:
#   pixi run python tests/artifacts/golden_trace_subtractive_v0.py --regenerate)
# Block 34 regenerated 2026-10-30 for exact ties-to-even coefficient products.
# Ten of 3072 samples change, by at most four PCM units. The former half-up
# reductions reproduce every old hash; an independent Fraction generator
# reproduces the corrected Python/JIT audio. See the evidence in
# regen_reports/subtractive_arithmetic_20261030.txt. Other blocks are unchanged.
# 2026-10-30: blocks except 37 regenerated for consistent Mode B2 spectrum
# selection, exact-pitch bases and enforced harmonic caps. The former matrix
# shortcut ignored selected weights/caps. Independent authored-weight laws
# and Fraction coefficient arithmetic reproduce the corrected audio in both
# Python and JIT. See regen_reports/carrier_spectrum_20261030.txt. The other
# three frozen traces and this input frame sequence remain unchanged.
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "d9a779e0b12079062fd306baea805e3a849be8daea97370482d97561c1b6c8a6",
    "dcb95982e803de39f373fdba7f4c5a340c722587ae44eb44f99f990497d3b110",
    "53b9967d96e226628b299486f6ad695401242adfcbf45ef9dd9d66893b2f0ca6",
    "0ca63af08e7dcb3ccc8f7ea22b52367251230d35c0e0857a50bea6094825be40",
    "fa31035acc8d51e7b9912e4aacfcd647c47ad631faab4939b47d5e80e98c80bf",
    "463b3b44a048dc870d323e07fcc0537209f9049c533254aa6da3c399de42f3a0",
    "28b486b30b10083d2b9f6415cd73f196ed40ded3beb403f73f1b530f8613d82a",
    "c15e55f88e764c0027edcb22b4bed5f14e8cfac9b6e2916ec06440557a63d5a7",
    "33e0ea9a30a937bff44dec9e78520f30262c0f292837f2a30f7efd266566d46a",
    "acbbfc4323e81ff12e3ebd381efaf635e76981d5f5639f8975617a663842954d",
    "c6089d367766c523965e9c11b7cab9d921c2e8c31a7b58f01a255df237823805",
    "05d0704b45ed137563f16b20b895135371ae942234b5e49fdd3e749cf46c2d5d",
    "7ad8b7c5fc7acb0f9fdb40a5e536d40548a44774f217aa10999fe33f0dece39a",
    "d2174f60519e2659095485c53a955fc73f6c76d60fc5f7863edfbbe7106c1044",
    "f5838f5fc541d8c5f4a048037191e65d0741df92933ce14be3282fdf73a31594",
    "e28e20c793f10b6964b585f09a4dba1ed2a754e3f0ee125fd32144551a902f28",
    "1ff1cd47329b813ac89ce390e678da6b36414ae9689fcba1ed214a16a6c2afc2",
    "8375928c2fe4a6df05eb327317430cc3982b180d66ef841a375d44cc74f5f88e",
    "c6411617850ac45da88a768adc18fcfa8d619a176fc767a036d6008572a8e47a",
    "d53149d5428e924211d2f55afa3956a5da6f81ea8b9c345875c6043a415fe455",
    "ee50264b1fc295c5edc7f10931b8ce6d3789e9a4cfdf6ea4840048ede293bc62",
    "2378a069456dc5d8baf8f2f04409ecea31c67218c20511a3e544910f4828165a",
    "9481c8d69e896f035d88d4cd7e89a644823e3213864484ee8e9350de9135ec81",
    "ae23ba5d4467cc6293938d51bb4bc263ce840d864ebb874f2800e61e8f5ecee8",
    "252da01215e0f2bff13c9c541233d7022771cb0a1747ad479aedff255e7116f5",
    "4a6b2f7d126ff5d90a4fa8cafccae2c4890634e5f9d1084fd7933d882e2fb705",
    "bfc538e22eb2f06e51b3096d6a921862ae5266167cab5f37139b3a07fc817dcc",
    "73780d7c98053a9b5760d67443709b1bd90412d89a1aaf1a22782982d94701cf",
    "7040401af2e89cab6a9925fa79d711da54ab502f7e24ac84431efcf3d487ec5a",
    "4ea53fd64d082d4ed840d67c97cca4402e83588943de9fd7759ab675b78ef125",
    "2fcad2dd90c20ace8bf7e8efe77031b00f6bbde69f0ff357286254fbe2932ca2",
    "7c4714c71fe44ab56bbbbf863d093f5f95c787afe589609d6a0e428f507ee878",
    "f076e33f4228a20f22e9b77b9a44f8cbc63dafc3b1b60fb084680320b71131f1",
    "f563f5527807081b5f52c5eff802e6d1946351752168794b0ab8b2be810c2c09",
    "371a1bce4c9f65d78765c3a7da0e71193b56ca4e512d2d746d6725e1b7a0d7f6",
    "6f2d507829cfcf9ae81b35f3977642c6ad3e8a45fe221a7fee027f3f3201bad5",
    "ea52ba40c813f0eca4068d08fcf5b5791c0b9e1f1ee08e01f31ae27e35b18645",
    "dbe57d8a4bc11039345d18b20a854e7de6b4f35a0bc804e16b7cfe9b808e188c",
    "c043dedb300e57161e66de937a3834f2cc8c462a72be6330b2d213c43c9e9edd",
    "ef1a8001234143a830fd81bcd585d23e9fa56ef1d013c3b24fa09968c67fdf2f",
    "5e0258a86f8c0209cad9e87c2a0a379096a1a1096176f9535f47f1f79068a459",
    "ad46613cc6357e5e4d18b5b8e4d94397fb30dd08c6b39b5f963e55766c433a19",
    "b60c88b66b3914bc71c254f071095cae8f3a41c284e44796f7dde45aa7f98ee3",
    "881f90897f0b1d2a9633bbd5e1b6aaa40599c46ba242206873b6b59d173d61bd",
    "dcdf09ecdd4900ac50c5dfeb0419e275a05f777f9705c6823e6bc21f0b46cf47",
    "af1a0b0758addcc52d4acefff3e4de12d72c54863528f7b22f833191d85b180c",
    "73f823b40e84509f9e85a43d6391b69bf0744ef4c5b96d3061010cc55a0ce285",
    "a05e09493ed2b80d76f90f4010142528a919f4fdeb1b8d260c9fff4d3238939a",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "d8a46aab67e7984565eda84a6225b211ddf06ea3e5fddfc4b387dfe04cabc4f5"


# =============================================================================
# HASH GENERATION
# =============================================================================

def compute_trace_hashes() -> Tuple[List[str], str]:
    """
    Run the golden trace through the VM and compute hashes.

    Returns:
        (block_hashes, trace_hash)
    """
    import numpy as np
    from symcrash_python_ap.vm import vm_step_block, VMState
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    # Create VM state (A4 = MIDI note 69)
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69, cents=0)

    # Generate frame sequence
    frames = generate_golden_sequence()

    # Run through VM and hash each block
    block_hashes = []
    all_bytes = b''

    for frame in frames:
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)

        # Hash this block
        block_bytes = output.tobytes()
        block_hash = hashlib.sha256(block_bytes).hexdigest()
        block_hashes.append(block_hash)

        # Accumulate for final hash
        all_bytes += block_bytes

    # Compute final trace hash
    trace_hash = hashlib.sha256(all_bytes).hexdigest()

    return block_hashes, trace_hash


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def regenerate_hashes():
    """
    Regenerate and print the frozen hashes.

    Run this when intentional behavior changes require updating the golden trace.
    """
    print("=" * 60)
    print("GOLDEN TRACE SUBTRACTIVE REGENERATION")
    print("=" * 60)
    print()

    # Print version info
    print("VERSION INFO:")
    for k, v in VERSION_INFO.items():
        print(f"  {k}: {v}")
    print()

    # Print trace info
    print(f"Trace: {TRACE_NAME}")
    print(f"Duration: {TRACE_DURATION_BLOCKS} blocks ({TRACE_DURATION_BLOCKS * BLOCK_SIZE / SAMPLE_RATE:.2f}s)")
    print()

    # Show old hash
    print(f"OLD TRACE HASH: {FROZEN_TRACE_HASH or '(not set)'}")

    # Compute new hashes
    block_hashes, trace_hash = compute_trace_hashes()

    # Show comparison
    print(f"NEW TRACE HASH: {trace_hash}")
    print()

    if trace_hash == FROZEN_TRACE_HASH:
        print("STATUS: UNCHANGED [OK]")
    else:
        print("STATUS: CHANGED [!]")
        print("  The VM output has changed. Review carefully before updating.")
    print()

    # Print hashes for copy-paste
    print("-" * 60)
    print("COPY-PASTE VALUES (if updating frozen hashes):")
    print("-" * 60)
    print()

    print("# Block hashes (first 5, last 5):")
    for i, h in enumerate(block_hashes[:5]):
        print(f"    # Block {i:2d}: {h}")
    print("    # ...")
    for i, h in enumerate(block_hashes[-5:], start=len(block_hashes)-5):
        print(f"    # Block {i:2d}: {h}")
    print()

    print(f"# Trace hash (all {len(block_hashes)} blocks):")
    print(f"FROZEN_TRACE_HASH = \"{trace_hash}\"")
    print()

    print("# Python list for FROZEN_BLOCK_HASHES:")
    print("FROZEN_BLOCK_HASHES = [")
    for h in block_hashes:
        print(f"    \"{h}\",")
    print("]")

    return block_hashes, trace_hash


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_hashes()
    else:
        print("Golden Trace SUBTRACTIVE v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_subtractive_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Print new hashes if you need to update frozen values")
