"""
Golden Trace Stress v0 - ABI Regression Artifact (Stress Test)
===============================================================

This file defines a stress-test VoiceFrame sequence that exercises
VM semantics not covered by the brass_stab trace:

Coverage targets:
- PRNG advancement: NoiseBand via event lane 2 (advances ONLY on those blocks)
- Permute ops: RotateNodes enabled window
- K ladder: pattern_id 0 (K=32) -> 2 (K=16) -> 0 (K=32)
- Tail mode ladder: FULL -> MIN -> FULL
- Event lanes: Impulse (0), NoiseBand (2), GateMask (4), CuspInjection (7)
- Smooth ramps: gain and beta changes within and across regions

Musical concept: "riser -> airy noise -> topology motion -> hit -> resolve"

Usage:
    python -m pytest pipeline_tests/artifacts/test_golden_trace_stress.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python pipeline_tests/artifacts/golden_trace_stress_v0.py --regenerate
"""

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# =============================================================================
# VERSION METADATA (Prevents silent ABI drift)
# =============================================================================

GOLDEN_TRACE_STRESS_VERSION = "v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
VOICEFRAME_STD_VERSION = "WSE-STD-020 v1"
SINE_CORE_LUT_VERSION = "1024-Q15-12bit-interp"

VERSION_INFO = {
    "golden_trace_stress": GOLDEN_TRACE_STRESS_VERSION,
    "vm_std": VM_STD_VERSION,
    "voiceframe_std": VOICEFRAME_STD_VERSION,
    "sine_core_lut": SINE_CORE_LUT_VERSION,
    "table_size": 1024,
    "q_bits": 15,
    "frac_bits": 12,
    "intent": "noise PRNG gating, permutes, K ladder, tail ladder, event lanes, ramps",
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

TRACE_NAME = "stress_riser_hit_resolve"
TRACE_DURATION_BLOCKS = 64
SAMPLE_RATE = 48000
BLOCK_SIZE = 64


@dataclass
class GoldenFrame:
    """Minimal frame definition for golden trace."""
    f0_hz: float
    gain_q15: int
    beta_target_q15: int
    ops_enable: List[str]
    families: dict
    # Required fields with defaults
    voice_id: int = 0
    note: int = 57  # A3 (220 Hz)
    cents: int = 0
    pattern_id: int = 0
    tail_mode: int = 0  # FULL
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    partial_amps_q15: List[int] = None
    event_mask: int = 0
    event_strength_q15: List[int] = None

    def __post_init__(self):
        if self.partial_amps_q15 is None:
            self.partial_amps_q15 = [32767, 16384, 8192, 4096]
        if self.event_strength_q15 is None:
            self.event_strength_q15 = [0] * 8


# =============================================================================
# HELPERS
# =============================================================================

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_i(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _q15_from_float(x: float) -> int:
    """Convert 0.0-1.0 float to Q15 (0-32767)."""
    return _clamp_int(int(round(x * 32767.0)), 0, 32767)


def _event_strengths(default: int = 0) -> List[int]:
    return [default] * 8


# =============================================================================
# STRESS GOLDEN SEQUENCE
# =============================================================================

def generate_golden_stress_sequence() -> List[GoldenFrame]:
    """
    64-block stress trace: "riser -> airy noise -> topology motion -> hit -> resolve"

    Coverage targets:
      - NoiseBand via event lane 2 (PRNG advances ONLY on these blocks)
      - Permute op enabled window (rotatenodes)
      - K ladder change via pattern_id (0=K32, 2=K16, back to 0)
      - Tail mode change FULL -> MIN -> FULL
      - Event lanes: impulse (lane 0), noise (lane 2), gatemask (lane 4), cusp (lane 7)
      - Smooth ramps: gain and beta_target_q15 changes within and across regions
    """
    frames: List[GoldenFrame] = []

    voice_id = 0
    note = 57  # A3
    f0_hz = 220.0  # Slightly lower than A4; stable carrier reference
    gain_q15 = 26000

    # =========================================================================
    # Region A: Blocks 0-15 (stable tone, brightness rises, beta ramps 0->0.6)
    # =========================================================================
    for i in range(16):
        t = i / 15.0
        beta = _lerp_i(0, _q15_from_float(0.60), t)

        # Backbone: classic-ish glue stack (no PRNG, no permutes)
        ops = ["tilt", "intramix", "diffusedown"]
        fam = {
            "tilt": _lerp(0.20, 0.70, t),  # brightness macro rising
            "intramix": 0.25,               # body/glue
            "diffusedown": 0.12,            # add detail from coarse->fine
        }

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=gain_q15,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=0,            # K=32
            tail_mode=0,             # FULL
            event_mask=0,
            event_strength_q15=_event_strengths(0),
        ))

    # =========================================================================
    # Region B: Blocks 16-23 (K->16, intermittent noise via event lane 2)
    # Noise on even blocks only => PRNG advances only on those blocks.
    # =========================================================================
    for j in range(8):
        # Keep beta ~0.6; slight gain trim to avoid energy jump with K drop
        beta = _q15_from_float(0.60)
        g = gain_q15 - 1200

        ops = ["tilt", "intramix"]  # still no permutes; keep it controlled
        fam = {"tilt": 0.65, "intramix": 0.22}

        event_mask = 0
        strengths = _event_strengths(0)

        if (j % 2) == 0:
            # Lane 2 = NoiseBand (advances PRNG)
            event_mask |= (1 << 2)
            strengths[2] = 9000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # K=16
            tail_mode=0,             # FULL (still; tail ladder exercised later)
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region C: Blocks 24-31 (permute window + tail downgrade to MIN)
    # This is the "topology motion" region.
    # =========================================================================
    for j in range(8):
        t = j / 7.0

        beta = _q15_from_float(0.60)
        g = gain_q15 - 1500

        # Enable a permute op (RotateNodes) plus mild Tilt
        ops = ["tilt", "rotatenodes"]
        fam = {
            "tilt": 0.60,
            "rotatenodes": _lerp(0.20, 0.70, t),
        }

        event_mask = 0
        strengths = _event_strengths(0)

        if j == 3:
            # Lane 4 = GateMask (deterministic; no PRNG)
            event_mask |= (1 << 4)
            strengths[4] = 14000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # still K=16
            tail_mode=2,             # MIN (FineClamp only)
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region D: Blocks 32-39 (hit/burst; impulse + cusp event, beta dip/recover)
    # =========================================================================
    for j in range(8):
        # Beta dip and recover: 0.60 -> 0.30 -> 0.60
        if j <= 3:
            beta = _lerp_i(_q15_from_float(0.60), _q15_from_float(0.30), j / 3.0)
        else:
            beta = _lerp_i(_q15_from_float(0.30), _q15_from_float(0.60), (j - 3) / 4.0)

        # Slight gain "punch" at onset
        g = gain_q15 - 800 if j == 0 else gain_q15 - 1400

        ops = ["tilt", "intramix"]
        fam = {"tilt": 0.72, "intramix": 0.28}

        event_mask = 0
        strengths = _event_strengths(0)

        if j in (0, 1, 2):
            # Lane 0 = Impulse (attack transient)
            event_mask |= (1 << 0)
            strengths[0] = 16000

        if j == 0:
            # Lane 7 = CuspInjection (dramatic but deterministic)
            event_mask |= (1 << 7)
            strengths[7] = 12000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # still K=16 for this "constrained" moment
            tail_mode=2,             # MIN continues
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region E: Blocks 40-63 (resolve: restore K=32, restore FULL tail, lowrank)
    # =========================================================================
    for j in range(24):
        t = j / 23.0

        beta = _q15_from_float(0.55)  # settle slightly less wet than before
        g = gain_q15

        # Add LowRankMorph gently to prove LOWRANK path is stable + deterministic.
        ops = ["tilt", "lowrankmorph", "diffuseup"]
        fam = {
            "tilt": 0.55,
            "lowrankmorph": _lerp(0.10, 0.30, t),
            "diffuseup": 0.10,
        }

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=0,            # back to K=32
            tail_mode=0,             # FULL restored
            event_mask=0,
            event_strength_q15=_event_strengths(0),
        ))

    assert len(frames) == 64
    return frames


# =============================================================================
# FROZEN HASHES (Generated 2026-01-01 - IDWPT now uses effect_mode)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "7bed9cc8a93ffbf92e9ab94dba9f7a0d679e2557684041598face9c3aa7cf4b4",
    "7b54d71896c2cc04805df2af22ca39d70ddc941924af009aac909c15b29262f2",
    "b4940c9510ff7255ffbb0198c2efe5e1d4150cf53ba996b581afb2de5b5f4b69",
    "09d6aba497d0a269e9b2de2f0c805c5ed83e9a9685d47a5c12696472c2ab0f7f",
    "e565973e2555a26f74b6aaa366188c8f0c508ba6d429c5f2e7574b091c775184",
    "9487a363ed3a6b76ab144107f03ac2e7b85e8000f359f66b4a901202e1231c23",
    "7cee32342e28424ae04ccda2f7192e4c9a859350248ee0e524ed8c2a65e8e613",
    "377a28d85910a23bcd592009effae3d90091740a84a41b8f3775e6577680c9dd",
    "e3acc8b48a0e4a2eff4c8c4e8de507bf7f6f3df54ca258af20aa856a3b53f78f",
    "a2aa2436f03fdfc35a40c0077311d361bc70d464d71474f6d5a4cb3acfaa2b78",
    "77dc3f7048e6e57c1a499a674df5820341ecf899719bcca458b32f3837045ea4",
    "e3234da384099c834a3436100eb6cae54e992b92f638795052c3b04fb9fd1071",
    "c1c67adee60e131773f93dba4dffcefa56a1f910c04959004aedd2ecb823d634",
    "71050b178894adf7d287222e96a6dde708d4a11884f4641bcb2003877ee18550",
    "13f85ff11c5806b0d765eca9cd15136db52a3ca0eb3ef238cc58035142c16a56",
    "59d365f1068a1d7f7750eb54b12a6c01a3dff667fa030d85c8a17bf3bcff96be",
    "350d0214e16db82a9f4c7b96f6246bd045579c747f5ec38d079c037e486b975d",
    "3770af919fddb33991bf292ed4fa10009aa87e1e936a6e44427a06914b8789a0",
    "e31733e3aed66513a672769b1ba7b717bc4c0d219f7b4019dac10ef9f01f220a",
    "977456768adf19b22c53d616764ddf1df77c355f429b5d51a020c2f54ed08c50",
    "56715435dd08516bcc596e31e6311644870e6a3f4024f4da827dd7ec58be18ed",
    "b6f61050905eade2f780d2a75fffc3fc2de11fe9c7c36f7828cb67b2e6d68471",
    "680d4505adeb1b4cc9bf005ed3418ad3563f9d7b7c4492bc1ee53d180329af61",
    "1656cf818a03e0b7c4c126960c4bc12b8284e4f548847c8cbcc43bd2d20201f7",
    "4bba7825269870703dcbf7f9fc8000bd2b3eba6e9e56c103947ed1e95f5a1b3d",
    "bf30aeb6f5f409f89287e5c2b040a2ec96e71491fc7598a917e72bbab3abd2ff",
    "2dc64022f8bdd5eac33a1f48bfd6f05733f6aaeede454c30c1a36e47b7b71210",
    "acf8beebf41899cf1d197f7e47b34d3adaa34b989596b261f50b521264a4a761",
    "a999215643960ecf6d4e6c275cc9686a70bdae498a89e3ccd5beafc2aaa05819",
    "2a21ae563163237a6f3b8253cebdbbe012c6bc7e478b95a15949fd58cb1fa4c9",
    "989fe7f3e9ca0948af1bf96f9815551a717ff3ff48518c651898b0c8ea62006f",
    "2e9fd3c21d4fa9ce4e3e15e2f6c28e9ff739f6c2eca5910ebf01db2897e2007e",
    "c407489654924b0d01ff1a9449c8f0257c9636e10a214af4ec5448277a5f02f1",
    "7cac7a87ef0a870d6c15a948e784d6ec1a6fa8cc9f3dce43d301f25ab3fa1f53",
    "23c97bae1f533641b1c82c222ba6be8208b3371c2744c4804035bd1612bec061",
    "5abaf97f902391f13b81fe1f9546303598a4c335b4d5c50f78659bb3fd97b4f3",
    "6f77dda96a23742d760e2a6f0c0371734d35a0a72bcb29933d216aa8c9d5bd33",
    "ae6d4e37ebafd55e9cadd70dd965fd1bb230f1ee88b552ad174a142fc1222a44",
    "59ee4a41b2856e242a6e75b9bf8c79a9f97c4df9bdc66c6da0277b3ea10c15e1",
    "2beda3161b8165dedb72cc345233d9774497f050dd86c4a81e043f508b52f6d8",
    "c450029b33bf9801de3f2f8f805c25cd405edca9372f6fce1012e92161f78ed6",
    "3d9e54977fef58c38dacb4e5d83c0aeda3d1c479e0e7293e172005a952f8bb17",
    "ee6e687006abf707e1c94cb80e7fcd99ead5d3b667e57589cd4488ebd63011ea",
    "5d52a1bc7cdf38df8de26279befa89ef774551247daae8e1ae0f172068d5c1bf",
    "ff1354a0a35c9decad9b1f32cfcf21e5ff4bc9c6d0dc8d287aad21469cab98c3",
    "dc55db8f3f4f991b0448c3f04bfd960589650d343593311fb4a924d82aa20505",
    "5261609e99c792cfb5809568b5ea0924429d0a662597b7ed3d8751e069831260",
    "7b0c1f291b2c70670292f7e37997e06840d27b2f01f57089e74e9534ae383623",
    "61cb5b393234d4b527cb3717803bfb7eb9d21cb50c58ed145618bfb67360c58e",
    "56a35984af3774ad80fd088de99339a7242d9902c8076f2ca3f3f7745bcc2891",
    "4ce346510778f34fea548ba6009c97e8da4fa58930ddf93e30a5b2a85805fab9",
    "9a8d5a715bb121e24a9dc6e32040c057c18d5d44ed85431596a7cd3373092a90",
    "249ed40439db187bee4f6e814693dccb4f85032056403470ccb8273eeefb6541",
    "69884aa3fa990afbc1112338e7bb514b4a364f98ed8a13d2059623a1de57266a",
    "b2566ec0b3524d4ff15f9e6001faa3f934483ac20a68c1dcdf535bcd61c8d38b",
    "d28a0899e3f36470843d38c900aaf552701b38ed685a34650f9b65a359392e44",
    "b56ed0fd439ce5f5f5d649401050ee87996daa33f8a252618f562aec50c6f3fb",
    "ef29eb3bd8e48239eb96605a5d33defc97a8724c1c207c6ce8dd4baed482540b",
    "2310f3accc59732c9b5cde29a4ce6af1de1c2f902411b3bb870366cb32773c2a",
    "140f878f81d8ae074d0cc71a07014f45371f39a22ff659d41d50b78ea3266a71",
    "ae832bcd3e9bcb24e6035ab0613549570abfe6b7a3879730eccc12abd2c7a799",
    "fe1b836f847251b5fef7c3e548651e55db859a4271f9efdc46f5a8b954a1f5ba",
    "222cb5a37113112d4ebfaabd234fe10e08554de05995b17fa28bb9220cb19dc4",
    "3d7415a5d5e9f544c548d496e576b76541ae700ab1e9af7a444d7eb6ead1ce59",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "03653d44144e11ab4b2bdf0d92bac90e937ad381561cc82baf4d08049b67c3de"


# =============================================================================
# HASH GENERATION
# =============================================================================

def compute_trace_hashes() -> Tuple[List[str], str]:
    """
    Run the golden stress trace through the VM and compute hashes.

    Returns:
        (block_hashes, trace_hash)
    """
    import numpy as np
    from symcrash_python_ap.vm import vm_step_block, VMState
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=220.0)

    # Generate frame sequence
    frames = generate_golden_stress_sequence()

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
            return result.stdout.strip()[:12]  # Short hash
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _write_regen_report(
    old_hash: str,
    new_hash: str,
    block_hashes: List[str],
) -> Optional[str]:
    """Write a regeneration report file for audit trail."""
    reports_dir = Path(__file__).parent / "regen_reports"
    try:
        reports_dir.mkdir(exist_ok=True)
    except OSError:
        return None

    timestamp = datetime.now()
    filename = f"regen_stress_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = reports_dir / filename

    git_commit = _get_git_commit()
    status = "UNCHANGED" if old_hash == new_hash else "CHANGED"

    report_lines = [
        "=" * 60,
        "GOLDEN TRACE STRESS REGENERATION REPORT",
        "=" * 60,
        "",
        f"Timestamp:    {timestamp.isoformat()}",
        f"Git Commit:   {git_commit or 'N/A'}",
        f"Trace Name:   {TRACE_NAME}",
        f"Trace Ver:    {GOLDEN_TRACE_STRESS_VERSION}",
        "",
        "--- VERSION INFO ---",
    ]
    for k, v in VERSION_INFO.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.extend([
        "",
        "--- HASH COMPARISON ---",
        f"Old Hash: {old_hash or '(not yet frozen)'}",
        f"New Hash: {new_hash}",
        f"Status:   {status}",
        "",
        f"Block Count: {len(block_hashes)}",
        "",
        "=" * 60,
    ])

    try:
        report_path.write_text("\n".join(report_lines))
        return str(report_path)
    except OSError:
        return None


def regenerate_hashes():
    """
    Regenerate and print the frozen hashes.

    Run this when intentional behavior changes require updating the golden trace.
    """
    print("=" * 60)
    print("GOLDEN TRACE STRESS REGENERATION")
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
    old_hash = FROZEN_TRACE_HASH if FROZEN_TRACE_HASH else "(not yet frozen)"
    print(f"OLD TRACE HASH: {old_hash}")

    # Compute new hashes
    block_hashes, trace_hash = compute_trace_hashes()

    # Show comparison
    print(f"NEW TRACE HASH: {trace_hash}")
    print()

    if FROZEN_TRACE_HASH and trace_hash == FROZEN_TRACE_HASH:
        print("STATUS: UNCHANGED [OK]")
    elif not FROZEN_TRACE_HASH:
        print("STATUS: INITIAL GENERATION")
    else:
        print("STATUS: CHANGED [!]")
        print("  The VM output has changed. Review carefully before updating.")
    print()

    # Write audit report
    report_path = _write_regen_report(FROZEN_TRACE_HASH, trace_hash, block_hashes)
    if report_path:
        print(f"Audit report written to: {report_path}")
    print()

    # Print hashes for copy-paste
    print("-" * 60)
    print("COPY-PASTE VALUES (update FROZEN_BLOCK_HASHES and FROZEN_TRACE_HASH):")
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
    print(f'FROZEN_TRACE_HASH = "{trace_hash}"')
    print()

    print("# Python list for FROZEN_BLOCK_HASHES:")
    print("FROZEN_BLOCK_HASHES: List[str] = [")
    for h in block_hashes:
        print(f'    "{h}",')
    print("]")

    return block_hashes, trace_hash


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_hashes()
    else:
        print("Golden Trace Stress v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_stress_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Write an audit report to regen_reports/")
        print("  4. Print new hashes if you need to update frozen values")
        print()
        print("Coverage targets:")
        print("  - PRNG advancement (NoiseBand on event lane 2)")
        print("  - Permute ops (RotateNodes)")
        print("  - K ladder (K=32 -> K=16 -> K=32)")
        print("  - Tail mode ladder (FULL -> MIN -> FULL)")
        print("  - Event lanes (Impulse, NoiseBand, GateMask, CuspInjection)")
        print("  - Smooth ramps (gain, beta)")
