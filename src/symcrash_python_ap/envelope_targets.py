"""
Envelope Target Registry
========================

Defines valid destinations for envelope routing in the matrix envelope system.

Envelopes can route their Q15 output to any of these targets. Multiple targets
can share the same envelope (e.g., carrier envelope → FM_OP1, FM_OP2, FM_OP3, FM_OP4).

For envelope generators and processors, see envelope_suite.py which provides:
- ADSR, AR, Decay (envelope generators)
- EnvelopeFollower, Slew (signal processors)
- LFO (modulator)

Usage:
    from symcrash_python_ap.envelope_targets import EnvelopeTarget
    from symcrash_python_ap.envelope_suite import ADSR, LFO

    targets = [EnvelopeTarget.FM_OP1, EnvelopeTarget.FM_OP2]
    env = ADSR(attack_ms=10, decay_ms=100, sustain=0.7, release_ms=200)
"""

from enum import IntEnum


class EnvelopeTarget(IntEnum):
    """Valid envelope routing destinations."""

    # FM6 operators (0-5)
    # Route envelope output to FM operator's env_value_q15
    FM_OP1 = 0
    FM_OP2 = 1
    FM_OP3 = 2
    FM_OP4 = 3
    FM_OP5 = 4
    FM_OP6 = 5

    # Voice-level (10-19)
    GAIN = 10  # Voice output gain

    # Filter (20-29) - reserved for future use
    FILTER_CUTOFF = 20
    FILTER_RESO = 21

    # Pitch (30-39) - reserved for future use
    PITCH = 30
    VIBRATO_DEPTH = 31

    # Material engine (40-49) - reserved for future use
    MATERIAL_MIX = 40
    MATERIAL_DRIVE = 41


# Convenience sets for target categories
FM_OPERATOR_TARGETS = frozenset([
    EnvelopeTarget.FM_OP1,
    EnvelopeTarget.FM_OP2,
    EnvelopeTarget.FM_OP3,
    EnvelopeTarget.FM_OP4,
    EnvelopeTarget.FM_OP5,
    EnvelopeTarget.FM_OP6,
])

VOICE_TARGETS = frozenset([
    EnvelopeTarget.GAIN,
])


def parse_envelope_target(name: str) -> EnvelopeTarget:
    """
    Parse envelope target from string name.

    Args:
        name: Target name (e.g., "FM_OP1", "GAIN")

    Returns:
        EnvelopeTarget enum value

    Raises:
        KeyError: If name is not a valid target
    """
    return EnvelopeTarget[name]


__all__ = [
    'EnvelopeTarget',
    'FM_OPERATOR_TARGETS',
    'VOICE_TARGETS',
    'parse_envelope_target',
]
