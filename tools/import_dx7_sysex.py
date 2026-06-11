#!/usr/bin/env python3
"""
DX7 SysEx Bank Importer (WS1b, AUTHORING-IMPROVEMENT-PLAN)
==========================================================

Converts a 32-voice DX7 bulk-dump SysEx bank (.syx, 4104 bytes: 6-byte
header + 32 x 128 packed voice bytes + checksum + F7) into
``instrument_ir_v0`` YAML presets for this repo's FM6 engine.

Field map per the WS1 spec (AUTHORING-IMPROVEMENT-PLAN.md section 1b):

    SysEx field                  IR field
    ---------------------------  -----------------------------------------
    algorithm (0-31)             graph.fm6.algorithm (same numbering, 1-32)
    coarse/fine/detune per op    operators[n].ratio
    output level 0-99            operators[n].level (DX7 nonlinear curve)
    per-op EG rates/levels       envelopes targeting FM_OP1..6 + amp ADSR
    velocity sensitivity 0-7     velocity_sens (0.0-1.0)
    keyboard rate scaling 0-7    key_scale_rate (0.0-1.0)
    feedback 0-7                 graph.fm6.feedback

Known-lossy points (handled explicitly, surfaced by ``--report``):
  * 4-rate/4-level EG -> ADSR approximation (poor fits flagged per op)
  * carrier auto-normalization to <= 0.5 sum (FM6->FWPT chain saturates
    above ~0.5 summed carrier levels; removed once WS1a gateway lands)
  * LFO AMS/AMD routing dropped; PMD/PMS mapped to a pitch LFO
  * pitch EG, keyboard level scaling, transpose, key sync dropped
  * fixed-frequency oscillator mode approximated as a ratio at middle C
  * ENGINE DIVERGENCE: this repo's fm_algorithms.py table does not match
    the real DX7 topology for many algorithm numbers (e.g. real DX7 alg 5
    is three 2-op pairs; repo alg 5 is a dual stack into one carrier).
    The importer keeps the WS1 "same numbering" map and reports voices
    whose engine carrier set differs from the real DX7 carrier set.

Usage:
    pixi run python tools/import_dx7_sysex.py <bank.syx> --out-dir <dir>
    pixi run python tools/import_dx7_sysex.py <bank.syx> --out-dir <dir> --report
    pixi run python tools/import_dx7_sysex.py <bank.syx> --report          # report only
    pixi run python tools/import_dx7_sysex.py --demo-bank tests/fixtures/dx7_demo.syx

Options:
    --out-dir DIR     Write one instrument_ir_v0 YAML per voice into DIR
    --report          Print per-voice lossy-conversion (divergence) report;
                      with --out-dir, also writes <out-dir>/import_report.txt
    --voices SPEC     Only import selected voices, e.g. "1,3,9-12" (1-based)
    --prefix STR      Filename/id prefix (default "dx7")
    --force           Overwrite existing YAML files
    --demo-bank PATH  Write a synthetic 32-voice test bank built from
                      publicly documented classic parameter sets
                      (approximate E.PIANO 1 / BRASS 1) and exit

Note on WS1c (Dexed differential validation): Dexed is not available in
this environment, so no render diff is performed. The ``--report`` mode is
the structural stand-in: it documents every lossy conversion per voice so
a later Dexed pass knows exactly where to look.
"""

import argparse
import re
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from symcrash_python_ap.vm.fm_algorithms import get_algorithm  # noqa: E402


# =============================================================================
# CONSTANTS
# =============================================================================

VOICE_BYTES = 128          # Packed voice record size
BANK_VOICES = 32           # Voices per bulk dump
BANK_DATA_BYTES = VOICE_BYTES * BANK_VOICES   # 4096
HEADER = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])  # F0 43 0n 09 20 00

# Carrier-sum headroom: FM6->FWPT chain saturates above ~0.5 summed carrier
# levels (see CLAUDE.md / WS1a). Modulators are exempt.
CARRIER_SUM_MAX = 0.5

# DX7 output-level nonlinear curve (Dexed ScaleOutLevel): levels < 20 use a
# lookup, >= 20 are linear in the scaled domain; one scaled step = 0.75 dB,
# scaled 127 (level 99) = 0 dB = 1.0 linear.
_LEVEL_LUT = [0, 5, 9, 13, 17, 20, 23, 25, 27, 29,
              31, 33, 35, 37, 39, 41, 42, 43, 45, 46]

# Approximation: DX7 detune (0-14, 7 = center) -> cents per step.
DETUNE_CENTS_PER_STEP = 1.5

# Approximation: EG attack segments rise much faster than the dB-linear
# decay at the same rate; scale the full-swing decay time down.
ATTACK_TIME_FACTOR = 0.25

# Clamps for converted envelope times (ms).
MIN_TIME_MS = 0.5
MAX_TIME_MS = 20000.0

# The Amp node ADSR rides the control plane (control_registry.py), which
# caps attack at 500 blocks, decay at 1000, release at 1500 (64-sample
# blocks at 48 kHz -> ~1.333 ms/block). Exceeding these triggers compile
# clamps, so the importer clamps the amp ADSR itself and reports it.
AMP_ATTACK_MAX_MS = 500 * 64000.0 / 48000.0    # ~666 ms
AMP_DECAY_MAX_MS = 1000 * 64000.0 / 48000.0    # ~1333 ms
AMP_RELEASE_MAX_MS = 1500 * 64000.0 / 48000.0  # ~2000 ms

# DX7 pitch-mod sensitivity table (Dexed pitchmodsenstab), normalized so
# PMS=7 with PMD=99 gives ~1200 cents peak vibrato.
_PMS_TAB = [0, 10, 20, 33, 55, 92, 153, 255]

# DX7 LFO waveform -> engine LFO waveform (saw_down and S&H approximated).
_LFO_WAVE_MAP = {
    0: ("triangle", None),
    1: ("saw_up", "LFO saw-down approximated as saw_up (inverted phase)"),
    2: ("saw_up", None),
    3: ("square", None),
    4: ("sine", None),
    5: ("square", "LFO sample&hold approximated as square"),
}

# Real DX7 carrier sets per algorithm (1-based), for divergence reporting
# against this repo's fm_algorithms.py table.
REAL_DX7_CARRIERS = {
    1: {1, 3}, 2: {1, 3}, 3: {1, 4}, 4: {1, 4},
    5: {1, 3, 5}, 6: {1, 3, 5}, 7: {1, 3}, 8: {1, 3},
    9: {1, 3}, 10: {1, 4}, 11: {1, 4}, 12: {1, 3},
    13: {1, 3}, 14: {1, 3}, 15: {1, 3}, 16: {1},
    17: {1}, 18: {1}, 19: {1, 4, 5}, 20: {1, 2, 4},
    21: {1, 2, 4, 5}, 22: {1, 3, 5}, 23: {1, 2, 4},
    24: {1, 2, 3}, 25: {1, 2, 3}, 26: {1, 2, 4},
    27: {1, 2, 4}, 28: {1, 3, 6}, 29: {1, 2, 3, 5},
    30: {1, 2, 3, 6}, 31: {1, 2, 3, 4, 5},
    32: {1, 2, 3, 4, 5, 6},
}


# =============================================================================
# PACKED VOICE FORMAT (decode / encode)
# =============================================================================
#
# Packed 128-byte voice = 6 operators x 17 bytes (stored OP6 first, OP1
# last) + 26 global bytes. Bit-packed fields per the Yamaha DX7 bulk spec:
#
#   op byte 11: RC(2) << 2 | LC(2)
#   op byte 12: DET(4) << 3 | RS(3)
#   op byte 13: KVS(3) << 2 | AMS(2)
#   op byte 15: FC/coarse(5) << 1 | mode(1)
#   global 111: OKS(1) << 3 | feedback(3)
#   global 116: LPMS(3) << 4 | wave(3) << 1 | LFO key sync(1)


def unpack_operator(data):
    """Decode one packed 17-byte operator record into a parameter dict."""
    if len(data) != 17:
        raise ValueError(f"Operator record must be 17 bytes, got {len(data)}")
    b11, b12, b13, b15 = data[11], data[12], data[13], data[15]
    return {
        "eg_rates": [data[0], data[1], data[2], data[3]],
        "eg_levels": [data[4], data[5], data[6], data[7]],
        "break_point": data[8],
        "left_depth": data[9],
        "right_depth": data[10],
        "left_curve": b11 & 0x03,
        "right_curve": (b11 >> 2) & 0x03,
        "rate_scaling": b12 & 0x07,
        "detune": (b12 >> 3) & 0x0F,
        "ams": b13 & 0x03,
        "kvs": (b13 >> 2) & 0x07,
        "output_level": data[14],
        "mode": b15 & 0x01,          # 0 = ratio, 1 = fixed frequency
        "coarse": (b15 >> 1) & 0x1F,
        "fine": data[16],
    }


def pack_operator(op):
    """Encode an operator parameter dict into the packed 17-byte record."""
    return bytes([
        op["eg_rates"][0] & 0x7F, op["eg_rates"][1] & 0x7F,
        op["eg_rates"][2] & 0x7F, op["eg_rates"][3] & 0x7F,
        op["eg_levels"][0] & 0x7F, op["eg_levels"][1] & 0x7F,
        op["eg_levels"][2] & 0x7F, op["eg_levels"][3] & 0x7F,
        op["break_point"] & 0x7F,
        op["left_depth"] & 0x7F,
        op["right_depth"] & 0x7F,
        ((op["right_curve"] & 0x03) << 2) | (op["left_curve"] & 0x03),
        ((op["detune"] & 0x0F) << 3) | (op["rate_scaling"] & 0x07),
        ((op["kvs"] & 0x07) << 2) | (op["ams"] & 0x03),
        op["output_level"] & 0x7F,
        ((op["coarse"] & 0x1F) << 1) | (op["mode"] & 0x01),
        op["fine"] & 0x7F,
    ])


def unpack_voice(data):
    """Decode one packed 128-byte voice record into a parameter dict.

    Operators are returned in OP1..OP6 order (sysex stores OP6 first).
    """
    if len(data) != VOICE_BYTES:
        raise ValueError(f"Voice record must be 128 bytes, got {len(data)}")
    ops = []
    for i in range(6):
        ops.append(unpack_operator(data[i * 17:(i + 1) * 17]))
    ops.reverse()  # sysex order OP6..OP1 -> OP1..OP6

    g = data[102:128]
    return {
        "operators": ops,
        "pitch_eg_rates": [g[0], g[1], g[2], g[3]],
        "pitch_eg_levels": [g[4], g[5], g[6], g[7]],
        "algorithm": g[8] & 0x1F,             # 0-31 (display 1-32)
        "feedback": g[9] & 0x07,
        "osc_key_sync": (g[9] >> 3) & 0x01,
        "lfo_speed": g[10],
        "lfo_delay": g[11],
        "lfo_pmd": g[12],
        "lfo_amd": g[13],
        "lfo_key_sync": g[14] & 0x01,
        "lfo_wave": (g[14] >> 1) & 0x07,
        "lfo_pms": (g[14] >> 4) & 0x07,
        "transpose": g[15],                   # 0-48, 24 = C3 (center)
        "name": bytes(g[16:26]).decode("ascii", errors="replace"),
    }


def pack_voice(voice):
    """Encode a voice parameter dict into the packed 128-byte record."""
    out = bytearray()
    for op in reversed(voice["operators"]):   # OP6 first on the wire
        out += pack_operator(op)
    out += bytes(r & 0x7F for r in voice["pitch_eg_rates"])
    out += bytes(l & 0x7F for l in voice["pitch_eg_levels"])
    out.append(voice["algorithm"] & 0x1F)
    out.append(((voice["osc_key_sync"] & 0x01) << 3) | (voice["feedback"] & 0x07))
    out.append(voice["lfo_speed"] & 0x7F)
    out.append(voice["lfo_delay"] & 0x7F)
    out.append(voice["lfo_pmd"] & 0x7F)
    out.append(voice["lfo_amd"] & 0x7F)
    out.append(((voice["lfo_pms"] & 0x07) << 4)
               | ((voice["lfo_wave"] & 0x07) << 1)
               | (voice["lfo_key_sync"] & 0x01))
    out.append(voice["transpose"] & 0x7F)
    name = voice["name"][:10].ljust(10)
    out += bytes(ord(c) & 0x7F for c in name)
    assert len(out) == VOICE_BYTES
    return bytes(out)


def bank_checksum(data):
    """DX7 bulk checksum: two's complement of the 7-bit sum of the data."""
    return (128 - (sum(data) & 0x7F)) & 0x7F


def build_bank(voices):
    """Build a complete 4104-byte 32-voice bulk dump from voice dicts.

    Pads to 32 voices with INIT VOICE copies.
    """
    if len(voices) > BANK_VOICES:
        raise ValueError(f"Bank holds at most {BANK_VOICES} voices")
    voices = list(voices) + [init_voice()] * (BANK_VOICES - len(voices))
    data = b"".join(pack_voice(v) for v in voices)
    return HEADER + data + bytes([bank_checksum(data), 0xF7])


def parse_bank(raw):
    """Parse .syx bytes into (voices, warnings).

    Accepts a headered 32-voice bulk dump (possibly with leading junk or
    multiple messages; the first bulk dump is used) or a raw 4096-byte blob.
    """
    warnings = []

    # Find a DX7 bulk-dump header: F0 43 0n 09 20 00
    start = -1
    i = 0
    while True:
        i = raw.find(b"\xF0\x43", i)
        if i < 0:
            break
        if (len(raw) >= i + 6 and raw[i + 3] == 0x09
                and raw[i + 4] == 0x20 and raw[i + 5] == 0x00):
            start = i
            break
        i += 1

    if start >= 0:
        data = raw[start + 6:start + 6 + BANK_DATA_BYTES]
        if len(data) < BANK_DATA_BYTES:
            raise ValueError(
                f"Truncated bank: expected {BANK_DATA_BYTES} data bytes after "
                f"header, got {len(data)}")
        tail = raw[start + 6 + BANK_DATA_BYTES:start + 6 + BANK_DATA_BYTES + 2]
        if len(tail) >= 1:
            expected = bank_checksum(data)
            if tail[0] != expected:
                warnings.append(
                    f"checksum mismatch: stored 0x{tail[0]:02X}, "
                    f"computed 0x{expected:02X} (continuing)")
        if len(tail) >= 2 and tail[1] != 0xF7:
            warnings.append("bank not terminated with F7 (continuing)")
    elif len(raw) >= BANK_DATA_BYTES and len(raw) % VOICE_BYTES in (0, 8):
        warnings.append("no SysEx header found; treating file as raw voice data")
        data = raw[:BANK_DATA_BYTES]
    else:
        raise ValueError(
            "Not a DX7 32-voice bulk dump (no F0 43 0n 09 20 00 header and "
            "not a raw 4096-byte blob)")

    voices = [unpack_voice(data[v * VOICE_BYTES:(v + 1) * VOICE_BYTES])
              for v in range(BANK_VOICES)]
    return voices, warnings


# =============================================================================
# PARAMETER CONVERSION (DX7 -> instrument_ir_v0)
# =============================================================================

def scale_out_level(level):
    """DX7 output level 0-99 -> scaled level 0-127 (Dexed ScaleOutLevel)."""
    level = max(0, min(99, int(level)))
    return 28 + level if level >= 20 else _LEVEL_LUT[level]


def level_to_linear(level):
    """DX7 output level 0-99 -> linear amplitude 0..1.

    One scaled step = 0.75 dB; scaled 127 (level 99) = 0 dB = 1.0.
    Note: the same curve is used for carriers (amplitude) and modulators
    (index) - a known approximation of the DX7's separate index mapping.
    """
    if level <= 0:
        return 0.0
    return 10.0 ** ((scale_out_level(level) - 127) * 0.75 / 20.0)


def level_to_db(level):
    """DX7 EG/output level 0-99 -> dB (0 dB at 99, ~-95 dB at 0)."""
    return (scale_out_level(level) - 127) * 0.75


def eg_rate_to_full_time_ms(rate):
    """DX7 EG rate 0-99 -> time (ms) for a full 96 dB dB-linear traverse.

    Derived from the Dexed EG implementation: qrate = (rate*41)>>6,
    decrement (4 + (qrate&3)) << (qrate>>2) per 64-sample block at 44.1 kHz
    over a 2^28 level range.
    """
    rate = max(0, min(99, int(rate)))
    q = min(63, (rate * 41) >> 6)
    inc = (4 + (q & 3)) << (8 + (q >> 2))
    blocks = float(1 << 28) / inc
    return blocks * 64.0 / 44100.0 * 1000.0


def _clamp_time(ms, lo=MIN_TIME_MS, hi=MAX_TIME_MS):
    return max(lo, min(hi, ms))


def eg_to_adsr(eg_rates, eg_levels):
    """Approximate a DX7 4-rate/4-level EG as ADSR.

    Returns (adsr_dict, flags) where flags lists poor-fit conditions.

    Mapping decisions (the WS1 spec marks this conversion as known-lossy
    but does not fix a formula):
      * attack  = full-swing time(R1) * ATTACK_TIME_FACTOR
      * decay   = dB-scaled time L1->L2 at R2 plus L2->L3 at R3
      * sustain = lin(L3) / lin(L1)
      * release = dB-scaled time L3->0 at R4
    """
    r1, r2, r3, r4 = eg_rates
    l1, l2, l3, l4 = eg_levels
    flags = []

    peak = level_to_linear(l1)
    if peak <= 0.0:
        flags.append("EG peak level L1=0 (op effectively silent)")
        peak = 0.0

    attack_ms = _clamp_time(eg_rate_to_full_time_ms(r1) * ATTACK_TIME_FACTOR)

    # Decay: traverse L1 -> L2 at R2, then L2 -> L3 at R3, in the dB domain.
    db1, db2, db3 = level_to_db(l1), level_to_db(l2), level_to_db(l3)
    seg1_db = abs(db1 - db2)
    seg2_db = abs(db2 - db3)
    seg1_ms = eg_rate_to_full_time_ms(r2) * (seg1_db / 96.0)
    seg2_ms = eg_rate_to_full_time_ms(r3) * (seg2_db / 96.0)
    decay_ms = _clamp_time(seg1_ms + seg2_ms, lo=1.0)

    sustain = 0.0
    if peak > 0.0:
        sustain = max(0.0, min(1.0, level_to_linear(l3) / peak))

    # Release: R4 governs the fall to silence. The note is usually released
    # mid-decay (above L3), so for low sustain levels use at least a
    # half-swing (48 dB) span rather than the vanishing L3->silence span.
    rel_db = max(abs(level_to_db(l3) - level_to_db(0)), 48.0)
    release_ms = _clamp_time(eg_rate_to_full_time_ms(r4) * (rel_db / 96.0),
                             lo=1.0)

    # Poor-fit flags.
    if l2 + 5 < l3:
        flags.append(f"non-monotonic decay (L2={l2} < L3={l3}): "
                     "ADSR cannot dip below sustain")
    if l2 > l1 + 5:
        flags.append(f"secondary peak (L2={l2} > L1={l1}): ADSR cannot "
                     "rise after the attack peak")
    if l4 > 5:
        flags.append(f"non-silent end level L4={l4}: ADSR release always "
                     "ends at silence")
    if seg1_db > 12.0 and seg2_db > 12.0:
        rate1 = seg1_db / max(seg1_ms, 1e-6)
        rate2 = seg2_db / max(seg2_ms, 1e-6)
        if max(rate1, rate2) / max(min(rate1, rate2), 1e-9) > 4.0:
            flags.append("two-stage decay with very different slopes "
                         "collapsed to a single ADSR decay")

    adsr = {
        "attack": round(attack_ms, 1),
        "decay": round(decay_ms, 1),
        "sustain": round(sustain, 4),
        "release": round(release_ms, 1),
    }
    return adsr, flags, peak


def op_frequency_ratio(op, notes):
    """DX7 coarse/fine/detune (+osc mode) -> frequency ratio.

    Appends lossy notes to ``notes`` for fixed mode and detune approximation.
    """
    if op["mode"] == 0:
        base = 0.5 if op["coarse"] == 0 else float(op["coarse"])
        ratio = base * (1.0 + op["fine"] / 100.0)
    else:
        # Fixed frequency: f = 10^(coarse mod 4) * 10^(fine/100) Hz.
        # Not representable in the ratio-based IR; anchor to middle C.
        freq_hz = 10.0 ** ((op["coarse"] % 4) + op["fine"] / 100.0)
        ratio = freq_hz / 261.6256
        notes.append(f"fixed-frequency mode ({freq_hz:.2f} Hz) approximated "
                     f"as ratio {ratio:.4f} anchored at middle C")
    if op["detune"] != 7:
        cents = (op["detune"] - 7) * DETUNE_CENTS_PER_STEP
        ratio *= 2.0 ** (cents / 1200.0)
    return ratio


def lfo_speed_to_hz(speed):
    """DX7 LFO speed 0-99 -> Hz (Dexed-derived formula, approximate)."""
    speed = max(0, min(99, int(speed)))
    sr = 1 if speed == 0 else (165 * speed) >> 6
    sr *= 11 if sr < 160 else (11 + ((sr - 160) >> 4))
    return 25190424.0 / (1 << 32) * sr


def lfo_delay_to_ms(delay):
    """DX7 LFO delay 0-99 -> onset delay in ms (rough approximation)."""
    delay = max(0, min(99, int(delay)))
    return (delay / 99.0) ** 2 * 5000.0


def slugify(name):
    """Voice name -> filesystem/id slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "unnamed"


# =============================================================================
# VOICE -> IR CONVERSION
# =============================================================================

class ConvertedVoice:
    """One converted voice: IR-shaped data + per-voice divergence notes."""

    def __init__(self, index, voice):
        self.index = index           # 1-based voice slot in the bank
        self.voice = voice
        self.name = voice["name"].strip() or f"VOICE {index}"
        self.notes = []              # lossy/approximation notes
        self.carrier_scale = 1.0
        self.carrier_sum_raw = 0.0
        self.ir = None               # filled by convert()
        self.engine_carriers = set()
        self.op_comments = {}
        self.envelopes = {}          # env name -> dict
        self.amp = {}
        self.lfo = None

    @property
    def algorithm_1based(self):
        return self.voice["algorithm"] + 1


def convert_voice(index, voice):
    """Convert one unpacked DX7 voice into IR-shaped data + lossy notes."""
    cv = ConvertedVoice(index, voice)
    notes = cv.notes
    alg_1b = cv.algorithm_1based

    alg = get_algorithm(alg_1b)
    engine_carriers = {i + 1 for i in alg.get_carriers()}   # 1-based
    cv.engine_carriers = engine_carriers

    real_carriers = REAL_DX7_CARRIERS[alg_1b]
    if engine_carriers != real_carriers:
        notes.append(
            f"TOPOLOGY DIVERGENCE: engine algorithm {alg_1b} has carriers "
            f"{sorted(engine_carriers)} but the real DX7 algorithm {alg_1b} "
            f"has carriers {sorted(real_carriers)} (fm_algorithms.py table "
            "diverges from DX7; engine fix tracked under WS1c)")

    # --- Operators: ratio + level (+ EG peak folded into the level) ------
    ratios, levels, adsr_per_op, op_flags, peaks = [], [], [], [], []
    for i, op in enumerate(voice["operators"]):
        op_notes = []
        ratio = op_frequency_ratio(op, op_notes)
        adsr, flags, peak = eg_to_adsr(op["eg_rates"], op["eg_levels"])
        level = level_to_linear(op["output_level"]) * peak
        ratios.append(round(ratio, 4))
        levels.append(round(level, 4) if level > 1e-4 else 0.0)
        adsr_per_op.append(adsr)
        peaks.append(peak)
        op_flags.append(flags)
        for n in op_notes:
            notes.append(f"op{i + 1}: {n}")
        for f in flags:
            if levels[i] > 0.0:
                notes.append(f"op{i + 1} EG: {f}")
        if op["ams"] > 0 and voice["lfo_amd"] > 0 and levels[i] > 0.0:
            notes.append(f"op{i + 1}: LFO amplitude modulation dropped "
                         f"(AMS={op['ams']}, AMD={voice['lfo_amd']})")
        if (op["left_depth"] > 0 or op["right_depth"] > 0) and levels[i] > 0.0:
            notes.append(f"op{i + 1}: keyboard level scaling dropped "
                         f"(depths L={op['left_depth']}/R={op['right_depth']})")

    # --- Carrier headroom normalization ----------------------------------
    carrier_idx = [c - 1 for c in sorted(engine_carriers)]
    carrier_sum = sum(levels[i] for i in carrier_idx)
    cv.carrier_sum_raw = carrier_sum
    if carrier_sum > CARRIER_SUM_MAX:
        scale = CARRIER_SUM_MAX / carrier_sum
        cv.carrier_scale = scale
        for i in carrier_idx:
            levels[i] = round(levels[i] * scale, 4)
        notes.append(
            f"carrier levels normalized x{scale:.3f} "
            f"(raw sum {carrier_sum:.3f} > {CARRIER_SUM_MAX} headroom limit)")

    # --- Amp node: dominant engine-carrier envelope -----------------------
    active = [i for i in range(6) if levels[i] > 0.0]
    active_carriers = [i for i in carrier_idx if levels[i] > 0.0]
    if active_carriers:
        dom = max(active_carriers, key=lambda i: levels[i])
    else:
        notes.append("no active carriers (silent voice?) - amp uses op1 EG")
        dom = 0
    dom_op = voice["operators"][dom]
    dom_adsr = adsr_per_op[dom]

    # The dominant carrier group shares the amp envelope (matches the
    # dx_marimba/dx_lately_bass idiom: carriers shaped by the Amp node,
    # modulators by per-op envelopes).
    def env_key(i):
        op = voice["operators"][i]
        return (tuple(sorted(adsr_per_op[i].items())), op["kvs"],
                op["rate_scaling"])

    dom_group = [i for i in active if env_key(i) == env_key(dom)
                 and i in carrier_idx]

    cv.amp = dict(dom_adsr)
    for key, cap in (("attack", AMP_ATTACK_MAX_MS),
                     ("decay", AMP_DECAY_MAX_MS),
                     ("release", AMP_RELEASE_MAX_MS)):
        if cv.amp[key] > cap:
            notes.append(f"amp {key} {cv.amp[key]:.0f} ms clamped to "
                         f"{cap:.0f} ms (control-plane block range)")
            cv.amp[key] = round(cap, 1)
    if dom_op["kvs"] > 0:
        cv.amp["velocity_sens"] = round(dom_op["kvs"] / 7.0, 3)
    if dom_op["rate_scaling"] > 0:
        cv.amp["key_scale_rate"] = round(dom_op["rate_scaling"] / 7.0, 3)

    # --- Per-op envelopes for everything else -----------------------------
    groups = {}   # env_key -> [op indices]
    for i in active:
        if i in dom_group:
            continue
        groups.setdefault(env_key(i), []).append(i)

    env_defs = {}
    n_car, n_mod, n_mix = 0, 0, 0
    for key, members in sorted(groups.items(), key=lambda kv: kv[1][0]):
        adsr = dict(adsr_per_op[members[0]])
        op0 = voice["operators"][members[0]]
        # Skip flat envelopes with no expressive payload: the op simply
        # follows the Amp gate.
        flat = (adsr["sustain"] >= 0.98 and adsr["attack"] <= 2.0)
        if flat and op0["kvs"] == 0 and op0["rate_scaling"] == 0:
            continue
        all_car = all((i + 1) in engine_carriers for i in members)
        all_mod = all((i + 1) not in engine_carriers for i in members)
        if all_mod:
            n_mod += 1
            name = "mod_env" if n_mod == 1 else f"mod_env{n_mod}"
        elif all_car:
            n_car += 1
            name = "carrier_env" if n_car == 1 else f"carrier_env{n_car}"
        else:
            n_mix += 1
            name = "op_env" if n_mix == 1 else f"op_env{n_mix}"
        env = dict(adsr)
        if op0["kvs"] > 0:
            env["velocity_sens"] = round(op0["kvs"] / 7.0, 3)
        if op0["rate_scaling"] > 0:
            env["key_scale_rate"] = round(op0["rate_scaling"] / 7.0, 3)
        env["targets"] = [f"FM_OP{i + 1}" for i in members]
        env_defs[name] = env
    cv.envelopes = env_defs

    # --- LFO: PMD/PMS -> pitch vibrato ------------------------------------
    if voice["lfo_pmd"] > 0 and voice["lfo_pms"] > 0:
        wave, wave_note = _LFO_WAVE_MAP.get(voice["lfo_wave"], ("sine", None))
        if wave_note:
            notes.append(wave_note)
        peak_cents = (_PMS_TAB[voice["lfo_pms"]] / 255.0) * 1200.0
        # Engine convention: peak cents = 32767 * depth * amount.
        cv.lfo = {
            "waveform": wave,
            "rate_hz": round(lfo_speed_to_hz(voice["lfo_speed"]), 2),
            "depth": round(voice["lfo_pmd"] / 99.0, 3),
            "delay_ms": round(lfo_delay_to_ms(voice["lfo_delay"]), 1),
            "amount": round(peak_cents / 32767.0, 6),
        }
        notes.append(
            f"LFO PMD={voice['lfo_pmd']}/PMS={voice['lfo_pms']} mapped to "
            f"pitch LFO ({cv.lfo['rate_hz']} Hz, peak "
            f"~{peak_cents * cv.lfo['depth']:.0f} cents) - approximate")
    elif voice["lfo_pmd"] > 0 or voice["lfo_amd"] > 0:
        notes.append("LFO present but not routed (PMS=0 or pitch-only "
                     "routing unavailable) - dropped")

    # --- Globals that have no IR representation ---------------------------
    if voice["pitch_eg_levels"] != [50, 50, 50, 50]:
        notes.append(f"non-flat pitch EG dropped "
                     f"(levels {voice['pitch_eg_levels']})")
    if voice["transpose"] != 24:
        semis = voice["transpose"] - 24
        notes.append(f"transpose {semis:+d} semitones dropped")
    if voice["osc_key_sync"] == 0:
        notes.append("oscillator key sync OFF not representable - dropped")

    # --- Assemble ----------------------------------------------------------
    cv.op_comments = {}
    for i in range(6):
        role = "carrier" if (i + 1) in engine_carriers else "modulator"
        extra = ""
        if levels[i] == 0.0:
            extra = " (silent)"
        elif alg.feedback_op == i and voice["feedback"] > 0:
            extra = " (feedback)"
        cv.op_comments[i] = f"Op{i + 1} {role}{extra}"

    cv.ir = {
        "algorithm": alg_1b,
        "feedback": voice["feedback"],
        "ratios": ratios,
        "levels": levels,
    }
    return cv


# =============================================================================
# YAML EMISSION (hand-formatted to match presets/instruments/dx_*.yaml)
# =============================================================================

def _fmt(v):
    """Format a number for YAML output."""
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e6:
            return f"{v:.1f}"
        return repr(round(v, 6))
    return str(v)


def voice_to_yaml(cv, source_name, prefix="dx7"):
    """Render a ConvertedVoice as instrument_ir_v0 YAML text (LF newlines;
    caller is responsible for CRLF on disk)."""
    slug = slugify(cv.name)
    preset_id = f"{prefix}-{slug.replace('_', '-')}-v0"
    seed = zlib.crc32(f"{source_name}:{cv.index}:{cv.name}".encode()) & 0x7FFFFFFF
    alg = cv.ir["algorithm"]
    carriers = sorted(cv.engine_carriers)

    L = []
    L.append(f"# {cv.name} - imported from DX7 SysEx")
    L.append("# " + "=" * max(28, len(cv.name) + 26))
    L.append("#")
    L.append(f"# Source: {source_name}, voice {cv.index}/32")
    L.append(f"# DX7 algorithm {alg} (engine carriers: "
             f"{', '.join('Op%d' % c for c in carriers)}), "
             f"feedback {cv.ir['feedback']}")
    L.append("# Imported by tools/import_dx7_sysex.py (WS1b). Lossy points")
    L.append("# for this voice are listed in the import report (--report).")
    if cv.carrier_scale != 1.0:
        L.append(f"# Carrier levels normalized x{cv.carrier_scale:.3f} to the "
                 f"~{CARRIER_SUM_MAX} headroom limit.")
    L.append("# Conformance: Expressive (full articulation)")
    L.append("")
    L.append("instrument_ir_v0:")
    L.append("  meta:")
    L.append(f'    id: "{preset_id}"')
    L.append(f'    name: "{cv.name}"')
    L.append("    version: 1")
    L.append('    author: "dx7-import"')
    L.append(f"    seed: 0x{seed:08X}")
    L.append("")
    L.append("  audio:")
    L.append("    sample_rate: 48000")
    L.append("    block_size: 64")
    L.append("")
    L.append("  graph:")
    L.append("    fm6:")
    L.append("      type: FM6")
    L.append("      output: amp1")
    L.append(f"      algorithm: {alg}")
    L.append(f"      feedback: {cv.ir['feedback']}")
    L.append("      operators:")
    for i in range(6):
        ratio = _fmt(cv.ir["ratios"][i])
        level = _fmt(cv.ir["levels"][i])
        pad = " " * max(1, 10 - len(ratio))
        L.append(f"        - ratio: {ratio}{pad}# {cv.op_comments[i]}")
        L.append(f"          level: {level}")
    L.append("    amp1:")
    L.append("      type: Amp")
    L.append(f"      attack: {_fmt(cv.amp['attack'])}")
    L.append(f"      decay: {_fmt(cv.amp['decay'])}")
    L.append(f"      sustain: {_fmt(cv.amp['sustain'])}")
    L.append(f"      release: {_fmt(cv.amp['release'])}")
    if "velocity_sens" in cv.amp:
        L.append(f"      velocity_sens: {_fmt(cv.amp['velocity_sens'])}")
    if "key_scale_rate" in cv.amp:
        L.append(f"      key_scale_rate: {_fmt(cv.amp['key_scale_rate'])}")

    if cv.envelopes:
        L.append("")
        L.append("  envelopes:")
        for name, env in cv.envelopes.items():
            L.append(f"    {name}:")
            L.append(f"      attack: {_fmt(env['attack'])}")
            L.append(f"      decay: {_fmt(env['decay'])}")
            L.append(f"      sustain: {_fmt(env['sustain'])}")
            L.append(f"      release: {_fmt(env['release'])}")
            if "velocity_sens" in env:
                L.append(f"      velocity_sens: {_fmt(env['velocity_sens'])}")
            if "key_scale_rate" in env:
                L.append(f"      key_scale_rate: {_fmt(env['key_scale_rate'])}")
            L.append(f"      targets: [{', '.join(env['targets'])}]")

    if cv.lfo:
        L.append("")
        L.append("  modulators:")
        L.append("    vibrato:")
        L.append(f'      waveform: "{cv.lfo["waveform"]}"')
        L.append(f"      rate: {_fmt(cv.lfo['rate_hz'])}")
        L.append(f"      depth: {_fmt(cv.lfo['depth'])}")
        L.append(f"      delay: {_fmt(cv.lfo['delay_ms'])}")
        L.append('      sync: "free"')
        L.append("      targets:")
        L.append(f"        - pitch: {_fmt(cv.lfo['amount'])}")

    L.append("")
    L.append("  behavior:")
    for state in ("idle", "attack", "stable", "recovery"):
        L.append(f"    {state}:")
        L.append("      emphasis: {}")
        L.append("      trees: {A: harmonic, B: harmonic, blend: 0.0}")
    L.append("")
    L.append("  compile:")
    L.append("    material:")
    L.append("      blend: 0.0")
    L.append("      target: 0.0")
    L.append("      ramp: 1")
    L.append("")
    return "\n".join(L)


# =============================================================================
# DIVERGENCE / LOSSY-POINTS REPORT
# =============================================================================

def build_report(converted, source_name, bank_warnings):
    """Build the per-voice lossy-conversion report text."""
    L = []
    L.append("DX7 SysEx import - lossy-conversion (divergence) report")
    L.append("=" * 60)
    L.append(f"Source bank: {source_name}")
    L.append("")
    L.append("Dexed differential rendering (WS1c) was NOT performed: Dexed")
    L.append("is unavailable in this environment. This report lists every")
    L.append("structural lossy point per voice so a later Dexed pass knows")
    L.append("where to listen first.")
    L.append("")
    for w in bank_warnings:
        L.append(f"BANK WARNING: {w}")
    if bank_warnings:
        L.append("")
    n_clean = 0
    for cv in converted:
        L.append(f"[{cv.index:2d}] {cv.name!r}  alg {cv.algorithm_1based}, "
                 f"fb {cv.ir['feedback']}, "
                 f"carrier sum {cv.carrier_sum_raw:.3f}"
                 + (f" -> x{cv.carrier_scale:.3f}"
                    if cv.carrier_scale != 1.0 else ""))
        if cv.notes:
            for n in cv.notes:
                L.append(f"      - {n}")
        else:
            n_clean += 1
            L.append("      (no lossy conversions)")
    L.append("")
    L.append(f"{len(converted)} voices, {n_clean} without lossy notes")
    return "\n".join(L) + "\n"


# =============================================================================
# DEMO BANK: classic voices encoded from publicly documented parameter sets
# =============================================================================

def init_voice():
    """DX7 INIT VOICE."""
    op_off = {
        "eg_rates": [99, 99, 99, 99], "eg_levels": [99, 99, 99, 0],
        "break_point": 39, "left_depth": 0, "right_depth": 0,
        "left_curve": 0, "right_curve": 0, "rate_scaling": 0,
        "detune": 7, "ams": 0, "kvs": 0, "output_level": 0,
        "mode": 0, "coarse": 1, "fine": 0,
    }
    ops = [dict(op_off) for _ in range(6)]
    ops[0]["output_level"] = 99
    return {
        "operators": ops,
        "pitch_eg_rates": [99, 99, 99, 99],
        "pitch_eg_levels": [50, 50, 50, 50],
        "algorithm": 0, "feedback": 0, "osc_key_sync": 1,
        "lfo_speed": 35, "lfo_delay": 0, "lfo_pmd": 0, "lfo_amd": 0,
        "lfo_key_sync": 1, "lfo_wave": 0, "lfo_pms": 3,
        "transpose": 24, "name": "INIT VOICE",
    }


def _mk_op(eg_rates, eg_levels, output_level, coarse=1, fine=0, detune=7,
           kvs=0, ams=0, rate_scaling=0, mode=0, break_point=39,
           left_depth=0, right_depth=0, left_curve=0, right_curve=0):
    return {
        "eg_rates": list(eg_rates), "eg_levels": list(eg_levels),
        "break_point": break_point, "left_depth": left_depth,
        "right_depth": right_depth, "left_curve": left_curve,
        "right_curve": right_curve, "rate_scaling": rate_scaling,
        "detune": detune, "ams": ams, "kvs": kvs,
        "output_level": output_level, "mode": mode,
        "coarse": coarse, "fine": fine,
    }


def epiano1_voice():
    """Approximation of the classic DX7 'E.PIANO 1' parameter set.

    Values follow the widely published ROM1A listing (approximate - used
    as a realistic round-trip/import fixture, not a byte-exact ROM copy).
    Real DX7 algorithm 5: pairs 2->1*, 4->3*, 6->5*, feedback on op6.
    """
    ops = [
        _mk_op([96, 25, 25, 67], [99, 75, 0, 0], 99, coarse=1, kvs=2,
               rate_scaling=3),                                   # OP1 carrier
        _mk_op([95, 50, 35, 78], [99, 75, 0, 0], 58, coarse=14, kvs=7,
               detune=0),                                         # OP2 tine mod
        _mk_op([95, 20, 20, 50], [99, 95, 0, 0], 99, coarse=1, kvs=2,
               rate_scaling=3, detune=10),                        # OP3 carrier
        _mk_op([95, 29, 20, 50], [99, 95, 0, 0], 89, coarse=1, kvs=6,
               detune=7),                                         # OP4 mod
        _mk_op([95, 20, 20, 50], [99, 95, 0, 0], 99, coarse=1, kvs=0,
               rate_scaling=3, detune=4),                         # OP5 carrier
        _mk_op([49, 99, 28, 68], [98, 98, 91, 0], 79, coarse=1, kvs=6,
               detune=14),                                        # OP6 mod (fb)
    ]
    return {
        "operators": ops,
        "pitch_eg_rates": [94, 67, 95, 60],
        "pitch_eg_levels": [50, 50, 50, 50],
        "algorithm": 4,            # 0-based -> displayed algorithm 5
        "feedback": 6, "osc_key_sync": 1,
        "lfo_speed": 34, "lfo_delay": 33, "lfo_pmd": 5, "lfo_amd": 0,
        "lfo_key_sync": 1, "lfo_wave": 4, "lfo_pms": 3,
        "transpose": 24, "name": "E.PIANO 1",
    }


def brass1_voice():
    """Approximation of the classic DX7 'BRASS 1' parameter set.

    Approximate published values (algorithm 18, heavy feedback, slowish
    carrier attacks with high sustain) - realistic fixture material.
    """
    ops = [
        _mk_op([72, 76, 99, 71], [99, 88, 96, 0], 99, coarse=1, kvs=0,
               rate_scaling=0, detune=7),                         # OP1 carrier
        _mk_op([62, 51, 29, 71], [82, 95, 96, 0], 86, coarse=0, fine=50,
               kvs=0, detune=7),                                  # OP2 mod
        _mk_op([77, 76, 82, 71], [99, 98, 98, 0], 98, coarse=1, kvs=2,
               detune=8),                                         # OP3 mod (fb chain)
        _mk_op([77, 36, 41, 71], [99, 98, 98, 0], 79, coarse=1, kvs=2,
               detune=10),                                        # OP4 mod
        _mk_op([77, 36, 41, 71], [99, 98, 98, 0], 72, coarse=1, kvs=2,
               detune=4),                                         # OP5 mod
        _mk_op([49, 99, 28, 68], [98, 98, 91, 0], 65, coarse=1, kvs=0,
               detune=7),                                         # OP6 mod
    ]
    return {
        "operators": ops,
        "pitch_eg_rates": [84, 95, 95, 60],
        "pitch_eg_levels": [50, 50, 50, 50],
        "algorithm": 17,           # 0-based -> displayed algorithm 18
        "feedback": 7, "osc_key_sync": 1,
        "lfo_speed": 37, "lfo_delay": 0, "lfo_pmd": 5, "lfo_amd": 0,
        "lfo_key_sync": 1, "lfo_wave": 0, "lfo_pms": 3,
        "transpose": 24, "name": "BRASS   1",
    }


def write_demo_bank(path):
    """Write the synthetic fixture bank (E.PIANO 1, BRASS 1, 30x INIT)."""
    bank = build_bank([epiano1_voice(), brass1_voice()])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bank)
    return path


# =============================================================================
# CLI
# =============================================================================

def parse_voice_spec(spec):
    """Parse a 1-based voice selection like '1,3,9-12' into a set of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = [v for v in out if not 1 <= v <= BANK_VOICES]
    if bad:
        raise ValueError(f"voice indices out of range 1-{BANK_VOICES}: {bad}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import a DX7 32-voice SysEx bank as instrument_ir_v0 "
                    "YAML presets")
    parser.add_argument("bank", nargs="?", help="Path to the .syx bank file")
    parser.add_argument("--out-dir", help="Directory for the generated YAML")
    parser.add_argument("--report", action="store_true",
                        help="Print the per-voice lossy-conversion report "
                             "(with --out-dir, also writes import_report.txt)")
    parser.add_argument("--voices", help="Voice selection, e.g. '1,3,9-12'")
    parser.add_argument("--prefix", default="dx7",
                        help="Filename/id prefix (default: dx7)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing YAML files")
    parser.add_argument("--demo-bank", metavar="PATH",
                        help="Write a synthetic fixture bank and exit")
    args = parser.parse_args(argv)

    if args.demo_bank:
        path = write_demo_bank(args.demo_bank)
        print(f"Wrote demo bank: {path} ({path.stat().st_size} bytes)")
        return 0

    if not args.bank:
        parser.error("a .syx bank path is required (or use --demo-bank)")
    if not args.out_dir and not args.report:
        parser.error("nothing to do: pass --out-dir and/or --report")

    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"ERROR: {bank_path} not found")
        return 1

    raw = bank_path.read_bytes()
    try:
        voices, warnings = parse_bank(raw)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    for w in warnings:
        print(f"WARNING: {w}")

    selection = parse_voice_spec(args.voices) if args.voices else None

    converted = []
    seen_slugs = {}
    for idx, voice in enumerate(voices, start=1):
        if selection and idx not in selection:
            continue
        name = voice["name"].strip()
        if not selection and (not name or name == "INIT VOICE"):
            continue   # skip pad voices unless explicitly selected
        cv = convert_voice(idx, voice)
        slug = slugify(cv.name)
        if slug in seen_slugs:
            seen_slugs[slug] += 1
            cv.slug = f"{slug}_{seen_slugs[slug]}"
        else:
            seen_slugs[slug] = 1
            cv.slug = slug
        converted.append(cv)

    if not converted:
        print("No voices selected/found to import.")
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cv in converted:
            out_path = out_dir / f"{args.prefix}_{cv.slug}_v0.yaml"
            if out_path.exists() and not args.force:
                print(f"SKIP (exists, use --force): {out_path}")
                continue
            text = voice_to_yaml(cv, bank_path.name, prefix=args.prefix)
            # Repo convention: CRLF line endings.
            with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
                f.write(text)
            print(f"Wrote {out_path}  ({cv.name!r}, alg {cv.algorithm_1based})")

    if args.report:
        report = build_report(converted, bank_path.name, warnings)
        print()
        print(report)
        if args.out_dir:
            report_path = Path(args.out_dir) / "import_report.txt"
            with open(report_path, "w", encoding="utf-8", newline="\r\n") as f:
                f.write(report)
            print(f"Report written to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
