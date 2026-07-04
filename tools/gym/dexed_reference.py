#!/usr/bin/env python3
"""
dexed_reference.py - render a DX7 voice through the REAL Dexed (msfa) engine.

This is the WS5 *control*: a bit-accurate DX7 reference independent of our
own engine, so patch-reversal effectiveness can be measured against a fixed
yardstick instead of a self-referential one (our engine rendering its own
import). See AUTHORING-IMPROVEMENT-PLAN WS1c / WS5.

Runs in WSL (Linux VST3 hosting via pedalboard); the repo lives on /mnt/f.
It loads a voice from a 32-voice DX7 SysEx bank by parsing the packed VMEM
record and setting Dexed's named VST parameters directly (SysEx MIDI-thru
is NOT forwarded by pedalboard, so the param path is the reliable one).

Usage (inside WSL, with the venv/pixi env that has pedalboard):
    python dexed_reference.py BANK.syx --voice 29 --note 60 --dur 2.5 \
        --out ref.wav --plugin ~/dx7_ref/dexed/Dexed.vst3
"""
import argparse
from pathlib import Path

import numpy as np

# --- Packed DX7 VMEM voice parsing (128 bytes/voice, ops stored OP6..OP1) ---

def parse_packed_voice(rec: bytes) -> dict:
    """Parse one 128-byte packed DX7 voice into raw DX7 field values.

    Returns a dict with per-op fields keyed op1..op6 (1-based, DX7 order)
    plus the global fields. All values are raw DX7 integers.
    """
    assert len(rec) == 128, f"packed voice must be 128 bytes, got {len(rec)}"
    ops = {}
    # Packed bank stores operators OP6 first, OP1 last -> 17 bytes each.
    for slot in range(6):
        o = rec[slot * 17: slot * 17 + 17]
        op_num = 6 - slot  # slot 0 = OP6 ... slot 5 = OP1
        detune = (o[12] >> 3) & 0x0F           # 0..14 -> DX7 detune -7..+7
        ops[op_num] = {
            "eg_rate": [o[0], o[1], o[2], o[3]],
            "eg_level": [o[4], o[5], o[6], o[7]],
            "break_point": o[8],
            "l_scale_depth": o[9],
            "r_scale_depth": o[10],
            "l_curve": o[11] & 0x03,
            "r_curve": (o[11] >> 2) & 0x03,
            "rate_scale": o[12] & 0x07,
            "detune": detune - 7,
            "amp_mod_sens": o[13] & 0x03,
            "key_vel_sens": (o[13] >> 2) & 0x07,
            "output_level": o[14],
            "mode": o[15] & 0x01,               # 0 RATIO, 1 FIXED
            "f_coarse": (o[15] >> 1) & 0x1F,
            "f_fine": o[16],
        }
    g = {
        "pitch_eg_rate": [rec[102], rec[103], rec[104], rec[105]],
        "pitch_eg_level": [rec[106], rec[107], rec[108], rec[109]],
        "algorithm": rec[110] & 0x1F,           # 0..31
        "feedback": rec[111] & 0x07,
        "osc_key_sync": (rec[111] >> 3) & 0x01,
        "lfo_speed": rec[112],
        "lfo_delay": rec[113],
        "lfo_pm_depth": rec[114],
        "lfo_am_depth": rec[115],
        "lfo_key_sync": rec[116] & 0x01,
        "lfo_wave": (rec[116] >> 1) & 0x07,
        "p_mode_sens": (rec[116] >> 4) & 0x07,
        "transpose": rec[117],
        "name": bytes(rec[118:128]).decode("ascii", "replace").strip(),
    }
    return {"ops": ops, "global": g}


# NB: "TRIANGE" is Dexed's own misspelling of the enum value — must match it.
_LFO_WAVE = ["TRIANGE", "SAW DOWN", "SAW UP", "SQUARE", "SINE", "S&HOLD"]


def apply_voice_to_dexed(plug, voice: dict):
    """Set Dexed's named VST parameters from a parsed packed voice.

    Dexed exposes each param in its natural DX7 range but with mixed python
    types: floats for numeric params, bool for the *_key_sync switches, and
    string enums for op mode / lfo wave. Coerce to the current value's type.
    """
    g = voice["global"]

    def setp(k, v):
        # Dexed params come in four flavours: plain float; bool (the
        # *_key_sync switches); string enum passed by name (op mode, lfo
        # wave); and *indexed* enum where the raw DX7 int selects from a
        # value list (break_point note names, keyboard curves). getattr
        # returns an opaque wrapper, so branch on the value we pass and use
        # the parameter's valid_values list to place indexed enums.
        if isinstance(v, str):
            setattr(plug, k, v)
            return
        vv = getattr(plug.parameters[k], "valid_values", None)
        if vv:
            idx = max(0, min(len(vv) - 1, int(round(float(v)))))
            setattr(plug, k, vv[idx])
            return
        try:
            setattr(plug, k, float(v))
        except (ValueError, TypeError):
            setattr(plug, k, bool(v))

    setp("algorithm", g["algorithm"] + 1)            # Dexed shows 1..32
    setp("feedback", g["feedback"])
    setp("osc_key_sync", g["osc_key_sync"])
    setp("lfo_speed", g["lfo_speed"])
    setp("lfo_delay", g["lfo_delay"])
    setp("lfo_pm_depth", g["lfo_pm_depth"])
    setp("lfo_am_depth", g["lfo_am_depth"])
    setp("lfo_key_sync", g["lfo_key_sync"])
    setp("lfo_wave", _LFO_WAVE[g["lfo_wave"] % 6])
    setp("p_mode_sens", g["p_mode_sens"])
    setp("transpose", g["transpose"])
    for i, r in enumerate(g["pitch_eg_rate"], 1):
        setp(f"pitch_eg_rate_{i}", r)
    for i, l in enumerate(g["pitch_eg_level"], 1):
        setp(f"pitch_eg_level_{i}", l)
    for op_num, o in voice["ops"].items():
        pre = f"op{op_num}"
        for i in range(4):
            setp(f"{pre}_eg_rate_{i+1}", o["eg_rate"][i])
            setp(f"{pre}_eg_level_{i+1}", o["eg_level"][i])
        setp(f"{pre}_output_level", o["output_level"])
        setp(f"{pre}_mode", "FIXED" if o["mode"] else "RATIO")
        setp(f"{pre}_f_coarse", o["f_coarse"])
        setp(f"{pre}_f_fine", o["f_fine"])
        setp(f"{pre}_osc_detune", o["detune"])
        setp(f"{pre}_break_point", o["break_point"])
        setp(f"{pre}_l_scale_depth", o["l_scale_depth"])
        setp(f"{pre}_r_scale_depth", o["r_scale_depth"])
        setp(f"{pre}_l_key_scale", o["l_curve"])
        setp(f"{pre}_r_key_scale", o["r_curve"])
        setp(f"{pre}_rate_scaling", o["rate_scale"])
        setp(f"{pre}_a_mod_sens", o["amp_mod_sens"])
        setp(f"{pre}_key_velocity", o["key_vel_sens"])
        setp(f"{pre}_switch", True)


def render(bank_path, voice_1based, note, dur, tail, plugin_path, out_path, velocity):
    import pedalboard
    import mido
    raw = Path(bank_path).read_bytes()
    # A 4104-byte bank = 6-byte header + 32*128 + checksum + F7.
    body = raw[6:6 + 32 * 128] if len(raw) >= 6 + 32 * 128 else raw
    rec = body[(voice_1based - 1) * 128: voice_1based * 128]
    voice = parse_packed_voice(rec)
    plug = pedalboard.load_plugin(str(plugin_path))
    apply_voice_to_dexed(plug, voice)
    sr = 48000
    notes = [mido.Message("note_on", note=note, velocity=velocity, time=0.0),
             mido.Message("note_off", note=note, time=dur)]
    audio = np.asarray(
        plug(notes, duration=dur + tail, sample_rate=sr, num_channels=2),
        dtype=np.float32)
    mono = audio.mean(axis=0)
    if out_path:
        import soundfile as sf
        sf.write(str(out_path), mono, sr)
    return voice, mono, sr


def fingerprint(mono, sr, n_harm=10):
    w = mono[int(0.05 * sr):int(0.30 * sr)]
    w = w * np.hanning(len(w))
    mag = np.abs(np.fft.rfft(w, 8 * len(w)))
    frq = np.fft.rfftfreq(8 * len(w), 1.0 / sr)
    band = (frq > 200) & (frq < 320)
    f0 = frq[band][np.argmax(mag[band])]
    ref = mag.max()
    out = {"f0": float(f0)}
    for k in range(1, n_harm + 1):
        b = (frq > k * f0 - 6) & (frq < k * f0 + 6)
        out[f"h{k}"] = float(20 * np.log10(mag[b].max() / ref + 1e-12)) if b.any() else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("bank")
    ap.add_argument("--voice", type=int, required=True, help="1-based voice number")
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--dur", type=float, default=1.5, help="gate-on seconds")
    ap.add_argument("--tail", type=float, default=1.0, help="release tail seconds")
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plugin", default=str(Path.home() / "dx7_ref/dexed/Dexed.vst3"))
    ap.add_argument("--fingerprint", action="store_true")
    args = ap.parse_args()
    voice, mono, sr = render(args.bank, args.voice, args.note, args.dur,
                             args.tail, args.plugin, args.out, args.velocity)
    g = voice["global"]
    print(f"VOICE {args.voice}: '{g['name']}'  alg {g['algorithm']+1}  fb {g['feedback']}")
    print(f"peak={float(np.max(np.abs(mono))):.4f}  rms={float(np.sqrt(np.mean(mono**2))):.5f}")
    if args.out:
        print(f"wrote {args.out}")
    if args.fingerprint:
        fp = fingerprint(mono, sr)
        print(f"f0={fp['f0']:.1f} Hz")
        for k in range(1, 11):
            v = fp.get(f"h{k}")
            print(f"  h{k}: {v:6.1f} dB" if v is not None else f"  h{k}: -")


if __name__ == "__main__":
    main()
