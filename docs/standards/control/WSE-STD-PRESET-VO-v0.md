## WSE-STD-PRESET-VO v0 (practical schema)

**Design choice:** VO presets are “compiler-friendly, human-readable patch sources” that compile into your **VoiceFrameEmitter rules** (block-rate frames + event_mask logic). The VM still only sees VoiceFrames.

**Important compatibility note (COMPRESS_K):** the classic docs use `COMPRESS_K` heavily (filter “reso proxy”).
Your current 24-op table you pasted does **not** include `compress_k` by name. So these presets include:

* `op: compress_k` with `fallback_op_id: 20` (**foldpressure**) as a safe default “bite/reso” substitute
* If you later add a real COMPRESS_K op, just update the alias table once.

---

## Preset pack (12 VO presets) — YAML

```yaml
wse_preset_std: "WSE-STD-PRESET-VO v0"
voiceframe_std: "WSE-STD-020 v1"
vm_std: "WSE-STD-VM-HYBRID v0"

# Global operator aliases for compilation (so docs remain canonical even if names drift)
op_aliases:
  compress_k:
    preferred_op_id: null          # set when you add a real COMPRESS_K
    fallback_op_id: 20             # foldpressure (current library)
    note: "Legacy reso/bite proxy used in Classic Presets docs"

# Event program semantics (compiler):
# - note_on: fires for first N blocks after note-on
# - periodic: fires when (block_index % period == phase)
# - early_decay: fires only while env.stage in {ATTACK, DECAY} and within max_blocks
event_program_semantics: "deterministic_block_rate"

presets:

  - preset_id: "classic.saw_lead.v0"
    name: "Saw Lead (Supersaw / Classic Lead)"
    carrier:
      wave: "saw"
      partials_N: 12
    defaults:
      pattern_id: 0        # K=32
      tail_mode: "FULL"
      gain_q15: 30000
      pan_q15: 16384
      env:
        attack_blocks: 2
        decay_blocks: 40
        sustain_level_q15: 25000
        release_blocks: 60
    backbone:
      - { op: "tilt",         op_id: 1,  strength_q15: 18022 }  # 0.55
      - { op: "compress_k",   fallback_op_id: 20, strength_q15: 6553 }   # 0.20
      - { op: "lowrankmorph", op_id: 7,  strength_q15: 14745 }  # 0.45
      - { op: "quasirotate",  op_id: 17, strength_q15: 8192 }   # 0.25
      - { op: "diffuseup",    op_id: 4,  strength_q15: 3277 }   # 0.10
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }# 0.80
    events:
      - trigger: "periodic"
        lane: 4
        op: "rotatenodes"
        op_id: 10
        period_blocks: 8
        phase_blocks: 0
        strength_q15: 6553   # 0.20
        target: { kind: "pattern_id", pattern_id: 0 }
      - trigger: "periodic"
        lane: 1
        op: "globalpulse"
        op_id: 8
        period_blocks: 32
        phase_blocks: 0
        strength_q15: 4915   # 0.15
        target: { kind: "pattern_id", pattern_id: 3 }

  - preset_id: "classic.square_bass.v0"
    name: "Square Bass (Subtractive Bass)"
    carrier:
      wave: "square"
      partials_N: 8
    defaults:
      pattern_id: 2          # K=16 (often enough for bass)
      tail_mode: "FULL"
      gain_q15: 29000
      env: { attack_blocks: 2, decay_blocks: 25, sustain_level_q15: 28000, release_blocks: 30 }
    backbone:
      - { op: "tilt",           op_id: 1,  strength_q15: 8192 }   # 0.25
      - { op: "diffusedown",    op_id: 5,  strength_q15: 8192 }   # 0.25
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
      - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
    events:
      - trigger: "periodic"
        lane: 0
        op: "impulse"
        op_id: 13
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 8192   # 0.25
        enabled: false       # optional accent
        target: { kind: "pattern_id", pattern_id: 6 }   # K=1 near f0 leaf (compiler decides)

  - preset_id: "classic.sync_lead.v0"
    name: "Sync Lead (Hard Sync Archetype)"
    carrier:
      wave: "saw"
      partials_N: 12
    defaults:
      pattern_id: 0
      tail_mode: "FULL"
      gain_q15: 30000
      env: { attack_blocks: 1, decay_blocks: 60, sustain_level_q15: 22000, release_blocks: 50 }
    backbone:
      - { op: "bandshear",      op_id: 6,  strength_q15: 14745 }  # 0.45
      - { op: "rotatenodes",    op_id: 10, strength_q15: 9830 }   # 0.30
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 9830 }  # 0.30
      - { op: "tilt",           op_id: 1,  strength_q15: 18022 }  # 0.55
      - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
    events:
      - trigger: "periodic"
        lane: 3
        op: "swapbands"
        op_id: 9
        period_blocks: 8
        phase_blocks: 0
        strength_q15: 11469   # 0.35
        target: { kind: "pattern_id", pattern_id: 0 }
      - trigger: "periodic"
        lane: 1
        op: "globalpulse"
        op_id: 8
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 6553    # 0.20
        target: { kind: "pattern_id", pattern_id: 0 }

  - preset_id: "classic.brass_stab.v0"
    name: "Brass Stab (Poly Subtractive Brass)"
    carrier:
      wave: "saw"
      partials_N: 10
    defaults:
      pattern_id: 0
      tail_mode: "FULL"
      gain_q15: 31000
      env: { attack_blocks: 2, decay_blocks: 90, sustain_level_q15: 16000, release_blocks: 75 }
    backbone:
      - { op: "sheafproject",   op_id: 23, strength_q15: 13107 }  # 0.40
      - { op: "tilt",           op_id: 1,  strength_q15: 14745 }  # 0.45
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 9830 }  # 0.30
      - { op: "intramix",       op_id: 3,  strength_q15: 8192 }   # 0.25
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
    events:
      - trigger: "note_on"
        lane: 1
        op: "globalpulse"
        op_id: 8
        active_blocks: 2
        strength_q15: 11469   # 0.35
        target: { kind: "pattern_id", pattern_id: 3 }
      - trigger: "note_on"
        lane: 0
        op: "impulse"
        op_id: 13
        active_blocks: 1
        strength_q15: 8192    # 0.25
        target: { kind: "pattern_id", pattern_id: 6 }
    notes:
      - "Classic recipe: fast attack + bark via operators; amp/brightness env lives in VoiceEngine." # see docs

  - preset_id: "classic.string_pad.v0"
    name: "String Pad (Analog Ensemble Pad)"
    carrier:
      wave: "saw"
      partials_N: 12
    defaults:
      pattern_id: 0
      tail_mode: "FULL"
      gain_q15: 27000
      env: { attack_blocks: 120, decay_blocks: 220, sustain_level_q15: 28000, release_blocks: 240 }
    backbone:
      - { op: "lowrankmorph",   op_id: 7,  strength_q15: 18022 }  # 0.55
      - { op: "quasirotate",    op_id: 17, strength_q15: 11469 }  # 0.35
      - { op: "diffuseup",      op_id: 4,  strength_q15: 4915 }   # 0.15
      - { op: "sheafproject",   op_id: 23, strength_q15: 6553 }   # 0.20
      - { op: "tilt",           op_id: 1,  strength_q15: 13107 }  # 0.40
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
    events:
      - trigger: "periodic"
        lane: 4
        op: "rotatenodes"
        op_id: 10
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 6553   # 0.20
        target: { kind: "pattern_id", pattern_id: 0 }
      - trigger: "periodic"
        lane: 7
        op: "gluepair"
        op_id: 12
        period_blocks: 64
        phase_blocks: 0
        strength_q15: 4915   # 0.15
        target: { kind: "pattern_id", pattern_id: 0 }

  - preset_id: "classic.fm_bell.v0"
    name: "FM Bell (2–3 Operator FM)"
    carrier:
      wave: "sine"
      partials_N: 3
    defaults:
      pattern_id: 2
      tail_mode: "FULL"
      gain_q15: 28000
      env: { attack_blocks: 1, decay_blocks: 140, sustain_level_q15: 0, release_blocks: 120 }
    backbone:
      - { op: "sheafproject",   op_id: 23, strength_q15: 18022 }  # 0.55
      - { op: "bandshear",      op_id: 6,  strength_q15: 11469 }  # 0.35
      - { op: "quasirotate",    op_id: 17, strength_q15: 8192 }   # 0.25
      - { op: "tilt",           op_id: 1,  strength_q15: 16384 }  # 0.50
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 8192 }  # 0.25
      - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
    events:
      - trigger: "note_on"
        lane: 0
        op: "impulse"
        op_id: 13
        active_blocks: 1
        strength_q15: 14745  # 0.45
        target: { kind: "pattern_id", pattern_id: 6 }
      - trigger: "early_decay"
        lane: 1
        op: "globalpulse"
        op_id: 8
        period_blocks: 12
        max_blocks: 60
        strength_q15: 6553   # 0.20
        target: { kind: "pattern_id", pattern_id: 3 }

  - preset_id: "classic.jump_brass.v0"
    name: "Jump Brass (Bright Unison Brass)"
    carrier:
      wave: "saw"
      partials_N: 12
    defaults:
      pattern_id: 0
      tail_mode: "FULL"
      gain_q15: 31500
      env: { attack_blocks: 2, decay_blocks: 120, sustain_level_q15: 14000, release_blocks: 90 }
    backbone:
      - { op: "lowrankmorph",   op_id: 7,  strength_q15: 14745 }  # 0.45
      - { op: "sheafproject",   op_id: 23, strength_q15: 11469 }  # 0.35
      - { op: "tilt",           op_id: 1,  strength_q15: 19660 }  # 0.60
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
      - { op: "intramix",       op_id: 3,  strength_q15: 6553 }   # 0.20
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
    events:
      - trigger: "note_on"
        lane: 1
        op: "globalpulse"
        op_id: 8
        active_blocks: 2
        strength_q15: 9830   # 0.30
        target: { kind: "pattern_id", pattern_id: 3 }
      - trigger: "periodic"
        lane: 4
        op: "rotatenodes"
        op_id: 10
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 4915   # 0.15
        target: { kind: "pattern_id", pattern_id: 0 }

  - preset_id: "classic.sub_bass.v0"
    name: "Sub Bass (Pure Fundamental)"
    carrier:
      wave: "sine"
      partials_N: 1
    defaults:
      pattern_id: 6          # K=1 is fine here
      tail_mode: "FULL"
      gain_q15: 30000
      env: { attack_blocks: 3, decay_blocks: 30, sustain_level_q15: 31000, release_blocks: 60 }
    backbone:
      - { op: "sheafproject",   op_id: 23, strength_q15: 24575 }  # 0.75
      - { op: "diffusedown",    op_id: 5,  strength_q15: 11469 }  # 0.35
      - { op: "tilt",           op_id: 1,  strength_q15: 3277 }   # 0.10
      - { op: "normalizescale", op_id: 11, strength_q15: 27852 }  # 0.85
    events:
      - trigger: "note_on"
        lane: 0
        op: "impulse"
        op_id: 13
        active_blocks: 1
        strength_q15: 3277   # 0.10
        enabled: false
        target: { kind: "pattern_id", pattern_id: 6 }

  - preset_id: "classic.pwm_pad.v0"
    name: "PWM Pad (Pulse Modulation Pad)"
    carrier:
      wave: "pulse"
      partials_N: 10
    defaults:
      pattern_id: 0
      tail_mode: "FULL"
      gain_q15: 27000
      env: { attack_blocks: 80, decay_blocks: 200, sustain_level_q15: 28000, release_blocks: 240 }
    backbone:
      - { op: "quasirotate",    op_id: 17, strength_q15: 14745 }  # 0.45
      - { op: "lowrankmorph",   op_id: 7,  strength_q15: 11469 }  # 0.35
      - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
      - { op: "tilt",           op_id: 1,  strength_q15: 13107 }  # 0.40
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
    events:
      - trigger: "periodic"
        lane: 4
        op: "rotatenodes"
        op_id: 10
        period_blocks: 8
        phase_blocks: 0
        strength_q15: 5898   # 0.18
        target: { kind: "pattern_id", pattern_id: 0 }
      - trigger: "periodic"
        lane: 5
        op: "bandshear"
        op_id: 6
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 3277   # 0.10
        enabled: false
        target: { kind: "pattern_id", pattern_id: 0 }

  - preset_id: "classic.pluck.v0"
    name: "Pluck (Resonant Decay Pluck)"
    carrier:
      wave: "saw"
      partials_N: 8
    defaults:
      pattern_id: 2
      tail_mode: "FULL"
      gain_q15: 30000
      env: { attack_blocks: 1, decay_blocks: 120, sustain_level_q15: 0, release_blocks: 90 }
    backbone:
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 18022 } # 0.55
      - { op: "tilt",           op_id: 1,  strength_q15: 18022 }  # 0.55
      - { op: "sheafproject",   op_id: 23, strength_q15: 11469 }  # 0.35
      - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
    events:
      - trigger: "note_on"
        lane: 0
        op: "impulse"
        op_id: 13
        active_blocks: 1
        strength_q15: 18022  # 0.55
        target: { kind: "pattern_id", pattern_id: 6 }
      - trigger: "note_on"
        lane: 2
        op: "gatemask"
        op_id: 16
        active_blocks: 16
        strength_q15: 8192   # 0.25
        target: { kind: "pattern_id", pattern_id: 3 }

  - preset_id: "classic.wobble_bass.v0"
    name: "Wobble Bass (LFO-Driven Filter Bass)"
    carrier:
      wave: "square"
      partials_N: 6
    defaults:
      pattern_id: 2
      tail_mode: "FULL"
      gain_q15: 31000
      env: { attack_blocks: 2, decay_blocks: 60, sustain_level_q15: 30000, release_blocks: 70 }
    backbone:
      - { op: "diffusedown",    op_id: 5,  strength_q15: 8192 }   # 0.25
      - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
      - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
      - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
      # wobble: tilt is modulated (compiler generates per-block strength update)
      - { op: "tilt",           op_id: 1,  strength_q15: 6553, mod: { source: "lfo0", add_q15: 11469 } }
    lfos:
      - lfo_id: "lfo0"
        shape: "sine_q15"
        period_blocks: 96      # ~0.128s at 750 blocks/s (edit later)
        phase_q15: 0
    events:
      - trigger: "periodic"
        lane: 1
        op: "globalpulse"
        op_id: 8
        period_blocks: 16
        phase_blocks: 0
        strength_q15: 4915   # 0.15
        enabled: false
        target: { kind: "pattern_id", pattern_id: 3 }

  - preset_id: "classic.theremin_lead.v0"
    name: "Theremin Lead (Continuous Pitch Lead)"
    carrier:
      wave: "sine"
      partials_N: 2
    defaults:
      pattern_id: 6
      tail_mode: "FULL"
      gain_q15: 26000
      env: { attack_blocks: 40, decay_blocks: 0, sustain_level_q15: 26000, release_blocks: 80 }
      glide:
        mode: "portamento"
        glide_blocks: 60
    backbone:
      - { op: "sheafproject",   op_id: 23, strength_q15: 19660 }  # 0.60
      - { op: "tilt",           op_id: 1,  strength_q15: 11469 }  # 0.35
      - { op: "quasirotate",    op_id: 17, strength_q15: 3277 }   # 0.10
      - { op: "normalizescale", op_id: 11, strength_q15: 27852 }  # 0.85
    events: []
```

### Where these come from (so you can trust them)

* The 12-patch “basis set” and lane meanings are in your classic preset notes / stub, including explicit backbone strengths and event cadences.
* Brass stab / string pad / FM bell / PWM pad / pluck / wobble bass / theremin lead are all spelled out with backbone + events in the same sources.

If you want, I can convert this pack into **your existing `brass_stab.json` style** (JSON output) *and* generate a tiny `PresetCompiler` stub that turns this YAML into: `(base_backbone_ops, strengths, pattern_id, tail_mode, event rules)` for your `VoiceFrameEmitter`.
