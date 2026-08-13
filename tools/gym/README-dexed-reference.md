# Dexed reference renderer (WS5 control)

`dexed_reference.py` renders a DX7 voice through the **real Dexed (msfa)
engine**, giving the reversal gym a bit-accurate reference *independent of
our own engine*. Without it the gym is self-referential (our engine scored
against its own import) and technique effectiveness is unmeasurable; with it,
every optimization technique is scored against a fixed yardstick.

## Why this exists / what it proved

- Headless VST3 hosting of Dexed **works** (WSL + pedalboard) — this was the
  handoff's "one genuine unknown."
- Per-voice loading works via **direct VST parameter setting** (parse the
  packed VMEM voice, set Dexed's named params). pedalboard does **not** forward
  SysEx MIDI-thru, so the bank-SysEx + program-change path silently renders the
  default patch — do not use it.
- First fidelity finding (HARP 1, rom1b v29, alg 3 / fb 7): even harmonics
  match our engine to <0.2 dB, but the **odd harmonics diverge 11–20 dB** —
  real Dexed h3 is louder than h2; our engine suppresses h3/h5. The gap tracks
  the max-feedback op6, i.e. our feedback approximation. This is the
  fidelity-contract compromise made measurable (WS1c: "engine bugs for free").

## Environment (WSL + Pixi)

Runs in WSL (Ubuntu-24.04) because the Linux Dexed VST3 hosts cleanly headless.
Python and all host libraries are locked in the repository's `dexed` Pixi
environment. The Dexed binary remains an external, gitignored reference asset.

```bash
# one-time: Dexed Linux build (binary, gitignored like the .syx banks)
mkdir -p ~/dx7_ref && cd ~/dx7_ref
curl -sSL https://github.com/asb2m10/dexed/releases/download/v1.0.1/Dexed-1.0.1-lnx.zip -o dexed-lnx.zip
mkdir -p dexed && (cd dexed && unzip -o -q ../dexed-lnx.zip)
# runtime libs the JUCE plugin needs (sudo):
sudo apt-get install -y libasound2t64 libfreetype6 libx11-6 libxext6 \
    libxrandr2 libxinerama1 libxcursor1 libgl1 libcurl4
# install the locked host environment (run from the repo in WSL)
pixi install --environment dexed --frozen
```

Render:

```bash
REPO=/mnt/f/Keytone/Documents/GitHub/symcrash_python_AP
pixi run --manifest-path "$REPO/pyproject.toml" --environment dexed --frozen \
  python "$REPO/tools/gym/dexed_reference.py" "$REPO/ingest/dx7_banks/rom1b.syx" \
    --voice 29 --note 60 --dur 1.5 --tail 1.0 --out ref.wav --fingerprint
```

## Automated core validation

From Windows, `pixi run dx7-core-refine` renders the eight core ROM voices
through this environment, sweeps the declared Symcrash parameter grid, and
uses the process-isolated perceptual committee as a hard acceptance gate.
References, candidates, audition WAVs, trajectories, and the summary report
land under `audio_out/dx7_core_refine/`; source presets are never overwritten
by the batch driver.
