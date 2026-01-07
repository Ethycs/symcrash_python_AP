# TODO List to Integrate System Level

1. Implement Low Rank Rotator on CCS Oscillator
2. Define CockpitEngine Standard
3. WPT to Polyphase filterbank/streaming filterbank


3. CoeffOsc modes (A & B)

You asked earlier “modes A & B of the wavelet generator”; here they become CoeffOsc modes.

3.1 Mode A: Table-Orbit Oscillator (v0 recommended)

Concept: tables store a coefficient “shape” (magnitudes/phases) and the oscillator advances phase to generate a periodic orbit.

Table format (v0)

Per table T:

mag_q15[256] (or [32][8])

phase_stride_q16[256] (optional; default derived from leaf/coeff index)

phase_bias_q16[256] (optional)

norm_rms_q15 (precomputed)

Runtime equation

For each coefficient i:

phi_i += phase_inc * stride_i + phase_jitter_i

coeff_base[i] = (mag[i] * sin_lut(phi_i + bias_i)) >> 15

Where:

phase_inc comes from pitch

stride_i sets harmonic/exotic structure (can be table-defined)

phase_jitter_i optional small PRNG-based perturbation (bounded)

Pros: cheap, deterministic, phase-continuous, FPGA-friendly.
Cons: requires sin LUT and per-coeff phase accumulators (can be compressed, see §3.3).

3.2 Mode B: Low-Rank Rotator (optional v0-lite, great v1)

Represent base coefficients as:

coeff_base = U[:,0]*s0(θ0) + U[:,1]*s1(θ1) + ... (rank 2–4)

Each θk advances with pitch-scaled increments

U lives in ROM (tables)

Pros: far fewer accumulators, expressive.
Cons: needs ROM basis + more MACs per block.

3.3 Phase accumulator budget options

Choose one:

Full: phi[256] per voice (heaviest)

Leaf-shared: phi_leaf[32] plus phi_sub[8] (recommended v0)

phi(i)=phi_leaf[leaf(i)] + phi_sub[j]

2–4 global phases: only a few phases per voice + fixed per-coeff offsets (cheapest)

4. Base coefficient injection policy (how base meets CMMM)

We need a single normative rule so implementations don’t diverge.

4.1 State objects

Per voice keep:

c_state[256] = persistent coefficient state (this is what CMMM evolves)

Per block we generate:

c_base[256] from CoeffOsc

4.2 Mix operator (normative)

Define base_mix_q15 (0..1) per voice per block.

Then:

c_cmd = lerp(c_state, c_base, base_mix)

c_cmd[i] = ((1-base_mix)*c_state[i] + base_mix*c_base[i])

Interpretation:

base_mix=1 means “pure oscillator base”

base_mix=0 means “free-running CMMM state”

In most instruments: use base_mix as a slow stabilizer (prevents drift)

4.3 Retrigger and note transitions

On gate_rise:

If retrigger_policy == RESET_PHASE: reset oscillator phase(s)

Else keep phase continuous.

On pitch change:

apply slewed phase increment (glide) not coefficient crossfade.

This gives you clean “glide vs discrete note” semantics:

Glide: phase_inc slews continuously

Rearticulation: envelope retrigger + optional phase reset

5. Updated wet/dry and ETA semantics (fix combing)

Your combing came from mixing a “carrier audio” with a transformed/residual that is phase-variant per block. This upgrade eliminates that by making one canonical render path.

5.1 New canonical meaning of ETA

ETA = base_mix (coefficient-domain wet/dry)

ETA=1: stick to oscillator orbit

ETA<1: allow CMMM to deviate

If you still want an audio-domain wet/dry, it must be time-aligned and phase-consistent:

dry_audio must be rendered from the same coefficient base with a “null CMMM” path:

audio_dry = IDWPT(TailQ(c_base))

audio_wet = IDWPT(TailQ(CMMM(...)))

out = lerp(audio_dry, audio_wet, eta_audio)
This avoids combing because both share the same rendering chain.

5.2 Remove FWPT-from-carrier from the synthesis path

FWPT is now only used when:

external audio input is being conditioned into coeffs (mimic/analysis)

debugging/visualization

optional “audio-follow” gestures

6. FWPT optional path (kept, but non-blocking)
6.1 Optional “mimic mode” path

If enabled for a voice (or global bus):

coeff_audio = FWPT(windowed_audio_in_block) (conditioning)

Provide coeff_audio as an additive perturbation into c_cmd:

c_cmd = c_cmd + audio_inject_gain * coeff_audio

Must pass through TailQ.

6.2 No dependency guarantee

A valid synthesis frame must render even if FWPT is disabled, missing, or skipped.

7. Tail safety predicate (mandatory tail ordering, frozen)

Tail ops are reserved outside the budget and always executed.

7.1 Tail predicate (what must be true after tail)

After TailQ, coefficients must satisfy:

Bounded amplitude per coefficient

|c[i]| <= C_MAX (Q format bound)

Energy bound per leaf
Let leaf_energy[l] = Σ_{j=0..7} c[l,j]^2

leaf_energy[l] <= E_LEAF_MAX

Global energy bound

Σ leaf_energy[l] <= E_GLOBAL_MAX

Regularity / slew bound (stateful)

|audio_out[n] - audio_out[n-1]| <= SLEW_MAX

This part can live in SafetyTail (audio) or in a coefficient regularity operator if you keep it.

No illegal sparsity explosion

active_leaf_count <= L_ACTIVE_MAX (optional, if you want sparsity)

7.2 Tail operator ordering (normative)

Tail is a deterministic pipeline:

FINE_CLAMP (hard coefficient clamp)

NORMALIZE_SCALE (energy normalization, bounded gain change)

GLUE_PAIR (optional but if enabled, always here)

SHEAF_PROJECT (structural projection)

EMERGENCY_SIMPLIFY (only if predicate would fail; must set telemetry bit)

Rule: tail ordering is not author-controlled.

8. Fixed-point formats (v0 normative)

Choose these and freeze them (matches your ABI habits):

Coefficients: i16 Q1.15 (-32768..32767)

Internal MAC: i32 or i64 accumulation

Squares: i64 for energy sums (avoid overflow)

Gains: u16 Q0.15

base_mix / ETA: u16 Q0.15

Phase: u32 full wrap; sin LUT index uses high bits.

Rounding:

multiplications use round-to-nearest ties-to-even when dropping bits

saturation is symmetric at final i16 clamp

9. VoiceFrame / UART implications (minimal)

You said: two UART ins to FPGA; controls later; deltas or absolutes; treeid primary selector.

This upgrade does not require new wire fields if you repurpose sensibly:

9.1 New required per-voice runtime parameters

table_id (or reuse program_id mapping table)

base_mix_q15 (can be computed from existing w_family or a macro)

osc_mode (A/B)

phase_reset_policy (optional, per program)

Where to put them v0:

if you have a ParamFrame/registry (you planned one), these become registered parameters.

If not, you can temporarily map:

tree_a_id as primary selector → maps to {tree_id, table_id, osc_mode} via program table.

10. Budget tracking (still applies, but easier)

This upgrade reduces total cost because the FWPT path can be skipped on synthesis voices.

Per block cost becomes:

CoeffOsc: cheap (few LUT ops, or a handful of MACs)

CMMM: same

Tail: same (reserved)

IDWPT: same

Scheduler/degrade logic stays the same, but now:

“skip FWPT first” is an automatic win.

11. Anti-click responsibilities (Wavelet vs Safety) — new table (requested)
Concern	Where	Why	Mechanism
Phase continuity across blocks (voice running)	CoeffOsc + Voice	musical timing/state	phase accumulators; glide as phase_inc slew
Note retrigger click	Voice + Safety	musical event, audio boundary	envelope de-click, short ramp, optional phase reset policy
Tree switch click	Wavelet (mechanical)	numeric hygiene	coefficient crossfade between tree A/B (not timeful)
Operator-induced “intentional clicks”	CMMM (non-tail)	effect content	explicit IMPULSE/SHOCK ops; never auto-removed
Unintentional discontinuities	Tail + Safety	guarantees	clamp/normalize + slew limiter + optional OLA
Block stitching	Safety	global audio continuity	output OLA/crossfade if needed

Critical: with CoeffOsc, you should no longer need “wet OLA” hacks to fight combing. Use Safety only for stitching/mix.

12. Migration plan (implementation guide)
12.1 Step-by-step (software prototype)

Implement CoeffOsc with leaf-shared phases (fast v0).

Replace synthesis path:

delete “carrier → FWPT → mask” for standard voices

set c_state ← lerp(c_state, c_base, base_mix)

Keep FWPT path behind a feature flag:

only used for external input voices or mimic mode

Update ETA:

make ETA = base_mix first

optionally add audio-domain wet/dry later using dual-IDWPT (aligned)

Run your click/combing tests:

expect combing to disappear because there is no delayed/misaligned mix.

12.2 FPGA port consequences (good news)

You can build the FPGA voice path without forward transform at all.

FWPT can be instantiated only if you want mimic mode hardware-side.

CoeffOsc is ROM + small phase logic → extremely FPGA-friendly.

13. Acceptance criteria (v0)
Must pass

No steady-state combing when ETA/base_mix is enabled

Pitch stable across blocks (no block-rate AM)

Glide smooth (no discontinuities at sample 32 / block edges)

Bit-exact determinism in the reference pipeline (given same PRNG seeds)

Nice-to-have

Mode B low-rank oscillator

External audio mimic injection into coefficient space

14. What is still a choice (explicitly deferred)

You asked to “defer 1–5 through experience.” Here’s what remains intentionally open:

exact control mapping (pots/encoders/macros)

delta vs absolute update semantics (ParamRegistry will handle)

table authoring workflow (handmade vs learned)

oscillator phase reset policy per instrument

whether w_family_q15[8] mixes tables, operator families, or both

If you want one concrete default (so you can code immediately)

Use this v0 default:

CoeffOsc Mode A

leaf-shared phases: phi_leaf[32] + phi_sub[8]

ETA := base_mix_q15

c_cmd = lerp(c_state, c_base, base_mix)

Tail always applied

Single render path only (no carrier audio mixing)

That’s the “minimum upgrade” that removes the combing and makes forward optional.

If you want, paste your current lanes.py/render loop structure and I’ll map these exact steps onto the functions you already have (where to store c_state, where to plug CoeffOsc.next(), and where to move ETA).