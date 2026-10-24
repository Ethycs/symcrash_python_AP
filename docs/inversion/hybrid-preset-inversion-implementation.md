# Hybrid Preset Inversion Implementation

**Status:** Proposed implementation specification

**Date:** 2026-08-14

**Theory:** [[../../00 - Theory/10 - Configuration-State Duality/00 - Configuration to State and Back]]

**Related:** [[../../00 - Theory/09 - Authoring Theory/02 - Optimization Geometry]], [[../../00 - Theory/09 - Authoring Theory/03 - Agentic Patch Analysis]], [[../../07 - Status Reports/AUTOMATED-REVERSAL-2026-08-13]], `WSE-STD-PRESET-AUTHORING` chapters 25–26

---

## 0. Outcome

Build an offline system that accepts one or more labelled reference recordings and returns a diverse, ranked set of legal Symcrash presets. It learns the discrete structure and continuous parameter distribution separately but conditionally, renders every finalist through the exact VM, runs the existing block-coordinate refiner as the final local step, and never writes into the factory preset library without an explicit promotion action.

The same render-and-measure infrastructure will audit the eleven `/system/*` controls and fit perceptually uniform, pleasant response curves. Dial profiles remain workbench artifacts in the first implementation; they do not change the frozen macro, capsule, VoiceFrame, or FPGA ABI until their behavior has been validated.

The intended end-to-end path is:

```text
reference probe set
    -> audio/set encoder
    -> masked discrete structure generator
    -> conditional continuous flow
    -> ConfigCodec decode + validity projection
    -> exact compiler and VM render
    -> multi-probe perceptual committee
    -> top-K diversity selection
    -> existing local refiner
    -> audition package
    -> explicit human promotion
```

No design decision blocks implementation. The choices below are defaults that can be replaced behind versioned interfaces if evidence disproves them.

## 1. Decisions fixed by this document

1. **`InstrumentIR` remains the semantic source of truth.** YAML is the tracked authoring representation; `ConfigCodec` is a model view of `InstrumentIR`, not a new preset system.
2. **The model never generates YAML or Python source.** It emits typed categorical tokens and bounded numerical fields.
3. **Structure and values remain distinct.** A masked discrete model proposes the stratum; a conditional continuous model fills the active parameter fiber.
4. **The inverse returns top-$K$, not one answer.** Non-identifiability is represented rather than averaged away.
5. **The exact compiler and VM are authoritative.** Surrogates may accelerate learning but cannot accept or promote a patch.
6. **Training and evaluation are multi-probe.** A lone WAV is supported as a low-confidence convenience input, never as a sufficient instrument test.
7. **Symmetry and dormant fields are explicit metadata.** Losses do not punish legal operator relabellings or inactive parameters.
8. **Model training is isolated in a Pixi environment.** The default synth/runtime environment does not acquire a mandatory ML dependency; no manual `pip` workflow is introduced.
9. **Generated data and models do not enter `presets/`.** Dataset/cache/checkpoint artifacts live under `_inversion_data/`; audition runs live under `audio_out/inversion/`; only reviewed YAML is promoted to `presets/instruments/`.
10. **Reward fine-tuning waits for metric calibration.** Paired synthetic pretraining and exact-render evaluation can begin immediately. An uncalibrated committee may shortlist but may not become an autonomous reward oracle.
11. **Dial optimization initially changes no runtime ABI.** It produces workbench-consumable profile files. A later standard may compile accepted profiles for hardware after their representation and memory cost are measured.
12. **Existing tools are reused rather than forked.** `render_preset_note`, `WorkbenchSession`, `timbre_distance`, and `auto_refine` gain shared library surfaces where necessary.

## 2. Scope and non-goals

### 2.1 In scope

- Typed configuration encoding and decoding.
- Canonicalization, active masks, legal symmetry metadata, and validity projection.
- Reproducible multi-note/multi-velocity probe rendering.
- Synthetic paired-dataset generation through the canonical VM.
- External target ingestion for DX7/Dexed, NSynth, and recorded references.
- A baseline inverse model followed by masked discrete diffusion and conditional flow matching.
- Exact-render candidate scoring, diversity selection, and local refinement.
- Dial sweep auditing and perceptual arc-length remapping.
- Run manifests, provenance, reproducibility, tests, and human promotion packages.

### 2.2 Out of scope for v0

- Running a neural model on the FPGA or in the realtime audio thread.
- Generating arbitrary executable operators or bypassing the operator registry.
- End-to-end waveform cloning independent of the Symcrash architecture.
- Automatic factory-preset promotion.
- Training directly from unlabelled NSynth audio as if it contained Symcrash parameter labels.
- Changing the macro, parameter-register, capsule, or VoiceFrame ABI for dial curves.
- Claiming a universal perceptual JND before a human preference corpus calibrates it.

## 3. Existing components and required adaptations

| Existing component | Reuse | Required adaptation |
|---|---|---|
| `instrument_ir.py` | Canonical typed preset object and loader | Expose stable field introspection; do not duplicate its schema |
| `library/compiler.py` / `preset_compiler.py` | Final legality and compilation gates | Return structured diagnostics suitable for dataset rejection and generation masks |
| `symcrash_workbench.params` | Stable addresses, domains, log flags, get/set codecs for exposed continuous fields | Factor generic field metadata out of UI-only assumptions |
| `WorkbenchSession` | Comment-preserving mutation, version bump, render, and JSONL trajectory | Add non-interactive transaction API and multi-probe batch render |
| `render_preset_note` | Exact canonical VM render | Add a side-effect-free render request wrapper with explicit seed, gate, tail, and control trajectory |
| `timbre_distance.py` | Current spectral/envelope/centroid/Zimtohrli committee | Factor `ProbeSuiteScore`; load versioned weights rather than hard-code bootstrap weights |
| `auto_refine.py` | Exact-render fixed-topology local improvement | Accept a probe manifest and candidate object; preserve the current one-WAV CLI |
| DX7 importer + Dexed renderer | Labelled structure source and independent reference renderer | Batch probe-manifest generation and render fingerprints |
| `/system/*` + `SystemAxisRuntime` | Live semantic control surface and frozen slew laws | Accept optional workbench-only dial remap profiles |
| Preset compiler/build | Final candidate conformance | Add an explicit promotion helper; never promote during inference |

The refactor rule is strict: shared behavior moves into importable modules, while current command-line interfaces continue to work.

## 4. Proposed source layout

```text
src/symcrash_inversion/
    __init__.py
    schema.py              FieldSpec, StructureSpec, schema versions
    codec.py               InstrumentIR <-> EncodedConfig
    canonical.py           active masks, graph automorphisms, gauges
    probes.py              ProbeSpec, ProbeSet, target manifests
    render_service.py      deterministic exact-render adapter
    dataset.py             manifests, shards, splits, sampling
    features.py            shared audio preprocessing
    scoring.py             multi-probe committee and Pareto gates
    candidates.py          candidate records, diversity, provenance
    inference.py           model sampling -> compile -> render -> refine
    dial_profiles.py       sweep audit and arc-length remapping
    promotion.py           explicit reviewed candidate promotion
    models/
        encoder.py         per-probe audio encoder + set aggregation
        discrete.py        masked categorical diffusion
        continuous.py      conditional flow matching
        baseline.py        small deterministic/mixture baseline
    training/
        generate.py
        train_encoder.py
        train_inverse.py
        exact_reward.py
        evaluate.py

tools/inversion/
    inspect_schema.py
    build_dataset.py
    train.py
    invert.py
    evaluate.py
    audit_dials.py
    fit_dials.py
    promote.py

configs/inversion/
    smoke-fm6-p0-v0.yaml
    fm6-p0-v0.yaml
    committee-bootstrap-v0.yaml

tests/inversion/
    test_codec.py
    test_canonical.py
    test_probes.py
    test_dataset.py
    test_model_shapes.py
    test_tiny_overfit.py
    test_inference.py
    test_dial_profiles.py
```

Tools remain thin argument parsers. Reusable logic belongs in `src/symcrash_inversion`.

## 5. Configuration codec

### 5.1 Public types

```python
class FieldKind(Enum):
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BOUNDED_LINEAR = "bounded_linear"
    BOUNDED_LOG = "bounded_log"
    BIPOLAR = "bipolar"

@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: FieldKind
    domain: tuple[float, float] | tuple[str, ...]
    group: str
    activation: str
    quantizer: str
    symmetry_role: str | None = None

@dataclass
class EncodedConfig:
    schema: str
    structure_tokens: list[int]
    continuous_values: list[float]
    active_mask: list[bool]
    field_keys: list[str]
    template_family: str
    opaque_digest: str | None
    provenance: dict
```

`FieldSpec.key` is a stable semantic key, not a YAML line or arbitrary list index. Existing workbench addresses are reused where their meaning is already stable, for example `/fm/op1/ratio` and `/system/brightness`. Structural keys use the same namespace style, for example `/structure/engine`, `/structure/fm/algorithm`, and `/structure/regime/count`.

### 5.2 Encoding layers

`ConfigCodec` has two representations:

1. **Canonical typed tree:** a normalized view of the complete `InstrumentIR`, retaining fields not yet used by a model.
2. **Model view:** fixed or padded categorical and numerical arrays plus masks for one declared conformance profile.

This prevents an early FM6 model from silently deleting granular, wavetable, CMMM, or future fields. When a source preset contains unsupported data, the codec either:

- preserves it as immutable template context while modifying supported fields; or
- rejects generation-from-scratch with a structured `unsupported_feature` diagnostic.

It never drops the field.

### 5.3 Initial schema: `symcrash.config-vector.fm6-p0.v0`

The first trainable schema targets the core FM6 presets.

Discrete fields:

- FM6 algorithm 1–32;
- feedback 0–7, initially categorical with an ordinal diagnostic;
- presence of Amp, Filter, noise exciter, and unison;
- noise color;
- number and targets of envelope bindings;
- presence of authored attack/stable emphasis families.

Continuous fields:

- six operator ratios, encoded in log-frequency space;
- six operator levels;
- global modulation index and post-filter cutoff;
- Amp ADSR, velocity sensitivity, and key-scale rate;
- every active operator-envelope binding, ordered by stable target ID;
- filter cutoff, resonance, and envelope amount when active;
- noise level/attack/decay when active;
- unison spread and width when active;
- attack/stable seven-family emphasis;
- eleven system-manifold defaults;
- seven temperament values if the preset uses CMMM behavior.

Discrete integer values such as unison voice count remain categorical or ordinal. They are not passed through a continuous output and rounded as an afterthought.

### 5.4 Normalization and quantization

- Positive time and frequency fields use bounded log normalization.
- Ratios use bounded $\log_2$ normalization; an auxiliary cents residual may be added after the baseline.
- Q15 fields normalize to $[0,1]$ but always decode through the engine's canonical rounding and clamping functions.
- Bipolar fields normalize to $[-1,1]$.
- Enum IDs remain categorical.
- The codec retains both normalized model values and decoded engine integers in debug reports.

Every decoded candidate is re-encoded. A codec round-trip passes only if categorical values and active masks are equal and continuous values differ by no more than one engine quantization step.

### 5.5 Active masks

An active mask is derived from structure, not learned independently. Examples:

- filter values are inactive if there is no Filter node;
- a noise color is inactive at zero noise level for audio loss but remains present for editable intent;
- operator-envelope parameters are inactive if no binding targets that operator;
- a regime template is inactive if no reachable regime emits it;
- fields owned by a disabled operator are excluded from parameter loss.

Two masks are useful:

- `semantic_mask`: field exists as authored intent;
- `audible_mask`: field can affect the selected probe set.

Training parameter loss uses the semantic mask. Identifiability reports and candidate simplification use the audible mask.

### 5.6 Symmetry handling

For each canonical FM algorithm, compute graph automorphisms that preserve:

- carrier/modulator role;
- directed modulation edges;
- feedback ownership;
- active envelope and routing types.

Six operators permit an exhaustive offline permutation check, but only graph-preserving permutations belong to the symmetry group. The canonical representative is the lexicographically smallest field signature over that group. Each claimed automorphism receives a bit-exact render-difference test across the standard probe set; a non-zero difference removes it from the legal group.

Training uses both:

- canonical representatives for stable storage and splitting;
- random legal symmetry augmentation or orbit-minimized losses so the model does not overfit an arbitrary label.

Gain tradeoffs are not assumed to be exact symmetries. Scores are loudness-normalized, but source levels are preserved unless a bit-exact or perceptually calibrated equivalence test proves a safe gauge transform.

## 6. Probe and render contract

### 6.1 Probe manifest

External targets and evaluation suites use a small YAML manifest:

```yaml
schema: symcrash.probe-set.v0
instrument_id: marimba_ref
sample_rate_hz: 48000
probes:
  - id: n48_v035
    audio: n48_v035.wav
    note: 48
    velocity: 0.35
    gate_s: 1.5
    tail_s: 1.0
    seed: 0
  - id: n60_v070
    audio: n60_v070.wav
    note: 60
    velocity: 0.70
    gate_s: 1.5
    tail_s: 1.0
    seed: 0
```

Unknown metadata is preserved. Required fields are audio, note, velocity, gate, tail, and seed. A legacy single WAV is wrapped into a one-probe manifest by CLI arguments and marked `identification_confidence: low`.

### 6.2 Standard evaluation suite

Factory candidates use the Cartesian set:

- notes: 48, 60, 72;
- velocities: 0.35, 0.70, 1.00;
- gate: 1.5 seconds;
- captured tail: 1.0 second;
- deterministic seed: 0.

This nine-probe suite is the release gate, not necessarily the training batch. Dataset records store individual probes keyed by canonical configuration hash; the loader samples one to four probes per training example and can assemble the full set for validation.

Specialized instruments may add probes but may not remove the standard suite without recording a reason. CMMM patches add at least one standardized system-control trajectory; stochastic patches add fixed seeds 0–3 and are scored on both mean and worst case.

### 6.3 Exact render request

```python
@dataclass(frozen=True)
class RenderRequest:
    preset: InstrumentIR
    note: int
    velocity: float
    gate_s: float
    tail_s: float
    seed: int
    system_trajectory: tuple[ControlPoint, ...] = ()

@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate_hz: int
    render_fingerprint: dict
    diagnostics: dict
```

The adapter must be side-effect free and quiet by default. It calls the same `IRFrameBuilder` and `vm_step_block` path as the current renderer. `render_fingerprint` contains at least Git revision when available, schema versions, sample rate, block size, seed, renderer implementation ID, and relevant compiler/VM version identifiers.

The dataset generator refuses to append to a shard when its render fingerprint differs from the shard fingerprint.

## 7. Storage and dataset design

### 7.1 Paths

```text
_inversion_data/
    datasets/<dataset-id>/
        dataset.json
        train.jsonl
        validation.jsonl
        test.jsonl
        shards/
    checkpoints/<run-id>/
    feature_cache/<feature-version>/
    external/nsynth/

audio_out/inversion/<run-id>/
    request.yaml
    candidates/
    renders/
    scores.json
    shortlist.html
    trajectory/
    run.json
```

Both roots are generated/ignored. `presets/` remains the only preset source root. A run may contain YAML candidates for audition, but they do not become library sources until promotion.

### 7.2 Record schema

Each JSONL record contains:

```json
{
  "schema": "symcrash.inversion-example.v0",
  "example_id": "sha256:...",
  "config_hash": "sha256:...",
  "parent_hash": "sha256:...",
  "config": {"schema": "symcrash.config-vector.fm6-p0.v0"},
  "probe": {"note": 60, "velocity": 0.7, "gate_s": 1.5, "tail_s": 1.0, "seed": 0},
  "audio_member": "shard-00017.tar/example.flac",
  "feature_member": "shard-00017.tar/example.f16.npy",
  "render_fingerprint": {},
  "source": {"kind": "synthetic", "family": "library-perturbation"},
  "diagnostics": {}
}
```

Audio and optional float16 features are stored in immutable, checksummed shards. JSONL remains the searchable index. Shards are replaced atomically and never edited in place.

### 7.3 Dataset sources

The initial mixture is:

1. **Library perturbations (40%):** legal local variation around shipped and imported presets. These densely cover musically useful regions.
2. **Motif grammar samples (40%):** generator, articulation, color, motion, and event motifs composed under Tower legality.
3. **Broad legal samples (20%):** coverage of underrepresented valid configurations, with rejection filters for silence and pathological saturation.

These are starting weights, recorded in `dataset.json`, not constants in code. Later dataset versions may rebalance based on coverage reports.

The first useful corpus target is 50,000 distinct configurations with three probes each. Before spending that render budget, a 1,000-configuration smoke corpus must pass every integrity and tiny-overfit test. Scale is increased only when held-out performance remains data-limited.

### 7.4 Sampling validity

The generator samples through constructors and registry domains, not by mutating arbitrary YAML. Every sample must:

1. decode to `InstrumentIR`;
2. pass structural validation;
3. compile successfully;
4. render all requested probes deterministically;
5. pass hard audio safety predicates;
6. produce a matching encode/decode round-trip.

Silence, sustained limiter engagement, NaN/overflow, invalid pitch, and render nondeterminism are rejection reasons with counters. Rejected examples are retained only as diagnostics, never silently discarded.

### 7.5 Split policy

All perturbations of a source parent and all legal symmetry variants share one split. Split assignment hashes the canonical parent/config family, not the rendered clip. This prevents nearly identical parameter neighbors from leaking between training and test.

The held-out test set includes:

- unseen synthetic configuration families;
- held-out DX7 voices;
- NSynth instruments unavailable during supervised training.

The marimba and harpsichord are product regression targets rather than claims of unseen generalization: their scores and verdicts are tracked across runs, but they are expected to influence development decisions.

## 8. Audio representation

The preprocessing contract is versioned independently from model code.

Per probe:

1. convert to mono for the v0 timbre model while retaining stereo width as auxiliary features;
2. preserve original RMS, peak, crest, onset, and stereo width diagnostics;
3. onset-align under the same rules as `timbre_distance`;
4. loudness-normalize the timbre branch;
5. compute log-magnitude STFT views at short, medium, and long windows;
6. compute a 128-band log-mel view for the neural encoder;
7. attach note, velocity, gate, tail, and seed embeddings.

The encoder has a transient branch over short windows and a harmonic/body branch over longer windows. A shared per-probe encoder produces embeddings which a permutation-invariant set transformer aggregates. Missing probes are masked; probe order cannot change the result.

Absolute loudness is excluded from the timbre distance but remains an explicit scalar feature and release diagnostic. This prevents gain from dominating identity without making dynamics unobservable.

## 9. Models

### 9.1 Baseline first

Before diffusion or flow, implement a small baseline with the same codec and encoder:

- categorical cross-entropy heads for structure;
- masked Gaussian-mixture or mean/log-scale heads for continuous fields;
- top-$K$ structure sampling followed by continuous sampling;
- exact-render reranking.

The baseline is not the target architecture. It proves dataset integrity, establishes training throughput, and gives every later model a measured bar to clear.

### 9.2 Masked discrete generator

The discrete model operates on typed field/graph tokens with `[MASK]`, `[INACTIVE]`, and `[PAD]` states. It receives the audio-set embedding, constraint tokens, and the partially denoised structure. Compiler-derived dependency masks remove illegal values before sampling.

V0 uses a fixed padded FM6-P0 token layout. Later Tower profiles may use a graph token sequence, but retain stable field-type embeddings and simultaneous masked denoising; array order must not imply an acoustical causal order.

Inference supports constraints:

- fixed engine or template family;
- fixed or allowed FM algorithms;
- locked fields from a seed preset;
- maximum Tower level and budget;
- required/forbidden motifs;
- maximum structural complexity.

### 9.3 Conditional continuous flow

Conditioned on audio embedding and sampled structure, the continuous model learns a masked velocity field from base noise to active normalized parameters. Inactive fields are zeroed and excluded from the flow loss. Bounded transforms guarantee valid normalized ranges before engine quantization.

The flow returns multiple samples for each discrete structure. Do not average them. Candidate diversity is retained until exact-render scoring.

### 9.4 Symmetry

Operator tokens use shared weights plus role, graph-neighborhood, target, and slot embeddings. Training alternates canonical data with legal permutation augmentation. Validation reports both raw-label accuracy and orbit-aware accuracy. Audio reconstruction and orbit-aware accuracy are release metrics; arbitrary slot accuracy is diagnostic only.

### 9.5 Model output

For each request, sample by default:

- 16 discrete structures;
- 4 continuous configurations per structure;
- 64 pre-render candidates total;
- compile-valid candidates only;
- 16 candidates after cheap feature scoring;
- 8 after exact multi-probe committee scoring;
- 3–5 after diversity selection and local refinement.

These counts are run-config defaults, not hard-coded limits. The run manifest records them.

## 10. Training procedure

### 10.1 Pixi isolation

Add a `training` feature/environment with Python 3.12, PyTorch, and training-only audio/data dependencies. CPU execution must support tests and tiny runs. Accelerator selection is a run setting; checkpoints must load on CPU.

Typical command shape:

```powershell
pixi install --environment training --frozen
pixi run -e training inversion-train --config configs/inversion/fm6_p0_v0.yaml
```

All commands use Pixi. The implementation documentation and scripts contain no standalone `pip install` instructions.

### 10.2 Stages

**T0 — Codec and data integrity**

- build 1,000-configuration smoke dataset;
- verify compilation, determinism, split isolation, and shard hashes;
- overfit 64 examples to near-zero training loss;
- decode and render every overfit prediction.

**T1 — Supervised baseline**

- train encoder and baseline heads;
- measure top-1/top-$K$ structure orbit accuracy;
- exact-render the held-out reconstructions;
- establish random-search and nearest-library baselines.

**T2 — Discrete diffusion**

- replace the categorical heads with masked denoising;
- preserve the same encoder, codec, constraints, and evaluation harness;
- require improved best-of-$K$ audio distance at equal render budget.

**T3 — Conditional flow**

- replace the continuous baseline with flow matching;
- require improved held-out exact-render distance and candidate diversity;
- verify smooth interpolation only within a fixed active structure.

**T4 — Exact-render self-improvement**

- sample candidates for training and external targets;
- render and score them through the exact VM;
- retain reward-weighted elites and diverse near-elites;
- fine-tune by weighted likelihood/self-distillation first;
- add policy-gradient/GRPO-style tuning only if self-distillation plateaus and the reward has passed human calibration.

**T5 — External adaptation**

- Dexed: known source configuration plus independent target audio;
- NSynth: unlabelled Symcrash target, conditioned on instrument/pitch/velocity metadata;
- acoustic references: unlabelled target plus explicit probe manifest.

External targets never enter supervised parameter loss without a generated pseudo-label and its exact-render score/provenance.

### 10.3 Checkpoint contract

A checkpoint directory contains:

```text
model.safetensors
model.json
config-schema.json
feature-schema.json
dataset-digests.json
training-history.jsonl
evaluation.json
```

`model.json` records code revision, architecture, dimensions, training environment lock digest, random seeds, and required schema versions. Loading fails closed on incompatible configuration or feature schemas.

Pickle-only checkpoints are not accepted as release artifacts.

## 11. Perceptual scoring and acceptance

### 11.1 Multi-probe score

`ProbeSuiteScore` calls the existing per-file committee for each matching probe and reports:

- weighted mean composite;
- worst-probe composite;
- each metric by probe;
- pitch and velocity consistency;
- release/tail diagnostics;
- hard safety failures;
- model complexity and budget.

Ranking is lexicographic at the outer gate:

1. valid compile and no hard safety failure;
2. no unacceptable worst-probe regression against the seed;
3. lower calibrated perceptual aggregate;
4. lower unnecessary complexity;
5. higher diversity from already selected candidates.

Before calibration, the report says `metric_status: bootstrap`. No threshold is labelled a JND, and human audition is mandatory.

### 11.2 Committee versioning

Move weights and preprocessing identifiers into a checked configuration record:

```yaml
schema: symcrash.metric-committee.v0
status: bootstrap
preprocessing: onset-loudness-v0
components:
  multiscale_spectrum: 0.30
  envelope: 0.10
  centroid: 0.10
  zimtohrli: 0.45
  level: 0.05
calibration_digest: null
```

The values above reproduce the current bootstrap policy; they are not declared optimal. A calibrated version requires a verdict-corpus digest and held-out preference accuracy report.

### 11.3 Candidate diversity

Greedy selection uses a minimum distance in both structural orbit and rendered audio embedding. Two candidates are not diverse merely because operator numbers differ under a legal permutation. Conversely, two perceptually close candidates with genuinely different structures may both survive when editability or cost differs.

## 12. Inference pipeline

### 12.1 Request

```yaml
schema: symcrash.inversion-request.v0
target: reference_audio/marimba/probes.yaml
model: _inversion_data/checkpoints/fm6-p0-v0
seed_preset: presets/instruments/dx7_marimba_v0.yaml
constraints:
  profile: fm6-p0
  lock: []
  allowed_algorithms: "1-32"
  max_candidates: 64
  tower_level: P0
scoring:
  committee: configs/inversion/committee-bootstrap-v0.yaml
  zimtohrli: required
refinement:
  enabled: true
  passes: 2
```

An omitted seed means inference from scratch. With a seed, locked fields and allowed structural edits are explicit; no hidden rule decides whether topology may change.

### 12.2 Execution

1. Validate and fingerprint the reference probe set.
2. Encode all probes and their performance metadata.
3. Sample structures under request constraints.
4. Sample continuous fibers and quantize through `ConfigCodec`.
5. Decode, validate, and compile; retain diagnostics for rejected candidates.
6. Run a cheap single/three-probe screen.
7. Render survivors on the full nine-probe suite.
8. Score with `ProbeSuiteScore` and choose diverse top-$K$.
9. Invoke the generalized `auto_refine` on each survivor's active semantic blocks.
10. Re-render and apply the full-committee veto against both pre-refinement candidate and supplied seed.
11. Write the audition package; do not touch `presets/`.

### 12.3 Output candidate

Every candidate YAML is accompanied by a machine-readable record:

```json
{
  "schema": "symcrash.inversion-candidate.v0",
  "rank": 0,
  "preset": "candidates/candidate_00.yaml",
  "source_request": "request.yaml",
  "model_digest": "sha256:...",
  "model_scores": {"structure": null, "continuous": null},
  "canonical_orbit": "sha256:...",
  "compile": {"ok": true, "diagnostics": []},
  "score": {},
  "refinement": {},
  "human_verdict": null,
  "promotion_status": "not-reviewed"
}
```

Model probability never substitutes for audio score; audio score never substitutes for human verdict while the committee is uncalibrated.

## 13. Pleasant dial implementation

### 13.1 Separate the two dial classes

**Authoring controls** such as `/fm/op2/level` edit preset coordinates. Their UI domain should reflect the field codec, valid range, and log/linear law. They are allowed to expose ugly sounds because they are laboratory controls, although clipping and invalid values remain forbidden.

**Performance controls** under `/system/*` are part of the playable instrument. Every position in their public range should be useful, safe, and reasonably smooth. Pleasant-path fitting applies to this class first.

### 13.2 Audit

For each factory preset and each of the eleven system axes:

1. hold the other axes at the preset defaults;
2. render positions $s = 0, 1/32, \ldots, 1$ over a reduced three-probe suite;
3. compute adjacent perceptual distance, loudness, crest, centroid, flux, pitch stability, limiter activity, and identity embedding drift;
4. mark dead intervals, cliffs, unsafe points, and identity breaks;
5. produce an HTML/audio sweep for human labels: pleasant, useful extreme, redundant, or reject.

The audit is useful before any learned inverse exists and should be implemented immediately after `ProbeSet`.

### 13.3 Scalar arc-length remapping v0

The first fitter does not invent new cross-parameter modulation. It finds a monotone map $h:[0,1]\to[0,1]$ for each existing system axis such that

$$
d_{\mathcal P}\left(Y(h(s+\Delta)),Y(h(s))\right)
$$

is approximately constant over accepted intervals. Rejected source intervals are skipped; endpoints are the best accepted useful extremes. The mapping is represented as 17 monotone Q15 knots with piecewise-linear interpolation.

This solves dead zones and perceptual cliffs without changing macro ownership or the existing family projection.

### 13.4 Multivariate path fitting v1

After scalar profiles pass audition, permit a dial to trace a bounded path through controller-owned system axes only. Use five to nine control knots and optimize:

- perceptual velocity variance;
- identity drift;
- hard safety violations;
- path curvature;
- useful endpoint separation.

The path may compensate, for example, brightness with focus or unity, but may not alter STATIC operator definitions or runtime-owned state. CMMM-owned `age` and envelope-owned `energy_state` remain inaccessible.

Optimization is gradient-free against exact renders. A learned model may propose initial paths later, but the acceptance loop is identical.

### 13.5 Profile artifact

V0 writes an untracked workbench artifact:

```yaml
schema: symcrash.dial-profile.v0
preset_digest: sha256:...
probe_set: standard-reduced-v0
committee: bootstrap-v0
controls:
  brightness:
    kind: scalar-remap
    input_q15:  [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18431, 20479, 22527, 24575, 26623, 28671, 30719, 32767]
    output_q15: [0, 1024, 2304, 3840, 5632, 7680, 9984, 12544, 16384, 19100, 21700, 24000, 26000, 27800, 29500, 31200, 32767]
    neutral_input_q15: 16384
    audition_status: pending
```

The example numbers illustrate the format only; generated files contain measured values. `SystemAxisRuntime` may load a profile explicitly for workbench audition. A missing profile is a strict no-op.

Only after profile behavior is accepted across several presets should a separate normative proposal define:

- storage in tracked `presets/macros/`;
- optional capsule chunk ID and size budget;
- interpolation rounding;
- Python/FPGA conformance vectors;
- old-reader behavior.

This postpones an irreversible ABI choice without postponing the perceptual work.

## 14. Commands

Proposed Pixi task surface:

```powershell
# Inspect the model-visible schema and unsupported preset features
pixi run inversion-schema --profile fm6-p0 presets/instruments/dx7_marimba_v0.yaml

# Generate and validate the small corpus first
pixi run inversion-build-dataset --config configs/inversion/smoke-fm6-p0-v0.yaml

# Train/evaluate in the isolated ML environment
pixi run -e training inversion-train --config configs/inversion/fm6-p0-v0.yaml
pixi run -e training inversion-evaluate --checkpoint _inversion_data/checkpoints/<run-id>

# Invert a labelled reference set
pixi run -e training invert-preset reference_audio/marimba/probes.yaml \
  --seed presets/instruments/dx7_marimba_v0.yaml \
  --output audio_out/inversion/marimba-<run-id>

# Audit and fit a performance dial profile
pixi run inversion-audit-dials presets/instruments/dx7_marimba_v0.yaml \
  --output audio_out/inversion/dials/dx7-marimba
pixi run inversion-fit-dials audio_out/inversion/dials/dx7-marimba/audit.json \
  --output audio_out/inversion/dials/dx7-marimba/profile.yaml

# Explicit promotion after a recorded human verdict
pixi run inversion-promote audio_out/inversion/marimba-<run-id> \
  --candidate 0 --verdict accepted
```

`inversion-promote` performs a dry run by default, shows the exact destination and diff, verifies lint/compile/tests, and requires `--apply` to write. It refuses promotion without a non-empty verdict and never overwrites a higher preset version.

## 15. Tests and acceptance gates

### Gate G0 — Contracts

- All shipped FM6-P0 presets encode and decode.
- Decode/re-encode differs by at most one field quantization step.
- Unsupported features are reported and preserved, never silently dropped.
- Every generated config compiles or returns a structured rejection.
- Legal symmetry transforms render bit-exact across the probe suite.

### Gate G1 — Render/data integrity

- Repeated render requests with the same fingerprint are bit-exact.
- Changing seed affects only declared stochastic paths.
- Dataset shards verify against their manifest hashes.
- Parent/symmetry variants never cross split boundaries.
- Interrupted generation resumes without duplicate or partial records.

### Gate G2 — Learning smoke test

- The baseline overfits 64 examples.
- Every decoded prediction compiles.
- A fixed checkpoint/request/seed reproduces candidate tokens and scores.
- Top-$K$ beats uniform legal random search at equal render budget on held-out self-reconstruction.

### Gate G3 — Generative model

- Discrete diffusion improves best-of-$K$ orbit-aware structure accuracy or exact-render score over the baseline.
- Conditional flow improves best-of-$K$ exact-render score and does not collapse diversity.
- Multi-probe training beats single-probe training on held-out nine-probe evaluation.

### Gate G4 — External targets

- Same-voice Dexed/Symcrash pairs rank ahead of cross-voice pairs.
- The core batch does not regress relative to its seed under full-committee guard.
- Marimba and harpsichord receive dedicated human verdicts.
- NSynth evaluation is reported by held-out instrument family, not merely random clip split.

### Gate G5 — Dial profiles

- Profile endpoints and every interpolation knot pass hard safety checks.
- Adjacent-distance variance improves over the raw control sweep.
- No new perceptual cliff appears on the full nine-probe suite.
- Neutral physical position reproduces the preset's authored default within one Q15 step.
- Human audition accepts the full sweep, not just its endpoints.
- Missing/disabled profile remains bit-exact with current workbench behavior.

### Gate G6 — Promotion

- Lint and capsule compilation pass.
- Candidate source provenance and reference rights are recorded.
- Full audition package exists.
- Human verdict is recorded verbatim.
- Preset version increases exactly once.
- Generated library output remains under `presets/.build/`.

## 16. Milestones

### M0 — Shared contracts

Deliver:

- `ConfigCodec` FM6-P0 schema;
- canonicalization and active masks;
- `ProbeSet` and exact render request;
- structured compile/render diagnostics;
- G0/G1 tests.

This milestone has value without ML: it makes systematic search, datasets, and dial audits reproducible.

### M1 — Multi-probe automation

Deliver:

- `ProbeSuiteScore`;
- multi-probe `auto_refine`;
- target-manifest builder for Dexed and local recordings;
- dial audit command;
- full core-preset baseline reports.

### M2 — Dataset and baseline

Deliver:

- 1,000-config smoke corpus, then 50,000-config FM6 corpus;
- training Pixi environment;
- encoder, baseline heads, nearest-preset baseline;
- exact-render top-$K$ evaluation.

### M3 — Hybrid generator

Deliver:

- masked discrete diffusion;
- conditional continuous flow;
- orbit-aware training and diversity selection;
- inference audition packages.

### M4 — External adaptation

Deliver:

- exact-render elite self-distillation;
- DX7/Dexed core-target runs;
- NSynth ingestion and held-out family evaluation;
- calibrated reward tuning only if the committee gate passes.

### M5 — Pleasant controls

Deliver:

- scalar dial profile fitter;
- workbench profile loading;
- human-labelled sweep reports for all core presets;
- optional multivariate controller paths after scalar acceptance.

### M6 — Standardization decision by evidence

Produce a normative dial-profile proposal only after M5 establishes:

- required knot count;
- whether scalar maps are sufficient;
- maximum observed path dimensionality;
- runtime memory and multiply/add cost;
- accepted interpolation and rounding law.

This is an engineering measurement milestone, not a request for an aesthetic decision.

## 17. Parallelism and resource policy

Dataset rendering is CPU-parallel by process, with one immutable output shard per worker. Workers receive explicit configuration/probe ranges; they do not append to one shared file. The coordinator validates and atomically publishes completed shards.

Training uses one writer for checkpoints and metrics. Exact-render evaluation can fan out by candidate/probe, then joins before ranking. Zimtohrli remains process-isolated and is applied only to shortlisted candidates unless a batch backend proves stable.

No worker launches an audio device or workbench server. Offline generation cannot leave a stuck note.

## 18. Failure and recovery

- Every long operation writes an append-only JSONL event log.
- Dataset and render jobs are content-addressed; rerunning skips verified outputs.
- Checkpoints are written to a temporary sibling and atomically renamed after validation.
- Candidate generation records compile failures without terminating the run unless all candidates fail.
- A failed local refinement restores the pre-refinement candidate.
- A metric or Zimtohrli failure leaves the candidate `unscored`; it never converts failure into a perfect or zero distance.
- Promotion is the only operation allowed to write tracked preset YAML, and its dry-run path is the default.

## 19. Definition of done

The inversion system v0 is complete when:

1. A fresh Pixi checkout can reproduce the codec, smoke dataset, baseline checkpoint, and one inference run from documented commands.
2. A multi-probe reference produces at least three diverse, compiling, playable candidate presets with complete provenance.
3. Exact-render top-$K$ performance beats nearest-library and uniform-random baselines on held-out synthetic and DX7 targets.
4. Marimba and harpsichord candidates survive full-committee comparison and human audition against their current seeds.
5. NSynth targets can drive reward/pseudo-label inference without being misrepresented as parameter-labelled data.
6. Each core preset has an eleven-axis dial audit; accepted profiles measurably reduce dead zones/cliffs and pass complete-sweep audition.
7. No automation writes to `presets/instruments/` except an explicit, verdict-bearing promotion command.
8. The default synth remains usable without installing the training environment.

At that point the system performs both halves of the theory in operational form: it learns legal explanations for reference audio, and it shapes the continuous paths through those explanations into playable controls.
