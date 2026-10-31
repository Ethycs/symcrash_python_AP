# Symcrash RTL components

## Current control-path qualification — 2026-10-31

The 2026-10-31 FIFO mapping follow-up preserves every RTL operation;
removing comments and the memory attribute yields the committed token stream.
Only the same two 128-bit feeder words change physical implementation:
**+256 FF and -32 distributed-RAM cells**, with measured LUT/mux changes.
Payload BRAM counts, N/K/P, datatypes, arithmetic, memory geometry, latency
and deadlines stay fixed. The isolated mapped FIFO passes an all-state,
all-input read/next-state proof with actual netlist corruption controls.

Current modified-path coverage is **20 RTL simulations, 47 formal/evidence
tests and four mapped-FIFO tests**. The 41 unchanged owner/burst/priority/
bounds tests retain matching sources: 112 distinct tests through scoped
runs. This is not a fresh whole-root run. Audio comparisons cover 141,824
stream samples and 21,504 windowed samples; cycle assertions are unchanged.

| Current component | 100 MHz routing |
|---|---|
| Burst | 3/3 pass; 122.59..132.19 MHz |
| Priority queue | 3/3 pass; 106.32..116.08 MHz |
| Owner | 9/9 pass; 110.38..120.26 MHz |
| Prepared-frame stream | 9/9 pass; 105.97..112.03 MHz |
| Shared DMA-window top | 9/9 pass; 102.82..106.79 MHz |

Combined prepared-frame transport/decode/window passes all nine 100 MHz out-of-context placements.
The full matrix uses ECP5-25K speed 6, CSFBGA285 and seeds 1/2/3. Every
current result and the prior failures are retained. Verified ABC9 mapping
requires clean subtool logs and successful mapping-equivalence checks.
Evidence and physical resource costs are in `tests/artifacts/regen_reports/hdl_fifo_mapping_20261031.txt` and the stream/window
implementation profiles. The preceding signed baseline is root `23377cc`
with docs `8651481`; its failed payload-BRAM-to-FIFO paths motivated this change.

The 40,000-clock DMA window and 133,333-clock block budgets remain fixed.
The mapped proof covers only the FIFO, and routing is out of context.
Board I/O/global clocks, PHY, bounded workload admission, full voice/operator
generation, global memory packing, clock load and board power remain open,
along with Mode A storage and K2 tier cost.

Earlier milestone measurements below remain historical; this section and
the linked implementation profiles give the current source-matched results.

`ccs_idwpt.v` implements the balanced 64-leaf, depth-six polyphase inverse
transform with K=2/4/8. Stored coefficients/audio remain int16; signed RNE,
wide inter-step results, node SAT16 and final headroom restoration match
production Python. This is a standalone decoder, with board integration open.
The [HDL profile](../../symcrash_docs/05%20-%20Standards/09%20-%20Wavelet/WSE-IMPL-CCS-IDWPT-HDL-v1.md)
records range proofs, measured costs and remaining acceptance gates.

## Interface

`coeff_valid && coeff_ready` accepts one leaf's `min(K,4)` int16 slots,
lane zero in the low bits. Send leaves 0..63. For K=8, wait through computation,
then send leaves 0..63 for slots 4..7. Keep a pending beat stable until accepted.

`audio_valid && audio_ready` accepts four chronological int16 samples,
earliest in the low bits. Sixteen beats form a block; `audio_last` marks the
last. Output remains stable under backpressure. Input pauses during compute
and output. Active-low reset aborts partial work; resume at leaf zero.

K and `HEADROOM_BITS` are compile-time parameters. Headroom defaults to 2,
with supported range 0..15. Coefficients are leaf-major; output selection is
`audio[k::K] = inverse(coefficients[:,k])[0::K]`.

## Pipeline and arithmetic

One source pair enters each clock. Undo-update results and their original
details stream directly into undo-predict. A pending register supplies the
left even, the next result supplies its right neighbor, and node ends repeat
their own even. The last pair flushes without another coefficient read.
The former wide-even scratch and second detail-read pass are removed.

Five levels take 32 issue + 10 drain clocks each. The root only needs even
samples, so its update uses 32+5. Compute is 247 clocks per group, down from
407. No-stall transactions take 327 clocks (K2/K4) or 638 (K8).

Boundary selection is registered before the sum, outside the block-RAM
output path. Node saturation checks the sign extension of an 18-bit value;
headroom compares against exact pre-shift bounds. These are equivalent to
the original clipping sequence, verified exhaustively. They do not change
stored precision, saturation boundaries or headroom.

## Reproduce the gates

From the repository root:

```
pixi run check-hdl
pixi run synth-hdl
pixi run route-hdl
pixi run python tools/route_hdl_idwpt.py --seeds 1 2 3
pixi run python -m pytest tests/conformance/test_hdl_idwpt.py tests/conformance/test_hdl_evidence.py -q
```

`check-hdl` requires Icarus Verilog and fails if absent. The actual RTL is
compared with independent Fraction transforms and production JIT. Coverage:
2,104 frames, every scalar impulse position at K=2/4/8, extrema/random data,
327,679 rounding sums, all 262,144 signed 18-bit saturation inputs, and all
2,097,152 combinations of signed 17-bit input/headroom 0..15. Eighteen reset
probes cover every pipeline age, final-pair flush, later levels, root-only
update, output and K8's second group, each followed by a checked block. Transfer stability, input
bubbles and exact cycles are asserted. Final RTL/guard pytest: 17 passed in 83.85s (0:01:23).

Pytest skips when the simulator is missing; a skip is not acceptance.
`--quick` on the simulation Python command reduces impulse coverage and
is diagnostic only. `synth-hdl` uses Yosys ECP5 synthesis/checks for K=2/4/8,
writing scripts, netlists, logs and source/netlist hashes to `build/hdl-tests`.

`route-hdl` requires matching source and netlist hashes, then runs nextpnr
at the fixed 100 MHz target, ECP5-25K, speed grade 6, CSFBGA285. It records
per-seed timing/utilization, logs and routed netlists; a failing or missing
clock result exits nonzero. Default seed is 1; the three-seed command checks
placement variation. See the profile/report for the measured result matrix.
This experimental out-of-context mode omits I/O buffers and global clock
promotion/routing. It has no board pins or external delay constraints, so
it does not qualify board timing or the complete synthesizer.

## Measured synthesis and cost

Yosys 0.33 (2584903a060), headroom 2:

| Primitive / cost | K=2 | K=4 | K=8 |
|---|---:|---:|---:|
| LUT4 | 643 | 1,035 | 1,036 |
| CCU2C | 130 | 254 | 254 |
| TRELLIS_FF | 912 | 1,732 | 1,732 |
| TRELLIS_DPR16X4 | 64 | 16 | 16 |
| DP16KD | 4 | 8 | 8 |
| DSP | 0 | 0 | 0 |
| Physical block RAM bytes | 9,216 | 18,432 | 18,432 |
| Logical array bytes | 640 | 1,152 | 1,152 |
| Compute clocks | 247 | 247 | 494 |
| Transaction clocks, no stalls | 327 | 327 | 638 |

LUT4 alone is not packed usage: carry cells and LUTRAM also use logic.
Streaming raises K4 FF use 1,512 -> 1,732, while removing 68 LUTRAM cells
and keeping block RAM unchanged. Current three-seed routing passes all nine
K2/K4/K8 configurations at 108.38..112.51 MHz against the fixed 100 MHz target.
The prior two-pass pipeline took 487/958 transaction clocks; the unpipelined
core took 784/1,552 and failed timing at 32.63 MHz. Both reports remain as
historical evidence with their own source hashes.

K4/K8 decoder compute fits within the T1/T0 estimates of 400/1,000. K2
still exceeds T2's 100 by 147 clocks. Carrier/operators, transfers and other
voice work need their own budget; these decoder counts do not qualify the
whole service. DMA/prefetch, bounded stalls, mixing and aggregate BRAM fit
remain open. No numeric format or documented budget was relaxed.

## DMA burst engine

`dma_burst.v` implements aligned 16-byte reads/writes using the standard's
64-bit descriptor. An ordered request/response adapter connects external
RAM; a synchronous 128-bit port connects BRAM. The payload is opaque: no
audio format or transform dimension changes. A two-entry FIFO preserves
write data under stalls. Completion waits for every response, reports errors
and overruns, and stays stable until consumed. A separate overdue signal
also works while a response remains outstanding.

The [DMA profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-DMA-BURST-v1.md)
specifies addresses, handshakes, reset and failure ownership. The 16-bit BRAM
field is preserved; this byte-address profile reaches the lower 64 KiB,
including the coefficient partition. Scatter/gather and upper-bank mapping
remain open. Reset must flush/reset the external adapter as well.

```
pixi run check-hdl-dma
pixi run synth-hdl-dma
pixi run route-hdl-dma
pixi run python -m pytest tests/conformance/test_hdl_dma.py tests/conformance/test_hdl_dma_evidence.py -q
```

Final validation: 11 tests pass in 25.06 seconds, covering 1,394 descriptors
and 2,486,336 requested bytes. Every length 1..255 is exercised in both
directions, with stalls, malformed descriptors, errors, deadline edges,
whole-BRAM checks and reset. Ideal service is L+1 clocks for reads and L+3
for writes; external stalls add time. The default per-transfer alarm is
40,000 clocks, with aggregate DMA/audio admission still required.

Synthesis uses 293 LUT4, 131 FF, 118 carry cells, 32 LUTRAM and no block RAM
or DSPs. The 32-byte FIFO occupies 256 bytes of physical distributed RAM.
All three standalone routing seeds pass 100 MHz at 122.73..131.29 MHz.
These counts exclude staging memory, queues, the RAM adapter/PHY and buffer
ownership. Double buffering, two-block prefetch, service-class integration and
real-device timing remain open. Evidence:
`tests/artifacts/regen_reports/hdl_dma_20261030.txt`.

## DMA priority queue

`dma_priority.v` wraps the actual burst engine with 256 unchanged 64-bit
descriptors: 2 KiB of payload. Wire priority 15 is highest; ties are FIFO.
Selection waits for an idle engine, and full queues backpressure admission.
Started bursts finish before another descriptor is dispatched. The
[priority profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-DMA-PRIORITY-v1.md)
defines service-class mapping, ownership boundaries and the additional costs.

```
pixi run check-hdl-dma-priority
pixi run synth-hdl-dma-priority
pixi run route-hdl-dma-priority
```

Eleven tests pass in 10.46 seconds, covering 3,328 descriptors, all 256 queue
slots, independent priority/FIFO order, payloads, errors, recycling, stalls
and fourteen reset probes. Initialization takes 256 clocks. Admission spacing
is four clocks when the engine is busy; dispatch acceptance is five clocks
after selection, or nine after admission into an empty idle queue. Queue
waiting and completion backpressure must be budgeted separately.

The wrapper including its burst engine uses 1,274 LUT4, 589 FF, 142 carry
cells, 32 LUTRAM and two block RAMs. Descriptor/link arrays consume 4,608
physical block-RAM bytes; metadata is additional to the 2 KiB payload.
All three standalone seeds pass 100 MHz at 108.20..113.13 MHz. Initial
inference and timing failures remain in the report. Aggregate memory,
DMA-window admission, buffer handoff, two-block prefetch and the adapter/PHY
remain open. Evidence: `tests/artifacts/regen_reports/hdl_dma_priority_20261030.txt`.

## Coefficient buffer ownership

`coeff_buffer_owner.v` reserves two complete N=64, signed-16-bit frames,
with existing K=2/4/8. It publishes only successful consumed DMA completions,
leases a ready frame to DSP until release, and drains expired fills before
reusing their banks. Misses return without waiting for RAM. Service classes
0..3 map to wire priorities 15..12. The
[ownership profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-COEFF-BUFFERS-v1.md)
specifies handshakes, reset/flush requirements, races and reserved addresses.

```
pixi run check-hdl-coeff-buffers
pixi run synth-hdl-coeff-buffers
pixi run route-hdl-coeff-buffers
```

Thirteen tests pass, including six workloads connected to the actual DMA
queue/engine and seven stale-evidence guards. They cover 1,674 admissions,
1,560 complete acquisitions, partial/error/late fills, cancellation, wrong
releases, reset, and a 256-block N+2 prefetch sequence for each K/stall mode.
All nine standalone ownership-control runs pass 100 MHz at 101.11..110.14 MHz.

The controller uses 902/902/906 LUT4 and 184 FF for K=2/4/8. Its external
two-frame payload is 512/1,024/2,048 bytes inside 0x8000..0xffff. Synthesis
excludes that memory, shared DMA and decoder costs. The caller still owns
block scheduling, real decoder handoff and fallback audio; complete shared
memory/timing admission and PHY integration remain open. Evidence is
`tests/artifacts/regen_reports/hdl_coeff_buffers_20261030.txt`.

## Prepared coefficient frames to audio

`ccs_stream.v` connects that owner to real payload RAM, a two-word feeder,
the unchanged CCS decoder, and a 128-byte replay buffer. The
[stream profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-CCS-STREAM-v1.md)
records the leaf-major layout, K8 slot-group rereads, exact final-input release
and retained-audio fallback. A cache miss replays the last complete block,
or silence after reset, without waiting for outstanding DMA.

```
pixi run check-hdl-ccs-stream
pixi run synth-hdl-ccs-stream
pixi run route-hdl-ccs-stream
```

Twenty tests pass: eight actual DMA-to-audio workloads and twelve evidence
guards. They verify 141,824 samples against independent rational arithmetic,
all input positions, stalls, source reuse during decoding, misses, errors and
reset. Ideal ready-frame service is 331 clocks for K2/K4 and 642 for K8;
fallback takes 18. DMA prefetch and arbitrary output stalls are additional.
K2 remains over its tier estimate; full voice/audio admission remains open.

Combined synthesis now includes owner, payload RAM, feeder, decoder and replay:
1,796/2,141/2,121 LUT4, 1,289/2,245/2,247 FF and 4/12/12 block RAMs for K2/K4/K8.
Physical block RAM is 9,216/27,648/27,648 bytes; distributed RAM adds
1,408/512/512 bytes. Shared DMA and the PHY remain outside this synthesis.
See `tests/artifacts/regen_reports/hdl_ccs_stream_20261030.txt` for final
source-matched synthesis/routing and validation evidence.
All nine combined stream runs pass 100 MHz at 102.06..113.56 MHz, using the
same out-of-context limits as the other hardware probes. Device/PHY, board
I/O and global clock timing still need qualification.

## Shared DMA-window deadline

`dma_window.v` gives the priority queue and burst engine one 40,000-clock
deadline, including queue wait and callbacks. `ccs_dma_window.v` connects
the complete prepared-frame path and registers payload writes/completions.
Expired work drains without publishing; obsolete/canceled prefetches are
acknowledged without claiming memory. Audio fallback proceeds during drain.
The [window profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-DMA-WINDOW-v1.md)
specifies handshakes, the exact inclusive deadline, reset, tagged rejection
pulses and separate ingress/descriptor counters.

```
pixi run check-hdl-dma-window
pixi run synth-hdl-dma-window
pixi run route-hdl-dma-window
```

Twenty-seven tests pass, including full-queue and cumulative deadline probes,
21,504 exact audio samples, actual payload writes and publication ordering.
Combined prepared-frame transport/decode/window passes all nine 100 MHz out-of-context placements.
The full top uses 6/14/14 block RAMs for K2/K4/K8, equivalent to
13,824/32,256/32,256 physical bytes. It excludes the adapter/PHY and other
synth engines. Ready-frame service remains 331/331/642 clocks; fallback 18.
Bounded adapter/PHY service, full-workload admission, voice scheduling, global memory packing and the 133,333-clock whole-block deadline remain open. K2 tier cost and Mode A storage remain unresolved.
Evidence: `tests/artifacts/regen_reports/hdl_dma_window_20261031.txt`.

## Tool setup

Native Icarus/Yosys/nextpnr executables on PATH are preferred. Windows also
supports isolated packages through WSL Ubuntu-24.04 (`SYMCRASH_HDL_WSL`
overrides the distribution). Ubuntu Noble packages were downloaded with
`apt download` and extracted via `dpkg-deb -x` without system installation:

- `iverilog=12.0-2build2` into `build/hdl-tools/iverilog`;
- `yosys=0.33-5build2`, `yosys-abc=0.33-5build2`, and
  `libtcl8.6=8.6.14+dfsg-1build1` into `build/hdl-tools/yosys`;
- `nextpnr-ecp5=0.6-3build5`, `nextpnr-ecp5-chipdb=0.6-3build5` and their
  Boost 1.83 runtime libraries into `build/hdl-tools/nextpnr`.

Drivers set the Icarus backend and Yosys ABC/library paths. nextpnr uses a
fixed database path: the previously absent `/usr/share/nextpnr` in Ubuntu
is linked to the extracted `build/hdl-tools/nextpnr/usr/share/nextpnr`.
Native installs need their usual runtime dependencies. Package/source hashes,
validation and timing details are retained in
`tests/artifacts/regen_reports/hdl_stream_20261030.txt`; the previous
pipeline report retains the isolated package hashes.

## Legacy modules

These earlier files retain historical interfaces; they are not current CCS:

- `wavelet_synth_balanced.v`: legacy 32-by-8 layout; only two slots per leaf
  enter synthesis. It does not decode current 64-by-K CCS.
- `wavelet_synth.v`: incomplete reconstruction body; its old six-cycle,
  five-tree and throughput claims are not verified behavior.
- `cdf53_inverse_lift.v` / `tb_cdf53_inverse_lift.v`: half-up/per-step policy.
- `tree_rom.v`, `sat16.v`, `generate_test_vectors.py`: legacy support files.

Old estimates of 800 LUTs, 1,100 FFs, zero BRAM and general Python bit
equivalence are superseded by the scoped evidence above.
