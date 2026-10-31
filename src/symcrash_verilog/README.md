# Symcrash RTL components

## Complete VM core pages — 2026-10-31

`voice/hardware_vm_page.py` prepares an exact 2 KiB page for the existing
cache/renderer: aligned control fields occupy bytes 0..836, bytes 837..1023
are reserved, and all coefficient slots start at byte 1024 (word offset 64).
The current reference bridge is K4. Complete optional engine/resource records
have explicit external addresses and ownership; the codec rejects overflow.

`pixi run check-hdl-vm-page` checks 18 real VM pages through the unchanged RTL
with 100-cycle adapter latency, stalls off/on, held reads, eviction, updates,
failed write/retry and reset. Both runs pass 14,720 samples; K4 service is
348 clocks ready / 354–357 stalled, max window 1,899/40,000 clocks. 74 Python
checks establish complete field and future-audio replay. Alignment adds 146
padding bytes inside the same page to eliminate 12 scalar word crossings.

These tests cover core-page transport/rendering. External engine RTL, shared
resource ownership, musical voice state, aggregate admission and global fit
remain open. The opaque cache does not parse the embedded page header; hardware
header admission is still required. No RTL arithmetic, widths, N/K or timing limits changed, and no
new route is claimed. See the
[VM page profile](../../symcrash_docs/05%20-%20Standards/03%20-%20Voice/WSE-IMPL-HARDWARE-VM-PAGE-v1.md)
and `tests/artifacts/regen_reports/hardware_vm_page_20261031.json`.

## Runtime prepared-frame transport — 2026-10-31

The worktree now includes runtime owner, stream and shared-window modules
for per-frame K2/K4/K8. N64, every int16 slot, wide intermediates,
RNE/SAT16/headroom and the existing deadlines remain fixed. Two 1,024-byte
logical banks support exact 256/512/1,024-byte DMA payloads. The two-entry
128-bit FIFO gathers adjacent slot pairs from the normative leaf-major
payload. K4/K8 reread the protected source; internal read traffic is
256/1,024/4,096 bytes. Ready-frame service is 183/346/672 clocks; fallback 18.
These counts exclude prefetch and the rest of the voice workload.

Twelve fresh worktree profiles pass with **221,056 exact samples**, as does
the isolated candidate. **29 scoped regression checks**, both physical
owner-encoding checks, the mapped FIFO proof and two mapped-owner window
runs (15,616 samples) pass. All six fresh 100 MHz routes pass: stream
103.520/101.605/105.053 MHz; window 107.921/102.176/107.365 MHz. ECP5-25K,
speed 6, CSFBGA285, out of context, seeds 1/2/3; verified ABC9 mapping.

| Top | LUT4 | FF | CCU2C | DP16KD | DPR cells | Packed COMB |
|---|---:|---:|---:|---:|---:|---:|
| Runtime stream | 4347 | 4460 | 511 | 4 | 176 | 6671 |
| Runtime window | 5352 | 5430 | 714 | 6 | 208 | 8362 |

Physical BRAM is 9,216/13,824 bytes; distributed RAM is 1,408/1,664 bytes;
DSP use is zero. Replay belongs to **one logical stream**. A shared
multi-voice context/cache, held coefficients, tier admission, T3 playback,
operators and mixing remain required. Neither the 133,333-clock full
workload nor global memory fit or board/PHY timing is established here.
The tests use the unchanged 40,000-clock shared DMA deadline.

[Runtime transport analysis](../../symcrash_docs/08%20-%20Analysis/RUNTIME-TRANSPORT-2026-10-31.md) records interfaces, tradeoffs and proof limits. Evidence:
`tests/artifacts/regen_reports/hdl_runtime_transport_20261031.txt`.
Run `pixi run check-hdl-runtime-transport`,
`check-hdl-runtime-transport-proofs`, and
`check-hdl-runtime-transport-mapping`; target tasks are
`synth-hdl-runtime-stream` / `route-hdl-runtime-stream` and
`synth-hdl-runtime-window` / `route-hdl-runtime-window`.
Fixed-K RTL and standalone runtime arithmetic remain unchanged. The
following sections record their separate qualifications. Signing is
deferred; both prepared indexes remain unchanged.

## Standalone runtime tier decoder — 2026-10-31

`ccs_idwpt_runtime.v` now shares one decoder across per-frame K2/K4/K8.
It preserves N64, every int16 coefficient slot, wide intermediates,
RNE/SAT16 and headroom. Thirty-two headroom/stall profiles compare 556,032
exact samples; 26 scoped regression checks and two complete mapped runs
(13,824 samples) pass. All three fresh 100 MHz worktree placements pass
at 126.678/123.047/123.747 MHz, ECP5-25K speed 6, out of context.

The explicit two-slot reuse tradeoff costs 180/343/669 standalone clocks,
including command, versus fixed-K 179/327/638. Compute is 99/198/396.
Mapped cost: 3,401 LUT4, 3,633 FF, 499 CCU2C, 160 DPR cells (1,280 physical
bytes; 640 logical working/output bytes), zero DP16KD/DSP. Output banking
removes replicated write ports without changing capacity or arithmetic.

The runtime transport qualification above now supplies gathering,
ownership and shared-window integration. Tier scheduling, ghost playback,
operators, mixing and the full 133,333-clock workload remain open, as do
board/PHY, global packing and Mode A storage. The qualified fixed-K
integration below is unchanged. Signing is deferred.

[Runtime analysis](../../symcrash_docs/08%20-%20Analysis/RUNTIME-DECODER-2026-10-31.md) records proof limits, both candidates and the
exact costs. Evidence: `tests/artifacts/regen_reports/hdl_runtime_decoder_20261031.txt`.
Run `pixi run check-hdl-runtime-matrix` for the 32-profile matrix,
`check-hdl-runtime-output` / `check-hdl-runtime-control` for compositional
proofs, `check-hdl-runtime-mapping`
for mapped simulation, and `synth-hdl-idwpt-runtime` /
`route-hdl-idwpt-runtime` for target mapping/routing.

## Parameterized production integration — 2026-10-31

The worktree now contains the qualified isolated K2/K4/K8 candidate.
N=64, every coefficient slot, int16 storage, wide intermediates, RNE,
saturation and headroom 0..15 are unchanged. K2 uses four pair positions
per clock and takes 99 compute clocks; K4/K8 retain 247 per group.
No-stall standalone service is 179/327/638 clocks, and integrated
ready-frame service is 183/331/642; fallback remains 18.

The owner maps its unchanged logical policy to seven state FF per bank
(+8 FF total). K2 compares all 64 descriptor bits using eight 16-bit
equality flags captured at the existing callback edge (-56 callback FF
net). K4/K8 retain direct matching. Per-bank admission removes the shared
duplicate/selection chain without adding a register or clock.

Both the isolated candidate and production netlists pass all nine
100 MHz placements at 102.29133..108.36584 MHz. Production validation
passes 122 scoped pytest checks plus four mapped-owner runs. The final
five trailing-space removals were resynthesized: all three complete
netlists are byte-identical, so measured timing is retained explicitly.
Evidence: `tests/artifacts/regen_reports/hdl_parameterized_production_20261031.txt`.
Source-matched production synthesis uses LUT4/FF/CCU2C/DPR/DP counts:
K2 5093/5261/717/272/2; K4 3098/3557/436/64/14;
K8 3104/3559/434/64/14. Packed utilization is recorded separately.

The [integration analysis](../../symcrash_docs/08%20-%20Analysis/CCS-INTEGRATION-2026-10-31.md)
records the numerical/resource tradeoffs, proofs, tests and earlier failures.
Permanent checks include:

```
pixi run check-hdl-owner-pipeline
pixi run check-hdl-owner-mapping
pixi run python tools/check_hdl_idwpt.py --all-headrooms
pixi run python -m pytest tests/conformance/test_hdl_owner_pipeline.py -q
```

The mapping check accepts `--library /path/to/ecp5/cells_sim.v` for native
Yosys installations; the isolated WSL installation is the default.
It maps only the owner and simulates it with the remaining real RTL.
Encoding equivalence uses matched proof points; reset-driven RTL and
mapped tests provide separate evidence. No board timing is implied.

This remains a compile-time K profile, not a runtime mixed-tier engine.
Full T2 workload cost, 40,000-clock shared DMA service, 133,333-clock block
admission, global memory packing, Mode A storage, PHY and board clocks/I/O
still require qualification. The earlier milestone results below retain
their historical source hashes. Prepared FIFO commit batches are unchanged;
the new work remains unstaged while signing is deferred.

## Prior FIFO control-path milestone — 2026-10-31

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

This FIFO section records the preceding staged source. Its evidence is
preserved independently of the subsequent parameterized worktree changes.

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

In K4/K8, one source pair enters each clock. Undo-update results and their original
details stream directly into undo-predict. A pending register supplies the
left even, the next result supplies its right neighbor, and node ends repeat
their own even. The last pair flushes without another coefficient read.
The former wide-even scratch and second detail-read pass are removed.

Five levels take 32 issue + 10 drain clocks each. The root only needs even
samples, so its update uses 32+5. Compute is 247 clocks per group, down from
407 in the prior implementation. K4/K8 transactions take 327/638 clocks.

K2 issues four pairs per clock. Each 64×32 working bank is split into eight
8×32 banks with `bank={xor(address[5:2]),address[1:0]}` and
`row=address[5:3]`. All eight pair reads are collision-free. The first
level's exact symmetric-boundary identity removes its redundant prediction
division while preserving wide intermediates and node saturation. Compute
is `(8+6)+4*(8+10)+(8+5)=99`, with 179 total standalone clocks.
Logical working storage stays 512 bytes and output stays 128 bytes;
physical distributed RAM has the measured depth-padding cost.

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

## Prior serial decoder synthesis and cost

These measurements predate the parallel K2 integration above. Yosys 0.33 (2584903a060), headroom 2:

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
and keeping block RAM unchanged. That source's three-seed routing passed all nine
K2/K4/K8 configurations at 108.38..112.51 MHz against the fixed 100 MHz target.
The prior two-pass pipeline took 487/958 transaction clocks; the unpipelined
core took 784/1,552 and failed timing at 32.63 MHz. Both reports remain as
historical evidence with their own source hashes.

K4/K8 decoder compute fits within the T1/T0 estimates of 400/1,000. This
older K2 decoder exceeded T2's 100 by 147 clocks; the current K2 compute
path takes 99. That alone does not qualify the complete T2 workload. Carrier/operators, transfers and other
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
the parameterized CCS decoder, and a 128-byte replay buffer. The
[stream profile](../../symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-CCS-STREAM-v1.md)
records the leaf-major layout, K8 slot-group rereads, exact final-input release
and retained-audio fallback. A cache miss replays the last complete block,
or silence after reset, without waiting for outstanding DMA.

```
pixi run check-hdl-ccs-stream
pixi run synth-hdl-ccs-stream
pixi run route-hdl-ccs-stream
```

The initial stream milestone passed twenty tests: eight actual DMA-to-audio workloads and twelve evidence
guards. They verify 141,824 samples against independent rational arithmetic,
all input positions, stalls, source reuse during decoding, misses, errors and
reset. Current ready-frame service is 183/331/642 clocks for K2/K4/K8;
fallback takes 18. DMA prefetch and arbitrary output stalls are additional.
K2 remains over its tier estimate; full voice/audio admission remains open.

That initial source's combined synthesis included owner, payload RAM, feeder, decoder and replay:
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

The initial window milestone passed twenty-seven tests, including full-queue and cumulative deadline probes,
21,504 exact audio samples, actual payload writes and publication ordering.
Combined prepared-frame transport/decode/window passes all nine 100 MHz out-of-context placements.
The current full top uses 2/14/14 block RAMs for K2/K4/K8, equivalent to
4,608/32,256/32,256 physical bytes. It excludes the adapter/PHY and other
synth engines. Ready-frame service is 183/331/642 clocks; fallback 18.
Bounded adapter/PHY service, full-workload admission, voice scheduling, global memory packing and the 133,333-clock whole-block deadline remain open. K2 tier cost and Mode A storage remain unresolved.
Evidence: `tests/artifacts/regen_reports/hdl_dma_window_20261031.txt`.

## Native voice cache and shared renderer — 2026-10-31

The production worktree now shares one decoder across retained voice pages
through native read-only/writable ownership. N64, K2/K4/K8, all int16 slots,
wide intermediates, RNE/SAT/headroom, sixteen 2 KiB pages and the existing
100 MHz / 40,000 / 133,333-clock budgets remain fixed. Cache acquisition is
two clocks, three through the wrapper; pending requests reserve their page.
Only the owner may consume a response or release the lease. Dirty stores
and full-page retries remain required before reuse.

Fresh production qualification: ten RTL profiles (454,400 exact samples),
six cache profiles, four complete mapped profiles (including 16,256 streamed
samples), ownership/bounds/FIFO proofs and 66 scoped tests. All six 100 MHz
placements pass: cache 119.474312/115.008629/120.452904 MHz; combined 111.719360/106.100792/112.157913 MHz.
These are ECP5-25K speed 6, CSFBGA285, seeds 1/2/3, out-of-context results;
combined headroom is 2. Shared DMA placement and board timing remain open.

Cache cost is 3,376 LUT4/1,684 FF/12 CCU2C/15 DP16KD; combined cost is
6,388 LUT4/5,719 FF/522 CCU2C/15 DP16KD/160 DPR cells. Physical memory is
34,560 BRAM bytes plus 1,280 distributed bytes in the combined top, with
zero DSP. Standalone cache growth is 871 LUT4/114 FF over its preceding
qualification. Ready K2/K4/K8 service is 185/348/798 clocks, including the
explicit inherited 124-clock K8 FIFO penalty. Normal DMA max is 26,603/40,000.

Complete state layout, shared-client DMA, scheduler/T3, operators/mix,
global memory fit, full workload and board/PHY remain open. Signing is
deferred and both prepared indexes are unchanged. The rejected `chparam`
mapping and legacy insufficient proof are retained in the evidence.
See [production analysis](../../symcrash_docs/08%20-%20Analysis/VOICE-STREAM-PRODUCTION-2026-10-31.md) for interfaces, costs, proof limits and tasks.
Evidence: `tests/artifacts/regen_reports/hdl_voice_stream_production_20261031.txt`.

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
