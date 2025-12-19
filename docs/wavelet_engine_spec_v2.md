# Ultimate Wavelet Engine: Implementation Specification v2.0

## Document Status

| Item | Status |
|------|--------|
| Architecture | FINAL |
| Feature audit | COMPLETE |
| Ready for implementation | YES |

---

## 1. Feature Audit: Original Requirements vs New Architecture

### 1.1 Core Features

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Inverse transform (synthesis) | ✓ | Stencil unit, mode=INVERSE |
| Forward transform (analysis) | ✓ | Same stencil, mode=FORWARD |
| Bidirectional hardware sharing | ✓ | Single stencil, +/- swap |
| CDF 5/3 lifting | ✓ | Hardcoded stencil |
| 32 leaves × 8 coefficients | ✓ | 256 coefficient buffer |
| 64-sample output blocks | ✓ | Overlap-add windowing |

### 1.2 Tree System

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Multiple tree structures | ✓ | Parametric from β ∈ [0,1] |
| Tree 0: Balanced | ✓ | β = 0.5 |
| Tree 1: Bass-heavy | ✓ | β = 0.7-0.9 |
| Tree 2: Treble-heavy | ✓ | β = 0.1-0.3 |
| Tree switching | ✓ | 6 cycles (delta load) |
| Tree crossfade | ✓ | Animate β, 50 cycles |
| Loadable trees from SD | ✓ | Legacy: convert to β |
| Unlimited trees | ✓ | Continuous β space |

### 1.3 Baby Fourier

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Harmonic oscillator bank | ✓ | 32 phase accumulators |
| η blend (wavelet ↔ sine) | ✓ | Output crossfade |
| 8-32 harmonic bins | ✓ | Active leaves only |
| Phase continuity | ✓ | Persistent accumulators |

### 1.4 Harmonic/Exotic System

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Integer harmonic ratios | ✓ | exotic = 0.0 |
| Metallic/bell ratios | ✓ | exotic = 1.0 |
| Continuous blend | ✓ | exotic ∈ [0,1] |
| Per-voice exotic | ✓ | Voice parameter |

### 1.5 Three Operating Modes

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Generator mode | ✓ | Coeffs → audio |
| Transformer mode | ✓ | Audio → coeffs → audio |
| Decomposer mode | ✓ | Audio → coeffs → analysis |
| Mode switching | ✓ | Single-cycle mode register |

### 1.6 Conditioning Frontend

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| DC blocker | ✓ | 1st-order HPF |
| Noise gate | ✓ | Threshold + hysteresis |
| Anti-pop slew limiter | ✓ | Delta clamping |

### 1.7 Voice System

| Original Requirement | Status | Implementation |
|---------------------|--------|----------------|
| Multi-voice support | ✓ | 64-128 voices |
| Voice multiplexing | ✓ | Inject to shared buffers |
| Per-voice parameters | ✓ | freq, amp, β, exotic, η |

### 1.8 New Capabilities (Bonus)

| Feature | Status | Notes |
|---------|--------|-------|
| β as modulation target | ✓ NEW | LFO/env → tree structure |
| Per-voice trees | ✓ NEW | Each voice can have unique β |
| Audio-rate tree modulation | ✓ NEW | Block-rate β changes |
| Tree structure automation | ✓ NEW | β in sequencer lanes |
| 43× faster tree switching | ✓ NEW | 6 vs 256 cycles |
| 88× faster morphing | ✓ NEW | 50 vs 4416 cycles |
| 4-7× more voices | ✓ NEW | 64-128 vs 18 |
| Zero tree storage | ✓ NEW | Parametric generation |

### 1.9 Feature Audit Summary

**All original features: PRESERVED**
**New capabilities: ADDED**
**Dropped features: NONE**

---

## 2. Architecture Overview

### 2.1 Block Diagram

```
                              ┌─────────────────────────────────────────────────┐
                              │           ULTIMATE WAVELET ENGINE               │
                              │                  (~1,500 LUTs)                  │
┌─────────────┐               │                                                 │
│   VOICE     │               │  ┌─────────────────────────────────────────┐   │
│  MANAGER    │──────────────►│  │         COEFFICIENT INJECTION          │   │
│  (external) │  voice params │  │  voices → freq_to_leaf → accum buffers │   │
└─────────────┘               │  └────────────────────┬────────────────────┘   │
                              │                       │                         │
┌─────────────┐               │                       ▼                         │
│     β       │               │  ┌─────────────────────────────────────────┐   │
│  PARAMETER  │──────────────►│  │         β → SCHEDULE GENERATOR         │   │
│  (mod src)  │               │  │     procedural tree from β (80 LUTs)   │   │
└─────────────┘               │  └────────────────────┬────────────────────┘   │
                              │                       │ topology tokens         │
                              │       ┌───────────────┼───────────────┐         │
                              │       ▼               ▼               ▼         │
                              │  ┌─────────┐    ┌─────────┐    ┌─────────┐     │
                              │  │STENCIL 0│    │STENCIL 1│    │STENCIL 2│ ... │
                              │  │ 100 LUTs│    │ 100 LUTs│    │ 100 LUTs│     │
                              │  └────┬────┘    └────┬────┘    └────┬────┘     │
                              │       │              │              │           │
                              │       ▼              ▼              ▼           │
                              │  ┌─────────────────────────────────────────┐   │
                              │  │              TREE MIXER                 │   │
                              │  │     sum partial outputs → wavelet_out   │   │
                              │  └────────────────────┬────────────────────┘   │
                              │                       │                         │
                              │                       ▼                         │
                              │  ┌─────────────────────────────────────────┐   │
                              │  │           BABY FOURIER UNIT             │   │
                              │  │   envelope × sin(ω) → fourier_out       │   │
                              │  └────────────────────┬────────────────────┘   │
                              │                       │                         │
                              │                       ▼                         │
                              │  ┌─────────────────────────────────────────┐   │
                              │  │             η BLEND MUX                 │   │
                              │  │  out = (1-η)·wavelet + η·fourier        │   │
                              │  └────────────────────┬────────────────────┘   │
                              │                       │                         │
                              │                       ▼                         │
                              │                   [audio_out]                   │
                              │                                                 │
                              │  ┌─────────────────────────────────────────┐   │
┌─────────────┐               │  │         CONDITIONING FRONTEND          │   │
│  AUDIO IN   │──────────────►│  │    DC block → gate → slew → forward    │   │
│ (transformer│               │  └────────────────────┬────────────────────┘   │
│  mode only) │               │                       │                         │
└─────────────┘               │                       ▼                         │
                              │               [to forward path]                 │
                              │                                                 │
                              └─────────────────────────────────────────────────┘
```

### 2.2 Data Flow by Mode

**Generator Mode:**
```
coeffs_in → [INJECTION] → [STENCILS inverse] → [BABY FOURIER] → [η BLEND] → audio_out
```

**Transformer Mode:**
```
audio_in → [CONDITION] → [STENCILS forward] → coeffs → [SYMBOL OPS external] → 
        → [STENCILS inverse] → [BABY FOURIER] → [η BLEND] → audio_out
```

**Decomposer Mode:**
```
audio_in → [CONDITION] → [STENCILS forward] → coeffs_out → [to analysis]
```

---

## 3. Module Specifications

### 3.1 Stencil Unit

The atomic compute element. All wavelet operations use this.

#### 3.1.1 Interface

```verilog
module stencil_unit #(
    parameter WIDTH = 16
)(
    input  wire                 clk,
    input  wire                 rst,
    
    // Control
    input  wire                 start,
    input  wire                 mode,           // 0=FORWARD, 1=INVERSE
    input  wire [4:0]           op_src_lo,      // Buffer index for approx/even
    input  wire [4:0]           op_src_hi,      // Buffer index for detail/odd
    input  wire [4:0]           op_length,      // Samples to process (1-32)
    
    // Coefficient buffer interface
    output wire [8:0]           buf_addr,
    output wire                 buf_we,
    output wire signed [WIDTH-1:0] buf_wdata,
    input  wire signed [WIDTH-1:0] buf_rdata,
    
    // Status
    output wire                 busy,
    output wire                 done
);
```

#### 3.1.2 Microarchitecture

```
Pipeline stages (8 cycles latency, 1 sample/cycle throughput):

Stage 1: Fetch d[n-1] or e[n-1] (boundary)
Stage 2: Fetch d[n] or e[n]
Stage 3: Fetch approx[n] or detail[n]
Stage 4: Compute sum (d_prev + d_curr) or (e_curr + e_next)
Stage 5: Shift (>> 2 or >> 1)
Stage 6: Add/subtract (mode-dependent)
Stage 7: Compute second output (odd from even, or vice versa)
Stage 8: Write results, interleave
```

#### 3.1.3 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| Adder 1 (d+d or e+e) | 16 | 16-bit |
| Adder 2 (result) | 16 | 16-bit |
| Adder 3 (second output) | 16 | 16-bit |
| Adder 4 (address gen) | 8 | 9-bit |
| Boundary muxes | 16 | 2× 16-bit 2:1 |
| Mode logic | 8 | +/- select |
| Pipeline regs | 0 | FFs, not LUTs |
| Control FSM | 20 | State machine |
| **Total** | **~100** | |

#### 3.1.4 Timing

| Operation | Cycles |
|-----------|--------|
| Pipeline fill | 8 |
| Process N samples | N |
| Pipeline drain | 8 |
| **Total for 8-sample node** | 24 |
| **Total for 256-sample tree (31 ops)** | ~1,330 |

---

### 3.2 β → Schedule Generator

Procedurally generates tree topology from single parameter.

#### 3.2.1 Interface

```verilog
module tree_generator (
    input  wire                 clk,
    input  wire                 rst,
    
    // Parameter input
    input  wire [15:0]          beta,           // Q0.16 format, 0.0-1.0
    input  wire                 generate,       // Pulse to regenerate
    
    // Schedule output (directly to stencil routing)
    output reg  [4:0]           op_src_lo [0:30],
    output reg  [4:0]           op_src_hi [0:30],
    output reg  [4:0]           op_length [0:30],
    output wire                 schedule_valid
);
```

#### 3.2.2 Algorithm

```
For each level L (0 to 4):
    nodes_at_level = 2^(4-L)
    threshold = beta * nodes_at_level
    
    For each node N at level L:
        if N < threshold:
            schedule[op] = (low_child, high_child)   // Bass emphasis
        else:
            schedule[op] = (high_child, low_child)   // Treble emphasis
        op++
```

#### 3.2.3 β → Tree Mapping

| β Value | Tree Character | Equivalent Legacy |
|---------|----------------|-------------------|
| 0.0 | Maximum treble resolution | Tree 2 |
| 0.25 | Treble-leaning | — |
| 0.5 | Balanced (standard dyadic) | Tree 0 |
| 0.75 | Bass-leaning | — |
| 1.0 | Maximum bass resolution | Tree 1 |

#### 3.2.4 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| Threshold comparators | 32 | 5 levels × ~6 |
| Multiplier (β × count) | 20 | Small, sequential |
| Schedule registers | 0 | FFs |
| Control logic | 28 | |
| **Total** | **~80** | |

---

### 3.3 Coefficient Injection Unit

Routes voice energy into coefficient buffers.

#### 3.3.1 Interface

```verilog
module coeff_injection #(
    parameter N_VOICES = 64,
    parameter N_BUFFERS = 4
)(
    input  wire                 clk,
    input  wire                 rst,
    
    // Voice inputs (from voice manager)
    input  wire                 voice_valid [0:N_VOICES-1],
    input  wire [15:0]          voice_freq  [0:N_VOICES-1],  // Hz in Q10.6
    input  wire [15:0]          voice_amp   [0:N_VOICES-1],  // Q1.15
    input  wire [15:0]          voice_phase [0:N_VOICES-1],  // Q0.16
    input  wire [15:0]          voice_beta  [0:N_VOICES-1],  // Q0.16
    input  wire [7:0]           voice_exotic[0:N_VOICES-1],  // Q0.8
    input  wire [3:0]           voice_harmonics [0:N_VOICES-1], // 1-16
    
    // Buffer write interface
    output wire [1:0]           buf_select,
    output wire [7:0]           buf_addr,
    output wire                 buf_we,
    output wire signed [15:0]   buf_wdata,
    
    // Status
    output wire                 done
);
```

#### 3.3.2 Frequency to Leaf Mapping

```verilog
// Logarithmic mapping: 20 Hz → leaf 0, 20 kHz → leaf 31
// log2(freq/20) / log2(1000) * 31

function [4:0] freq_to_leaf;
    input [15:0] freq_hz;
    reg [31:0] log_val;
    begin
        // Approximation using leading-zero count + interpolation
        // Or: 64-entry LUT for common frequencies
        freq_to_leaf = log_approx(freq_hz);
    end
endfunction
```

#### 3.3.3 Harmonic Injection with Exotic Ratios

```verilog
// Partial ratios based on exotic parameter
// exotic=0: [1, 2, 3, 4, 5, 6, 7, 8]
// exotic=255: [1.0, 2.4, 3.9, 5.2, 6.8, 8.5, 10.3, 12.1]

wire [15:0] partial_ratio [0:7];
assign partial_ratio[0] = 16'h0100;  // Always 1.0
assign partial_ratio[1] = 16'h0200 + (exotic * 16'h0066 >> 8);  // 2.0 → 2.4
assign partial_ratio[2] = 16'h0300 + (exotic * 16'h00E6 >> 8);  // 3.0 → 3.9
// ... etc

// For each voice, for each harmonic:
//   partial_freq = voice_freq * partial_ratio[h]
//   leaf = freq_to_leaf(partial_freq)
//   amplitude = voice_amp / (h + 1)
//   inject_to_buffer(leaf, amplitude, phase * ratio)
```

#### 3.3.4 Buffer Allocation by β

```verilog
// Route voice to appropriate buffer based on β
// Quantize β into N_BUFFERS buckets

function [1:0] beta_to_buffer;
    input [15:0] beta;
    begin
        beta_to_buffer = beta[15:14];  // Top 2 bits → buffer 0-3
    end
endfunction
```

#### 3.3.5 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| freq_to_leaf LUT | 64 | 64-entry, 5-bit out |
| Partial ratio calc | 50 | 8 ratios × interpolation |
| Buffer routing | 30 | β quantization |
| Accumulator | 32 | 16-bit add |
| Address generation | 20 | |
| Control FSM | 40 | 64-voice sequencing |
| **Total** | **~236** | |

---

### 3.4 Baby Fourier Unit

Sine bank synthesis from coefficient envelopes.

#### 3.4.1 Interface

```verilog
module baby_fourier #(
    parameter N_LEAVES = 32
)(
    input  wire                 clk,
    input  wire                 rst,
    
    // Envelope input (from coefficient magnitudes)
    input  wire [15:0]          envelope [0:N_LEAVES-1],  // RMS per leaf
    input  wire [15:0]          phase_in [0:N_LEAVES-1],  // Phase hint
    
    // Control
    input  wire                 start,
    input  wire [5:0]           sample_idx,     // 0-63 within block
    
    // Output
    output reg signed [15:0]    sample_out,
    output wire                 valid
);
```

#### 3.4.2 Phase Accumulator Bank

```verilog
// 32 phase accumulators, one per leaf
reg [23:0] phase_acc [0:31];

// Frequency increment per leaf (precomputed from leaf_to_freq)
wire [23:0] freq_inc [0:31];  // ROM or computed

always @(posedge clk) begin
    if (start && sample_idx == 0) begin
        // Block start: optionally resync phases
    end
    
    for (i = 0; i < 32; i = i + 1) begin
        phase_acc[i] <= phase_acc[i] + freq_inc[i];
    end
end
```

#### 3.4.3 Sine Lookup and Accumulation

```verilog
// Sequential: process one leaf per cycle
reg [4:0] leaf_idx;
reg signed [23:0] accumulator;

wire [7:0] sin_addr = phase_acc[leaf_idx][23:16];
wire signed [15:0] sin_val;  // From BRAM LUT

sin_lut_256x16 sin_rom (
    .addr(sin_addr),
    .data(sin_val)
);

always @(posedge clk) begin
    if (start_leaf_scan) begin
        leaf_idx <= 0;
        accumulator <= 0;
    end else if (leaf_idx < 32) begin
        // Multiply envelope by sin, accumulate
        accumulator <= accumulator + (envelope[leaf_idx] * sin_val);
        leaf_idx <= leaf_idx + 1;
    end else begin
        sample_out <= accumulator[23:8];  // Scale down
    end
end
```

#### 3.4.4 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| Phase accumulators | 0 | 32 × 24-bit FFs |
| Freq increment ROM | 0 | BRAM |
| Sin LUT | 0 | BRAM (256 × 16) |
| Multiplier | 0 | DSP block |
| Accumulator | 24 | 24-bit add |
| Envelope registers | 0 | FFs |
| Control FSM | 30 | |
| Address logic | 16 | |
| **Total** | **~70 LUTs + 1 DSP** | |

---

### 3.5 η Blend Multiplexer

Crossfades between wavelet and Baby Fourier outputs.

#### 3.5.1 Interface

```verilog
module eta_blend (
    input  wire                 clk,
    input  wire signed [15:0]   wavelet_in,
    input  wire signed [15:0]   fourier_in,
    input  wire [15:0]          eta,            // Q0.16, 0=wavelet, 1=fourier
    output reg signed [15:0]    audio_out
);
```

#### 3.5.2 Implementation

```verilog
wire signed [31:0] wavelet_scaled = wavelet_in * (16'hFFFF - eta);
wire signed [31:0] fourier_scaled = fourier_in * eta;
wire signed [31:0] sum = wavelet_scaled + fourier_scaled;

always @(posedge clk) begin
    audio_out <= sum[31:16];
end
```

#### 3.5.3 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| Multipliers | 0 | 2 DSP blocks, or... |
| Multipliers (fabric) | 64 | If no DSP |
| Adder | 16 | 32-bit |
| **Total** | **~80 LUTs or 2 DSP** | |

---

### 3.6 Conditioning Frontend

Audio input preprocessing for Transformer/Decomposer modes.

#### 3.6.1 Interface

```verilog
module conditioning (
    input  wire                 clk,
    input  wire                 rst,
    input  wire signed [15:0]   audio_in,
    input  wire                 enable,
    
    // Configuration
    input  wire                 dc_block_en,
    input  wire                 gate_en,
    input  wire [15:0]          gate_threshold,
    input  wire                 slew_en,
    input  wire [15:0]          max_slew,
    
    output reg signed [15:0]    audio_out,
    output wire                 gate_open
);
```

#### 3.6.2 DC Blocker

```verilog
// y[n] = x[n] - x[n-1] + α·y[n-1]
// α = 0.995 ≈ 255/256 for ~10 Hz cutoff at 48kHz

reg signed [15:0] x_prev, y_prev;
wire signed [15:0] dc_removed = audio_in - x_prev + y_prev - (y_prev >>> 8);

always @(posedge clk) begin
    x_prev <= audio_in;
    y_prev <= dc_removed;
end
```

#### 3.6.3 Noise Gate

```verilog
// Simple threshold with hysteresis
reg [15:0] rms_estimate;
reg gate_state;

wire [15:0] abs_in = audio_in[15] ? -audio_in : audio_in;

always @(posedge clk) begin
    // Leaky integrator for RMS estimate
    rms_estimate <= rms_estimate - (rms_estimate >>> 6) + (abs_in >>> 6);
    
    // Hysteresis
    if (rms_estimate > gate_threshold + 16'd100)
        gate_state <= 1;
    else if (rms_estimate < gate_threshold - 16'd100)
        gate_state <= 0;
end

assign gate_open = gate_state;
```

#### 3.6.4 Slew Limiter

```verilog
reg signed [15:0] slew_prev;

wire signed [15:0] delta = audio_in - slew_prev;
wire signed [15:0] clamped_delta = 
    (delta > $signed(max_slew))  ? $signed(max_slew) :
    (delta < -$signed(max_slew)) ? -$signed(max_slew) : delta;

wire signed [15:0] slew_out = slew_prev + clamped_delta;

always @(posedge clk) begin
    slew_prev <= slew_out;
end
```

#### 3.6.5 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| DC blocker | 24 | 2 adds, 1 shift |
| RMS estimator | 20 | Leaky integrator |
| Gate comparators | 16 | Threshold + hysteresis |
| Slew limiter | 24 | Delta clamp |
| Mux chain | 16 | Enable selects |
| **Total** | **~100** | |

---

### 3.7 Parallel Stencil Bank

Multiple stencils for high voice counts.

#### 3.7.1 Interface

```verilog
module stencil_bank #(
    parameter N_STENCILS = 4
)(
    input  wire                 clk,
    input  wire                 rst,
    
    // Global control
    input  wire                 start_all,
    input  wire                 mode,           // FORWARD/INVERSE
    
    // Per-stencil β values
    input  wire [15:0]          beta [0:N_STENCILS-1],
    
    // Shared coefficient buffer (multi-port or arbitrated)
    // ... buffer interfaces ...
    
    // Outputs
    output wire signed [15:0]   audio_out [0:N_STENCILS-1],
    output wire                 all_done
);

    // Instantiate N_STENCILS stencil units
    genvar i;
    generate
        for (i = 0; i < N_STENCILS; i = i + 1) begin : stencils
            
            // Each stencil has its own schedule generator
            tree_generator tree_gen_i (
                .beta(beta[i]),
                // ...
            );
            
            stencil_unit stencil_i (
                // ...
            );
            
        end
    endgenerate
    
endmodule
```

#### 3.7.2 Resource Estimate (4 stencils)

| Component | LUTs | Notes |
|-----------|------|-------|
| Stencil units × 4 | 400 | 100 each |
| Tree generators × 4 | 320 | 80 each |
| Buffer arbitration | 100 | 4-way |
| Output mixing | 60 | 4-input sum |
| **Total** | **~880** | |

---

### 3.8 Mode Controller

Top-level state machine for three operating modes.

#### 3.8.1 Interface

```verilog
module mode_controller (
    input  wire                 clk,
    input  wire                 rst,
    
    // Mode selection
    input  wire [1:0]           mode,           // 0=GEN, 1=TRANS, 2=DECOMP
    
    // Block synchronization
    input  wire                 block_start,
    
    // Submodule control
    output reg                  cond_enable,
    output reg                  forward_start,
    output reg                  inverse_start,
    output reg                  fourier_enable,
    output reg                  coeffs_output_en,
    
    // Data routing
    output reg                  use_external_coeffs,
    output reg                  output_coeffs_not_audio
);
```

#### 3.8.2 State Machine

```
GENERATOR:
  block_start → inverse_start → fourier (if η>0) → output audio

TRANSFORMER:
  block_start → cond_enable → forward_start → [external symbol ops] →
              → inverse_start → fourier → output audio
  Latency: 2 blocks

DECOMPOSER:
  block_start → cond_enable → forward_start → output coefficients
```

#### 3.8.3 Resource Estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| State machine | 30 | 3 modes × ~5 states |
| Control outputs | 20 | Registered |
| **Total** | **~50** | |

---

## 4. Memory Map

### 4.1 Coefficient Buffers (BRAM)

```
Buffer 0: 0x000 - 0x0FF  (256 bytes, tree β ≈ 0.125)
Buffer 1: 0x100 - 0x1FF  (256 bytes, tree β ≈ 0.375)
Buffer 2: 0x200 - 0x2FF  (256 bytes, tree β ≈ 0.625)
Buffer 3: 0x300 - 0x3FF  (256 bytes, tree β ≈ 0.875)

Each buffer: 32 leaves × 8 coefficients × 16 bits = 512 bytes
Total: 2 KB
```

### 4.2 Sine LUT (BRAM)

```
0x000 - 0x1FF: 256 entries × 16 bits = 512 bytes
              sin(2π × addr/256) in Q1.15 format
```

### 4.3 Frequency Increment ROM (BRAM or LUT)

```
0x000 - 0x03F: 32 entries × 24 bits
              Phase increment for each leaf center frequency
              
Leaf 0:  20 Hz    → inc = 20 * 2^24 / 48000 = 6,990
Leaf 15: 632 Hz   → inc = 632 * 2^24 / 48000 = 220,893
Leaf 31: 20 kHz   → inc = 20000 * 2^24 / 48000 = 6,990,506
```

---

## 5. Register Interface

### 5.1 Control Registers

| Address | Name | Bits | Description |
|---------|------|------|-------------|
| 0x00 | MODE | [1:0] | Operating mode: 0=GEN, 1=TRANS, 2=DECOMP |
| 0x02 | BETA_GLOBAL | [15:0] | Global β parameter (Q0.16) |
| 0x04 | ETA | [15:0] | Wavelet/Fourier blend (Q0.16) |
| 0x06 | EXOTIC_GLOBAL | [7:0] | Global exotic parameter |
| 0x08 | HARMONICS | [3:0] | Number of harmonics (1-16) |
| 0x0A | COND_EN | [2:0] | [0]=DC, [1]=gate, [2]=slew |
| 0x0C | GATE_THRESH | [15:0] | Noise gate threshold |
| 0x0E | MAX_SLEW | [15:0] | Slew limiter delta |

### 5.2 Status Registers

| Address | Name | Bits | Description |
|---------|------|------|-------------|
| 0x10 | STATUS | [7:0] | [0]=busy, [1]=block_done, [2]=gate_open |
| 0x12 | VOICE_COUNT | [7:0] | Active voices this block |
| 0x14 | PEAK_LEVEL | [15:0] | Peak output level |
| 0x16 | CYCLE_COUNT | [15:0] | Cycles used this block |

### 5.3 Voice Parameter Interface

```
Voice parameters are written via streaming interface, not memory-mapped.
See Section 6.3 for voice protocol.
```

---

## 6. Interface Protocols

### 6.1 Block Synchronization

```
48 kHz sample rate, 64-sample blocks = 750 blocks/second
Block period = 1.333 ms

     ┌────┐         ┌────┐         ┌────┐
BCLK ┘    └─────────┘    └─────────┘    └─────────
     │←─ Block N ──►│←─ Block N+1 ►│

block_start pulses at start of each block
All processing must complete before next block_start
```

### 6.2 Audio I/O

```
Input (Transformer/Decomposer):
  I2S or parallel, 48 kHz, 16-bit signed
  Directly to conditioning frontend

Output (Generator/Transformer):
  I2S or parallel, 48 kHz, 16-bit signed
  From η blend mux
```

### 6.3 Voice Parameter Protocol

```
Voice injection happens at block boundaries.
Voice manager sends voice list before block_start.

VOICE_PACKET (96 bits each):
  [15:0]   freq      (Hz, Q10.6)
  [31:16]  amplitude (Q1.15)
  [47:32]  phase     (Q0.16)
  [63:48]  beta      (Q0.16, per-voice tree)
  [71:64]  exotic    (Q0.8)
  [75:72]  harmonics (1-16)
  [79:76]  reserved
  [95:80]  voice_id  (for tracking)

Protocol:
  voice_valid pulse + 96-bit data
  Repeat for all active voices
  voice_list_done pulse
  block_start pulse
```

### 6.4 Coefficient Output (Decomposer)

```
COEFF_PACKET (per leaf):
  [15:0]   coeff_rms     (energy in this leaf)
  [31:16]  coeff_phase   (phase estimate)
  [39:32]  leaf_idx
  
Output 32 packets per block (one per leaf)
Or: only leaves with energy > threshold
```

---

## 7. Timing Budget

### 7.1 Per-Block Cycle Budget

```
Clock: 50 MHz
Block period: 1.333 ms = 66,666 cycles
```

### 7.2 Operation Costs

| Operation | Cycles | Notes |
|-----------|--------|-------|
| Voice injection (64 voices) | 1,120 | 64 × 70 ÷ 4 parallel |
| Tree generation (4 trees) | 100 | Parallel |
| Wavelet synthesis (4 trees) | 1,330 | Parallel stencils |
| Baby Fourier | 320 | 32 leaves × 10 |
| η blend | 64 | 1 per sample |
| Output write | 64 | 1 per sample |
| **Total** | **~3,000** | |

### 7.3 Margin

```
Budget:    66,666 cycles
Used:       3,000 cycles
Remaining: 63,666 cycles
Margin:    22×
```

### 7.4 Transformer Mode Timing

```
Forward pass:  1,330 cycles
External ops:  (variable, not our budget)
Inverse pass:  1,330 cycles
Latency:       2 blocks minimum
```

---

## 8. Resource Summary

### 8.1 LUT Budget

| Component | LUTs |
|-----------|------|
| Stencil units × 4 | 400 |
| Tree generators × 4 | 320 |
| Coefficient injection | 236 |
| Baby Fourier | 70 |
| η blend | 80 |
| Conditioning | 100 |
| Mode controller | 50 |
| Buffer arbitration | 100 |
| Output mixing | 60 |
| Misc glue | 64 |
| **Total** | **1,480** |

### 8.2 BRAM Budget

| Buffer | Size |
|--------|------|
| Coefficient buffers × 4 | 2 KB |
| Sine LUT | 0.5 KB |
| Freq increment ROM | 96 bytes |
| Working buffer | 0.5 KB |
| **Total** | **~3.1 KB** |

### 8.3 DSP Budget

| Use | DSP Blocks |
|-----|------------|
| Baby Fourier multiply | 1 |
| η blend | 2 |
| **Total** | **3** |

### 8.4 FF (Register) Budget

| Use | FFs |
|-----|-----|
| Phase accumulators (32 × 24) | 768 |
| Pipeline registers | ~200 |
| Control state | ~100 |
| **Total** | **~1,100** |

---

## 9. Implementation Checklist

### Phase 1: Stencil Core (Week 1)
- [ ] 9.1.1 Bidirectional stencil_unit module
- [ ] 9.1.2 Stencil testbench (forward + inverse)
- [ ] 9.1.3 Verify 1 sample/cycle throughput
- [ ] 9.1.4 Verify bit-exact with Python reference

### Phase 2: Tree System (Week 1-2)
- [ ] 9.2.1 tree_generator module
- [ ] 9.2.2 β sweep test (0.0 → 1.0)
- [ ] 9.2.3 Verify all β produce valid schedules
- [ ] 9.2.4 Legacy tree ID → β mapping

### Phase 3: Coefficient Injection (Week 2)
- [ ] 9.3.1 freq_to_leaf function
- [ ] 9.3.2 Exotic ratio calculation
- [ ] 9.3.3 Multi-voice injection sequencer
- [ ] 9.3.4 Buffer allocation by β

### Phase 4: Stencil Bank (Week 2-3)
- [ ] 9.4.1 Parameterized stencil_bank module
- [ ] 9.4.2 4-stencil instantiation
- [ ] 9.4.3 Buffer arbitration
- [ ] 9.4.4 Output mixing

### Phase 5: Baby Fourier (Week 3)
- [ ] 9.5.1 Phase accumulator bank
- [ ] 9.5.2 Sine LUT integration
- [ ] 9.5.3 Envelope extraction
- [ ] 9.5.4 Sample accumulation

### Phase 6: Integration (Week 3-4)
- [ ] 9.6.1 η blend mux
- [ ] 9.6.2 Conditioning frontend
- [ ] 9.6.3 Mode controller
- [ ] 9.6.4 Full datapath integration

### Phase 7: Verification (Week 4)
- [ ] 9.7.1 Generator mode test
- [ ] 9.7.2 Transformer mode test
- [ ] 9.7.3 Decomposer mode test
- [ ] 9.7.4 64-voice stress test
- [ ] 9.7.5 β modulation test
- [ ] 9.7.6 Bit-exact Python comparison

### Phase 8: Optimization (Week 4+)
- [ ] 9.8.1 Synthesis for target FPGA
- [ ] 9.8.2 Timing closure
- [ ] 9.8.3 Resource optimization if needed
- [ ] 9.8.4 Power analysis

---

## 10. Test Vectors

### 10.1 Stencil Unit Tests

```
Test 1: Forward transform, 8 samples
  Input:  [100, 120, 105, 115, 108, 112, 103, 117]
  Expected approx: [computed...]
  Expected detail: [computed...]

Test 2: Inverse transform, verify round-trip
  Forward then inverse should reconstruct original
  Max error: 0 (perfect reconstruction)

Test 3: All-zeros input
Test 4: Impulse input
Test 5: Full-scale sine
```

### 10.2 Tree Generator Tests

```
Test 1: β = 0.5 (balanced)
  Verify schedule matches known balanced tree

Test 2: β = 0.0, 1.0 (extremes)
  Verify valid schedules

Test 3: β sweep 0.0 → 1.0 in 0.1 steps
  Verify smooth schedule transitions
```

### 10.3 Full System Tests

```
Test 1: 30-voice convergence
  Replicate convergence_crescendo through pipeline
  Compare output to Python reference

Test 2: 64-voice polyphony
  Random frequencies, verify no overflow

Test 3: β modulation
  Sweep β with LFO, verify no artifacts

Test 4: Mode switching
  Generator → Transformer → Decomposer
  Verify clean transitions
```

---

## 11. Appendix: Python Reference

The Python reference implementation is in the following files:

| File | Contents |
|------|----------|
| `stencil_reference.py` | CDF 5/3 lifting, forward + inverse |
| `tree_generator.py` | β → schedule algorithm |
| `coeff_injection.py` | Voice → coefficient mapping |
| `baby_fourier.py` | Oscillator bank synthesis |
| `wavelet_engine.py` | Full integrated model |

All Verilog must produce bit-exact results vs Python reference.

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | — | Original spec (RAM-based trees) |
| 2.0 | — | Architecture pivot: hardcoded stencils + parametric trees |

**Key changes in v2.0:**
- Trees generated from β parameter (was: loaded from RAM)
- 1,480 LUTs (was: 2,450)
- 64-128 voices (was: 18)
- Tree morphing: 50 cycles (was: 4,416)
- Tree switching: 6 cycles (was: 256)
- β modulation: native (was: impossible)
- Per-voice trees: supported (was: impossible)
