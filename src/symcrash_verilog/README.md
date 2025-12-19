# SymCrash Verilog - CDF 5/3 Wavelet Synthesis

FPGA implementation of the CDF 5/3 wavelet packet synthesis pipeline.

## Overview

This module performs wavelet packet synthesis using the CDF 5/3 lifting scheme.
It takes 256 Q1.15 coefficients (32 leaves × 8 coefficients) and produces 64
Q1.15 audio samples.

**Key Features:**
- Bit-exact match with Python reference implementation
- No multipliers - only adders and shifters
- No LUTs for transcendentals - pure integer arithmetic
- 5 tree types (balanced, harmonic, texture, transient, drone)
- Configurable pipelining for timing closure

## Architecture

```
coeff_in[256] ──► Leaf Processing ──► Level 5→4 ──► Level 4→3 ──► ... ──► Level 1→0 ──► samples_out[64]
   (32×8)           (32×2)            (16×4)        (8×8)                  (1×64)
```

### CDF 5/3 Inverse Lifting Step

Each level applies the inverse CDF 5/3 lifting:

1. **Undo Update** (compute even samples):
   ```
   x[2i] = s[i] - ((d[i-1] + d[i] + 2) >> 2)
   ```

2. **Undo Predict** (compute odd samples):
   ```
   x[2i+1] = d[i] + ((x[2i] + x[2i+2] + 1) >> 1)
   ```

Boundary conditions use symmetric extension: `d[-1] = d[0]`, `x[2N] = x[2N-2]`.

## Files

| File | Description |
|------|-------------|
| `sat16.v` | Saturation module for Q1.15 arithmetic |
| `cdf53_inverse_lift.v` | Single-level inverse lifting |
| `wavelet_synth_balanced.v` | Full synthesis for Tree 0 (balanced) |
| `wavelet_synth.v` | Generic synthesis with state machine |
| `tree_rom.v` | Tree structure ROM tables (all 5 types) |
| `tb_cdf53_inverse_lift.v` | Testbench for inverse lifting |
| `generate_test_vectors.py` | Python script to generate golden vectors |

## Usage

### Simulation

```bash
# Generate test vectors from Python
cd src/symcrash_verilog
python generate_test_vectors.py

# Run Verilog simulation (Icarus Verilog example)
iverilog -o tb_lift sat16.v cdf53_inverse_lift.v tb_cdf53_inverse_lift.v
vvp tb_lift
```

### Synthesis

The modules are designed for standard FPGA synthesis. For Xilinx:

```tcl
read_verilog sat16.v
read_verilog cdf53_inverse_lift.v
read_verilog wavelet_synth_balanced.v
read_verilog tree_rom.v
synth_design -top wavelet_synth_balanced
```

## Resource Estimates

For Tree 0 (balanced) on a typical FPGA:

| Resource | Count | Notes |
|----------|-------|-------|
| LUTs | ~800 | Adders, saturation logic |
| FFs | ~1100 | Pipeline registers |
| DSPs | 0 | No multipliers needed |
| BRAMs | 0 | ROM fits in LUTs |

## Q1.15 Format

All values use Q1.15 fixed-point:
- 1 sign bit + 15 fractional bits
- Range: [-1.0, +0.99997] mapped to [-32768, +32767]
- Saturation prevents overflow

## Tree Types

| ID | Name | Description |
|----|------|-------------|
| 0 | Balanced | Uniform resolution, all leaves at depth 5 |
| 1 | Harmonic | Wider bass bands, detailed highs |
| 2 | Texture | High-frequency detail for texture sounds |
| 3 | Transient | Mid-band emphasis for attacks |
| 4 | Drone | Low-frequency detail for sustained bass |

## Verification

The Python reference is in `src/symcrash_python_ap/wavelet_packet_trees.py`.
The `generate_test_vectors.py` script produces bit-exact test cases.

To verify bit-exactness:
1. Run Python to generate expected outputs
2. Run Verilog simulation
3. Compare results

## License

Same as parent project.
