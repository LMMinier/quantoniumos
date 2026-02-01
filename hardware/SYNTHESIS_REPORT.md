# RFTPU FPGA Synthesis Report

**Date:** February 1, 2026  
**Tool:** Yosys 0.33  
**Target:** iCE40UP5K (WebFPGA compatible)

## Files

| File | Lines | Purpose | WebFPGA Compatible |
|------|-------|---------|-------------------|
| `fpga_top.sv` | 1086 | Full 16-mode RFTPU | ⚠️ Use `.v` version |
| `fpga_top_minimal.sv` | 500 | 4-mode verified only | ⚠️ Use `.v` version |
| `fpga_top_webfpga.v` | 500 | **WebFPGA ready** | ✅ YES |

## Synthesis Results

### Full Version (fpga_top.sv)
```
Resource        Count    iCE40UP5K    Utilization
─────────────────────────────────────────────────
SB_LUT4         686      5,280        13.0%
SB_DFF*         372      5,280         7.0%
SB_CARRY        366      -            -
SB_RAM40_4K     4        30           13.3%
```

### Minimal/WebFPGA Version (fpga_top_webfpga.v)
```
Resource        Count    iCE40UP5K    Utilization
─────────────────────────────────────────────────
SB_LUT4         454      5,280         8.6%
SB_DFF*         189      5,280         3.6%
SB_CARRY        159      -            -
SB_RAM40_4K     1        30            3.3%
```

## WebFPGA Upload Instructions

1. Go to https://beta.webfpga.io/dashboard
2. Upload `hardware/fpga_top_webfpga.v`
3. Click "Synthesize"
4. Wait for bitstream generation
5. Flash to device

## Modes Available (Minimal Version)

| Mode | Name | Function |
|------|------|----------|
| 0 | RFT-GOLDEN | Canonical Golden Ratio Transform |
| 1 | RFT-CASCADE | H3 Hybrid Compression |
| 2 | SIS-HASH | Lattice-based Hash |
| 3 | QUANTUM-SIM | Symbolic Quantum Engine |

## Known Issues

1. **File Extension:** WebFPGA requires `.v` not `.sv`
2. **Size Limit:** Free tier may have LUT limits - use minimal version
3. **Memory Warning:** "Replacing memory with registers" is expected (small arrays)

## Verification

```bash
# Local synthesis test
cd hardware
yosys -p "read_verilog fpga_top_webfpga.v; synth_ice40 -top fpga_top; stat"
```
