# mozjpeg-rs Development Guide

## Project Overview

Rust port of Mozilla's mozjpeg JPEG encoder, following the jpegli-rs methodology.

**Scope**: Encoder only + mozjpeg extensions (trellis, progressive scans, deringing)
**API**: Idiomatic Rust (not C-compatible)
**Validation**: FFI dual-execution against C mozjpeg

## Current Status

**147 tests passing** (127 unit + 5 codec comparison + 10 FFI comparison + 5 FFI validation)

### Compression Results vs C mozjpeg

**With matching settings (progressive + trellis + Huffman opt):**

| Mode | Rust | C mozjpeg | Ratio | Notes |
|------|------|-----------|-------|-------|
| Progressive + Trellis | 71,176 bytes | 72,721 bytes | **0.98x** | Rust is 2.1% smaller! |
| Baseline + Trellis | 73,834 bytes | 72,721 bytes | 1.02x | C defaults to progressive |

**Small test images (16x16):**
| Quality | Rust Size | C Size | Ratio | Rust PSNR | C PSNR |
|---------|-----------|--------|-------|-----------|--------|
| Q50 | 491 bytes | 498 bytes | **0.99x** | 40.33 dB | 40.16 dB |
| Q75 | 518 bytes | 558 bytes | **0.93x** | 44.90 dB | 45.20 dB |
| Q85 | 590 bytes | 631 bytes | **0.94x** | 47.32 dB | 47.49 dB |

**Note:** C mozjpeg uses JCP_MAX_COMPRESSION profile by default which enables progressive
mode. Use `Encoder::max_compression()` for equivalent behavior.

### Completed Layers
- Layer 0: Constants, types, error handling
- Layer 1: Quantization tables, Huffman table construction
- Layer 2: Forward DCT, color conversion, chroma subsampling
- Layer 3: Bitstream writer with byte stuffing
- Layer 4: Entropy encoder, **trellis quantization (AC + DC)**
- Layer 5: **Progressive scan generation (integrated)**
- Layer 6: Marker emission, **Full encoder pipeline**
- **FFI Validation**: Granular comparison tests against C mozjpeg

### Working Encoder
The encoder produces valid JPEG files with mozjpeg-quality compression:
```rust
use mozjpeg::{Encoder, Subsampling, TrellisConfig};

// Default encoding (trellis + Huffman optimization enabled)
let encoder = Encoder::new().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Maximum compression (progressive + trellis + optimized Huffman)
let encoder = Encoder::max_compression();
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Fastest encoding (no optimizations)
let encoder = Encoder::fastest().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Custom configuration
let encoder = Encoder::new()
    .quality(75)
    .progressive(true)
    .subsampling(Subsampling::S420)
    .trellis(TrellisConfig::default())
    .optimize_huffman(true);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
```

### Implemented Features
- **Baseline JPEG encoding** - Standard sequential DCT
- **Progressive JPEG encoding** - Multi-scan with DC first, then AC bands
- **Trellis quantization** - Rate-distortion optimized AC + DC quantization (mozjpeg core feature)
- **DC trellis optimization** - Dynamic programming across blocks for optimal DC encoding
- **Huffman table optimization** - 2-pass encoding for optimal tables
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Quality presets** - `max_compression()` and `fastest()`

### Remaining Work
- **Bug: Progressive encoder MCU column issue** - Progressive encoding produces quality degradation
  when images have multiple MCU columns (width > 16px with 4:2:0). Tall images (1 MCU column)
  work correctly. See `mozjpeg/tests/codec_comparison.rs:test_progressive_debug_64x64` for details.
- Optional: EOB optimization integration (`trellis_eob_opt` - disabled by default in C mozjpeg)
- Optional: deringing, arithmetic coding
- Performance optimization (SIMD)

**Baseline mode works well!** With baseline + trellis + Huffman optimization, Rust produces
files with quality matching C mozjpeg. Progressive mode has a known bug affecting multi-column
MCU images that needs investigation.

## Workflow Rules

### Commit Strategy
- **Commit when new tests pass** - After fixing/completing a module
- **Commit when new tests are added** - Even if they're failing (documents expected behavior)
- Write descriptive commit messages explaining what was ported

### Validation Approach
- Validate equivalence **layer by layer**, not just end-to-end
- Use `mozjpeg-sys` from crates.io for basic FFI validation
- Use `mozjpeg-sys-local` (in workspace) for granular internal function testing
  - Builds from local `../mozjpeg` C source with test exports
  - C code has been instrumented with `mozjpeg_test_*` functions
  - Tests in `mozjpeg/tests/ffi_comparison.rs` compare Rust vs C implementations

### Golden Rule: Never Delete Instrumentation
**NEVER delete tests, FFI comparisons, or instrumentation code.** These are essential for:
- Validating correctness against C mozjpeg
- Catching regressions during development
- Documenting expected behavior
- Ensuring byte-exact parity with C implementation

If a test seems obsolete, comment it out with explanation rather than deleting.

## Key Learnings

### mozjpeg Specifics
1. **Default quant tables**: mozjpeg uses ImageMagick tables (index 3), not JPEG Annex K (index 0)
2. **Quality scaling**: Q50 = 100% scale factor (use tables as-is)
3. **DCT scaling**: Output is scaled by factor of 8 (sqrt(8) per dimension)
4. **Huffman pseudo-symbol**: Symbol 256 ensures no real symbol gets all-ones code

### Trellis Quantization (Critical!)
The trellis algorithm requires raw DCT coefficients (scaled by 8):
1. **Lambda calculation**: `lambda = 2^scale1 / (2^scale2 + block_norm)` with `lambda_base = 1.0`
2. **Per-coefficient weights**: `weight[i] = 1/quantval[i]^2`
3. **Quantization divisor**: `q = 8 * quantval[i]` (includes DCT scaling)
4. **Distortion**: `(candidate * q - original)^2 * lambda * weight`
5. **Cost**: `rate + distortion` where rate is Huffman code size + value bits

### Implementation Notes
1. **Huffman tree construction**: Use sentinel values carefully to avoid overflow
   - `FREQ_INITIAL_MAX = 1_000_000_000` for comparison
   - `FREQ_MERGED = 1_000_000_001` for merged nodes
2. **Bitstream stuffing**: 0xFF bytes ALWAYS require 0x00 stuffing in entropy data
3. **Bit buffer**: Use 64-bit buffer, flush when full, pad with 1-bits at end
4. **Progressive encoding**:
   - DC scans can be interleaved (multiple components)
   - AC scans must be single-component
   - Successive approximation uses Ah/Al for bit refinement

### Testing Patterns
1. Use `#[cfg(test)]` modules within each source file
2. FFI validation tests in `tests/ffi_validation.rs`
3. Test both positive cases and error conditions

## Architecture

```
mozjpeg-rs/
├── mozjpeg/                    # Main library crate
│   ├── src/
│   │   ├── lib.rs              # Module exports, public API
│   │   ├── consts.rs           # Layer 0: Constants, tables, markers
│   │   ├── types.rs            # Layer 0: ColorSpace, ScanInfo, etc.
│   │   ├── error.rs            # Error types
│   │   ├── quant.rs            # Layer 1: Quantization tables
│   │   ├── huffman.rs          # Layer 1: Huffman table construction
│   │   ├── dct.rs              # Layer 2: Forward DCT (Loeffler)
│   │   ├── color.rs            # Layer 2: RGB→YCbCr conversion
│   │   ├── sample.rs           # Layer 2: Chroma subsampling
│   │   ├── bitstream.rs        # Layer 3: Bit-level I/O
│   │   ├── entropy.rs          # Layer 4: Huffman encoding
│   │   ├── trellis.rs          # Layer 4: Trellis quantization
│   │   ├── progressive.rs      # Layer 5: Progressive scans
│   │   ├── marker.rs           # Layer 6: JPEG markers
│   │   └── encode.rs           # Layer 6: Encoder pipeline
│   └── tests/
│       ├── ffi_validation.rs   # crates.io mozjpeg-sys tests
│       └── ffi_comparison.rs   # Local FFI granular comparison
├── mozjpeg-sys/                # Local FFI bindings (builds from ../mozjpeg)
│   ├── build.rs                # CMake integration
│   └── src/lib.rs              # FFI declarations + test exports
└── ../mozjpeg/                 # Instrumented C mozjpeg fork
    ├── mozjpeg_test_exports.c  # Test export implementations
    └── mozjpeg_test_exports.h  # Test export declarations
```

## Build & Test

```bash
cargo test                           # Run all tests
cargo test huffman                  # Run specific module tests
cargo test --test ffi_validation    # Run crates.io FFI tests
cargo test --test ffi_comparison    # Run local FFI comparison tests
cargo test -p mozjpeg-sys-local     # Run local mozjpeg-sys tests
```

## Dependencies

- `mozjpeg-sys = "2.2"` (dev) - FFI validation against C mozjpeg
- `mozjpeg-sys-local` (workspace) - Local FFI with granular test exports
- `bytemuck = "1.14"` - Safe transmutes (for future SIMD)
