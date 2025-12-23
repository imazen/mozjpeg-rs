# mozjpeg-rs Development Guide

## Project Overview

Rust port of Mozilla's mozjpeg JPEG encoder, following the jpegli-rs methodology.

**Scope**: Encoder only + mozjpeg extensions (trellis, progressive scans, deringing)
**API**: Idiomatic Rust (not C-compatible)
**Validation**: FFI dual-execution against C mozjpeg

## Current Status

**134 tests passing** (116 unit + 8 FFI comparison + 6 mozjpeg-sys + 4 ffi_validation)

### Completed Layers
- Layer 0: Constants, types, error handling
- Layer 1: Quantization tables, Huffman table construction
- Layer 2: Forward DCT, color conversion, chroma subsampling
- Layer 3: Bitstream writer with byte stuffing
- Layer 4: Entropy encoder, trellis quantization
- Layer 5: Progressive scan generation
- Layer 6: Marker emission
- **FFI Validation**: Granular comparison tests against C mozjpeg

### Remaining Work
- High-level Encoder struct/builder API
- End-to-end encoding pipeline
- Optional: deringing, arithmetic coding

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

## Key Learnings

### mozjpeg Specifics
1. **Default quant tables**: mozjpeg uses ImageMagick tables (index 3), not JPEG Annex K (index 0)
2. **Quality scaling**: Q50 = 100% scale factor (use tables as-is)
3. **DCT scaling**: Output is scaled by factor of 64 (8 per dimension)
4. **Huffman pseudo-symbol**: Symbol 256 ensures no real symbol gets all-ones code

### Implementation Notes
1. **Huffman tree construction**: Use sentinel values carefully to avoid overflow
   - `FREQ_INITIAL_MAX = 1_000_000_000` for comparison
   - `FREQ_MERGED = 1_000_000_001` for merged nodes
2. **Bitstream stuffing**: 0xFF bytes ALWAYS require 0x00 stuffing in entropy data
3. **Bit buffer**: Use 64-bit buffer, flush when full, pad with 1-bits at end
4. **Trellis quantization**: Core mozjpeg innovation
   - Cost = Rate + Lambda * Distortion
   - Lambda calculated from block energy and quant table
   - Per-coefficient lambda weights = 1/q^2
5. **Progressive encoding**:
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
│   │   └── marker.rs           # Layer 6: JPEG markers
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
