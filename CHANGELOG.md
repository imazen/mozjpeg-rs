# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-02

### Breaking Changes
- **`Encoder::new()` now requires a `Preset` argument** - Explicit preset selection replaces the previous boolean progressive parameter
- Crate renamed from `mozjpeg-oxide` to `mozjpeg-rs`

### Added
- **Preset enum** with four encoding modes:
  - `BaselineFastest` - No optimizations, maximum speed
  - `BaselineBalanced` - Baseline with trellis + Huffman optimization
  - `ProgressiveBalanced` - Progressive with optimizations (default)
  - `ProgressiveSmallest` - Maximum compression with optimize_scans
- **`mozjpeg-sys-config` feature** - Configure a C mozjpeg encoder from Rust `Encoder` settings via `Encoder::configure_sys()`
- **Smoothing filter** - `.smoothing(0-100)` for noise reduction on dithered images
- **Robidoux quant table alias** - `QuantTableIdx::Robidoux` (alias for MssimTuned)
- **CLI-style method names** - `.quant_method()`, `.dct_method()` aliases
- **Grayscale progressive JPEG support**
- **Trellis speed optimization** - Adaptive search limiting for high-entropy blocks
- **EOB cross-block optimization** - Opt-in via `TrellisConfig::eob_optimization(true)`

### Fixed
- **AC refinement ZRL encoding** - Fixed decoder errors on 8/24 Kodak images with successive approximation
- Progressive encoding now works correctly for all image sizes including non-MCU-aligned dimensions

### Changed
- Public API cleanup: internal modules now hidden with `#[doc(hidden)]`
- SIMD: `multiversion` is now the default (safe autovectorization)
- SIMD: Hand-written AVX2 intrinsics available via `simd-intrinsics` feature
- Encoder settings matrix added to README

## [0.2.5] - 2024-12-30

### Fixed
- Example `test_refine.rs` used wrong crate name

## [0.2.4] - 2024-12-29

### Added
- Grayscale progressive JPEG support
- Trellis speed optimization (`speed_level`)

## [0.2.3] - 2024-12-28

### Fixed
- AC refinement ZRL loop must run for both `temp>1` and `temp==1` cases
- Bytewise parity analysis documentation

## [0.2.2] - 2024-12-27

### Fixed
- Match C mozjpeg progressive encoding defaults exactly
- Per-scan Huffman tables for all progressive modes

## [0.1.0] - 2024-12-27

### Added
- Initial release of mozjpeg-rs
- Pure Rust JPEG encoder based on Mozilla's mozjpeg
- **Trellis quantization** - Rate-distortion optimized AC and DC coefficient selection
- **Progressive JPEG encoding** - Multi-scan with DC-first, AC-band progression
- **Huffman optimization** - 2-pass encoding for optimal entropy coding
- **Overshoot deringing** - Reduces ringing artifacts near hard edges
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0, 4:4:0 modes
- **Quality presets** - `Encoder::new()`, `Encoder::max_compression()`, `Encoder::fastest()`
- **Optimize scans** - Tries multiple scan configurations for progressive mode
- SIMD acceleration via `multiversion` crate (AVX2, SSE4.1, NEON)
- Runtime CPU feature detection and dispatch
- Comprehensive test suite (177+ tests)
- FFI validation against C mozjpeg
- Multi-platform CI (Linux, macOS, Windows on x64 and ARM64)

### Performance
- Trellis mode: Rust is ~10% faster than C mozjpeg
- With matching settings, produces files 2% smaller than C mozjpeg
- AVX2 intrinsics provide additional ~15% DCT speedup (opt-in feature)

### Compatibility
- MSRV: Rust 1.89.0
- Platforms: Linux, macOS, Windows (x64 and ARM64)
- Output compatible with all standard JPEG decoders

[0.3.0]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/imazen/mozjpeg-rs/compare/v0.1.0...v0.2.2
[0.1.0]: https://github.com/imazen/mozjpeg-rs/releases/tag/v0.1.0
