# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Public API cleanup: internal modules now hidden with `#[doc(hidden)]`
- SIMD: `multiversion` is now the default (safe autovectorization)
- SIMD: Hand-written AVX2 intrinsics available via `simd-intrinsics` feature

### Fixed
- Example `test_refine.rs` used wrong crate name

## [0.1.0] - 2024-12-27

### Added
- Initial release of mozjpeg-oxide
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

[Unreleased]: https://github.com/imazen/mozjpeg-oxide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/imazen/mozjpeg-oxide/releases/tag/v0.1.0
