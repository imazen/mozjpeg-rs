# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

### Added
- **RGBX8 descriptor in zencodec dispatch** — `RGBX8_SRGB` added to `ENCODE_DESCRIPTORS` alongside `RGBA8_SRGB`, routing through `encode_rgba_with_stop` since mozjpeg reads 4 bytes per pixel and ignores byte 3 regardless of whether it's alpha or padding (ed47868)
  - Handled in both the one-shot encode path and the `push_rows` / `finish` streaming path
  - Tests confirm RGBX produces byte-identical output to RGBA and to RGB when colour bytes match

### Changed
- Routed all README badges through shields.io with `?style=flat-square` for consistent sizing (9f10ab2)

### Fixed
- `fast_color` doctest in `src/encode.rs` now passes `Preset::default()` to `Encoder::new()` to match the current API signature (9f10ab2)

## [0.9.1] - 2026-04-14

### Added
- **RGBA8 encoding** — Native 4-channel input support without intermediate RGB buffer (7a55b85, 6f0ffcc)
  - `Encoder::encode_rgba()`, `encode_rgba_with_stop()`, `encode_rgba_to_writer()`
  - `color::convert_rgba_to_ycbcr_c_compat()` — direct RGBA→YCbCr, alpha ignored
  - For BGRA input, swizzle to RGBA first with `garb::bytes::bgra_to_rgba_inplace`
- **zencodec RGBA8 support** — `RGBA8_SRGB` added to supported encode descriptors (d5d60ab, 06c5e4b)
  - `encode()` routes to native RGBA path based on pixel descriptor
  - `encode_srgba8()` uses native RGBA path for contiguous data, `garb::rgba_to_rgb_strided` for strided
  - `push_rows()` / `finish()` handle RGB8, RGBA8, and GRAY8
- **garb dependency** (optional, via `zencodec` feature) for SIMD-optimized pixel format conversions

### Changed
- Swapped `yuv` crate for `zenyuv` in the `fast-yuv` path (2667a98) — internal Imazen crate, same 15-bit fixed-point BT.601 output, enables future Sharp YUV opt-in
- Refactored `encode_rgb_to_writer` to share downsample+MCU+encode pipeline via `encode_ycbcr_planes_to_writer` helper (6f0ffcc)

## [0.8.0] - 2026-02-08

### Added
- **ARM NEON SIMD support** — Safe SIMD implementation for aarch64 using archmage
  - Automatic runtime detection and dispatch
  - Byte-identical output to scalar implementation
  - Zero unsafe code
- **100% safe Rust** — Achieved `#![forbid(unsafe_code)]` with no exceptions
  - Uses archmage 0.5 for safe SIMD intrinsics
  - Uses safe_unaligned_simd 0.2.4 for safe load/store operations
- **CI coverage for ARM platforms** — Tests on native arm64 runners plus QEMU for aarch64, armv7, and i686
- **WASM support** — Builds successfully for wasm32-wasip1 with SIMD128

### Changed
- Updated to archmage 0.5 (from 0.4)
- Updated to safe_unaligned_simd 0.2.4 (from 0.2.3)
- Trellis encoding is now **6% faster** than C mozjpeg (was 7%, updated measurements)
- CI now tests 15 platform configurations including ARM and WASM

### Fixed
- Conditional compilation for forbid(unsafe_code) when mozjpeg-sys-config feature is enabled
- wasm32 unused import warnings

## [0.7.0] - 2026-02-02

### Added
- **`fast_color(bool)`** - Ergonomic API for color conversion mode
  - `fast_color(true)` — ~40% faster RGB→YCbCr using `yuv` crate (±1 rounding vs C)
  - `fast_color(false)` — exact C mozjpeg parity, bytewise identical output (default)
- **Byte-exact C mozjpeg parity** — Baseline and Progressive modes now produce identical output to C mozjpeg (0.00% file size delta)
- **AVX2-accelerated C-compatible color conversion** — 3.6 Gpix/s, exact match to C mozjpeg's `jccolor.c`

### Changed
- Default color conversion now produces **byte-exact** output matching C mozjpeg
- `kodak_dir()` now returns the actual Kodak corpus directory (was aliasing `cid22_dir()`)
- With trellis enabled, Rust produces **smaller files** than C mozjpeg (-0.05% to -0.47%)
- Trellis encoding is now **7% faster** than C mozjpeg

### Deprecated
- `c_compat_color(bool)` — use `fast_color(bool)` instead (clearer semantics)

## [0.6.0] - 2026-02-02

### Added
- **`imgref` feature (default)** - Type-safe encoding with `encode_imgref()` accepting `ImgRef<P>` directly
  - Supports `RGB<u8>`, `RGBA<u8>` (alpha discarded), `Gray<u8>`, `[u8; 3]`, `[u8; 4]`, `u8`
  - Automatic stride handling for subimages
  - No dimension mix-ups (width/height baked into type)
- **Strided encoding** - `encode_rgb_strided()` and `encode_gray_strided()` for:
  - Memory-aligned buffers (e.g., 64-byte aligned rows)
  - Cropping without copy (point into larger buffer)
  - GPU texture formats with padding
- **`EncodeablePixel` trait** - Implement for custom pixel types
- **`Error::InvalidStride`** - New error variant for stride validation

### Changed
- `imgref` and `rgb` crates are now default dependencies (can be disabled with `default-features = false`)

## [0.5.5] - 2026-02-01

### Changed
- Recommend `mozjpeg` crate for users who need both encoding and decoding

## [0.5.4] - 2026-01-13

### Fixed
- Clippy warnings in benchmarks and examples

### Changed
- Removed all unsafe from SIMD modules (now uses safe wrappers)

## [0.5.3] - 2026-01-10

### Changed
- Migrated from archmage 0.1 to 0.4 with safe_unaligned_simd

## [0.5.1] - 2026-01-08

### Fixed
- Scan optimizer parity improvements with C mozjpeg

## [0.5.0] - 2026-01-05

### Added
- Improved scan optimization algorithm
- Better progressive scan selection

### Fixed
- optimize_scans parity with C mozjpeg (now within ±0.4%)

## [0.4.1] - 2025-01-03

### Added
- **`encode_ycbcr_planar_strided()`** - Encode strided YCbCr data without extra copies. Accepts separate stride for each plane.

## [0.4.0] - 2025-01-03

### Added
- **`encode_ycbcr_planar()`** - Encode pre-converted planar YCbCr data directly, bypassing RGB-to-YCbCr conversion. Supports all subsampling modes (4:4:4, 4:2:2, 4:2:0).
- **`CMozjpeg::encode_ycbcr_planar()`** - Same capability for C mozjpeg wrapper
- **Decoder round-trip validation** - CI now validates encoded JPEGs against multiple decoders

### Changed
- **`configure_sys()` renamed to `to_c_mozjpeg()`** - Clearer naming for C mozjpeg interop
- `sys-local` is now an optional dev dependency gated by feature flag

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
- **AC refinement ZRL encoding** - Fixed decoder errors on 8/24 test corpus images with successive approximation
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

[0.9.1]: https://github.com/imazen/mozjpeg-rs/compare/v0.9.0...v0.9.1
[0.6.0]: https://github.com/imazen/mozjpeg-rs/compare/v0.5.5...v0.6.0
[0.5.5]: https://github.com/imazen/mozjpeg-rs/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/imazen/mozjpeg-rs/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/imazen/mozjpeg-rs/compare/v0.5.1...v0.5.3
[0.5.1]: https://github.com/imazen/mozjpeg-rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/imazen/mozjpeg-rs/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/imazen/mozjpeg-rs/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/imazen/mozjpeg-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/imazen/mozjpeg-rs/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/imazen/mozjpeg-rs/compare/v0.1.0...v0.2.2
[0.1.0]: https://github.com/imazen/mozjpeg-rs/releases/tag/v0.1.0
