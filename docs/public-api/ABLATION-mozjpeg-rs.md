# mozjpeg-rs public-API ablation report

**Date:** 2026-06-11
**Snapshot commit:** 2cad252
**Crate analyzed:** `mozjpeg-rs` (478 default / 722 all-features items)
**Grep template:** `ugrep -r --include="*.rs" --include="*.toml" "<symbol>" /home/lilith/work/ --exclude-dir=target --exclude-dir=.jj`

## Consumer context

No org-internal consumers found: `ugrep -r --include="*.toml" "mozjpeg" /home/lilith/work/` returns only the repo itself, `mozjpeg-rs-sys` and `codec-eval` dev deps. mozjpeg-rs is a standalone published crate; the expected consumers per the Crate Index are imageflow and comparison harnesses.

## Summary

**0 items flagged for action.**

## Structure

mozjpeg-rs's surface has four main concerns:

1. **Core encode API** — `Encoder` (builder), `StreamingEncoder`, `EncodingStream<W>`, `Encode` trait. Exhaustive encode methods: `encode_rgb`, `encode_gray`, `encode_rgba`, `encode_ycbcr_planar`, plus `_strided`, `_to_writer`, `_with_stop`, `_cancellable` variants. Well-structured builder; all methods are documented.

2. **Configuration types** — `Preset`, `Subsampling`, `TrellisConfig`, `TrellisSpeedMode`, `PixelDensity`, `DensityUnit`, `Limits`, `ResourceEstimate`. All are Copy/Clone/Debug/Default with appropriate derives. `TrellisConfig` exposes pub fields for all trellis knobs; this is intentional — the struct is the config value, and callers are expected to read/write fields directly (also has a method API for builder-style chaining).

3. **Optional feature surface:**
   - `mozjpeg-sys-config` feature: `compat::CMozjpeg`, `compat::ConfigError`, `compat::ConfigWarnings` (also re-exported at top level as `mozjpeg_rs::CMozjpeg` etc.)
   - `zencodec` feature: `codec::MozjpegEncoderConfig`, `codec::MozjpegEncodeJob`, `codec::MozjpegEncoder` (also re-exported at top level)
   - `imgref` feature: `imgref_ext::EncodeablePixel` trait

4. **Error/result types** — `Error` (`#[non_exhaustive]`), `Result<T>` alias.

## Observations (informational, no action needed)

1. **`#[doc(hidden)] pub mod` internal modules** — `bitstream`, `color`, `color_avx2`, `consts`, `dct`, `deringing`, `entropy`, `fast_entropy`, `huffman`, `progressive`, `quant`, `sample`, `simd`, `trellis`. All are `#[doc(hidden)]` in lib.rs. They appear as `pub` at the module level so tests and FFI comparison harnesses can reach internal types. The `cargo public-api` snapshot correctly omits them from the rendered surface (they do not appear in `mozjpeg-rs.txt`). Intentional.

2. **`#[doc(hidden)] pub mod test_encoder` / `pub mod corpus`** — Both are explicitly marked `#[doc(hidden)]` with a comment "Test support modules - hidden from public API." Neither appears in the snapshot. Intentional.

3. **`Encoder::simd_ops(self, SimdOps) -> Self`** — This method is public and documented ("Override SIMD operations dispatch for testing alternative DCT implementations"). `SimdOps` lives in the `#[doc(hidden)] pub mod simd` module. Callers can reach it via `mozjpeg_rs::simd::SimdOps` but it won't appear in rendered docs. The method exists specifically to test experimental paths like `SimdOps::avx2_i16()` (the known DCT overflow path documented in mozilla/mozjpeg#453). This is an explicitly documented testing escape hatch, not an accidental leak. KEEP.

4. **`TrellisConfig` pub fields** — `enabled`, `dc_enabled`, `delta_dc_weight`, `eob_opt`, `freq_split`, `lambda_log_scale1`, `lambda_log_scale2`, `num_loops`, `q_opt`, `speed_mode`, `use_lambda_weight_tbl`, `use_scans_in_trellis`. These are all exposed. CLAUDE.md documents which fields are not-yet-wired vs implemented. The struct is Copy + has builder methods for the common cases; pub fields are needed for callers who want to inspect or modify individual knobs. Intentional for an expert-config struct. The CLAUDE.md lists `freq_split`, `q_opt`, `use_lambda_weight_tbl`, `num_loops` as "not fully wired" — but that's a correctness/docs issue, not a visibility issue. KEEP.

5. **Duplicate top-level re-exports under all-features** — The snapshot shows both `mozjpeg_rs::CMozjpeg` and `mozjpeg_rs::compat::CMozjpeg` (same for `MozjpegEncoderConfig` etc.). These are `pub use compat::*` re-exports in lib.rs, all with full docstrings and feature gates. `cargo public-api` records both the canonical path and the re-export path. Not a leak; this is the ergonomic top-level access pattern documented in the CLAUDE.md API examples. Intentional.

6. **`CompatWarnings::has_custom_markers: bool`, `has_exif: bool`, `has_icc_profile: bool` pub fields** — Returned from `CMozjpeg::configure_cinfo`. Callers read them to know which metadata could not be applied. Intentional diagnostic output struct.

7. **`pub unsafe fn CMozjpeg::configure_cinfo`** — The only `unsafe fn` in the public surface. Takes `&mut mozjpeg_sys::jpeg_compress_struct`. Documented as unsafe because direct cinfo mutation bypasses mozjpeg-sys's safety invariants. This is behind `mozjpeg-sys-config` feature, gated appropriately. Intentional.

8. **`imgref_ext::EncodeablePixel` impls for `[u8; 3]`, `[u8; 4]`, `u8`, `rgb::Rgb<u8>`, `rgb::Rgba<u8>`, `rgb::Gray<u8>`** — All appear in the snapshot as impl blocks because the trait is public and the impls are on stdlib/external types. Not a concern; this is the standard Rust trait-impl pattern for the `imgref` integration.

## Flagged items

| # | Item | Category | Proposal | Confidence |
|---|------|----------|----------|------------|
| — | (none) | — | — | — |

**0 flagged. 0 % of surface.**

## Digest

mozjpeg-rs has a well-organized public surface. The core encode API (`Encoder`/`StreamingEncoder`/`Encode`) is clean. Internal modules are correctly `#[doc(hidden)]` and absent from the snapshot. Test-support modules (`test_encoder`, `corpus`) are `#[doc(hidden)]` and absent from the snapshot. `Encoder::simd_ops` is an explicit, documented testing escape hatch for experimental DCT paths. `TrellisConfig` pub fields are intentional for expert-configuration access. Top-level re-exports from `compat` and `codec` are feature-gated with full documentation. No leaks, no accidental exposures, no zero-consumer internals mistakenly surfaced.
