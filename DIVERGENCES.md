# Divergences from C mozjpeg

Where mozjpeg-rs output differs from C mozjpeg, and why. Use it to tell an
intended difference from a parity bug: if a difference you hit is not listed
here, treat it as a bug.

Two C references are in play:

- **Upstream mozjpeg** (4.x, what the `mozjpeg-sys` crate vendors). The
  comparisons in `tests/ffi_validation.rs` and `tests/preset_parity.rs` run
  against it.
- **The imazen fork** (`imazen/mozjpeg`, built at `../mozjpeg`). It adds
  `trellis_speed_level` and test exports, and is the oracle for
  `tests/exact_trellis_parity.rs`.

mozjpeg-rs has two trellis modes, which differ in how closely they track C:

- `TrellisMode::Optimized` (default) favors speed and file size.
- `TrellisMode::MozjpegExact` reproduces C's emission and trellis pass
  structure byte for byte, quirks included. The exceptions are listed below.

## Byte-identical to C

These have tests and are expected to stay byte-identical:

| Configuration | C reference | Test |
|---|---|---|
| Baseline and progressive, no trellis, MCU-aligned 4:2:0 | upstream | `tests/parity_benchmark.rs` (Kodak) |
| Grayscale output from RGB input (`Subsampling::Gray`) | upstream, `jpeg_set_colorspace(JCS_GRAYSCALE)` | `ffi_validation::test_gray_from_rgb_matches_c_grayscale` |
| RGB output (`JpegColorSpace::Rgb`) | upstream, `jpeg_set_colorspace(JCS_RGB)` | `ffi_validation::test_rgb_color_space_matches_c_jcs_rgb` |
| `MozjpegExact`, trellis off, standard tables (C `-notrellis`) | upstream | `ffi_validation::test_exact_mode_notrellis_standard_tables_match_c` |
| `MozjpegExact { speed_level: 0 }` | upstream | `ffi_validation` color-space tests |
| `MozjpegExact` (speed level 7) | imazen fork | `tests/exact_trellis_parity.rs` |

The `ffi_validation` comparisons set `pixel_density(PixelDensity::aspect_ratio(1, 1))`,
because the default JFIF density differs (see below).

## Fixed C bugs

These are places where C writes a broken file and mozjpeg-rs deliberately
does not. Everywhere C's output is valid, mozjpeg-rs output stays
byte-identical.

### Exact trellis with `optimize_coding = FALSE` can emit undecodable files

- **Applies to:** `TrellisMode::MozjpegExact` with trellis enabled,
  `optimize_huffman(false)`, baseline.
- **What C does:** C's trellis passes run as single-component scans, and the
  statistics they gather overwrite the component's Huffman table slots.
  Without `optimize_coding`, C emits whatever those slots hold at the end.
  Those tables were gathered from component 0's blocks alone. The real
  interleaved scan can need symbols they never saw:
  - Under 4:2:0, right-edge and bottom dummy blocks, and luma DC deltas
    taken in MCU order rather than row order.
  - Under `JCS_RGB`, G and B share R's slot.

  libjpeg-turbo's encoder doesn't check for missing codes, so it writes
  zero-length codes and the file fails to decode, e.g. 37×29 at 4:2:0, or a
  512×512 photo in RGB.
- **What mozjpeg-rs does:** after the trellis passes it counts the real
  scan's symbols. Any slot whose table can't code them is replaced with the
  optimal table for that scan, which is the table `optimize_coding = TRUE`
  would emit. Coefficients are unchanged; only the DHT differs. When C's
  tables already cover the scan, nothing changes. See
  `cover_main_scan` in `src/trellis_exact.rs`.
- **Test:** `ffi_validation::test_exact_trellis_standard_tables_always_decodable`
  covers 4:2:0, 4:4:4, RGB and gray across sizes, qualities and restart
  intervals. It requires byte equality wherever C's file decodes, and a
  decodable Rust file everywhere. The `baseline+!opt` cases in
  `tests/exact_trellis_parity.rs` apply the same rule against the fork.

## Differences by design

| Area | mozjpeg-rs | C mozjpeg |
|---|---|---|
| **Default JFIF density** | 72×72 dpi (unit 1) | 1:1 aspect ratio, no unit (unit 0). Set `PixelDensity::aspect_ratio(1, 1)` to match. |
| **Progressive with `optimize_huffman(false)`** | Uses the standard Huffman tables, in both trellis modes. | Forces `optimize_coding = TRUE` for progressive. |
| **`optimize_scans` with `JpegColorSpace::Rgb`** | Runs the scan search with the YCbCr-shaped candidate set. On a 37×29 test image it picked 4 scans (658 bytes). | `jpeg_search_progression` only handles YCbCr and grayscale, so it falls back to the 13-scan all-purpose script (953 bytes). Without `optimize_scans`, both use the all-purpose script. |
| **Trellis in `Optimized` mode** | Adaptive search limits (`TrellisSpeedMode`) and its own rate-distortion choices; typically 0.05–0.80% smaller files. | Use `MozjpegExact` for C's decisions. |
| **`MozjpegExact` speed level** | Defaults to `trellis_speed_level` 7, the imazen fork's default. | Upstream mozjpeg has no speed limiting. Use `MozjpegExact { speed_level: 0 }` to match it; at 7, high-entropy blocks differ. |
| **Edges of non-MCU-aligned subsampled images (`Optimized`)** | Pads by edge replication and runs the DCT on the padding. Decoded pixels near the right and bottom edges can differ, e.g. 37×29 at 4:2:0. | Synthesizes dummy blocks (zero AC, copied DC) and downsamples chroma over the padded width. `MozjpegExact` reproduces this. |
| **Marker layout (`Optimized`)** | Groups the standard DHTs into one segment in its own order. Bytes differ, pixels don't. | Per-component DHT order. `MozjpegExact` matches. |
| **Input smoothing (`Optimized`)** | Smooths in the RGB domain before color conversion. | Smooths the converted planes inside downsampling (`jcsample.c`). `MozjpegExact` matches. |
| **cjpeg's `-quality` sampling rule** | Applied only in `MozjpegExact` and only when `subsampling` wasn't set explicitly. | Part of cjpeg's command line (`rdswitch.c`), not the library. |
| **Color conversion with `fast_color(true)`** | ±1 rounding difference, opt-in. | Exact `jccolor.c` arithmetic. The default `fast_color(false)` matches. |
| **Presets** | `Encoder::default()` / `baseline_optimized()` are baseline. `Preset::BaselineFastest` uses the ImageMagick quant tables. | `jpeg_set_defaults()` uses `JCP_MAX_COMPRESSION`, which is progressive with `optimize_scans`; `Encoder::max_compression()` matches it. `JCP_FASTEST` uses the Annex K tables. |

## Not implemented

- Arithmetic coding.
- Multipass trellis (`use_scans_in_trellis`). C's own benchmarks show
  slightly larger files for it.
- `trellis_q_opt`, which is also a placeholder in C.

## C quirks reproduced on purpose (`MozjpegExact`)

These look like bugs but are C's real behavior. Exact mode keeps them so its
output stays byte-identical:

- **Only component 0 is trellised without `optimize_coding`.** C never
  re-selects the scan component for these trellis passes, so it trellises
  component 0 `num_components` times, and components 1..N keep normal
  quantization (see `run_exact_trellis`).
- **`huffval` order.** In the non-optimized trellis tables, `huffval` is
  ordered by each symbol's original code length (`gen_optimal_table_c`).

## Adding to this file

Give each new divergence a test that pins it down: an exact comparison where
mozjpeg-rs matches C, and an assertion on the intended behavior where it
doesn't. Mark the code with a `DIVERGENCE from C (see DIVERGENCES.md)`
comment.
