# aarch64 SIMD coverage — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/neon_tiers.rs`

The crate advertises *"safe SIMD (archmage) on x86_64 (AVX2) and aarch64 (NEON)"*. This checks
that the ARM half is real and dispatching, rather than a placeholder — a distinction that
mattered elsewhere in this sweep, where an entire crate's `_neon` arms turned out to be scalar
bodies wrapped in `#[arcane]`.

## CRITICAL: the NEON DCT is CORRECT-BROKEN and has been disabled

`aarch64::neon::forward_dct_8x8_neon` reproduces **mozilla/mozjpeg#453** — the i16 forward-DCT
overflow that inverts entire 8×8 blocks.

CLAUDE.md documents the bug and states *"mozjpeg-rs status: Production paths use i32
intermediates — immune."* That is true on x86_64 and was **not** true here: the NEON kernel
carries the whole transform in s16 lanes (104 `s16` ops vs 72 `s32`), and the column-pass final
butterfly reaches 8 × 5056 = 40448, past `i16::MAX` (32767). Wrapping flips the sign of the
block.

Causality proven by toggling exactly that dispatch branch:

| NEON DCT | `test_issue444_deringing_overflow_pattern` | `test_issue444_across_quality_range` |
|---|---|---|
| enabled | **FAIL** — "Left half should be dark (got mean 206.0). Sign flip bug?" | **FAIL** — "Q2: left half (231.2) should be darker than right half (24.2)" |
| disabled | pass | pass |

**Why nobody saw it:** `cargo test` did not COMPILE on aarch64. Five x86-only AVX2 debug
examples used `core::arch::x86_64` and `is_x86_feature_detected!` with no arch gate, so
`--all-targets` failed to build and the two issue-444 regression tests — which exist precisely
to catch this — had never run on ARM.

**Action taken:** the aarch64 branch of `SimdOps::detect()` no longer selects the NEON DCT, and
the five examples are gated so the suite builds. 329 tests now run on ARM and pass.

**Cost:** the autovectorized scalar DCT is ~1.67× slower (20.50 vs 12.28 ns per 8×8 block).
That is the right trade against silently inverted blocks on any image with a sharp vertical
edge at Q≤57 with deringing on — text, UI captures, line art.

**To re-enable:** widen the column-pass butterfly in `aarch64/neon.rs` to s32, mirroring the
x86 production path, then flip the branch back and confirm both issue-444 tests still pass.

## Both paths are real and dispatching

| kernel | NEON | scalar reference | |
|---|---|---|---|
| forward_dct_8x8 | 12.28 ns | 20.50 ns | 1.67× (NEON now disabled — see above) |

`SimdOps::detect()` reports `dct_variant_name() == "neon_archmage"`, and
`aarch64::neon::forward_dct_8x8_neon` is genuine intrinsic code (`vld1q_s16`, `vtrnq_s16`, a
full 8×8 transpose and butterfly) — not a scalar body behind a `#[target_feature]` no-op.

## A gap I thought I found, and did not

Reading `SimdOps::detect()`, the colour-conversion selection appears to fall through to
`scalar::convert_rgb_to_ycbcr` on every non-x86 target (`src/simd/mod.rs:154`), which would
mean RGB→YCbCr runs scalar on ARM — per-pixel over every image, a far bigger cost than the
per-block DCT.

That reading was wrong. The **first** cfg branch is `feature = "fast-yuv"`, which is a DEFAULT
feature, and it routes colour through the `zenyuv` crate — which carries its own aarch64 NEON
path (`mod neon_encode`, plus a magetypes `neon` tier). The scalar arm on line 154 is only
reached with `--no-default-features`.

Recorded because the mistake is easy to repeat: in a `detect()` built from stacked `#[cfg]`
branches, the *last* assignment in source order is not the active one.

## Note on the DCT margin

1.67× is modest next to the 4–9× seen on byte-wise kernels elsewhere in this sweep, and that
is expected: an 8×8 DCT is a butterfly network with a transpose, so it is dominated by data
movement and dependent arithmetic rather than by wide independent lanes. The scalar arm is
also autovectorized (NEON is baseline on aarch64), so this measures hand-written NEON against
LLVM's best effort, not against unvectorized code.
