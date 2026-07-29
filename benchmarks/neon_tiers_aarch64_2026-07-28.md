# aarch64 SIMD coverage — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/neon_tiers.rs`

The crate advertises *"safe SIMD (archmage) on x86_64 (AVX2) and aarch64 (NEON)"*. This checks
that the ARM half is real and dispatching, rather than a placeholder — a distinction that
mattered elsewhere in this sweep, where an entire crate's `_neon` arms turned out to be scalar
bodies wrapped in `#[arcane]`.

## The NEON DCT reproduced mozjpeg#453 — found, fixed, kept

`aarch64::neon::forward_dct_8x8_neon` inverted entire 8×8 blocks. CLAUDE.md documents the bug
and says *"Production paths use i32 intermediates — immune"* — true on x86_64, not here: the
NEON kernel carried the whole transform in s16 lanes.

**Localised to the pass-2 (column) final butterfly.** Pass-2 inputs are pass-1 outputs, already
~5056 with overshoot deringing, so `tmp10` reaches 20224 (still fine in i16) but the final
`tmp10 ± tmp11` spans ±40448, past `i16::MAX`. Pass 1 is safe — its inputs are level-shifted
samples ≤158.

**Fix:** widen only that add/sub to i32 and narrow with `vrshrn_n_s32`.

| | ns per 8×8 block | correct |
|---|---|---|
| scalar (autovectorized) | 21.23 | yes |
| NEON, original | 12.28 | **NO — inverts blocks** |
| NEON, widened pass-2 | **13.15** | **yes** |

Correctness cost ~7%; the kernel is still **1.61×** over scalar. The first response was to
disable it outright — the right reflex, since silently inverted blocks beat any speedup — but
the actual fix was one butterfly, so the disable was reverted.

**Why nobody saw it:** `cargo test` did not COMPILE on aarch64. Five x86-only AVX2 debug
examples used `core::arch::x86_64` and `is_x86_feature_detected!` with no arch gate, so
`--all-targets` failed to build and the two issue-444 regression tests — which exist precisely
to catch this — had never run on ARM. Same shape as the ungated AVX2 externs that blocked
zenav1-svt's C-parity gates in this same sweep.

Liveness-checked: reverting the widening (NEON still enabled) fails both issue-444 tests;
restoring it passes them, plus encode_tests 42/42 and lib 287/287.

## Both paths are real and dispatching

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
