# aarch64 SIMD coverage — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/neon_tiers.rs`

The crate advertises *"safe SIMD (archmage) on x86_64 (AVX2) and aarch64 (NEON)"*. This checks
that the ARM half is real and dispatching, rather than a placeholder — a distinction that
mattered elsewhere in this sweep, where an entire crate's `_neon` arms turned out to be scalar
bodies wrapped in `#[arcane]`.

## Result: both paths are real and dispatching

| kernel | dispatched | scalar reference | |
|---|---|---|---|
| forward_dct_8x8 | 12.28 ns | 20.50 ns | **1.67×** |

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
