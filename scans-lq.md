# Investigation: optimize_scans divergence at low quality — RESOLVED

## Root Cause

`encode_rust()` in `src/test_encoder.rs` was missing `.optimize_scans(config.optimize_scans)`
in the `Encoder` builder chain. The Rust scan optimizer was **never called** — the encoder
always used the fixed 9-scan script regardless of the `optimize_scans` config flag.

Meanwhile, the C encoder correctly passed `optimize_scans` via FFI, so C's scan search
found simpler, more efficient scripts at low quality (4-5 scans without successive
approximation), while Rust always used the default 9-scan SA script.

## Evidence

Before fix (R=Rust optimize_scans, C=C optimize_scans):
- R(optsc) == R(fixed) at ALL quality levels — Rust optimizer was never invoked
- C correctly found smaller scripts at low Q (C saves 4.6% at Q10 vs fixed script)

After fix — Rust scan optimizer runs and finds similar scripts as C:
```
  Q  R(optsc) C(optsc)   Δopt%
 10    183834   184696  -0.47%   (was +4.18%)
 20    354621   357038  -0.68%   (was +2.40%)
 30    507224   509658  -0.48%   (was +1.54%)
 40    643993   645968  -0.31%   (was +1.11%)
 50    769538   771715  -0.28%   (was +0.77%)
 75   1263157  1259526  +0.29%   (was +0.59%)
 85   1766757  1760481  +0.36%   (was +0.41%)
 95   3218288  3209303  +0.28%   (was +0.40%)
```

At low quality, Rust is now **smaller** than C (the scan optimizer works well).

## Fix

One-line fix in `src/test_encoder.rs:134`:
```rust
.optimize_scans(config.optimize_scans)
```

## Remaining Observations

Some images still show Rust choosing different scripts than C (different Al levels
or frequency splits). This is expected — the scan search is a greedy heuristic and
can find different local optima. The per-image max deviation is ~1.6% at Q55, which
is within acceptable range.
