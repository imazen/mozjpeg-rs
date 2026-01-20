# Safety Review: mozjpeg-rs

**Date:** 2026-01-19
**Reviewed by:** Claude Code
**Branch:** safety-review

## Executive Summary

The codebase has a well-designed safety architecture with `#![deny(unsafe_code)]` at the crate level. Unsafe code is confined to specific modules that explicitly allow it. The dynamic CPU dispatch is handled correctly via runtime feature detection before calling SIMD intrinsics.

**Risk Level:** Low
**Critical Issues:** 0
**Medium Issues:** 1 (misleading safety comment)
**Low Issues:** 3 (documentation/style)

---

## 1. Unsafe Code Inventory

### Modules with `#![allow(unsafe_code)]`

| Module | Purpose | Lines of unsafe | Risk |
|--------|---------|-----------------|------|
| `simd/x86_64/avx2.rs` | AVX2 intrinsics for DCT and color conversion | ~200 | Low |
| `simd/x86_64/entropy.rs` | SSE2 intrinsics for entropy encoding | ~150 | Low |
| `dct.rs` (inline `avx2` module) | AVX2 intrinsics for DCT | ~180 | Low |
| `color_avx2.rs` | AVX2 intrinsics for color conversion | ~350 | Low |
| `compat.rs` | FFI wrapper for C mozjpeg | ~400 | Low |
| `test_encoder.rs` | Test FFI calls | ~80 | Low (test only) |

**Note:** `lib.rs` has `#![deny(unsafe_code)]` at crate level, enforcing opt-in for unsafe.

---

## 2. Dynamic CPU Dispatch Analysis

### Dispatch Mechanisms

1. **`multiversion` crate (primary safe dispatch)**
   - Used in `dct.rs:forward_dct_8x8()` and `simd/scalar.rs:convert_rgb_to_ycbcr()`
   - Automatically generates multiple versions for different targets
   - Runtime dispatch handled by the crate (safe)
   - Targets: `x86_64+avx2`, `x86_64+sse4.1`, `x86+avx2`, `x86+sse4.1`, `aarch64+neon`

2. **`is_x86_feature_detected!` macro (manual runtime dispatch)**
   - Used in `simd/mod.rs:SimdOps::detect()` (lines 123, 134, 163)
   - Checks AVX2 availability before selecting function pointers
   - **Status: SAFE** - Runtime check happens before function pointer assignment

3. **`#[cfg(target_feature = "avx2")]` (compile-time dispatch)**
   - Used in `dct.rs:forward_dct()` (lines 889-900)
   - **Status: SAFE** - Code only compiled when target supports AVX2

### SimdOps Dispatch Table

```rust
// simd/mod.rs:117-146
pub fn detect() -> Self {
    #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
    let dct_fn: ForwardDctFn = if is_x86_feature_detected!("avx2") {
        x86_64::avx2::forward_dct_8x8      // ← Only called after runtime check
    } else {
        scalar::forward_dct_8x8
    };
    // ...
}
```

**Verdict:** All dispatch paths are safe. Runtime checks precede unsafe intrinsic calls.

---

## 3. Issues Found

### MEDIUM: Misleading Safety Comment

**Location:** `src/simd/x86_64/avx2.rs:304-307`

```rust
pub fn forward_dct_8x8(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // SAFETY: This module is only compiled when target_feature = "avx2"  ← INCORRECT
    unsafe { forward_dct_8x8_avx2(samples, coeffs) }
}
```

**Problem:** The comment claims the module is only compiled with `target_feature = "avx2"`, but the `x86_64` submodule is compiled unconditionally on x86_64. The function is `pub`, meaning it could theoretically be called directly.

**Actual Safety:** The function is safe in practice because:
1. `SimdOps::detect()` checks `is_x86_feature_detected!("avx2")` at runtime
2. The `simd` module is `#[doc(hidden)]` (not part of public API)
3. `SimdOps::avx2_intrinsics()` returns `Option<Self>` and checks at runtime

**Recommendation:** Fix the comment:
```rust
// SAFETY: Callers must ensure AVX2 is available via is_x86_feature_detected!("avx2")
// or by going through SimdOps::detect() which performs this check.
```

### LOW: Typo in Comment

**Location:** `src/color_avx2.rs:385`

```rust
if is_x86_feature_detected!("avx2") {
    / SAFETY: We just checked for AVX2 support   ← Missing leading slash
```

### LOW: `std::mem::transmute` Usage

**Locations:**
- `simd/x86_64/avx2.rs:427-429`
- `simd/x86_64/entropy.rs:225`

**Pattern:**
```rust
let y_arr: [i32; 8] = core::mem::transmute(y);  // __m256i → [i32; 8]
```

**Assessment:** Safe - both types are 32 bytes with compatible alignment.

**Recommendation:** Consider using `_mm256_storeu_si256` or dedicated extract intrinsics for clarity, though current code is correct.

### LOW: Bounds Not Explicitly Checked in Color Conversion

**Location:** `simd/x86_64/avx2.rs:341-378`

The `convert_rgb_to_ycbcr_avx2_inner` function relies on slice bounds checking rather than explicit length validation. While safe (Rust panics on OOB), explicit validation would provide clearer error messages.

---

## 4. FFI Safety Analysis

### C mozjpeg FFI (`compat.rs`)

**Patterns Used:**

1. **Input validation** (lines 261-267):
   ```rust
   if rgb.len() != expected_size {
       return Err(Error::BufferSizeMismatch { ... });
   }
   ```

2. **Struct initialization:**
   ```rust
   let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
   ```
   **Status:** Safe for C structs with all-zero valid state.

3. **Memory management** (lines 340-342):
   ```rust
   if !outbuffer.is_null() {
       libc::free(outbuffer as *mut libc::c_void);
   }
   ```
   **Status:** Proper cleanup of C-allocated memory.

4. **Pointer arithmetic** (line 326):
   ```rust
   row_pointer[0] = rgb.as_ptr().add(row_offset);
   ```
   **Status:** Safe - bounds validated earlier.

**No issues found in FFI code.**

---

## 5. Recommendations

### Immediate (before next release)

1. **Fix misleading safety comment** in `simd/x86_64/avx2.rs:305`
2. **Fix typo** in `color_avx2.rs:385`

### Future Improvements

1. Consider making `simd::x86_64::avx2` module `pub(crate)` instead of `pub` to prevent direct external calls
2. Add explicit length checks with descriptive panic messages in SIMD hot paths
3. Consider using `bytemuck` for SIMD type conversions instead of `transmute`

---

## 6. CPU Dispatch Summary

| Function | Dispatch Method | Safety Check |
|----------|-----------------|--------------|
| `forward_dct_8x8` (multiversion) | Compile-time + runtime | Automatic (multiversion crate) |
| `SimdOps::detect()` DCT | Runtime | `is_x86_feature_detected!("avx2")` |
| `SimdOps::detect()` color | Runtime | `is_x86_feature_detected!("avx2")` |
| `forward_dct()` (dct.rs) | Compile-time | `#[cfg(target_feature = "avx2")]` |
| `convert_rgb_to_ycbcr` (color_avx2) | Runtime | `is_x86_feature_detected!("avx2")` |

**All paths are safe.** No SIGILL risk in production use.

---

## Conclusion

The codebase demonstrates good safety practices:
- Crate-level `#![deny(unsafe_code)]` with explicit opt-in
- Proper runtime feature detection before SIMD intrinsics
- Correct FFI memory management
- Input validation on public API boundaries

The one medium issue (misleading comment) should be addressed for code clarity but does not represent an actual safety vulnerability.
