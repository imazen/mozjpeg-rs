# API Improvement Notes

Analysis of [mozjpeg-rust PR #49](https://github.com/ImageOptim/mozjpeg-rust/pull/49) for potential mozjpeg-rs improvements.

## Implementation Status

| Feature | Status | Commit |
|---------|--------|--------|
| Strided pixel data support | ✅ Done | `feat: add strided encoding support` |
| imgref integration | ✅ Done | `feat: add imgref integration` |
| Generic pixel types | ⏳ Deferred | Covered by imgref feature |

## What mozjpeg-rs already has (good)

- **Preset enum** - `BaselineFastest`, `BaselineBalanced`, `ProgressiveBalanced`, `ProgressiveSmallest`
- **Builder pattern** - `.quality()`, `.subsampling()`, `.progressive()`, etc.
- **Type-safe enums** - `Subsampling`, `QuantTableIdx`
- **No stateful API gotchas** - We don't wrap C's stateful `jpeg_compress_struct`, so no parameter ordering issues

## Implemented Improvements

### 1. Strided pixel data support ✅

Added `encode_rgb_strided()` and `encode_gray_strided()` methods:

```rust
// Memory-aligned buffer (rows padded to 256 bytes)
let stride = 256;
let buffer: Vec<u8> = vec![128; stride * 100];
let jpeg = encoder.encode_rgb_strided(&buffer, 100, 100, stride)?;

// Crop without copy - point into larger buffer
let crop_data = &full_image[crop_offset..];
let jpeg = encoder.encode_rgb_strided(crop_data, crop_w, crop_h, full_stride)?;
```

**Features:**
- Fast path when stride == row_bytes (zero overhead)
- Validates stride >= minimum required
- New `Error::InvalidStride` variant for validation errors

### 2. imgref integration ✅

Added optional `imgref` feature with `encode_imgref()` method:

```toml
[dependencies]
mozjpeg-rs = { version = "0.5", features = ["imgref"] }
```

```rust
use imgref::ImgVec;
use rgb::RGB8;

let img = ImgVec::new(pixels, 640, 480);
let jpeg = encoder.encode_imgref(img.as_ref())?;
```

**Supported pixel types:**
- `RGB<u8>` / `[u8; 3]` → Color JPEG
- `RGBA<u8>` / `[u8; 4]` → Color JPEG (alpha discarded)
- `Gray<u8>` / `u8` → Grayscale JPEG

**Benefits:**
- Type-safe pixel formats (compile-time distinction)
- Automatic stride handling (subimages work automatically)
- No dimension mix-ups (width/height baked into type)

### 3. Generic pixel types (covered by imgref)

The imgref feature provides this functionality via the `EncodeablePixel` trait.
Direct generic support on `encode_rgb()` deferred since imgref covers the use case.

## What we DON'T need from PR #49

1. **Parameter ordering fixes** - We don't have this problem because we don't wrap C's stateful API
2. **Type-safe color space combinations** - Our API is simpler (RGB→YCbCr always for color, Gray for grayscale)
3. **Raw MCU mode** - Not needed for our use case (encoder-only crate)

## Remaining ideas (low priority)

- `encode_rgb_to_writer_strided()` - strided variant for writer API (add if needed)
- 16-bit pixel support via imgref - `RGB<u16>` with precision downsampling
