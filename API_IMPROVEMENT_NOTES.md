# API Improvement Notes

Analysis of [mozjpeg-rust PR #49](https://github.com/ImageOptim/mozjpeg-rust/pull/49) for potential mozjpeg-rs improvements.

## What mozjpeg-rs already has (good)

- **Preset enum** - `BaselineFastest`, `BaselineBalanced`, `ProgressiveBalanced`, `ProgressiveSmallest`
- **Builder pattern** - `.quality()`, `.subsampling()`, `.progressive()`, etc.
- **Type-safe enums** - `Subsampling`, `QuantTableIdx`
- **No stateful API gotchas** - We don't wrap C's stateful `jpeg_compress_struct`, so no parameter ordering issues

## Improvements to consider

### 1. Strided pixel data support (HIGH VALUE)

**Current:** No stride support - assumes tightly packed rows
```rust
encoder.encode_rgb(&pixels, width, height)?;
```

**Proposed:** Add strided variants
```rust
encoder.encode_rgb_strided(&pixels, width, height, stride)?;
encoder.encode_rgb_to_writer_strided(&pixels, width, height, stride, &mut writer)?;
```

**Use case:** Memory-aligned buffers, cropping without copy, GPU textures

### 2. imgref integration (HIGH VALUE, feature-gated)

**Current:** Separate width/height params, easy to swap
```rust
encoder.encode_rgb(&pixels, width, height)?;  // which is which?
```

**Proposed:** Accept `ImgRef` directly
```rust
// Feature: imgref
encoder.encode(img.as_ref())?;  // dimensions baked in, can't mess up
```

Benefits:
- Type safety (`ImgRef<RGB8>` vs `ImgRef<u8>`)
- Stride support built-in (subimages work automatically)
- No dimension mix-ups

### 3. Generic pixel type support (MEDIUM VALUE)

**Current:** Only `&[u8]`
```rust
fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;
```

**Proposed:** Accept various pixel types via trait
```rust
fn encode<P: RgbPixels>(&self, pixels: P, width: u32, height: u32) -> Result<Vec<u8>>;

// Supports:
encoder.encode(&rgb8_slice, w, h)?;           // &[RGB8]
encoder.encode(&array_slice, w, h)?;          // &[[u8; 3]]
encoder.encode(&flat_bytes, w, h)?;           // &[u8] (existing)
```

### 4. One-liner convenience on JpegConfig (LOW VALUE)

PR #49 adds:
```rust
let jpeg = JpegConfig::from_preset(Preset::ProgressiveBalanced, 85.0)
    .encode_rgb(&pixels, 640, 480)?;
```

We already have this pattern:
```rust
let jpeg = Encoder::new(Preset::ProgressiveBalanced)
    .quality(85)
    .encode_rgb(&pixels, 640, 480)?;
```

No change needed - our API is already ergonomic.

## What we DON'T need from PR #49

1. **Parameter ordering fixes** - We don't have this problem because we don't wrap C's stateful API
2. **Type-safe color space combinations** - Our API is simpler (RGB→YCbCr always for color, Gray for grayscale)
3. **Raw MCU mode** - Not needed for our use case (encoder-only crate)

## Implementation priority

1. **Strided support** - Easy win, ~50 lines, enables cropping/aligned buffers
2. **imgref feature** - Medium effort, high ergonomics payoff
3. **Generic pixel types** - Nice to have, lower priority

## API sketch for imgref

```rust
// Cargo.toml
[features]
imgref = ["dep:imgref", "dep:rgb"]

// src/encode.rs
#[cfg(feature = "imgref")]
impl Encoder {
    /// Encode from imgref with automatic dimension/stride handling
    pub fn encode<'a, P>(&self, img: imgref::ImgRef<'a, P>) -> Result<Vec<u8>>
    where
        P: AsRgbBytes,
    {
        let (width, height) = (img.width() as u32, img.height() as u32);
        let stride = img.stride();
        // ... call internal strided encoder
    }
}

// Trait for pixel types
pub trait AsRgbBytes {
    fn as_rgb_bytes(&self) -> &[u8];
    fn bytes_per_pixel() -> usize;
}

impl AsRgbBytes for rgb::RGB8 { ... }
impl AsRgbBytes for [u8; 3] { ... }
```
