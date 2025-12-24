# mozjpeg-oxide

Pure Rust JPEG encoder based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg), featuring trellis quantization for optimal compression.

[![Crates.io](https://img.shields.io/crates/v/mozjpeg-oxide.svg)](https://crates.io/crates/mozjpeg-oxide)
[![Documentation](https://docs.rs/mozjpeg-oxide/badge.svg)](https://docs.rs/mozjpeg-oxide)
[![License](https://img.shields.io/crates/l/mozjpeg-oxide.svg)](LICENSE)

## Features

- **Trellis quantization** - Rate-distortion optimized coefficient selection for smaller files
- **Progressive JPEG** - Multi-scan encoding with DC-first, AC-band progression
- **Huffman optimization** - 2-pass encoding for optimal entropy coding
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Quality presets** - `max_compression()` and `fastest()` for common use cases

## Usage

```rust
use mozjpeg_oxide::{Encoder, Subsampling};

// Default encoding (trellis + Huffman optimization enabled)
let encoder = Encoder::new().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Maximum compression (progressive + trellis + optimized Huffman)
let encoder = Encoder::max_compression();
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Fastest encoding (no optimizations)
let encoder = Encoder::fastest().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Custom configuration
let encoder = Encoder::new()
    .quality(75)
    .progressive(true)
    .subsampling(Subsampling::S420)
    .optimize_huffman(true);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
```

## Compression Quality vs C mozjpeg

Benchmarked against C mozjpeg using the [Kodak](http://r0k.us/graphics/kodak/) (24 images) and [CLIC](https://www.compression.cc/) (32 images) test corpora with perceptual quality metrics.

### Methodology

Both encoders use identical settings:
- Progressive mode with default scan configuration
- Trellis quantization (AC + DC)
- Optimized Huffman tables
- 4:2:0 chroma subsampling

Quality measured using:
- **SSIMULACRA2** - Perceptual quality metric (higher = better, 90+ is excellent)
- **DSSIM** - Structural dissimilarity (lower = better)
- **BPP** - Bits per pixel (file size normalized by resolution)

### Results

| Quality | Rust BPP | C BPP | Size Δ | Rust SSIM2 | C SSIM2 | SSIM2 Δ |
|---------|----------|-------|--------|------------|---------|---------|
| 20 | 0.251 | 0.252 | -0.3% | 21.5 | 21.8 | -0.30 |
| 50 | 0.534 | 0.532 | +0.4% | 57.3 | 57.5 | -0.18 |
| 75 | 0.881 | 0.870 | +1.3% | 71.8 | 72.0 | -0.13 |
| 85 | 1.251 | 1.224 | +2.2% | 77.8 | 77.9 | -0.08 |
| 95 | 2.367 | 2.270 | +4.3% | 86.1 | 86.1 | -0.03 |

**Summary**: mozjpeg-oxide produces files 0-4% larger than C mozjpeg with nearly identical perceptual quality (SSIM2 within 0.3 points). The quality difference is imperceptible - both encoders are on the same Pareto frontier.

### Pareto Front Visualization

![SSIMULACRA2 vs BPP](../benchmark/pareto_ssimulacra2.svg)

### Reproducibility

Run the benchmark yourself using Docker:

```bash
docker build -t mozjpeg-oxide-bench benchmark/
docker run --rm -v $(pwd)/results:/results mozjpeg-oxide-bench
python3 benchmark/plot_pareto.py results/benchmark_results.csv
```

## Performance

Tested on 512x512 images in release mode:

| Configuration | vs C mozjpeg | Notes |
|---------------|--------------|-------|
| Baseline (no opts) | 7.5x slower | C has SIMD DCT |
| Trellis AC | 0.87x (faster) | |
| Max compression | 0.60x (faster) | |

Baseline encoding is slower due to lack of SIMD, but trellis quantization (the main feature) is competitive or faster

## License

BSD-3-Clause - Same license as the original mozjpeg.

## Acknowledgments

Based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg), which is itself based on libjpeg-turbo and the Independent JPEG Group's libjpeg.
