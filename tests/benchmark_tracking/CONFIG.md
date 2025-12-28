# Benchmark Tracking Configuration

## Purpose

Track file size and perceptual quality (DSSIM) across commits to detect
regressions and improvements in compression efficiency.

## Test Image

- **File**: `tests/images/1.png`
- **Dimensions**: 512x512
- **Format**: 8-bit RGB PNG
- **Content**: Natural photo (suitable for real-world compression testing)

## Quality Levels Tested

20 quality values: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99

## Encoder Configuration

### Rust (mozjpeg-oxide)

```rust
Encoder::new()
    .quality(Q)
    .subsampling(Subsampling::S420)
    .progressive(false)           // Baseline mode for consistent comparison
    .optimize_huffman(true)       // Enable Huffman optimization
    .trellis(TrellisConfig::default())  // Trellis enabled (AC + DC)
    .overshoot_deringing(true)    // Deringing enabled
```

### C mozjpeg Baseline

```c
jpeg_set_defaults(&cinfo);
jpeg_c_set_int_param(&cinfo, JINT_BASE_QUANT_TBL_IDX, 3);  // ImageMagick tables
jpeg_set_quality(&cinfo, Q, TRUE);

// 4:2:0 subsampling
cinfo.comp_info[0].h_samp_factor = 2;
cinfo.comp_info[0].v_samp_factor = 2;
cinfo.comp_info[1].h_samp_factor = 1;
cinfo.comp_info[1].v_samp_factor = 1;
cinfo.comp_info[2].h_samp_factor = 1;
cinfo.comp_info[2].v_samp_factor = 1;

// Optimizations
cinfo.optimize_coding = TRUE;                              // Huffman optimization
jpeg_c_set_bool_param(&cinfo, JBOOLEAN_TRELLIS_QUANT, 1);  // AC trellis
jpeg_c_set_bool_param(&cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1); // DC trellis
jpeg_c_set_bool_param(&cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);
```

## Metrics

- **File Size**: Bytes of encoded JPEG
- **DSSIM**: Structural dissimilarity (lower is better, 0 = identical)
  - Computed using the `dssim` crate
  - Compares decoded JPEG against original PNG

## Result Format

Results stored in `results/` as JSON files named by commit hash:

```json
{
  "commit": "abc1234",
  "timestamp": "2024-12-27T12:00:00Z",
  "image": "tests/images/1.png",
  "results": [
    {"quality": 5, "size": 12345, "dssim": 0.0123},
    ...
  ]
}
```

## Baseline

`baseline_c_mozjpeg.json` contains C mozjpeg results for comparison.
This baseline should not change unless the C mozjpeg version changes.
