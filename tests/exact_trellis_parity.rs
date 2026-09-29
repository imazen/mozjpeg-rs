//! Byte-exactness tests: `TrellisMode::MozjpegExact` vs the patched C mozjpeg
//! oracle (`../mozjpeg`, the imazen fork with the SIMD overshoot fix).
//!
//! The oracle is invoked through `../mozjpeg/build/cjpeg-static` with
//! `JSIMD_FORCENONE=1`, which disables SIMD so C uses the scalar i32 ISLOW
//! forward DCT — the same exact arithmetic Rust's DCT computes on every
//! tier. (The patched C SIMD path *saturates* at i16 on deringing-overshoot
//! blocks, which is a different result than scalar/exact arithmetic; scalar
//! C is the mathematically exact reference for byte parity.)
//!
//! Tests are skipped (not failed) when the oracle binary is missing.

use mozjpeg_rs::{Encoder, PixelDensity, Subsampling, TrellisConfig, TrellisMode};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn cjpeg_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mozjpeg/build/cjpeg-static");
    p.exists().then_some(p)
}

/// Compile (once per test binary run) and return the `optimize_coding=FALSE`
/// baseline oracle driver. No `cjpeg` flag combination produces that config:
/// `-baseline` still uses JCP_MAX_COMPRESSION defaults (optimize_coding=TRUE)
/// and `-revert` disables trellis entirely. The driver clears the scan script
/// and optimize_coding after `jpeg_set_defaults` — see
/// `tests/oracle/trellis_noopt_oracle.c`.
fn noopt_oracle_path() -> Option<PathBuf> {
    use std::sync::OnceLock;
    static ORACLE: OnceLock<Option<PathBuf>> = OnceLock::new();
    ORACLE
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let src = root.join("tests/oracle/trellis_noopt_oracle.c");
            let libjpeg = root.join("../mozjpeg/build/libjpeg.a");
            if !src.exists() || !libjpeg.exists() {
                return None;
            }
            let out_dir = root.join("target/oracle");
            let _ = std::fs::create_dir_all(&out_dir);
            let out = out_dir.join("trellis_noopt_oracle");
            let status = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
                .args([
                    "-O2",
                    "-I",
                    "../mozjpeg",
                    "-I",
                    "../mozjpeg/build",
                    src.to_str().unwrap(),
                    libjpeg.to_str().unwrap(),
                    "-lm",
                    "-o",
                    out.to_str().unwrap(),
                ])
                .current_dir(&root)
                .status();
            match status {
                Ok(s) if s.success() => Some(out),
                _ => None,
            }
        })
        .clone()
}

/// Run a C oracle binary on raw pixels via stdin PPM/PGM.
fn encode_oracle(
    path: &PathBuf,
    rgb_or_gray: &[u8],
    width: u32,
    height: u32,
    grayscale: bool,
    args: &[String],
) -> Vec<u8> {
    let mut child = Command::new(path)
        .args(args)
        .env("JSIMD_FORCENONE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn cjpeg-static");

    let (magic, cpp) = if grayscale {
        (b"P5", 1usize)
    } else {
        (b"P6", 3usize)
    };
    let mut stdin = child.stdin.take().unwrap();
    let header = format!(
        "{}\n{} {}\n255\n",
        std::str::from_utf8(magic).unwrap(),
        width,
        height
    );
    stdin.write_all(header.as_bytes()).unwrap();
    assert_eq!(rgb_or_gray.len(), (width * height) as usize * cpp);
    stdin.write_all(rgb_or_gray).unwrap();
    drop(stdin);

    let out = child.wait_with_output().expect("oracle failed");
    assert!(out.status.success(), "oracle exited with {:?}", out.status);
    out.stdout
}

/// Run the cjpeg-static oracle on raw pixels via stdin PPM/PGM.
fn encode_c(
    rgb_or_gray: &[u8],
    width: u32,
    height: u32,
    grayscale: bool,
    args: &[String],
) -> Vec<u8> {
    let path = cjpeg_path().expect("cjpeg-static oracle not built");
    encode_oracle(&path, rgb_or_gray, width, height, grayscale, args)
}

/// Run the nonoptimized-trellis oracle driver on raw pixels via stdin PPM/PGM.
fn encode_c_noopt(
    rgb_or_gray: &[u8],
    width: u32,
    height: u32,
    grayscale: bool,
    args: &[String],
) -> Vec<u8> {
    let path = noopt_oracle_path().expect("trellis_noopt_oracle not built");
    encode_oracle(&path, rgb_or_gray, width, height, grayscale, args)
}

fn diff_report(c: &[u8], r: &[u8]) -> String {
    if c == r {
        return "identical".into();
    }
    let n = c.len().min(r.len());
    let first = (0..n).find(|&i| c[i] != r[i]);
    let mut s = format!("sizes C={} Rust={}", c.len(), r.len());
    if let Some(i) = first {
        let lo = i.saturating_sub(8);
        s += &format!(
            "; first diff at byte {}: C={:02x?} Rust={:02x?}",
            i,
            &c[lo..(i + 8).min(c.len())],
            &r[lo..(i + 8).min(r.len())]
        );
    }
    s
}

fn exact_config() -> TrellisConfig {
    TrellisConfig::default().mode(TrellisMode::mozjpeg_exact())
}

/// Assert byte-exact equality; on failure decode-compare so we see whether
/// pixels also differ or just markers/table layout.
fn assert_bytes(name: &str, c: &[u8], r: &[u8]) {
    if c == r {
        return;
    }
    let rep = diff_report(c, r);
    panic!("{name}: JPEG byte mismatch ({rep})");
}

// ============================================================================
// Test images
// ============================================================================

fn gen_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.push((x * 255 / w.max(1)) as u8);
            v.push((y * 255 / h.max(1)) as u8);
            v.push(((x + y) * 255 / (w + h).max(1)) as u8);
        }
    }
    v
}

fn gen_noise(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = 0x9e3779b9u32;
    for _ in 0..w * h {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push((s & 0xff) as u8);
        v.push((s >> 8 & 0xff) as u8);
        v.push((s >> 16 & 0xff) as u8);
    }
    v
}

/// Sharp black/white checkerboard — exercises deringing and high-entropy AC.
fn gen_checker(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let p = if ((x / 4) + (y / 4)) % 2 == 0 {
                255u8
            } else {
                0
            };
            v.extend_from_slice(&[p, p, p]);
        }
    }
    v
}

/// Color blocks — exercises chroma trellis.
fn gen_colorblocks(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    let cols = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ];
    for y in 0..h {
        for x in 0..w {
            let c = cols[((x / 8) + (y / 8)) % cols.len()];
            v.extend_from_slice(&c);
        }
    }
    v
}

fn to_gray(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .take(w * h)
        .map(|c| ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8) as u8)
        .collect()
}

struct Img {
    name: &'static str,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn test_images() -> Vec<Img> {
    let mut imgs = vec![
        Img {
            name: "gradient64",
            rgb: gen_gradient(64, 64),
            w: 64,
            h: 64,
        },
        Img {
            name: "noise48",
            rgb: gen_noise(48, 48),
            w: 48,
            h: 48,
        },
        Img {
            name: "checker64",
            rgb: gen_checker(64, 64),
            w: 64,
            h: 64,
        },
        Img {
            name: "colorblocks64",
            rgb: gen_colorblocks(64, 64),
            w: 64,
            h: 64,
        },
        // Non-MCU-aligned: exercises right-edge + bottom dummy synthesis.
        Img {
            name: "gradient51x37",
            rgb: gen_gradient(51, 37),
            w: 51,
            h: 37,
        },
        Img {
            name: "noise33x19",
            rgb: gen_noise(33, 19),
            w: 33,
            h: 19,
        },
        // Single MCU.
        Img {
            name: "tiny8x8",
            rgb: gen_gradient(8, 8),
            w: 8,
            h: 8,
        },
        Img {
            name: "flat32",
            rgb: vec![128u8; 32 * 32 * 3],
            w: 32,
            h: 32,
        },
    ];
    // Bundled photographic image, if present.
    let png = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/images/1.png");
    if let Ok(file) = std::fs::File::open(&png) {
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        if let Ok(mut reader) = decoder.read_info() {
            let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
            if let Ok(info) = reader.next_frame(&mut buf) {
                let rgb: Vec<u8> = match info.color_type {
                    png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
                    png::ColorType::Rgba => buf[..info.buffer_size()]
                        .chunks(4)
                        .flat_map(|c| [c[0], c[1], c[2]])
                        .collect(),
                    _ => Vec::new(),
                };
                if !rgb.is_empty() {
                    imgs.push(Img {
                        name: "bundled_png",
                        rgb,
                        w: info.width,
                        h: info.height,
                    });
                }
            }
        }
    }
    imgs
}

// ============================================================================
// Config matrix
// ============================================================================

#[test]
fn exact_baseline_opt_color() {
    if cjpeg_path().is_none() {
        eprintln!("skipping: cjpeg-static oracle not built");
        return;
    }
    for img in test_images() {
        for &(q, samp_c, samp_r, sname) in &[
            (75u8, "2x2", Subsampling::S420, "s420"),
            (90, "1x1", Subsampling::S444, "s444"),
            (90, "2x1", Subsampling::S422, "s422"),
            (50, "2x2", Subsampling::S420, "s420"),
        ] {
            let args = vec![
                "-baseline".into(),
                "-optimize".into(),
                "-quality".into(),
                q.to_string(),
                "-sample".into(),
                samp_c.into(),
            ];
            let c = encode_c(&img.rgb, img.w, img.h, false, &args);
            let r = Encoder::baseline_optimized()
                .quality(q)
                .subsampling(samp_r)
                .trellis(exact_config())
                .force_baseline(true)
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_rgb(&img.rgb, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(
                &format!("{} baseline+opt q{} {}", img.name, q, sname),
                &c,
                &r,
            );
        }
    }
}

#[test]
fn exact_baseline_noopt_color() {
    if noopt_oracle_path().is_none() {
        eprintln!("skipping: trellis_noopt_oracle not built");
        return;
    }
    for img in test_images() {
        for &(q, samp_c, samp_r, sname) in &[
            (75u8, "2x2", Subsampling::S420, "s420"),
            (90, "1x1", Subsampling::S444, "s444"),
        ] {
            // True optimize_coding=FALSE baseline oracle: cjpeg -baseline
            // still runs optimize_coding=TRUE under the JCP_MAX_COMPRESSION
            // defaults, so this goes through the library driver instead.
            let args = vec![
                "-quality".into(),
                q.to_string(),
                "-sample".into(),
                samp_c.into(),
            ];
            let c = encode_c_noopt(&img.rgb, img.w, img.h, false, &args);
            let r = Encoder::baseline_optimized()
                .quality(q)
                .subsampling(samp_r)
                .optimize_huffman(false)
                .trellis(exact_config())
                .force_baseline(true)
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_rgb(&img.rgb, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(
                &format!("{} baseline+!opt q{} {}", img.name, q, sname),
                &c,
                &r,
            );
        }
    }
}

#[test]
fn exact_progressive_color() {
    if cjpeg_path().is_none() {
        eprintln!("skipping: cjpeg-static oracle not built");
        return;
    }
    for img in test_images() {
        for &(q, samp_c, samp_r, sname) in &[
            (75u8, "2x2", Subsampling::S420, "s420"),
            (90, "1x1", Subsampling::S444, "s444"),
        ] {
            let args = vec![
                "-progressive".into(),
                "-quality".into(),
                q.to_string(),
                "-sample".into(),
                samp_c.into(),
            ];
            let c = encode_c(&img.rgb, img.w, img.h, false, &args);
            let r = Encoder::max_compression()
                .quality(q)
                .subsampling(samp_r)
                .trellis(exact_config())
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_rgb(&img.rgb, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(
                &format!("{} progressive q{} {}", img.name, q, sname),
                &c,
                &r,
            );
        }
    }
}

#[test]
fn exact_grayscale() {
    if cjpeg_path().is_none() {
        eprintln!("skipping: cjpeg-static oracle not built");
        return;
    }
    for img in test_images() {
        let gray = to_gray(&img.rgb, img.w as usize, img.h as usize);
        for q in [75u8, 90] {
            // Baseline + optimize
            let args = vec![
                "-baseline".into(),
                "-optimize".into(),
                "-quality".into(),
                q.to_string(),
            ];
            let c = encode_c(&gray, img.w, img.h, true, &args);
            let r = Encoder::baseline_optimized()
                .quality(q)
                .trellis(exact_config())
                .force_baseline(true)
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_gray(&gray, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(&format!("{} gray baseline+opt q{}", img.name, q), &c, &r);

            // Baseline + no optimize (true optimize_coding=FALSE driver)
            if let Some(_p) = noopt_oracle_path() {
                let args = vec!["-quality".into(), q.to_string()];
                let c = encode_c_noopt(&gray, img.w, img.h, true, &args);
                let r = Encoder::baseline_optimized()
                    .quality(q)
                    .optimize_huffman(false)
                    .trellis(exact_config())
                    .force_baseline(true)
                    .pixel_density(PixelDensity::aspect_ratio(1, 1))
                    .encode_gray(&gray, img.w, img.h)
                    .expect("rust encode failed");
                assert_bytes(&format!("{} gray baseline+!opt q{}", img.name, q), &c, &r);
            }

            // Progressive
            let args = vec!["-progressive".into(), "-quality".into(), q.to_string()];
            let c = encode_c(&gray, img.w, img.h, true, &args);
            let r = Encoder::max_compression()
                .quality(q)
                .trellis(exact_config())
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_gray(&gray, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(&format!("{} gray progressive q{}", img.name, q), &c, &r);
        }
    }
}

#[test]
fn exact_restart_interval() {
    if cjpeg_path().is_none() {
        eprintln!("skipping: cjpeg-static oracle not built");
        return;
    }
    for img in test_images() {
        for &(q, rst) in &[(75u8, 8u16), (90, 1)] {
            let args = vec![
                "-baseline".into(),
                "-optimize".into(),
                "-quality".into(),
                q.to_string(),
                "-sample".into(),
                "2x2".into(),
                "-restart".into(),
                format!("{}B", rst),
            ];
            let c = encode_c(&img.rgb, img.w, img.h, false, &args);
            let r = Encoder::baseline_optimized()
                .quality(q)
                .subsampling(Subsampling::S420)
                .restart_interval(rst)
                .trellis(exact_config())
                .force_baseline(true)
                .pixel_density(PixelDensity::aspect_ratio(1, 1))
                .encode_rgb(&img.rgb, img.w, img.h)
                .expect("rust encode failed");
            assert_bytes(&format!("{} restart {} q{}", img.name, rst, q), &c, &r);
        }
    }
}

#[test]
fn exact_trellis_speed_levels() {
    if cjpeg_path().is_none() {
        eprintln!("skipping: cjpeg-static oracle not built");
        return;
    }
    let img = Img {
        name: "noise48",
        rgb: gen_noise(48, 48),
        w: 48,
        h: 48,
    };
    for level in [0u8, 3, 7, 10] {
        let args = vec![
            "-baseline".into(),
            "-optimize".into(),
            "-quality".into(),
            "95".into(),
            // Pin sampling: cjpeg auto-selects factors by quality otherwise.
            "-sample".into(),
            "2x2".into(),
            "-trellis-speed".into(),
            level.to_string(),
        ];
        let c = encode_c(&img.rgb, img.w, img.h, false, &args);
        let r = Encoder::baseline_optimized()
            .quality(95)
            .subsampling(Subsampling::S420)
            .trellis(
                TrellisConfig::default().mode(TrellisMode::MozjpegExact { speed_level: level }),
            )
            .force_baseline(true)
            .pixel_density(PixelDensity::aspect_ratio(1, 1))
            .encode_rgb(&img.rgb, img.w, img.h)
            .expect("rust encode failed");
        assert_bytes(&format!("trellis-speed {}", level), &c, &r);
    }
}
