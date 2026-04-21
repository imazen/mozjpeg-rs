//! Integration tests for the new `Encoder::chroma_quality` knob.
//!
//! Three properties to verify (mirrors zenjpeg PR #109):
//!
//!   1. **Identity** — `chroma_quality(None)` and `chroma_quality(Some(q))`
//!      where `q == quality` both produce bit-identical output to a
//!      config that never touched the setter.
//!   2. **Monotonicity** — on a chroma-rich image, lower
//!      `chroma_quality` at fixed luma `quality` produces a smaller
//!      file than higher `chroma_quality`.
//!   3. **Clamping** — out-of-range inputs are clamped to 1..=100, not
//!      rejected.

use mozjpeg_rs::{Encoder, Preset, Subsampling};

fn sharp_chroma_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(((x * 255) / w.max(1)) as u8);
            out.push(((y * 255) / h.max(1)) as u8);
            out.push((((x + y) * 255) / (w + h).max(1)) as u8);
        }
    }
    out
}

fn encode_rgb(encoder: &Encoder, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    encoder.encode_rgb(rgb, w, h).expect("encode")
}

#[test]
fn chroma_quality_none_equals_unset() {
    let (w, h) = (128u32, 128u32);
    let rgb = sharp_chroma_rgb(w, h);

    for q in [40u8, 75, 90] {
        for sub in [Subsampling::S444, Subsampling::S420, Subsampling::S422] {
            let base = Encoder::new(Preset::default()).quality(q).subsampling(sub);
            let explicit_none = base.clone().chroma_quality(None);
            let same_as_luma = base.clone().chroma_quality(Some(q));

            let a = encode_rgb(&base, &rgb, w, h);
            let b = encode_rgb(&explicit_none, &rgb, w, h);
            let c = encode_rgb(&same_as_luma, &rgb, w, h);

            assert_eq!(
                a, b,
                "q={q}, sub={sub:?}: chroma_quality(None) must be bit-identical to unset"
            );
            assert_eq!(
                a, c,
                "q={q}, sub={sub:?}: chroma_quality(Some(q)) must be bit-identical to unset"
            );
        }
    }
}

#[test]
fn chroma_quality_monotone_file_size() {
    // Gradient RGB has real Cb/Cr content; lower chroma_quality
    // should shrink the file at fixed luma quality.
    let (w, h) = (256u32, 256u32);
    let rgb = sharp_chroma_rgb(w, h);
    let luma_q = 85u8;

    let base = Encoder::new(Preset::default())
        .quality(luma_q)
        .subsampling(Subsampling::S420);

    let high = encode_rgb(&base.clone().chroma_quality(Some(90)), &rgb, w, h);
    let same = encode_rgb(&base.clone().chroma_quality(Some(85)), &rgb, w, h);
    let low = encode_rgb(&base.clone().chroma_quality(Some(50)), &rgb, w, h);

    assert!(
        high.len() >= same.len(),
        "chroma_quality 90 ({}) should be ≥ 85 ({})",
        high.len(),
        same.len()
    );
    assert!(
        same.len() >= low.len(),
        "chroma_quality 85 ({}) should be ≥ 50 ({})",
        same.len(),
        low.len()
    );
    assert!(
        high.len() > low.len(),
        "chroma_quality 90 ({}) should be strictly larger than 50 ({})",
        high.len(),
        low.len()
    );
}

#[test]
fn chroma_quality_clamped_to_1_100() {
    let e = Encoder::new(Preset::default()).chroma_quality(Some(150));
    assert_eq!(e.get_chroma_quality(), Some(100));

    let e = Encoder::new(Preset::default()).chroma_quality(Some(0));
    assert_eq!(e.get_chroma_quality(), Some(1));

    let e = Encoder::new(Preset::default()).chroma_quality(None);
    assert_eq!(e.get_chroma_quality(), None);

    let e = Encoder::new(Preset::default());
    assert_eq!(
        e.get_chroma_quality(),
        None,
        "default encoder should have chroma_quality = None"
    );
}
