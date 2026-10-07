//! Integration tests for the JPEG encoder.
//!
//! These tests verify the public API of the encoder.

use dssim::Dssim;
use mozjpeg_rs::{Encode, Encoder, QuantTableIdx, StreamingEncoder, Subsampling, TrellisConfig};

/// Verify JPEG output can be decoded by an external decoder
#[test]
fn test_decode_with_jpeg_decoder() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let encoder = Encoder::baseline_optimized()
        .quality(90)
        .subsampling(Subsampling::S444);
    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_encode_small_image() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for i in 0..(width * height) as usize {
        rgb_data[i * 3] = 255; // R
        rgb_data[i * 3 + 1] = 0; // G
        rgb_data[i * 3 + 2] = 0; // B
    }

    let encoder = Encoder::baseline_optimized().quality(75);
    let result = encoder.encode_rgb(&rgb_data, width, height);

    assert!(result.is_ok());
    let jpeg_data = result.unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI
    assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
    assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI
}

#[test]
fn test_encode_gradient() {
    let width = 8u32;
    let height = 8u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x + y) * 16) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let encoder = Encoder::baseline_optimized()
        .quality(90)
        .subsampling(Subsampling::S444);
    let result = encoder.encode_rgb(&rgb_data, width, height);

    assert!(result.is_ok());
}

#[test]
fn test_encode_grayscale() {
    let width = 16u32;
    let height = 16u32;
    let mut gray_data = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            gray_data[i] = ((x + y) * 8) as u8;
        }
    }

    let encoder = Encoder::baseline_optimized().quality(85);
    let result = encoder.encode_gray(&gray_data, width, height);

    assert!(result.is_ok());
    let jpeg_data = result.unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI
    assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
    assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode grayscale JPEG");
    let info = decoder.info().unwrap();

    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height) as usize);
}

#[test]
fn test_encode_with_exif() {
    let width = 16u32;
    let height = 16u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    let exif_data = vec![
        0x4D, 0x4D, // Big-endian TIFF
        0x00, 0x2A, // TIFF magic
        0x00, 0x00, 0x00, 0x08, // Offset to IFD
    ];

    let encoder = Encoder::baseline_optimized()
        .quality(75)
        .exif_data(exif_data.clone());
    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut found_app1 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xE1 {
            found_app1 = true;
            if i + 4 < jpeg_data.len() {
                let identifier = &jpeg_data[i + 4..i + 10];
                assert_eq!(identifier, b"Exif\0\0");
            }
            break;
        }
    }
    assert!(found_app1, "APP1 (EXIF) marker not found in output");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode JPEG with EXIF");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

/// Walk the JPEG header (SOI through SOS) and collect the payload of every
/// APPn segment, in order. Returns `(app_number, payload)` pairs; payload
/// excludes the two length bytes.
fn app_segments(jpeg: &[u8]) -> Vec<(u8, &[u8])> {
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "missing SOI");
    let mut out = Vec::new();
    let mut pos = 2;
    while pos + 4 <= jpeg.len() {
        assert_eq!(jpeg[pos], 0xFF, "expected marker at {pos}");
        let marker = jpeg[pos + 1];
        if marker == 0xDA || marker == 0xD9 {
            break; // SOS or EOI
        }
        let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        assert!(
            len >= 2 && pos + 2 + len <= jpeg.len(),
            "bad segment at {pos}"
        );
        if (0xE0..=0xEF).contains(&marker) {
            out.push((marker - 0xE0, &jpeg[pos + 4..pos + 2 + len]));
        }
        pos += 2 + len;
    }
    out
}

const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const XMP_PACKET: &[u8] = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/></x:xmpmeta>"#;

#[test]
fn test_encode_with_xmp() {
    let width = 16u32;
    let height = 16u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    let jpeg_data = Encoder::baseline_optimized()
        .quality(75)
        .xmp_data(XMP_PACKET.to_vec())
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    let xmp_segments: Vec<_> = app_segments(&jpeg_data)
        .into_iter()
        .filter(|(app, data)| *app == 1 && data.starts_with(XMP_NS))
        .collect();
    assert_eq!(xmp_segments.len(), 1, "expected exactly one XMP APP1");
    assert_eq!(&xmp_segments[0].1[XMP_NS.len()..], XMP_PACKET);

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    decoder.decode().expect("Failed to decode JPEG with XMP");
}

#[test]
fn test_encode_xmp_grayscale() {
    let width = 16u32;
    let height = 16u32;
    let gray_data = vec![128u8; (width * height) as usize];

    let jpeg_data = Encoder::baseline_optimized()
        .quality(75)
        .xmp_data(XMP_PACKET.to_vec())
        .encode_gray(&gray_data, width, height)
        .unwrap();

    let has_xmp = app_segments(&jpeg_data)
        .iter()
        .any(|(app, data)| *app == 1 && data.starts_with(XMP_NS));
    assert!(has_xmp, "XMP APP1 not found in grayscale output");
}

#[test]
fn test_xmp_emitted_after_exif() {
    let width = 16u32;
    let height = 16u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];
    let exif = vec![0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];

    let jpeg_data = Encoder::baseline_optimized()
        .quality(75)
        .exif_data(exif)
        .xmp_data(XMP_PACKET.to_vec())
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    let app1s: Vec<&[u8]> = app_segments(&jpeg_data)
        .into_iter()
        .filter(|(app, _)| *app == 1)
        .map(|(_, data)| data)
        .collect();
    assert_eq!(app1s.len(), 2, "expected EXIF and XMP APP1 segments");
    assert!(app1s[0].starts_with(b"Exif\0\0"), "EXIF must come first");
    assert!(app1s[1].starts_with(XMP_NS), "XMP must follow EXIF");
}

#[test]
fn test_xmp_empty_omitted() {
    let rgb_data = vec![128u8; 16 * 16 * 3];
    let jpeg_data = Encoder::baseline_optimized()
        .quality(75)
        .xmp_data(Vec::new())
        .encode_rgb(&rgb_data, 16, 16)
        .unwrap();

    let has_xmp = app_segments(&jpeg_data)
        .iter()
        .any(|(app, data)| *app == 1 && data.starts_with(XMP_NS));
    assert!(!has_xmp, "empty XMP must not emit an APP1 segment");
}

#[test]
fn test_xmp_oversized_rejected() {
    let rgb_data = vec![128u8; 16 * 16 * 3];
    // 2 (length) + 29 (namespace) + data must fit in u16::MAX = 65535,
    // so the largest valid packet is 65504 bytes.
    let oversized = vec![0u8; 65505];
    let result = Encoder::baseline_optimized()
        .xmp_data(oversized)
        .encode_rgb(&rgb_data, 16, 16);
    assert!(result.is_err(), "oversized XMP packet must be rejected");

    // The boundary value itself encodes fine.
    let max_packet = vec![0u8; 65504];
    let result = Encoder::baseline_optimized()
        .xmp_data(max_packet)
        .encode_rgb(&rgb_data, 16, 16);
    assert!(result.is_ok(), "largest valid XMP packet should encode");
}

#[test]
fn test_xmp_counts_against_marker_limit() {
    use mozjpeg_rs::{Error, Limits};
    let rgb_data = vec![128u8; 16 * 16 * 3];
    let result = Encoder::baseline_optimized()
        .xmp_data(XMP_PACKET.to_vec())
        .limits(Limits::none().max_marker_bytes(10))
        .encode_rgb(&rgb_data, 16, 16);
    assert!(
        matches!(result, Err(Error::MarkerDataTooLarge { .. })),
        "XMP must count against max_marker_bytes, got {result:?}"
    );
}

#[test]
fn test_streaming_encoder_xmp() {
    let width = 16u32;
    let height = 16u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    let jpeg_data = StreamingEncoder::baseline_fastest()
        .quality(85)
        .xmp_data(XMP_PACKET.to_vec())
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    let xmp_segments: Vec<_> = app_segments(&jpeg_data)
        .into_iter()
        .filter(|(app, data)| *app == 1 && data.starts_with(XMP_NS))
        .collect();
    assert_eq!(xmp_segments.len(), 1);
    assert_eq!(&xmp_segments[0].1[XMP_NS.len()..], XMP_PACKET);
}

#[test]
fn test_encode_with_restart_markers() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 4) as u8;
            rgb_data[i * 3 + 1] = (y * 4) as u8;
            rgb_data[i * 3 + 2] = 128;
        }
    }

    let encoder = Encoder::baseline_optimized()
        .quality(75)
        .subsampling(Subsampling::S444)
        .optimize_huffman(false)
        .trellis(TrellisConfig::disabled())
        .restart_interval(4);

    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut found_dri = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDD {
            found_dri = true;
            if i + 5 < jpeg_data.len() {
                let len = ((jpeg_data[i + 2] as u16) << 8) | (jpeg_data[i + 3] as u16);
                assert_eq!(len, 4, "DRI marker length should be 4");
                let interval = ((jpeg_data[i + 4] as u16) << 8) | (jpeg_data[i + 5] as u16);
                assert_eq!(interval, 4, "Restart interval should be 4");
            }
            break;
        }
    }
    assert!(found_dri, "DRI marker not found in output");

    let mut rst_count = 0;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] >= 0xD0 && jpeg_data[i + 1] <= 0xD7 {
            rst_count += 1;
        }
    }
    assert_eq!(
        rst_count, 15,
        "Expected 15 RST markers, found {}",
        rst_count
    );

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder
        .decode()
        .expect("Failed to decode JPEG with restart markers");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_encode_invalid_size() {
    let rgb_data = vec![0u8; 100];
    let encoder = Encoder::baseline_optimized();
    let result = encoder.encode_rgb(&rgb_data, 16, 16);

    assert!(result.is_err());
}

#[test]
fn test_encode_zero_dimensions() {
    use mozjpeg_rs::Error;

    let encoder = Encoder::baseline_optimized();

    let result = encoder.encode_rgb(&[], 0, 16);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 0,
            height: 16
        })
    ));

    let result = encoder.encode_rgb(&[], 16, 0);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 16,
            height: 0
        })
    ));

    let result = encoder.encode_rgb(&[], 0, 0);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 0,
            height: 0
        })
    ));
}

#[test]
fn test_encode_overflow_dimensions() {
    let encoder = Encoder::baseline_optimized();
    let result = encoder.encode_rgb(&[], u32::MAX, u32::MAX);
    assert!(result.is_err());
}

#[test]
fn test_progressive_encode_decode() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let encoder = Encoder::baseline_optimized()
        .quality(85)
        .progressive(true)
        .subsampling(Subsampling::S420);

    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI

    let mut has_sof2 = false;
    let mut i = 2;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC2 {
            has_sof2 = true;
            break;
        }
        i += 1;
    }
    assert!(has_sof2, "Progressive JPEG should have SOF2 marker");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode progressive JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_progressive_vs_baseline_size() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as f32 * 0.1).sin() * 127.0 + 128.0) as u8)
                .wrapping_add((((y as f32) * 0.1).cos() * 50.0) as u8);
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val.wrapping_add(30);
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let baseline = Encoder::baseline_optimized()
        .quality(75)
        .progressive(false)
        .subsampling(Subsampling::S420);
    let baseline_data = baseline.encode_rgb(&rgb_data, width, height).unwrap();

    let progressive = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .subsampling(Subsampling::S420);
    let progressive_data = progressive.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!baseline_data.is_empty());
    assert!(!progressive_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline_data));
    decoder.decode().expect("Failed to decode baseline JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive_data));
    decoder.decode().expect("Failed to decode progressive JPEG");
}

#[test]
fn test_trellis_quantization_enabled() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as i32 - y as i32).abs() * 10) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = val / 2;
        }
    }

    let no_trellis = Encoder::baseline_optimized()
        .quality(75)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::disabled());
    let no_trellis_data = no_trellis.encode_rgb(&rgb_data, width, height).unwrap();

    let with_trellis = Encoder::baseline_optimized()
        .quality(75)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::default());
    let with_trellis_data = with_trellis.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!no_trellis_data.is_empty());
    assert!(!with_trellis_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_trellis_data));
    decoder.decode().expect("Failed to decode non-trellis JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_trellis_data));
    decoder.decode().expect("Failed to decode trellis JPEG");
}

#[test]
fn test_trellis_presets() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as i32 - y as i32).abs() * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = (val / 2).wrapping_add(64);
        }
    }

    let quality = 97;

    let default = Encoder::baseline_optimized()
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::default());
    let default_data = default.encode_rgb(&rgb_data, width, height).unwrap();

    let favor_size = Encoder::baseline_optimized()
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::favor_size());
    let favor_size_data = favor_size.encode_rgb(&rgb_data, width, height).unwrap();

    let favor_quality = Encoder::baseline_optimized()
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::favor_quality());
    let favor_quality_data = favor_quality.encode_rgb(&rgb_data, width, height).unwrap();

    for (name, data) in [
        ("default", &default_data),
        ("favor_size", &favor_size_data),
        ("favor_quality", &favor_quality_data),
    ] {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
        decoder
            .decode()
            .unwrap_or_else(|_| panic!("Failed to decode {} JPEG", name));
    }

    assert!(
        favor_size_data.len() != favor_quality_data.len(),
        "Presets should produce different sizes"
    );
}

#[test]
fn test_trellis_rd_factor() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 8 + y * 4) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let factor_1 = Encoder::baseline_optimized()
        .quality(85)
        .trellis(TrellisConfig::default().rd_factor(1.0));
    let factor_1_data = factor_1.encode_rgb(&rgb_data, width, height).unwrap();

    let factor_2 = Encoder::baseline_optimized()
        .quality(85)
        .trellis(TrellisConfig::default().rd_factor(2.0));
    let factor_2_data = factor_2.encode_rgb(&rgb_data, width, height).unwrap();

    jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_1_data))
        .decode()
        .expect("Failed to decode rd_factor(1.0) JPEG");
    jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_2_data))
        .decode()
        .expect("Failed to decode rd_factor(2.0) JPEG");
}

#[test]
fn test_huffman_optimization() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 8 + y * 4) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let no_opt = Encoder::baseline_optimized()
        .quality(75)
        .subsampling(Subsampling::S420)
        .optimize_huffman(false);
    let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let with_opt = Encoder::baseline_optimized()
        .quality(75)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);
    let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!no_opt_data.is_empty());
    assert!(!with_opt_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
    decoder
        .decode()
        .expect("Failed to decode non-optimized JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
    decoder.decode().expect("Failed to decode optimized JPEG");
}

#[test]
fn test_color_encoding_accuracy() {
    let test_cases = [
        ("black", 0u8, 0u8, 0u8),
        ("red", 255, 0, 0),
        ("green", 0, 255, 0),
        ("blue", 0, 0, 255),
        ("white", 255, 255, 255),
        ("gray", 128, 128, 128),
    ];

    let width = 16u32;
    let height = 16u32;

    for (name, r, g, b) in &test_cases {
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];
        for i in 0..(width * height) as usize {
            rgb_data[i * 3] = *r;
            rgb_data[i * 3 + 1] = *g;
            rgb_data[i * 3 + 2] = *b;
        }

        let encoder = Encoder::baseline_optimized()
            .quality(95)
            .subsampling(Subsampling::S444);
        let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        let decoded = decoder.decode().expect("decode failed");

        let dr = decoded[0];
        let dg = decoded[1];
        let db = decoded[2];

        let tolerance = 2i16;
        let r_diff = (dr as i16 - *r as i16).abs();
        let g_diff = (dg as i16 - *g as i16).abs();
        let b_diff = (db as i16 - *b as i16).abs();

        assert!(
            r_diff <= tolerance,
            "{}: R mismatch - expected {}, got {} (diff {})",
            name,
            r,
            dr,
            r_diff
        );
        assert!(
            g_diff <= tolerance,
            "{}: G mismatch - expected {}, got {} (diff {})",
            name,
            g,
            dg,
            g_diff
        );
        assert!(
            b_diff <= tolerance,
            "{}: B mismatch - expected {}, got {} (diff {})",
            name,
            b,
            db,
            b_diff
        );
    }
}

#[test]
fn test_optimize_scans() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 4 + y * 3) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = ((val as u16 + 128) % 256) as u8;
        }
    }

    let no_opt = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .optimize_scans(false)
        .subsampling(Subsampling::S420);
    let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let with_opt = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .optimize_scans(true)
        .subsampling(Subsampling::S420);
    let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
    decoder
        .decode()
        .expect("Failed to decode non-optimized JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
    decoder
        .decode()
        .expect("Failed to decode scan-optimized JPEG");

    assert!(!with_opt_data.is_empty());
}

/// Regression test: Progressive encoding with non-MCU-aligned dimensions.
#[test]
fn test_progressive_non_mcu_aligned_regression() {
    let failing_sizes = [17, 24, 33, 40, 49];

    for &size in &failing_sizes {
        let s = size as usize;
        let mut rgb = vec![0u8; s * s * 3];
        for y in 0..s {
            for x in 0..s {
                let idx = (y * s + x) * 3;
                rgb[idx] = (x * 15).min(255) as u8;
                rgb[idx + 1] = (y * 15).min(255) as u8;
                rgb[idx + 2] = 128;
            }
        }

        let baseline = Encoder::baseline_optimized()
            .quality(95)
            .subsampling(Subsampling::S420)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let progressive = Encoder::baseline_optimized()
            .quality(95)
            .subsampling(Subsampling::S420)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
            .decode()
            .expect("baseline decode failed");
        let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
            .decode()
            .expect("progressive decode failed");

        let base_psnr = calculate_psnr(&rgb, &base_dec);
        let prog_psnr = calculate_psnr(&rgb, &prog_dec);

        let diff = (prog_psnr - base_psnr).abs();
        assert!(
            diff < 3.0,
            "{}x{}: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
            size,
            size,
            prog_psnr,
            base_psnr,
            diff
        );

        let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
        let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
        assert!(
            base_dssim < 0.01,
            "{}x{}: Baseline DSSIM too high: {:.6}",
            size,
            size,
            base_dssim
        );
        assert!(
            prog_dssim < 0.01,
            "{}x{}: Progressive DSSIM too high: {:.6}",
            size,
            size,
            prog_dssim
        );
    }
}

#[test]
fn test_progressive_422_non_mcu_aligned_regression() {
    let size = 17u32;
    let s = size as usize;
    let mut rgb = vec![0u8; s * s * 3];
    for y in 0..s {
        for x in 0..s {
            let idx = (y * s + x) * 3;
            rgb[idx] = (x * 15).min(255) as u8;
            rgb[idx + 1] = (y * 15).min(255) as u8;
            rgb[idx + 2] = 128;
        }
    }

    let baseline = Encoder::baseline_optimized()
        .quality(95)
        .subsampling(Subsampling::S422)
        .progressive(false)
        .encode_rgb(&rgb, size, size)
        .unwrap();

    let progressive = Encoder::baseline_optimized()
        .quality(95)
        .subsampling(Subsampling::S422)
        .progressive(true)
        .encode_rgb(&rgb, size, size)
        .unwrap();

    let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
        .decode()
        .unwrap();
    let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
        .decode()
        .unwrap();

    let base_psnr = calculate_psnr(&rgb, &base_dec);
    let prog_psnr = calculate_psnr(&rgb, &prog_dec);
    let diff = (prog_psnr - base_psnr).abs();

    assert!(
        diff < 3.0,
        "4:2:2 17x17: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
        prog_psnr,
        base_psnr,
        diff
    );

    let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
    let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
    assert!(
        base_dssim < 0.01,
        "4:2:2 17x17: Baseline DSSIM too high: {:.6}",
        base_dssim
    );
    assert!(
        prog_dssim < 0.01,
        "4:2:2 17x17: Progressive DSSIM too high: {:.6}",
        prog_dssim
    );
}

#[test]
fn test_streaming_encoder_rgb() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let streaming = StreamingEncoder::baseline_fastest().quality(85);
    let streaming_data = streaming.encode_rgb(&rgb_data, width, height).unwrap();

    assert_eq!(streaming_data[0], 0xFF);
    assert_eq!(streaming_data[1], 0xD8); // SOI
    assert_eq!(streaming_data[streaming_data.len() - 2], 0xFF);
    assert_eq!(streaming_data[streaming_data.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
    let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_streaming_encoder_scanlines() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let mut output = Vec::new();
    let mut stream = StreamingEncoder::baseline_fastest()
        .quality(85)
        .subsampling(Subsampling::S420)
        .start_rgb(width, height, &mut output)
        .unwrap();

    let bytes_per_line = (width * 3) as usize;
    for chunk in rgb_data.chunks(bytes_per_line * 8) {
        stream.write_scanlines(chunk).unwrap();
    }

    stream.finish().unwrap();

    assert_eq!(output[0], 0xFF);
    assert_eq!(output[1], 0xD8); // SOI
    assert_eq!(output[output.len() - 2], 0xFF);
    assert_eq!(output[output.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&output));
    let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_streaming_encoder_gray() {
    let width = 16u32;
    let height = 16u32;
    let mut gray_data = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            gray_data[i] = ((x * 16 + y * 16) % 256) as u8;
        }
    }

    let streaming = StreamingEncoder::baseline_fastest().quality(85);
    let streaming_data = streaming.encode_gray(&gray_data, width, height).unwrap();

    assert_eq!(streaming_data[0], 0xFF);
    assert_eq!(streaming_data[1], 0xD8); // SOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
    let decoded = decoder
        .decode()
        .expect("Failed to decode grayscale streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height) as usize);
}

// Helper functions

fn calculate_psnr(orig: &[u8], decoded: &[u8]) -> f64 {
    let mse: f64 = orig
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / orig.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    use rgb::RGB8;

    let attr = Dssim::new();

    let orig_rgb: Vec<RGB8> = original
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = attr
        .create_image_rgb(&orig_rgb, width as usize, height as usize)
        .expect("Failed to create original image");

    let dec_rgb: Vec<RGB8> = decoded
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = attr
        .create_image_rgb(&dec_rgb, width as usize, height as usize)
        .expect("Failed to create decoded image");

    let (dssim_val, _) = attr.compare(&orig_img, dec_img);
    dssim_val.into()
}

#[test]
fn test_eob_optimization_produces_valid_jpeg() {
    use mozjpeg_rs::TrellisConfig;

    let width = 64u32;
    let height = 64u32;

    // Create test image with some areas that will produce zero blocks
    let mut rgb = vec![128u8; (width * height * 3) as usize];
    // Add some variation in one quadrant
    for y in 0..32 {
        for x in 0..32 {
            let idx = ((y * width + x) * 3) as usize;
            rgb[idx] = ((x * 8) % 256) as u8;
            rgb[idx + 1] = ((y * 8) % 256) as u8;
            rgb[idx + 2] = (((x + y) * 4) % 256) as u8;
        }
    }

    // Encode with EOB optimization enabled
    let trellis_with_eob = TrellisConfig::default().eob_optimization(true);
    let with_eob = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .trellis(trellis_with_eob)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding with EOB opt failed");

    // Encode with EOB optimization disabled
    let trellis_without_eob = TrellisConfig::default().eob_optimization(false);
    let without_eob = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .trellis(trellis_without_eob)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding without EOB opt failed");

    // Both should produce valid JPEGs
    assert!(with_eob.len() > 100, "EOB-optimized JPEG too small");
    assert!(without_eob.len() > 100, "Non-EOB JPEG too small");

    // Both should decode successfully
    let mut decoder1 = jpeg_decoder::Decoder::new(&with_eob[..]);
    let decoded1 = decoder1
        .decode()
        .expect("Failed to decode EOB-optimized JPEG");
    let info1 = decoder1.info().unwrap();
    assert_eq!(info1.width, width as u16);
    assert_eq!(info1.height, height as u16);

    let mut decoder2 = jpeg_decoder::Decoder::new(&without_eob[..]);
    let decoded2 = decoder2.decode().expect("Failed to decode non-EOB JPEG");
    let info2 = decoder2.info().unwrap();
    assert_eq!(info2.width, width as u16);
    assert_eq!(info2.height, height as u16);

    // EOB optimization may zero some coefficients for encoding efficiency,
    // so we allow reasonable quality differences. The important thing is
    // both produce valid, decodable JPEGs.
    let max_diff: i32 = decoded1
        .iter()
        .zip(decoded2.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);

    // Allow up to ~27% difference (70/255) since EOB optimization trades quality for size
    assert!(
        max_diff <= 100,
        "EOB optimization changed output too much: max_diff={}",
        max_diff
    );

    println!(
        "EOB optimization: with={} bytes, without={} bytes, diff={}",
        with_eob.len(),
        without_eob.len(),
        without_eob.len() as i64 - with_eob.len() as i64
    );
}

#[test]
fn test_eob_optimization_grayscale() {
    use mozjpeg_rs::TrellisConfig;

    let width = 64u32;
    let height = 64u32;

    // Create grayscale test image with some flat areas
    let mut gray = vec![128u8; (width * height) as usize];
    // Add gradient in one area
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * width + x) as usize;
            gray[idx] = ((x * 4 + y * 4) % 256) as u8;
        }
    }

    // Encode with EOB optimization
    let trellis_with_eob = TrellisConfig::default().eob_optimization(true);
    let with_eob = Encoder::baseline_optimized()
        .quality(75)
        .progressive(true)
        .trellis(trellis_with_eob)
        .encode_gray(&gray, width, height)
        .expect("Grayscale encoding with EOB opt failed");

    // Should produce valid JPEG
    assert!(
        with_eob.len() > 50,
        "EOB-optimized grayscale JPEG too small"
    );

    // Should decode successfully
    let mut decoder = jpeg_decoder::Decoder::new(&with_eob[..]);
    let decoded = decoder
        .decode()
        .expect("Failed to decode EOB-optimized grayscale JPEG");
    assert_eq!(decoded.len(), (width * height) as usize);

    println!("Grayscale EOB optimization: {} bytes", with_eob.len());
}

/// Comprehensive test of all encoder setting permutations with encode+decode round-trip.
/// Uses jpeg_decoder for decoding.
#[test]
fn test_encode_decode_permutations() {
    use mozjpeg_rs::{Subsampling, TrellisConfig};

    let width = 32u32;
    let height = 32u32;

    // Create test image
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();

    // Test matrix of settings
    let progressives = [false, true];
    let subsamplings = [Subsampling::S444, Subsampling::S422, Subsampling::S420];
    let optimize_huffmans = [false, true];
    let trellis_enabled = [false, true];
    let eob_opts = [false, true];
    let qualities = [50u8, 85];

    let mut test_count = 0;

    for &progressive in &progressives {
        for &subsampling in &subsamplings {
            for &optimize_huffman in &optimize_huffmans {
                for &trellis in &trellis_enabled {
                    for &eob_opt in &eob_opts {
                        // Skip eob_opt=true when trellis=false (eob_opt requires trellis)
                        if eob_opt && !trellis {
                            continue;
                        }

                        for &quality in &qualities {
                            let trellis_config = if trellis {
                                TrellisConfig::default().eob_optimization(eob_opt)
                            } else {
                                TrellisConfig::disabled()
                            };

                            let encoder = Encoder::baseline_optimized()
                                .quality(quality)
                                .progressive(progressive)
                                .subsampling(subsampling)
                                .optimize_huffman(optimize_huffman)
                                .trellis(trellis_config);

                            let result = encoder.encode_rgb(&rgb, width, height);

                            let jpeg = match result {
                                Ok(data) => data,
                                Err(e) => {
                                    panic!(
                                        "Encoding failed: prog={}, sub={:?}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                        progressive,
                                        subsampling,
                                        optimize_huffman,
                                        trellis,
                                        eob_opt,
                                        quality,
                                        e
                                    );
                                }
                            };

                            // Verify JPEG markers
                            assert_eq!(jpeg[0], 0xFF, "Missing SOI");
                            assert_eq!(jpeg[1], 0xD8, "Missing SOI");

                            // Decode the JPEG
                            let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
                            let decoded = decoder.decode().unwrap_or_else(|e| {
                                panic!(
                                    "Decode failed: prog={}, sub={:?}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                    progressive, subsampling, optimize_huffman, trellis, eob_opt, quality, e
                                );
                            });

                            let info = decoder.info().unwrap();
                            assert_eq!(info.width, width as u16);
                            assert_eq!(info.height, height as u16);
                            assert_eq!(
                                decoded.len(),
                                (width * height * 3) as usize,
                                "Decoded size mismatch"
                            );

                            test_count += 1;
                        }
                    }
                }
            }
        }
    }

    println!(
        "Successfully tested {} encode+decode permutations",
        test_count
    );
}

/// Test grayscale encoding with all setting permutations.
#[test]
fn test_grayscale_encode_decode_permutations() {
    use mozjpeg_rs::TrellisConfig;

    let width = 32u32;
    let height = 32u32;

    // Create grayscale test image
    let gray: Vec<u8> = (0..width * height)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();

    let progressives = [false, true];
    let optimize_huffmans = [false, true];
    let trellis_enabled = [false, true];
    let eob_opts = [false, true];
    let qualities = [50u8, 85];

    let mut test_count = 0;

    for &progressive in &progressives {
        for &optimize_huffman in &optimize_huffmans {
            for &trellis in &trellis_enabled {
                for &eob_opt in &eob_opts {
                    // Skip eob_opt=true when trellis=false
                    if eob_opt && !trellis {
                        continue;
                    }

                    for &quality in &qualities {
                        let trellis_config = if trellis {
                            TrellisConfig::default().eob_optimization(eob_opt)
                        } else {
                            TrellisConfig::disabled()
                        };

                        let encoder = Encoder::baseline_optimized()
                            .quality(quality)
                            .progressive(progressive)
                            .optimize_huffman(optimize_huffman)
                            .trellis(trellis_config);

                        let result = encoder.encode_gray(&gray, width, height);

                        let jpeg = match result {
                            Ok(data) => data,
                            Err(e) => {
                                panic!(
                                    "Grayscale encoding failed: prog={}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                    progressive, optimize_huffman, trellis, eob_opt, quality, e
                                );
                            }
                        };

                        // Decode
                        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
                        let decoded = decoder.decode().unwrap_or_else(|e| {
                            panic!(
                                "Grayscale decode failed: prog={}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                progressive, optimize_huffman, trellis, eob_opt, quality, e
                            );
                        });

                        let info = decoder.info().unwrap();
                        assert_eq!(info.width, width as u16);
                        assert_eq!(info.height, height as u16);
                        assert_eq!(decoded.len(), (width * height) as usize);

                        test_count += 1;
                    }
                }
            }
        }
    }

    println!(
        "Successfully tested {} grayscale encode+decode permutations",
        test_count
    );
}

/// Test smoothing filter for dithered image simulation.
#[test]
fn test_smoothing_filter() {
    // Create a "dithered" image with alternating pixels
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let val = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb_data[i] = val;
            rgb_data[i + 1] = val;
            rgb_data[i + 2] = val;
        }
    }

    // Encode without smoothing
    let no_smooth = Encoder::baseline_optimized()
        .quality(75)
        .smoothing(0)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // Encode with smoothing
    let with_smooth = Encoder::baseline_optimized()
        .quality(75)
        .smoothing(50)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // Both should produce valid JPEGs
    let mut decoder = jpeg_decoder::Decoder::new(&no_smooth[..]);
    decoder.decode().expect("No-smoothing JPEG should decode");

    let mut decoder = jpeg_decoder::Decoder::new(&with_smooth[..]);
    decoder.decode().expect("Smoothed JPEG should decode");

    // Smoothing on dithered images typically produces smaller files
    // (less high-frequency content after smoothing)
    println!(
        "Dithered image: no_smooth={} bytes, with_smooth={} bytes",
        no_smooth.len(),
        with_smooth.len()
    );
}

/// Test alternate quantization tables.
#[test]
fn test_alternate_quant_tables() {
    let width = 64u32;
    let height = 64u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    let tables = [
        QuantTableIdx::JpegAnnexK,
        QuantTableIdx::Flat,
        QuantTableIdx::MssimTuned,
        QuantTableIdx::ImageMagick, // default
        QuantTableIdx::PsnrHvsM,
        QuantTableIdx::Klein,
        QuantTableIdx::Watson,
        QuantTableIdx::Ahumada,
        QuantTableIdx::Peterson,
    ];

    for table in &tables {
        let jpeg = Encoder::baseline_optimized()
            .quality(75)
            .quant_tables(*table)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder
            .decode()
            .unwrap_or_else(|e| panic!("Failed to decode with {:?} table: {:?}", table, e));

        println!("{:?}: {} bytes", table, jpeg.len());
    }
}

/// Test custom quantization tables.
#[test]
fn test_custom_quant_tables() {
    let width = 64u32;
    let height = 64u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    // Custom flat table with value 16
    let custom_table = [16u16; 64];

    let jpeg = Encoder::baseline_optimized()
        .quality(75)
        .custom_luma_qtable(custom_table)
        .custom_chroma_qtable(custom_table)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    decoder
        .decode()
        .expect("Custom quant table JPEG should decode");

    println!("Custom quant table: {} bytes", jpeg.len());
}

/// Regression test for mozilla/mozjpeg#444: overshoot deringing + SIMD DCT overflow.
///
/// An 8x8 block split half-black/half-white triggers maximum overshoot deringing,
/// which pushes level-shifted sample values to +-158. In SIMD forward DCT
/// implementations using 16-bit arithmetic, the column pass final butterfly
/// (tmp10+tmp11) reaches 8*5056 = 40448, overflowing signed 16-bit and causing
/// catastrophic sign flips that invert the entire block's brightness.
///
/// The Rust encoder uses 32-bit DCT intermediates and is immune, but this test
/// documents the pattern and ensures correctness is maintained.
#[test]
fn test_issue444_deringing_overflow_pattern() {
    let width = 8u32;
    let height = 8u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    // Half-black, half-white vertical split — the worst case for deringing overflow
    for y in 0..height {
        for x in 4..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = 255;
            rgb_data[i * 3 + 1] = 255;
            rgb_data[i * 3 + 2] = 255;
        }
    }

    // Test at Q25 (where the C SIMD bug is most visible) with deringing enabled
    let encoder = Encoder::baseline_optimized()
        .quality(25)
        .subsampling(Subsampling::S444);
    let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    let decoded = decoder.decode().expect("Failed to decode");

    // Left half (columns 0-3) must be dark, right half (columns 4-7) must be bright.
    // The SIMD overflow bug inverts this completely (left=~197, right=~70).
    let mut left_sum: u64 = 0;
    let mut right_sum: u64 = 0;
    for y in 0..8 {
        for x in 0..4 {
            let i = (y * 8 + x) * 3;
            left_sum += decoded[i] as u64;
        }
        for x in 4..8 {
            let i = (y * 8 + x) * 3;
            right_sum += decoded[i] as u64;
        }
    }
    let left_mean = left_sum as f64 / 32.0;
    let right_mean = right_sum as f64 / 32.0;

    assert!(
        left_mean < 80.0,
        "Left half should be dark (got mean {:.1}). Sign flip bug?",
        left_mean
    );
    assert!(
        right_mean > 180.0,
        "Right half should be bright (got mean {:.1}). Sign flip bug?",
        right_mean
    );
}

/// Extended regression test for issue #444 across quality range Q2-Q57.
///
/// The overflow occurs when the DC quantization value is large enough that
/// 2*quantval[0] >= 28 (the overshoot threshold for overflow). This is true
/// for Q1-Q57 with the default ImageMagick quant tables.
#[test]
fn test_issue444_across_quality_range() {
    let width = 8u32;
    let height = 8u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 4..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = 255;
            rgb_data[i * 3 + 1] = 255;
            rgb_data[i * 3 + 2] = 255;
        }
    }

    for q in [2, 10, 25, 50, 57, 75, 90] {
        let encoder = Encoder::baseline_optimized()
            .quality(q)
            .subsampling(Subsampling::S444);
        let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        let decoded = decoder.decode().unwrap_or_else(|e| {
            panic!("Failed to decode Q{}: {:?}", q, e);
        });

        let mut left_sum: f64 = 0.0;
        let mut right_sum: f64 = 0.0;
        for y in 0..8usize {
            for x in 0..4usize {
                left_sum += decoded[(y * 8 + x) * 3] as f64;
            }
            for x in 4..8usize {
                right_sum += decoded[(y * 8 + x) * 3] as f64;
            }
        }
        let left_mean = left_sum / 32.0;
        let right_mean = right_sum / 32.0;

        assert!(
            left_mean < right_mean,
            "Q{}: left half ({:.1}) should be darker than right half ({:.1})",
            q,
            left_mean,
            right_mean
        );
    }
}

// =============================================================================
// Strided encoding tests
// =============================================================================

/// Test that encode_rgb_strided produces identical output to encode_rgb
/// when stride equals row_bytes (tight packing).
#[test]
fn test_strided_rgb_tight_packing() {
    let width = 64u32;
    let height = 48u32;
    let row_bytes = (width * 3) as usize;

    // Create test image
    let mut rgb_data = vec![0u8; row_bytes * height as usize];
    for (i, byte) in rgb_data.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let encoder = Encoder::baseline_optimized().quality(85);

    let jpeg_normal = encoder.encode_rgb(&rgb_data, width, height).unwrap();
    let jpeg_strided = encoder
        .encode_rgb_strided(&rgb_data, width, height, row_bytes)
        .unwrap();

    // Should be byte-identical
    assert_eq!(jpeg_normal, jpeg_strided);
}

/// Test that encode_rgb_strided correctly handles padded rows.
#[test]
fn test_strided_rgb_with_padding() {
    let width = 64u32;
    let height = 48u32;
    let row_bytes = (width * 3) as usize; // 192 bytes
    let stride = 256; // Padded to 256 bytes per row (64-byte aligned)

    // Create tight buffer
    let mut tight_rgb = vec![0u8; row_bytes * height as usize];
    for (i, byte) in tight_rgb.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    // Create padded buffer
    let mut padded_rgb = vec![0xFFu8; stride * height as usize]; // Fill padding with 0xFF
    for y in 0..height as usize {
        let src_start = y * row_bytes;
        let dst_start = y * stride;
        padded_rgb[dst_start..dst_start + row_bytes]
            .copy_from_slice(&tight_rgb[src_start..src_start + row_bytes]);
    }

    let encoder = Encoder::baseline_optimized().quality(85);

    let jpeg_tight = encoder.encode_rgb(&tight_rgb, width, height).unwrap();
    let jpeg_strided = encoder
        .encode_rgb_strided(&padded_rgb, width, height, stride)
        .unwrap();

    // Should produce identical output (padding is ignored)
    assert_eq!(jpeg_tight, jpeg_strided);
}

/// Test that encode_gray_strided produces identical output to encode_gray
/// when stride equals width (tight packing).
#[test]
fn test_strided_gray_tight_packing() {
    let width = 64u32;
    let height = 48u32;

    let mut gray_data = vec![0u8; (width * height) as usize];
    for (i, byte) in gray_data.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let encoder = Encoder::baseline_optimized().quality(85);

    let jpeg_normal = encoder.encode_gray(&gray_data, width, height).unwrap();
    let jpeg_strided = encoder
        .encode_gray_strided(&gray_data, width, height, width as usize)
        .unwrap();

    assert_eq!(jpeg_normal, jpeg_strided);
}

/// Test that encode_gray_strided correctly handles padded rows.
#[test]
fn test_strided_gray_with_padding() {
    let width = 64u32;
    let height = 48u32;
    let stride = 128; // Padded to 128 bytes per row

    // Create tight buffer
    let mut tight_gray = vec![0u8; (width * height) as usize];
    for (i, byte) in tight_gray.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    // Create padded buffer
    let mut padded_gray = vec![0xFFu8; stride * height as usize];
    for y in 0..height as usize {
        let src_start = y * width as usize;
        let dst_start = y * stride;
        padded_gray[dst_start..dst_start + width as usize]
            .copy_from_slice(&tight_gray[src_start..src_start + width as usize]);
    }

    let encoder = Encoder::baseline_optimized().quality(85);

    let jpeg_tight = encoder.encode_gray(&tight_gray, width, height).unwrap();
    let jpeg_strided = encoder
        .encode_gray_strided(&padded_gray, width, height, stride)
        .unwrap();

    assert_eq!(jpeg_tight, jpeg_strided);
}

/// Test that encode_rgb_strided rejects stride smaller than row bytes.
#[test]
fn test_strided_rgb_invalid_stride() {
    let width = 64u32;
    let height = 48u32;
    let row_bytes = (width * 3) as usize;
    let rgb_data = vec![0u8; row_bytes * height as usize];

    let encoder = Encoder::baseline_optimized();
    let result = encoder.encode_rgb_strided(&rgb_data, width, height, row_bytes - 1);

    assert!(matches!(
        result,
        Err(mozjpeg_rs::Error::InvalidStride { stride, minimum })
            if stride == row_bytes - 1 && minimum == row_bytes
    ));
}

/// Test that encode_gray_strided rejects stride smaller than width.
#[test]
fn test_strided_gray_invalid_stride() {
    let width = 64u32;
    let height = 48u32;
    let gray_data = vec![0u8; (width * height) as usize];

    let encoder = Encoder::baseline_optimized();
    let result = encoder.encode_gray_strided(&gray_data, width, height, (width - 1) as usize);

    assert!(matches!(
        result,
        Err(mozjpeg_rs::Error::InvalidStride { stride, minimum })
            if stride == (width - 1) as usize && minimum == width as usize
    ));
}

/// Test strided encoding with a subregion (simulates cropping).
#[test]
fn test_strided_crop_simulation() {
    // Simulate encoding a 32x32 crop from a 128x96 image starting at (16, 8)
    let full_width = 128usize;
    let full_height = 96usize;
    let crop_x = 16usize;
    let crop_y = 8usize;
    let crop_width = 32u32;
    let crop_height = 32u32;

    // Create full image
    let mut full_rgb = vec![0u8; full_width * full_height * 3];
    for y in 0..full_height {
        for x in 0..full_width {
            let i = (y * full_width + x) * 3;
            full_rgb[i] = (x % 256) as u8;
            full_rgb[i + 1] = (y % 256) as u8;
            full_rgb[i + 2] = ((x + y) % 256) as u8;
        }
    }

    // Point to crop region (stride = full width * 3)
    let crop_offset = (crop_y * full_width + crop_x) * 3;
    let crop_data = &full_rgb[crop_offset..];
    let stride = full_width * 3;

    let encoder = Encoder::baseline_optimized().quality(85);
    let jpeg = encoder
        .encode_rgb_strided(crop_data, crop_width, crop_height, stride)
        .unwrap();

    // Decode and verify we got the crop region
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    let decoded = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    assert_eq!(info.width, crop_width as u16);
    assert_eq!(info.height, crop_height as u16);

    // Verify a few pixels match the expected crop
    // Top-left of crop (16, 8) in full image
    let expected_r = (crop_x % 256) as u8;
    let expected_g = (crop_y % 256) as u8;
    assert!(
        (decoded[0] as i16 - expected_r as i16).abs() < 5,
        "Top-left R mismatch: {} vs {}",
        decoded[0],
        expected_r
    );
    assert!(
        (decoded[1] as i16 - expected_g as i16).abs() < 5,
        "Top-left G mismatch: {} vs {}",
        decoded[1],
        expected_g
    );
}

// ============================================================================
// RGBA encode tests
// ============================================================================

/// Verify encode_rgba produces valid, decodable JPEG output.
#[test]
fn test_encode_rgba() {
    let w = 64u32;
    let h = 64u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        rgba[i * 4] = 220; // R
        rgba[i * 4 + 1] = 128; // G
        rgba[i * 4 + 2] = 30; // B
        rgba[i * 4 + 3] = 100; // A (ignored)
    }

    let encoder = Encoder::default().quality(85);
    let jpeg = encoder.encode_rgba(&rgba, w, h).unwrap();

    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);

    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    let pixels = decoder.decode().unwrap();

    let r_diff = (pixels[0] as i16 - 220).abs();
    let g_diff = (pixels[1] as i16 - 128).abs();
    let b_diff = (pixels[2] as i16 - 30).abs();
    assert!(r_diff < 5, "R diff too large: {r_diff}");
    assert!(g_diff < 5, "G diff too large: {g_diff}");
    assert!(b_diff < 5, "B diff too large: {b_diff}");
}

/// RGBA and RGB encode the same pixels — output must be byte-identical.
#[test]
fn test_rgba_rgb_parity() {
    let w = 48u32;
    let h = 48u32;
    let (r, g, b) = (180u8, 90, 45);

    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
        rgba[i * 4] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = 200;
    }

    let encoder = Encoder::default().quality(75);
    let jpeg_rgb = encoder.encode_rgb(&rgb, w, h).unwrap();
    let jpeg_rgba = encoder.encode_rgba(&rgba, w, h).unwrap();

    assert_eq!(
        jpeg_rgb, jpeg_rgba,
        "RGBA encode must produce identical JPEG to RGB encode"
    );
}

/// encode_rgba rejects wrong buffer size.
#[test]
fn test_encode_rgba_buffer_validation() {
    let encoder = Encoder::default();
    // Too small buffer
    let result = encoder.encode_rgba(&[0u8; 10], 64, 64);
    assert!(result.is_err());
}

/// encode_rgba_with_stop respects cancellation.
#[test]
fn test_encode_rgba_with_stop() {
    use enough::{Stop, StopReason};

    struct AlreadyCancelled;
    impl Stop for AlreadyCancelled {
        fn check(&self) -> core::result::Result<(), StopReason> {
            Err(StopReason::Cancelled)
        }
        fn may_stop(&self) -> bool {
            true
        }
    }

    let rgba = vec![128u8; 64 * 64 * 4];
    let encoder = Encoder::default();
    let result = encoder.encode_rgba_with_stop(&rgba, 64, 64, &AlreadyCancelled);
    assert!(result.is_err());
}

// ============================================================================
// Shared helpers for the color-space tests below
// ============================================================================

const ALL_PRESETS: [mozjpeg_rs::Preset; 4] = [
    mozjpeg_rs::Preset::BaselineFastest,
    mozjpeg_rs::Preset::BaselineBalanced,
    mozjpeg_rs::Preset::ProgressiveBalanced,
    mozjpeg_rs::Preset::ProgressiveSmallest,
];

/// Deterministic RGB image whose three channels carry different content.
fn three_channel_image(width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut state = 0x2545_f491u32;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            rgb.push(((x * 255 / w.max(1)) as u8).wrapping_add((state & 7) as u8));
            rgb.push((y * 255 / h.max(1)) as u8);
            rgb.push((((x + y) * 9) % 256) as u8);
        }
    }
    rgb
}

/// C mozjpeg's `rgb_gray_convert`: Y = 0.299 R + 0.587 G + 0.114 B in 16-bit
/// fixed point with rounding — the Y of its YCbCr conversion.
fn c_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            ((19595 * p[0] as u32 + 38470 * p[1] as u32 + 7471 * p[2] as u32 + 32768) >> 16) as u8
        })
        .collect()
}

/// Walk the header and return every `(marker, payload)` segment before SOS
/// (payload excludes the length bytes), plus each SOS header payload.
fn header_segments(jpeg: &[u8]) -> Vec<(u8, &[u8])> {
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "missing SOI");
    let mut out = Vec::new();
    let mut pos = 2;
    while pos + 4 <= jpeg.len() {
        assert_eq!(jpeg[pos], 0xFF, "expected marker at {pos}");
        let marker = jpeg[pos + 1];
        if marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        out.push((marker, &jpeg[pos + 4..pos + 2 + len]));
        pos += 2 + len;
        if marker == 0xDA {
            // Skip entropy-coded data to the next marker (not RSTn / stuffing).
            while pos + 1 < jpeg.len()
                && !(jpeg[pos] == 0xFF
                    && jpeg[pos + 1] != 0
                    && !(0xD0..=0xD7).contains(&jpeg[pos + 1]))
            {
                pos += 1;
            }
        }
    }
    out
}

/// Frame components as `(id, h, v, tq)` from the SOF segment.
fn sof_components(jpeg: &[u8]) -> Vec<(u8, u8, u8, u8)> {
    let (_, sof) = header_segments(jpeg)
        .into_iter()
        .find(|(m, _)| matches!(m, 0xC0..=0xC2))
        .expect("no SOF");
    (0..sof[5] as usize)
        .map(|c| {
            let s = &sof[6 + 3 * c..9 + 3 * c];
            (s[0], s[1] >> 4, s[1] & 0x0F, s[2])
        })
        .collect()
}

// ============================================================================
// Subsampling::Gray with color input (GitHub #9)
// ============================================================================

/// The exact reproduction from GitHub #9: progressive presets panicked with
/// "index out of bounds" in the SOS writer.
#[test]
fn test_issue9_gray_subsampling_progressive_no_panic() {
    let px = vec![128u8; 8 * 8 * 3];
    for preset in [
        mozjpeg_rs::Preset::ProgressiveBalanced,
        mozjpeg_rs::Preset::ProgressiveSmallest,
    ] {
        let jpeg = Encoder::new(preset)
            .quality(75)
            .subsampling(Subsampling::Gray)
            .trellis(TrellisConfig::default())
            .encode_rgb(&px, 8, 8)
            .unwrap();
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        let pixels = decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
        assert_eq!(pixels.len(), 64);
    }
}

/// Color input with `Subsampling::Gray` is a grayscale encode of its luma:
/// byte-identical to `encode_gray` of C's RGB->gray conversion, for every
/// preset and color entry point. Baseline presets used to write a
/// 1-component SOF over 3-component scan data (undecodable), progressive
/// presets panicked.
#[test]
fn test_gray_subsampling_with_color_input_encodes_luma() {
    for &(width, height) in &[(8u32, 8u32), (13, 7), (37, 29), (64, 48)] {
        let rgb = three_channel_image(width, height);
        let luma = c_luma(&rgb);
        let rgba: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 0x7F])
            .collect();
        let stride = width as usize * 3 + 5;
        let mut padded = vec![0xAAu8; stride * height as usize];
        for (dst, src) in padded
            .chunks_mut(stride)
            .zip(rgb.chunks(width as usize * 3))
        {
            dst[..src.len()].copy_from_slice(src);
        }

        for preset in ALL_PRESETS {
            let encoder = Encoder::new(preset)
                .quality(80)
                .subsampling(Subsampling::Gray);
            let expected = encoder.encode_gray(&luma, width, height).unwrap();
            let ctx = format!("{preset:?} {width}x{height}");

            assert_eq!(
                encoder.encode_rgb(&rgb, width, height).unwrap(),
                expected,
                "rgb {ctx}"
            );
            assert_eq!(
                encoder.encode_rgba(&rgba, width, height).unwrap(),
                expected,
                "rgba {ctx}"
            );
            assert_eq!(
                encoder
                    .encode_rgb_strided(&padded, width, height, stride)
                    .unwrap(),
                expected,
                "strided {ctx}"
            );

            assert_eq!(sof_components(&expected), [(1, 1, 1, 0)], "{ctx}");
            let mut decoder = jpeg_decoder::Decoder::new(&expected[..]);
            let decoded = decoder.decode().unwrap();
            assert_eq!(
                decoder.info().unwrap().pixel_format,
                jpeg_decoder::PixelFormat::L8
            );
            assert_eq!(decoded.len(), luma.len(), "{ctx}");
        }
    }
}

/// `encode_ycbcr_planar` with `Subsampling::Gray` encodes the Y plane (C's
/// YCbCr -> JCS_GRAYSCALE); the chroma planes are not read.
#[test]
fn test_gray_subsampling_ycbcr_planar_uses_luma_plane() {
    let (width, height) = (37u32, 29u32);
    let y = c_luma(&three_channel_image(width, height));
    for preset in ALL_PRESETS {
        let encoder = Encoder::new(preset).subsampling(Subsampling::Gray);
        let expected = encoder.encode_gray(&y, width, height).unwrap();
        assert_eq!(
            encoder
                .encode_ycbcr_planar(&y, &[], &[], width, height)
                .unwrap(),
            expected,
            "{preset:?}"
        );

        let stride = width as usize + 3;
        let mut padded = vec![0u8; stride * height as usize];
        for (dst, src) in padded.chunks_mut(stride).zip(y.chunks(width as usize)) {
            dst[..src.len()].copy_from_slice(src);
        }
        assert_eq!(
            encoder
                .encode_ycbcr_planar_strided(&padded, stride, &[], 0, &[], 0, width, height)
                .unwrap(),
            expected,
            "{preset:?} strided"
        );
    }
}

/// The streaming encoder honors `Subsampling::Gray` for RGB scanlines too
/// (it used to ignore it and write a 4:4:4 YCbCr file).
#[test]
fn test_streaming_gray_subsampling_with_rgb_input() {
    // Height not a multiple of 8 and writes that straddle MCU rows exercise
    // both the buffered conversion and finish()'s last-row padding.
    let (width, height) = (37u32, 29u32);
    let rgb = three_channel_image(width, height);
    let luma = c_luma(&rgb);

    let mut from_rgb = Vec::new();
    let mut stream = StreamingEncoder::baseline_fastest()
        .quality(80)
        .subsampling(Subsampling::Gray)
        .start_rgb(width, height, &mut from_rgb)
        .unwrap();
    for rows in rgb.chunks(5 * width as usize * 3) {
        stream.write_scanlines(rows).unwrap();
    }
    stream.finish().unwrap();

    let expected = StreamingEncoder::baseline_fastest()
        .quality(80)
        .subsampling(Subsampling::Gray)
        .encode_gray(&luma, width, height)
        .unwrap();
    assert_eq!(from_rgb, expected);
    assert_eq!(sof_components(&from_rgb), [(1, 1, 1, 0)]);
    let decoded = jpeg_decoder::Decoder::new(&from_rgb[..]).decode().unwrap();
    assert_eq!(decoded.len(), luma.len());
}

// ============================================================================
// JpegColorSpace::Rgb (GitHub #10)
// ============================================================================

/// RGB output is flagged the way libjpeg's JCS_RGB is: component IDs 'R','G',
/// 'B', all 1x1 on quant table 0 (whatever `subsampling` says), one DQT, all
/// Huffman tables in slot 0, an Adobe APP14 with transform 0 and no JFIF.
#[test]
fn test_rgb_color_space_markers() {
    use mozjpeg_rs::JpegColorSpace;

    let (width, height) = (37u32, 29u32);
    let rgb = three_channel_image(width, height);
    for preset in ALL_PRESETS {
        let jpeg = Encoder::new(preset)
            .color_space(JpegColorSpace::Rgb)
            .encode_rgb(&rgb, width, height)
            .unwrap();
        let segments = header_segments(&jpeg);

        assert_eq!(
            sof_components(&jpeg),
            [(b'R', 1, 1, 0), (b'G', 1, 1, 0), (b'B', 1, 1, 0)],
            "{preset:?}"
        );
        assert!(
            !segments.iter().any(|(m, _)| *m == 0xE0),
            "{preset:?}: JFIF APP0 must not be written for RGB"
        );
        let adobe: Vec<_> = segments.iter().filter(|(m, _)| *m == 0xEE).collect();
        assert_eq!(adobe.len(), 1, "{preset:?}");
        assert_eq!(
            adobe[0].1, b"Adobe\x00\x64\x00\x00\x00\x00\x00",
            "{preset:?}"
        );

        // One 8-bit table 0: Pq/Tq byte + 64 entries
        let dqt: Vec<_> = segments.iter().filter(|(m, _)| *m == 0xDB).collect();
        assert_eq!(dqt.len(), 1, "{preset:?}");
        assert_eq!(dqt[0].1.len(), 65, "{preset:?}");
        assert_eq!(dqt[0].1[0], 0x00, "{preset:?}");

        for (_, dht) in segments.iter().filter(|(m, _)| *m == 0xC4) {
            let mut p = 0;
            while p < dht.len() {
                assert_eq!(dht[p] & 0x0F, 0, "{preset:?}: Huffman table outside slot 0");
                let count: usize = dht[p + 1..p + 17].iter().map(|&n| n as usize).sum();
                p += 17 + count;
            }
        }
        for (_, sos) in segments.iter().filter(|(m, _)| *m == 0xDA) {
            for c in 0..sos[0] as usize {
                assert_eq!(
                    sos[2 + 2 * c],
                    0x00,
                    "{preset:?}: scan table selector not 0"
                );
            }
        }
    }
}

/// The point of RGB mode: no color transform and no subsampling, so a channel
/// never picks up another channel's quantization error. Constant G and B
/// planes must decode exactly constant while R carries detail.
#[test]
fn test_rgb_color_space_keeps_channels_independent() {
    use mozjpeg_rs::{JpegColorSpace, TrellisMode};

    let (width, height) = (48u32, 40u32);
    let pattern = three_channel_image(width, height);
    let rgb: Vec<u8> = pattern
        .chunks_exact(3)
        .flat_map(|p| [p[0], 0, 255])
        .collect();

    let mut encoders: Vec<(String, Encoder)> = ALL_PRESETS
        .iter()
        .map(|&p| (format!("{p:?}"), Encoder::new(p).quality(85)))
        .collect();
    for progressive in [false, true] {
        encoders.push((
            format!("MozjpegExact progressive={progressive}"),
            Encoder::new(mozjpeg_rs::Preset::BaselineBalanced)
                .progressive(progressive)
                .trellis(TrellisConfig::default().mode(TrellisMode::mozjpeg_exact())),
        ));
    }

    for (name, encoder) in encoders {
        let jpeg = encoder
            .clone()
            .color_space(JpegColorSpace::Rgb)
            .encode_rgb(&rgb, width, height)
            .unwrap();
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        let decoded = decoder.decode().unwrap();
        assert_eq!(
            decoder.info().unwrap().pixel_format,
            jpeg_decoder::PixelFormat::RGB24
        );
        assert!(
            decoded.chunks(3).all(|p| p[1] == 0 && p[2] == 255),
            "{name}: G/B leaked"
        );
        let r_psnr = calculate_psnr(
            &rgb.iter().step_by(3).copied().collect::<Vec<_>>(),
            &decoded.iter().step_by(3).copied().collect::<Vec<_>>(),
        );
        assert!(r_psnr > 30.0, "{name}: R PSNR {r_psnr:.1} dB");

        // The same input through YCbCr 4:4:4 does leak R's detail into G/B,
        // so the assertion above is a real distinction.
        let ycc = encoder
            .subsampling(Subsampling::S444)
            .encode_rgb(&rgb, width, height)
            .unwrap();
        let ycc_decoded = jpeg_decoder::Decoder::new(&ycc[..]).decode().unwrap();
        assert!(
            !ycc_decoded.chunks(3).all(|p| p[1] == 0 && p[2] == 255),
            "{name}"
        );
    }
}

/// jpeg-decoder and zune-jpeg both recognize the file as RGB and agree on
/// the pixels (up to IDCT rounding).
#[test]
fn test_rgb_color_space_decoders_agree() {
    use mozjpeg_rs::JpegColorSpace;
    use zune_jpeg::zune_core::bytestream::ZCursor;

    let (width, height) = (37u32, 29u32);
    let rgb = three_channel_image(width, height);
    for preset in ALL_PRESETS {
        let jpeg = Encoder::new(preset)
            .quality(90)
            .color_space(JpegColorSpace::Rgb)
            .encode_rgb(&rgb, width, height)
            .unwrap();
        let a = jpeg_decoder::Decoder::new(&jpeg[..]).decode().unwrap();
        let mut zune = zune_jpeg::JpegDecoder::new(ZCursor::new(&jpeg[..]));
        let b = zune.decode().unwrap();
        assert_eq!(
            zune.input_colorspace(),
            Some(zune_jpeg::zune_core::colorspace::ColorSpace::RGB),
            "{preset:?}"
        );
        assert_eq!(a.len(), b.len());
        let max_diff = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(max_diff <= 1, "{preset:?}: decoders differ by {max_diff}");
    }
}

/// In RGB mode chroma settings have nothing to act on: every subsampling
/// mode, `chroma_quality` and a custom chroma table give the same bytes.
/// RGBA and strided input match RGB.
#[test]
fn test_rgb_color_space_ignores_chroma_settings() {
    use mozjpeg_rs::JpegColorSpace;

    let (width, height) = (37u32, 29u32);
    let rgb = three_channel_image(width, height);
    let rgba: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 0])
        .collect();
    for preset in ALL_PRESETS {
        let base = Encoder::new(preset)
            .quality(80)
            .color_space(JpegColorSpace::Rgb);
        let expected = base.encode_rgb(&rgb, width, height).unwrap();
        for subsampling in [
            Subsampling::S444,
            Subsampling::S422,
            Subsampling::S420,
            Subsampling::S440,
        ] {
            let jpeg = base
                .clone()
                .subsampling(subsampling)
                .encode_rgb(&rgb, width, height)
                .unwrap();
            assert_eq!(jpeg, expected, "{preset:?} {subsampling:?}");
        }
        let jpeg = base
            .clone()
            .chroma_quality(Some(20))
            .encode_rgb(&rgb, width, height)
            .unwrap();
        assert_eq!(jpeg, expected, "{preset:?} chroma_quality");
        let jpeg = base
            .clone()
            .custom_chroma_qtable([99; 64])
            .encode_rgb(&rgb, width, height)
            .unwrap();
        assert_eq!(jpeg, expected, "{preset:?} custom_chroma_qtable");
        assert_eq!(
            base.encode_rgba(&rgba, width, height).unwrap(),
            expected,
            "{preset:?} rgba"
        );
    }
}

/// Without optimize_scans, progressive RGB uses C's all-purpose script:
/// interleaved DC, then luma-style successive approximation for every
/// channel (13 scans) instead of the YCbCr script's luma-only SA.
#[test]
fn test_rgb_color_space_progressive_script() {
    use mozjpeg_rs::JpegColorSpace;

    let rgb = three_channel_image(32, 32);
    let jpeg = Encoder::new(mozjpeg_rs::Preset::ProgressiveBalanced)
        .color_space(JpegColorSpace::Rgb)
        .encode_rgb(&rgb, 32, 32)
        .unwrap();
    // (components, Ss, Se, Ah, Al) per scan
    let scans: Vec<(Vec<u8>, u8, u8, u8, u8)> = header_segments(&jpeg)
        .into_iter()
        .filter(|(m, _)| *m == 0xDA)
        .map(|(_, s)| {
            let n = s[0] as usize;
            let comps = (0..n).map(|c| s[1 + 2 * c]).collect();
            let tail = &s[1 + 2 * n..];
            (comps, tail[0], tail[1], tail[2] >> 4, tail[2] & 0x0F)
        })
        .collect();
    let mut expected = vec![(b"RGB".to_vec(), 0, 0, 0, 0)];
    for (ss, se, ah, al) in [(1, 8, 0, 2), (9, 63, 0, 2), (1, 63, 2, 1), (1, 63, 1, 0)] {
        for c in *b"RGB" {
            expected.push((vec![c], ss, se, ah, al));
        }
    }
    assert_eq!(scans, expected);
}

/// Contradictory or impossible requests are errors, not silent fallbacks.
#[test]
fn test_rgb_color_space_rejected_combinations() {
    use mozjpeg_rs::{Error, JpegColorSpace};

    let rgb = three_channel_image(16, 16);
    let encoder = Encoder::default().color_space(JpegColorSpace::Rgb);

    // Grayscale output and RGB output at once
    let result = encoder
        .clone()
        .subsampling(Subsampling::Gray)
        .encode_rgb(&rgb, 16, 16);
    assert!(
        matches!(result, Err(Error::UnsupportedFeature(_))),
        "{result:?}"
    );

    // Planar input is already YCbCr
    let plane = vec![128u8; 16 * 16];
    let result = encoder.encode_ycbcr_planar(&plane, &plane, &plane, 16, 16);
    assert!(
        matches!(result, Err(Error::UnsupportedFeature(_))),
        "{result:?}"
    );
}

/// Grayscale input stays grayscale; metadata still goes in, after the Adobe
/// marker (C's write_file_header order).
#[test]
fn test_rgb_color_space_gray_input_and_metadata() {
    use mozjpeg_rs::JpegColorSpace;

    let gray: Vec<u8> = (0..32 * 32).map(|i| (i * 7 % 256) as u8).collect();
    let encoder = Encoder::default().quality(80);
    assert_eq!(
        encoder
            .clone()
            .color_space(JpegColorSpace::Rgb)
            .encode_gray(&gray, 32, 32)
            .unwrap(),
        encoder.encode_gray(&gray, 32, 32).unwrap()
    );

    let exif = b"MM\x00\x2a\x00\x00\x00\x08\x00\x00".to_vec();
    let icc = vec![0x42u8; 300];
    let jpeg = Encoder::default()
        .color_space(JpegColorSpace::Rgb)
        .exif_data(exif)
        .icc_profile(icc)
        .encode_rgb(&three_channel_image(32, 32), 32, 32)
        .unwrap();
    let apps: Vec<u8> = app_segments(&jpeg).iter().map(|(n, _)| *n).collect();
    assert_eq!(apps, [14, 1, 2]);
}

// ============================================================================
// Input validation hardening (dimensions, streaming row counts)
// ============================================================================

/// Dimensions past the JPEG SOF 2-byte field (65535) must be rejected, not
/// silently truncated into the `u16` header (65537 -> a 1-px header).
#[test]
fn test_dimension_over_65535_rejected() {
    use mozjpeg_rs::Error;
    let enc = Encoder::new(mozjpeg_rs::Preset::BaselineFastest);
    // Buffer length is checked after the dimension gate, so a tiny buffer is
    // fine — we assert on the error variant, and that nothing is encoded.
    for (w, h) in [(65_536u32, 1u32), (1, 65_536), (70_000, 70_000)] {
        assert!(
            matches!(
                enc.encode_rgb(&[], w, h),
                Err(Error::InvalidDimensions { .. })
            ),
            "rgb {w}x{h}"
        );
        assert!(
            matches!(
                enc.encode_gray(&[], w, h),
                Err(Error::InvalidDimensions { .. })
            ),
            "gray {w}x{h}"
        );
    }
    // 65535 is the largest valid value: it must pass the dimension gate and
    // fail only on the buffer-length check (proving the gate let it through).
    assert!(matches!(
        enc.encode_gray(&[0u8; 4], 65_535, 1),
        Err(Error::BufferSizeMismatch { .. })
    ));
}

/// Streaming must receive exactly the declared number of scanlines: too few
/// by `finish()` is an error, and writing more than the height is an error.
#[test]
fn test_streaming_scanline_count_enforced() {
    use mozjpeg_rs::Error;
    let (w, h) = (16u32, 32u32);
    let row = vec![90u8; (w * 3) as usize];

    // Too few rows: finish() rejects.
    let mut out = Vec::new();
    let mut s = StreamingEncoder::baseline_fastest()
        .start_rgb(w, h, &mut out)
        .unwrap();
    for _ in 0..16 {
        s.write_scanlines(&row).unwrap();
    }
    assert!(matches!(
        s.finish(),
        Err(Error::ScanlineCountMismatch {
            expected: 32,
            received: 16
        })
    ));

    // Too many rows: write_scanlines rejects as soon as the total exceeds h.
    let mut out = Vec::new();
    let mut s = StreamingEncoder::baseline_fastest()
        .start_rgb(w, h, &mut out)
        .unwrap();
    for _ in 0..32 {
        s.write_scanlines(&row).unwrap();
    }
    assert!(matches!(
        s.write_scanlines(&row),
        Err(Error::ScanlineCountMismatch { expected: 32, .. })
    ));

    // Exactly h rows still succeeds and decodes to the full height.
    let mut out = Vec::new();
    let mut s = StreamingEncoder::baseline_fastest()
        .start_rgb(w, h, &mut out)
        .unwrap();
    for _ in 0..32 {
        s.write_scanlines(&row).unwrap();
    }
    s.finish().unwrap();
    let mut d = jpeg_decoder::Decoder::new(&out[..]);
    d.decode().unwrap();
    assert_eq!(d.info().unwrap().height, 32);
}
