//! Utilities for locating test corpus images.
//!
//! This module provides cross-platform utilities for finding test images,
//! whether from the local corpus directory, environment variables, or
//! the bundled test images.

use std::path::{Path, PathBuf};

/// Returns the path to the corpus directory.
///
/// Checks in order:
/// 1. `MOZJPEG_CORPUS_DIR` environment variable
/// 2. `CODEC_CORPUS_DIR` environment variable
/// 3. `./corpus/` relative to project root
/// 4. `../codec-corpus/` sibling directory (legacy)
///
/// Returns `None` if no corpus directory is found.
pub fn corpus_dir() -> Option<PathBuf> {
    // Check environment variables first
    if let Ok(dir) = std::env::var("MOZJPEG_CORPUS_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }

    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }

    // Check ./corpus/ relative to project root
    let project_root = project_root()?;
    let corpus = project_root.join("corpus");
    if corpus.is_dir() {
        return Some(corpus);
    }

    // Legacy: check sibling codec-corpus directory
    let sibling = project_root.parent()?.join("codec-corpus");
    if sibling.is_dir() {
        return Some(sibling);
    }

    None
}

/// Returns the path to the Kodak test images.
pub fn kodak_dir() -> Option<PathBuf> {
    let corpus = corpus_dir()?;
    let kodak = corpus.join("kodak");
    if kodak.is_dir() {
        Some(kodak)
    } else {
        None
    }
}

/// Returns the path to the CLIC validation images.
pub fn clic_validation_dir() -> Option<PathBuf> {
    let corpus = corpus_dir()?;

    // Try clic2025/validation first
    let clic = corpus.join("clic2025").join("validation");
    if clic.is_dir() {
        return Some(clic);
    }

    // Try just clic2025
    let clic = corpus.join("clic2025");
    if clic.is_dir() {
        return Some(clic);
    }

    None
}

/// Returns paths to all available corpus directories.
pub fn all_corpus_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(kodak) = kodak_dir() {
        dirs.push(kodak);
    }

    if let Some(clic) = clic_validation_dir() {
        dirs.push(clic);
    }

    dirs
}

/// Returns the path to the bundled test images (always available).
///
/// These are small test images bundled with the crate for CI.
pub fn bundled_test_images_dir() -> Option<PathBuf> {
    let project_root = project_root()?;
    let test_images = project_root.join("tests").join("images");
    if test_images.is_dir() {
        Some(test_images)
    } else {
        None
    }
}

/// Returns a specific bundled test image path.
pub fn bundled_test_image(name: &str) -> Option<PathBuf> {
    let dir = bundled_test_images_dir()?;
    let path = dir.join(name);
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

/// Returns the project root directory.
fn project_root() -> Option<PathBuf> {
    // Try CARGO_MANIFEST_DIR first (works in tests/examples)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        return Some(PathBuf::from(manifest_dir));
    }

    // Fall back to current directory
    std::env::current_dir().ok()
}

/// Returns PNG files from a directory, sorted by name.
pub fn png_files_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("png") {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

/// Loads an image from a path, returning RGB data, width, and height.
///
/// Supports PNG files. Returns `None` on error or unsupported format.
#[cfg(feature = "png")]
pub fn load_png_as_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    use std::fs::File;

    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => bytes.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => {
            bytes.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect()
        }
        _ => return None,
    };

    Some((rgb_data, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_images_available() {
        let dir = bundled_test_images_dir();
        assert!(dir.is_some(), "Bundled test images directory should exist");

        let image = bundled_test_image("1.png");
        assert!(image.is_some(), "Bundled 1.png should exist");
    }

    #[test]
    fn test_bundled_test_image_nonexistent() {
        let image = bundled_test_image("nonexistent_image_12345.png");
        assert!(image.is_none(), "Nonexistent image should return None");
    }

    #[test]
    fn test_project_root() {
        let root = project_root();
        assert!(root.is_some(), "Should find project root");

        let root = root.unwrap();
        assert!(
            root.join("Cargo.toml").is_file(),
            "Project root should contain Cargo.toml"
        );
    }

    #[test]
    fn test_corpus_dir_returns_valid_or_none() {
        // This test works whether or not corpus exists
        let result = corpus_dir();
        if let Some(dir) = result {
            assert!(
                dir.is_dir(),
                "If corpus_dir returns Some, it should be a directory"
            );
        }
        // If None, that's also valid - corpus might not be set up
    }

    #[test]
    fn test_kodak_dir_returns_valid_or_none() {
        let result = kodak_dir();
        if let Some(dir) = result {
            assert!(
                dir.is_dir(),
                "If kodak_dir returns Some, it should be a directory"
            );
            // Should have PNG files if it exists
            let files = png_files_in_dir(&dir);
            assert!(!files.is_empty(), "Kodak dir should have PNG files");
        }
    }

    #[test]
    fn test_clic_validation_dir_returns_valid_or_none() {
        let result = clic_validation_dir();
        if let Some(dir) = result {
            assert!(
                dir.is_dir(),
                "If clic_validation_dir returns Some, it should be a directory"
            );
        }
    }

    #[test]
    fn test_all_corpus_dirs_returns_valid() {
        let dirs = all_corpus_dirs();
        for dir in &dirs {
            assert!(dir.is_dir(), "All corpus dirs should be directories");
        }
    }

    #[test]
    fn test_png_files_in_dir_bundled() {
        let dir = bundled_test_images_dir().expect("Bundled images should exist");
        let files = png_files_in_dir(&dir);
        assert!(
            !files.is_empty(),
            "Bundled images dir should have PNG files"
        );

        // Files should be sorted
        let mut sorted_files = files.clone();
        sorted_files.sort();
        assert_eq!(files, sorted_files, "Files should be sorted");

        // All files should have .png extension
        for file in &files {
            assert_eq!(
                file.extension().and_then(|e| e.to_str()),
                Some("png"),
                "All files should have .png extension"
            );
        }
    }

    #[test]
    fn test_png_files_in_dir_empty() {
        // Test with a directory that exists but has no PNGs
        let temp_dir = std::env::temp_dir();
        let files = png_files_in_dir(&temp_dir);
        // Should return empty vec or find some PNGs (either is fine)
        // Main thing is it shouldn't crash
        let _ = files;
    }

    #[test]
    fn test_png_files_in_dir_nonexistent() {
        // Test with a nonexistent directory
        let nonexistent = PathBuf::from("/nonexistent/path/that/does/not/exist");
        let files = png_files_in_dir(&nonexistent);
        assert!(files.is_empty(), "Nonexistent dir should return empty vec");
    }

    #[cfg(feature = "png")]
    #[test]
    fn test_load_png_as_rgb() {
        // Test loading a bundled PNG
        let path = bundled_test_image("1.png").expect("Bundled 1.png should exist");
        let result = load_png_as_rgb(&path);
        assert!(result.is_some(), "Should successfully load bundled PNG");

        let (rgb, width, height) = result.unwrap();
        assert!(width > 0, "Width should be positive");
        assert!(height > 0, "Height should be positive");
        assert_eq!(
            rgb.len(),
            (width * height * 3) as usize,
            "RGB data should have correct size"
        );
    }

    #[cfg(feature = "png")]
    #[test]
    fn test_load_png_as_rgb_nonexistent() {
        let path = PathBuf::from("/nonexistent/image.png");
        let result = load_png_as_rgb(&path);
        assert!(result.is_none(), "Nonexistent file should return None");
    }
}
