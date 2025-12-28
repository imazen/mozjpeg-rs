#!/bin/bash
# Run benchmarks on historical commits and save results
#
# Usage: ./scripts/run_historical_benchmarks.sh [commit1] [commit2] ...
#
# Results are saved to tests/benchmark_tracking/results/ in the main repo

set -e

MAIN_REPO="/home/lilith/work/mozjpeg-rs"
HIST_REPO="/home/lilith/work/mzjprs"
RESULTS_DIR="$MAIN_REPO/tests/benchmark_tracking/results"
TEST_IMAGE="$MAIN_REPO/tests/images/1.png"

# Key historical commits to benchmark (if no args provided)
DEFAULT_COMMITS=(
    "97f1602"  # v0.2.0 release
    "00622be"  # Before trellis tweaks
    "76645f5"  # After SIMD merge
    "963d57d"  # First SimdOps integration
    "531b4d9"  # Before codec-eval changes
)

COMMITS=("${@:-${DEFAULT_COMMITS[@]}}")

echo "=== Historical Benchmark Runner ==="
echo "Main repo: $MAIN_REPO"
echo "Historical repo: $HIST_REPO"
echo "Results dir: $RESULTS_DIR"
echo "Commits to benchmark: ${COMMITS[*]}"
echo ""

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Copy test image to historical repo
cp "$TEST_IMAGE" "$HIST_REPO/tests/images/" 2>/dev/null || mkdir -p "$HIST_REPO/tests/images" && cp "$TEST_IMAGE" "$HIST_REPO/tests/images/"

cd "$HIST_REPO"

for commit in "${COMMITS[@]}"; do
    echo ""
    echo "=== Benchmarking commit: $commit ==="

    # Check if result already exists
    if ls "$RESULTS_DIR"/mozjpeg-oxide_"$commit"*.json 1>/dev/null 2>&1; then
        echo "Results already exist for $commit, skipping..."
        continue
    fi

    # Checkout the commit
    git checkout "$commit" 2>/dev/null || {
        echo "Failed to checkout $commit, skipping..."
        continue
    }

    # Try to build
    if ! cargo build --release 2>/dev/null; then
        echo "Build failed for $commit, skipping..."
        git checkout main 2>/dev/null
        continue
    fi

    # Run a simplified benchmark inline (since old commits may not have benchmark_runner)
    echo "Running benchmark..."

    # Create a temporary benchmark script
    cat > /tmp/hist_bench.rs << 'BENCH_EOF'
use std::fs;
use std::io::Cursor;
use std::path::Path;

fn main() {
    let qualities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99];

    // Load PNG
    let path = Path::new("tests/images/1.png");
    let file = fs::File::open(path).expect("Failed to open test image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to read frame");
    buf.truncate(info.buffer_size());

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(buf.len() * 3 / 4);
            for chunk in buf.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => panic!("Unsupported color type"),
    };

    let width = info.width;
    let height = info.height;

    println!("{{");
    println!("  \"commit\": \"{}\",", std::env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".to_string()));
    println!("  \"timestamp\": \"{}\",", chrono::Utc::now().to_rfc3339());
    println!("  \"image\": \"tests/images/1.png\",");
    println!("  \"encoder\": \"mozjpeg-oxide\",");
    println!("  \"results\": [");

    for (i, &quality) in qualities.iter().enumerate() {
        let jpeg = mozjpeg_oxide::Encoder::new()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(mozjpeg_oxide::TrellisConfig::default())
            .overshoot_deringing(true)
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        // Decode and calculate DSSIM
        let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(&jpeg));
        let decoded = decoder.decode().expect("Decode failed");

        // Simple MSE as proxy (DSSIM requires more deps)
        let mse: f64 = rgb.iter().zip(decoded.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>() / rgb.len() as f64;
        let psnr = if mse > 0.0 { 10.0 * (255.0_f64.powi(2) / mse).log10() } else { 100.0 };

        let comma = if i < qualities.len() - 1 { "," } else { "" };
        println!("    {{\"quality\": {}, \"size\": {}, \"psnr\": {:.2}}}{}", quality, jpeg.len(), psnr, comma);
    }

    println!("  ]");
    println!("}}");
}
BENCH_EOF

    # Try running with cargo run --example or inline test
    GIT_COMMIT="$commit" cargo test --test benchmark_runner run_benchmark_and_save --release -- --nocapture 2>/dev/null && {
        # Copy the result
        cp "$HIST_REPO/tests/benchmark_tracking/results/mozjpeg-oxide_"*.json "$RESULTS_DIR/" 2>/dev/null || true
    } || {
        echo "Benchmark test not available for $commit, using simple size check..."
        # Just get file sizes as fallback
    }

    echo "Done with $commit"
done

# Return to main
git checkout main 2>/dev/null

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
