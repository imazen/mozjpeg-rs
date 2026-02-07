# mozjpeg-rs development tasks

# Run all tests (native)
test:
    cargo test -p mozjpeg-rs --lib
    cargo test --test codec_comparison
    cargo test --test ffi_validation
    cargo test --test preset_parity

# Run unit tests only
test-lib:
    cargo test -p mozjpeg-rs --lib

# Cross-compile and test for i686 (32-bit x86)
test-i686:
    cross build -p mozjpeg-rs --target i686-unknown-linux-gnu
    cross test -p mozjpeg-rs --lib --target i686-unknown-linux-gnu

# Cross-compile and test for armv7 (32-bit ARM)
test-armv7:
    cross build -p mozjpeg-rs --target armv7-unknown-linux-gnueabihf
    cross test -p mozjpeg-rs --lib --target armv7-unknown-linux-gnueabihf

# Cross-compile and test for aarch64 (64-bit ARM)
test-aarch64:
    cross build -p mozjpeg-rs --target aarch64-unknown-linux-gnu
    cross test -p mozjpeg-rs --lib --target aarch64-unknown-linux-gnu

# Run all cross targets
test-cross: test-i686 test-armv7 test-aarch64

# Build and check (no tests)
check:
    cargo check -p mozjpeg-rs

# Clippy lint
clippy:
    cargo clippy -p mozjpeg-rs --lib -- -D warnings

# Format check
fmt-check:
    cargo fmt --all -- --check

# Format
fmt:
    cargo fmt --all
