# Reproducible benchmark environment for mozjpeg-oxide vs C mozjpeg
#
# Build:   docker build -t mozjpeg-oxide-bench benchmark/
# Run:     docker run --rm -v $(pwd)/results:/results mozjpeg-oxide-bench
#
# The benchmark produces:
#   - benchmark_results.csv: Raw data for all images/qualities
#   - summary.txt: Summary statistics by quality level

FROM rust:1.83-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    nasm \
    pkg-config \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY . .

# Build the benchmark binary
RUN cargo build --release --example pareto_benchmark

# Fetch corpus
RUN ./scripts/fetch-corpus.sh --full

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libpng16-16 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /benchmark

# Copy built binary and corpus
COPY --from=builder /build/target/release/examples/pareto_benchmark /benchmark/
COPY --from=builder /build/corpus /benchmark/corpus

# Create output directory
RUN mkdir -p /results

# Default: run full benchmark with both corpora
ENTRYPOINT ["/benchmark/pareto_benchmark"]
CMD ["--corpus", "/benchmark/corpus", "--output", "/results/benchmark_results.csv"]
