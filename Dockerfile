# ==============================================================================
# Builder Stage
# ==============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    pkg-config \
    clang \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

WORKDIR /usr/src/whisper-lrc

# Copy manifests and source
COPY Cargo.toml ./
COPY src ./src

# Build release binary with CUDA acceleration
RUN cargo build --release --features cuda

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HOST=0.0.0.0 \
    PORT=9500 \
    MODEL_DIR=/models \
    TORCH_HOME=/models/torch \
    CPU_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    python3-minimal \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Demucs for native vocal isolation
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir numpy demucs

COPY --from=builder /usr/src/whisper-lrc/target/release/whisper-lrc /usr/local/bin/whisper-lrc

EXPOSE 9500

CMD ["whisper-lrc"]
