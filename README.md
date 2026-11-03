# Whisper LRC

A high-performance, stateless audio-to-LRC synchronized lyrics microservice and Rust library crate powered by **Rust**, **Axum**, **Symphonia**, and **Whisper.cpp** (`whisper-rs`).

---

## Key Features

* **Pure Native Audio Pipeline**: Decodes MP3, WAV, FLAC, and AAC directly to 16 kHz mono PCM in pure Rust using `symphonia` (no external `ffmpeg` binary required).
* **Synchronized LRC Output**: Generates standard `[mm:ss.xx]Lyrics` timestamps for music players.
* **In-Memory LRU Cache**: Instant $\sim$0.1 ms responses for identical audio hashes (`X-Cache: HIT`).
* **Preemptive Priority Queue (Concurrency = 1)**:
  * Guarantees single-flight inference to prevent GPU VRAM out-of-memory crashes.
  * **Shortest Job First (SJF)**: Shorter tracks jump ahead in line.
  * **Preemption**: Short songs ($\le$ 10 min) cancel running podcasts ($>$ 10 min) and re-queue them so music gets the right of way.
* **Idle VRAM/RAM Eviction (10-min TTL)**:
  * When the queue is idle for 10 minutes, the model is evicted from VRAM/RAM (`cudaFree`), freeing resources for video transcoding or background tasks.
  * Reloads on-demand from SSD in $\sim$1.5s when a new track arrives.
* **Dual Accuracy Levels**:
  * `normal` *(default)*: Uses `medium` model with GPU/CUDA acceleration ($\sim$2.5 GB VRAM).
  * `high`: Uses `large-v3` model on multi-threaded CPU for maximum precision.

---

## 1. REST API Usage (Docker)

### Quick Start with Docker Compose

```bash
docker compose up -d --build
```

The service will start listening on port `9500`.

### Endpoints

#### `POST /lyrics`

Accepts an uploaded audio file or server-local file path.

**Multipart Form Parameters:**
* `file`: Audio file binary (MP3, WAV, FLAC, AAC)
* `filepath`: Server-local file path (optional zero-copy alternative to upload)
* `accuracy`: `normal` (default) or `high`
* `lang`: Language code, e.g. `ja` (default), `en`, or `auto`
* `format`: `lrc` (default, returns `text/plain`) or `json`
* `refresh`: `false` (default) or `true` (forces fresh inference, bypassing cache)

#### Example: Transcribe to Synced LRC
```bash
curl -X POST "http://localhost:9500/lyrics" \
  -F "file=@song.mp3" \
  -F "accuracy=normal" \
  -F "lang=ja"
```

**Output:**
```lrc
[00:14.20]静かな夜の空に
[00:18.55]星が一つ光ってる
[00:23.10]見上げた瞳の先に
[00:28.40]君の面影を探していた
```

#### Example: Force Refresh & JSON Output
```bash
curl -X POST "http://localhost:9500/lyrics" \
  -F "file=@song.mp3" \
  -F "accuracy=high" \
  -F "format=json" \
  -F "refresh=true"
```

#### `GET /queue`
Check active jobs, queue depth, and cached track count:
```bash
curl http://localhost:9500/queue
```

#### `GET /health`
```bash
curl http://localhost:9500/health
```

---

## 2. Using as a Rust Library Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
whisper-lrc = { path = "../whisper-lrc" }
```

### Example

```rust
use std::path::PathBuf;
use whisper_lrc::{Accuracy, to_lrc};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let audio_bytes = tokio::fs::read("song.mp3").await?;
    let model_dir = PathBuf::from("./models");

    let lrc = to_lrc(
        &audio_bytes,
        Accuracy::Normal,
        "ja",
        model_dir,
        8, // CPU threads
    ).await?;

    println!("{lrc}");
    Ok(())
}
```

---

## Configuration

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `9500` | Port for the HTTP server |
| `MODEL_DIR` | `/models` | Directory where GGML models are cached |
| `CPU_THREADS` | `8` | Worker threads allocated for CPU inference |
| `IDLE_TIMEOUT_SECS` | `600` | Inactivity seconds before dropping VRAM/RAM context (0 = keep forever) |
| `MAX_QUEUE_CAPACITY`| `100` | Maximum queued jobs before returning HTTP 429 |
| `CACHE_CAPACITY` | `5000` | Max entries in the in-memory LRU cache |
