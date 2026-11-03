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
* **Direct Hotlink Streaming**: Accepts direct audio stream URLs (`GET /lyrics?url=...` or `POST`) and fetches audio on-the-fly, eliminating manual download/upload steps.
* **Native Vocal Isolation (Demucs)**:
  * Built-in multi-stem separator isolates pure singing phonemes from loud instruments before transcription.
  * Eliminates acoustic distortion from drums, bass, and guitars for crystal-clear lyrics.
* **State-of-the-Art Whisper Model**:
  * Powered by full 32-layer **`large-v3-q5_0`** (5-bit quantization, $\sim$1.08 GB).
  * Runs 100% on GPU (CUDA) using only $\sim$1.4 GB VRAM for maximum accuracy and fast inference.

---

## 1. REST API Usage (Docker)

### Quick Start with Docker Compose

```bash
docker compose up -d --build
```

The service will start listening on port `9500`.

### Endpoints

#### `POST /lyrics` & `GET /lyrics`

Accepts an uploaded audio file, server-local file path, or **hotlink audio stream URL**.

**Parameters (Multipart Form, Query Params, or JSON):**
* `file`: Audio file binary upload (MP3, WAV, FLAC, AAC)
* `url`: Direct hotlink / streaming audio URL (e.g. `https://listen.example.com/stream/...`)
* `filepath`: Server-local file path (optional zero-copy alternative to upload)
* `lang`: Language code, e.g. `ja` (default), `en`, `tr`, or `auto`
* `isolate`: `true` (default) or `false` (skips vocal isolation for podcasts or speech)
* `format`: `lrc` (default, returns `text/plain`) or `json`
* `refresh`: `false` (default) or `true` (forces fresh inference, bypassing cache)

#### Example 1: Transcribe via Direct Hotlink URL (GET or POST)
```bash
# Simple GET request with URL query parameter
curl "http://localhost:9500/lyrics?url=https://listen.example.com/stream/track.mp3&lang=tr"

# Or via POST multipart form
curl -X POST "http://localhost:9500/lyrics" \
  -F "url=https://listen.example.com/stream/track.mp3" \
  -F "lang=tr"
```

#### Example 2: Transcribe via File Upload
```bash
curl -X POST "http://localhost:9500/lyrics" \
  -F "file=@song.mp3" \
  -F "lang=ja"
```

#### Example 3: Transcribe via JSON Body (`POST /lyrics/json`)
```bash
curl -X POST "http://localhost:9500/lyrics/json" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://listen.example.com/stream/track.mp3", "lang": "tr"}'
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
use whisper_lrc::to_lrc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let audio_bytes = tokio::fs::read("song.mp3").await?;
    let model_dir = PathBuf::from("./models");

    let lrc = to_lrc(
        &audio_bytes,
        "ja",
        true, // isolate_vocals_first (via Demucs)
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
| `HOST` | `0.0.0.0` | Bind address |
| `MODEL_DIR` | `/models` | Directory where GGML models are cached |
| `TORCH_HOME` | `/models/torch` | Directory where Demucs PyTorch models are cached |
| `CPU_THREADS` | `8` | Worker threads allocated for Demucs & CPU inference |
| `IDLE_VRAM_TTL_SECS` | `600` | Inactivity seconds before dropping VRAM context (0 = keep forever) |
| `MAX_QUEUE_CAPACITY` | `32` | Maximum queued jobs before returning HTTP 429 |
| `CACHE_SIZE` | `500` | Max entries in the in-memory LRU cache |
| `CACHE_TTL_SECS` | `86400` | Expiration time for cached lyrics (24 hours) |
