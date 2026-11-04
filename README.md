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
* **Native Rust Vocal Isolation (`demucs-rs` / HTDemucs v4)**:
  * Built-in pure-Rust multi-stem separator (`demucs-core` powered by the **Burn** deep learning framework) isolates pure singing phonemes from loud instruments in-memory before transcription—zero Python, PyTorch, or FFmpeg runtime dependencies required.
  * Automatically downloads and caches `htdemucs.safetensors` ($\sim$84 MB) into `MODEL_DIR`.
* **Multi-Layer Anti-Hallucination & Metadata Conditioning**:
  * **Auto-Prompting via Metadata (`artist`, `title`, `prompt`)**: Automatically extracts embedded ID3/Vorbis `Artist` and `TrackTitle` tags via `symphonia` (or uses request parameters) to seed Whisper's `initial_prompt`, preventing phonetic katakana fallback on archaic/stylized lyrics.
  * **RMS Vocal Silence Gate**: Mutes sustained non-vocal passages ($\ge 1.0\text{s}$ below $-42\text{ dBFS}$) to pure digital zero so instrumental percussion/bell outros never trigger phantom speech.
  * **Greedy Decoding with Temperature Fallback**: Uses greedy single-token argmax with dynamic temperature fallback (`temperature_inc = 0.2`) to break out of repetition traps during silence or heavy instrumentals.
  * **Confidence & Hallucination Filtering**: Enforces `logprob_thold = -1.0`, `entropy_thold = 2.4`, per-segment token probability pruning (`avg_prob >= 0.20`), and a post-processing denylist for common training artifacts (`"アーメン"`, `"ご視聴ありがとうございました"`, Turkish subtitle watermarks, repetition loops).
* **State-of-the-Art Whisper Model (`large-v3-turbo`)**:
  * Powered by **`large-v3-turbo`** (4-layer pruned decoder, $\sim$1.6 GB fp16), offering 4x faster inference with significantly reduced hallucination risk on music.
  * Configurable via `WHISPER_MODEL` environment variable (e.g. `ggml-large-v3-turbo.bin`, `ggml-large-v2.bin`, `ggml-large-v3-q5_0.bin`).
  * Runs 100% on GPU (CUDA) using only $\sim$1.6–2.1 GB VRAM.

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
* `artist`: Optional artist name (auto-extracted from ID3/Vorbis tags if omitted)
* `title`: Optional song title (auto-extracted from ID3/Vorbis tags if omitted)
* `prompt`: Optional custom context/vocabulary hint appended to `initial_prompt`
* `isolate`: `false` (default, fastest) or `true` (enables Demucs vocal isolation before transcription)
* `format`: `lrc` (default, returns `text/plain`) or `json`
* `refresh`: `false` (default) or `true` (forces fresh inference, bypassing cache)

#### Example 1: Transcribe via Direct Hotlink URL (GET or POST)
```bash
# Simple GET request with URL, artist, and title query parameters
curl "http://localhost:9500/lyrics?url=https://listen.example.com/stream/track.mp3&lang=ja&artist=Kenji%20Kawai&title=Making%20of%20a%20Cyborg"

# Or via POST multipart form
curl -X POST "http://localhost:9500/lyrics" \
  -F "url=https://listen.example.com/stream/track.mp3" \
  -F "lang=ja" \
  -F "artist=Kenji Kawai" \
  -F "title=Making of a Cyborg"
```

#### Example 2: Transcribe via File Upload
```bash
curl -X POST "http://localhost:9500/lyrics" \
  -F "file=@song.mp3" \
  -F "lang=ja" \
  -F "artist=Kenji Kawai" \
  -F "title=Making of a Cyborg"
```

#### Example 3: Transcribe via JSON Body (`POST /lyrics/json`)
```bash
curl -X POST "http://localhost:9500/lyrics/json" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://listen.example.com/stream/track.mp3",
    "lang": "ja",
    "artist": "Kenji Kawai",
    "title": "Making of a Cyborg"
  }'
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
use whisper_lrc::{to_lrc, to_lrc_with_metadata};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let audio_bytes = tokio::fs::read("song.mp3").await?;
    let model_dir = PathBuf::from("./models");

    // Automatically extracts ID3/Vorbis artist & title tags if present:
    let lrc = to_lrc(
        &audio_bytes,
        "ja",
        true, // isolate_vocals_first (via Demucs)
        model_dir.clone(),
        8, // CPU threads
    ).await?;

    // Or supply explicit artist / title / prompt hints:
    let lrc_seeded = to_lrc_with_metadata(
        &audio_bytes,
        "ja",
        true,
        Some("Kenji Kawai"),
        Some("Making of a Cyborg"),
        None,
        model_dir,
        8,
    ).await?;

    println!("{lrc_seeded}");
    Ok(())
}
```

---

## Configuration

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `9500` | Port for the HTTP server |
| `HOST` | `0.0.0.0` | Bind address |
| `MODEL_DIR` | `/models` | Directory where Whisper GGML and Demucs `.safetensors` models are cached |
| `WHISPER_MODEL` | `ggml-large-v3-turbo.bin` | GGML Whisper model filename (e.g. `ggml-large-v3-turbo.bin`, `ggml-large-v2.bin`) |
| `CPU_THREADS` | `8` | Worker threads allocated for Demucs & CPU inference |
| `IDLE_VRAM_TTL_SECS` | `600` | Inactivity seconds before dropping VRAM context (0 = keep forever) |
| `MAX_QUEUE_CAPACITY` | `32` | Maximum queued jobs before returning HTTP 429 |
| `CACHE_SIZE` | `500` | Max entries in the in-memory LRU cache |
| `CACHE_TTL_SECS` | `86400` | Expiration time for cached lyrics (24 hours) |
