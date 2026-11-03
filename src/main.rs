use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use whisper_lrc::audio::{
    apply_rms_silence_gate, decode_audio_to_16k_mono, extract_audio_metadata,
};
use whisper_lrc::cache::{CacheKey, LrcCache};
use whisper_lrc::isolation::isolate_vocals_to_16k_mono;
use whisper_lrc::queue::{start_worker_loop, PreemptiveQueue};
use whisper_lrc::transcriber::{build_initial_prompt, TranscriberService};

#[derive(Clone)]
struct AppState {
    queue: Arc<PreemptiveQueue>,
    cache: LrcCache,
    model_dir: PathBuf,
}

#[derive(Deserialize, Default)]
struct LyricsQuery {
    url: Option<String>,
    hotlink: Option<String>,
    filepath: Option<String>,
    lang: Option<String>,
    artist: Option<String>,
    title: Option<String>,
    prompt: Option<String>,
    isolate: Option<bool>,
    format: Option<String>,
    refresh: Option<bool>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "whisper_lrc=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "9500".to_string())
        .parse()
        .unwrap_or(9500);

    let model_dir = PathBuf::from(
        std::env::var("MODEL_DIR").unwrap_or_else(|_| "/models".to_string()),
    );
    let cpu_threads: i32 = std::env::var("CPU_THREADS")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap_or(8);

    let cache_size: u64 = std::env::var("CACHE_SIZE")
        .unwrap_or_else(|_| "500".to_string())
        .parse()
        .unwrap_or(500);

    let cache_ttl_secs: u64 = std::env::var("CACHE_TTL_SECS")
        .unwrap_or_else(|_| "86400".to_string())
        .parse()
        .unwrap_or(86400);

    let idle_vram_ttl_secs: u64 = std::env::var("IDLE_VRAM_TTL_SECS")
        .unwrap_or_else(|_| "600".to_string())
        .parse()
        .unwrap_or(600);

    let max_queue_capacity: usize = std::env::var("MAX_QUEUE_CAPACITY")
        .unwrap_or_else(|_| "32".to_string())
        .parse()
        .unwrap_or(32);

    tracing::info!("Initializing Whisper Transcriber Service...");
    let transcriber = Arc::new(TranscriberService::new(model_dir.clone(), cpu_threads));

    tracing::info!("Starting Preemptive Priority Queue (Concurrency = 1)...");
    let queue = Arc::new(PreemptiveQueue::new(max_queue_capacity));

    start_worker_loop(queue.clone(), transcriber.clone(), idle_vram_ttl_secs);

    let cache = LrcCache::new(cache_size, cache_ttl_secs);

    let state = AppState {
        queue,
        cache,
        model_dir,
    };

    // Max 250MB audio upload limit
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/queue", get(handle_queue_status))
        .route("/lyrics", post(handle_lyrics_multipart).get(handle_lyrics_query))
        .route("/lyrics/json", post(handle_lyrics_json))
        .layer(DefaultBodyLimit::max(250 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Whisper LRC server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_health() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "whisper-lrc",
        "model": "ggml-large-v3-q5_0",
        "vocal_isolation": "demucs-rs-htdemucs",
        "hotlink_supported": true,
        "runtime": "rust-axum"
    }))
}

async fn handle_queue_status(State(state): State<AppState>) -> impl IntoResponse {
    let q_status = state.queue.status().await;
    Json(json!({
        "queue": q_status,
        "cached_entries": state.cache.entry_count()
    }))
}

async fn fetch_hotlink_audio(url: &str) -> Result<Vec<u8>, (StatusCode, String)> {
    tracing::info!("Fetching hotlink audio from URL: {url}");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("HTTP client error: {e}")))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed fetching hotlink URL: {e}")))?;

    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Hotlink URL returned status HTTP {}", resp.status()),
        ));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed reading hotlink stream: {e}")))?;

    tracing::info!("Fetched hotlink audio ({} bytes).", bytes.len());
    Ok(bytes.to_vec())
}

#[allow(clippy::too_many_arguments)]
async fn process_audio_pipeline(
    state: &AppState,
    raw_bytes: Vec<u8>,
    lang: String,
    artist: Option<String>,
    title: Option<String>,
    prompt: Option<String>,
    isolate: bool,
    output_format: String,
    refresh: bool,
) -> Result<Response, (StatusCode, String)> {
    // Auto-extract embedded ID3/Vorbis artist & title tags if not supplied in request
    let (effective_artist, effective_title) = if artist.is_none() || title.is_none() {
        let embedded = extract_audio_metadata(&raw_bytes);
        (artist.or(embedded.artist), title.or(embedded.title))
    } else {
        (artist, title)
    };

    let initial_prompt = build_initial_prompt(
        effective_artist.as_deref(),
        effective_title.as_deref(),
        prompt.as_deref(),
    );

    // Calculate SHA-256 hash for in-memory LRU cache
    let mut hasher = Sha256::new();
    hasher.update(&raw_bytes);
    let audio_hash = hex::encode(hasher.finalize());

    let cache_key = CacheKey {
        audio_hash: audio_hash.clone(),
        lang: lang.clone(),
        isolate,
        prompt: initial_prompt.clone(),
    };

    // 1. In-memory Cache Check
    if !refresh {
        if let Some(cached_lrc) = state.cache.get(&cache_key).await {
            if output_format == "json" {
                return Ok((
                    StatusCode::OK,
                    [(header::HeaderName::from_static("x-cache"), "HIT")],
                    Json(json!({
                        "lrc": cached_lrc,
                        "lang": lang,
                        "prompt": initial_prompt,
                        "isolated": isolate,
                        "cached": true
                    })),
                )
                    .into_response());
            } else {
                return Ok((
                    StatusCode::OK,
                    [
                        (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
                        (header::HeaderName::from_static("x-cache"), "HIT"),
                    ],
                    cached_lrc,
                )
                    .into_response());
            }
        }
    }

    // 2. Vocal Isolation (pure-Rust demucs-rs) & Decode to 16kHz mono PCM
    let mut decoded = if isolate {
        match isolate_vocals_to_16k_mono(&raw_bytes, &state.model_dir).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("demucs-rs vocal isolation failed, falling back to mixed audio: {e}");
                tokio::task::spawn_blocking(move || decode_audio_to_16k_mono(&raw_bytes))
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Thread error: {e}")))?
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Audio decoding error: {e}")))?
            }
        }
    } else {
        tokio::task::spawn_blocking(move || decode_audio_to_16k_mono(&raw_bytes))
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Thread error: {e}")))?
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Audio decoding error: {e}")))?
    };

    // 3. Apply RMS silence gate to mute sustained instrumental/drum tails
    apply_rms_silence_gate(&mut decoded.pcm, 16000, -42.0);

    // 4. Submit to Preemptive Priority Queue
    let rx = state
        .queue
        .submit(
            decoded.pcm,
            decoded.duration_secs,
            lang.clone(),
            initial_prompt.clone(),
        )
        .await
        .map_err(|e| (StatusCode::TOO_MANY_REQUESTS, e))?;

    // 5. Await worker completion
    let result = rx
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Worker dropped job channel".to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 6. Store in LRU cache
    state.cache.insert(cache_key, result.lrc.clone()).await;

    if output_format == "json" {
        Ok((
            StatusCode::OK,
            [(header::HeaderName::from_static("x-cache"), "MISS")],
            Json(result),
        )
            .into_response())
    } else {
        Ok((
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
                (header::HeaderName::from_static("x-cache"), "MISS"),
            ],
            result.lrc,
        )
            .into_response())
    }
}

async fn handle_lyrics_multipart(
    State(state): State<AppState>,
    Query(query): Query<LyricsQuery>,
    mut multipart: Multipart,
) -> Result<Response, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut filepath = query.filepath;
    let mut hotlink_url = query.url.or(query.hotlink);
    let mut lang = query.lang.unwrap_or_else(|| "ja".to_string());
    let mut artist = query.artist;
    let mut title = query.title;
    let mut prompt = query.prompt;
    let mut output_format = query.format.unwrap_or_else(|| "lrc".to_string());
    let mut refresh = query.refresh.unwrap_or(false);
    let mut isolate = query.isolate.unwrap_or(true);

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {e}")))?
    {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "file" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed reading file: {e}")))?;
                audio_bytes = Some(data.to_vec());
            }
            "url" | "hotlink" | "link" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid url field: {e}")))?;
                hotlink_url = Some(text.trim().to_string());
            }
            "filepath" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid filepath field: {e}")))?;
                filepath = Some(text.trim().to_string());
            }
            "artist" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid artist field: {e}")))?;
                artist = Some(text.trim().to_string());
            }
            "title" | "song" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid title field: {e}")))?;
                title = Some(text.trim().to_string());
            }
            "prompt" | "initial_prompt" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid prompt field: {e}")))?;
                prompt = Some(text.trim().to_string());
            }
            "isolate" | "isolate_vocals" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid isolate field: {e}")))?;
                isolate = !text.trim().eq_ignore_ascii_case("false") && text.trim() != "0";
            }
            "lang" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid lang field: {e}")))?;
                lang = text.trim().to_string();
            }
            "format" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid format field: {e}")))?;
                output_format = text.trim().to_lowercase();
            }
            "refresh" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid refresh field: {e}")))?;
                refresh = text.trim().eq_ignore_ascii_case("true") || text.trim() == "1";
            }
            _ => (),
        }
    }

    let raw_bytes = if let Some(bytes) = audio_bytes {
        bytes
    } else if let Some(url) = hotlink_url {
        fetch_hotlink_audio(&url).await?
    } else if let Some(path_str) = filepath {
        tokio::fs::read(&path_str).await.map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                format!("Failed reading file at '{path_str}': {e}"),
            )
        })?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Provide 'file' upload, 'url' hotlink, or 'filepath' form field.".to_string(),
        ));
    };

    process_audio_pipeline(
        &state,
        raw_bytes,
        lang,
        artist,
        title,
        prompt,
        isolate,
        output_format,
        refresh,
    )
    .await
}

async fn handle_lyrics_query(
    State(state): State<AppState>,
    Query(query): Query<LyricsQuery>,
) -> Result<Response, (StatusCode, String)> {
    let lang = query.lang.unwrap_or_else(|| "ja".to_string());
    let output_format = query.format.unwrap_or_else(|| "lrc".to_string()).to_lowercase();
    let refresh = query.refresh.unwrap_or(false);
    let isolate = query.isolate.unwrap_or(true);

    let raw_bytes = if let Some(url) = query.url.or(query.hotlink) {
        fetch_hotlink_audio(&url).await?
    } else if let Some(path_str) = query.filepath {
        tokio::fs::read(&path_str).await.map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                format!("Failed reading file at '{path_str}': {e}"),
            )
        })?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Provide 'url' or 'filepath' query parameter.".to_string(),
        ));
    };

    process_audio_pipeline(
        &state,
        raw_bytes,
        lang,
        query.artist,
        query.title,
        query.prompt,
        isolate,
        output_format,
        refresh,
    )
    .await
}

async fn handle_lyrics_json(
    State(state): State<AppState>,
    Json(payload): Json<LyricsQuery>,
) -> Result<Response, (StatusCode, String)> {
    let lang = payload.lang.unwrap_or_else(|| "ja".to_string());
    let output_format = payload.format.unwrap_or_else(|| "lrc".to_string()).to_lowercase();
    let refresh = payload.refresh.unwrap_or(false);
    let isolate = payload.isolate.unwrap_or(true);

    let raw_bytes = if let Some(url) = payload.url.or(payload.hotlink) {
        fetch_hotlink_audio(&url).await?
    } else if let Some(path_str) = payload.filepath {
        tokio::fs::read(&path_str).await.map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                format!("Failed reading file at '{path_str}': {e}"),
            )
        })?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Provide 'url' or 'filepath' in JSON payload.".to_string(),
        ));
    };

    process_audio_pipeline(
        &state,
        raw_bytes,
        lang,
        payload.artist,
        payload.title,
        payload.prompt,
        isolate,
        output_format,
        refresh,
    )
    .await
}
