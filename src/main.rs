use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use whisper_lrc::audio::decode_audio_to_16k_mono;
use whisper_lrc::cache::{CacheKey, LrcCache};
use whisper_lrc::queue::{start_worker_loop, PreemptiveQueue};
use whisper_lrc::transcriber::{Accuracy, TranscriberService};

#[derive(Clone)]
struct AppState {
    queue: Arc<PreemptiveQueue>,
    cache: LrcCache,
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

    let model_dir = PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| "/models".to_string()));
    let cpu_threads: i32 = std::env::var("CPU_THREADS")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap_or(8);

    let idle_ttl_secs: u64 = std::env::var("IDLE_TIMEOUT_SECS")
        .unwrap_or_else(|_| "600".to_string())
        .parse()
        .unwrap_or(600);

    let max_queue_capacity: usize = std::env::var("MAX_QUEUE_CAPACITY")
        .unwrap_or_else(|_| "100".to_string())
        .parse()
        .unwrap_or(100);

    let cache_capacity: u64 = std::env::var("CACHE_CAPACITY")
        .unwrap_or_else(|_| "5000".to_string())
        .parse()
        .unwrap_or(5000);

    let transcriber = Arc::new(TranscriberService::new(model_dir, cpu_threads));
    let queue = Arc::new(PreemptiveQueue::new(max_queue_capacity));
    let cache = LrcCache::new(cache_capacity, 0); // 0 = no expiration until evicted by LRU

    // Start background single-concurrency worker loop
    start_worker_loop(queue.clone(), transcriber, idle_ttl_secs);

    let state = AppState { queue, cache };

    let app = Router::new()
        .route("/lyrics", post(handle_lyrics))
        .route("/queue", get(handle_queue))
        .route("/health", get(handle_health))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(axum::extract::DefaultBodyLimit::max(250 * 1024 * 1024))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Whisper LRC Rust microservice listening on http://{addr}");
    tracing::info!("Preemptive queue active. Idle VRAM eviction TTL: {}s", idle_ttl_secs);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_health() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "whisper-lrc",
        "runtime": "rust-axum"
    }))
}

async fn handle_queue(State(state): State<AppState>) -> impl IntoResponse {
    let q_status = state.queue.status().await;
    Json(json!({
        "queue": q_status,
        "cached_tracks": state.cache.entry_count()
    }))
}

async fn handle_lyrics(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut filepath: Option<String> = None;
    let mut accuracy = Accuracy::Normal;
    let mut lang = "ja".to_string();
    let mut output_format = "lrc".to_string();
    let mut refresh = false;

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
            "filepath" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid filepath field: {e}")))?;
                filepath = Some(text.trim().to_string());
            }
            "accuracy" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid accuracy field: {e}")))?;
                if let Ok(acc) = Accuracy::from_str(&text) {
                    accuracy = acc;
                }
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
            "Provide either 'file' upload or 'filepath' form field.".to_string(),
        ));
    };

    // Calculate SHA-256 hash for in-memory LRU cache
    let mut hasher = Sha256::new();
    hasher.update(&raw_bytes);
    let audio_hash = hex::encode(hasher.finalize());

    let cache_key = CacheKey {
        audio_hash: audio_hash.clone(),
        accuracy,
        lang: lang.clone(),
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
                        "accuracy": accuracy,
                        "lang": lang,
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

    // 2. Decode audio to 16kHz mono PCM & get duration
    let decoded = tokio::task::spawn_blocking(move || decode_audio_to_16k_mono(&raw_bytes))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Thread error: {e}")))?
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Audio decoding error: {e}")))?;

    // 3. Submit to Preemptive Priority Queue
    let rx = state
        .queue
        .submit(decoded.pcm, decoded.duration_secs, accuracy, lang.clone())
        .await
        .map_err(|e| (StatusCode::TOO_MANY_REQUESTS, e))?;

    // 4. Await worker completion
    let result = rx
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Worker dropped job channel".to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 5. Store in LRU cache
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
