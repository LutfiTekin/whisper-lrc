use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub const DEFAULT_MODEL_FILENAME: &str = "ggml-large-v3-q5_0.bin";

pub fn format_lrc_timestamp(seconds: f64) -> String {
    let s = if seconds < 0.0 { 0.0 } else { seconds };
    let mins = (s / 60.0) as u32;
    let remaining_secs = s % 60.0;
    format!("[{:02}:{:05.2}]", mins, remaining_secs)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrcSegment {
    pub start: f64,
    pub end: f64,
    pub timestamp: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub lrc: String,
    pub lang: String,
    pub segments: Vec<LrcSegment>,
}

pub struct TranscriberService {
    model_dir: PathBuf,
    cpu_threads: i32,
    context: Arc<Mutex<Option<Arc<WhisperContext>>>>,
    last_used: Arc<Mutex<Instant>>,
}

impl TranscriberService {
    pub fn new(model_dir: PathBuf, cpu_threads: i32) -> Self {
        Self {
            model_dir,
            cpu_threads,
            context: Arc::new(Mutex::new(None)),
            last_used: Arc::new(Mutex::new(Instant::now())),
        }
    }

    async fn ensure_model(&self) -> Result<PathBuf> {
        let path = self.model_dir.join(DEFAULT_MODEL_FILENAME);

        if path.exists() {
            return Ok(path);
        }

        tokio::fs::create_dir_all(&self.model_dir).await?;

        tracing::info!(
            "Model {} not found. Downloading to {:?}...",
            DEFAULT_MODEL_FILENAME,
            path
        );
        let url = format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            DEFAULT_MODEL_FILENAME
        );

        let resp = reqwest::get(&url)
            .await
            .context(format!("Failed to request model from {url}"))?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "Failed downloading model {DEFAULT_MODEL_FILENAME}: HTTP {}",
                resp.status()
            );
        }

        let bytes = resp.bytes().await?;
        tokio::fs::write(&path, &bytes).await?;
        tracing::info!("Model {} downloaded successfully.", DEFAULT_MODEL_FILENAME);

        Ok(path)
    }

    async fn get_or_load_context(&self) -> Result<Arc<WhisperContext>> {
        let mut guard = self.context.lock().await;

        *self.last_used.lock().await = Instant::now();

        if let Some(ctx) = guard.as_ref() {
            return Ok(ctx.clone());
        }

        let model_path = self.ensure_model().await?;
        let path_str = model_path.to_str().context("Invalid UTF-8 in path")?;

        tracing::info!("Loading large-v3-q5_0 into GPU VRAM (CUDA)...");
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed initializing Whisper context: {e:?}"))?;

        let arc_ctx = Arc::new(ctx);
        *guard = Some(arc_ctx.clone());
        Ok(arc_ctx)
    }

    /// Evicts loaded model from VRAM if idle time exceeds ttl_secs.
    pub async fn evict_if_idle(&self, ttl_secs: u64) -> bool {
        if ttl_secs == 0 {
            return false;
        }

        let last = *self.last_used.lock().await;
        if last.elapsed().as_secs() >= ttl_secs {
            let mut guard = self.context.lock().await;

            if guard.is_some() {
                tracing::info!(
                    "Idle TTL ({}s) reached. Evicting model from GPU VRAM.",
                    ttl_secs
                );
                *guard = None;
                return true;
            }
        }
        false
    }

    pub async fn transcribe(
        &self,
        pcm_16k_mono: &[f32],
        lang: &str,
    ) -> Result<TranscriptionResult> {
        let ctx = self.get_or_load_context().await?;
        let mut state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {e:?}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.cpu_threads);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_no_speech_thold(0.6);

        let lang_clean = lang.trim().to_lowercase();
        if lang_clean != "auto" && !lang_clean.is_empty() {
            params.set_language(Some(&lang_clean));
        } else {
            params.set_language(None);
        }

        state
            .full(params, pcm_16k_mono)
            .map_err(|e| anyhow::anyhow!("Whisper inference error: {e:?}"))?;

        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed reading segment count: {e:?}"))?;

        let mut segments = Vec::new();
        let mut lrc_lines = Vec::new();
        let mut last_text = String::new();

        for i in 0..num_segments {
            let start_sec = (state.full_get_segment_t0(i).unwrap_or(0) as f64) / 100.0;
            let end_sec = (state.full_get_segment_t1(i).unwrap_or(0) as f64) / 100.0;
            let text = state
                .full_get_segment_text(i)
                .unwrap_or_default()
                .trim()
                .to_string();

            if text.is_empty() || text == last_text {
                continue;
            }
            last_text = text.clone();

            let ts = format_lrc_timestamp(start_sec);
            lrc_lines.push(format!("{ts}{text}"));

            segments.push(LrcSegment {
                start: (start_sec * 100.0).round() / 100.0,
                end: (end_sec * 100.0).round() / 100.0,
                timestamp: ts,
                text,
            });
        }

        let lrc_output = if lrc_lines.is_empty() {
            String::new()
        } else {
            format!("{}\n", lrc_lines.join("\n"))
        };

        *self.last_used.lock().await = Instant::now();

        Ok(TranscriptionResult {
            lrc: lrc_output,
            lang: lang_clean,
            segments,
        })
    }
}
