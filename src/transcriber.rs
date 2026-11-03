use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accuracy {
    Normal,
    High,
}

impl Default for Accuracy {
    fn default() -> Self {
        Accuracy::Normal
    }
}

impl std::str::FromStr for Accuracy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "normal" => Ok(Accuracy::Normal),
            "high" => Ok(Accuracy::High),
            other => anyhow::bail!("Invalid accuracy '{other}'. Allowed: 'normal', 'high'."),
        }
    }
}

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
    pub accuracy: Accuracy,
    pub lang: String,
    pub segments: Vec<LrcSegment>,
}

pub struct TranscriberService {
    model_dir: PathBuf,
    cpu_threads: i32,
    normal_context: Arc<Mutex<Option<Arc<WhisperContext>>>>,
    high_context: Arc<Mutex<Option<Arc<WhisperContext>>>>,
    last_used: Arc<Mutex<Instant>>,
}

impl TranscriberService {
    pub fn new(model_dir: PathBuf, cpu_threads: i32) -> Self {
        Self {
            model_dir,
            cpu_threads,
            normal_context: Arc::new(Mutex::new(None)),
            high_context: Arc::new(Mutex::new(None)),
            last_used: Arc::new(Mutex::new(Instant::now())),
        }
    }

    fn model_filename(&self, accuracy: Accuracy) -> &'static str {
        match accuracy {
            Accuracy::Normal => "ggml-medium.bin",
            Accuracy::High => "ggml-large-v3.bin",
        }
    }

    async fn ensure_model(&self, accuracy: Accuracy) -> Result<PathBuf> {
        let filename = self.model_filename(accuracy);
        let path = self.model_dir.join(filename);

        if path.exists() {
            return Ok(path);
        }

        tokio::fs::create_dir_all(&self.model_dir).await?;

        tracing::info!("Model {} not found. Downloading to {:?}...", filename, path);
        let url = format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            filename
        );

        let resp = reqwest::get(&url)
            .await
            .context(format!("Failed to request model from {url}"))?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed downloading model {filename}: HTTP {}", resp.status());
        }

        let bytes = resp.bytes().await?;
        tokio::fs::write(&path, &bytes).await?;
        tracing::info!("Model {} downloaded successfully.", filename);

        Ok(path)
    }

    async fn get_or_load_context(&self, accuracy: Accuracy) -> Result<Arc<WhisperContext>> {
        let mut guard = match accuracy {
            Accuracy::Normal => self.normal_context.lock().await,
            Accuracy::High => self.high_context.lock().await,
        };

        *self.last_used.lock().await = Instant::now();

        if let Some(ctx) = guard.as_ref() {
            return Ok(ctx.clone());
        }

        let model_path = self.ensure_model(accuracy).await?;
        let path_str = model_path.to_str().context("Invalid UTF-8 in path")?;

        tracing::info!("Loading {:?} into memory/VRAM...", accuracy);
        let ctx_params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed initializing Whisper context: {e:?}"))?;

        let arc_ctx = Arc::new(ctx);
        *guard = Some(arc_ctx.clone());
        Ok(arc_ctx)
    }

    /// Evicts loaded models from VRAM and RAM if idle time exceeds ttl_secs.
    pub async fn evict_if_idle(&self, ttl_secs: u64) -> bool {
        if ttl_secs == 0 {
            return false;
        }

        let last = *self.last_used.lock().await;
        if last.elapsed().as_secs() >= ttl_secs {
            let mut normal_guard = self.normal_context.lock().await;
            let mut high_guard = self.high_context.lock().await;

            if normal_guard.is_some() || high_guard.is_some() {
                tracing::info!(
                    "Idle TTL ({}s) reached. Evicting models from VRAM and RAM.",
                    ttl_secs
                );
                *normal_guard = None;
                *high_guard = None;
                return true;
            }
        }
        false
    }

    pub async fn transcribe(
        &self,
        pcm_16k_mono: &[f32],
        accuracy: Accuracy,
        lang: &str,
        cancel_token: Option<CancellationToken>,
    ) -> Result<TranscriptionResult> {
        let ctx = self.get_or_load_context(accuracy).await?;
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

        let lang_clean = lang.trim().to_lowercase();
        if lang_clean != "auto" && !lang_clean.is_empty() {
            params.set_language(Some(&lang_clean));
        } else {
            params.set_language(None);
        }

        // Cooperative cancellation check
        let is_aborted = Arc::new(AtomicBool::new(false));
        let abort_flag = is_aborted.clone();

        if let Some(token) = cancel_token {
            tokio::spawn(async move {
                token.cancelled().await;
                abort_flag.store(true, Ordering::SeqCst);
            });
        }

        let abort_checker = is_aborted.clone();
        params.set_abort_callback(Some(move |_| abort_checker.load(Ordering::SeqCst)));

        state
            .full(params, pcm_16k_mono)
            .map_err(|e| anyhow::anyhow!("Whisper inference error: {e:?}"))?;

        if is_aborted.load(Ordering::SeqCst) {
            anyhow::bail!("Inference was cancelled by preemption");
        }

        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed reading segment count: {e:?}"))?;

        let mut segments = Vec::new();
        let mut lrc_lines = Vec::new();

        for i in 0..num_segments {
            let start_sec = (state.full_get_segment_t0(i).unwrap_or(0) as f64) / 100.0;
            let end_sec = (state.full_get_segment_t1(i).unwrap_or(0) as f64) / 100.0;
            let text = state
                .full_get_segment_text(i)
                .unwrap_or_default()
                .trim()
                .to_string();

            if text.is_empty() {
                continue;
            }

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
            accuracy,
            lang: lang_clean,
            segments,
        })
    }
}
