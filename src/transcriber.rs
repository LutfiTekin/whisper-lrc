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

/// Builds a Whisper `initial_prompt` from optional `artist`, `title`, and custom `prompt` fields.
pub fn build_initial_prompt(
    artist: Option<&str>,
    title: Option<&str>,
    custom_prompt: Option<&str>,
) -> Option<String> {
    let artist_clean = artist.map(str::trim).filter(|s| !s.is_empty());
    let title_clean = title.map(str::trim).filter(|s| !s.is_empty());
    let prompt_clean = custom_prompt.map(str::trim).filter(|s| !s.is_empty());

    let meta_part = match (artist_clean, title_clean) {
        (Some(a), Some(t)) => Some(format!("{a} - {t}")),
        (Some(a), None) => Some(a.to_string()),
        (None, Some(t)) => Some(t.to_string()),
        (None, None) => None,
    };

    let combined = match (meta_part, prompt_clean) {
        (Some(m), Some(p)) => Some(format!("{m}. {p}")),
        (Some(m), None) => Some(m),
        (None, Some(p)) => Some(p.to_string()),
        (None, None) => None,
    };

    combined.map(|s| s.replace('\0', ""))
}

/// Returns `true` if a transcribed segment matches known Whisper silence/outro hallucinations
/// or degenerate repetition loops.
pub fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Strip common punctuation, brackets, and music symbols
    let stripped: String = trimmed
        .chars()
        .filter(|c| {
            !c.is_whitespace()
                && !matches!(
                    c,
                    '.' | ',' | '!' | '?' | '。' | '、' | '！' | '？' | '…' | '・' | '〜' | '~'
                        | '-' | '—' | '(' | ')' | '（' | '）' | '[' | ']' | '「' | '」' | '『'
                        | '』' | '♪' | '♫' | '♬' | '"' | '\''
                )
        })
        .collect();

    if stripped.is_empty() {
        return true;
    }

    let lower = stripped.to_lowercase();

    // Exact match hallucinations (frequent during percussion/bell outros & silence)
    const EXACT_DENYLIST: &[&str] = &[
        "アーメン",
        "あーめん",
        "amen",
        "ハレルヤ",
        "hallelujah",
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "視聴ありがとうございました",
        "ありがとうございました",
        "チャンネル登録お願いします",
        "チャンネル登録よろしくお願いします",
        "チャンネル登録",
        "高評価",
        "字幕",
        "翻訳",
        "おやすみなさい",
    ];

    if EXACT_DENYLIST.contains(&lower.as_str()) {
        return true;
    }

    // Substring match for YouTube/subtitle watermark hallucinations
    const SUBSTRING_DENYLIST: &[&str] = &[
        "ご視聴ありがとう",
        "チャンネル登録",
        "subtitlesby",
        "translatedby",
        "amara.org",
        "thankyouforwatching",
        "thanksforwatching",
    ];

    if SUBSTRING_DENYLIST.iter().any(|pat| lower.contains(pat)) {
        return true;
    }

    // Detect single-character degenerate repetition (e.g. "ああああああああ")
    let mut prev_char = None;
    let mut run_len = 0usize;
    for ch in stripped.chars() {
        if Some(ch) == prev_char {
            run_len += 1;
            if run_len >= 8 {
                return true;
            }
        } else {
            prev_char = Some(ch);
            run_len = 1;
        }
    }

    false
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
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
        prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let ctx = self.get_or_load_context().await?;
        let mut state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {e:?}"))?;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
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
        params.set_logprob_thold(-1.0);
        params.set_entropy_thold(2.4);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);

        let clean_prompt = prompt
            .map(|p| p.replace('\0', "").trim().to_string())
            .filter(|p| !p.is_empty());

        if let Some(ref p) = clean_prompt {
            tracing::info!("Conditioning Whisper decoder with initial_prompt: \"{p}\"");
            params.set_initial_prompt(p);
        }

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
                .full_get_segment_text_lossy(i)
                .unwrap_or_default()
                .trim()
                .to_string();

            if text.is_empty() || text == last_text || is_hallucination(&text) {
                continue;
            }

            // Filter out low-confidence noise segments by checking average token probability
            if let Ok(n_tokens) = state.full_n_tokens(i) {
                if n_tokens > 0 {
                    let mut prob_sum = 0.0_f32;
                    let mut prob_count = 0usize;
                    for t in 0..n_tokens {
                        if let Ok(p) = state.full_get_token_prob(i, t) {
                            prob_sum += p;
                            prob_count += 1;
                        }
                    }
                    if prob_count > 0 {
                        let avg_prob = prob_sum / (prob_count as f32);
                        if avg_prob < 0.20 {
                            tracing::debug!(
                                "Dropping low-confidence segment (avg_prob={:.2}): {}",
                                avg_prob,
                                text
                            );
                            continue;
                        }
                    }
                }
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
            prompt: clean_prompt,
            segments,
        })
    }
}
