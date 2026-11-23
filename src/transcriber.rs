use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

use crate::alignment::{
    align_lyrics_to_words, extract_lyrics_prompt_hint, format_lrc_with_breaks, BreakConfig,
    TimedWord, VocalEnergyProfile,
};

pub const DEFAULT_MODEL_FILENAME: &str = "ggml-large-v3-turbo.bin";

pub fn get_model_filename() -> String {
    std::env::var("WHISPER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_FILENAME.to_string())
}

pub fn is_dtw_enabled() -> bool {
    std::env::var("WHISPER_DTW")
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "false" | "0" | "no" | "off"))
        .unwrap_or(true)
}

pub fn infer_dtw_mode(model_filename: &str) -> DtwMode<'static> {
    let lower = model_filename.trim().to_ascii_lowercase();
    if lower.contains("large-v3-turbo") || lower.contains("turbo") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::LargeV3Turbo,
        }
    } else if lower.contains("large-v3") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::LargeV3,
        }
    } else if lower.contains("large-v2") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::LargeV2,
        }
    } else if lower.contains("large-v1") || lower.contains("large") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::LargeV1,
        }
    } else if lower.contains("medium.en") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::MediumEn,
        }
    } else if lower.contains("medium") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Medium,
        }
    } else if lower.contains("small.en") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::SmallEn,
        }
    } else if lower.contains("small") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Small,
        }
    } else if lower.contains("base.en") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::BaseEn,
        }
    } else if lower.contains("base") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Base,
        }
    } else if lower.contains("tiny.en") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::TinyEn,
        }
    } else if lower.contains("tiny") {
        DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Tiny,
        }
    } else {
        DtwMode::TopMost { n_top: 4 }
    }
}

pub fn format_lrc_timestamp(seconds: f64) -> String {
    let s = if seconds.is_finite() && seconds > 0.0 {
        seconds
    } else {
        0.0
    };
    let total_centis = (s * 100.0).round() as u64;
    let mins = total_centis / 6000;
    let secs_centis = total_centis % 6000;
    format!("[{:02}:{:02}.{:02}]", mins, secs_centis / 100, secs_centis % 100)
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
        "altyazı",
        "altyazi",
        "seslibetimleme",
        "seslendiren",
        "metinyazarı",
        "metinyazari",
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
        "seslibetimleme",
        "seslendiren",
        "altyazı",
        "altyazi",
        "metinyazarı",
        "metinyazari",
        "derneğineyaptırılmıştır",
        "dernegineyaptirilmistir",
        "seslibetimlemedernegi",
        "seslibetimlemederneğine",
        "seslibetimlemedernegine",
        "aboneol",
        "beğenmeyiunutmayın",
        "begenmeyiunutmayin",
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

    // Detect word and n-gram repetition loops inside a segment (e.g. "Alou Alou..." or "of the city of the city...")
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 6 {
        let mut repeat_count = 1;
        for i in 1..words.len() {
            if words[i].eq_ignore_ascii_case(words[i - 1]) {
                repeat_count += 1;
                if repeat_count >= 5 {
                    return true;
                }
            } else {
                repeat_count = 1;
            }
        }

        for window_size in 2..=4 {
            if words.len() >= window_size * 4 {
                let mut loop_count = 1;
                for i in (window_size..=words.len() - window_size).step_by(window_size) {
                    let prev = &words[i - window_size..i];
                    let curr = &words[i..i + window_size];
                    if prev == curr {
                        loop_count += 1;
                        if loop_count >= 4 {
                            return true;
                        }
                    } else {
                        loop_count = 1;
                    }
                }
            }
        }
    }

    false
}

/// Checks whether an `initial_prompt` contains characters compatible with a non-Latin target language.
/// Prevents Latin English metadata from forcing non-Latin Whisper decoders (Arabic, Japanese, Cyrillic, etc.)
/// into broken transliteration or English repetition loops.
pub fn is_prompt_compatible_with_lang(prompt: &str, lang: &str) -> bool {
    let lang = lang.trim().to_lowercase();
    match lang.as_str() {
        "ar" | "fa" | "ur" => prompt.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)),
        "ja" => prompt.chars().any(|c| {
            ('\u{3040}'..='\u{309F}').contains(&c)
                || ('\u{30A0}'..='\u{30FF}').contains(&c)
                || ('\u{4E00}'..='\u{9FFF}').contains(&c)
        }),
        "zh" | "yue" => prompt.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
        "ko" => prompt.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c)),
        "ru" | "uk" | "be" | "bg" | "mk" | "sr" => {
            prompt.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c))
        }
        "el" => prompt.chars().any(|c| ('\u{0370}'..='\u{03FF}').contains(&c)),
        "he" | "yi" => prompt.chars().any(|c| ('\u{0590}'..='\u{05FF}').contains(&c)),
        "th" => prompt.chars().any(|c| ('\u{0E00}'..='\u{0E7F}').contains(&c)),
        "hi" | "mr" | "ne" => prompt.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c)),
        _ => true,
    }
}

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub lang: String,
    pub prompt: Option<String>,
    pub reference_lyrics: Option<String>,
    pub isolated: bool,
    pub breaks: BreakConfig,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            lang: "ja".to_string(),
            prompt: None,
            reference_lyrics: None,
            isolated: false,
            breaks: BreakConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrcSegment {
    pub start: f64,
    pub end: f64,
    pub timestamp: String,
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub words: Vec<TimedWord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub lrc: String,
    pub lang: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub segments: Vec<LrcSegment>,
    #[serde(default)]
    pub aligned: bool,
    #[serde(default)]
    pub breaks_inserted: usize,
}

#[derive(Debug, Clone)]
struct KeptSegmentMeta {
    norm_text: String,
    start_sec: f64,
}

fn round_centis(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn is_unspaced_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x309F
            | 0x30A0..=0x30FF
            | 0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0xAC00..=0xD7AF
            | 0x0E00..=0x0E7F
    )
}

fn is_hallucination_loop(
    history: &[KeptSegmentMeta],
    text: &str,
    start_sec: f64,
    end_sec: f64,
    avg_prob: f32,
) -> bool {
    let norm = text.trim().to_lowercase();
    if norm.is_empty() {
        return true;
    }

    if let Some(prev) = history.last() {
        if prev.norm_text == norm {
            if (start_sec - prev.start_sec) < 0.40 || (end_sec - start_sec) < 0.20 {
                return true;
            }
            let consecutive_same = history
                .iter()
                .rev()
                .take_while(|h| h.norm_text == norm)
                .count();
            if consecutive_same >= 3 && avg_prob < 0.50 {
                return true;
            }
        }
    }

    if history.len() >= 5 && avg_prob < 0.50 {
        let tail = &history[history.len() - 5..];
        let a = &tail[0].norm_text;
        let b = &tail[1].norm_text;
        if a != b
            && tail[2].norm_text == *a
            && tail[3].norm_text == *b
            && tail[4].norm_text == *a
            && norm == *b
        {
            return true;
        }
    }

    false
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
        let model_filename = get_model_filename();
        let path = self.model_dir.join(&model_filename);

        if path.exists() {
            return Ok(path);
        }

        tokio::fs::create_dir_all(&self.model_dir).await?;

        tracing::info!(
            "Model {} not found. Downloading to {:?}...",
            model_filename,
            path
        );
        let url = format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            model_filename
        );

        let resp = reqwest::get(&url)
            .await
            .context(format!("Failed to request model from {url}"))?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "Failed downloading model {model_filename}: HTTP {}",
                resp.status()
            );
        }

        let bytes = resp.bytes().await?;
        tokio::fs::write(&path, &bytes).await?;
        tracing::info!("Model {} downloaded successfully.", model_filename);

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
        let model_filename = get_model_filename();

        tracing::info!("Loading Whisper model '{model_filename}' into GPU VRAM (CUDA)...");
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);
        ctx_params.flash_attn(false);

        if is_dtw_enabled() {
            let dtw_mode = infer_dtw_mode(&model_filename);
            tracing::info!("Enabling Whisper DTW token timestamps: {:?}", dtw_mode);
            ctx_params.dtw_parameters(DtwParameters {
                mode: dtw_mode,
                dtw_mem_size: 1024 * 1024 * 128,
            });
        }

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
        let options = TranscribeOptions {
            lang: lang.to_string(),
            prompt: prompt.map(str::to_string),
            ..TranscribeOptions::default()
        };
        let duration_secs = (pcm_16k_mono.len() as f64) / 16000.0;
        self.transcribe_with_options(pcm_16k_mono, duration_secs, &options)
            .await
    }

    pub async fn transcribe_with_options(
        &self,
        pcm_16k_mono: &[f32],
        duration_secs: f64,
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult> {
        let ctx = self.get_or_load_context().await?;
        let eot_id = ctx.token_eot();
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
        params.set_token_timestamps(true);
        params.set_split_on_word(true);
        params.set_no_speech_thold(0.6);
        params.set_logprob_thold(-1.0);
        params.set_entropy_thold(2.4);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);

        let lang_clean = options.lang.trim().to_lowercase();
        if lang_clean != "auto" && !lang_clean.is_empty() {
            params.set_language(Some(&lang_clean));
        } else {
            params.set_language(None);
        }

        let clean_prompt = options
            .prompt
            .as_deref()
            .map(|p| p.replace('\0', "").trim().to_string())
            .filter(|p| !p.is_empty())
            .or_else(|| {
                options
                    .reference_lyrics
                    .as_deref()
                    .and_then(|lyr| extract_lyrics_prompt_hint(lyr, 180))
            });

        if let Some(ref p) = clean_prompt {
            if is_prompt_compatible_with_lang(p, &lang_clean) {
                tracing::info!("Conditioning Whisper decoder with initial_prompt: \"{p}\"");
                params.set_initial_prompt(p);
            } else {
                tracing::info!(
                    "Skipping initial_prompt \"{p}\" due to script mismatch with target language '{lang_clean}'"
                );
            }
        }

        let t0 = Instant::now();
        state
            .full(params, pcm_16k_mono)
            .map_err(|e| anyhow::anyhow!("Whisper inference error: {e:?}"))?;
        let inference_dur = t0.elapsed();

        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed reading segment count: {e:?}"))?;

        tracing::info!(
            "Whisper inference finished in {:.2?} (raw segments: {})",
            inference_dur,
            num_segments
        );

        let effective_duration = if duration_secs > 0.0 {
            duration_secs
        } else {
            (pcm_16k_mono.len() as f64) / 16000.0
        };
        let energy = VocalEnergyProfile::from_pcm(pcm_16k_mono, 16000);

        let mut segments = Vec::new();
        let mut all_words = Vec::new();
        let mut history: Vec<KeptSegmentMeta> = Vec::new();

        for i in 0..num_segments {
            let seg_start = (state.full_get_segment_t0(i).unwrap_or(0) as f64) / 100.0;
            let seg_end = ((state.full_get_segment_t1(i).unwrap_or(0) as f64) / 100.0)
                .max(seg_start);
            let text = state
                .full_get_segment_text_lossy(i)
                .unwrap_or_default()
                .trim()
                .to_string();

            if text.is_empty() || is_hallucination(&text) {
                continue;
            }

            let mut seg_words: Vec<TimedWord> = Vec::new();
            let mut prob_sum = 0.0_f32;
            let mut prob_count = 0usize;

            if let Ok(n_tokens) = state.full_n_tokens(i) {
                for t in 0..n_tokens {
                    let Ok(data) = state.full_get_token_data(i, t) else {
                        continue;
                    };
                    if data.id >= eot_id {
                        continue;
                    }
                    let raw_tok = state.full_get_token_text_lossy(i, t).unwrap_or_default();
                    let tok_trimmed = raw_tok.trim();
                    if tok_trimmed.is_empty()
                        || tok_trimmed.starts_with("[_")
                        || tok_trimmed.starts_with("<|")
                    {
                        continue;
                    }

                    prob_sum += data.p;
                    prob_count += 1;

                    let mut tok_start = (data.t0 as f64 / 100.0).max(seg_start);
                    let mut tok_end = (data.t1 as f64 / 100.0).min(seg_end).max(tok_start);

                    if data.t_dtw >= 0 {
                        let dtw_sec = (data.t_dtw as f64) / 100.0;
                        if dtw_sec >= seg_start - 0.25 && dtw_sec <= seg_end + 0.25 {
                            if dtw_sec - tok_start > 0.60 {
                                tok_start = (dtw_sec - 0.15).max(tok_start);
                            }
                            if tok_end - dtw_sec > 0.80 {
                                tok_end = (dtw_sec + 0.35).min(tok_end);
                            }
                            if tok_end <= tok_start {
                                tok_end = (dtw_sec + 0.15).min(seg_end).max(tok_start + 0.05);
                            }
                        }
                    }

                    let starts_new_word = seg_words.is_empty()
                        || raw_tok.starts_with(|c: char| c.is_whitespace())
                        || tok_trimmed.chars().next().is_some_and(is_unspaced_char)
                        || seg_words
                            .last()
                            .and_then(|w| w.text.chars().last())
                            .is_some_and(is_unspaced_char);

                    if starts_new_word {
                        seg_words.push(TimedWord {
                            start: tok_start,
                            end: tok_end,
                            text: tok_trimmed.to_string(),
                            prob: data.p,
                        });
                    } else if let Some(prev_word) = seg_words.last_mut() {
                        prev_word.text.push_str(tok_trimmed);
                        prev_word.end = prev_word.end.max(tok_end);
                        prev_word.prob = (prev_word.prob + data.p) * 0.5;
                    }
                }
            }

            let avg_prob = if prob_count > 0 {
                prob_sum / (prob_count as f32)
            } else {
                1.0
            };

            if prob_count > 0 && avg_prob < 0.20 {
                tracing::debug!(
                    "Dropping low-confidence segment (avg_prob={:.2}): {}",
                    avg_prob,
                    text
                );
                continue;
            }

            let raw_start = seg_words.first().map(|w| w.start).unwrap_or(seg_start);
            let raw_end = seg_words
                .last()
                .map(|w| w.end)
                .unwrap_or(seg_end)
                .max(raw_start);

            let (refined_start, refined_end) =
                energy.refine_bounds(raw_start, raw_end, options.isolated);

            if energy.peak_db_in_range(refined_start, refined_end) < -52.0 {
                tracing::debug!(
                    "Dropping silent/non-vocal phantom segment at {:.2}s: {}",
                    refined_start,
                    text
                );
                continue;
            }

            if is_hallucination_loop(&history, &text, refined_start, refined_end, avg_prob) {
                tracing::debug!(
                    "Dropping hallucination loop segment (avg_prob={:.2}): {}",
                    avg_prob,
                    text
                );
                continue;
            }

            history.push(KeptSegmentMeta {
                norm_text: text.trim().to_lowercase(),
                start_sec: refined_start,
            });
            if history.len() > 8 {
                history.remove(0);
            }

            for w in &mut seg_words {
                let ws = w.start.max(refined_start).min(refined_end);
                let we = w.end.min(refined_end).max(ws);
                w.start = round_centis(ws);
                w.end = round_centis(we);
            }
            all_words.extend(seg_words.iter().cloned());

            let start = round_centis(refined_start);
            let end = round_centis(refined_end).max(start);
            let ts = format_lrc_timestamp(start);

            segments.push(LrcSegment {
                start,
                end,
                timestamp: ts,
                text,
                words: seg_words,
            });
        }

        let mut aligned = false;
        let mut final_segments = segments;
        if let Some(ref raw_lyrics) = options.reference_lyrics {
            if !raw_lyrics.trim().is_empty() {
                let aligned_segments = align_lyrics_to_words(
                    raw_lyrics,
                    &all_words,
                    &energy,
                    effective_duration,
                    options.isolated,
                );
                if !aligned_segments.is_empty() {
                    final_segments = aligned_segments;
                    aligned = true;
                }
            }
        }

        let (lrc_output, breaks_inserted) =
            format_lrc_with_breaks(&final_segments, effective_duration, &options.breaks);

        *self.last_used.lock().await = Instant::now();

        Ok(TranscriptionResult {
            lrc: lrc_output,
            lang: lang_clean,
            prompt: clean_prompt,
            segments: final_segments,
            aligned,
            breaks_inserted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_lrc_timestamp_no_60s_overflow() {
        assert_eq!(format_lrc_timestamp(59.996), "[01:00.00]");
        assert_eq!(format_lrc_timestamp(0.0), "[00:00.00]");
        assert_eq!(format_lrc_timestamp(151.35), "[02:31.35]");
    }

    #[test]
    fn test_infer_dtw_mode_presets() {
        assert!(matches!(
            infer_dtw_mode("ggml-large-v3-turbo.bin"),
            DtwMode::ModelPreset {
                model_preset: DtwModelPreset::LargeV3Turbo
            }
        ));
        assert!(matches!(
            infer_dtw_mode("ggml-large-v3.bin"),
            DtwMode::ModelPreset {
                model_preset: DtwModelPreset::LargeV3
            }
        ));
        assert!(matches!(
            infer_dtw_mode("ggml-medium.en.bin"),
            DtwMode::ModelPreset {
                model_preset: DtwModelPreset::MediumEn
            }
        ));
        assert!(matches!(
            infer_dtw_mode("custom-finetune.bin"),
            DtwMode::TopMost { n_top: 4 }
        ));
    }

    #[test]
    fn test_music_safe_repetition_keeps_chorus_drops_hallucination_loop() {
        let mut history = Vec::new();
        for (idx, t) in [10.0, 13.0, 16.0, 19.0].iter().enumerate() {
            let line = if idx % 2 == 0 {
                "The boys wanna be her"
            } else {
                "The girls wanna be her"
            };
            assert!(!is_hallucination_loop(&history, line, *t, t + 1.8, 0.85));
            history.push(KeptSegmentMeta {
                norm_text: line.to_lowercase(),
                start_sec: *t,
            });
        }

        // High-confidence hook repeating 4th time is kept
        assert!(!is_hallucination_loop(
            &history,
            "Turn around, bright eyes",
            22.0,
            24.0,
            0.88
        ));

        // Rapid collapsed duplicate (<0.40s gap) is dropped
        let collapsed_hist = vec![KeptSegmentMeta {
            norm_text: "turn around".to_string(),
            start_sec: 30.0,
        }];
        assert!(is_hallucination_loop(
            &collapsed_hist,
            "Turn around",
            30.2,
            30.8,
            0.90
        ));

        // Low-confidence alternating A-B-A-B-A-B loop is dropped
        let loop_hist = vec![
            KeptSegmentMeta { norm_text: "you".into(), start_sec: 200.0 },
            KeptSegmentMeta { norm_text: "me".into(), start_sec: 201.0 },
            KeptSegmentMeta { norm_text: "you".into(), start_sec: 202.0 },
            KeptSegmentMeta { norm_text: "me".into(), start_sec: 203.0 },
            KeptSegmentMeta { norm_text: "you".into(), start_sec: 204.0 },
        ];
        assert!(is_hallucination_loop(&loop_hist, "Me", 205.0, 206.0, 0.30));
        // Same alternating pattern with high singing confidence is preserved
        assert!(!is_hallucination_loop(&loop_hist, "Me", 205.0, 206.0, 0.85));
    }
}
