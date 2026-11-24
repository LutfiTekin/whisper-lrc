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
    align_lyrics_to_words, format_lrc_with_breaks, BreakConfig, TimedWord, VocalEnergyProfile,
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
        "musicplaying",
        "captioning",
        "subtitles",
        "music",
        "yourewelcome",
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
        "wellberightback",
        "staytuned",
        "commercialbreak",
        "guitarsolo",
        "pianosolo",
        "drumsolo",
        "instrumentalbreak",
        "songlyrics",
    ];

    if SUBSTRING_DENYLIST.iter().any(|pat| lower.contains(pat)) {
        return true;
    }

    // Drop pure non-lexical vocalise loops (e.g. "Ooh, ooh, ooh, ooh") during instrumental fills
    let ws_words: Vec<&str> = trimmed.split_whitespace().collect();
    if ws_words.len() >= 3 {
        let mut first_vocalise: Option<String> = None;
        let mut all_same_vocalise = true;
        for w in &ws_words {
            let clean: String = w
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !matches!(
                clean.as_str(),
                "ooh" | "oo" | "uh" | "um" | "hm" | "hmm" | "mm" | "mmm" | "ah" | "aah" | "ahh"
            ) {
                all_same_vocalise = false;
                break;
            }
            match &first_vocalise {
                None => first_vocalise = Some(clean),
                Some(first) if *first != clean => {
                    all_same_vocalise = false;
                    break;
                }
                _ => {}
            }
        }
        if all_same_vocalise {
            return true;
        }
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

fn is_pronoun_i(word: &str) -> bool {
    let trimmed = word.trim_matches(|c: char| !c.is_alphabetic() && c != '\'' && c != '’');
    trimmed == "I" || trimmed.starts_with("I'") || trimmed.starts_with("I’")
}

fn is_trailing_connector(word: &str) -> bool {
    let clean: String = word
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’')
        .to_ascii_lowercase();
    matches!(
        clean.as_str(),
        "the"
            | "a"
            | "an"
            | "of"
            | "in"
            | "to"
            | "my"
            | "your"
            | "our"
            | "his"
            | "her"
            | "and"
            | "or"
            | "but"
            | "i"
            | "i'm"
            | "i’m"
            | "bit"
            | "gonna"
            | "can"
            | "we'll"
            | "we’ll"
            | "don't"
            | "don’t"
            | "fall"
    )
}

fn is_lone_filler_word(text: &str) -> bool {
    let mut parts = text.split_whitespace();
    let Some(first) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    let clean: String = first
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();
    matches!(
        clean.as_str(),
        "the" | "a" | "an" | "and" | "or" | "of" | "to" | "in" | "it" | "is" | "so" | "unknown"
    )
}

fn join_words_text(words: &[TimedWord]) -> String {
    let mut out = String::new();
    for w in words {
        let t = w.text.trim();
        if t.is_empty() {
            continue;
        }
        if !out.is_empty() {
            let prev_unspaced = out.chars().last().is_some_and(is_unspaced_char);
            let next_unspaced = t.chars().next().is_some_and(is_unspaced_char);
            if !(prev_unspaced && next_unspaced) {
                out.push(' ');
            }
        }
        out.push_str(t);
    }
    out
}

/// Splits multi-line Whisper segments into individual lyric lines at word-level pauses
/// or capitalized phrase starts so choruses and verses don't collapse into 20s blocks.
fn split_segment_by_word_pauses(words: &[TimedWord]) -> Vec<Vec<TimedWord>> {
    if words.len() <= 1 {
        return if words.is_empty() {
            Vec::new()
        } else {
            vec![words.to_vec()]
        };
    }

    let mut groups = Vec::new();
    let mut start_idx = 0usize;
    let mut cur_chars = words[0].text.chars().count();

    for idx in 1..words.len() {
        let prev = &words[idx - 1];
        let next = &words[idx];
        let word_count = idx - start_idx;
        let remaining_words = words.len() - idx;
        let gap = (next.start - prev.end).max(0.0);

        let prev_ends_comma = prev.text.trim_end().ends_with(',');
        let next_starts_upper = next
            .text
            .chars()
            .next()
            .is_some_and(|c| c.is_uppercase())
            && !is_pronoun_i(&next.text);

        let pause_split = (gap >= 1.50)
            || (gap >= 0.85
                && word_count >= 2
                && remaining_words >= 2
                && !(is_trailing_connector(&prev.text) && !next_starts_upper))
            || (gap >= 0.55 && word_count >= 2 && remaining_words >= 2 && next_starts_upper);
        let phrase_split = cur_chars >= 22
            && word_count >= 3
            && remaining_words >= 2
            && !prev_ends_comma
            && next_starts_upper;

        if pause_split || phrase_split {
            groups.push(words[start_idx..idx].to_vec());
            start_idx = idx;
            cur_chars = next.text.chars().count();
        } else {
            cur_chars += 1 + next.text.chars().count();
        }
    }

    groups.push(words[start_idx..].to_vec());
    groups
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
            if consecutive_same >= 2 && (start_sec - prev.start_sec) <= 1.60 {
                return true;
            }
            if consecutive_same >= 2
                && norm.split_whitespace().count() <= 2
                && (start_sec - prev.start_sec) <= 4.50
            {
                return true;
            }
            if consecutive_same >= 4
                && (start_sec - prev.start_sec) <= 4.50
                && avg_prob < 0.85
            {
                return true;
            }
            if consecutive_same >= 3 && (avg_prob < 0.50 || (start_sec - prev.start_sec) <= 2.50) {
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

        let lang_clean = options.lang.trim().to_lowercase();
        let effective_duration = if duration_secs > 0.0 {
            duration_secs
        } else {
            (pcm_16k_mono.len() as f64) / 16000.0
        };
        let energy = VocalEnergyProfile::from_pcm(pcm_16k_mono, 16000);

        let mut chunks = energy.plan_audio_chunks(22.0, 29.0);
        if chunks.is_empty() && effective_duration > 0.05 {
            chunks.push((0.0, effective_duration));
        }

        let explicit_prompt = options
            .prompt
            .as_deref()
            .map(|p| p.replace('\0', "").trim().to_string())
            .filter(|p| !p.is_empty());

        let clean_prompt = explicit_prompt.clone();

        tracing::info!(
            "Whisper inference across {} energy-guided chunk(s) (total {:.1}s)...",
            chunks.len(),
            effective_duration
        );

        let t0 = Instant::now();
        let mut total_raw_segments: i32 = 0;
        let mut segments = Vec::new();
        let mut all_words = Vec::new();
        let mut history: Vec<KeptSegmentMeta> = Vec::new();

        for (chunk_idx, &(chunk_start, chunk_end)) in chunks.iter().enumerate() {
            let s_idx = ((chunk_start * 16000.0).round() as usize).min(pcm_16k_mono.len());
            let e_idx = ((chunk_end * 16000.0).round() as usize).clamp(s_idx, pcm_16k_mono.len());
            let raw_chunk_pcm = &pcm_16k_mono[s_idx..e_idx];
            if raw_chunk_pcm.len() < 1600 {
                continue;
            }

            let chunk_vocal_peak_db = energy.peak_db_in_range(chunk_start, chunk_end);
            let raw_peak = raw_chunk_pcm
                .iter()
                .fold(0.0_f32, |m, &x| m.max(x.abs()));
            let boosted_pcm: Option<Vec<f32>> =
                if chunk_vocal_peak_db >= -38.0 && raw_peak > 1e-4 && raw_peak < 0.35 {
                    let gain = (0.70 / raw_peak).min(5.6);
                    Some(
                        raw_chunk_pcm
                            .iter()
                            .map(|&x| (x * gain).clamp(-1.0, 1.0))
                            .collect(),
                    )
                } else {
                    None
                };
            let decode_pcm: &[f32] = boosted_pcm.as_deref().unwrap_or(raw_chunk_pcm);

            let chunk_prompt = explicit_prompt.as_deref();

            let allow_retry = chunk_end - chunk_start >= 8.0 && chunk_vocal_peak_db >= -32.0;
            let attempts: &[(f32, f32)] = if allow_retry {
                &[(0.0, 0.82), (0.2, 0.90)]
            } else {
                &[(0.0, 0.82)]
            };

            for &(temp, no_speech_thold) in attempts {
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
                params.set_max_tokens(180);
                params.set_no_speech_thold(no_speech_thold);
                params.set_logprob_thold(-1.3);
                params.set_entropy_thold(2.6);
                params.set_temperature(temp);
                params.set_temperature_inc(0.0);

                if lang_clean != "auto" && !lang_clean.is_empty() {
                    params.set_language(Some(&lang_clean));
                } else {
                    params.set_language(None);
                }

                if let Some(p) = chunk_prompt {
                    if is_prompt_compatible_with_lang(p, &lang_clean) {
                        if chunk_idx == 0 && temp == 0.0 {
                            tracing::info!("Conditioning Whisper decoder with initial_prompt: \"{p}\"");
                        }
                        params.set_initial_prompt(p);
                    } else if chunk_idx == 0 && temp == 0.0 {
                        tracing::info!(
                            "Skipping initial_prompt \"{p}\" due to script mismatch with target language '{lang_clean}'"
                        );
                    }
                }

                state
                    .full(params, decode_pcm)
                    .map_err(|e| anyhow::anyhow!("Whisper inference error: {e:?}"))?;

                let num_segments = state
                    .full_n_segments()
                    .map_err(|e| anyhow::anyhow!("Failed reading segment count: {e:?}"))?;
                total_raw_segments += num_segments;

                let mut attempt_history = history.clone();
                let mut chunk_segments = Vec::new();
                let mut chunk_words = Vec::new();

                for i in 0..num_segments {
                    let seg_start = (chunk_start
                        + (state.full_get_segment_t0(i).unwrap_or(0) as f64) / 100.0)
                        .clamp(chunk_start, chunk_end);
                    let seg_end = (chunk_start
                        + (state.full_get_segment_t1(i).unwrap_or(0) as f64) / 100.0)
                        .clamp(seg_start, chunk_end);
                    let raw_text = state
                        .full_get_segment_text_lossy(i)
                        .unwrap_or_default()
                        .trim()
                        .to_string();

                    if raw_text.is_empty() || is_hallucination(&raw_text) {
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

                            let mut tok_start = (chunk_start + data.t0 as f64 / 100.0).max(seg_start);
                            let mut tok_end = (chunk_start + data.t1 as f64 / 100.0)
                                .min(seg_end)
                                .max(tok_start);

                            if data.t_dtw >= 0 {
                                let dtw_sec = chunk_start + (data.t_dtw as f64) / 100.0;
                                if dtw_sec >= seg_start - 0.50 && dtw_sec <= seg_end + 0.50 {
                                    if dtw_sec < tok_start - 0.30 {
                                        tok_start = (dtw_sec - 0.10).max(seg_start);
                                    } else if dtw_sec - tok_start > 0.60 {
                                        tok_start = (dtw_sec - 0.15).max(tok_start);
                                    }
                                    let rel_drop = if options.isolated { 16.0 } else { 8.0 };
                                    if tok_end - dtw_sec > 0.90
                                        && energy.mean_db_in_range(dtw_sec + 0.45, tok_end)
                                            < energy.peak_db_in_range(tok_start, dtw_sec + 0.45)
                                                - rel_drop
                                    {
                                        tok_end = (dtw_sec + 0.55).min(tok_end);
                                    }
                                    if tok_end <= tok_start {
                                        tok_end =
                                            (dtw_sec + 0.20).min(seg_end).max(tok_start + 0.05);
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

                    for w in &mut seg_words {
                        if w.end - w.start > 1.60 {
                            w.end = energy.trim_trailing_tail(
                                w.start,
                                w.end,
                                (w.start + 0.50).min(w.end),
                                options.isolated,
                            );
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
                            raw_text
                        );
                        continue;
                    }

                    let sub_groups = if seg_words.is_empty() {
                        vec![Vec::new()]
                    } else {
                        split_segment_by_word_pauses(&seg_words)
                    };
                    let split_used_raw = sub_groups.len() == 1 && seg_words.is_empty();

                    for mut sub_words in sub_groups {
                        let sub_text = if split_used_raw {
                            raw_text.clone()
                        } else {
                            join_words_text(&sub_words)
                        };
                        if sub_text.is_empty() || is_hallucination(&sub_text) {
                            continue;
                        }

                        let sub_prob = if sub_words.is_empty() {
                            avg_prob
                        } else {
                            sub_words.iter().map(|w| w.prob).sum::<f32>()
                                / (sub_words.len() as f32)
                        };
                        if !sub_words.is_empty() && sub_prob < 0.20 {
                            continue;
                        }
                        if (sub_words.len() == 1 || sub_text.split_whitespace().count() == 1)
                            && is_lone_filler_word(&sub_text)
                            && sub_prob < 0.80
                        {
                            continue;
                        }

                        let raw_start = sub_words.first().map(|w| w.start).unwrap_or(seg_start);
                        let raw_end = sub_words
                            .last()
                            .map(|w| w.end)
                            .unwrap_or(seg_end)
                            .max(raw_start);

                        let (refined_start, refined_end) =
                            energy.refine_bounds(raw_start, raw_end, options.isolated);

                        if sub_text.split_whitespace().count() <= 3
                            && (refined_end - refined_start) >= 10.0
                        {
                            tracing::debug!(
                                "Dropping stretched instrumental-solo phantom ({:.2}s..{:.2}s): {}",
                                refined_start,
                                refined_end,
                                sub_text
                            );
                            continue;
                        }

                        if energy.peak_db_in_range(refined_start, refined_end) < -52.0 {
                            tracing::debug!(
                                "Dropping silent/non-vocal phantom segment at {:.2}s: {}",
                                refined_start,
                                sub_text
                            );
                            continue;
                        }

                        if is_hallucination_loop(
                            &attempt_history,
                            &sub_text,
                            refined_start,
                            refined_end,
                            sub_prob,
                        ) {
                            tracing::debug!(
                                "Dropping hallucination loop segment (avg_prob={:.2}): {}",
                                sub_prob,
                                sub_text
                            );
                            continue;
                        }

                        attempt_history.push(KeptSegmentMeta {
                            norm_text: sub_text.trim().to_lowercase(),
                            start_sec: refined_start,
                        });
                        if attempt_history.len() > 8 {
                            attempt_history.remove(0);
                        }

                        for w in &mut sub_words {
                            let ws = w.start.max(refined_start).min(refined_end);
                            let we = w.end.min(refined_end).max(ws);
                            w.start = round_centis(ws);
                            w.end = round_centis(we);
                        }
                        chunk_words.extend(sub_words.iter().cloned());

                        let start = round_centis(refined_start);
                        let end = round_centis(refined_end).max(start);
                        let ts = format_lrc_timestamp(start);

                        chunk_segments.push(LrcSegment {
                            start,
                            end,
                            timestamp: ts,
                            text: sub_text,
                            words: sub_words,
                        });
                    }
                }

                if !chunk_segments.is_empty() {
                    history = attempt_history;
                    segments.extend(chunk_segments);
                    all_words.extend(chunk_words);
                    break;
                }
            }
        }

        let inference_dur = t0.elapsed();
        tracing::info!(
            "Whisper inference finished in {:.2?} (raw segments: {}, kept: {})",
            inference_dur,
            total_raw_segments,
            segments.len()
        );

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

        // High-confidence 1s-spaced consecutive greedy loop is dropped after 2 repetitions
        let fast_loop = vec![
            KeptSegmentMeta { norm_text: "and i'm always in the dark".into(), start_sec: 140.0 },
            KeptSegmentMeta { norm_text: "and i'm always in the dark".into(), start_sec: 141.0 },
        ];
        assert!(is_hallucination_loop(
            &fast_loop,
            "And I'm always in the dark",
            142.0,
            143.0,
            0.95
        ));

        // Legitimate spaced hook repeats (>= 2.6s apart) are preserved
        let hook_hist = vec![
            KeptSegmentMeta { norm_text: "listen, listen, listen".into(), start_sec: 178.46 },
            KeptSegmentMeta { norm_text: "listen, listen, listen".into(), start_sec: 182.10 },
            KeptSegmentMeta { norm_text: "listen, listen, listen".into(), start_sec: 185.73 },
        ];
        assert!(!is_hallucination_loop(
            &hook_hist,
            "Listen, listen, listen",
            189.37,
            191.0,
            0.90
        ));
    }

    #[test]
    fn test_split_segment_by_word_pauses_splits_chorus_lines_cleanly() {
        let mk = |s, e, t: &str| TimedWord {
            start: s,
            end: e,
            text: t.to_string(),
            prob: 0.9,
        };
        let words = vec![
            mk(145.25, 145.6, "There's"),
            mk(145.6, 146.2, "nothing"),
            mk(146.2, 146.4, "I"),
            mk(146.4, 147.2, "can"),
            mk(147.2, 147.7, "do"),
            mk(147.72, 148.0, "A"),
            mk(148.0, 148.6, "total"),
            mk(148.6, 149.2, "eclipse"),
            mk(149.2, 149.4, "of"),
            mk(149.4, 149.6, "the"),
            mk(149.6, 149.9, "heart"),
        ];
        let groups = split_segment_by_word_pauses(&words);
        assert_eq!(groups.len(), 2);
        assert_eq!(join_words_text(&groups[0]), "There's nothing I can do");
        assert_eq!(join_words_text(&groups[1]), "A total eclipse of the heart");

        // Connector word + brief pause must NOT orphan a single trailing lowercase word ("heart")
        let connected = vec![
            mk(147.24, 147.90, "A"),
            mk(147.90, 148.30, "total"),
            mk(148.30, 148.70, "eclipse"),
            mk(148.70, 148.90, "of"),
            mk(148.90, 149.10, "the"),
            mk(149.70, 150.20, "heart"),
        ];
        let connected_groups = split_segment_by_word_pauses(&connected);
        assert_eq!(connected_groups.len(), 1);
        assert_eq!(
            join_words_text(&connected_groups[0]),
            "A total eclipse of the heart"
        );
    }

    #[test]
    fn test_is_hallucination_drops_vocalise_loops_keeps_real_lyrics() {
        assert!(is_hallucination("Ooh, ooh, ooh, ooh"));
        assert!(!is_hallucination("Listen, listen, listen"));
        assert!(!is_hallucination("La, la, la, la"));
    }
}
