pub mod alignment;
pub mod audio;
pub mod cache;
pub mod isolation;
pub mod queue;
pub mod transcriber;

pub use alignment::{
    align_lyrics_to_words, extract_lyrics_prompt_hint, format_lrc_with_breaks,
    parse_reference_lyrics, BreakConfig, TimedWord, VocalEnergyProfile,
};
pub use audio::{
    apply_rms_silence_gate, decode_audio_to_16k_mono, decode_audio_to_stereo,
    extract_audio_metadata, stereo_to_16k_mono, AudioMetadata, DecodedAudio, StereoAudio,
};
pub use cache::{CacheKey, LrcCache};
pub use isolation::{isolate_vocals, isolate_vocals_to_16k_mono};
pub use queue::PreemptiveQueue;
pub use transcriber::{
    build_initial_prompt, format_lrc_timestamp, get_model_filename, infer_dtw_mode,
    is_dtw_enabled, is_hallucination, LrcSegment, TranscribeOptions, TranscriberService,
    TranscriptionResult,
};

/// High-level library helper to transcribe audio bytes directly to LRC text.
pub async fn to_lrc(
    audio_bytes: &[u8],
    lang: &str,
    isolate_vocals_first: bool,
    model_dir: std::path::PathBuf,
    cpu_threads: i32,
) -> anyhow::Result<String> {
    to_lrc_with_metadata(
        audio_bytes,
        lang,
        isolate_vocals_first,
        None,
        None,
        None,
        model_dir,
        cpu_threads,
    )
    .await
}

/// High-level library helper to transcribe audio bytes with optional `artist`, `title`, and `prompt` hints.
/// Automatically falls back to embedded ID3/Vorbis tags if `artist` or `title` is not provided.
pub async fn to_lrc_with_metadata(
    audio_bytes: &[u8],
    lang: &str,
    isolate_vocals_first: bool,
    artist: Option<&str>,
    title: Option<&str>,
    prompt: Option<&str>,
    model_dir: std::path::PathBuf,
    cpu_threads: i32,
) -> anyhow::Result<String> {
    let options = TranscribeOptions {
        lang: lang.to_string(),
        prompt: prompt.map(str::to_string),
        isolated: isolate_vocals_first,
        ..TranscribeOptions::default()
    };
    let result = to_lrc_with_options(
        audio_bytes,
        isolate_vocals_first,
        artist,
        title,
        options,
        model_dir,
        cpu_threads,
    )
    .await?;
    Ok(result.lrc)
}

/// Full-featured library helper supporting reference-lyrics alignment and custom instrumental-break config.
pub async fn to_lrc_with_options(
    audio_bytes: &[u8],
    isolate_vocals_first: bool,
    artist: Option<&str>,
    title: Option<&str>,
    mut options: TranscribeOptions,
    model_dir: std::path::PathBuf,
    cpu_threads: i32,
) -> anyhow::Result<TranscriptionResult> {
    let embedded = if artist.is_none() || title.is_none() {
        extract_audio_metadata(audio_bytes)
    } else {
        AudioMetadata::default()
    };

    let effective_artist = artist.or(embedded.artist.as_deref());
    let effective_title = title.or(embedded.title.as_deref());

    let prompt_hint = options
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            options
                .reference_lyrics
                .as_deref()
                .and_then(|lyr| extract_lyrics_prompt_hint(lyr, 180))
        });

    options.prompt = build_initial_prompt(effective_artist, effective_title, prompt_hint.as_deref());

    let (mut decoded, actually_isolated) = if isolate_vocals_first {
        match isolate_vocals_to_16k_mono(audio_bytes, &model_dir).await {
            Ok(d) => (d, true),
            Err(e) => {
                tracing::warn!("demucs-rs vocal isolation failed, falling back to raw audio: {e}");
                (decode_audio_to_16k_mono(audio_bytes)?, false)
            }
        }
    } else {
        (decode_audio_to_16k_mono(audio_bytes)?, false)
    };
    options.isolated = actually_isolated;

    apply_rms_silence_gate(&mut decoded.pcm, 16000, -42.0);

    let service = TranscriberService::new(model_dir, cpu_threads);
    service
        .transcribe_with_options(&decoded.pcm, decoded.duration_secs, &options)
        .await
}
