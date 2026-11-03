pub mod audio;
pub mod cache;
pub mod queue;
pub mod transcriber;

pub use audio::{decode_audio_to_16k_mono, DecodedAudio};
pub use cache::{CacheKey, LrcCache};
pub use queue::PreemptiveQueue;
pub use transcriber::{
    format_lrc_timestamp, LrcSegment, TranscriberService, TranscriptionResult,
};

/// High-level library helper to transcribe audio bytes directly to LRC text.
pub async fn to_lrc(
    audio_bytes: &[u8],
    lang: &str,
    model_dir: std::path::PathBuf,
    cpu_threads: i32,
) -> anyhow::Result<String> {
    let decoded = decode_audio_to_16k_mono(audio_bytes)?;
    let service = TranscriberService::new(model_dir, cpu_threads);
    let result = service.transcribe(&decoded.pcm, lang).await?;
    Ok(result.lrc)
}
