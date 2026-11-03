pub mod audio;
pub mod cache;
pub mod isolation;
pub mod queue;
pub mod transcriber;

pub use audio::{decode_audio_to_16k_mono, DecodedAudio};
pub use cache::{CacheKey, LrcCache};
pub use isolation::isolate_vocals;
pub use queue::PreemptiveQueue;
pub use transcriber::{
    format_lrc_timestamp, LrcSegment, TranscriberService, TranscriptionResult,
};

/// High-level library helper to transcribe audio bytes directly to LRC text.
pub async fn to_lrc(
    audio_bytes: &[u8],
    lang: &str,
    isolate_vocals_first: bool,
    model_dir: std::path::PathBuf,
    cpu_threads: i32,
) -> anyhow::Result<String> {
    let raw_to_decode = if isolate_vocals_first {
        match isolate_vocals(audio_bytes).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Vocal isolation failed, falling back to raw audio: {e}");
                audio_bytes.to_vec()
            }
        }
    } else {
        audio_bytes.to_vec()
    };

    let decoded = decode_audio_to_16k_mono(&raw_to_decode)?;
    let service = TranscriberService::new(model_dir, cpu_threads);
    let result = service.transcribe(&decoded.pcm, lang).await?;
    Ok(result.lrc)
}
