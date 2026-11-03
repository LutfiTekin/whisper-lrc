use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use burn::backend::NdArray;
use demucs_core::model::metadata::{download_url, StemId, HTDEMUCS};
use demucs_core::{Demucs, ModelOptions};
use tokio::sync::oneshot;

use crate::audio::{decode_audio_to_stereo, stereo_to_16k_mono, DecodedAudio};

type CpuBackend = NdArray<f32>;

async fn ensure_demucs_weights(model_dir: &Path) -> Result<PathBuf> {
    let path = model_dir.join(HTDEMUCS.filename);
    if path.exists() {
        return Ok(path);
    }

    tokio::fs::create_dir_all(model_dir).await?;

    let url = download_url(&HTDEMUCS);
    tracing::info!(
        "Demucs weights {} not found. Downloading (~{} MB) from {}...",
        HTDEMUCS.filename,
        HTDEMUCS.size_mb,
        url
    );

    let resp = reqwest::get(&url)
        .await
        .with_context(|| format!("Failed to request Demucs weights from {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "Failed downloading Demucs weights {}: HTTP {}",
            HTDEMUCS.filename,
            resp.status()
        );
    }

    let bytes = resp.bytes().await?;
    tokio::fs::write(&path, &bytes).await?;
    tracing::info!("Demucs weights {} downloaded successfully.", HTDEMUCS.filename);

    Ok(path)
}

/// Isolates vocals from mixed audio bytes in-memory using pure-Rust `demucs-rs` (`demucs-core` HTDemucs v4)
/// on CPU (Burn NdArray SIMD + multi-threading) and returns 16 kHz mono PCM `DecodedAudio`.
pub async fn isolate_vocals_to_16k_mono(
    audio_bytes: &[u8],
    model_dir: &Path,
) -> Result<DecodedAudio> {
    let weights_path = ensure_demucs_weights(model_dir).await?;
    let weights_bytes = tokio::fs::read(&weights_path)
        .await
        .with_context(|| format!("Failed reading Demucs weights at {:?}", weights_path))?;

    let raw_owned = audio_bytes.to_vec();
    let (tx, rx) = oneshot::channel();

    tracing::info!("demucs-rs: isolating vocal stem in pure Rust (HTDemucs v4)...");

    std::thread::Builder::new()
        .name("demucs-rs-worker".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let res = (|| -> Result<DecodedAudio> {
                let stereo = decode_audio_to_stereo(&raw_owned)?;
                let device = Default::default();
                let model = Demucs::<CpuBackend>::from_bytes(
                    ModelOptions::FourStem,
                    &weights_bytes,
                    device,
                )
                .map_err(|e| anyhow::anyhow!("Failed loading HTDemucs weights: {e}"))?;

                let stems = pollster::block_on(model.separate(
                    &stereo.left,
                    &stereo.right,
                    stereo.sample_rate,
                ))
                .map_err(|e| anyhow::anyhow!("HTDemucs separation failed: {e}"))?;

                let vocals = stems
                    .into_iter()
                    .find(|s| s.id == StemId::Vocals)
                    .ok_or_else(|| anyhow::anyhow!("Vocals stem missing from HTDemucs output"))?;

                Ok(stereo_to_16k_mono(
                    &vocals.left,
                    &vocals.right,
                    stereo.sample_rate,
                ))
            })();
            let _ = tx.send(res);
        })
        .context("Failed spawning demucs-rs worker thread")?;

    let decoded = rx
        .await
        .context("demucs-rs worker thread panicked or dropped channel")??;

    tracing::info!(
        "demucs-rs: vocal isolation complete ({:.1}s, {} samples at 16kHz).",
        decoded.duration_secs,
        decoded.pcm.len()
    );

    Ok(decoded)
}

/// Convenience wrapper using `MODEL_DIR` env var (or `/models` default).
pub async fn isolate_vocals(audio_bytes: &[u8]) -> Result<DecodedAudio> {
    let model_dir = PathBuf::from(
        std::env::var("MODEL_DIR").unwrap_or_else(|_| "/models".to_string()),
    );
    isolate_vocals_to_16k_mono(audio_bytes, &model_dir).await
}
