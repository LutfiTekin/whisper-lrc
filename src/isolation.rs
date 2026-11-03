use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Context, Result};
use tokio::process::Command;

static ISO_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Isolates vocals from mixed audio bytes using Demucs htdemucs on CPU.
/// Returns isolated vocal audio bytes (WAV format).
pub async fn isolate_vocals(audio_bytes: &[u8]) -> Result<Vec<u8>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let count = ISO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_dir = std::env::temp_dir().join(format!("demucs_{now}_{count}"));

    tokio::fs::create_dir_all(&temp_dir).await?;

    let input_path = temp_dir.join("input.audio");
    tokio::fs::write(&input_path, audio_bytes).await?;

    let out_dir = temp_dir.join("out");

    tracing::info!("Demucs: isolating vocal stem on CPU...");
    let status = Command::new("demucs")
        .arg("-d")
        .arg("cpu")
        .arg("--two-stems")
        .arg("vocals")
        .arg("--other-method")
        .arg("none")
        .arg("-n")
        .arg("htdemucs")
        .arg("-o")
        .arg(&out_dir)
        .arg(&input_path)
        .status()
        .await
        .context("Failed executing demucs process")?;

    if !status.success() {
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        anyhow::bail!("Demucs exited with error status: {status}");
    }

    // Demucs outputs to <out_dir>/htdemucs/input/vocals.wav
    let vocals_path = out_dir.join("htdemucs").join("input").join("vocals.wav");
    let vocals_bytes = match tokio::fs::read(&vocals_path).await {
        Ok(bytes) => {
            tracing::info!(
                "Demucs: vocal isolation complete ({} bytes).",
                bytes.len()
            );
            Ok(bytes)
        }
        Err(e) => {
            Err(anyhow::anyhow!(
                "Failed reading isolated vocals at {:?}: {e}",
                vocals_path
            ))
        }
    };

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    vocals_bytes
}
