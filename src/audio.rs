use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, StandardTagKey};
use symphonia::core::probe::Hint;

#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub pcm: Vec<f32>,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Default)]
pub struct AudioMetadata {
    pub artist: Option<String>,
    pub title: Option<String>,
}

/// Extracts `artist` and `title` tags from container/ID3/Vorbis metadata if present.
pub fn extract_audio_metadata(bytes: &[u8]) -> AudioMetadata {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let Ok(mut probed) = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) else {
        return AudioMetadata::default();
    };

    let mut artist: Option<String> = None;
    let mut title: Option<String> = None;

    let mut inspect_rev = |rev: &symphonia::core::meta::MetadataRevision| {
        for tag in rev.tags() {
            match tag.std_key {
                Some(StandardTagKey::Artist)
                | Some(StandardTagKey::AlbumArtist)
                | Some(StandardTagKey::ArtistPerformer)
                | Some(StandardTagKey::LeadArtistPerformerSoloist) => {
                    if artist.is_none() {
                        let val = tag.value.to_string().trim().to_string();
                        if !val.is_empty() {
                            artist = Some(val);
                        }
                    }
                }
                Some(StandardTagKey::TrackTitle) => {
                    if title.is_none() {
                        let val = tag.value.to_string().trim().to_string();
                        if !val.is_empty() {
                            title = Some(val);
                        }
                    }
                }
                _ => {}
            }
        }
    };

    if let Some(mut meta) = probed.metadata.get() {
        if let Some(rev) = meta.skip_to_latest() {
            inspect_rev(rev);
        }
    }

    if let Some(rev) = probed.format.metadata().skip_to_latest() {
        inspect_rev(rev);
    }

    AudioMetadata { artist, title }
}

/// Applies an RMS silence gate to 16kHz mono PCM audio.
/// Sustained non-vocal sections (>= 1.0s where RMS energy is below `threshold_db`, e.g. -42 dBFS)
/// are zeroed out to pure digital silence so Whisper does not hallucinate during instrumental outros.
pub fn apply_rms_silence_gate(pcm: &mut [f32], sample_rate: usize, threshold_db: f32) {
    if pcm.is_empty() || sample_rate == 0 {
        return;
    }

    let window_size = (sample_rate / 4).max(1); // 250ms windows
    let min_sustained_windows = 4; // 4 * 250ms = 1.0s of sustained silence
    let threshold_linear = 10.0_f32.powf(threshold_db / 20.0);

    let num_windows = (pcm.len() + window_size - 1) / window_size;
    let mut is_below = vec![false; num_windows];

    for (idx, chunk) in pcm.chunks(window_size).enumerate() {
        let sum_sq: f32 = chunk.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / (chunk.len() as f32)).sqrt();
        is_below[idx] = rms < threshold_linear;
    }

    // Identify runs of >= min_sustained_windows consecutive quiet windows
    let mut run_start = None;
    for i in 0..=num_windows {
        let below = if i < num_windows { is_below[i] } else { false };
        match (below, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(start)) => {
                if i - start >= min_sustained_windows {
                    let sample_start = start * window_size;
                    let sample_end = (i * window_size).min(pcm.len());
                    pcm[sample_start..sample_end].fill(0.0);
                }
                run_start = None;
            }
            _ => {}
        }
    }
}

/// Decodes audio bytes (MP3, WAV, FLAC, AAC, etc.) into 16,000 Hz mono f32 PCM.
pub fn decode_audio_to_16k_mono(bytes: &[u8]) -> anyhow::Result<DecodedAudio> {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("No supported audio track found"))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("Unknown sample rate in audio track"))?;

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut mono_samples = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let channels = audio_buf.spec().channels.count();
                let mut sample_buf =
                    SampleBuffer::<f32>::new(audio_buf.capacity() as u64, *audio_buf.spec());
                sample_buf.copy_interleaved_ref(audio_buf);

                let samples = sample_buf.samples();

                if channels == 1 {
                    mono_samples.extend_from_slice(samples);
                } else {
                    for chunk in samples.chunks(channels) {
                        let sum: f32 = chunk.iter().sum();
                        mono_samples.push(sum / (channels as f32));
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => (),
            Err(e) => return Err(e.into()),
        }
    }

    let pcm = if sample_rate == 16000 {
        mono_samples
    } else {
        resample_linear(&mono_samples, sample_rate, 16000)
    };

    let duration_secs = (pcm.len() as f64) / 16000.0;

    Ok(DecodedAudio { pcm, duration_secs })
}

pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;

        let s0 = input[idx0];
        let s1 = input[idx1];
        output.push(s0 + frac * (s1 - s0));
    }

    output
}
