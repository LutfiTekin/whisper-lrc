use std::f64::consts::{FRAC_1_SQRT_2, PI};
use serde::{Deserialize, Serialize};

use crate::transcriber::{format_lrc_timestamp, LrcSegment};

/// A single word (merged from Whisper subword tokens) with DTW-refined start/end bounds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimedWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub prob: f32,
}

/// Controls insertion of blank LRC timestamp lines during instrumental breaks and outros.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BreakConfig {
    pub insert_breaks: bool,
    pub break_offset_ms: u64,
    pub min_break_gap_ms: u64,
}

impl Default for BreakConfig {
    fn default() -> Self {
        Self {
            insert_breaks: true,
            break_offset_ms: 1500,
            min_break_gap_ms: 4000,
        }
    }
}

#[derive(Debug, Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn highpass(sample_rate: f64, cutoff_hz: f64) -> Self {
        let fc = cutoff_hz.clamp(1.0, sample_rate * 0.45);
        let w0 = 2.0 * PI * fc / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) * 0.5) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: ((1.0 + cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn lowpass(sample_rate: f64, cutoff_hz: f64) -> Self {
        let fc = cutoff_hz.clamp(1.0, sample_rate * 0.45);
        let w0 = 2.0 * PI * fc / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) * 0.5) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Band-passed (300 Hz - 3400 Hz) vocal-energy envelope sampled in 20 ms frames.
#[derive(Debug, Clone)]
pub struct VocalEnergyProfile {
    frame_db: Vec<f32>,
    frame_sec: f64,
    total_duration_secs: f64,
}

impl VocalEnergyProfile {
    pub fn from_pcm(pcm: &[f32], sample_rate: usize) -> Self {
        if pcm.is_empty() || sample_rate == 0 {
            return Self {
                frame_db: Vec::new(),
                frame_sec: 0.02,
                total_duration_secs: 0.0,
            };
        }

        let sr = sample_rate as f64;
        let frame_samples = (sr * 0.020).round().max(1.0) as usize;
        let frame_sec = (frame_samples as f64) / sr;
        let total_duration_secs = (pcm.len() as f64) / sr;

        let mut hp = Biquad::highpass(sr, 300.0);
        let mut lp = Biquad::lowpass(sr, 3400.0);

        let mut frame_db = Vec::with_capacity((pcm.len() + frame_samples - 1) / frame_samples);
        for chunk in pcm.chunks(frame_samples) {
            let mut sum_sq = 0.0_f64;
            for &sample in chunk {
                let bp = lp.process(hp.process(sample as f64));
                sum_sq += bp * bp;
            }
            let rms = (sum_sq / (chunk.len() as f64)).sqrt();
            let db = if rms > 1e-5 {
                (20.0 * rms.log10()).max(-100.0) as f32
            } else {
                -100.0
            };
            frame_db.push(db);
        }

        Self {
            frame_db,
            frame_sec,
            total_duration_secs,
        }
    }

    pub fn peak_db_in_range(&self, start_sec: f64, end_sec: f64) -> f32 {
        if self.frame_db.is_empty() || end_sec <= start_sec {
            return -100.0;
        }
        let (f0, f1) = self.frame_range(start_sec, end_sec);
        if f0 > f1 || f0 >= self.frame_db.len() {
            return -100.0;
        }
        self.frame_db[f0..=f1]
            .iter()
            .copied()
            .fold(-100.0_f32, f32::max)
    }

    /// Tightens coarse segment boundaries onto actual vocal-band energy onset and offset.
    pub fn refine_bounds(&self, start_sec: f64, end_sec: f64, is_isolated: bool) -> (f64, f64) {
        let s = start_sec.clamp(0.0, self.total_duration_secs);
        let e = end_sec.clamp(s, self.total_duration_secs);
        if self.frame_db.is_empty() || e - s <= 0.25 {
            return (s, e);
        }

        let local_peak = self.peak_db_in_range(s, e);
        if local_peak < -52.0 {
            return (s, e);
        }

        let onset_thresh = if is_isolated {
            (local_peak - 24.0).max(-45.0)
        } else {
            (local_peak - 22.0).max(-38.0)
        };
        let offset_thresh = if is_isolated {
            (local_peak - 26.0).max(-48.0)
        } else {
            (local_peak - 24.0).max(-40.0)
        };

        let (f_start, f_end) = self.frame_range(s, e);
        if f_start > f_end {
            return (s, e);
        }

        let mut refined_start = s;
        if let Some(rel_idx) = self.frame_db[f_start..=f_end]
            .iter()
            .position(|&db| db >= onset_thresh)
        {
            let onset_frame = (f_start + rel_idx).saturating_sub(1).max(f_start);
            let mut candidate = (onset_frame as f64) * self.frame_sec;
            if !is_isolated {
                candidate = candidate.min(s + 1.5);
            }
            refined_start = candidate.clamp(s, (e - 0.25).max(s));
        }

        let mut refined_end = e;
        if let Some(rev_idx) = self.frame_db[f_start..=f_end]
            .iter()
            .rev()
            .position(|&db| db >= offset_thresh)
        {
            let last_active = f_end - rev_idx;
            let offset_frame = (last_active + 2).min(f_end);
            let mut candidate = (((offset_frame + 1) as f64) * self.frame_sec).min(e);
            if !is_isolated {
                candidate = candidate.max(e - 1.5);
            }
            refined_end = candidate
                .max((refined_start + 0.20).min(e))
                .min(e);
        }

        (refined_start, refined_end)
    }

    /// Finds the first vocal-energy onset within `[start_sec, end_sec]`, if one exceeds `-45 dBFS`.
    pub fn find_onset_in_window(&self, start_sec: f64, end_sec: f64) -> Option<f64> {
        let s = start_sec.clamp(0.0, self.total_duration_secs);
        let e = end_sec.clamp(s, self.total_duration_secs);
        if self.frame_db.is_empty() || e - s < 0.20 {
            return None;
        }
        let peak = self.peak_db_in_range(s, e);
        if peak < -45.0 {
            return None;
        }
        let thresh = (peak - 18.0).max(-42.0);
        let (f_start, f_end) = self.frame_range(s, e);
        let mut scan_start = f_start;
        if f_start > 0 && self.frame_db[f_start - 1] >= thresh {
            while scan_start <= f_end && self.frame_db[scan_start] >= thresh {
                scan_start += 1;
            }
            if scan_start > f_end {
                return None;
            }
        }
        self.frame_db[scan_start..=f_end]
            .iter()
            .position(|&db| db >= thresh)
            .map(|rel| {
                let frame = (scan_start + rel).saturating_sub(1).max(f_start);
                ((frame as f64) * self.frame_sec).clamp(s, e)
            })
    }

    fn frame_range(&self, start_sec: f64, end_sec: f64) -> (usize, usize) {
        if self.frame_db.is_empty() {
            return (0, 0);
        }
        let max_idx = self.frame_db.len() - 1;
        let f_start = ((start_sec.max(0.0) / self.frame_sec).floor() as usize).min(max_idx);
        let f_end = ((end_sec.max(0.0) / self.frame_sec).ceil() as usize)
            .min(max_idx)
            .max(f_start);
        (f_start, f_end)
    }
}

/// Formats segments into LRC text and inserts blank `[mm:ss.xx]` lines during instrumental breaks
/// and song outros (`break_offset_ms` after the previous line's last word end timestamp).
pub fn format_lrc_with_breaks(
    segments: &[LrcSegment],
    audio_duration_secs: f64,
    cfg: &BreakConfig,
) -> (String, usize) {
    if segments.is_empty() {
        return (String::new(), 0);
    }

    const MIN_VISIBLE_BLANK_SECS: f64 = 2.0;
    let offset_secs = (cfg.break_offset_ms as f64) / 1000.0;
    let min_gap_secs = (cfg.min_break_gap_ms as f64) / 1000.0;

    let mut lines = Vec::with_capacity(segments.len() + 8);
    let mut breaks_inserted = 0usize;

    for (idx, seg) in segments.iter().enumerate() {
        lines.push(format!("{} {}", seg.timestamp, seg.text));

        if !cfg.insert_breaks {
            continue;
        }

        if let Some(next) = segments.get(idx + 1) {
            let gap = next.start - seg.end;
            let blank_ts = seg.end + offset_secs;
            if gap >= min_gap_secs && (next.start - blank_ts) >= MIN_VISIBLE_BLANK_SECS {
                lines.push(format_lrc_timestamp(blank_ts));
                breaks_inserted += 1;
            }
        } else {
            let outro_gap = audio_duration_secs - seg.end;
            let blank_ts = seg.end + offset_secs;
            if outro_gap >= min_gap_secs
                && (audio_duration_secs - blank_ts) >= MIN_VISIBLE_BLANK_SECS
            {
                lines.push(format_lrc_timestamp(blank_ts));
                breaks_inserted += 1;
            }
        }
    }

    (format!("{}\n", lines.join("\n")), breaks_inserted)
}

fn strip_leading_lrc_tags(line: &str) -> &str {
    let mut rest = line.trim_start();
    while rest.starts_with('[') {
        let Some(close_idx) = rest.find(']') else {
            break;
        };
        let inside = &rest[1..close_idx];
        if is_lrc_timestamp_tag(inside) {
            rest = rest[close_idx + 1..].trim_start();
        } else {
            break;
        }
    }
    rest
}

fn is_lrc_timestamp_tag(inside: &str) -> bool {
    let Some((mins, secs)) = inside.split_once(':') else {
        return false;
    };
    if mins.is_empty() || !mins.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let (sec_whole, sec_frac) = match secs.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (secs, None),
    };
    if sec_whole.is_empty() || !sec_whole.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    match sec_frac {
        Some(f) => !f.is_empty() && f.chars().all(|c| c.is_ascii_digit()),
        None => true,
    }
}

fn is_lrc_metadata_header(trimmed: &str) -> bool {
    if !(trimmed.starts_with('[') && trimmed.ends_with(']')) {
        return false;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.contains(']') {
        return false;
    }
    let Some((key, _val)) = inner.split_once(':') else {
        return false;
    };
    !key.is_empty() && key.chars().all(|c| c.is_ascii_alphabetic())
}

/// Parses plain or LRC-prefixed reference lyrics into clean non-empty lyric lines.
pub fn parse_reference_lyrics(raw: &str) -> Vec<String> {
    raw.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || is_lrc_metadata_header(trimmed) {
                return None;
            }
            let clean = strip_leading_lrc_tags(trimmed).trim();
            if clean.is_empty() || is_lrc_metadata_header(clean) {
                None
            } else {
                Some(clean.to_string())
            }
        })
        .collect()
}

/// Extracts up to `max_chars` (on a UTF-8 boundary) of clean reference lyrics for Whisper seeding.
pub fn extract_lyrics_prompt_hint(raw: &str, max_chars: usize) -> Option<String> {
    if max_chars == 0 {
        return None;
    }
    let lines = parse_reference_lyrics(raw);
    if lines.is_empty() {
        return None;
    }

    let mut hint = String::new();
    for line in lines {
        let sep = if hint.is_empty() { "" } else { ", " };
        let next_len = hint.chars().count() + sep.chars().count() + line.chars().count();
        if next_len <= max_chars {
            hint.push_str(sep);
            hint.push_str(&line);
        } else if hint.is_empty() {
            let clipped: String = line.chars().take(max_chars).collect();
            hint.push_str(clipped.trim_end());
            break;
        } else {
            break;
        }
    }

    let trimmed = hint.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn is_unspaced_script_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x309F // Hiragana
            | 0x30A0..=0x30FF // Katakana
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0x3400..=0x4DBF // CJK Extension A
            | 0xAC00..=0xD7AF // Hangul Syllables
            | 0x0E00..=0x0E7F // Thai
    )
}

/// Normalizes text into alignment tokens: alphanumeric runs for spaced scripts,
/// individual characters for unspaced CJK/Kana/Hangul/Thai scripts.
pub fn tokenize_for_alignment(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if is_unspaced_script_char(ch) {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                current.push(lc);
            }
        } else if ch == '\'' || ch == '’' {
            // Keep internal apostrophes out so "causin'" and "causing" stay close without splitting words
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn token_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len());
    let dist = levenshtein_chars(&a_chars, &b_chars);
    1.0 - (dist as f32) / (max_len as f32)
}

fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b.len()]
}

const DIR_DIAG: u8 = 1;
const DIR_UP: u8 = 2;
const DIR_LEFT: u8 = 3;

const GAP_REF: i32 = -25; // Hypothesis missed a reference lyric word
const GAP_HYP: i32 = -15; // Hypothesis emitted an extra word

#[inline]
fn substitution_score(ref_tok: &str, hyp_tok: &str) -> (i32, f32) {
    if ref_tok == hyp_tok {
        return (100, 1.0);
    }
    let sim = token_similarity(ref_tok, hyp_tok);
    if sim >= 0.60 {
        ((sim * 90.0).round() as i32, sim)
    } else {
        (-40, sim)
    }
}

fn global_dp_align(ref_tokens: &[(usize, String)], hyp_tokens: &[(usize, String)]) -> Vec<(usize, usize)> {
    let r = ref_tokens.len();
    let h = hyp_tokens.len();
    let cols = h + 1;
    let mut dp = vec![i32::MIN / 4; (r + 1) * cols];
    let mut dir = vec![0u8; (r + 1) * cols];

    dp[0] = 0;
    for i in 1..=r {
        dp[i * cols] = (i as i32) * GAP_REF;
        dir[i * cols] = DIR_UP;
    }
    for j in 1..=h {
        dp[j] = (j as i32) * GAP_HYP;
        dir[j] = DIR_LEFT;
    }

    for i in 1..=r {
        let ref_tok = &ref_tokens[i - 1].1;
        for j in 1..=h {
            let (sub, _) = substitution_score(ref_tok, &hyp_tokens[j - 1].1);
            let diag = dp[(i - 1) * cols + (j - 1)] + sub;
            let up = dp[(i - 1) * cols + j] + GAP_REF;
            let left = dp[i * cols + (j - 1)] + GAP_HYP;

            let (best, best_dir) = if diag >= up && diag >= left {
                (diag, DIR_DIAG)
            } else if up >= left {
                (up, DIR_UP)
            } else {
                (left, DIR_LEFT)
            };
            dp[i * cols + j] = best;
            dir[i * cols + j] = best_dir;
        }
    }

    let mut matches = Vec::new();
    let mut i = r;
    let mut j = h;
    while i > 0 || j > 0 {
        match dir[i * cols + j] {
            DIR_DIAG => {
                let (_, sim) = substitution_score(&ref_tokens[i - 1].1, &hyp_tokens[j - 1].1);
                if sim >= 0.55 {
                    matches.push((i - 1, hyp_tokens[j - 1].0));
                }
                i -= 1;
                j -= 1;
            }
            DIR_UP => i -= 1,
            DIR_LEFT => j -= 1,
            _ => break,
        }
    }
    matches.reverse();
    matches
}

fn banded_dp_align(ref_tokens: &[(usize, String)], hyp_tokens: &[(usize, String)]) -> Vec<(usize, usize)> {
    let r = ref_tokens.len();
    let h = hyp_tokens.len();
    let band = (r.abs_diff(h) + 200).max(200);

    let mut row_j_start = Vec::with_capacity(r + 1);
    let mut row_offsets = Vec::with_capacity(r + 1);
    let mut total_cells = 0usize;

    for i in 0..=r {
        let center = (i * h) / r.max(1);
        let j0 = center.saturating_sub(band);
        let j1 = (center + band).min(h);
        row_j_start.push(j0);
        row_offsets.push(total_cells);
        total_cells += j1 - j0 + 1;
    }

    const NEG_INF: i32 = i32::MIN / 4;
    let mut dp = vec![NEG_INF; total_cells];
    let mut dir = vec![0u8; total_cells];

    let cell_idx = |row: usize, col: usize, j_start: &[usize], offs: &[usize]| -> Option<usize> {
        let j0 = j_start[row];
        let next_off = offs.get(row + 1).copied().unwrap_or(total_cells);
        let len = next_off - offs[row];
        if col < j0 || col >= j0 + len {
            None
        } else {
            Some(offs[row] + (col - j0))
        }
    };

    dp[0] = 0;
    let row0_len = row_offsets[1] - row_offsets[0];
    for offset_j in 1..row0_len {
        let j = row_j_start[0] + offset_j;
        dp[offset_j] = (j as i32) * GAP_HYP;
        dir[offset_j] = DIR_LEFT;
    }

    for i in 1..=r {
        let j0 = row_j_start[i];
        let next_off = row_offsets.get(i + 1).copied().unwrap_or(total_cells);
        let row_len = next_off - row_offsets[i];
        let ref_tok = &ref_tokens[i - 1].1;

        for offset_j in 0..row_len {
            let j = j0 + offset_j;
            let cur = row_offsets[i] + offset_j;

            if j == 0 {
                if let Some(up_idx) = cell_idx(i - 1, 0, &row_j_start, &row_offsets) {
                    dp[cur] = dp[up_idx] + GAP_REF;
                    dir[cur] = DIR_UP;
                }
                continue;
            }

            let diag = cell_idx(i - 1, j - 1, &row_j_start, &row_offsets).map(|idx| {
                let (sub, _) = substitution_score(ref_tok, &hyp_tokens[j - 1].1);
                dp[idx] + sub
            });
            let up = cell_idx(i - 1, j, &row_j_start, &row_offsets).map(|idx| dp[idx] + GAP_REF);
            let left = cell_idx(i, j - 1, &row_j_start, &row_offsets).map(|idx| dp[idx] + GAP_HYP);

            let mut best = NEG_INF;
            let mut best_dir = 0u8;
            if let Some(v) = diag {
                if v >= best {
                    best = v;
                    best_dir = DIR_DIAG;
                }
            }
            if let Some(v) = up {
                if v > best {
                    best = v;
                    best_dir = DIR_UP;
                }
            }
            if let Some(v) = left {
                if v > best {
                    best = v;
                    best_dir = DIR_LEFT;
                }
            }
            dp[cur] = best;
            dir[cur] = best_dir;
        }
    }

    let mut matches = Vec::new();
    let mut i = r;
    let mut j = h;
    while i > 0 || j > 0 {
        let Some(cur) = cell_idx(i, j, &row_j_start, &row_offsets) else {
            break;
        };
        match dir[cur] {
            DIR_DIAG => {
                let (_, sim) = substitution_score(&ref_tokens[i - 1].1, &hyp_tokens[j - 1].1);
                if sim >= 0.55 {
                    matches.push((i - 1, hyp_tokens[j - 1].0));
                }
                i -= 1;
                j -= 1;
            }
            DIR_UP => i -= 1,
            DIR_LEFT => j -= 1,
            _ => break,
        }
    }
    matches.reverse();
    matches
}

#[derive(Debug, Clone)]
struct AlignedLineAnchor {
    text: String,
    token_count: usize,
    start: f64,
    end: f64,
    anchored: bool,
    words: Vec<TimedWord>,
}

/// Aligns known reference lyrics monotonically against Whisper word-level timestamps,
/// refining boundaries on vocal-band RMS energy and interpolating any missed lines.
pub fn align_lyrics_to_words(
    raw_lyrics: &str,
    words: &[TimedWord],
    energy: &VocalEnergyProfile,
    audio_duration_secs: f64,
    is_isolated: bool,
) -> Vec<LrcSegment> {
    let ref_lines = parse_reference_lyrics(raw_lyrics);
    if ref_lines.is_empty() {
        return Vec::new();
    }

    let total_dur = audio_duration_secs.max(ref_lines.len() as f64 * 0.20);
    let mut line_token_ranges = Vec::with_capacity(ref_lines.len());
    let mut ref_tokens: Vec<(usize, String)> = Vec::new();

    for (line_idx, line) in ref_lines.iter().enumerate() {
        let toks = tokenize_for_alignment(line);
        let start_tok_idx = ref_tokens.len();
        for tok in toks {
            ref_tokens.push((line_idx, tok));
        }
        line_token_ranges.push(start_tok_idx..ref_tokens.len());
    }

    let mut hyp_tokens: Vec<(usize, String)> = Vec::new();
    for (word_idx, word) in words.iter().enumerate() {
        for tok in tokenize_for_alignment(&word.text) {
            hyp_tokens.push((word_idx, tok));
        }
    }

    let mut anchors: Vec<AlignedLineAnchor> = ref_lines
        .iter()
        .enumerate()
        .map(|(idx, text)| AlignedLineAnchor {
            text: text.clone(),
            token_count: line_token_ranges[idx].len(),
            start: 0.0,
            end: 0.0,
            anchored: false,
            words: Vec::new(),
        })
        .collect();

    if !ref_tokens.is_empty() && !hyp_tokens.is_empty() {
        let matches = if ref_tokens.len().saturating_mul(hyp_tokens.len()) <= 4_000_000 {
            global_dp_align(&ref_tokens, &hyp_tokens)
        } else {
            banded_dp_align(&ref_tokens, &hyp_tokens)
        };

        let mut per_line_matches: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ref_lines.len()];
        for (ref_idx, word_idx) in matches {
            let line_idx = ref_tokens[ref_idx].0;
            let pos_in_line = ref_idx - line_token_ranges[line_idx].start;
            per_line_matches[line_idx].push((pos_in_line, word_idx));
        }

        for (line_idx, line_matches) in per_line_matches.into_iter().enumerate() {
            if line_matches.is_empty() {
                continue;
            }
            let &(first_pos, first_word_idx) = line_matches.first().unwrap();
            let &(last_pos, last_word_idx) = line_matches.last().unwrap();
            let tok_len = anchors[line_idx].token_count.max(1);

            let missed_leading = first_pos;
            let missed_trailing = tok_len.saturating_sub(1 + last_pos);

            let raw_start = (words[first_word_idx].start - 0.35 * (missed_leading as f64)).max(0.0);
            let raw_end =
                (words[last_word_idx].end + 0.35 * (missed_trailing as f64)).min(total_dur);
            let (refined_start, refined_end) = energy.refine_bounds(raw_start, raw_end, is_isolated);

            let mut line_words = Vec::with_capacity(line_matches.len());
            let mut last_seen_word = None;
            for (_, w_idx) in line_matches {
                if last_seen_word != Some(w_idx) {
                    line_words.push(words[w_idx].clone());
                    last_seen_word = Some(w_idx);
                }
            }

            anchors[line_idx].start = refined_start;
            anchors[line_idx].end = refined_end.max(refined_start + 0.15).min(total_dur);
            anchors[line_idx].anchored = true;
            anchors[line_idx].words = line_words;
        }
    }

    // Interpolate any contiguous runs of unanchored lines proportionally between anchored neighbors
    let mut i = 0;
    while i < anchors.len() {
        if anchors[i].anchored {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < anchors.len() && !anchors[i].anchored {
            i += 1;
        }
        let run_end = i;

        let win_start = if run_start > 0 {
            anchors[run_start - 1].end
        } else {
            0.0
        };
        let win_end = if run_end < anchors.len() {
            anchors[run_end].start.max(win_start)
        } else {
            total_dur.max(win_start)
        };

        let span = (win_end - win_start).max(0.0);
        let total_weight: usize = anchors[run_start..run_end]
            .iter()
            .map(|a| a.token_count.max(1))
            .sum();

        let mut cursor = win_start;
        for anchor in &mut anchors[run_start..run_end] {
            let weight = anchor.token_count.max(1) as f64;
            let slot_dur = span * (weight / (total_weight as f64));
            let slot_start = cursor;
            let slot_end = (cursor + slot_dur).min(total_dur);
            cursor = slot_end;

            let snapped_start = if slot_end - slot_start >= 0.35 {
                energy
                    .find_onset_in_window(slot_start, (slot_end - 0.20).max(slot_start))
                    .unwrap_or(slot_start)
            } else {
                slot_start
            };

            let (refined_start, refined_end) =
                energy.refine_bounds(snapped_start, slot_end, is_isolated);
            anchor.start = refined_start;
            anchor.end = refined_end.max(refined_start + 0.15).min(total_dur);
        }
    }

    // Enforce strict monotonic timestamps across all lines
    for idx in 0..anchors.len() {
        if idx > 0 {
            let prev_start = anchors[idx - 1].start;
            if anchors[idx].start <= prev_start + 0.05 {
                anchors[idx].start = (prev_start + 0.05).min(total_dur);
            }
            if anchors[idx - 1].end > anchors[idx].start {
                anchors[idx - 1].end = anchors[idx].start.max(prev_start + 0.05);
            }
        }
        anchors[idx].end = anchors[idx]
            .end
            .max(anchors[idx].start + 0.15)
            .min(total_dur);
    }

    anchors
        .into_iter()
        .map(|a| {
            let start = (a.start * 100.0).round() / 100.0;
            let end = ((a.end * 100.0).round() / 100.0).max(start);
            LrcSegment {
                start,
                end,
                timestamp: format_lrc_timestamp(start),
                text: a.text,
                words: a.words,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> LrcSegment {
        LrcSegment {
            start,
            end,
            timestamp: format_lrc_timestamp(start),
            text: text.to_string(),
            words: Vec::new(),
        }
    }

    #[test]
    fn test_format_lrc_with_breaks_inter_line_and_outro() {
        let segments = vec![
            seg(10.0, 12.5, "Line one"),
            seg(15.0, 18.0, "Line two"), // gap 2.5s (< 4.0s) -> no blank
            seg(28.0, 31.0, "Line three"), // gap 10.0s (>= 4.0s) -> blank at 19.50s
        ];
        let cfg = BreakConfig::default();
        let (lrc, breaks) = format_lrc_with_breaks(&segments, 40.0, &cfg);

        assert_eq!(breaks, 2);
        assert_eq!(
            lrc,
            "[00:10.00] Line one\n\
             [00:15.00] Line two\n\
             [00:19.50]\n\
             [00:28.00] Line three\n\
             [00:32.50]\n"
        );
    }

    #[test]
    fn test_format_lrc_with_breaks_disabled_and_custom_offset() {
        let segments = vec![seg(5.0, 8.0, "First"), seg(20.0, 22.0, "Second")];

        let disabled = BreakConfig {
            insert_breaks: false,
            ..BreakConfig::default()
        };
        let (lrc_off, count_off) = format_lrc_with_breaks(&segments, 35.0, &disabled);
        assert_eq!(count_off, 0);
        assert_eq!(lrc_off, "[00:05.00] First\n[00:20.00] Second\n");

        let custom = BreakConfig {
            insert_breaks: true,
            break_offset_ms: 1000,
            min_break_gap_ms: 5000,
        };
        let (lrc_custom, count_custom) = format_lrc_with_breaks(&segments, 35.0, &custom);
        assert_eq!(count_custom, 2);
        assert_eq!(
            lrc_custom,
            "[00:05.00] First\n[00:09.00]\n[00:20.00] Second\n[00:23.00]\n"
        );
    }

    #[test]
    fn test_vocal_energy_refines_coarse_bounds() {
        let sr = 16000usize;
        let total_samples = sr * 4;
        let mut pcm = vec![0.0_f32; total_samples];

        // 0..1s and 2.5..4s: sub-bass 60 Hz rumble (should be attenuated by 300 Hz HPF)
        // 1.0..2.5s: 600 Hz vocal-band tone
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = i as f64 / sr as f64;
            let rumble = 0.08 * (2.0 * PI * 60.0 * t).sin();
            let vocal = if (1.0..2.5).contains(&t) {
                0.25 * (2.0 * PI * 600.0 * t).sin()
            } else {
                0.0
            };
            *sample = (rumble + vocal) as f32;
        }

        let energy = VocalEnergyProfile::from_pcm(&pcm, sr);
        let (refined_start, refined_end) = energy.refine_bounds(0.2, 3.8, true);

        assert!(
            (refined_start - 1.0).abs() <= 0.10,
            "expected onset near 1.00s, got {refined_start}"
        );
        assert!(
            (refined_end - 2.5).abs() <= 0.10,
            "expected offset near 2.50s, got {refined_end}"
        );
    }

    #[test]
    fn test_parse_reference_lyrics_strips_lrc_and_headers() {
        let raw = "\
[ar:Peaches]
[ti:Boys Wanna Be Her]
[00:00.00] You've got them all, by the balls
[00:01.00][00:02.00] Causin' waterfalls

[by:someone]
   ";
        let parsed = parse_reference_lyrics(raw);
        assert_eq!(
            parsed,
            vec![
                "You've got them all, by the balls".to_string(),
                "Causin' waterfalls".to_string()
            ]
        );
    }

    #[test]
    fn test_align_lyrics_handles_misspellings_repeats_and_missing_line() {
        let raw_lyrics = "\
Causin' waterfalls
Parents peeved
The boys wanna be her
The girls wanna be her
The boys wanna be her";

        let word_specs = [
            (10.0, 10.4, "causing"),
            (10.4, 11.0, "waterfalls"),
            // "Parents peeved" completely omitted from ASR output
            (15.0, 15.2, "the"),
            (15.2, 15.5, "boys"),
            (15.5, 15.8, "wanna"),
            (15.8, 16.0, "be"),
            (16.0, 16.3, "her"),
            (17.0, 17.2, "the"),
            (17.2, 17.5, "girls"),
            (17.5, 17.8, "wanna"),
            (17.8, 18.0, "be"),
            (18.0, 18.3, "her"),
            (19.0, 19.2, "the"),
            (19.2, 19.5, "boys"),
            (19.5, 19.8, "wanna"),
            (19.8, 20.0, "be"),
            (20.0, 20.3, "her"),
        ];

        let words: Vec<TimedWord> = word_specs
            .iter()
            .map(|&(s, e, t)| TimedWord {
                start: s,
                end: e,
                text: t.to_string(),
                prob: 0.9,
            })
            .collect();

        // Build PCM with vocal energy during each active word
        let sr = 16000usize;
        let mut pcm = vec![0.0_f32; sr * 25];
        for (i, s) in pcm.iter_mut().enumerate() {
            let t = i as f64 / sr as f64;
            if words.iter().any(|w| t >= w.start && t <= w.end) || (12.5..13.5).contains(&t) {
                *s = (0.25 * (2.0 * PI * 500.0 * t).sin()) as f32;
            }
        }
        let energy = VocalEnergyProfile::from_pcm(&pcm, sr);
        let aligned = align_lyrics_to_words(raw_lyrics, &words, &energy, 25.0, true);

        assert_eq!(aligned.len(), 5);
        assert!((aligned[0].start - 10.0).abs() <= 0.15);
        // Unmatched middle line should fall between line 0 end (11.0) and line 2 start (15.0)
        assert!(aligned[1].start >= aligned[0].end && aligned[1].end <= aligned[2].start);
        assert!((aligned[1].start - 12.5).abs() <= 0.15);
        assert!((aligned[2].start - 15.0).abs() <= 0.15);
        assert!((aligned[3].start - 17.0).abs() <= 0.15);
        assert!((aligned[4].start - 19.0).abs() <= 0.15);
    }

    #[test]
    fn test_align_lyrics_cjk_characters() {
        let raw = "静かな夜\n星が光る";
        let words = vec![
            TimedWord {
                start: 14.0,
                end: 14.8,
                text: "静かな".into(),
                prob: 0.9,
            },
            TimedWord {
                start: 14.8,
                end: 15.4,
                text: "夜".into(),
                prob: 0.9,
            },
            TimedWord {
                start: 18.2,
                end: 18.7,
                text: "星が".into(),
                prob: 0.9,
            },
            TimedWord {
                start: 18.7,
                end: 19.4,
                text: "光る".into(),
                prob: 0.9,
            },
        ];
        let sr = 16000usize;
        let mut pcm = vec![0.0_f32; sr * 22];
        for (i, s) in pcm.iter_mut().enumerate() {
            let t = i as f64 / sr as f64;
            if (14.0..15.4).contains(&t) || (18.2..19.4).contains(&t) {
                *s = (0.25 * (2.0 * PI * 440.0 * t).sin()) as f32;
            }
        }
        let energy = VocalEnergyProfile::from_pcm(&pcm, sr);
        let aligned = align_lyrics_to_words(raw, &words, &energy, 22.0, true);
        assert_eq!(aligned.len(), 2);
        assert!((aligned[0].start - 14.0).abs() <= 0.15);
        assert!((aligned[1].start - 18.2).abs() <= 0.15);
    }
}
