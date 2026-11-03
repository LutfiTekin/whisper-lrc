use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{oneshot, Mutex, Notify};
use tokio_util::sync::CancellationToken;

use crate::transcriber::{TranscriberService, TranscriptionResult};

pub const DEFAULT_LONG_AUDIO_THRESHOLD: f64 = 600.0; // 10 minutes

pub struct Job {
    pub id: u64,
    pub pcm: Vec<f32>,
    pub duration_secs: f64,
    pub lang: String,
    pub response_tx: oneshot::Sender<Result<TranscriptionResult, String>>,
    pub cancel_token: CancellationToken,
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Job {}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        // Preemptive Shortest Job First:
        // 1. Tracks <= 600s come before tracks > 600s
        let self_is_short = self.duration_secs <= DEFAULT_LONG_AUDIO_THRESHOLD;
        let other_is_short = other.duration_secs <= DEFAULT_LONG_AUDIO_THRESHOLD;

        match (self_is_short, other_is_short) {
            (true, false) => Ordering::Greater, // self has higher priority in BinaryHeap
            (false, true) => Ordering::Less,
            _ => {
                // Within same tier, shorter duration has higher priority
                other
                    .duration_secs
                    .partial_cmp(&self.duration_secs)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| other.id.cmp(&self.id))
            }
        }
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct RunningJobInfo {
    id: u64,
    duration_secs: f64,
    pcm: Vec<f32>,
    lang: String,
    cancel_token: CancellationToken,
}

pub struct PreemptiveQueue {
    heap: Arc<Mutex<BinaryHeap<Job>>>,
    active_job: Arc<Mutex<Option<RunningJobInfo>>>,
    notify: Arc<Notify>,
    next_id: AtomicU64,
    max_capacity: usize,
}

impl PreemptiveQueue {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            heap: Arc::new(Mutex::new(BinaryHeap::new())),
            active_job: Arc::new(Mutex::new(None)),
            notify: Arc::new(Notify::new()),
            next_id: AtomicU64::new(1),
            max_capacity,
        }
    }

    pub async fn submit(
        &self,
        pcm: Vec<f32>,
        duration_secs: f64,
        lang: String,
    ) -> Result<oneshot::Receiver<Result<TranscriptionResult, String>>, String> {
        let mut heap = self.heap.lock().await;

        if heap.len() >= self.max_capacity {
            return Err("Queue capacity exceeded (429 Too Many Requests)".to_string());
        }

        let id = self.next_id.fetch_add(1, AtomicOrdering::SeqCst);
        let (tx, rx) = oneshot::channel();
        let cancel_token = CancellationToken::new();

        let job = Job {
            id,
            pcm,
            duration_secs,
            lang,
            response_tx: tx,
            cancel_token,
        };

        // Preemption check: if incoming track is short (<= 10m) and running track is long (> 10m)
        if duration_secs <= DEFAULT_LONG_AUDIO_THRESHOLD {
            let active = self.active_job.lock().await;
            if let Some(ref running) = *active {
                if running.duration_secs > DEFAULT_LONG_AUDIO_THRESHOLD {
                    tracing::info!(
                        "⚡ Preempting long audio (id: {}, {:.1}s) for short audio (id: {}, {:.1}s)",
                        running.id,
                        running.duration_secs,
                        job.id,
                        job.duration_secs
                    );
                    running.cancel_token.cancel();
                }
            }
        }

        heap.push(job);
        self.notify.notify_one();

        Ok(rx)
    }

    pub async fn status(&self) -> serde_json::Value {
        let heap = self.heap.lock().await;
        let active = self.active_job.lock().await;

        serde_json::json!({
            "is_busy": active.is_some(),
            "queued_jobs": heap.len(),
            "active_job_id": active.as_ref().map(|j| j.id),
            "active_job_duration": active.as_ref().map(|j| round_2(j.duration_secs))
        })
    }
}

fn round_2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub fn start_worker_loop(
    queue: Arc<PreemptiveQueue>,
    transcriber: Arc<TranscriberService>,
    idle_ttl_secs: u64,
) {
    tokio::spawn(async move {
        loop {
            // Check idle eviction
            transcriber.evict_if_idle(idle_ttl_secs).await;

            let maybe_job = {
                let mut heap = queue.heap.lock().await;
                heap.pop()
            };

            let job = match maybe_job {
                Some(j) => j,
                None => {
                    // Wait for new work or timeout check
                    tokio::select! {
                        _ = queue.notify.notified() => continue,
                        _ = tokio::time::sleep(Duration::from_secs(30)) => continue,
                    }
                }
            };

            // Set as active job
            {
                let mut active = queue.active_job.lock().await;
                *active = Some(RunningJobInfo {
                    id: job.id,
                    duration_secs: job.duration_secs,
                    pcm: job.pcm.clone(),
                    lang: job.lang.clone(),
                    cancel_token: job.cancel_token.clone(),
                });
            }

            tracing::info!(
                "Processing job id: {} ({:.1}s, lang: {})",
                job.id,
                job.duration_secs,
                job.lang
            );

            let cancel_token = job.cancel_token.clone();
            let res = tokio::select! {
                r = transcriber.transcribe(&job.pcm, &job.lang) => Some(r),
                _ = cancel_token.cancelled() => None,
            };

            if let Some(r) = res {
                let _ = job.response_tx.send(r.map_err(|e| e.to_string()));
            } else {
                tracing::info!("Job id: {} was preempted. Re-queuing to heap.", job.id);
                let re_queued = Job {
                    id: job.id,
                    pcm: job.pcm,
                    duration_secs: job.duration_secs,
                    lang: job.lang,
                    response_tx: job.response_tx,
                    cancel_token: CancellationToken::new(),
                };
                let mut heap = queue.heap.lock().await;
                heap.push(re_queued);
            }

            // Clear active job
            {
                let mut active = queue.active_job.lock().await;
                *active = None;
            }
        }
    });
}
