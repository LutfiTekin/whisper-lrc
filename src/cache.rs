use moka::future::Cache;
use std::time::Duration;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    pub audio_hash: String,
    pub lang: String,
    pub isolate: bool,
    pub prompt: Option<String>,
}

#[derive(Clone)]
pub struct LrcCache {
    inner: Cache<CacheKey, String>,
}

impl LrcCache {
    pub fn new(max_capacity: u64, ttl_secs: u64) -> Self {
        let builder = Cache::builder().max_capacity(max_capacity);
        let cache = if ttl_secs > 0 {
            builder.time_to_idle(Duration::from_secs(ttl_secs)).build()
        } else {
            builder.build()
        };

        Self { inner: cache }
    }

    pub async fn get(&self, key: &CacheKey) -> Option<String> {
        self.inner.get(key).await
    }

    pub async fn insert(&self, key: CacheKey, lrc: String) {
        self.inner.insert(key, lrc).await;
    }

    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }
}
