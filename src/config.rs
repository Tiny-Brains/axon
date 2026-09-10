//! The environment block. `budget_ops` and `deadline_ms` are deliberately absent: they are the
//! game's, they arrive on every call, and putting them here would put a game's rules in a
//! game-agnostic process.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Beside a Kalam replica. Fetches by hash, refuses a URL, serves `/play`.
    Replica,
    /// Beside Soma. Accepts URLs, mirrors what it verified, serves `/inspect` and `/validate`.
    Admission,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Replica => "replica",
            Mode::Admission => "admission",
        }
    }

    fn parse(s: &str) -> Option<Mode> {
        match s {
            "replica" => Some(Mode::Replica),
            "admission" => Some(Mode::Admission),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StoreSpec {
    Dir(PathBuf),
    Http {
        base: String,
    },
    /// The only spec that works across hosts, and therefore the only one a fleet can use.
    S3 {
        endpoint: String,
        bucket: String,
        region: String,
        access_key: String,
        secret_key: String,
    },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub mode: Mode,
    pub bind: String,
    pub auth_token: Option<String>,
    pub memory_budget_bytes: u64,
    pub max_weights_bytes: u64,
    pub max_adapter_bytes: u64,
    pub default_idle_ttl_s: u64,
    pub threads: usize,
    pub max_in_flight: usize,
    pub store: StoreSpec,
    /// Empty on a replica, which is the property: only one instance can reach the internet.
    pub fetch_allow_hosts: Vec<String>,
}

impl Config {
    pub fn from_env() -> Result<Config, String> {
        let mode = Mode::parse(&var("AXON_MODE").unwrap_or_else(|| "replica".into()))
            .ok_or("AXON_MODE must be 'replica' or 'admission'")?;
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        Ok(Config {
            mode,
            store: store_spec()?,
            fetch_allow_hosts: allow_hosts(mode),
            bind: var("AXON_BIND").unwrap_or_else(|| "127.0.0.1:9090".into()),
            auth_token: var("AXON_AUTH_TOKEN"),
            memory_budget_bytes: env_or("AXON_MEMORY_BUDGET_BYTES", 6 * 1024 * 1024 * 1024),
            max_weights_bytes: env_or("AXON_MAX_WEIGHTS_BYTES", 96 * 1024 * 1024),
            max_adapter_bytes: env_or("AXON_MAX_ADAPTER_BYTES", 4 * 1024 * 1024),
            default_idle_ttl_s: env_or("AXON_IDLE_TTL_S", 900),
            threads: env_or("AXON_THREADS", cores.saturating_sub(1).max(1)),
            max_in_flight: env_or(
                "AXON_MAX_IN_FLIGHT",
                if mode == Mode::Admission { 4 } else { 1 },
            ),
        })
    }
}

/// S3 first: a deployment that names a bucket means it, and falling back to a directory because one
/// variable was missing would give every replica its own empty store — the split-store failure,
/// silently.
fn store_spec() -> Result<StoreSpec, String> {
    if let Some(bucket) = var("AXON_STORE_S3_BUCKET") {
        let need =
            |k: &str| var(k).ok_or(format!("{k} is required when AXON_STORE_S3_BUCKET is set"));
        return Ok(StoreSpec::S3 {
            endpoint: need("AXON_STORE_S3_ENDPOINT")?.trim_end_matches('/').to_string(),
            bucket,
            // R2 signs against "auto"; MinIO and AWS want a real region.
            region: var("AXON_STORE_S3_REGION").unwrap_or_else(|| "auto".into()),
            access_key: need("AXON_STORE_S3_ACCESS_KEY")?,
            secret_key: need("AXON_STORE_S3_SECRET_KEY")?,
        });
    }
    if let Some(dir) = var("AXON_STORE_DIR") {
        return Ok(StoreSpec::Dir(PathBuf::from(dir)));
    }
    if let Some(url) = var("AXON_STORE_URL") {
        return Ok(StoreSpec::Http { base: url.trim_end_matches('/').to_string() });
    }
    Err("set AXON_STORE_S3_BUCKET (with endpoint and keys), AXON_STORE_DIR, or AXON_STORE_URL"
        .into())
}

/// Empty on a replica whatever the environment says: a replica that could be told to fetch from the
/// internet would be one compromised workflow away from pulling arbitrary bytes, and everything it
/// needs is already in the store, by hash.
fn allow_hosts(mode: Mode) -> Vec<String> {
    if mode == Mode::Replica {
        return Vec::new();
    }
    var("AXON_FETCH_ALLOW_HOSTS")
        .unwrap_or_else(|| "github.com,objects.githubusercontent.com".into())
        .split(',')
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .collect()
}

fn var(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn env_or<T: std::str::FromStr>(key: &str, dflt: T) -> T {
    var(key).and_then(|v| v.parse().ok()).unwrap_or(dflt)
}
