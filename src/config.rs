//! docs/design.md §11: one block, and nothing that names a host, a budget or a secret is in code.
//!
//! `budget_ops` and `deadline_ms` are deliberately absent. They are the *game's*, they arrive on
//! every call, and putting them here would put a game's rules in a game-agnostic process.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Beside a Kalam replica. Fetches by hash, refuses a URL, serves `/play`.
    Replica,
    /// Beside Soma. Accepts URLs, mirrors what it verified, serves `/inspect` and `/validate`,
    /// and refuses `/play`.
    Admission,
}

impl Mode {
    fn parse(s: &str) -> Option<Mode> {
        match s {
            "replica" => Some(Mode::Replica),
            "admission" => Some(Mode::Admission),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Replica => "replica",
            Mode::Admission => "admission",
        }
    }
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
    pub adapter_threads: usize,
    pub max_in_flight: usize,
    /// Where the bytes are: a directory, an HTTP base URL, or S3/R2 signed with SigV4.
    pub store: StoreSpec,
    /// Empty on a replica, which is the property: only one instance can reach the internet.
    pub fetch_allow_hosts: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum StoreSpec {
    Dir(PathBuf),
    Http { base: String },
    /// devops/docs/deployment.md §8.1. The only spec that works across hosts, and therefore the only one a fleet
    /// can use: the admission instance and every replica must share one store, and there is no
    /// shared volume between machines.
    S3 { endpoint: String, bucket: String, region: String, access_key: String, secret_key: String },
}

fn env_or<T: std::str::FromStr>(key: &str, dflt: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}

impl Config {
    /// From the environment, with docs/design.md §11's provisional values as defaults.
    pub fn from_env() -> Result<Config, String> {
        let mode = Mode::parse(&std::env::var("AXON_MODE").unwrap_or_else(|_| "replica".into()))
            .ok_or("AXON_MODE must be 'replica' or 'admission'")?;

        // S3 first: a deployment that names a bucket means it, and falling back to a directory
        // because one variable was missing would give every replica its own empty store -- the
        // exact split-store failure §8.1 exists to prevent, and silent.
        let store = match (
            std::env::var("AXON_STORE_S3_BUCKET"),
            std::env::var("AXON_STORE_DIR"),
            std::env::var("AXON_STORE_URL"),
        ) {
            (Ok(bucket), _, _) => {
                let need = |k: &str| {
                    std::env::var(k).map_err(|_| format!("{k} is required when AXON_STORE_S3_BUCKET is set"))
                };
                StoreSpec::S3 {
                    endpoint: need("AXON_STORE_S3_ENDPOINT")?.trim_end_matches('/').to_string(),
                    bucket,
                    // R2 signs against "auto"; MinIO and AWS want a real region.
                    region: std::env::var("AXON_STORE_S3_REGION").unwrap_or_else(|_| "auto".into()),
                    access_key: need("AXON_STORE_S3_ACCESS_KEY")?,
                    secret_key: need("AXON_STORE_S3_SECRET_KEY")?,
                }
            }
            (_, Ok(d), _) => StoreSpec::Dir(PathBuf::from(d)),
            (_, _, Ok(u)) => StoreSpec::Http { base: u.trim_end_matches('/').to_string() },
            _ => return Err("set AXON_STORE_S3_BUCKET (with endpoint and keys), AXON_STORE_DIR, or AXON_STORE_URL".into()),
        };

        // The allowlist is empty on a replica whatever the environment says. A replica that could
        // be told to fetch from the internet would be one compromised workflow away from pulling
        // arbitrary bytes, and it has no reason to: everything it needs is in the store, by hash.
        let fetch_allow_hosts = match mode {
            Mode::Replica => Vec::new(),
            Mode::Admission => std::env::var("AXON_FETCH_ALLOW_HOSTS")
                .unwrap_or_else(|_| "github.com,objects.githubusercontent.com".into())
                .split(',')
                .map(|h| h.trim().to_string())
                .filter(|h| !h.is_empty())
                .collect(),
        };

        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        Ok(Config {
            mode,
            bind: std::env::var("AXON_BIND").unwrap_or_else(|_| "127.0.0.1:9090".into()),
            auth_token: std::env::var("AXON_AUTH_TOKEN").ok().filter(|t| !t.is_empty()),
            memory_budget_bytes: env_or("AXON_MEMORY_BUDGET_BYTES", 6 * 1024 * 1024 * 1024),
            max_weights_bytes: env_or("AXON_MAX_WEIGHTS_BYTES", 96 * 1024 * 1024),
            max_adapter_bytes: env_or("AXON_MAX_ADAPTER_BYTES", 4 * 1024 * 1024),
            default_idle_ttl_s: env_or("AXON_IDLE_TTL_S", 900),
            threads: env_or("AXON_THREADS", cores.saturating_sub(1).max(1)),
            adapter_threads: env_or("AXON_ADAPTER_THREADS", cores),
            max_in_flight: env_or(
                "AXON_MAX_IN_FLIGHT",
                if mode == Mode::Admission { 4 } else { 1 },
            ),
            store,
            fetch_allow_hosts,
        })
    }
}
