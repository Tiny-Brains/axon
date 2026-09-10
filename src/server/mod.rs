//! The service — docs/design.md §3. Six calls, JSON in, JSON out.
//!
//! Blocking and threaded rather than async: inference and adapter evaluation are CPU-bound, ORT is
//! blocking, and a replica's loader serves one caller.

mod admission;
mod http;
mod play;

pub use http::serve;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::api::*;
use crate::config::{Config, Mode, StoreSpec};
use crate::dialect::{self, Adapter};
use crate::model::{Graph, GraphError};
use crate::residency::Residency;
use crate::store::{self, DirStore, HttpStore, Kind, S3Store, Store, StoreError};

/// Why a model could not be made resident, in the two classes Kalam's barrier branches on: a
/// `loader` fault releases the row with no attempt spent, a `model` fault fails it with the seat.
struct Refusal {
    reason: &'static str,
    fault: &'static str,
    detail: Option<String>,
}

impl Refusal {
    fn loader(reason: &'static str, detail: String) -> Refusal {
        Refusal { reason, fault: "loader", detail: Some(detail) }
    }

    fn model(reason: &'static str, detail: String) -> Refusal {
        Refusal { reason, fault: "model", detail: Some(detail) }
    }

    fn bare(reason: &'static str, fault: &'static str) -> Refusal {
        Refusal { reason, fault, detail: None }
    }
}

pub struct Axon {
    pub cfg: Config,
    pub store: Box<dyn Store>,
    pub residency: Mutex<Residency>,
}

impl Axon {
    pub fn new(cfg: Config) -> Axon {
        let store: Box<dyn Store> = match &cfg.store {
            StoreSpec::Dir(p) => Box::new(DirStore::new(p.clone())),
            StoreSpec::Http { base } => Box::new(HttpStore::new(base.clone())),
            StoreSpec::S3 { endpoint, bucket, region, access_key, secret_key } => {
                Box::new(S3Store::new(
                    endpoint.clone(),
                    bucket.clone(),
                    region.clone(),
                    access_key.clone(),
                    secret_key.clone(),
                ))
            }
        };
        let residency = Mutex::new(Residency::new(cfg.memory_budget_bytes));
        Axon { cfg, store, residency }
    }

    pub fn load(&self, req: LoadRequest) -> LoadReply {
        let ttl = Duration::from_secs(req.idle_ttl_s.unwrap_or(self.cfg.default_idle_ttl_s));
        let models = req
            .models
            .iter()
            .map(|m| {
                let (state, reason, fault, detail) = match self.acquire(m, ttl) {
                    Ok(()) => ("resident", None, None, None),
                    Err(r) => ("refused", Some(r.reason), Some(r.fault), r.detail),
                };
                ModelState {
                    weights_hash: m.weights_hash.clone(),
                    adapter_hash: m.adapter_hash.clone(),
                    state,
                    reason,
                    fault,
                    detail,
                }
            })
            .collect();
        LoadReply {
            models,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        }
    }

    fn acquire(&self, m: &ModelRef, ttl: Duration) -> Result<(), Refusal> {
        // A replica cannot be told where to fetch from.
        if (m.weights_url.is_some() || m.adapter_url.is_some()) && self.cfg.mode == Mode::Replica {
            return Err(Refusal::bare("URL_NOT_ACCEPTED", "model"));
        }
        if store::key(Kind::Weights, &m.weights_hash).is_none()
            || store::key(Kind::Adapter, &m.adapter_hash).is_none()
        {
            return Err(Refusal::model("HASH_MISMATCH", "not a sha256 hash".into()));
        }

        // Already held? A hold is added and nothing is fetched, whatever URL was given.
        {
            let mut r = self.residency.lock().unwrap();
            if r.is_resident(&m.weights_hash, &m.adapter_hash) {
                r.hold(&m.weights_hash, &m.adapter_hash);
                return Ok(());
            }
        }

        let weights = self.fetch(Kind::Weights, &m.weights_hash, m.weights_url.as_deref())?;
        let adapter_bytes = self.fetch(Kind::Adapter, &m.adapter_hash, m.adapter_url.as_deref())?;

        if adapter_bytes.len() as u64 > self.cfg.max_adapter_bytes {
            let n = adapter_bytes.len();
            return Err(Refusal::model("TOO_LARGE", format!("adapter is {n} bytes")));
        }
        let adapter = Adapter::parse(&adapter_bytes)
            .map(Arc::new)
            .map_err(|e| Refusal::model("ADAPTER_INVALID", e.to_string()))?;

        // Memory is checked before a session is built, so a refusal costs a fetch and not a
        // session. Nothing held is ever evicted to make the room.
        if !self.residency.lock().unwrap().make_room_for(weights.len() as u64) {
            return Err(Refusal::bare("MEMORY", "loader"));
        }

        let graph = Graph::build(&weights, self.cfg.threads, self.cfg.max_weights_bytes)
            .map(Arc::new)
            .map_err(|e| match e {
                GraphError::TooLarge { bytes, limit } => {
                    Refusal::model("TOO_LARGE", format!("{bytes} bytes over {limit}"))
                }
                GraphError::Invalid(d) => Refusal::model("GRAPH_INVALID", d),
            })?;

        // Admission mirrors what it verified inside `/load`, so there is no state in which a
        // version is admitted and its bytes are not in the store.
        if self.cfg.mode == Mode::Admission {
            for (kind, hash, bytes) in [
                (Kind::Weights, &m.weights_hash, &weights),
                (Kind::Adapter, &m.adapter_hash, &adapter_bytes),
            ] {
                if !self.store.has(kind, hash) {
                    self.store
                        .put(kind, hash, bytes)
                        .map_err(|e| Refusal::loader("STORE_UNAVAILABLE", format!("{e:?}")))?;
                }
            }
        }

        let mut r = self.residency.lock().unwrap();
        r.insert(&m.weights_hash, graph, &m.adapter_hash, adapter, ttl);
        r.hold(&m.weights_hash, &m.adapter_hash);
        Ok(())
    }

    /// By hash from the store; by URL only on admission, and only to an allowlisted host. The bytes
    /// are hashed either way, and a mismatch is the model's fault: on admission the release does not
    /// match what was declared, on a replica the store is corrupt, and either way the row can never
    /// be played.
    fn fetch(&self, kind: Kind, hash: &str, url: Option<&str>) -> Result<Vec<u8>, Refusal> {
        let bytes = match url {
            Some(u) if self.cfg.mode == Mode::Admission => self.fetch_url(u)?,
            _ => self.store.get(kind, hash).map_err(|e| match e {
                StoreError::NotFound => Refusal::loader("FETCH_FAILED", "not in the store".into()),
                StoreError::Unavailable(e) => Refusal::loader("STORE_UNAVAILABLE", e),
            })?,
        };
        let actual = store::digest(&bytes);
        if actual != hash {
            return Err(Refusal::model("HASH_MISMATCH", format!("got {actual}, declared {hash}")));
        }
        Ok(bytes)
    }

    fn fetch_url(&self, url: &str) -> Result<Vec<u8>, Refusal> {
        let host = url.split("://").nth(1).and_then(|r| r.split('/').next()).unwrap_or("");
        if !self.cfg.fetch_allow_hosts.iter().any(|h| h == host) {
            let d = format!("host '{host}' is not allowlisted");
            return Err(Refusal::loader("FETCH_FAILED", d));
        }
        match ureq::get(url).call() {
            Ok(mut resp) => {
                store::read_body(&mut resp).map_err(|e| Refusal::loader("FETCH_FAILED", e))
            }
            // A 4xx is the release's fault: a competitor who forgot to attach `adapter.json` is
            // told so rather than given three retries and a TIMED_OUT. Anything else stays ours.
            Err(ureq::Error::StatusCode(code)) if (400..500).contains(&code) => {
                Err(Refusal::model("ASSET_MISSING", format!("{url} answered {code}")))
            }
            Err(e) => Err(Refusal::loader("FETCH_FAILED", e.to_string())),
        }
    }

    pub fn unload(&self, req: UnloadRequest) -> UnloadReply {
        let mut r = self.residency.lock().unwrap();
        UnloadReply {
            models: req
                .models
                .iter()
                .map(|m| UnloadState {
                    weights_hash: m.weights_hash.clone(),
                    adapter_hash: m.adapter_hash.clone(),
                    state: if r.release(&m.weights_hash, &m.adapter_hash) {
                        "released"
                    } else {
                        "not_held"
                    },
                })
                .collect(),
        }
    }

    pub fn resident(&self) -> ResidentReply {
        let r = self.residency.lock().unwrap();
        ResidentReply {
            weights: r.resident_weights(),
            // `/load` is synchronous, so a model is resident when it answers or it was refused.
            // The field stays because Kalam's claim excludes it, and that must keep working when
            // something does load in the background.
            loading: Vec::new(),
            adapters: r.resident_adapters(),
            memory_bytes: r.used_bytes(),
            memory_budget_bytes: r.budget_bytes,
        }
    }
}
