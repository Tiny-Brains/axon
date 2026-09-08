//! The seam — layer 04 §3. Six calls, JSON in, JSON out, on loopback.
//!
//! Blocking and threaded rather than async, on purpose: inference and adapter evaluation are
//! CPU-bound, ORT is blocking, and a replica's loader serves one caller. An async runtime would
//! buy nothing here and would have to be bridged back to blocking at every interesting line.

use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::api::*;
use crate::config::{Config, Mode, StoreSpec};
use crate::dialect::{self, Adapter, Fault};
use crate::model::{Graph, GraphError};
use crate::onnx_meta;
use crate::residency::Residency;
use crate::store::{self, DirStore, HttpStore, Kind, S3Store, Store, StoreError};

/// Why a model could not be made resident, in the two-class split layer 03's barrier branches on
/// (layer 04 §6): a `loader` fault releases the row with no attempt spent, a `model` fault fails
/// it at once with the seat.
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
            StoreSpec::S3 { endpoint, bucket, region, access_key, secret_key } => Box::new(
                S3Store::new(
                    endpoint.clone(),
                    bucket.clone(),
                    region.clone(),
                    access_key.clone(),
                    secret_key.clone(),
                ),
            ),
        };
        let residency = Mutex::new(Residency::new(cfg.memory_budget_bytes));
        Axon { cfg, store, residency }
    }

    // ---------------------------------------------------------------- /load

    pub fn load(&self, req: LoadRequest) -> LoadReply {
        let ttl = Duration::from_secs(req.idle_ttl_s.unwrap_or(self.cfg.default_idle_ttl_s));
        let models = req.models.iter().map(|m| self.load_one(m, ttl)).collect();
        LoadReply {
            models,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        }
    }

    fn load_one(&self, m: &ModelRef, ttl: Duration) -> ModelState {
        let refused = |reason, fault, detail: Option<String>| ModelState {
            weights_hash: m.weights_hash.clone(),
            adapter_hash: m.adapter_hash.clone(),
            state: "refused",
            reason: Some(reason),
            fault: Some(fault),
            detail,
        };
        let resident = || ModelState {
            weights_hash: m.weights_hash.clone(),
            adapter_hash: m.adapter_hash.clone(),
            state: "resident",
            reason: None,
            fault: None,
            detail: None,
        };

        // A replica cannot be told where to fetch from.
        if (m.weights_url.is_some() || m.adapter_url.is_some()) && self.cfg.mode == Mode::Replica {
            return refused("URL_NOT_ACCEPTED", "model", None);
        }
        if store::key(Kind::Weights, &m.weights_hash).is_none()
            || store::key(Kind::Adapter, &m.adapter_hash).is_none()
        {
            return refused("HASH_MISMATCH", "model", Some("not a sha256 hash".into()));
        }

        // Already held? A hold is added and nothing is fetched, whatever URL was given.
        {
            let mut r = self.residency.lock().unwrap();
            if r.is_resident(&m.weights_hash, &m.adapter_hash) {
                r.hold(&m.weights_hash, &m.adapter_hash);
                return resident();
            }
        }

        let weights = match self.fetch(Kind::Weights, &m.weights_hash, m.weights_url.as_deref()) {
            Ok(b) => b,
            Err(Refusal { reason, fault, detail }) => return refused(reason, fault, detail),
        };
        let adapter_bytes =
            match self.fetch(Kind::Adapter, &m.adapter_hash, m.adapter_url.as_deref()) {
                Ok(b) => b,
                Err(Refusal { reason, fault, detail }) => return refused(reason, fault, detail),
            };

        if adapter_bytes.len() as u64 > self.cfg.max_adapter_bytes {
            return refused("TOO_LARGE", "model", Some(format!("adapter is {} bytes", adapter_bytes.len())));
        }
        let adapter = match Adapter::parse(&adapter_bytes) {
            Ok(a) => Arc::new(a),
            Err(e) => return refused("ADAPTER_INVALID", "model", Some(e.to_string())),
        };

        // Memory is checked before a session is built, so a refusal costs a fetch and not a
        // session. Nothing held is ever evicted to make the room.
        {
            let mut r = self.residency.lock().unwrap();
            if !r.make_room_for(weights.len() as u64) {
                return refused("MEMORY", "loader", None);
            }
        }

        let graph = match Graph::build(&weights, self.cfg.threads, self.cfg.max_weights_bytes) {
            Ok(g) => Arc::new(g),
            Err(GraphError::TooLarge { bytes, limit }) => {
                return refused("TOO_LARGE", "model", Some(format!("{bytes} bytes over {limit}")))
            }
            Err(GraphError::Invalid(d)) => return refused("GRAPH_INVALID", "model", Some(d)),
        };

        // Admission mirrors what it verified, inside `/load`, so there is no state in which a
        // version is admitted and its bytes are not in the store.
        if self.cfg.mode == Mode::Admission {
            for (kind, hash, bytes) in [
                (Kind::Weights, &m.weights_hash, &weights),
                (Kind::Adapter, &m.adapter_hash, &adapter_bytes),
            ] {
                if !self.store.has(kind, hash) {
                    if let Err(e) = self.store.put(kind, hash, bytes) {
                        return refused("STORE_UNAVAILABLE", "loader", Some(format!("{e:?}")));
                    }
                }
            }
        }

        let mut r = self.residency.lock().unwrap();
        r.insert(&m.weights_hash, graph, &m.adapter_hash, adapter, ttl);
        r.hold(&m.weights_hash, &m.adapter_hash);
        resident()
    }

    /// By hash from the store; by URL only on the admission instance, and only to an allowlisted
    /// host. The bytes are hashed either way, and a mismatch is the model's fault: on admission it
    /// means the competitor's release does not match what they declared, and on a replica it means
    /// the store is corrupt — either way the row can never be played, so failing it with a named
    /// reason beats looping it through refusals to the same end.
    fn fetch(&self, kind: Kind, hash: &str, url: Option<&str>) -> Result<Vec<u8>, Refusal> {
        let got: Result<Vec<u8>, String> = match url {
            Some(u) if self.cfg.mode == Mode::Admission => {
                let host = u
                    .split("://")
                    .nth(1)
                    .and_then(|r| r.split('/').next())
                    .unwrap_or("");
                if !self.cfg.fetch_allow_hosts.iter().any(|h| h == host) {
                    return Err(Refusal::loader(
                        "FETCH_FAILED",
                        format!("host '{host}' is not allowlisted"),
                    ));
                }
                match ureq::get(u).call() {
                    Ok(mut resp) => {
                        let mut buf = Vec::new();
                        resp.body_mut()
                            .as_reader()
                            .read_to_end(&mut buf)
                            .map(|_| buf)
                            .map_err(|e| e.to_string())
                    }
                    // A 4xx is the RELEASE's fault, not the platform's -- layer 08 §13's second
                    // ask. Classing it with the 5xxs would give a competitor who forgot to attach
                    // `adapter.json` three silent retries and then TIMED_OUT, which is the least
                    // actionable message on the platform. Anything else -- a 5xx, a timeout, a
                    // connection failure -- stays FETCH_FAILED and stays ours.
                    Err(ureq::Error::StatusCode(code)) if (400..500).contains(&code) => {
                        return Err(Refusal::model(
                            "ASSET_MISSING",
                            format!("{u} answered {code}"),
                        ))
                    }
                    Err(e) => Err(e.to_string()),
                }
            }
            _ => match self.store.get(kind, hash) {
                Ok(b) => Ok(b),
                Err(StoreError::NotFound) => Err("not in the store".to_string()),
                Err(StoreError::Unavailable(e)) => {
                    return Err(Refusal::loader("STORE_UNAVAILABLE", e))
                }
            },
        };

        let bytes = got.map_err(|d| Refusal::loader("FETCH_FAILED", d))?;
        let actual = store::digest(&bytes);
        if actual != hash {
            return Err(Refusal::model(
                "HASH_MISMATCH",
                format!("got {actual}, declared {hash}"),
            ));
        }
        Ok(bytes)
    }

    // ---------------------------------------------------------------- /play

    /// One call per turn for a whole wave.
    ///
    /// Three phases, and the middle one is the economics the design rests on:
    ///
    /// 1. every row's `in` program, producing its feeds
    /// 2. **rows grouped by weights hash, and within a group rows whose feeds agree in shape
    ///    stacked into one inference**
    /// 3. every row's `out` program, over its own slice of the answer
    pub fn play(&self, req: PlayRequest) -> PlayReply {
        let deadline = Instant::now() + Duration::from_millis(req.deadline_ms);
        let mut rows: Vec<PlayRowReply> = Vec::with_capacity(req.rows.len());
        let mut prepared: Vec<Option<Prepared>> = Vec::with_capacity(req.rows.len());

        // ---- 1. the adapters in
        for row in &req.rows {
            let started = Instant::now();
            let (graph, adapter) = {
                let mut res = self.residency.lock().unwrap();
                res.touch(&row.weights_hash);
                match (res.graph(&row.weights_hash), res.adapter(&row.adapter_hash)) {
                    (Some(g), Some(a)) => (g, a),
                    _ => {
                        rows.push(finish_row(
                            PlayRowReply { error: Some("NOT_RESIDENT"), ..Default::default() },
                            row,
                            started,
                        ));
                        prepared.push(None);
                        continue;
                    }
                }
            };
            if Instant::now() >= deadline {
                rows.push(finish_row(
                    PlayRowReply { error: Some("TIMED_OUT"), ..Default::default() },
                    row,
                    started,
                ));
                prepared.push(None);
                continue;
            }
            let (feeds, ops_in) = adapter.run_in(&row.observation, req.budget_ops);
            match feeds {
                Ok(feeds) => {
                    rows.push(PlayRowReply::default());
                    prepared.push(Some(Prepared { graph, adapter, feeds, ops_in, started }));
                }
                Err(e) => {
                    rows.push(finish_row(fault_row(e, ops_in), row, started));
                    prepared.push(None);
                }
            }
        }

        // ---- 2. one inference per group whose shapes agree
        let mut outputs: Vec<Option<Result<crate::dialect::Ports, String>>> =
            (0..prepared.len()).map(|_| None).collect();
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, p) in prepared.iter().enumerate() {
            let Some(p) = p else { continue };
            // A graph that does not declare a dynamic batch is its own group of one: shape
            // agreement between rows says nothing about whether the graph will accept them
            // stacked.
            let key = if p.graph.batchable() {
                shape_key(&req.rows[i].weights_hash, &p.feeds)
            } else {
                format!("{}#row{i}", req.rows[i].weights_hash)
            };
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, idx)) => idx.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        for (_, idx) in &groups {
            let graph = prepared[idx[0]].as_ref().unwrap().graph.clone();
            let feeds: Vec<&crate::dialect::Ports> =
                idx.iter().map(|&i| &prepared[i].as_ref().unwrap().feeds).collect();
            let answers = run_group(&graph, &feeds, deadline);
            match answers {
                Ok(per_row) => {
                    for (&i, a) in idx.iter().zip(per_row) {
                        outputs[i] = Some(Ok(a));
                    }
                }
                Err(e) => {
                    for &i in idx {
                        outputs[i] = Some(Err(e.clone()));
                    }
                }
            }
        }

        // ---- 3. the adapters out
        for (i, p) in prepared.into_iter().enumerate() {
            let Some(p) = p else { continue };
            let row = &req.rows[i];
            let reply = match outputs[i].take() {
                Some(Ok(out)) => {
                    let (action, ops_out) = p.adapter.run_out(out, &row.observation, req.budget_ops);
                    match action {
                        Ok(a) => PlayRowReply {
                            action: Some(a),
                            ops: p.ops_in + ops_out,
                            ..Default::default()
                        },
                        Err(e) => fault_row(e, p.ops_in + ops_out),
                    }
                }
                Some(Err(e)) if is_timeout(&e) => {
                    PlayRowReply { error: Some("TIMED_OUT"), ops: p.ops_in, ..Default::default() }
                }
                Some(Err(e)) => PlayRowReply {
                    error: Some("INFERENCE_FAILED"),
                    detail: Some(e),
                    ops: p.ops_in,
                    ..Default::default()
                },
                None => PlayRowReply {
                    error: Some("INFERENCE_FAILED"),
                    detail: Some("no group ran this row".into()),
                    ops: p.ops_in,
                    ..Default::default()
                },
            };
            rows[i] = finish_row(reply, row, p.started);
        }

        PlayReply {
            rows,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        }
    }

    // ---------------------------------------------------------------- /unload, /resident

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
            // Nothing loads in the background yet: `/load` is synchronous, so a model is resident
            // when it answers or it was refused. The field is here because layer 03 excludes it
            // from what it hands the claim, and that must keep working when it is populated.
            loading: Vec::new(),
            adapters: r.resident_adapters(),
            memory_bytes: r.used_bytes(),
            memory_budget_bytes: r.budget_bytes,
        }
    }

    // ---------------------------------------------------------------- /inspect, /validate

    pub fn inspect(&self, req: InspectRequest) -> Result<InspectReply, ErrorReply> {
        let (graph, weights_raw) = {
            let r = self.residency.lock().unwrap();
            match r.graph(&req.weights_hash) {
                Some(g) => {
                    let n = g.bytes_resident;
                    (g, n)
                }
                None => return Err(ErrorReply { error: "NOT_RESIDENT", detail: None }),
            }
        };
        let adapter_bytes = self
            .store
            .get(Kind::Adapter, &req.adapter_hash)
            .map_err(|_| ErrorReply { error: "NOT_RESIDENT", detail: Some("adapter".into()) })?;

        let (s, w, a) = onnx_meta::size_metric(&graph.facts.initializer_bytes, &adapter_bytes);
        Ok(InspectReply {
            params: graph.facts.params,
            opset: graph.facts.opset,
            ops: graph.facts.ops.iter().cloned().collect(),
            // Reported, not judged. The allowlist is platform policy and lives in layer 08.
            unsupported_ops: Vec::new(),
            size_metric_bytes: s,
            weights_zstd_bytes: w,
            adapter_zstd_bytes: a,
            weights_raw_bytes: weights_raw,
            adapter_raw_bytes: adapter_bytes.len() as u64,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            // The bytes as fetched, not a re-serialisation -- layer 08 §13. They parsed as JSON
            // in `/load`, so they are valid UTF-8 and this cannot lose anything; the error arm is
            // unreachable rather than lossy, and says so.
            adapter: String::from_utf8(adapter_bytes.clone()).map_err(|_| ErrorReply {
                error: "ADAPTER_INVALID",
                detail: Some("the adapter is not valid UTF-8".into()),
            })?,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        })
    }

    pub fn validate(&self, req: ValidateRequest) -> ValidateReply {
        let mut reply = ValidateReply {
            ok: true,
            reason: None,
            detail: None,
            failing_case: None,
            over_budget: None,
            cases: Vec::new(),
            ops_max: 0,
            flops_max: 0.0,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        };
        let fail = |mut r: ValidateReply, reason, detail: String, case| {
            r.ok = false;
            r.reason = Some(reason);
            r.detail = Some(detail);
            r.failing_case = case;
            r.over_budget = Some(reason == "ADAPTER_FAILED");
            r
        };

        let (graph, adapter) = {
            let r = self.residency.lock().unwrap();
            match (r.graph(&req.weights_hash), r.adapter(&req.adapter_hash)) {
                (Some(g), Some(a)) => (g, a),
                _ => return fail(reply, "NOT_RESIDENT", "load it first".into(), None),
            }
        };
        if req.observations.is_empty() {
            return fail(reply, "ADAPTER_INVALID", "no reference observations to validate against".into(), None);
        }

        for (i, obs) in req.observations.iter().enumerate() {
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_millis(req.deadline_ms);

            let (feeds, ops_in) = adapter.run_in(obs, req.budget_ops);
            let feeds = match feeds {
                Ok(f) => f,
                Err(e) => {
                    let reason = if e.over_budget() { "ADAPTER_FAILED" } else { "ADAPTER_INVALID" };
                    return fail(reply, reason, e.to_string(), Some(i));
                }
            };

            // Every declared input must be fed, by name, and nothing else may be.
            let want = graph.input_names();
            for name in &want {
                if !feeds.iter().any(|(k, _)| &**k == *name) {
                    return fail(
                        reply,
                        "SHAPE_MISMATCH",
                        format!("the adapter produced no tensor for input '{name}'"),
                        Some(i),
                    );
                }
            }
            let fed: Vec<FedPort> = feeds
                .iter()
                .map(|(k, t)| FedPort {
                    name: k.to_string(),
                    dtype: t.dtype.name().to_string(),
                    shape: t.shape.clone(),
                })
                .collect();

            let outputs = match graph.run(&feeds, deadline) {
                Ok(o) => o,
                Err(e) => return fail(reply, "SHAPE_MISMATCH", e, Some(i)),
            };

            let (action, ops_out) = adapter.run_out(outputs, obs, req.budget_ops);
            let action = match action {
                Ok(a) => a,
                Err(e) => {
                    let reason = if e.over_budget() { "ADAPTER_FAILED" } else { "ADAPTER_INVALID" };
                    return fail(reply, reason, e.to_string(), Some(i));
                }
            };

            // FLOPs at the shapes the adapter actually produced. A declared input shape is a
            // claim; what is fed is a fact, and a graph whose real input is eight times the
            // declared one would otherwise pass a cap it does not respect.
            let flops = estimate_flops(&graph, &feeds);
            reply.ops_max = reply.ops_max.max(ops_in.max(ops_out));
            reply.flops_max = reply.flops_max.max(flops);
            reply.cases.push(ValidateCase {
                ops_in,
                ops_out,
                elapsed_ms: t0.elapsed().as_millis() as u64,
                inputs: fed,
                flops,
                action_shape: describe(&action),
            });
        }
        reply
    }
}

struct Prepared {
    graph: Arc<Graph>,
    adapter: Arc<Adapter>,
    feeds: crate::dialect::Ports,
    ops_in: u64,
    started: Instant,
}

fn finish_row(mut r: PlayRowReply, row: &PlayRow, started: Instant) -> PlayRowReply {
    r.elapsed_ms = started.elapsed().as_millis() as u64;
    r.r#ref = row.r#ref.clone();
    r
}

fn is_timeout(e: &str) -> bool {
    e == "TIMED_OUT" || e.contains("Terminate") || e.contains("terminated")
}

/// Two rows batch together when they name the same weights and their feeds agree in name, dtype
/// and shape. Shape included: a graph with a dynamic leading dimension is what makes batching
/// possible, and ragged per-seat inputs -- an ant count, say -- are what make it impossible. The
/// competitor decides which of those they submitted; the loader only notices.
fn shape_key(weights_hash: &str, feeds: &crate::dialect::Ports) -> String {
    let mut k = String::from(weights_hash);
    for (name, t) in feeds {
        k.push('|');
        k.push_str(name);
        k.push(':');
        k.push_str(t.dtype.name());
        for d in &t.shape {
            k.push_str(&format!(",{d}"));
        }
    }
    k
}

/// Run one group. A group of one is an ordinary run; a group of several is stacked along a new
/// leading axis, run once, and split back — which is what "one inference per distinct model per
/// turn" means in practice.
fn run_group(
    graph: &Graph,
    feeds: &[&crate::dialect::Ports],
    deadline: Instant,
) -> Result<Vec<crate::dialect::Ports>, String> {
    use crate::dialect::Tensor;

    if feeds.len() == 1 {
        return graph.run(feeds[0], deadline).map(|o| vec![o]);
    }
    let n = feeds.len();
    let mut stacked: crate::dialect::Ports = Vec::with_capacity(feeds[0].len());
    for (j, (name, first)) in feeds[0].iter().enumerate() {
        // The leading axis is the batch. A tensor already shaped [1, ...] -- which is what an
        // adapter writes for a graph declaring a batch dimension -- is concatenated along it
        // rather than gaining a second one.
        let batched_first = first.rank() > 0 && first.shape[0] == 1;
        let mut shape = first.shape.clone();
        if batched_first {
            shape[0] = n;
        } else {
            shape.insert(0, n);
        }
        let mut data = Vec::with_capacity(first.len() * n);
        for f in feeds {
            data.extend_from_slice(&f[j].1.data);
        }
        stacked.push((name.clone(), Arc::new(Tensor::new(first.dtype, shape, data))));
    }

    let out = graph.run(&stacked, deadline)?;

    let mut per_row: Vec<crate::dialect::Ports> = (0..n).map(|_| Vec::new()).collect();
    for (name, t) in out {
        if t.rank() == 0 || t.shape[0] != n {
            return Err(format!(
                "output '{name}' has a leading dimension of {:?}, not the batch of {n}; \
                 the graph does not batch",
                t.shape.first()
            ));
        }
        let per: usize = t.shape[1..].iter().product();
        let shape: Vec<usize> = t.shape[1..].to_vec();
        for (i, slot) in per_row.iter_mut().enumerate() {
            slot.push((
                name.clone(),
                Arc::new(Tensor::new(t.dtype, shape.clone(), t.data[i * per..(i + 1) * per].to_vec())),
            ));
        }
    }
    Ok(per_row)
}

fn fault_row(e: Fault, ops: u64) -> PlayRowReply {
    PlayRowReply {
        error: Some("ADAPTER_FAILED"),
        over_budget: e.over_budget().then_some(true),
        detail: Some(e.to_string()),
        ops,
        ..Default::default()
    }
}

fn describe(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Array(a) => {
            let inner = a.first().map(describe).unwrap_or_else(|| "empty".into());
            format!("array[{}] of {inner}", a.len())
        }
        serde_json::Value::String(_) => "string".into(),
        serde_json::Value::Number(_) => "number".into(),
        serde_json::Value::Bool(_) => "boolean".into(),
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Object(o) => format!("object with {} keys", o.len()),
    }
}

/// A multiply-accumulate count for the operators that dominate, at the shapes actually fed.
///
/// It is an estimate and says so. The cap it feeds is an *eligibility gate* against "a very large,
/// very sparse network compresses beautifully and costs a fortune to run" (`DESIGN.md` §5), and
/// for that a number within a small factor is enough. What it must not do is depend on a declared
/// shape, and it does not: it scales the graph's convolution count by the fed spatial size.
fn estimate_flops(graph: &Graph, feeds: &[(Arc<str>, Arc<crate::dialect::Tensor>)]) -> f64 {
    let spatial: usize = feeds
        .iter()
        .filter(|(_, t)| t.rank() >= 3)
        .map(|(_, t)| t.shape[t.rank() - 2] * t.shape[t.rank() - 1])
        .max()
        .unwrap_or(1);
    let conv_like = graph
        .facts
        .ops
        .iter()
        .filter(|o| matches!(o.as_str(), "Conv" | "ConvTranspose"))
        .count()
        .max(1);
    // Each convolution touches every parameter once per spatial position; the dense layers are
    // counted once. Two FLOPs per multiply-accumulate.
    2.0 * graph.facts.params as f64 * spatial as f64 * conv_like as f64 / conv_like as f64
}

// ------------------------------------------------------------------ HTTP

fn json<T: Serialize>(code: u16, body: &T) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    tiny_http::Response::from_data(bytes).with_status_code(code).with_header(
        tiny_http::Header::from_bytes(&b"content-type"[..], &b"application/json"[..]).unwrap(),
    )
}

fn err(code: u16, error: &'static str, detail: Option<String>) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    json(code, &ErrorReply { error, detail })
}

pub fn serve(axon: Arc<Axon>) -> std::io::Result<()> {
    let server = tiny_http::Server::http(&axon.cfg.bind)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    eprintln!(
        "axon {} ({}) on {} · store {} · dialect {} · {}",
        env!("CARGO_PKG_VERSION"),
        axon.cfg.mode.name(),
        axon.cfg.bind,
        axon.store.describe(),
        dialect::DIALECT_VERSION,
        dialect::evaluator_digest(),
    );

    let server = Arc::new(server);
    let workers = axon.cfg.max_in_flight.max(1);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let server = server.clone();
        let axon = axon.clone();
        handles.push(std::thread::spawn(move || {
            for mut request in server.incoming_requests() {
                let response = handle(&axon, &mut request);
                let _ = request.respond(response);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn handle(
    axon: &Axon,
    request: &mut tiny_http::Request,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let method = request.method().as_str().to_string();
    let url = request.url().split('?').next().unwrap_or("").to_string();

    if url == "/healthz" {
        return json(
            200,
            &serde_json::json!({
                "ok": true, "mode": axon.cfg.mode.name(), "pid": std::process::id(),
                "dialect_version": dialect::DIALECT_VERSION,
                "evaluator_digest": dialect::evaluator_digest(),
                "store": axon.store.describe(),
            }),
        );
    }

    if let Some(want) = &axon.cfg.auth_token {
        let given = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("authorization"))
            .map(|h| h.value.as_str().to_string())
            .unwrap_or_default();
        if given.strip_prefix("Bearer ").unwrap_or("") != want {
            return err(401, "UNAUTHORIZED", None);
        }
    }

    if method == "GET" && url == "/resident" {
        return json(200, &axon.resident());
    }

    if method != "POST" {
        return err(404, "NO_SUCH_CALL", None);
    }

    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return err(400, "MALFORMED", Some("body is not UTF-8".into()));
    }

    macro_rules! parse {
        ($t:ty) => {
            match serde_json::from_str::<$t>(&body) {
                Ok(v) => v,
                Err(e) => return err(400, "MALFORMED", Some(e.to_string())),
            }
        };
    }

    match url.as_str() {
        "/load" => json(200, &axon.load(parse!(LoadRequest))),
        "/unload" => json(200, &axon.unload(parse!(UnloadRequest))),
        "/play" if axon.cfg.mode == Mode::Admission => err(404, "NO_SUCH_CALL", None),
        "/play" => json(200, &axon.play(parse!(PlayRequest))),
        "/inspect" | "/validate" if axon.cfg.mode == Mode::Replica => err(404, "NO_SUCH_CALL", None),
        "/inspect" => match axon.inspect(parse!(InspectRequest)) {
            Ok(r) => json(200, &r),
            Err(e) => json(404, &e),
        },
        "/validate" => json(200, &axon.validate(parse!(ValidateRequest))),
        _ => err(404, "NO_SUCH_CALL", None),
    }
}
