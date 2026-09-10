//! The wire shapes — docs/design.md §3. Serde types only; the behaviour is in `server/`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct LoadRequest {
    #[serde(default)]
    pub models: Vec<ModelRef>,
    pub idle_ttl_s: Option<u64>,
    #[serde(default)]
    pub wait_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRef {
    pub weights_hash: String,
    pub adapter_hash: String,
    /// Accepted only by the admission instance; a replica answers `URL_NOT_ACCEPTED`.
    pub weights_url: Option<String>,
    pub adapter_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LoadReply {
    pub models: Vec<ModelState>,
    pub evaluator_digest: String,
    pub dialect_version: u32,
}

#[derive(Debug, Serialize)]
pub struct ModelState {
    pub weights_hash: String,
    pub adapter_hash: String,
    /// `resident` | `loading` | `refused`
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    /// `model` | `loader`. Kalam's barrier branches on this field rather than on the reason word,
    /// so a reason added later costs no workflow change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fault: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PlayRequest {
    #[serde(default)]
    pub rows: Vec<PlayRow>,
    #[serde(default = "default_deadline")]
    pub deadline_ms: u64,
    /// The game's `budgets.adapter_ops_max`. A parameter, not a setting: passing it per call is
    /// what keeps this process game-agnostic.
    #[serde(default = "default_budget")]
    pub budget_ops: u64,
}

fn default_deadline() -> u64 {
    1000
}
fn default_budget() -> u64 {
    1_000_000
}

#[derive(Debug, Deserialize)]
pub struct PlayRow {
    pub weights_hash: String,
    pub adapter_hash: String,
    pub observation: serde_json::Value,
    /// Echoed verbatim and never interpreted. Without it Kalam cannot count strikes.
    #[serde(default)]
    pub r#ref: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct PlayReply {
    pub rows: Vec<PlayRowReply>,
    pub evaluator_digest: String,
    pub dialect_version: u32,
}

#[derive(Debug, Serialize, Default)]
pub struct PlayRowReply {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub over_budget: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub ops: u64,
    /// Wall clock from this row entering the call to leaving it, so it includes waiting behind
    /// every group ahead of it. A latency, and **not** comparable between competitors.
    pub elapsed_ms: u64,
    /// This row's share of its own group's inference: the group ran once for `k` rows and each is
    /// charged `1/k` of it. Microseconds because a nano-class forward pass is under a millisecond.
    ///
    /// This is the number that compares two models. Both seats of a match are rows of the same
    /// call, on the same replica, at the same instant, so machine and load are shared and the
    /// comparison is paired.
    pub infer_us: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UnloadRequest {
    #[serde(default)]
    pub models: Vec<ModelRef>,
}

#[derive(Debug, Serialize)]
pub struct UnloadReply {
    pub models: Vec<UnloadState>,
}

#[derive(Debug, Serialize)]
pub struct UnloadState {
    pub weights_hash: String,
    pub adapter_hash: String,
    /// `released` | `not_held`
    pub state: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ResidentReply {
    pub weights: Vec<String>,
    pub loading: Vec<String>,
    pub adapters: Vec<String>,
    pub memory_bytes: u64,
    pub memory_budget_bytes: u64,
}

#[derive(Debug, Deserialize)]
pub struct InspectRequest {
    pub weights_hash: String,
    pub adapter_hash: String,
}

#[derive(Debug, Serialize)]
pub struct InspectReply {
    pub params: u64,
    pub opset: i64,
    pub ops: Vec<String>,
    pub unsupported_ops: Vec<String>,
    /// `S`, both terms. Reported, not classified: the class table is jodi's, so a threshold change
    /// is not a redeploy of this binary.
    pub size_metric_bytes: usize,
    pub weights_zstd_bytes: usize,
    pub adapter_zstd_bytes: usize,
    pub weights_raw_bytes: u64,
    pub adapter_raw_bytes: u64,
    pub inputs: Vec<crate::model::Port>,
    pub outputs: Vec<crate::model::Port>,
    /// The adapter's exact bytes as text — the ones that hashed to `adapter_hash`, never a
    /// re-serialisation: `models.adapter` stores this text under a CHECK that recomputes the hash.
    pub adapter: String,
    pub evaluator_digest: String,
    pub dialect_version: u32,
}

#[derive(Debug, Deserialize)]
pub struct ValidateRequest {
    pub weights_hash: String,
    pub adapter_hash: String,
    #[serde(default = "default_budget")]
    pub budget_ops: u64,
    /// The game's reference set. It must contain a worst case or admission is theatre: the budget
    /// is checked per call at play, so a gate that saw only a small sample tested nothing.
    #[serde(default)]
    pub observations: Vec<serde_json::Value>,
    #[serde(default = "default_validate_deadline")]
    pub deadline_ms: u64,
}

fn default_validate_deadline() -> u64 {
    5000
}

#[derive(Debug, Serialize)]
pub struct ValidateReply {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failing_case: Option<usize>,
    /// Set when `ok` is false: the difference between "too expensive" and "wrong". Collapsing them
    /// tells a competitor whose adapter merely costs too much to re-read the specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub over_budget: Option<bool>,
    pub cases: Vec<ValidateCase>,
    pub ops_max: u64,
    /// The slowest reference case's inference, in microseconds. Reported, never a gate: wall clock
    /// is the admission host's, and rejecting on it would make a verdict depend on a noisy
    /// neighbour. It exists so a competitor can see the headroom the turn deadline leaves them.
    pub infer_us_max: u64,
    pub evaluator_digest: String,
    pub dialect_version: u32,
}

#[derive(Debug, Serialize)]
pub struct ValidateCase {
    pub ops_in: u64,
    pub ops_out: u64,
    pub elapsed_ms: u64,
    pub inputs: Vec<FedPort>,
    /// The graph run alone, at the shapes the adapter actually produced -- a declared input shape
    /// is a claim, what is fed is a fact. `elapsed_ms` covers the adapter's two directions too.
    pub infer_us: u64,
    pub action_shape: String,
}

#[derive(Debug, Serialize)]
pub struct FedPort {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct ErrorReply {
    pub error: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
