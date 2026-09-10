//! `/inspect` and `/validate` — the admission role's two calls. Both report; neither judges. The
//! class table and the thresholds are platform policy and live in jodi/docs/admission.md.

use std::time::{Duration, Instant};

use super::Axon;
use crate::api::*;
use crate::dialect;
use crate::model;
use crate::store::Kind;

impl Axon {
    pub fn inspect(&self, req: InspectRequest) -> Result<InspectReply, ErrorReply> {
        let graph = self
            .residency
            .lock()
            .unwrap()
            .graph(&req.weights_hash)
            .ok_or(ErrorReply { error: "NOT_RESIDENT", detail: None })?;
        let adapter_bytes = self
            .store
            .get(Kind::Adapter, &req.adapter_hash)
            .map_err(|_| ErrorReply { error: "NOT_RESIDENT", detail: Some("adapter".into()) })?;

        let (s, w, a) = model::size_metric(&graph.facts.initializer_bytes, &adapter_bytes);
        Ok(InspectReply {
            params: graph.facts.params,
            opset: graph.facts.opset,
            ops: graph.facts.ops.iter().cloned().collect(),
            unsupported_ops: Vec::new(),
            size_metric_bytes: s,
            weights_zstd_bytes: w,
            adapter_zstd_bytes: a,
            weights_raw_bytes: graph.bytes_resident,
            adapter_raw_bytes: adapter_bytes.len() as u64,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            // The bytes as fetched, never a re-serialisation: `models.adapter` stores this text and
            // a CHECK recomputes the hash over it. They parsed as JSON at `/load`, so the error arm
            // is unreachable rather than lossy.
            adapter: String::from_utf8(adapter_bytes).map_err(|_| ErrorReply {
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
            infer_us_max: 0,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        };
        // `over_budget` is set on every failure, because "too expensive" and "wrong" are different
        // things to tell a competitor.
        let fail = |mut r: ValidateReply, reason, detail: String, case| {
            r.ok = false;
            r.reason = Some(reason);
            r.detail = Some(detail);
            r.failing_case = case;
            r.over_budget = Some(reason == "ADAPTER_FAILED");
            r
        };

        let held = {
            let r = self.residency.lock().unwrap();
            r.graph(&req.weights_hash).zip(r.adapter(&req.adapter_hash))
        };
        let Some((graph, adapter)) = held else {
            return fail(reply, "NOT_RESIDENT", "load it first".into(), None);
        };
        if req.observations.is_empty() {
            let d = "no reference observations to validate against".into();
            return fail(reply, "ADAPTER_INVALID", d, None);
        }

        for (i, obs) in req.observations.iter().enumerate() {
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_millis(req.deadline_ms);

            let (feeds, ops_in) = adapter.run_in(obs, req.budget_ops);
            let feeds = match feeds {
                Ok(f) => f,
                Err(e) => return fail(reply, adapter_reason(&e), e.to_string(), Some(i)),
            };

            // Every declared input must be fed, by name.
            for name in graph.input_names() {
                if !feeds.iter().any(|(k, _)| &**k == name) {
                    let d = format!("the adapter produced no tensor for input '{name}'");
                    return fail(reply, "SHAPE_MISMATCH", d, Some(i));
                }
            }

            // Timed on its own, so the number reported is the graph's cost and not the adapter's.
            let t_infer = Instant::now();
            let outputs = match graph.run(&feeds, deadline) {
                Ok(o) => o,
                Err(e) => return fail(reply, "SHAPE_MISMATCH", e, Some(i)),
            };
            let infer_us = t_infer.elapsed().as_micros() as u64;

            let (action, ops_out) = adapter.run_out(outputs, obs, req.budget_ops);
            let action = match action {
                Ok(a) => a,
                Err(e) => return fail(reply, adapter_reason(&e), e.to_string(), Some(i)),
            };

            reply.ops_max = reply.ops_max.max(ops_in.max(ops_out));
            reply.infer_us_max = reply.infer_us_max.max(infer_us);
            reply.cases.push(ValidateCase {
                ops_in,
                ops_out,
                elapsed_ms: t0.elapsed().as_millis() as u64,
                inputs: feeds
                    .iter()
                    .map(|(k, t)| FedPort {
                        name: k.to_string(),
                        dtype: t.dtype.name().to_string(),
                        shape: t.shape.clone(),
                    })
                    .collect(),
                infer_us,
                action_shape: describe(&action),
            });
        }
        reply
    }
}

fn adapter_reason(e: &dialect::Fault) -> &'static str {
    if e.over_budget() { "ADAPTER_FAILED" } else { "ADAPTER_INVALID" }
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
