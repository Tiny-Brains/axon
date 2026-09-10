//! The adapter dialect — docs/design.md §4. A competitor's `adapter.json` is two programs:
//!
//! ```jsonc
//! { "dialect": 1,
//!   "in":  <program: observation -> { input_name: tensor }>,
//!   "out": <program: { output_name: tensor } -> action> }
//! ```
//!
//! It is not built on `datalogic-rs`, the engine Orion evaluates workflow logic with, because that
//! crate exposes no evaluation hook, step budget or fuel — and the count in docs/dialect.md §4 is
//! the competitor-facing fairness rule. The core JSONLogic subset must still agree with it, which
//! `tests/differential.rs` checks on every case; there is one deliberate divergence,
//! `{"==": [0, null]}`, documented in `value.rs`.

pub mod digest;
pub mod eval;
pub mod ops;
pub mod tensor;
pub mod value;

pub use digest::{evaluator_digest, DIALECT_VERSION};
pub use eval::{Fault, Res};
pub use tensor::{DType, Tensor};
pub use value::Value;

use std::sync::Arc;

/// The graph's inputs, or its outputs: a named tensor per port.
pub type Ports = Vec<(Arc<str>, Arc<Tensor>)>;

/// A result plus what it cost. Every entry point reports its own count: a caller that cannot see
/// the count cannot enforce the budget.
pub type Counted<T> = (Result<T, Fault>, u64);

/// A compiled adapter: the two programs, as parsed.
#[derive(Debug)]
pub struct Adapter {
    pub dialect: u32,
    pub in_program: serde_json::Value,
    pub out_program: serde_json::Value,
}

impl Adapter {
    pub fn parse(bytes: &[u8]) -> Result<Adapter, Fault> {
        let doc: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| Fault::invalid(format!("adapter is not JSON: {e}")))?;
        let dialect = doc.get("dialect").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        if dialect != DIALECT_VERSION {
            return Err(Fault::invalid(format!(
                "adapter declares dialect {dialect}; this evaluator implements {DIALECT_VERSION}"
            )));
        }
        let take = |k: &str| -> Result<serde_json::Value, Fault> {
            doc.get(k)
                .cloned()
                .ok_or_else(|| Fault::invalid(format!("adapter has no '{k}' program")))
        };
        Ok(Adapter { dialect, in_program: take("in")?, out_program: take("out")? })
    }

    /// The observation to the graph's inputs. The result must be an object of tensors, checked
    /// here rather than at the graph so the error names the adapter.
    pub fn run_in(&self, observation: &serde_json::Value, budget: u64) -> Counted<Ports> {
        let data = Value::from_json(observation);
        let (out, ops) = eval::run(&self.in_program, &data, budget);
        let checked = out.and_then(|v| match v {
            Value::Obj(fields) => fields
                .iter()
                .map(|(k, v)| match v {
                    Value::Tensor(t) => Ok((k.clone(), t.clone())),
                    other => Err(Fault::invalid(format!(
                        "the 'in' program produced a {} for input '{k}'; it must produce a tensor",
                        other.type_name()
                    ))),
                })
                .collect(),
            other => Err(Fault::invalid(format!(
                "the 'in' program produced a {}; it must produce an object of tensors",
                other.type_name()
            ))),
        });
        (checked, ops)
    }

    /// The graph's outputs to an action. The result must be JSON: a tensor that escaped into it
    /// would serialise as `null` and reach the engine as a malformed action.
    ///
    /// The document is `{ "outputs": {...}, "observation": {...} }`, not the outputs alone —
    /// without the observation a dense policy map cannot be turned into per-unit moves, and a
    /// dense map is the shape that batches across seats.
    pub fn run_out(
        &self,
        outputs: Ports,
        observation: &serde_json::Value,
        budget: u64,
    ) -> Counted<serde_json::Value> {
        let data = Value::obj(vec![
            (
                Arc::from("outputs"),
                Value::obj(outputs.into_iter().map(|(k, t)| (k, Value::Tensor(t))).collect()),
            ),
            (Arc::from("observation"), Value::from_json(observation)),
        ]);
        let (out, ops) = eval::run(&self.out_program, &data, budget);
        let checked = out.and_then(|v| {
            if contains_tensor(&v) {
                Err(Fault::invalid(
                    "the 'out' program produced a tensor; an action must be JSON. \
                     Read it back with tb.argmax, tb.gather or tb.to_list",
                ))
            } else {
                Ok(v.to_json())
            }
        });
        (checked, ops)
    }
}

fn contains_tensor(v: &Value) -> bool {
    match v {
        Value::Tensor(_) => true,
        Value::Arr(a) => a.iter().any(contains_tensor),
        Value::Obj(o) => o.iter().any(|(_, v)| contains_tensor(v)),
        _ => false,
    }
}
