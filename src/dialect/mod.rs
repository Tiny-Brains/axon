//! The adapter dialect — layer 04 §4.
//!
//! A competitor's `adapter.json` is two programs:
//!
//! ```jsonc
//! { "dialect": 1,
//!   "in":  <program: observation -> { input_name: tensor }>,
//!   "out": <program: { output_name: tensor } -> action> }
//! ```
//!
//! # Why this is not built on `datalogic-rs`
//!
//! Layer 04 said "wrap or fork `datalogic-rs`", which is the engine Orion evaluates workflow logic
//! with. Neither turned out to be right, and the reasoning is worth keeping because the obvious
//! objection — two JSONLogic engines on one platform — is a real cost that has to be paid for.
//!
//! **Wrapping does not work.** The count in layer 04 §4.4 is "every node evaluated costs 1", and
//! `datalogic-rs` 5.4 exposes no evaluation hook, no step budget and no fuel. Its `CustomOperator`
//! trait covers the `tb.*` operators and nothing else: the core operators — `map`, `if`, `+` — are
//! where an adapter actually spends, and they are unreachable. Its trace API records every step,
//! but materialises the context and the result as `serde_json::Value` per node, which for a
//! program that touches a 128x128 plane is not instrumentation, it is a copy of the observation
//! per node.
//!
//! **Forking is worse than it looks.** It pins the platform's competitor-facing fairness rule to
//! the internals of a crate whose own documentation says they may evolve, and every 5.x upgrade
//! becomes a re-patch of an evaluation loop that no test of ours covers.
//!
//! **And the dialect is not JSONLogic anyway.** It is a fixed subset plus 19 tensor operators over
//! an opaque value JSON does not have, with no arithmetic on tensors (§4.6) and an operator table
//! that `evaluator_digest` hashes. Owning it means layer 04 §4.3 and §4.4 *are* the implementation
//! rather than a description of one.
//!
//! The cost of that choice is paid in `tests/differential.rs`: the core subset is run through both
//! engines on every case, and they must agree except where this dialect diverges **on purpose**,
//! which is one place — `{"==": [0, null]}`. `datalogic-rs` answers true; JavaScript, the
//! JSONLogic specification and this dialect answer false. The spike found that quirk the hard way
//! (`design/v2/03-spike/FINDINGS.md` §2.6): a join written against a path that does not resolve
//! silently selects the falsy elements and looks correct for exactly as long as the value it is
//! compared against is zero. An adapter is the worst possible place to rediscover that.

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

/// A result plus what it cost. Every entry point reports its own count, because the count is the
/// competitor-facing budget and a caller that cannot see it cannot enforce it.
pub type Counted<T> = (Result<T, Fault>, u64);

/// A compiled adapter: the two programs, plus the exact bytes they were parsed from, so the row's
/// copy stays self-verifying (layer 01 §3.2).
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
            doc.get(k).cloned().ok_or_else(|| Fault::invalid(format!("adapter has no '{k}' program")))
        };
        Ok(Adapter { dialect, in_program: take("in")?, out_program: take("out")? })
    }

    /// The observation to the graph's inputs. The result must be an object of tensors, and that is
    /// checked here rather than at the graph, so the error names the adapter.
    pub fn run_in(
        &self,
        observation: &serde_json::Value,
        budget: u64,
    ) -> Counted<Ports> {
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

    /// The graph's outputs to an action. The result must be JSON — a tensor that escaped into it
    /// would serialise as `null` and reach the engine as a malformed action, so it is refused here
    /// where the message can say why.
    ///
    /// **The document is `{ "outputs": {...}, "observation": {...} }`**, not the outputs alone.
    /// The observation is there because without it the batchable architecture is not expressible:
    /// a graph that answers a dense policy map for the whole board — which is the shape that
    /// actually batches across seats, since one preset per wave means every board has one shape —
    /// can only be turned into per-unit moves by gathering at positions the observation holds. It
    /// is the seat's own observation, already in the loader's hand, so it leaks nothing and costs
    /// a reference.
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
