//! `dialect_version` and `evaluator_digest` — docs/dialect.md §5.
//!
//! The digest is sha256 over a canonical description of the dialect — version, operator tables,
//! counting rules, semantics — and NOT of the binary, so a rebuild reports the same digest and the
//! re-validation sweep fires on a change to what an adapter means rather than on a release.

use sha2::{Digest, Sha256};

use super::eval::CORE;
use super::ops::TENSOR_OPS;

pub const DIALECT_VERSION: u32 = 1;

/// In the digest because changing one changes every adapter's cost without touching an operator.
const COUNTING_RULES: &[&str] = &[
    "node:1",
    "tensor_op:1+max(read,produced)",
    "reshape:1",
    "metadata:1",
    "literal:1",
    "per-direction",
    "abort-immediate",
];

/// Semantic choices that are not implied by the operator list and that an adapter can observe.
const SEMANTICS: &[&str] = &[
    "loose-eq:javascript",
    "null-eq:null-only",
    "truthy:jsonlogic",
    "object-rule:single-known-operator-is-an-operation",
    "map-scope:element-only",
    "reduce-seed:outer-scope",
    "narrowing:saturating",
    "argmax:first-maximum",
    "scatter-oob:dropped",
    "div-by-zero:null",
    "max-depth:64",
];

pub fn canonical_description() -> String {
    let mut s = String::new();
    s.push_str(&format!("dialect {DIALECT_VERSION}\n"));
    s.push_str("core\n");
    for op in CORE {
        s.push_str(op);
        s.push('\n');
    }
    s.push_str("tensor\n");
    for (name, sig) in TENSOR_OPS {
        s.push_str(name);
        s.push(' ');
        s.push_str(sig);
        s.push('\n');
    }
    s.push_str("counting\n");
    for r in COUNTING_RULES {
        s.push_str(r);
        s.push('\n');
    }
    s.push_str("semantics\n");
    for r in SEMANTICS {
        s.push_str(r);
        s.push('\n');
    }
    s
}

pub fn evaluator_digest() -> String {
    let mut h = Sha256::new();
    h.update(canonical_description().as_bytes());
    format!("sha256:{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_is_stable_and_is_not_the_binary() {
        // Called twice in one process, and it must not depend on anything but the description.
        assert_eq!(evaluator_digest(), evaluator_digest());
        assert!(evaluator_digest().starts_with("sha256:"));
        assert_eq!(evaluator_digest().len(), 7 + 64);

        // The description must actually mention every operator: a digest that did not cover the
        // table would let an operator change without firing a re-validation sweep.
        let d = canonical_description();
        for op in CORE {
            assert!(d.contains(op), "core operator {op} is not in the digest's description");
        }
        for (name, _) in TENSOR_OPS {
            assert!(d.contains(name), "tensor operator {name} is not in the digest's description");
        }
    }
}
