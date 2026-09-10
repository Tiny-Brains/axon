//! The core subset, through both engines, on every case.
//!
//! Orion evaluates workflow JSONLogic with `datalogic-rs`; Axon evaluates adapter JSONLogic with
//! its own. Where they overlap they must agree, or the platform has two dialects and nobody is
//! told which one they are writing. Divergences are listed, not discovered. There is one.

use serde_json::{Value as J, json};

fn axon_eval(program: &J, data: &J) -> Result<J, String> {
    let d = axon::dialect::Value::from_json(data);
    let (out, _ops) = axon::dialect::eval::run(program, &d, 100_000_000);
    out.map(|v| v.to_json()).map_err(|e| e.to_string())
}

fn datalogic_eval(program: &J, data: &J) -> Result<J, String> {
    // Templating on: it is what makes a multi-key object a literal whose values are evaluated,
    // and it is how Orion evaluates the object literals its own workflows build.
    let engine = datalogic_rs::EngineBuilder::new().with_templating(true).build();
    let ps = program.to_string();
    let compiled = engine.compile(ps.as_str()).map_err(|e| e.to_string())?;
    let mut session = engine.session();
    let ds = data.to_string();
    let out = session.eval_str(&compiled, ds.as_str()).map_err(|e| e.to_string())?;
    serde_json::from_str(&out).map_err(|e| format!("{e}: {out}"))
}

/// Every case both engines must agree on.
fn cases() -> Vec<(&'static str, J, J)> {
    let obs = json!({
        "size": [8, 8],
        "mine": [[1, 2], [3, 4], [5, 6]],
        "foes": [[7, 7, 1]],
        "food": [[0, 0], [2, 2]],
        "n": 3, "flag": true, "name": "ants", "zero": 0, "empty": [], "nothing": null
    });
    vec![
        ("var-scalar", json!({"var": "n"}), obs.clone()),
        ("var-nested", json!({"var": "size.0"}), obs.clone()),
        ("var-array-index", json!({"var": "mine.1.0"}), obs.clone()),
        ("var-missing", json!({"var": "nope"}), obs.clone()),
        ("var-default", json!({"var": ["nope", 42]}), obs.clone()),
        ("var-root", json!({"var": ""}), obs.clone()),
        ("if-chain", json!({"if": [{"var": "flag"}, "yes", "no"]}), obs.clone()),
        ("if-elseif", json!({"if": [false, "a", false, "b", "c"]}), obs.clone()),
        ("and-value", json!({"and": [1, "x", 3]}), obs.clone()),
        ("and-short", json!({"and": [1, 0, 3]}), obs.clone()),
        ("or-value", json!({"or": [0, "", "z"]}), obs.clone()),
        ("not", json!({"!": [{"var": "zero"}]}), obs.clone()),
        ("truthy", json!({"!!": [{"var": "empty"}]}), obs.clone()),
        ("eq-num-str", json!({"==": [1, "1"]}), obs.clone()),
        ("eq-strict", json!({"===": [1, "1"]}), obs.clone()),
        ("neq", json!({"!=": [1, 2]}), obs.clone()),
        ("lt", json!({"<": [1, 2]}), obs.clone()),
        ("between", json!({"<": [1, {"var": "n"}, 5]}), obs.clone()),
        ("gte-str", json!({">=": ["b", "a"]}), obs.clone()),
        ("plus", json!({"+": [1, 2, 3]}), obs.clone()),
        ("minus", json!({"-": [10, 3, 2]}), obs.clone()),
        ("negate", json!({"-": [5]}), obs.clone()),
        ("times", json!({"*": [2, 3, 4]}), obs.clone()),
        ("modulo", json!({"%": [7, 3]}), obs.clone()),
        ("max", json!({"max": [1, 9, 3]}), obs.clone()),
        ("min", json!({"min": [1, 9, 3]}), obs.clone()),
        ("cat", json!({"cat": ["a", 1, "b"]}), obs.clone()),
        ("substr", json!({"substr": ["abcdef", 1, 3]}), obs.clone()),
        ("substr-neg", json!({"substr": ["abcdef", -2]}), obs.clone()),
        ("in-array", json!({"in": [3, [1, 2, 3]]}), obs.clone()),
        ("in-string", json!({"in": ["nt", "ants"]}), obs.clone()),
        ("merge", json!({"merge": [[1, 2], [3]]}), obs.clone()),
        ("map-scalar", json!({"map": [{"var": "mine"}, {"var": "0"}]}), obs.clone()),
        (
            "map-object",
            json!({"map": [{"var": "mine"}, {"r": {"var": "0"}, "c": {"var": "1"}}]}),
            obs.clone(),
        ),
        (
            "map-arith",
            json!({"map": [{"var": "mine"}, {"+": [{"var": "0"}, {"var": "1"}]}]}),
            obs.clone(),
        ),
        ("filter", json!({"filter": [{"var": "mine"}, {">": [{"var": "0"}, 2]}]}), obs.clone()),
        (
            "reduce-sum",
            json!({"reduce": [{"var": "mine"}, {"+": [{"var": "accumulator"}, {"var": "current.0"}]}, 0]}),
            obs.clone(),
        ),
        (
            "reduce-seed-from-root",
            json!({"reduce": [{"var": "mine"}, {"+": [{"var": "accumulator"}, 1]}, {"var": "n"}]}),
            obs.clone(),
        ),
        (
            "reduce-flatten",
            json!({"reduce": [[[1, 2], [3]], {"merge": [{"var": "accumulator"}, {"var": "current"}]}, []]}),
            obs.clone(),
        ),
        ("all", json!({"all": [{"var": "mine"}, {">=": [{"var": "0"}, 1]}]}), obs.clone()),
        ("some", json!({"some": [{"var": "mine"}, {">": [{"var": "0"}, 4]}]}), obs.clone()),
        ("none", json!({"none": [{"var": "mine"}, {">": [{"var": "0"}, 9]}]}), obs.clone()),
        ("array-literal", json!([1, {"var": "n"}, 3]), obs.clone()),
        ("object-literal", json!({"a": 1, "b": {"var": "n"}}), obs.clone()),
        (
            "nested-map-filter",
            json!({"map": [{"filter": [{"var": "mine"}, {">": [{"var": "1"}, 2]}]}, {"var": "1"}]}),
            obs.clone(),
        ),
        // The scope rule: a body cannot reach the outer document. Both must answer null.
        ("map-cannot-see-root", json!({"map": [{"var": "mine"}, {"var": "size"}]}), obs.clone()),
        (
            "filter-cannot-see-root",
            json!({"filter": [{"var": "mine"}, {"var": "flag"}]}),
            obs.clone(),
        ),
    ]
}

#[test]
fn the_core_subset_agrees_with_datalogic() {
    let mut disagreed = Vec::new();
    for (name, program, data) in cases() {
        let a = axon_eval(&program, &data);
        let d = datalogic_eval(&program, &data);
        match (&a, &d) {
            (Ok(x), Ok(y)) if x == y => {}
            _ => disagreed.push(format!("  {name:<26} axon={a:?}\n  {:<28} dl  ={d:?}", "")),
        }
    }
    assert!(
        disagreed.is_empty(),
        "{} of {} cases disagree:\n{}",
        disagreed.len(),
        cases().len(),
        disagreed.join("\n")
    );
}

#[test]
fn the_one_deliberate_divergence() {
    // `{"==": [0, null]}`: JavaScript and the JSONLogic specification say false, datalogic-rs says
    // true. If this test fails because datalogic-rs changed, the divergence is gone and the notes
    // in dialect/mod.rs and value.rs should go with it.
    let p = json!({"==": [0, null]});
    let d = json!({});
    assert_eq!(axon_eval(&p, &d).unwrap(), json!(false), "axon must follow JavaScript");
    assert_eq!(
        datalogic_eval(&p, &d).unwrap(),
        json!(true),
        "datalogic-rs used to answer true here; if it no longer does, delete this divergence"
    );

    // And the case that actually bites: a comparison against a path that does not resolve.
    let join = json!({"filter": [{"var": "rows"}, {"==": [{"var": "m"}, {"var": "data.head"}]}]});
    let rows = json!({"rows": [{"m": 0}, {"m": 1}], "data": {"head": 1}});
    assert_eq!(axon_eval(&join, &rows).unwrap(), json!([]), "no row matches an unresolvable path");
    assert_eq!(
        datalogic_eval(&join, &rows).unwrap(),
        json!([{"m": 0}]),
        "datalogic-rs selects the falsy elements, which is the trap"
    );
}
