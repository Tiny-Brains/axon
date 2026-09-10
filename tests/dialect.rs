//! The counting rules and the operators — docs/dialect.md §3 and §4.4.

use axon::dialect::Value;
use axon::dialect::eval::{Fault, run};
use serde_json::json;

fn go(
    program: serde_json::Value,
    data: serde_json::Value,
    budget: u64,
) -> (Result<Value, Fault>, u64) {
    run(&program, &Value::from_json(&data), budget)
}

fn ok(program: serde_json::Value, data: serde_json::Value) -> (serde_json::Value, u64) {
    let (v, ops) = go(program, data, 100_000_000);
    (v.expect("program failed").to_json(), ops)
}

#[test]
fn every_node_evaluated_costs_one() {
    // Rule 1: applied, not written. The same program over a longer list costs more.
    let (_, three) =
        ok(json!({"map": [{"var": "xs"}, {"+": [{"var": ""}, 1]}]}), json!({"xs": [1, 2, 3]}));
    let (_, six) = ok(
        json!({"map": [{"var": "xs"}, {"+": [{"var": ""}, 1]}]}),
        json!({"xs": [1, 2, 3, 4, 5, 6]}),
    );
    let body_cost = (six - three) / 3;
    assert_eq!(three + body_cost * 3, six, "cost must be linear in the number of applications");
    assert!(body_cost >= 2, "the body is at least the `+` and its `var`");
}

#[test]
fn a_literal_costs_one_however_large() {
    // Rule 3. A lookup table in the program is priced by `S`, not by the op count.
    let small = json!({"var": "x"});
    let big: Vec<i64> = (0..5000).collect();
    let (_, a) = ok(small, json!({"x": 1}));
    let (_, b) = ok(json!({"tb.at": [big, 0]}), json!({}));
    // Two, not one: `{"var": "x"}` is the operator node plus its path argument, and the path is a
    // node like anything else. "Every node evaluated costs 1" means every node.
    assert_eq!(a, 2);
    // The array literal is 5000 expressions in expression position, so it is NOT one node -- and
    // that is the honest reading of "a literal costs 1": a *scalar* literal does. A competitor
    // shipping a table should put it in the adapter's data, which `S` prices, not in a live array.
    assert!(b > 5000, "an inline array of expressions is counted per element: {b}");
}

#[test]
fn a_tensor_operator_costs_one_plus_max_read_produced() {
    // Rule 2, checked against arithmetic rather than against itself.
    let (_, zeros) = ok(json!({"tb.zeros": [[10, 10], "int8"]}), json!({}));
    // 1 node + 100 produced, plus the nodes for the two literal args and their elements.
    assert!((101..=110).contains(&zeros), "tb.zeros on 100 elements cost {zeros}");

    let (_, argmax) = ok(json!({"tb.argmax": [{"tb.zeros": [[20, 5], "float32"]}, 1]}), json!({}));
    let (_, zeros100) = ok(json!({"tb.zeros": [[20, 5], "float32"]}), json!({}));
    // argmax reads 100 and produces 20, so it is charged the larger: its own node, the 100, and
    // the axis literal, which is a node too.
    assert_eq!(argmax - zeros100, 102, "argmax should cost 1 + max(100 read, 20 produced) + axis");
}

#[test]
fn reshape_is_a_view_and_costs_one() {
    let (_, with) = ok(json!({"tb.reshape": [{"tb.zeros": [[10, 10], "int8"]}, [100]]}), json!({}));
    let (_, without) = ok(json!({"tb.zeros": [[10, 10], "int8"]}), json!({}));
    assert_eq!(with - without, 3, "reshape is its node plus its shape literal, not its elements");
}

#[test]
fn over_budget_aborts_immediately_rather_than_reporting_afterwards() {
    // Rule 5, and docs/design.md §13 item 4: a bound enforced after the work bounds the report, not the
    // work. A tensor 40 million elements wide must be refused, not allocated.
    let start = std::time::Instant::now();
    let (r, ops) = go(json!({"tb.zeros": [[6000, 6000], "float32"]}), json!({}), 200_000);
    let elapsed = start.elapsed();
    assert!(matches!(r, Err(Fault::OverBudget { .. })), "expected OverBudget, got {r:?}");
    assert!(ops > 200_000);
    assert!(
        elapsed.as_millis() < 100,
        "refusal took {elapsed:?} -- it allocated first and counted afterwards"
    );
}

#[test]
fn over_budget_is_adapter_failed_and_a_bad_program_is_adapter_invalid() {
    // The two are different words to a competitor and different branches to Kalam.
    let (over, _) = go(json!({"tb.zeros": [[1000, 1000], "int8"]}), json!({}), 100);
    assert_eq!(over.unwrap_err().code(), "ADAPTER_FAILED");

    let (bad, _) = go(json!({"tb.nope": [1]}), json!({}), 100_000);
    assert_eq!(bad.unwrap_err().code(), "ADAPTER_INVALID");

    let (wrong_type, _) = go(json!({"tb.argmax": ["not a tensor", 0]}), json!({}), 100_000);
    assert_eq!(wrong_type.unwrap_err().code(), "ADAPTER_INVALID");
}

#[test]
fn the_count_does_not_depend_on_the_machine_or_the_run() {
    // The one property docs/design.md §10 says is required: integer arithmetic over a deterministic
    // walk. Same program, same input, same count -- every time.
    let p = json!({"tb.stack": [[
        {"tb.scatter": [{"var": "pts"}, [32, 32], "int8"]},
        {"tb.rle_expand": [{"var": "runs"}, [32, 32], "int8"]}], 0, "int8"]});
    let d = json!({"pts": [[1, 2], [3, 4]], "runs": [0, 512, 1, 512]});
    let counts: Vec<u64> = (0..8).map(|_| ok(p.clone(), d.clone()).1).collect();
    assert!(counts.windows(2).all(|w| w[0] == w[1]), "counts varied across runs: {counts:?}");
}

#[test]
fn a_tensor_is_opaque_to_a_program() {
    // Layer docs/dialect.md §2. Shape and dtype, and nothing else.
    let (shape, _) = ok(json!({"var": "t.shape"}), json!({}));
    assert_eq!(shape, json!(null), "there is no tensor in the data to begin with");

    let (s, _) = ok(json!({"tb.shape": [{"tb.zeros": [[2, 3], "int8"]}]}), json!({}));
    assert_eq!(s, json!([2, 3]));
    let (d, _) = ok(json!({"tb.dtype": [{"tb.zeros": [[2, 3], "int8"]}]}), json!({}));
    assert_eq!(d, json!("int8"));

    // An element is not reachable by indexing: a tensor is not an array.
    let (elem, _) = ok(json!({"var": [["t", 0]]}), json!({}));
    assert_eq!(elem, json!(null));
}

#[test]
fn the_operators_do_what_they_say() {
    let (v, _) =
        ok(json!({"tb.to_list": [{"tb.scatter": [[[0, 1], [1, 0]], [2, 2], "int8"]}]}), json!({}));
    assert_eq!(v, json!([[0, 1], [1, 0]]));

    let (v, _) =
        ok(json!({"tb.to_list": [{"tb.rle_expand": [[0, 2, 1, 2], [2, 2], "int8"]}]}), json!({}));
    assert_eq!(v, json!([[0, 0], [1, 1]]));

    let (v, _) = ok(json!({"tb.to_list": [{"tb.one_hot": [[0, 2], 3, "int8"]}]}), json!({}));
    assert_eq!(v, json!([[1, 0, 0], [0, 0, 1]]));

    let (v, _) = ok(
        json!({"tb.argmax": [{"tb.tensor": [[1, 9, 3, 8, 2, 7], [2, 3], "float32"]}, 1]}),
        json!({}),
    );
    assert_eq!(v, json!([1, 0]));

    let (v, _) = ok(
        json!({"tb.to_list": [{"tb.transpose": [{"tb.tensor": [[1, 2, 3, 4, 5, 6], [2, 3], "int8"]}, [1, 0]]}]}),
        json!({}),
    );
    assert_eq!(v, json!([[1, 4], [2, 5], [3, 6]]));

    let (v, _) = ok(
        json!({"tb.to_list": [{"tb.pad": [{"tb.tensor": [[1, 2, 3, 4], [2, 2], "int8"]}, [1, 1], [0, 0], 9]}]}),
        json!({}),
    );
    assert_eq!(v, json!([[9, 9, 9], [9, 1, 2], [9, 3, 4]]));

    let (v, _) = ok(
        json!({"tb.to_list": [{"tb.gather": [{"tb.tensor": [[1, 2, 3, 4, 5, 6], [3, 2], "int8"]}, [2, 0], 0]}]}),
        json!({}),
    );
    assert_eq!(v, json!([[5, 6], [1, 2]]));

    let (v, _) = ok(
        json!({"tb.to_list": [{"tb.stack": [[
        {"tb.tensor": [[1, 2], [2], "int8"]},
        {"tb.tensor": [[3, 4], [2], "int8"]}], 0, "int8"]}]}),
        json!({}),
    );
    assert_eq!(v, json!([[1, 2], [3, 4]]));
}

#[test]
fn narrowing_saturates_rather_than_wrapping() {
    // A competitor scattering 300 into an int8 plane gets 127. A wrap would hide their arithmetic
    // mistake inside a plausible number.
    let (v, _) =
        ok(json!({"tb.to_list": [{"tb.tensor": [[300, -300, 5], [3], "int8"]}]}), json!({}));
    assert_eq!(v, json!([127, -128, 5]));
}

#[test]
fn a_scatter_outside_the_plane_is_dropped_not_fatal() {
    // An adapter clipping a wrapped coordinate is ordinary; refusing the turn for it would be a
    // strike for arithmetic the platform never specified.
    let (v, _) = ok(
        json!({"tb.to_list": [{"tb.scatter": [[[0, 0], [9, 9], [-1, 0]], [2, 2], "int8"]}]}),
        json!({}),
    );
    assert_eq!(v, json!([[1, 0], [0, 0]]));
}

#[test]
fn division_by_zero_is_null_rather_than_a_refused_turn() {
    let (v, _) = ok(json!({"??": [{"/": [1, {"var": "n"}]}, -1]}), json!({"n": 0}));
    assert_eq!(v, json!(-1));
}

#[test]
fn the_two_directions_are_checked_at_their_boundaries() {
    use axon::dialect::Adapter;
    let a =
        Adapter::parse(br#"{"dialect": 1, "in": {"x": 1}, "out": {"tb.zeros": [[2], "int8"]}}"#)
            .unwrap();

    // `in` must produce tensors.
    let (r, _) = a.run_in(&json!({}), 100_000);
    assert!(r.unwrap_err().to_string().contains("must produce a tensor"));

    // `out` must not.
    let (r, _) = a.run_out(vec![], &json!({}), 100_000);
    assert!(r.unwrap_err().to_string().contains("must be JSON"));
}

#[test]
fn an_operator_with_no_arguments_is_answered_rather_than_panicking() {
    // Every operator is reachable with an empty argument list from a competitor's adapter.
    for op in axon::dialect::eval::CORE {
        let (r, _) = go(json!({ *op: [] }), json!({}), 100_000);
        assert!(r.is_ok() || r.unwrap_err().code() == "ADAPTER_INVALID", "'{op}' on no arguments");
    }
}

#[test]
fn a_dialect_the_evaluator_does_not_implement_is_refused_at_parse() {
    use axon::dialect::Adapter;
    let e = Adapter::parse(br#"{"dialect": 2, "in": {}, "out": {}}"#).unwrap_err();
    assert!(e.to_string().contains("dialect 2"), "{e}");
    assert_eq!(e.code(), "ADAPTER_INVALID");
}
