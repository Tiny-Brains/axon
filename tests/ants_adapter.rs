//! What the reference Ants adapter costs — the measurement behind `budgets.adapter_ops_max`, which
//! ants/docs/protocol.md §4 carries provisionally at 200,000.
//!
//! `cargo test --test ants_adapter -- --nocapture` prints the tables.

mod common;

use axon::dialect::{Adapter, DType, Ports, Tensor};
use common::{OBS, obs, reference_adapter};
use serde_json::json;
use std::sync::Arc;

const PROVISIONAL_BUDGET: u64 = 200_000;
const UNLIMITED: u64 = 100_000_000;

/// Deriving the view mask from ant positions — ants/docs/protocol.md §8.3's concrete case — without
/// `tb.dilate`, which means unrolling the 317-cell disk as literal offsets per ant.
///
/// The natural nested iteration cannot be written: the inner body sees only the offset and there is
/// no path back to the ant. Written that way it does not error — `{"var": "current.0"}` is `null`,
/// the points are the raw offsets, half of them are negative, `tb.scatter` drops those, and the
/// plane comes out empty.
fn visibility_unrolled() -> Vec<u8> {
    let mut pairs = Vec::new();
    for dr in -8i64..=8 {
        for dc in -8i64..=8 {
            if dr * dr + dc * dc <= 77 {
                pairs.push(json!([
                    {"+": [{"var": "current.0"}, dr]},
                    {"+": [{"var": "current.1"}, dc]}
                ]));
            }
        }
    }
    let points =
        json!({"reduce": [{"var": "mine"}, {"merge": [{"var": "accumulator"}, pairs]}, []]});
    visibility_adapter(json!({"tb.scatter": [points, {"var": "size"}, "int8"]}))
}

/// The same mask with the operator: scatter the ants onto a plane and dilate it once.
fn visibility_dilate() -> Vec<u8> {
    visibility_adapter(json!({"tb.dilate": [
        {"tb.scatter": [{"var": "mine"}, {"var": "size"}, "int8"]}, 77]}))
}

fn visibility_adapter(vis: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "dialect": 1,
        "in": {"vis": vis},
        "out": {"map": [{"tb.argmax": [{"var": "outputs.policy"}, 1]},
                        {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    }))
    .unwrap()
}

/// What the graph would answer: a score per direction per ant.
fn policy(n: usize) -> Ports {
    let data = (0..n * 5).map(|i| ((i * 37) % 11) as f64).collect();
    vec![(Arc::from("policy"), Arc::new(Tensor::new(DType::F32, vec![n, 5], data)))]
}

fn marked_cells(adapter: &[u8]) -> (u64, usize) {
    let (planes, ops) = Adapter::parse(adapter).unwrap().run_in(&obs(), UNLIMITED);
    let planes = planes.expect("the in program failed");
    (ops, planes[0].1.data.iter().filter(|&&v| v != 0.0).count())
}

#[test]
fn the_reference_adapter_runs_and_what_it_costs() {
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let o = obs();
    let n_ants = o["mine"].as_array().unwrap().len();
    let cells = o["size"][0].as_u64().unwrap() * o["size"][1].as_u64().unwrap();

    let (inputs, ops_in) = a.run_in(&o, UNLIMITED);
    let inputs = inputs.expect("the in program failed");
    let names: Vec<&str> = inputs.iter().map(|(k, _)| &**k).collect();
    assert!(names.contains(&"board") && names.contains(&"ant_r") && names.contains(&"ant_c"));

    let board = &inputs.iter().find(|(k, _)| &**k == "board").unwrap().1;
    assert_eq!(board.shape, vec![1, 6, 128, 128]);
    assert_eq!(board.dtype, DType::I8);

    let (action, ops_out) = a.run_out(policy(n_ants), &o, UNLIMITED);
    let action = action.expect("the out program failed");
    let action = action.as_array().unwrap();
    assert_eq!(action.len(), n_ants, "len(action) must equal len(mine) -- protocol §6");
    assert!(action.iter().all(|v| matches!(v.as_str(), Some("N" | "E" | "S" | "W" | "-"))));

    println!("\n=== the reference Ants adapter, on a real worst-case observation ===");
    println!("  observation      {} bytes, {n_ants} ants, {cells} cells", OBS.len());
    println!("  in               {ops_in:>9} ops");
    println!("  out              {ops_out:>9} ops");
    println!("  the larger       {:>9} ops   <- what the budget must exceed", ops_in.max(ops_out));
    println!("  provisional      {PROVISIONAL_BUDGET:>9} ops");
    println!("  headroom         {:>8.2}x", PROVISIONAL_BUDGET as f64 / ops_in.max(ops_out) as f64);
}

#[test]
fn deriving_visibility_costs_what_it_costs() {
    let (ops_unrolled, marked_unrolled) = marked_cells(&visibility_unrolled());
    let (ops_dilate, marked_dilate) = marked_cells(&visibility_dilate());

    println!("\n=== deriving visibility -- protocol §8.3's concrete case ===");
    println!("  unrolled kernel   {ops_unrolled:>9} ops   {marked_unrolled} cells marked");
    println!("  tb.dilate         {ops_dilate:>9} ops   {marked_dilate} cells marked");
    println!("  the operator is   {:.0}x cheaper", ops_unrolled as f64 / ops_dilate as f64);

    // Both must actually derive a mask: the first version of this test compared a program against
    // one that silently did nothing and cost 256,518 operations to derive an empty plane.
    assert!(marked_unrolled > 2000, "the unrolled kernel marked only {marked_unrolled} cells");
    assert!(marked_dilate > 2000, "tb.dilate marked only {marked_dilate} cells");

    // And they do not agree, which is the stronger half of the argument for the operator: Ants maps
    // wrap, `tb.dilate` wraps, and the unrolled kernel's out-of-range coordinates are dropped by
    // `tb.scatter`, so a competitor's mask is quietly wrong along every border. They cannot fix it
    // -- the modulo needs a map size that is not in scope where the ant is.
    assert!(
        marked_dilate > marked_unrolled,
        "the unrolled kernel should lose the wrapped border cells: {marked_unrolled} vs {marked_dilate}"
    );
    println!("  the unrolled kernel loses {} border cells", marked_dilate - marked_unrolled);
}

#[test]
fn the_adapter_is_refused_at_the_provisional_budget_or_it_is_not() {
    // The assertion the number has to satisfy: run the reference adapter under the published budget
    // and see whether a real competitor would be struck every turn.
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let (r, ops) = a.run_in(&obs(), PROVISIONAL_BUDGET);
    match r {
        Ok(_) => println!("\nthe reference adapter fits 200,000 at {ops} ops"),
        Err(e) => println!("\nthe reference adapter does NOT fit 200,000: {e} (spent {ops})"),
    }
}

#[test]
fn what_an_operation_costs_in_wall_clock() {
    // The budget is a fairness rule, not a performance one, but the number still has to be one a
    // replica can afford K x seats times a turn. This turns a proposed budget into milliseconds.
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let o = obs();
    for _ in 0..3 {
        let _ = a.run_in(&o, UNLIMITED);
    }
    let n = 50;
    let t0 = std::time::Instant::now();
    let mut ops = 0;
    for _ in 0..n {
        let (r, c) = a.run_in(&o, UNLIMITED);
        assert!(r.is_ok());
        ops = c;
    }
    let per_run_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("\n=== what an operation costs ===");
    println!("  reference adapter  {ops} ops in {per_run_ms:.2} ms");
    println!("  per million ops    {:.2} ms", per_run_ms * 1_000_000.0 / ops as f64);
    for budget in [200_000u64, 500_000, 1_000_000, 2_000_000] {
        let ms = per_run_ms * budget as f64 / ops as f64;
        println!(
            "  a budget of {budget:>9}  =  {ms:>6.1} ms per seat, {:>7.1} ms for 32 seats serial",
            ms * 32.0
        );
    }
}
