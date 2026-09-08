//! The reference Ants adapter, and what it costs.
//!
//! This is the P1 counter spike (the tracker) and it answers **decision 6** — the number in
//! `budgets.adapter_ops_max`, which ants/docs/protocol.md §4 carries provisionally at 200,000.
//!
//! The observation is real: `tests/fixtures/ants-observation.json` was dumped from the spike engine
//! at turn 600 of a 128x128 match, when the known-water mask is at its most fragmented and the
//! payload is at its largest (`the wave-turn spikeFINDINGS.md` §3.4). 90 ants, 670 water runs,
//! 2,437 bytes.
//!
//! `cargo test --test ants_adapter -- --nocapture` prints the table.

use axon::dialect::{Adapter, DType, Tensor};
use serde_json::{json, Value as J};
use std::sync::Arc;

const OBS: &str = include_str!("fixtures/ants-observation.json");
const PROVISIONAL_BUDGET: u64 = 200_000;

fn obs() -> J {
    serde_json::from_str(OBS).expect("fixture")
}

/// The board an adapter of this shape builds: six int8 planes at the map's size.
///
/// Writing it is instructive, and two things about it are findings rather than style.
///
/// **The map body cannot see the observation.** `{"var": "size"}` inside a `map` is `null`, so the
/// width is not reachable where a per-ant computation happens. Every place this adapter needs the
/// map's size inside an iteration it either passes it through `reduce`'s seed or restructures to
/// avoid needing it — here, by handing the graph ant rows and columns as separate tensors and
/// letting it do the gathering, which is what an ONNX `GatherND` is for anyway.
///
/// **Filtering hills by owner needs the owner in the element**, which it is — `[r, c, owner]` — so
/// that one is ordinary.
fn reference_adapter() -> Vec<u8> {
    let plane = |points: J| json!({"tb.scatter": [points, {"var": "size"}, "int8"]});
    let rc = |src: &str| json!({"map": [{"var": src}, [{"var": "0"}, {"var": "1"}]]});

    let program = json!({
        "dialect": 1,
        "in": {
            "board": {"tb.reshape": [
                {"tb.stack": [[
                    plane(json!({"var": "mine"})),
                    plane(rc("foes")),
                    plane(json!({"var": "food"})),
                    {"tb.rle_expand": [{"var": "water.rle"}, {"var": "size"}, "int8"]},
                    plane(json!({"map": [
                        {"filter": [{"var": "hills"}, {"==": [{"var": "2"}, 0]}]},
                        [{"var": "0"}, {"var": "1"}]]})),
                    plane(json!({"map": [
                        {"filter": [{"var": "hills"}, {"!=": [{"var": "2"}, 0]}]},
                        [{"var": "0"}, {"var": "1"}]]}))
                ], 0, "int8"]},
                {"merge": [[1, 6], {"var": "size"}]}]},
            "ant_r": {"tb.tensor": [
                {"map": [{"var": "mine"}, {"var": "0"}]},
                [{"length": [{"var": "mine"}]}], "int32"]},
            "ant_c": {"tb.tensor": [
                {"map": [{"var": "mine"}, {"var": "1"}]},
                [{"length": [{"var": "mine"}]}], "int32"]}
        },
        // The graph gathers at the ant positions itself and answers [N, 5]. All the adapter does
        // on the way back is take the best channel per ant and name it.
        "out": {"map": [
            {"tb.argmax": [{"var": "outputs.policy"}, 1]},
            {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    });
    serde_json::to_vec(&program).unwrap()
}

/// A visibility-deriving adapter — ants/docs/protocol.md §8.3 names it as the concrete case the budget has
/// to accommodate, because deriving the view mask from ant positions is the expensive thing a real
/// adapter wants to do.
///
/// **Writing it is where the dialect's ergonomics fail, and the failure is silent.** The natural
/// expression is a nested iteration — for each ant, for each of the 317 disk offsets, a point —
/// and it cannot be written, because the inner `map`'s body sees only the offset and there is no
/// path back to the ant. Written that way it does not error: `{"var": "current.0"}` is `null`,
/// `null + dr` is `dr`, the points are the raw offsets, half of them are negative, `tb.scatter`
/// drops those as out of bounds, and the plane comes out **empty**. The adapter runs, costs a
/// quarter of a million operations, and derives nothing.
///
/// `reduce`'s seed is evaluated in the outer scope, so a *nested reduce* can capture the ant — but
/// the inner one has to accumulate the ant alongside the points, and there is no way to project a
/// field out of a computed value (`var` reads the document, not an expression's result), so the
/// ant cannot be dropped again on the way out.
///
/// What is left is unrolling the kernel: 317 literal offset pairs in the body, per ant. That is
/// what this builds, and the cost is the finding.
fn visibility_adapter_unrolled() -> Vec<u8> {
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
    let pts = json!({"reduce": [
        {"var": "mine"},
        {"merge": [{"var": "accumulator"}, pairs]},
        []]});
    let program = json!({
        "dialect": 1,
        "in": {"vis": {"tb.scatter": [pts, {"var": "size"}, "int8"]}},
        "out": {"map": [{"tb.argmax": [{"var": "outputs.policy"}, 1]},
                        {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    });
    serde_json::to_vec(&program).unwrap()
}

/// The same thing with the operator the measurement argues for: scatter the ants onto a plane and
/// dilate it once.
fn visibility_adapter_dilate() -> Vec<u8> {
    let program = json!({
        "dialect": 1,
        "in": {"vis": {"tb.dilate": [
            {"tb.scatter": [{"var": "mine"}, {"var": "size"}, "int8"]}, 77]}},
        "out": {"map": [{"tb.argmax": [{"var": "outputs.policy"}, 1]},
                        {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    });
    serde_json::to_vec(&program).unwrap()
}

fn policy(n: usize) -> Vec<(Arc<str>, Arc<Tensor>)> {
    // What the graph would answer: a score per direction per ant.
    let data = (0..n * 5).map(|i| ((i * 37) % 11) as f64).collect();
    vec![(Arc::from("policy"), Arc::new(Tensor::new(DType::F32, vec![n, 5], data)))]
}

#[test]
fn the_reference_adapter_runs_and_what_it_costs() {
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let o = obs();
    let n_ants = o["mine"].as_array().unwrap().len();
    let cells = o["size"][0].as_u64().unwrap() * o["size"][1].as_u64().unwrap();

    let (inputs, ops_in) = a.run_in(&o, 100_000_000);
    let inputs = inputs.expect("the in program failed");
    let names: Vec<&str> = inputs.iter().map(|(k, _)| &**k).collect();
    assert!(names.contains(&"board") && names.contains(&"ant_r") && names.contains(&"ant_c"));

    let board = &inputs.iter().find(|(k, _)| &**k == "board").unwrap().1;
    assert_eq!(board.shape, vec![1, 6, 128, 128]);
    assert_eq!(board.dtype, DType::I8);

    let (action, ops_out) = a.run_out(policy(n_ants), &o, 100_000_000);
    let action = action.expect("the out program failed");
    assert_eq!(
        action.as_array().unwrap().len(),
        n_ants,
        "len(action) must equal len(mine) -- ants/docs/protocol.md §6"
    );
    assert!(action.as_array().unwrap().iter().all(|v| {
        matches!(v.as_str(), Some("N" | "E" | "S" | "W" | "-"))
    }));

    println!("\n=== the reference Ants adapter, on a real worst-case observation ===");
    println!("  observation      {} bytes, {n_ants} ants, {cells} cells", OBS.len());
    println!("  in               {ops_in:>9} ops");
    println!("  out              {ops_out:>9} ops");
    println!("  the larger       {:>9} ops   <- what the budget must exceed", ops_in.max(ops_out));
    println!("  provisional      {PROVISIONAL_BUDGET:>9} ops   (ants/docs/protocol.md §4)");
    println!(
        "  headroom         {:>8.2}x",
        PROVISIONAL_BUDGET as f64 / ops_in.max(ops_out) as f64
    );
    println!("\n  where it goes: six planes of {cells} = {} produced, and the stack reads and", 6 * cells);
    println!("  produces them again. The ants and the JSON are the small part.");
}

#[test]
fn deriving_visibility_costs_what_it_costs() {
    let o = obs();
    let n_ants = o["mine"].as_array().unwrap().len();

    let unrolled = Adapter::parse(&visibility_adapter_unrolled()).unwrap();
    let (r1, ops1) = unrolled.run_in(&o, 100_000_000);
    let vis1 = r1.expect("unrolled failed");
    let marked1 = vis1[0].1.data.iter().filter(|&&v| v != 0.0).count();

    let dilated = Adapter::parse(&visibility_adapter_dilate()).unwrap();
    let (r2, ops2) = dilated.run_in(&o, 100_000_000);
    let vis2 = r2.expect("dilate failed");
    let marked2 = vis2[0].1.data.iter().filter(|&&v| v != 0.0).count();

    println!("\n=== deriving visibility -- ants/docs/protocol.md §8.3's concrete case ===");
    println!("  {n_ants} ants, a 317-cell disk, on a {} cell map", vis1[0].1.len());
    println!("  unrolled kernel   {ops1:>9} ops   {marked1} cells marked");
    println!("  tb.dilate         {ops2:>9} ops   {marked2} cells marked");
    println!("  the operator is   {:.0}x cheaper", ops1 as f64 / ops2 as f64);
    println!("  provisional       {PROVISIONAL_BUDGET:>9} ops (ants/docs/protocol.md §4)");

    // Both must actually derive a fog mask -- the first version of this test compared a program
    // against a program that silently did nothing, and cost 256,518 operations to derive an empty
    // plane.
    assert!(marked1 > 2000, "the unrolled kernel marked only {marked1} cells");
    assert!(marked2 > 2000, "tb.dilate marked only {marked2} cells");

    // And they do NOT agree, which is the stronger half of the argument for the operator.
    //
    // Ants maps wrap (ants/docs/protocol.md §1). `tb.dilate` wraps; the unrolled kernel produces
    // out-of-range coordinates near the edges and `tb.scatter` drops them, so a competitor's fog
    // mask is quietly wrong along every border. They cannot fix it: the modulo needs the map's
    // size, the map's size is not reachable inside the iteration that has the ant, and the
    // accumulator trick that would carry it cannot give the points back again -- there is no way
    // to project a field out of a computed value. The mask is not expressible correctly without
    // the operator.
    assert!(
        marked2 > marked1,
        "the unrolled kernel should lose the wrapped border cells: {marked1} vs {marked2}"
    );
    println!(
        "  the unrolled kernel loses {} border cells it cannot recover: the map wraps and the",
        marked2 - marked1
    );
    println!("  modulo needs a size that is not in scope where the ant is.");
}

#[test]
fn the_adapter_is_refused_at_the_provisional_budget_or_it_is_not() {
    // The assertion the number actually has to satisfy: run the reference adapter under the
    // published budget and see whether a real competitor would be struck every turn.
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let (r, ops) = a.run_in(&obs(), PROVISIONAL_BUDGET);
    match r {
        Ok(_) => println!("\nthe reference adapter fits 200,000 at {ops} ops"),
        Err(e) => println!("\nthe reference adapter does NOT fit 200,000: {e} (spent {ops})"),
    }
}

#[test]
fn what_an_operation_costs_in_wall_clock() {
    // The budget is a fairness rule, not a performance one -- but the number still has to be a
    // number a replica can afford K x seats times a turn. This is what one op costs on this
    // machine, so a proposed budget can be turned into milliseconds.
    let a = Adapter::parse(&reference_adapter()).unwrap();
    let o = obs();
    // Warm.
    for _ in 0..3 {
        let _ = a.run_in(&o, 100_000_000);
    }
    let n = 50;
    let t0 = std::time::Instant::now();
    let mut ops = 0;
    for _ in 0..n {
        let (r, c) = a.run_in(&o, 100_000_000);
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
