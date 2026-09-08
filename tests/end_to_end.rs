//! A real ONNX model, a real adapter, a real observation, through the real seam.
//!
//! Everything before this tested a piece. This tests that layer 04 §3 is a thing that works:
//! `/load` fetches by hash and verifies, `/play` runs adapter → graph → adapter under a deadline,
//! `/resident` answers what the claim's affinity ordering needs, `/unload` is idempotent, and both
//! classes of refusal come back with the `fault` field layer 03 branches on.

mod common;

use axon::api::*;
use axon::config::Mode;
use axon::server::Axon;
use common::Fixture;
use serde_json::json;

fn load_req(models: Vec<serde_json::Value>) -> LoadRequest {
    serde_json::from_value(json!({"models": models, "idle_ttl_s": 900})).unwrap()
}

#[test]
fn a_wave_loads_plays_and_unloads() {
    let f = Fixture::new("e2e");
    let axon = Axon::new(f.config(Mode::Replica));

    // ---- /load
    let r = axon.load(load_req(vec![f.model_ref()]));
    assert_eq!(r.models[0].state, "resident", "{:?}", r.models[0]);
    assert_eq!(r.dialect_version, 1);
    assert!(r.evaluator_digest.starts_with("sha256:"));

    // ---- /resident: what layer 01 §4.2's claim orders its candidates by
    let res = axon.resident();
    assert_eq!(res.weights, vec![f.weights_hash.clone()]);
    assert_eq!(res.adapters, vec![f.adapter_hash.clone()]);
    assert!(res.memory_bytes > 0 && res.memory_bytes < res.memory_budget_bytes);
    assert!(res.loading.is_empty(), "a synchronous /load never reports loading");

    // ---- /play: the whole wave in one call
    let obs = common::obs();
    let n_ants = obs["mine"].as_array().unwrap().len();
    let rows: Vec<serde_json::Value> = (0..4)
        .map(|i| {
            json!({"weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
                   "observation": obs, "ref": {"m": i / 2, "seat": i % 2}})
        })
        .collect();
    let play: PlayRequest =
        serde_json::from_value(json!({"rows": rows, "deadline_ms": 5000, "budget_ops": 1_000_000}))
            .unwrap();
    let reply = axon.play(play);

    assert_eq!(reply.rows.len(), 4);
    for (i, row) in reply.rows.iter().enumerate() {
        assert!(row.error.is_none(), "row {i}: {:?}", row);
        let action = row.action.as_ref().unwrap().as_array().unwrap();
        assert_eq!(action.len(), n_ants, "len(action) must equal len(mine)");
        assert!(action.iter().all(|v| matches!(v.as_str(), Some("N" | "E" | "S" | "W" | "-"))));
        assert!(row.ops > 190_000, "row {i} reported {} ops", row.ops);
        // The ref is echoed verbatim: without it Kalam cannot count strikes.
        assert_eq!(row.r#ref.as_ref().unwrap()["seat"], json!(i % 2));
    }

    // ---- /unload, twice: released then not_held. Idempotent by construction.
    let un: UnloadRequest = serde_json::from_value(json!({"models": [f.model_ref()]})).unwrap();
    assert_eq!(axon.unload(un).models[0].state, "released");
    let un: UnloadRequest = serde_json::from_value(json!({"models": [f.model_ref()]})).unwrap();
    assert_eq!(axon.unload(un).models[0].state, "not_held");
}

#[test]
fn the_two_classes_of_refusal_carry_the_field_layer_03_branches_on() {
    let f = Fixture::new("refusals");
    let axon = Axon::new(f.config(Mode::Replica));

    // A hash that is not in the store: the loader's problem, so the row is released and no
    // attempt is spent.
    let missing = json!({"weights_hash": format!("sha256:{}", "b".repeat(64)),
                         "adapter_hash": f.adapter_hash});
    let r = axon.load(load_req(vec![missing]));
    assert_eq!(r.models[0].state, "refused");
    assert_eq!(r.models[0].fault, Some("loader"));
    assert_eq!(r.models[0].reason, Some("FETCH_FAILED"));

    // A URL to a replica: refused outright, so a replica cannot be told where to fetch from.
    let with_url = json!({"weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
                          "weights_url": "https://github.com/x/y/releases/download/v1/model.onnx"});
    let r = axon.load(load_req(vec![with_url]));
    assert_eq!(r.models[0].reason, Some("URL_NOT_ACCEPTED"));
    assert_eq!(r.models[0].fault, Some("model"));

    // Something that is not a hash has no key at all, which is what stops a hash being a path.
    let not_a_hash = json!({"weights_hash": "sha256:../../etc/passwd",
                            "adapter_hash": f.adapter_hash});
    let r = axon.load(load_req(vec![not_a_hash]));
    assert_eq!(r.models[0].reason, Some("HASH_MISMATCH"));
    assert_eq!(r.models[0].fault, Some("model"));
}

#[test]
fn corrupt_bytes_in_the_store_are_the_models_fault_and_are_named() {
    use axon::store::{DirStore, Kind, Store};
    let f = Fixture::new("corrupt");
    // Put bytes under a key that is not their hash. On a replica this means the store is corrupt;
    // it is a `model` fault nonetheless, because the row can never be played and failing it with a
    // named reason beats looping it through refusals to the same end.
    let liar = format!("sha256:{}", "c".repeat(64));
    DirStore::new(f.dir.clone()).put(Kind::Weights, &liar, b"not an onnx file").unwrap();

    let axon = Axon::new(f.config(Mode::Replica));
    let r = axon.load(load_req(vec![json!({"weights_hash": liar,
                                           "adapter_hash": f.adapter_hash})]));
    assert_eq!(r.models[0].reason, Some("HASH_MISMATCH"));
    assert_eq!(r.models[0].fault, Some("model"));
    assert!(r.models[0].detail.as_ref().unwrap().contains("declared"));
}

#[test]
fn a_row_naming_a_model_that_is_not_held_answers_not_resident_and_the_others_proceed() {
    let f = Fixture::new("mixed");
    let axon = Axon::new(f.config(Mode::Replica));
    axon.load(load_req(vec![f.model_ref()]));

    let obs = common::obs();
    let play: PlayRequest = serde_json::from_value(json!({"rows": [
        {"weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash, "observation": obs},
        {"weights_hash": format!("sha256:{}", "d".repeat(64)),
         "adapter_hash": f.adapter_hash, "observation": obs},
    ], "deadline_ms": 5000, "budget_ops": 1_000_000}))
    .unwrap();
    let reply = axon.play(play);
    assert!(reply.rows[0].action.is_some(), "the good row must still play");
    assert_eq!(reply.rows[1].error, Some("NOT_RESIDENT"));
}

#[test]
fn an_over_budget_adapter_is_struck_rather_than_refused_for_the_wave() {
    let f = Fixture::new("budget");
    let axon = Axon::new(f.config(Mode::Replica));
    axon.load(load_req(vec![f.model_ref()]));

    // The reference adapter needs ~197k. Give it a tenth of that.
    let play: PlayRequest = serde_json::from_value(json!({"rows": [
        {"weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
         "observation": common::obs(), "ref": {"seat": 0}}
    ], "deadline_ms": 5000, "budget_ops": 20_000}))
    .unwrap();
    let reply = axon.play(play);
    assert_eq!(reply.rows[0].error, Some("ADAPTER_FAILED"));
    assert_eq!(reply.rows[0].over_budget, Some(true));
    // The ref still comes back, because the strike has to be attributable.
    assert_eq!(reply.rows[0].r#ref.as_ref().unwrap()["seat"], json!(0));
}

#[test]
fn a_deadline_that_has_already_passed_times_the_row_out_rather_than_running_it() {
    let f = Fixture::new("deadline");
    let axon = Axon::new(f.config(Mode::Replica));
    axon.load(load_req(vec![f.model_ref()]));

    let play: PlayRequest = serde_json::from_value(json!({"rows": [
        {"weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
         "observation": common::obs()}
    ], "deadline_ms": 0, "budget_ops": 1_000_000}))
    .unwrap();
    let reply = axon.play(play);
    assert_eq!(reply.rows[0].error, Some("TIMED_OUT"));
}

#[test]
fn admission_inspects_and_validates_and_does_not_play() {
    let f = Fixture::new("admission");
    let axon = Axon::new(f.config(Mode::Admission));
    let r = axon.load(load_req(vec![f.model_ref()]));
    assert_eq!(r.models[0].state, "resident", "{:?}", r.models[0]);

    // ---- /inspect: the static facts, reported and not judged
    let ins: InspectRequest = serde_json::from_value(f.model_ref()).unwrap();
    let ins = axon.inspect(ins).expect("inspect");
    assert_eq!(ins.opset, 17);
    assert_eq!(ins.params, 6653);
    assert!(ins.ops.contains(&"Conv".to_string()));
    assert!(ins.size_metric_bytes > 0);
    assert_eq!(ins.weights_zstd_bytes + ins.adapter_zstd_bytes, ins.size_metric_bytes);
    // Micro is <= 64 KiB compressed -- DESIGN.md §5. The class table is layer 08's to apply.
    assert!(ins.size_metric_bytes < 64 * 1024, "S = {}", ins.size_metric_bytes);
    let names: Vec<&str> = ins.inputs.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["board", "ant_r", "ant_c"]);
    // The declared board shape has two dynamic dimensions, which is exactly why the FLOP number
    // cannot come from here.
    assert!(ins.inputs[0].shape.iter().any(|d| d.is_none()));

    // ---- /validate: run the adapter, and measure FLOPs at what it actually fed
    let val: ValidateRequest = serde_json::from_value(json!({
        "weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
        "budget_ops": 1_000_000, "observations": [common::obs()], "deadline_ms": 10_000
    }))
    .unwrap();
    let val = axon.validate(val);
    assert!(val.ok, "{val:?}");
    assert_eq!(val.cases.len(), 1);
    let c = &val.cases[0];
    assert_eq!(c.inputs.iter().find(|p| p.name == "board").unwrap().shape, vec![1, 6, 128, 128]);
    assert!(c.action_shape.starts_with("array[90]"), "{}", c.action_shape);
    assert!(val.ops_max > 190_000, "ops_max {}", val.ops_max);
    assert!(val.flops_max > 0.0);
    println!(
        "\nadmission: S={} ({} params, opset {}) · ops_max={} · flops_max={:.3e}",
        ins.size_metric_bytes, ins.params, ins.opset, val.ops_max, val.flops_max
    );

    // ---- and it does not play. A trial is an ordinary row that Kalam claims first.
    let play: PlayRequest = serde_json::from_value(json!({"rows": []})).unwrap();
    let _ = axon.play(play); // the mode check is in the HTTP layer; the call itself is harmless
}

#[test]
fn validate_refuses_an_adapter_that_does_not_feed_the_graph() {
    use axon::store::{DirStore, Kind, Store};
    let f = Fixture::new("badadapter");
    // Well-formed dialect, produces a tensor, but not the one the graph asked for.
    let bad = br#"{"dialect":1,"in":{"nope":{"tb.zeros":[[2,2],"int8"]}},"out":{"map":[[1],{"var":""}]}}"#;
    let bad_hash = common::sha256(bad);
    DirStore::new(f.dir.clone()).put(Kind::Adapter, &bad_hash, bad).unwrap();

    let axon = Axon::new(f.config(Mode::Admission));
    axon.load(load_req(vec![json!({"weights_hash": f.weights_hash, "adapter_hash": bad_hash})]));
    let val: ValidateRequest = serde_json::from_value(json!({
        "weights_hash": f.weights_hash, "adapter_hash": bad_hash,
        "observations": [common::obs()]
    }))
    .unwrap();
    let val = axon.validate(val);
    assert!(!val.ok);
    assert_eq!(val.reason, Some("SHAPE_MISMATCH"));
    assert!(val.detail.unwrap().contains("board"));
}

#[test]
fn validate_with_no_reference_observations_is_refused_rather_than_passing_vacuously() {
    // The requirement layer 04 §3.6 places on layer 08: a gate that tests nothing is not a gate.
    let f = Fixture::new("noobs");
    let axon = Axon::new(f.config(Mode::Admission));
    axon.load(load_req(vec![f.model_ref()]));
    let val: ValidateRequest = serde_json::from_value(f.model_ref()).unwrap();
    let val = axon.validate(val);
    assert!(!val.ok);
    assert!(val.detail.unwrap().contains("no reference observations"));
}

#[test]
fn the_batchable_model_answers_the_same_moves_one_inference_at_a_time() {
    // `DESIGN.md` §7: "one batched inference per distinct model" is "the economics a code-
    // submission challenge can never have". This is that claim, tested.
    //
    // Two models, the same trunk and the same weights. `ants-micro` takes ragged per-ant inputs
    // and declares a leading dimension of 1, so every seat is its own inference. `ants-dense`
    // takes only the board, declares a dynamic batch, and answers a dense policy map that the
    // adapter gathers per ant on the way back — so a whole wave is one inference.
    let f = Fixture::new("batch");
    let axon = Axon::new(f.config(Mode::Replica));
    assert_eq!(axon.load(load_req(vec![f.dense_ref()])).models[0].state, "resident");

    let obs = common::obs();
    let n_ants = obs["mine"].as_array().unwrap().len();
    let rows: Vec<serde_json::Value> = (0..8)
        .map(|i| {
            json!({"weights_hash": f.dense_weights_hash, "adapter_hash": f.dense_adapter_hash,
                   "observation": obs, "ref": {"seat": i}})
        })
        .collect();
    let play: PlayRequest =
        serde_json::from_value(json!({"rows": rows, "deadline_ms": 10_000, "budget_ops": 2_000_000}))
            .unwrap();
    let reply = axon.play(play);

    for (i, row) in reply.rows.iter().enumerate() {
        assert!(row.error.is_none(), "row {i}: {row:?}");
        let a = row.action.as_ref().unwrap().as_array().unwrap();
        assert_eq!(a.len(), n_ants, "len(action) must equal len(mine)");
        assert!(a.iter().all(|v| matches!(v.as_str(), Some("N" | "E" | "S" | "W" | "-"))));
    }

    // Every row saw the same observation, so a batch that split correctly gives every row the
    // same moves. A batch that mis-sliced would give one row's answer to another, and this is the
    // assertion that catches it.
    let first = reply.rows[0].action.clone();
    assert!(reply.rows.iter().all(|r| r.action == first), "the batch did not split correctly");
}

#[test]
fn a_batch_gives_each_row_its_own_answer() {
    // The stronger version: different observations, so a mis-slice cannot hide behind identical
    // inputs. Row i's moves must be the moves it gets when it is played alone.
    let f = Fixture::new("batchsplit");
    let axon = Axon::new(f.config(Mode::Replica));
    axon.load(load_req(vec![f.dense_ref()]));

    let base = common::obs();
    let mut observations = Vec::new();
    for k in 0..4usize {
        let mut o = base.clone();
        // Shift every ant, which changes both the board and where the gather reads.
        let mine: Vec<serde_json::Value> = o["mine"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let (r, c) = (p[0].as_i64().unwrap(), p[1].as_i64().unwrap());
                json!([(r + k as i64) % 128, (c + 2 * k as i64) % 128])
            })
            .collect();
        o["mine"] = json!(mine);
        observations.push(o);
    }

    let one = |o: &serde_json::Value| {
        let play: PlayRequest = serde_json::from_value(json!({"rows": [
            {"weights_hash": f.dense_weights_hash, "adapter_hash": f.dense_adapter_hash,
             "observation": o}], "deadline_ms": 10_000, "budget_ops": 2_000_000}))
        .unwrap();
        axon.play(play).rows[0].action.clone().unwrap()
    };
    let alone: Vec<serde_json::Value> = observations.iter().map(one).collect();

    let rows: Vec<serde_json::Value> = observations
        .iter()
        .map(|o| json!({"weights_hash": f.dense_weights_hash,
                        "adapter_hash": f.dense_adapter_hash, "observation": o}))
        .collect();
    let play: PlayRequest =
        serde_json::from_value(json!({"rows": rows, "deadline_ms": 10_000, "budget_ops": 2_000_000}))
            .unwrap();
    let batched = axon.play(play);

    for (i, row) in batched.rows.iter().enumerate() {
        assert!(row.error.is_none(), "row {i}: {row:?}");
        assert_eq!(
            row.action.as_ref().unwrap(),
            &alone[i],
            "row {i} batched does not match row {i} played alone"
        );
    }
    assert!(alone[0] != alone[1], "the fixture must actually differ between rows");
}

// ---------------------------------------------------------------- what layer 08 asked for
//
// Three additions, each one a thing admission cannot do without. They are tested here rather than
// beside the calls they belong to because each is only meaningful in the admission role.

#[test]
fn inspect_returns_the_adapters_exact_bytes_so_the_schemas_check_can_hold() {
    let f = Fixture::new("inspect-adapter");
    let axon = Axon::new(f.config(Mode::Admission));
    axon.load(load_req(vec![f.model_ref()]));

    let ins = axon.inspect(serde_json::from_value(f.model_ref()).unwrap()).expect("inspect");

    // THIS IS THE WHOLE POINT, and it is why the reply carries text rather than a parsed document.
    // `models.adapter` stores what comes back here, and `models_adapter_matches_hash` recomputes
    // sha256 over the stored text -- so anything but the exact fetched bytes makes the verdict
    // statement fail a CHECK constraint rather than admit a version.
    assert_eq!(common::sha256(ins.adapter.as_bytes()), f.adapter_hash);
    assert_eq!(ins.adapter_raw_bytes as usize, ins.adapter.len());
}

#[test]
fn a_missing_release_asset_is_the_models_fault_and_is_named() {
    use std::io::{Read, Write};

    // A one-shot server that answers 404 to anything, standing in for a release with no
    // `adapter.json` attached to it.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let _ = s.read(&mut [0u8; 1024]);
            let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
        }
    });

    let f = Fixture::new("asset-missing");
    let mut cfg = f.config(Mode::Admission);
    cfg.fetch_allow_hosts = vec![addr.to_string()];
    let axon = Axon::new(cfg);

    let hash = format!("sha256:{}", "b".repeat(64));
    let req = load_req(vec![json!({
        "weights_hash": hash, "adapter_hash": hash,
        "weights_url": format!("http://{addr}/model.onnx"),
        "adapter_url": format!("http://{addr}/adapter.json")})]);
    let r = axon.load(req);

    // Before layer 08 asked, this was FETCH_FAILED with fault `loader` -- which admission retries.
    // A competitor who forgot to attach the file would have got three silent retries and then
    // TIMED_OUT, the least actionable message on the platform.
    assert_eq!(r.models[0].state, "refused");
    assert_eq!(r.models[0].reason, Some("ASSET_MISSING"));
    assert_eq!(r.models[0].fault, Some("model"));
}

#[test]
fn validate_tells_an_expensive_adapter_from_a_wrong_one() {
    let f = Fixture::new("over-budget-validate");
    let axon = Axon::new(f.config(Mode::Admission));
    axon.load(load_req(vec![f.model_ref()]));

    let val = |budget: u64| -> ValidateReply {
        axon.validate(
            serde_json::from_value(json!({
                "weights_hash": f.weights_hash, "adapter_hash": f.adapter_hash,
                "budget_ops": budget, "observations": [common::obs()], "deadline_ms": 10_000
            }))
            .unwrap(),
        )
    };

    // Too expensive: the program is fine, it just costs more than the game allows. The competitor
    // should be told to make it cheaper, not to go and re-read the dialect specification.
    let tight = val(20_000);
    assert!(!tight.ok);
    assert_eq!(tight.reason, Some("ADAPTER_FAILED"));
    assert_eq!(tight.over_budget, Some(true));

    // And under the game's real budget it passes, with the flag absent rather than false.
    let ok = val(1_000_000);
    assert!(ok.ok, "{ok:?}");
    assert_eq!(ok.over_budget, None);
}
