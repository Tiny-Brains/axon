//! Shared fixtures: the two reference Ants adapters, and the observation they run against.
//!
//! `ants-observation.json` was dumped from the spike engine at turn 600 of a 128x128 match, when
//! the known-water mask is at its most fragmented: 90 ants, 670 water runs, 2,437 bytes.

// Each test binary uses a subset of these.
#![allow(dead_code)]

use axon::store::{DirStore, Kind, Store, digest};
use serde_json::{Value as J, json};

pub const OBS: &str = include_str!("../fixtures/ants-observation.json");
pub const MODEL: &[u8] = include_bytes!("../fixtures/ants-micro.onnx");
pub const DENSE_MODEL: &[u8] = include_bytes!("../fixtures/ants-dense.onnx");

pub fn obs() -> J {
    serde_json::from_str(OBS).expect("fixture")
}

pub fn sha256(bytes: &[u8]) -> String {
    digest(bytes)
}

/// The six int8 planes both adapters build: mine, foes, food, water, own hills, foreign hills.
fn board_planes() -> J {
    let plane = |points: J| json!({"tb.scatter": [points, {"var": "size"}, "int8"]});
    let rc = |src: &str| json!({"map": [{"var": src}, [{"var": "0"}, {"var": "1"}]]});
    let hills = |owned: J| {
        plane(json!({"map": [{"filter": [{"var": "hills"}, owned]},
                             [{"var": "0"}, {"var": "1"}]]}))
    };
    json!({"tb.reshape": [
        {"tb.stack": [[
            plane(json!({"var": "mine"})),
            plane(rc("foes")),
            plane(json!({"var": "food"})),
            {"tb.rle_expand": [{"var": "water.rle"}, {"var": "size"}, "int8"]},
            hills(json!({"==": [{"var": "2"}, 0]})),
            hills(json!({"!=": [{"var": "2"}, 0]}))
        ], 0, "int8"]},
        {"merge": [[1, 6], {"var": "size"}]}]})
}

/// Name the best channel per ant. Both adapters end this way.
fn name_moves(scores: J) -> J {
    json!({"map": [scores, {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]})
}

/// The adapter a competitor would submit for `ants-micro.onnx`: the planes, plus the ants' rows and
/// columns as separate tensors, so the graph gathers per ant itself and answers `[N, 5]`.
///
/// It is shaped that way because a `map` body cannot see the observation — `{"var": "size"}` inside
/// one is `null` — so the map's width is not reachable where a per-ant computation happens.
pub fn reference_adapter() -> Vec<u8> {
    let ant_axis = |i: &str| {
        json!({"tb.tensor": [{"map": [{"var": "mine"}, {"var": i}]},
                             [{"length": [{"var": "mine"}]}], "int32"]})
    };
    adapter(
        json!({"board": board_planes(), "ant_r": ant_axis("0"), "ant_c": ant_axis("1")}),
        name_moves(json!({"tb.argmax": [{"var": "outputs.policy"}, 1]})),
    )
}

/// The adapter for the batchable model: the same planes, but the graph answers a dense `[5, H, W]`
/// policy map and the per-ant gather happens here, on the way back.
///
/// It is the program that could not be written before `tb.get`: the flat indices need the map's
/// width inside an iteration over the ants, `reduce`'s seed is the only channel from the outer
/// scope into a body, and carrying the width in the accumulator means the points have to be
/// projected back out of it.
pub fn dense_adapter() -> Vec<u8> {
    let acc = |k: &str| json!({"tb.get": [{"var": "accumulator"}, k]});
    let flat = json!({"tb.get": [
        {"reduce": [
            {"var": "observation.mine"},
            {"w": acc("w"),
             "idx": {"merge": [acc("idx"),
                               [{"+": [{"*": [acc("w"), {"var": "current.0"}]},
                                       {"var": "current.1"}]}]]}},
            {"w": {"var": "observation.size.1"}, "idx": []}]},
        "idx"]});
    let cells = json!({"*": [{"var": "observation.size.0"}, {"var": "observation.size.1"}]});

    adapter(
        json!({"board": board_planes()}),
        name_moves(json!({"tb.argmax": [
            {"tb.transpose": [
                {"tb.gather": [
                    {"tb.reshape": [{"var": "outputs.policy"}, [5, cells]]},
                    flat, 1]},
                [1, 0]]},
            1]})),
    )
}

fn adapter(in_program: J, out_program: J) -> Vec<u8> {
    serde_json::to_vec(&json!({"dialect": 1, "in": in_program, "out": out_program})).unwrap()
}

/// A directory store seeded with both models and both adapters, in a temp dir the caller drops.
pub struct Fixture {
    pub dir: std::path::PathBuf,
    pub weights_hash: String,
    pub adapter_hash: String,
    pub dense_weights_hash: String,
    pub dense_adapter_hash: String,
}

impl Fixture {
    pub fn new(tag: &str) -> Fixture {
        let dir = std::env::temp_dir().join(format!("axon-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = DirStore::new(dir.clone());
        let put = |kind, bytes: &[u8]| {
            let h = sha256(bytes);
            store.put(kind, &h, bytes).unwrap();
            h
        };
        Fixture {
            weights_hash: put(Kind::Weights, MODEL),
            adapter_hash: put(Kind::Adapter, &reference_adapter()),
            dense_weights_hash: put(Kind::Weights, DENSE_MODEL),
            dense_adapter_hash: put(Kind::Adapter, &dense_adapter()),
            dir,
        }
    }

    pub fn config(&self, mode: axon::config::Mode) -> axon::config::Config {
        axon::config::Config {
            mode,
            bind: "127.0.0.1:0".into(),
            auth_token: None,
            memory_budget_bytes: 512 * 1024 * 1024,
            max_weights_bytes: 96 * 1024 * 1024,
            max_adapter_bytes: 4 * 1024 * 1024,
            default_idle_ttl_s: 900,
            threads: 2,
            max_in_flight: 1,
            store: axon::config::StoreSpec::Dir(self.dir.clone()),
            fetch_allow_hosts: Vec::new(),
        }
    }

    pub fn model_ref(&self) -> J {
        json!({"weights_hash": self.weights_hash, "adapter_hash": self.adapter_hash})
    }

    pub fn dense_ref(&self) -> J {
        json!({"weights_hash": self.dense_weights_hash, "adapter_hash": self.dense_adapter_hash})
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
