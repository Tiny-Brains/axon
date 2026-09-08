//! Shared fixtures: the reference Ants adapter, and the observation it runs against.

use serde_json::{json, Value as J};

pub const OBS: &str = include_str!("../fixtures/ants-observation.json");
pub const MODEL: &[u8] = include_bytes!("../fixtures/ants-micro.onnx");
pub const DENSE_MODEL: &[u8] = include_bytes!("../fixtures/ants-dense.onnx");

pub fn obs() -> J {
    serde_json::from_str(OBS).expect("fixture")
}

/// The adapter a competitor would submit for `ants-micro.onnx`: six int8 planes at the map's size,
/// plus the ants' rows and columns, answering the graph's three inputs; and on the way back, the
/// best of five channels per ant, named.
pub fn reference_adapter() -> Vec<u8> {
    let plane = |points: J| json!({"tb.scatter": [points, {"var": "size"}, "int8"]});
    let rc = |src: &str| json!({"map": [{"var": src}, [{"var": "0"}, {"var": "1"}]]});

    serde_json::to_vec(&json!({
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
        "out": {"map": [
            {"tb.argmax": [{"var": "outputs.policy"}, 1]},
            {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    }))
    .unwrap()
}

/// The adapter for the batchable model: the same six planes, but the graph answers a dense policy
/// map and the per-ant gather happens here, on the way back.
///
/// It is the program that could not be written before `tb.get` existed. The flat indices need the
/// map's width inside an iteration over the ants, `reduce`'s seed is the only channel from outer
/// scope into a body, and carrying the width in the accumulator means the accumulator is an object
/// that the points then have to be projected back out of. `var` reads the document; `tb.get` reads
/// a value.
pub fn dense_adapter() -> Vec<u8> {
    let plane = |points: J| json!({"tb.scatter": [points, {"var": "size"}, "int8"]});
    let rc = |src: &str| json!({"map": [{"var": src}, [{"var": "0"}, {"var": "1"}]]});

    // reduce(mine) carrying the width, then project the points back out.
    let flat = json!({"tb.get": [
        {"reduce": [
            {"var": "observation.mine"},
            {"w": {"tb.get": [{"var": "accumulator"}, "w"]},
             "idx": {"merge": [
                 {"tb.get": [{"var": "accumulator"}, "idx"]},
                 [{"+": [{"*": [{"tb.get": [{"var": "accumulator"}, "w"]}, {"var": "current.0"}]},
                         {"var": "current.1"}]}]]}},
            {"w": {"var": "observation.size.1"}, "idx": []}]},
        "idx"]});

    serde_json::to_vec(&json!({
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
                {"merge": [[1, 6], {"var": "size"}]}]}
        },
        // policy is [5, H, W] for this row. Flatten the map, gather at the ants' flat positions,
        // transpose to [N, 5], take the best channel per ant, and name it.
        "out": {"map": [
            {"tb.argmax": [
                {"tb.transpose": [
                    {"tb.gather": [
                        {"tb.reshape": [{"var": "outputs.policy"},
                                        [5, {"*": [{"var": "observation.size.0"},
                                                   {"var": "observation.size.1"}]}]]},
                        flat, 1]},
                    [1, 0]]},
                1]},
            {"tb.at": [["N", "E", "S", "W", "-"], {"var": ""}]}]}
    }))
    .unwrap()
}

pub fn sha256(bytes: &[u8]) -> String {
    axon::store::digest(bytes)
}

/// A directory store seeded with the model and the adapter, in a temp dir the caller drops.
pub struct Fixture {
    pub dir: std::path::PathBuf,
    pub weights_hash: String,
    pub adapter_hash: String,
    pub dense_weights_hash: String,
    pub dense_adapter_hash: String,
}

impl Fixture {
    pub fn new(tag: &str) -> Fixture {
        use axon::store::{DirStore, Kind, Store};
        let dir = std::env::temp_dir().join(format!("axon-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = DirStore::new(dir.clone());
        let adapter = reference_adapter();
        let weights_hash = sha256(MODEL);
        let adapter_hash = sha256(&adapter);
        s.put(Kind::Weights, &weights_hash, MODEL).unwrap();
        s.put(Kind::Adapter, &adapter_hash, &adapter).unwrap();
        let dense = dense_adapter();
        let dense_weights_hash = sha256(DENSE_MODEL);
        let dense_adapter_hash = sha256(&dense);
        s.put(Kind::Weights, &dense_weights_hash, DENSE_MODEL).unwrap();
        s.put(Kind::Adapter, &dense_adapter_hash, &dense).unwrap();
        Fixture { dir, weights_hash, adapter_hash, dense_weights_hash, dense_adapter_hash }
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
            adapter_threads: 2,
            max_in_flight: 1,
            store: axon::config::StoreSpec::Dir(self.dir.clone()),
            fetch_allow_hosts: Vec::new(),
        }
    }

    pub fn model_ref(&self) -> serde_json::Value {
        json!({"weights_hash": self.weights_hash, "adapter_hash": self.adapter_hash})
    }

    pub fn dense_ref(&self) -> serde_json::Value {
        json!({"weights_hash": self.dense_weights_hash, "adapter_hash": self.dense_adapter_hash})
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
