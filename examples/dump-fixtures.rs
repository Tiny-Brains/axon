//! Write the test fixtures into a directory store, so a running axon can be driven with the same
//! bytes the test suite uses. Development only.
//!
//!     cargo run --release --example dump-fixtures -- /path/to/store

#[path = "../tests/common/mod.rs"]
mod common;

use axon::store::{DirStore, Kind, Store, digest};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: dump-fixtures <store-dir>");
    let store = DirStore::new(dir.clone().into());
    let put = |kind, bytes: &[u8]| {
        let h = digest(bytes);
        store.put(kind, &h, bytes).unwrap();
        h
    };
    let out = serde_json::json!({
        "ragged": { "weights": put(Kind::Weights, common::MODEL),
                    "adapter": put(Kind::Adapter, &common::reference_adapter()) },
        "dense":  { "weights": put(Kind::Weights, common::DENSE_MODEL),
                    "adapter": put(Kind::Adapter, &common::dense_adapter()) },
    });
    std::fs::write(format!("{dir}/hashes.json"), serde_json::to_vec_pretty(&out).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
