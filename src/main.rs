//! Axon — the model loader. `--dialect` prints the dialect identity without starting a server.

use std::sync::Arc;

use axon::{config::Config, dialect, server};

fn main() {
    if std::env::args().any(|a| a == "--dialect") {
        println!("dialect_version  {}", dialect::DIALECT_VERSION);
        println!("evaluator_digest {}", dialect::evaluator_digest());
        return;
    }
    let cfg = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("axon: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = server::serve(Arc::new(server::Axon::new(cfg))) {
        eprintln!("axon: {e}");
        std::process::exit(1);
    }
}
