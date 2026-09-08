//! Axon — the Model Loader. docs/design.md.

use std::sync::Arc;

fn main() {
    if std::env::args().any(|a| a == "--dialect") {
        println!("dialect_version  {}", axon::dialect::DIALECT_VERSION);
        println!("evaluator_digest {}", axon::dialect::evaluator_digest());
        return;
    }
    let cfg = match axon::config::Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("axon: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = axon::server::serve(Arc::new(axon::server::Axon::new(cfg))) {
        eprintln!("axon: {e}");
        std::process::exit(1);
    }
}
