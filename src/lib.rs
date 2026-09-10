//! Axon — the model loader. One binary, two roles: an admission process that verifies and mirrors
//! submissions, and a replica process that holds models resident and answers Kalam's wave.

pub mod api;
pub mod config;
pub mod dialect;
pub mod model;
pub mod residency;
pub mod server;
pub mod store;
