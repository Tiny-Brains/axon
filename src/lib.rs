//! Axon — the Model Loader.
//!
//! Specified by docs/design.md. Built in the order the risk is in: the dialect
//! first, because it carries the competitor-facing budget and is the part no other system has;
//! then the seam; then the runtime.

pub mod api;
pub mod config;
pub mod dialect;
pub mod model;
pub mod onnx_meta;
pub mod residency;
pub mod server;
pub mod store;
