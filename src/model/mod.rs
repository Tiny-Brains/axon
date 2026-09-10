//! A resident model: the ONNX session, its static facts, and one batched run.
//!
//! A *model* on this seam is the pair `(weights_hash, adapter_hash)`; two pairs sharing a
//! `weights_hash` share one session, which is why `/resident` answers weights hashes.

mod meta;

pub use meta::{size_metric, GraphFacts};

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ort::session::{builder::GraphOptimizationLevel, RunOptions, Session, SessionInputValue};
use ort::value::{Outlet, Tensor as OrtTensor, Value as OrtValue};

use crate::dialect::{DType, Ports, Tensor};

pub struct Graph {
    pub session: Mutex<Session>,
    pub facts: GraphFacts,
    pub bytes_resident: u64,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Port {
    pub name: String,
    pub dtype: String,
    /// A dynamic dimension is `null` — what the competitor declared, not what the adapter feeds.
    pub shape: Vec<Option<i64>>,
}

#[derive(Debug)]
pub enum GraphError {
    Invalid(String),
    TooLarge { bytes: u64, limit: u64 },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Invalid(m) => write!(f, "{m}"),
            GraphError::TooLarge { bytes, limit } => {
                write!(f, "the weights are {bytes} bytes, over the {limit} byte ceiling")
            }
        }
    }
}

impl Graph {
    pub fn build(bytes: &[u8], threads: usize, max_bytes: u64) -> Result<Graph, GraphError> {
        if bytes.len() as u64 > max_bytes {
            return Err(GraphError::TooLarge { bytes: bytes.len() as u64, limit: max_bytes });
        }
        let build = || -> ort::Result<Session> {
            let mut b = Session::builder()?;
            b = b.with_optimization_level(GraphOptimizationLevel::Level3)?;
            b = b.with_intra_threads(threads.max(1))?;
            b.commit_from_memory(bytes)
        };
        let session = build().map_err(|e| {
            GraphError::Invalid(format!("the file parses but a session cannot be built: {e}"))
        })?;

        Ok(Graph {
            bytes_resident: bytes.len() as u64,
            facts: meta::read(bytes),
            inputs: session.inputs().iter().map(port).collect(),
            outputs: session.outputs().iter().map(port).collect(),
            session: Mutex::new(session),
        })
    }

    /// Whether this graph can be run on several rows at once.
    ///
    /// Shape agreement between rows is not sufficient: two seats with the same observation produce
    /// identical feeds and still fail if the board was exported with a leading dimension of exactly
    /// one. The condition is that the graph *declares* a dynamic leading dimension on every port,
    /// which is visible statically and so decided at load rather than by a failed run.
    pub fn batchable(&self) -> bool {
        let dynamic_leading = |p: &Port| matches!(p.shape.first(), Some(None));
        !self.inputs.is_empty()
            && self.inputs.iter().all(dynamic_leading)
            && self.outputs.iter().all(dynamic_leading)
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|p| p.name.as_str()).collect()
    }

    /// Run the graph once. `feeds` is one named tensor per input; the answer is one named tensor per
    /// output, always float32 — one element type on that boundary is one fewer thing for an adapter
    /// to get wrong.
    pub fn run(&self, feeds: &Ports, deadline: Instant) -> Result<Ports, String> {
        let mut session = self.session.lock().map_err(|_| "the session is poisoned".to_string())?;

        let given: BTreeMap<&str, &Arc<Tensor>> = feeds.iter().map(|(k, v)| (&**k, v)).collect();
        let mut values: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(self.inputs.len());
        for p in &self.inputs {
            let t = given.get(p.name.as_str()).ok_or_else(|| {
                format!("the adapter produced no tensor for the graph's input '{}'", p.name)
            })?;
            values.push((Cow::Owned(p.name.clone()), to_ort(t)?.into()));
        }

        // ORT is told to stop rather than waited on, so a slow graph is cut off and the other rows
        // in the call are unaffected.
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("TIMED_OUT".into());
        }
        let opts = Arc::new(RunOptions::new().map_err(|e| e.to_string())?);
        let watcher = opts.clone();
        let stop = std::thread::spawn(move || {
            std::thread::sleep(remaining);
            let _ = watcher.terminate();
        });
        let out = session.run_with_options(values, &opts);
        drop(stop); // the watcher exits on its own; terminating a finished run is a no-op
        let out = out.map_err(|e| e.to_string())?;

        let mut answer = Vec::with_capacity(self.outputs.len());
        for p in &self.outputs {
            let v = out.get(p.name.as_str()).ok_or_else(|| format!("no output '{}'", p.name))?;
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("output '{}' is not float32: {e}", p.name))?;
            answer.push((
                Arc::from(p.name.as_str()),
                Arc::new(Tensor::new(
                    DType::F32,
                    shape.iter().map(|&d| d.max(0) as usize).collect(),
                    data.iter().map(|&x| x as f64).collect(),
                )),
            ));
        }
        Ok(answer)
    }
}

fn port(o: &Outlet) -> Port {
    Port {
        name: o.name().to_string(),
        dtype: o
            .dtype()
            .tensor_type()
            .map(|t| format!("{t:?}").to_lowercase())
            .unwrap_or_else(|| "non-tensor".into()),
        shape: o
            .dtype()
            .tensor_shape()
            .map(|d| d.iter().map(|&x| if x < 0 { None } else { Some(x) }).collect())
            .unwrap_or_default(),
    }
}

/// The dialect holds every element as `f64` so an operator is written once rather than five times.
/// This is the one place that becomes the graph's element type.
fn to_ort(t: &Tensor) -> Result<OrtValue, String> {
    let shape: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
    let err = |e: ort::Error| e.to_string();
    macro_rules! narrowed {
        ($ty:ty) => {{
            let d: Vec<$ty> = t.data.iter().map(|&v| v as $ty).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }};
    }
    Ok(match t.dtype {
        DType::F32 => narrowed!(f32),
        DType::I8 => narrowed!(i8),
        DType::U8 => narrowed!(u8),
        DType::I16 => narrowed!(i16),
        DType::I32 => narrowed!(i32),
    })
}
