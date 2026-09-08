//! A resident model: the ONNX session, its static facts, and one batched run.
//!
//! Layer 04 §3.5, §3.6 and §10. A **model** on this seam is the pair `(weights_hash,
//! adapter_hash)`; two pairs sharing a `weights_hash` share one session, which is why `/resident`
//! answers weights hashes rather than pairs.

use std::collections::BTreeMap;
use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor as OrtTensor, Value as OrtValue};

use crate::dialect::tensor::{DType, Tensor};
use crate::dialect::Ports;
use crate::onnx_meta::{self, GraphFacts};

pub struct Graph {
    pub session: std::sync::Mutex<Session>,
    pub facts: GraphFacts,
    pub bytes_resident: u64,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Port {
    pub name: String,
    pub dtype: String,
    /// A dynamic dimension is `null`, which is what a competitor declared rather than what the
    /// adapter will actually feed. `/validate` reports the shapes that were fed.
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
        let facts = onnx_meta::read(bytes);
        let build = || -> ort::Result<Session> {
            let mut b = Session::builder()?;
            b = b.with_optimization_level(GraphOptimizationLevel::Level3)?;
            b = b.with_intra_threads(threads.max(1))?;
            b.commit_from_memory(bytes)
        };
        let session = build().map_err(|e| {
            GraphError::Invalid(format!("the file parses but a session cannot be built: {e}"))
        })?;

        let port = |o: &ort::value::Outlet| Port {
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
        };
        let inputs = session.inputs().iter().map(port).collect();
        let outputs = session.outputs().iter().map(port).collect();

        Ok(Graph {
            bytes_resident: bytes.len() as u64,
            facts,
            inputs,
            outputs,
            session: std::sync::Mutex::new(session),
        })
    }

    /// Whether this graph can be run on several rows at once.
    ///
    /// Shape agreement between two rows is **not** sufficient, which a test found the direct way:
    /// two seats with the same observation produce identical feeds, so they group — and then the
    /// graph refuses them, because its board was exported with a leading dimension of exactly 1.
    ///
    /// The condition is that the graph *declares* a dynamic leading dimension on every input and
    /// every output. That is the competitor's authoring choice and it is visible statically, so it
    /// is decided once at load rather than discovered per call by a failed run.
    pub fn batchable(&self) -> bool {
        let dynamic_leading = |p: &Port| matches!(p.shape.first(), Some(None));
        !self.inputs.is_empty()
            && self.inputs.iter().all(dynamic_leading)
            && self.outputs.iter().all(dynamic_leading)
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|p| p.name.as_str()).collect()
    }

    /// Run the graph once. `feeds` is one named tensor per input; the answer is one named tensor
    /// per output, always as `float32` — the adapter reads them back with `tb.argmax` and friends,
    /// and a single element type on that boundary is one fewer thing for an adapter to get wrong.
    pub fn run(
        &self,
        feeds: &[(Arc<str>, Arc<Tensor>)],
        deadline: std::time::Instant,
    ) -> Result<Ports, String> {
        let mut session = self.session.lock().map_err(|_| "the session is poisoned".to_string())?;

        let want: Vec<String> = self.inputs.iter().map(|p| p.name.clone()).collect();
        let mut given: BTreeMap<&str, &Arc<Tensor>> = BTreeMap::new();
        for (k, v) in feeds {
            given.insert(k, v);
        }
        let mut values: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(want.len());
        for name in &want {
            let t = given.get(name.as_str()).ok_or_else(|| {
                format!("the adapter produced no tensor for the graph's input '{name}'")
            })?;
            values.push((std::borrow::Cow::Owned(name.clone()), to_ort(t)?.into()));
        }

        // The deadline. ORT is told to stop rather than being waited on, so a slow graph is cut
        // off and the other rows in the call are unaffected -- layer 04 §5.
        let opts = ort::session::RunOptions::new().map_err(|e| e.to_string())?;
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err("TIMED_OUT".into());
        }
        let flag = std::sync::Arc::new(opts);
        let watcher = flag.clone();
        let stop = std::thread::spawn(move || {
            std::thread::sleep(remaining);
            let _ = watcher.terminate();
        });

        let out = session.run_with_options(values, &flag);
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

/// The dialect holds every element as `f64` so an operator is written once rather than five times
/// (see `dialect/tensor.rs`). This is the one place that becomes the graph's element type.
fn to_ort(t: &Tensor) -> Result<OrtValue, String> {
    let shape: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
    let err = |e: ort::Error| e.to_string();
    Ok(match t.dtype {
        DType::F32 => {
            let d: Vec<f32> = t.data.iter().map(|&v| v as f32).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }
        DType::I8 => {
            let d: Vec<i8> = t.data.iter().map(|&v| v as i8).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }
        DType::U8 => {
            let d: Vec<u8> = t.data.iter().map(|&v| v as u8).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }
        DType::I16 => {
            let d: Vec<i16> = t.data.iter().map(|&v| v as i16).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }
        DType::I32 => {
            let d: Vec<i32> = t.data.iter().map(|&v| v as i32).collect();
            OrtTensor::from_array((shape, d)).map_err(err)?.into_dyn()
        }
    })
}
