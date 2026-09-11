//! The static facts `/inspect` reports, read straight out of the ONNX protobuf: `ort` runs a graph,
//! it does not describe one.
//!
//! This walks the wire format rather than generating a schema — a length-delimited message can be
//! scanned for the handful of field numbers that matter, skipping every unknown field by its wire
//! type. The cost is the dependence on those field numbers, which have not changed since ONNX 1.0.
//!
//! ```text
//! ModelProto   .7  graph (GraphProto)      .8  opset_import (OperatorSetIdProto)
//! GraphProto   .1  node (NodeProto)        .5  initializer (TensorProto)
//! NodeProto    .4  op_type (string)
//! TensorProto  .1  dims (int64, packed)    .2  data_type (int32)   .9  raw_data (bytes)
//! OperatorSetIdProto  .1 domain (string)   .2  version (int64)
//! ```

use std::collections::BTreeSet;

#[derive(Debug, Default)]
pub struct GraphFacts {
    pub opset: i64,
    pub ops: BTreeSet<String>,
    pub params: u64,
    /// The concatenated initializer payloads: what the size metric compresses. Weights, not file.
    pub initializer_bytes: Vec<u8>,
    /// Every distinct element type the WEIGHTS are stored in, lowercased and sorted.
    ///
    /// The graph's input and output dtypes are already reported through `Port`, and they say
    /// nothing about this: a network whose ports are float32 may hold int8 weights, which is the
    /// whole point of quantisation. This is the fact a season needs to require one.
    ///
    /// Read off TensorProto's `data_type`, not inferred from which payload field carried the
    /// bytes — an exporter may put int8 data in `raw_data` or in `int32_data`, and the declared
    /// type is the one the runtime reads.
    pub initializer_dtypes: BTreeSet<String>,
}

/// ONNX `TensorProto.DataType`, as the names a competitor writes in a season rule.
///
/// Unknown numbers become `type-<n>` rather than being dropped: a dtype the platform cannot name
/// must still be visible to a rule that lists what is allowed, or a future ONNX type would pass a
/// quantised-only season by being unrecognised.
fn dtype_name(n: u64) -> String {
    match n {
        1 => "float32", 2 => "uint8", 3 => "int8", 4 => "uint16", 5 => "int16",
        6 => "int32", 7 => "int64", 8 => "string", 9 => "bool", 10 => "float16",
        11 => "float64", 12 => "uint32", 13 => "uint64", 14 => "complex64",
        15 => "complex128", 16 => "bfloat16", 17 => "float8e4m3fn", 18 => "float8e4m3fnuz",
        19 => "float8e5m2", 20 => "float8e5m2fnuz", 21 => "uint4", 22 => "int4",
        23 => "float4e2m1",
        other => return format!("type-{other}"),
    }
    .to_string()
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, at: 0 }
    }
    fn done(&self) -> bool {
        self.at >= self.b.len()
    }
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self.b.get(self.at)?;
            self.at += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
            if shift > 63 {
                return None;
            }
        }
    }
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    /// The next field's number, with its payload consumed. `None` at the end or on anything
    /// malformed: a truncated file reports what it read, because `/inspect`'s caller wants to know
    /// the file is broken, not to get a parse error.
    fn field(&mut self) -> Option<(u64, Field<'a>)> {
        let key = self.varint()?;
        let (num, wire) = (key >> 3, key & 7);
        Some((
            num,
            match wire {
                0 => Field::Varint(self.varint()?),
                1 => {
                    self.bytes(8)?;
                    Field::Fixed
                }
                2 => {
                    let n = self.varint()? as usize;
                    Field::Len(self.bytes(n)?)
                }
                5 => {
                    self.bytes(4)?;
                    Field::Fixed
                }
                _ => return None,
            },
        ))
    }
}

enum Field<'a> {
    Varint(u64),
    Len(&'a [u8]),
    /// Consumed and discarded: no field this scans is fixed-width.
    Fixed,
}

impl<'a> Field<'a> {
    fn len(self) -> Option<&'a [u8]> {
        match self {
            Field::Len(b) => Some(b),
            _ => None,
        }
    }
    fn varint(self) -> Option<u64> {
        match self {
            Field::Varint(v) => Some(v),
            _ => None,
        }
    }
}

pub fn read(model: &[u8]) -> GraphFacts {
    let mut f = GraphFacts::default();
    let mut r = Reader::new(model);
    while !r.done() {
        let Some((num, val)) = r.field() else { break };
        match num {
            7 => {
                if let Some(g) = val.len() {
                    read_graph(g, &mut f);
                }
            }
            // The default domain's version is the opset. A model carrying several imports reports
            // the largest, which is what a reader means by "the opset".
            8 => {
                if let Some(o) = val.len() {
                    let mut rr = Reader::new(o);
                    let mut domain_is_default = true;
                    let mut version = 0i64;
                    while !rr.done() {
                        let Some((n, v)) = rr.field() else { break };
                        match n {
                            1 => domain_is_default = v.len().is_some_and(|d| d.is_empty()),
                            2 => version = v.varint().unwrap_or(0) as i64,
                            _ => {}
                        }
                    }
                    if domain_is_default {
                        f.opset = f.opset.max(version);
                    }
                }
            }
            _ => {}
        }
    }
    f
}

fn read_graph(g: &[u8], f: &mut GraphFacts) {
    let mut r = Reader::new(g);
    while !r.done() {
        let Some((num, val)) = r.field() else { break };
        match num {
            1 => {
                if let Some(node) = val.len() {
                    let mut rr = Reader::new(node);
                    while !rr.done() {
                        let Some((n, v)) = rr.field() else { break };
                        if n == 4
                            && let Some(op) = v.len().and_then(|b| std::str::from_utf8(b).ok())
                        {
                            f.ops.insert(op.to_string());
                        }
                    }
                }
            }
            5 => {
                if let Some(t) = val.len() {
                    read_initializer(t, f);
                }
            }
            _ => {}
        }
    }
}

fn read_initializer(t: &[u8], f: &mut GraphFacts) {
    let mut r = Reader::new(t);
    let mut count = 1u64;
    let mut any_dim = false;
    while !r.done() {
        let Some((num, val)) = r.field() else { break };
        match num {
            1 => match val {
                // `dims` is int64, and protobuf may send it packed or one field at a time.
                Field::Varint(d) => {
                    count = count.saturating_mul(d);
                    any_dim = true;
                }
                Field::Len(b) => {
                    let mut rr = Reader::new(b);
                    while !rr.done() {
                        match rr.varint() {
                            Some(d) => {
                                count = count.saturating_mul(d);
                                any_dim = true;
                            }
                            None => break,
                        }
                    }
                }
                Field::Fixed => {}
            },
            // 2 is `data_type`: the element type this tensor is stored in.
            2 => {
                if let Field::Varint(d) = val {
                    f.initializer_dtypes.insert(dtype_name(d));
                }
            }
            9 => {
                if let Some(raw) = val.len() {
                    f.initializer_bytes.extend_from_slice(raw);
                }
            }
            _ => {}
        }
    }
    // A scalar initializer has no dims and one element.
    f.params = f.params.saturating_add(if any_dim { count } else { 1 });
}

/// The size metric, both terms: `S = len(zstd-19(initializer data)) + len(zstd-19(adapter))`.
pub fn size_metric(initializer_bytes: &[u8], adapter: &[u8]) -> (usize, usize, usize) {
    let w = zstd::encode_all(initializer_bytes, 19).map(|v| v.len()).unwrap_or(0);
    let a = zstd::encode_all(adapter, 19).map(|v| v.len()).unwrap_or(0);
    (w + a, w, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &[u8] = include_bytes!("../../tests/fixtures/ants-micro.onnx");

    #[test]
    fn it_reads_a_real_model() {
        let f = read(MODEL);
        assert_eq!(f.opset, 17, "the fixture was exported at opset 17");
        // The architecture make-model.py builds: a conv trunk with ReLU, then a gather.
        assert!(f.ops.contains("Conv"), "ops: {:?}", f.ops);
        assert!(f.ops.contains("Relu"), "ops: {:?}", f.ops);
        assert_eq!(f.params, 6653, "the exporter reported 6653 parameters");
        assert!(!f.initializer_bytes.is_empty());

        let (s, w, a) = size_metric(&f.initializer_bytes, b"{}");
        assert!(s > 0 && w > 0 && a > 0);
        // 6653 fp32 parameters is ~26 KB raw; random weights barely compress, so the metric
        // should land in the same order and inside the Micro class's 64 KiB.
        assert!((10_000..64 * 1024).contains(&w), "compressed initializers were {w} bytes");
        println!(
            "\nfixture: opset {} · {} params · S={s} (weights {w}, adapter {a})\n  ops: {:?}",
            f.opset, f.params, f.ops
        );
    }

    #[test]
    fn a_truncated_model_reports_what_it_read_rather_than_failing() {
        let f = read(&MODEL[..MODEL.len() / 2]);
        assert!(f.params < 6653);
        let _ = read(b"not a protobuf at all");
        let _ = read(&[]);
    }
}
