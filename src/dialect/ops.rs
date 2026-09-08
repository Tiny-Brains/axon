//! The tensor operators — dialect 1. Twenty-one, plus four metadata helpers: 25 in all.
//! See `docs/dialect.md` §3; `TENSOR_OPS` below is the normative list.
//!
//! Every one costs `1 + max(elements read, elements produced)`: the node's own 1 is charged by the
//! evaluator, and each operator charges the rest here. The charge happens **before** the work, so
//! an operator that would produce a hundred million elements is refused rather than run and then
//! reported — see `docs/design.md` §13.
//!
//! **There is no arithmetic here beyond `cast` and `normalise`**, and that is the most important
//! property in the file. Computation belongs in the graph, where the FLOP cap prices it; an
//! adapter that could multiply matrices would be a second, unpriced model in front of the priced
//! one, and a Nano entry could carry a Small policy in its adapter. The line, when an operator is
//! requested: does it move or reshape information, or does it compute with it?

use std::sync::Arc;

use super::eval::{Ctx, Fault, Res};
use super::tensor::{saturate, DType, Tensor};
use super::value::Value;

/// The dialect's operator table. Order is fixed: `digest.rs` hashes it, and a reordering that
/// changed the digest without changing the meaning would fire a re-validation sweep for nothing.
pub const TENSOR_OPS: &[(&str, &str)] = &[
    ("tb.zeros", "(shape, dtype) -> T"),
    ("tb.full", "(shape, dtype, value) -> T"),
    ("tb.tensor", "(values, shape, dtype) -> T"),
    ("tb.scatter", "(points, shape, dtype, value?) -> T"),
    ("tb.rle_expand", "(runs, shape, dtype) -> T"),
    ("tb.one_hot", "(indices, depth, dtype) -> T"),
    ("tb.range", "(n) -> list"),
    ("tb.stack", "(tensors, axis, dtype?) -> T"),
    ("tb.concat", "(tensors, axis) -> T"),
    ("tb.unstack", "(T, axis) -> [T]"),
    ("tb.reshape", "(T, shape) -> T"),
    ("tb.transpose", "(T, perm) -> T"),
    ("tb.pad", "(T, before, after, value) -> T"),
    ("tb.crop", "(T, offset, shape) -> T"),
    ("tb.cast", "(T, dtype) -> T"),
    ("tb.normalise", "(T, mean, scale) -> T"),
    ("tb.argmax", "(T, axis) -> list"),
    ("tb.gather", "(T, indices, axis) -> T"),
    ("tb.dilate", "(T, radius2) -> T"),
    ("tb.to_list", "(T) -> nested list"),
    ("tb.shape", "(T) -> list"),
    ("tb.dtype", "(T) -> string"),
    ("tb.len", "(list) -> int"),
    ("tb.at", "(list, i) -> value"),
    ("tb.get", "(value, path) -> value"),
];

pub fn is_tensor_op(name: &str) -> bool {
    TENSOR_OPS.iter().any(|(n, _)| *n == name)
}

/// `1 + max(read, produced)`. The evaluator has already charged the node's 1.
fn charge(ctx: &mut Ctx, read: usize, produced: usize) -> Res<()> {
    ctx.charge(read.max(produced) as u64)
}

pub fn apply(op: &str, a: &[Value], ctx: &mut Ctx) -> Res<Value> {
    match op {
        // ------------------------------------------------------------ metadata: free but for the node
        "tb.shape" => {
            let t = tensor(a, 0, op)?;
            Ok(Value::arr(t.shape.iter().map(|&n| Value::Num(n as f64)).collect()))
        }
        "tb.dtype" => Ok(Value::str(tensor(a, 0, op)?.dtype.name())),
        "tb.len" => Ok(Value::Num(match a.first() {
            Some(Value::Arr(v)) => v.len() as f64,
            Some(Value::Str(s)) => s.chars().count() as f64,
            _ => 0.0,
        })),
        "tb.get" => {
            // Read a path out of an already-evaluated value.
            //
            // `var` reads the DOCUMENT. There was no way to read a field out of an expression's
            // result, and that gap blocked three separate real adapters before it was noticed:
            // any nested accumulation has to carry its loop-invariants in the accumulator
            // (because `reduce`'s seed is the only channel from outer scope into a body), and
            // then it cannot drop them again on the way out. Deriving a visibility mask, and
            // computing flat indices from a map width, both die exactly there.
            //
            // One operator, cost 1, no arithmetic. It moves information; it does not compute.
            let v = a.first().cloned().unwrap_or(Value::Null);
            let path = a.get(1).cloned().unwrap_or(Value::Null);
            Ok(super::eval::lookup(&v, &path).unwrap_or(Value::Null))
        }
        "tb.at" => {
            let list = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let i = num(a, 1, op)? as i64;
            let i = if i < 0 { list.len() as i64 + i } else { i };
            Ok(if i >= 0 && (i as usize) < list.len() {
                list[i as usize].clone()
            } else {
                Value::Null
            })
        }
        "tb.range" => {
            let n = num(a, 0, op)?.max(0.0) as usize;
            charge(ctx, 0, n)?;
            Ok(Value::arr((0..n).map(|i| Value::Num(i as f64)).collect()))
        }

        // ------------------------------------------------------------ making a tensor
        "tb.zeros" | "tb.full" => {
            let shape = shape(a, 0, op)?;
            let dt = dtype(a, 1, op)?;
            let v = if op == "tb.full" { num(a, 2, op)? } else { 0.0 };
            let n = elems(&shape)?;
            charge(ctx, 0, n)?;
            Ok(t(Tensor::filled(dt, shape, v)))
        }
        "tb.tensor" => {
            let values = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let shape = shape(a, 1, op)?;
            let dt = dtype(a, 2, op)?;
            let n = elems(&shape)?;
            charge(ctx, values.len(), n)?;
            if values.len() != n {
                return Err(Fault::invalid(format!(
                    "'{op}': {} values for a shape of {n}",
                    values.len()
                )));
            }
            let data = values
                .iter()
                .map(|v| v.to_num().map(|x| saturate(x, dt)).unwrap_or(0.0))
                .collect();
            Ok(t(Tensor::new(dt, shape, data)))
        }
        "tb.scatter" => {
            // The natural encoding of `mine`, `foes`, `food`: a list of index lists onto a plane.
            // `[r, c]` writes `value` (default 1); `[r, c, v]` writes v.
            let points = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let shape = shape(a, 1, op)?;
            let dt = dtype(a, 2, op)?;
            let dflt = a.get(3).and_then(Value::to_num).unwrap_or(1.0);
            let n = elems(&shape)?;
            charge(ctx, points.len(), n)?;
            let mut out = Tensor::filled(dt, shape.clone(), 0.0);
            let strides = out.strides();
            for p in points {
                let idx = p.as_arr().unwrap_or(&[]);
                let mut off = 0usize;
                let mut ok = true;
                for (d, s) in strides.iter().enumerate() {
                    let c = idx.get(d).and_then(Value::to_num).unwrap_or(-1.0);
                    if c < 0.0 || c as usize >= shape[d] {
                        // Out of bounds is dropped, not an error: an adapter clipping a wrapped
                        // coordinate is ordinary, and refusing the turn for it would be a strike
                        // for arithmetic the platform never specified.
                        ok = false;
                        break;
                    }
                    off += c as usize * s;
                }
                if ok {
                    let v = idx.get(strides.len()).and_then(Value::to_num).unwrap_or(dflt);
                    out.data[off] = saturate(v, dt);
                }
            }
            Ok(t(out))
        }
        "tb.rle_expand" => {
            // The natural encoding of `water`: [v0, n0, v1, n1, ...] row-major.
            let runs = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let shape = shape(a, 1, op)?;
            let dt = dtype(a, 2, op)?;
            let n = elems(&shape)?;
            charge(ctx, runs.len(), n)?;
            let mut data = Vec::with_capacity(n);
            for pair in runs.chunks(2) {
                let v = pair.first().and_then(Value::to_num).unwrap_or(0.0);
                let count = pair.get(1).and_then(Value::to_num).unwrap_or(0.0).max(0.0) as usize;
                if data.len() + count > n {
                    return Err(Fault::invalid(format!(
                        "'{op}': runs sum past the shape's {n} elements"
                    )));
                }
                data.resize(data.len() + count, saturate(v, dt));
            }
            data.resize(n, 0.0);
            Ok(t(Tensor::new(dt, shape, data)))
        }
        "tb.one_hot" => {
            let idx = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let depth = num(a, 1, op)?.max(0.0) as usize;
            let dt = dtype(a, 2, op)?;
            let n = idx.len().saturating_mul(depth);
            charge(ctx, idx.len(), n)?;
            let mut data = vec![0.0; n];
            for (i, v) in idx.iter().enumerate() {
                if let Some(k) = v.to_num() {
                    if k >= 0.0 && (k as usize) < depth {
                        data[i * depth + k as usize] = 1.0;
                    }
                }
            }
            Ok(t(Tensor::new(dt, vec![idx.len(), depth], data)))
        }

        // ------------------------------------------------------------ shaping
        "tb.stack" | "tb.concat" => {
            let list = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let mut ts = Vec::with_capacity(list.len());
            for v in list {
                ts.push(
                    v.as_tensor()
                        .ok_or_else(|| Fault::invalid(format!("'{op}' takes a list of tensors")))?
                        .clone(),
                );
            }
            if ts.is_empty() {
                return Err(Fault::invalid(format!("'{op}' got an empty list")));
            }
            let axis = num(a, 1, op)?.max(0.0) as usize;
            let read: usize = ts.iter().map(|x| x.len()).sum();
            charge(ctx, read, read)?;
            if op == "tb.stack" {
                let base = &ts[0];
                if ts.iter().any(|x| x.shape != base.shape) {
                    return Err(Fault::invalid("'tb.stack' needs tensors of one shape"));
                }
                if axis > base.rank() {
                    return Err(Fault::invalid("'tb.stack' axis is past the rank"));
                }
                let dt = match a.get(2) {
                    Some(v) if !matches!(v, Value::Null) => dtype(a, 2, op)?,
                    _ => base.dtype,
                };
                let mut shape = base.shape.clone();
                shape.insert(axis, ts.len());
                // Stacking on axis 0 is a concatenation of whole tensors; deeper axes interleave.
                let inner: usize = base.shape[axis..].iter().product();
                let outer: usize = base.shape[..axis].iter().product();
                let mut data = Vec::with_capacity(read);
                for o in 0..outer {
                    for x in &ts {
                        data.extend_from_slice(&x.data[o * inner..(o + 1) * inner]);
                    }
                }
                let data = data.into_iter().map(|v| saturate(v, dt)).collect();
                Ok(t(Tensor::new(dt, shape, data)))
            } else {
                let base = &ts[0];
                if axis >= base.rank() {
                    return Err(Fault::invalid("'tb.concat' axis is past the rank"));
                }
                let mut shape = base.shape.clone();
                shape[axis] = ts.iter().map(|x| x.shape[axis]).sum();
                let outer: usize = base.shape[..axis].iter().product();
                let mut data = Vec::with_capacity(read);
                for o in 0..outer {
                    for x in &ts {
                        let chunk: usize = x.shape[axis..].iter().product();
                        data.extend_from_slice(&x.data[o * chunk..(o + 1) * chunk]);
                    }
                }
                Ok(t(Tensor::new(base.dtype, shape, data)))
            }
        }
        "tb.unstack" => {
            let x = tensor(a, 0, op)?;
            let axis = num(a, 1, op)?.max(0.0) as usize;
            if axis >= x.rank() {
                return Err(Fault::invalid("'tb.unstack' axis is past the rank"));
            }
            charge(ctx, x.len(), x.len())?;
            let count = x.shape[axis];
            let inner: usize = x.shape[axis + 1..].iter().product();
            let outer: usize = x.shape[..axis].iter().product();
            let mut shape = x.shape.clone();
            shape.remove(axis);
            let mut out = Vec::with_capacity(count);
            for k in 0..count {
                let mut data = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    let base = (o * count + k) * inner;
                    data.extend_from_slice(&x.data[base..base + inner]);
                }
                out.push(t(Tensor::new(x.dtype, shape.clone(), data)));
            }
            Ok(Value::arr(out))
        }
        "tb.reshape" => {
            // A view. It reads nothing and produces nothing, so it costs only its node.
            let x = tensor(a, 0, op)?;
            let shape = shape(a, 1, op)?;
            if elems(&shape)? != x.len() {
                return Err(Fault::invalid(format!(
                    "'{op}': {} elements do not fit {shape:?}",
                    x.len()
                )));
            }
            Ok(t(Tensor::new(x.dtype, shape, x.data.clone())))
        }
        "tb.transpose" => {
            let x = tensor(a, 0, op)?;
            let perm: Vec<usize> = a
                .get(1)
                .and_then(Value::as_arr)
                .map(|p| p.iter().filter_map(|v| v.to_num()).map(|n| n as usize).collect())
                .unwrap_or_else(|| (0..x.rank()).rev().collect());
            if perm.len() != x.rank() || perm.iter().any(|&p| p >= x.rank()) {
                return Err(Fault::invalid("'tb.transpose' permutation does not match the rank"));
            }
            charge(ctx, x.len(), x.len())?;
            let src_strides = x.strides();
            let shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
            let mut data = Vec::with_capacity(x.len());
            let mut idx = vec![0usize; shape.len()];
            for _ in 0..x.len() {
                let off: usize =
                    idx.iter().enumerate().map(|(d, &i)| i * src_strides[perm[d]]).sum();
                data.push(x.data[off]);
                for d in (0..shape.len()).rev() {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
            Ok(t(Tensor::new(x.dtype, shape, data)))
        }
        "tb.pad" | "tb.crop" => {
            let x = tensor(a, 0, op)?;
            let p1 = usizes(a, 1, op)?;
            let p2 = usizes(a, 2, op)?;
            if p1.len() != x.rank() || p2.len() != x.rank() {
                return Err(Fault::invalid(format!("'{op}' needs one entry per dimension")));
            }
            let shape: Vec<usize> = if op == "tb.pad" {
                (0..x.rank()).map(|d| x.shape[d] + p1[d] + p2[d]).collect()
            } else {
                p2.clone()
            };
            let n = elems(&shape)?;
            charge(ctx, x.len(), n)?;
            let fill = if op == "tb.pad" { a.get(3).and_then(Value::to_num).unwrap_or(0.0) } else { 0.0 };
            let mut data = vec![saturate(fill, x.dtype); n];
            let src_strides = x.strides();
            let dst = Tensor::new(x.dtype, shape.clone(), vec![0.0; n]);
            let dst_strides = dst.strides();
            let mut idx = vec![0usize; x.rank()];
            'outer: for _ in 0..x.len() {
                // Where this source element lands, if it lands at all.
                let mut off = 0usize;
                let mut inside = true;
                for d in 0..x.rank() {
                    let target = if op == "tb.pad" {
                        idx[d] as i64 + p1[d] as i64
                    } else {
                        idx[d] as i64 - p1[d] as i64
                    };
                    if target < 0 || target as usize >= shape[d] {
                        inside = false;
                        break;
                    }
                    off += target as usize * dst_strides[d];
                }
                if inside {
                    let src: usize = idx.iter().enumerate().map(|(d, &i)| i * src_strides[d]).sum();
                    data[off] = x.data[src];
                }
                for d in (0..x.rank()).rev() {
                    idx[d] += 1;
                    if idx[d] < x.shape[d] {
                        continue 'outer;
                    }
                    idx[d] = 0;
                }
                break;
            }
            Ok(t(Tensor::new(x.dtype, shape, data)))
        }

        // ------------------------------------------------------------ values
        "tb.cast" => {
            let x = tensor(a, 0, op)?;
            let dt = dtype(a, 1, op)?;
            charge(ctx, x.len(), x.len())?;
            let data = x.data.iter().map(|&v| saturate(v, dt)).collect();
            Ok(t(Tensor::new(dt, x.shape.clone(), data)))
        }
        "tb.normalise" => {
            // `(x - mean) * scale`, producing float32. On the arithmetic line and allowed across
            // it because a fixed affine per tensor cannot encode a policy. If a second operator
            // ever needs this argument, the line has moved and should be redrawn deliberately.
            let x = tensor(a, 0, op)?;
            let mean = num(a, 1, op)?;
            let scale = a.get(2).and_then(Value::to_num).unwrap_or(1.0);
            charge(ctx, x.len(), x.len())?;
            let data = x.data.iter().map(|&v| (v - mean) * scale).collect();
            Ok(t(Tensor::new(DType::F32, x.shape.clone(), data)))
        }

        // ------------------------------------------------------------ back to JSON
        "tb.argmax" => {
            // The inverse of one_hot, and how a policy head becomes moves.
            let x = tensor(a, 0, op)?;
            let axis = num(a, 1, op)?.max(0.0) as usize;
            if axis >= x.rank() {
                return Err(Fault::invalid("'tb.argmax' axis is past the rank"));
            }
            let count = x.shape[axis];
            let produced = x.len().checked_div(count).unwrap_or(0);
            charge(ctx, x.len(), produced)?;
            let inner: usize = x.shape[axis + 1..].iter().product();
            let outer: usize = x.shape[..axis].iter().product();
            let mut out = Vec::with_capacity(produced);
            for o in 0..outer {
                for i in 0..inner {
                    let mut best = 0usize;
                    let mut bestv = f64::NEG_INFINITY;
                    for k in 0..count {
                        let v = x.data[(o * count + k) * inner + i];
                        // Strictly greater, so the first maximum wins and two machines agree.
                        if v > bestv {
                            bestv = v;
                            best = k;
                        }
                    }
                    out.push(Value::Num(best as f64));
                }
            }
            Ok(Value::arr(out))
        }
        "tb.gather" => {
            let x = tensor(a, 0, op)?;
            let idx = usizes(a, 1, op)?;
            let axis = a.get(2).and_then(Value::to_num).unwrap_or(0.0).max(0.0) as usize;
            if axis >= x.rank() {
                return Err(Fault::invalid("'tb.gather' axis is past the rank"));
            }
            let inner: usize = x.shape[axis + 1..].iter().product();
            let outer: usize = x.shape[..axis].iter().product();
            let produced = outer * idx.len() * inner;
            charge(ctx, x.len(), produced)?;
            let mut shape = x.shape.clone();
            shape[axis] = idx.len();
            let mut data = Vec::with_capacity(produced);
            for o in 0..outer {
                for &k in &idx {
                    if k >= x.shape[axis] {
                        return Err(Fault::invalid("'tb.gather' index is out of range"));
                    }
                    let base = (o * x.shape[axis] + k) * inner;
                    data.extend_from_slice(&x.data[base..base + inner]);
                }
            }
            Ok(t(Tensor::new(x.dtype, shape, data)))
        }
        "tb.dilate" => {
            // Mark every cell within euclidean radius^2 of a non-zero cell, on the last two
            // dimensions. Structure-preserving and fixed: it moves information outward by a
            // constant geometry and cannot encode a policy, so it is on the same side of §4.6's
            // line as `scatter` and `rle_expand`.
            //
            // It is here because the measurement put it here. `PROTOCOL.md` §8.3 names deriving
            // visibility as the concrete case the budget must accommodate, and without this the
            // only expressible form is a 317-pair kernel unrolled per ant -- 30x the cost, for a
            // program a competitor cannot write correctly on the first attempt (see
            // tests/ants_adapter.rs).
            let x = tensor(a, 0, op)?;
            let r2 = num(a, 1, op)?.max(0.0);
            if x.rank() < 2 {
                return Err(Fault::invalid("'tb.dilate' needs at least two dimensions"));
            }
            charge(ctx, x.len(), x.len())?;
            let (h, w) = (x.shape[x.rank() - 2], x.shape[x.rank() - 1]);
            let planes = x.len() / (h * w).max(1);
            let reach = (r2.sqrt().floor()) as i64;
            let offsets: Vec<(i64, i64)> = (-reach..=reach)
                .flat_map(|dr| (-reach..=reach).map(move |dc| (dr, dc)))
                .filter(|(dr, dc)| (dr * dr + dc * dc) as f64 <= r2)
                .collect();
            let mut data = vec![0.0; x.len()];
            for p in 0..planes {
                let base = p * h * w;
                for r in 0..h {
                    for c in 0..w {
                        if x.data[base + r * w + c] == 0.0 {
                            continue;
                        }
                        for (dr, dc) in &offsets {
                            // The map wraps, as every grid game this operator is for does. A
                            // cartridge whose map does not wrap crops afterwards.
                            let rr = (r as i64 + dr).rem_euclid(h as i64) as usize;
                            let cc = (c as i64 + dc).rem_euclid(w as i64) as usize;
                            data[base + rr * w + cc] = 1.0;
                        }
                    }
                }
            }
            Ok(t(Tensor::new(x.dtype, x.shape.clone(), data)))
        }
        "tb.to_list" => {
            // The general escape, and expensive by the count on purpose.
            let x = tensor(a, 0, op)?;
            charge(ctx, x.len(), x.len())?;
            Ok(nest(&x.data, &x.shape))
        }

        other => Err(Fault::invalid(format!("no operator '{other}' in this dialect"))),
    }
}

// ------------------------------------------------------------------ helpers

fn t(x: Tensor) -> Value {
    Value::Tensor(Arc::new(x))
}

fn nest(data: &[f64], shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::arr(data.iter().map(|&v| Value::Num(v)).collect());
    }
    let chunk: usize = shape[1..].iter().product();
    Value::arr(data.chunks(chunk.max(1)).map(|c| nest(c, &shape[1..])).collect())
}

fn tensor<'a>(a: &'a [Value], i: usize, op: &str) -> Res<&'a Tensor> {
    a.get(i)
        .and_then(Value::as_tensor)
        .map(|x| &**x)
        .ok_or_else(|| Fault::invalid(format!("'{op}' argument {i} must be a tensor")))
}

fn num(a: &[Value], i: usize, op: &str) -> Res<f64> {
    a.get(i)
        .and_then(Value::to_num)
        .ok_or_else(|| Fault::invalid(format!("'{op}' argument {i} must be a number")))
}

fn dtype(a: &[Value], i: usize, op: &str) -> Res<DType> {
    a.get(i)
        .and_then(Value::as_str)
        .and_then(DType::parse)
        .ok_or_else(|| Fault::invalid(format!("'{op}' argument {i} must be a dtype")))
}

fn shape(a: &[Value], i: usize, op: &str) -> Res<Vec<usize>> {
    let v = usizes(a, i, op)?;
    if v.is_empty() {
        return Err(Fault::invalid(format!("'{op}' argument {i} must be a shape")));
    }
    Ok(v)
}

fn usizes(a: &[Value], i: usize, op: &str) -> Res<Vec<usize>> {
    a.get(i)
        .and_then(Value::as_arr)
        .map(|s| s.iter().filter_map(Value::to_num).map(|n| n.max(0.0) as usize).collect())
        .ok_or_else(|| Fault::invalid(format!("'{op}' argument {i} must be a list of numbers")))
}

/// The element count of a shape, refusing one that would overflow before anything is allocated.
fn elems(shape: &[usize]) -> Res<usize> {
    let mut n = 1usize;
    for &d in shape {
        n = n
            .checked_mul(d)
            .ok_or_else(|| Fault::invalid("shape is larger than can be counted"))?;
    }
    Ok(n)
}
