//! `/play` — one call per turn for a whole wave.
//!
//! Three phases, and the middle one is the economics the design rests on:
//!
//!   1. every row's `in` program, producing its feeds
//!   2. rows grouped by weights hash, and within a group rows whose feeds agree in shape stacked
//!      into one inference
//!   3. every row's `out` program, over its own slice of the answer

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::Axon;
use crate::api::{PlayReply, PlayRequest, PlayRow, PlayRowReply};
use crate::dialect::{self, Fault, Ports, Tensor};
use crate::model::Graph;

struct Prepared {
    graph: Arc<Graph>,
    adapter: Arc<dialect::Adapter>,
    feeds: Ports,
    ops_in: u64,
    started: Instant,
}

impl Axon {
    pub fn play(&self, req: PlayRequest) -> PlayReply {
        let deadline = Instant::now() + Duration::from_millis(req.deadline_ms);
        // Every row owns an equal share of the call's deadline, and a group of k rows owns k
        // shares. Without this the deadline is one wall consumed in group order, so an expensive
        // model spends the budget and the groups behind it answer TIMED_OUT -- other competitors'
        // rows, struck for someone else's graph. With the FLOP cap gone the clock IS the fairness
        // control, and a shared clock is not one.
        let share = Duration::from_millis(req.deadline_ms) / req.rows.len().max(1) as u32;
        let mut rows: Vec<PlayRowReply> = Vec::with_capacity(req.rows.len());
        let mut prepared: Vec<Option<Prepared>> = Vec::with_capacity(req.rows.len());

        // ---- 1. the adapters in
        for row in &req.rows {
            let started = Instant::now();
            let held = {
                let mut res = self.residency.lock().unwrap();
                res.touch(&row.weights_hash);
                res.graph(&row.weights_hash).zip(res.adapter(&row.adapter_hash))
            };
            let Some((graph, adapter)) = held else {
                rows.push(finish(errored("NOT_RESIDENT"), row, started));
                prepared.push(None);
                continue;
            };
            if Instant::now() >= deadline {
                rows.push(finish(errored("TIMED_OUT"), row, started));
                prepared.push(None);
                continue;
            }

            let (feeds, ops_in) = adapter.run_in(&row.observation, req.budget_ops);
            match feeds {
                Ok(feeds) => {
                    rows.push(PlayRowReply::default());
                    prepared.push(Some(Prepared { graph, adapter, feeds, ops_in, started }));
                }
                Err(e) => {
                    rows.push(finish(fault_row(e, ops_in), row, started));
                    prepared.push(None);
                }
            }
        }

        // ---- 2. one inference per group whose shapes agree
        let mut outputs: Vec<Option<Result<Ports, String>>> =
            (0..prepared.len()).map(|_| None).collect();
        let mut infer_us: Vec<u64> = vec![0; prepared.len()];
        for idx in groups(&req.rows, &prepared) {
            let of = |i: usize| prepared[i].as_ref().expect("a group holds only prepared rows");
            let feeds: Vec<&Ports> = idx.iter().map(|&i| &of(i).feeds).collect();
            // Its own shares, never past the call's own deadline. A group that overruns times
            // itself out and the rows behind it still get theirs.
            let group_deadline = (Instant::now() + share * idx.len() as u32).min(deadline);
            let t_infer = Instant::now();
            let ran = run_group(&of(idx[0]).graph, &feeds, group_deadline);
            // The group ran once for k rows, so each is charged 1/k. This is the model's cost;
            // `elapsed_ms` is its latency and includes the wait behind everyone else.
            let per_row = t_infer.elapsed().as_micros() as u64 / idx.len() as u64;
            for &i in &idx {
                infer_us[i] = per_row;
            }
            match ran {
                Ok(per_row) => {
                    for (&i, a) in idx.iter().zip(per_row) {
                        outputs[i] = Some(Ok(a));
                    }
                }
                Err(e) => {
                    for &i in &idx {
                        outputs[i] = Some(Err(e.clone()));
                    }
                }
            }
        }

        // ---- 3. the adapters out
        for (i, p) in prepared.into_iter().enumerate() {
            let Some(p) = p else { continue };
            let row = &req.rows[i];
            let reply = match outputs[i].take() {
                Some(Ok(out)) => {
                    let (action, ops_out) =
                        p.adapter.run_out(out, &row.observation, req.budget_ops);
                    let ops = p.ops_in + ops_out;
                    match action {
                        Ok(a) => PlayRowReply { action: Some(a), ops, ..Default::default() },
                        Err(e) => fault_row(e, ops),
                    }
                }
                Some(Err(e)) if is_timeout(&e) => {
                    PlayRowReply { error: Some("TIMED_OUT"), ops: p.ops_in, ..Default::default() }
                }
                Some(Err(e)) => PlayRowReply {
                    error: Some("INFERENCE_FAILED"),
                    detail: Some(e),
                    ops: p.ops_in,
                    ..Default::default()
                },
                None => PlayRowReply {
                    error: Some("INFERENCE_FAILED"),
                    detail: Some("no group ran this row".into()),
                    ops: p.ops_in,
                    ..Default::default()
                },
            };
            rows[i] = finish(PlayRowReply { infer_us: infer_us[i], ..reply }, row, p.started);
        }

        PlayReply {
            rows,
            evaluator_digest: dialect::evaluator_digest(),
            dialect_version: dialect::DIALECT_VERSION,
        }
    }
}

fn errored(error: &'static str) -> PlayRowReply {
    PlayRowReply { error: Some(error), ..Default::default() }
}

fn finish(mut r: PlayRowReply, row: &PlayRow, started: Instant) -> PlayRowReply {
    r.elapsed_ms = started.elapsed().as_millis() as u64;
    r.r#ref = row.r#ref.clone();
    r
}

fn fault_row(e: Fault, ops: u64) -> PlayRowReply {
    PlayRowReply {
        error: Some("ADAPTER_FAILED"),
        over_budget: e.over_budget().then_some(true),
        detail: Some(e.to_string()),
        ops,
        ..Default::default()
    }
}

fn is_timeout(e: &str) -> bool {
    e == "TIMED_OUT" || e.contains("Terminate") || e.contains("terminated")
}

/// Rows that can share one inference, in the order they arrived.
///
/// Two rows batch together when they name the same weights and their feeds agree in name, dtype and
/// shape — and only when the graph declares a dynamic batch, because shape agreement between rows
/// says nothing about whether the graph will accept them stacked.
fn groups(rows: &[PlayRow], prepared: &[Option<Prepared>]) -> Vec<Vec<usize>> {
    let mut keyed: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, p) in prepared.iter().enumerate() {
        let Some(p) = p else { continue };
        let key = if p.graph.batchable() {
            shape_key(&rows[i].weights_hash, &p.feeds)
        } else {
            format!("{}#row{i}", rows[i].weights_hash)
        };
        match keyed.iter_mut().find(|(k, _)| *k == key) {
            Some((_, idx)) => idx.push(i),
            None => keyed.push((key, vec![i])),
        }
    }
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

fn shape_key(weights_hash: &str, feeds: &Ports) -> String {
    use std::fmt::Write;
    let mut k = String::from(weights_hash);
    for (name, t) in feeds {
        let _ = write!(k, "|{name}:{}", t.dtype.name());
        for d in &t.shape {
            let _ = write!(k, ",{d}");
        }
    }
    k
}

/// A group of one is an ordinary run; a group of several is stacked along the leading axis, run
/// once, and split back.
fn run_group(graph: &Graph, feeds: &[&Ports], deadline: Instant) -> Result<Vec<Ports>, String> {
    if feeds.len() == 1 {
        return graph.run(feeds[0], deadline).map(|o| vec![o]);
    }
    let n = feeds.len();
    let mut stacked: Ports = Vec::with_capacity(feeds[0].len());
    for (j, (name, first)) in feeds[0].iter().enumerate() {
        // A tensor already shaped [1, ...] -- what an adapter writes for a graph declaring a batch
        // dimension -- is concatenated along that axis rather than gaining a second one.
        let mut shape = first.shape.clone();
        if first.rank() > 0 && first.shape[0] == 1 {
            shape[0] = n;
        } else {
            shape.insert(0, n);
        }
        let mut data = Vec::with_capacity(first.len() * n);
        for f in feeds {
            data.extend_from_slice(&f[j].1.data);
        }
        stacked.push((name.clone(), Arc::new(Tensor::new(first.dtype, shape, data))));
    }

    let out = graph.run(&stacked, deadline)?;

    let mut per_row: Vec<Ports> = (0..n).map(|_| Vec::new()).collect();
    for (name, t) in out {
        if t.rank() == 0 || t.shape[0] != n {
            return Err(format!(
                "output '{name}' has a leading dimension of {:?}, not the batch of {n}; \
                 the graph does not batch",
                t.shape.first()
            ));
        }
        let per: usize = t.shape[1..].iter().product();
        let shape: Vec<usize> = t.shape[1..].to_vec();
        for (i, slot) in per_row.iter_mut().enumerate() {
            let data = t.data[i * per..(i + 1) * per].to_vec();
            slot.push((name.clone(), Arc::new(Tensor::new(t.dtype, shape.clone(), data))));
        }
    }
    Ok(per_row)
}
