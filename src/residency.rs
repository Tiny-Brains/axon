//! Residency and holds — docs/design.md §6.
//!
//! ```text
//! /load      hold += 1              refused if it cannot fit and nothing is evictable
//! /play      last_touched = now     for every model a row names
//! /unload    hold -= 1
//! expiry     hold  = 0              when now - last_touched > idle_ttl_s
//! eviction   LRU over hold == 0     only under the memory budget, never a model with holds
//! ```
//!
//! Two properties make this a hold table rather than a cache. Nothing held is ever evicted, whatever
//! the pressure — the budget is enforced at `/load`, by refusing, so a wave that started can finish.
//! And a crashed replica cannot pin memory forever: its holds lapse `idle_ttl_s` after its last
//! `/play`, which is why the TTL is set above the longest match rather than above a turn.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::dialect::Adapter;
use crate::model::Graph;

pub struct Resident {
    pub graph: Arc<Graph>,
    pub holds: u32,
    pub last_touched: Instant,
    pub idle_ttl: Duration,
}

#[derive(Default)]
pub struct Residency {
    weights: HashMap<String, Resident>,
    adapters: HashMap<String, Arc<Adapter>>,
    /// Refcounted with the weights they were loaded beside, so an adapter is dropped when no pair
    /// naming it is held.
    adapter_holds: HashMap<String, u32>,
    pub budget_bytes: u64,
}

impl Residency {
    pub fn new(budget_bytes: u64) -> Residency {
        Residency { budget_bytes, ..Default::default() }
    }

    pub fn used_bytes(&self) -> u64 {
        self.weights.values().map(|r| r.graph.bytes_resident).sum()
    }

    pub fn graph(&self, weights_hash: &str) -> Option<Arc<Graph>> {
        self.weights.get(weights_hash).map(|r| r.graph.clone())
    }

    pub fn adapter(&self, adapter_hash: &str) -> Option<Arc<Adapter>> {
        self.adapters.get(adapter_hash).cloned()
    }

    pub fn is_resident(&self, weights_hash: &str, adapter_hash: &str) -> bool {
        self.weights.contains_key(weights_hash) && self.adapters.contains_key(adapter_hash)
    }

    /// Advisory and allowed to be stale: what `/resident` answers, and what the claim orders its
    /// candidates by.
    pub fn resident_weights(&self) -> Vec<String> {
        let mut v: Vec<String> = self.weights.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn resident_adapters(&self) -> Vec<String> {
        let mut v: Vec<String> = self.adapters.keys().cloned().collect();
        v.sort();
        v
    }

    /// The keep-alive: every model a `/play` row named. There is no separate call.
    pub fn touch(&mut self, weights_hash: &str) {
        if let Some(r) = self.weights.get_mut(weights_hash) {
            r.last_touched = Instant::now();
        }
    }

    /// Whether `bytes` more would fit, after expiring what has lapsed and evicting what is
    /// evictable. Returns false rather than evicting anything held.
    pub fn make_room_for(&mut self, bytes: u64) -> bool {
        if bytes > self.budget_bytes {
            return false;
        }
        self.expire();
        if self.used_bytes() + bytes <= self.budget_bytes {
            return true;
        }
        // LRU over the unheld, oldest first, and only as far as it needs to go.
        let mut evictable: Vec<(String, Instant)> = self
            .weights
            .iter()
            .filter(|(_, r)| r.holds == 0)
            .map(|(h, r)| (h.clone(), r.last_touched))
            .collect();
        evictable.sort_by_key(|(_, t)| *t);
        for (h, _) in evictable {
            if self.used_bytes() + bytes <= self.budget_bytes {
                break;
            }
            self.weights.remove(&h);
        }
        self.used_bytes() + bytes <= self.budget_bytes
    }

    /// Drop the holds of a caller that never unloaded. The crash backstop.
    fn expire(&mut self) {
        let now = Instant::now();
        for r in self.weights.values_mut() {
            if r.holds > 0 && now.duration_since(r.last_touched) > r.idle_ttl {
                r.holds = 0;
            }
        }
    }

    pub fn insert(
        &mut self,
        weights_hash: &str,
        graph: Arc<Graph>,
        adapter_hash: &str,
        adapter: Arc<Adapter>,
        idle_ttl: Duration,
    ) {
        self.weights.insert(
            weights_hash.to_string(),
            Resident { graph, holds: 0, last_touched: Instant::now(), idle_ttl },
        );
        self.adapters.insert(adapter_hash.to_string(), adapter);
    }

    pub fn hold(&mut self, weights_hash: &str, adapter_hash: &str) {
        if let Some(r) = self.weights.get_mut(weights_hash) {
            r.holds += 1;
            r.last_touched = Instant::now();
        }
        *self.adapter_holds.entry(adapter_hash.to_string()).or_insert(0) += 1;
    }

    /// One hold removed. `false` is `not_held` rather than an error, so unloading is idempotent.
    pub fn release(&mut self, weights_hash: &str, adapter_hash: &str) -> bool {
        let had = match self.weights.get_mut(weights_hash) {
            Some(r) if r.holds > 0 => {
                r.holds -= 1;
                true
            }
            _ => false,
        };
        if let Some(n) = self.adapter_holds.get_mut(adapter_hash) {
            *n = n.saturating_sub(1);
            // An adapter is cheap to reparse and the memory budget counts only weights, so it
            // goes at zero holds rather than waiting for pressure.
            if *n == 0 {
                self.adapter_holds.remove(adapter_hash);
                self.adapters.remove(adapter_hash);
            }
        }
        had
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> Arc<Adapter> {
        Arc::new(Adapter::parse(br#"{"dialect":1,"in":{},"out":{}}"#).unwrap())
    }

    #[test]
    fn a_budget_that_cannot_fit_a_model_refuses_rather_than_evicting_everything() {
        let mut r = Residency::new(100);
        assert!(r.make_room_for(100));
        assert!(!r.make_room_for(101), "a model larger than the whole budget can never fit");
    }

    #[test]
    fn unloading_is_idempotent() {
        let mut r = Residency::new(1000);
        assert!(!r.release("w", "a"), "releasing what was never held is not_held, not an error");
        r.hold("w", "a");
        assert!(!r.release("w", "a"), "there is no such weights hash resident");
    }

    #[test]
    fn an_adapter_is_dropped_at_zero_holds() {
        let mut r = Residency::new(1000);
        r.adapters.insert("a".into(), adapter());
        r.hold("w", "a");
        r.hold("w", "a");
        r.release("w", "a");
        assert!(r.adapter("a").is_some(), "still held once");
        r.release("w", "a");
        assert!(r.adapter("a").is_none(), "dropped at zero");
    }
}
