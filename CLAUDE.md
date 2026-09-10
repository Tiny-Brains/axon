# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`axon` is one Rust binary that runs competitors' ONNX models for TinyBrains, in **two roles chosen
by `AXON_MODE`**: a `replica` beside each Kalam replica (holds models resident, answers `/play` once
per turn for a whole wave) and an `admission` instance beside Soma (accepts URLs, verifies, mirrors,
answers `/inspect` and `/validate`). No Postgres, no Orion, no game rules — observations and actions
are opaque JSON, and tensors never leave this process.

`docs/design.md` (the six calls, the turn clock, residency, the store, the error table) and
`docs/dialect.md` (the adapter dialect — normative) are the specification: **where the code and those
documents disagree, the code is the bug.** Read the relevant section before changing behaviour, and
update the doc, `README.md`'s Status block and `../design/tracker.md` when work lands.

## Commands

```sh
cargo test                                   # 52 tests; the 3 in tests/s3_live.rs no-op without AXON_S3_LIVE
cargo test --test dialect                    # one file
cargo test a_wave_loads_plays_and_unloads    # one test by name
cargo test --test ants_adapter -- --nocapture   # prints the op-count tables
cargo fmt                                    # rustfmt.toml: max_width 100, small heuristics Max
cargo run -- --dialect                       # dialect_version + evaluator_digest, no server

# a local replica over the committed fixtures
AXON_DEMO_STORE=$(mktemp -d)
cargo run --release --example dump-fixtures -- "$AXON_DEMO_STORE"
AXON_MODE=replica AXON_STORE_DIR="$AXON_DEMO_STORE" AXON_BIND=127.0.0.1:9090 cargo run --release

# the live S3 checks, against MinIO from devops (`docker compose up -d minio`)
AXON_S3_LIVE=1 cargo test --test s3_live -- --nocapture

python3 tests/fixtures/make-model.py         # regenerate the committed ONNX fixtures (needs torch, onnx)
```

First build downloads ONNX Runtime through `ort`'s `download-binaries`; the HTTP tests bind loopback.
The image builds on `rust:1-trixie` / `debian:trixie-slim` — bookworm's libstdc++ cannot link ORT's
prebuilt aarch64 binary, and both stages must move together.

## Architecture

**`src/server/http.rs` is the whole routing and role story**: `/load` and `/unload` on both roles,
`/play` replica-only, `/inspect` and `/validate` admission-only, everything else `404 NO_SUCH_CALL`.
`/healthz` is unauthenticated; every other call checks the bearer token when one is configured.
Blocking and threaded (`max_in_flight` acceptor threads), not async: ORT is blocking and the work is
CPU-bound.

**Hashes are the identity *and* the address.** `store::key` maps `sha256:<hex>` to
`weights/sha256/<hex>` or `adapters/sha256/<hex>`; a hash not in that form has no key, which is what
stops a hash from becoming a path. A replica refuses a URL outright (`URL_NOT_ACCEPTED`) and
`config::allow_hosts` returns empty for `Mode::Replica` **whatever the environment says** — one
instance in the system can reach the internet. Admission mirrors verified bytes to the store *inside*
`/load`, so no version is ever admitted with its bytes missing (`server/mod.rs::acquire`).

**`/play` is three phases** (`src/server/play.rs`): every row's `in` program → rows grouped and run
as one inference → every row's `out` program. Two rows batch only when they name the same weights,
their feeds agree in name/dtype/shape, **and `Graph::batchable()` is true** — the graph must *declare*
a dynamic leading dimension on every input and output. Shape agreement between rows is not
sufficient; a graph exported with a leading `1` refuses stacked feeds, so batchability is decided
statically at load, never by a failed run.

**Residency (`src/residency.rs`) is a hold table, not a cache.** A held model is never evicted
whatever the pressure — the budget is enforced at `/load` by refusing `MEMORY` — and `idle_ttl_s` is
only the crash backstop for a replica that died mid-wave. `/play` is the keep-alive; `/unload` is
idempotent (`not_held` is not an error).

**Every refusal carries `reason` *and* `fault`** (`model` or `loader`). Kalam branches on `fault`
alone: a `loader` fault releases the row with no attempt spent, a `model` fault fails the seat. Add
reasons freely; adding them costs no workflow change, which is the point of having both fields.
Per-row `/play` errors carry no `fault` — a known gap recorded in `docs/design.md` §3.2.

**The dialect (`src/dialect/`) is not built on `datalogic-rs`** (it exposes no fuel or step budget)
but its core JSONLogic subset must agree with it, because Orion evaluates workflow logic with that
crate. `tests/differential.rs` runs both engines on every case and pins the single deliberate
divergence, `{"==": [0, null]}`. A tensor is opaque to a program: operators bridge JSON→tensor and
tensor→JSON, so the count is exact arithmetic instead of instrumentation of a 16,384-element `map`.
Every node costs 1; every tensor operator costs `1 + max(read, produced)`, charged **before** the
work so the budget bounds the work rather than the report.

## What breaks if you forget it

- **`evaluator_digest` must move exactly when adapter meaning moves.** It is sha256 over
  `digest::canonical_description()` — `eval::CORE`, `ops::TENSOR_OPS` *in order*, `COUNTING_RULES`,
  `SEMANTICS` — and never over the binary. Reordering `TENSOR_OPS`, renaming a core operator, or
  changing a cost fires a re-validation sweep of the entire roster; a refactor that changes no
  semantics must leave the digest untouched (the 10 September 2026 reorganisation is the precedent).
  A semantic choice an adapter can observe but no operator name implies belongs in `SEMANTICS`.
- **No tensor arithmetic beyond `cast` and `normalise`.** Computation belongs in the ONNX graph where
  the FLOP cap prices it; an adapter that could multiply matrices is a second, unpriced model. The
  test for a proposed operator: does it move or reshape information, or compute with it?
- **`budget_ops` and `deadline_ms` are never configuration.** They are the game's, they arrive on
  every `/play` and `/validate` call, and putting them in `src/config.rs` would put a game's rules in
  a game-agnostic process. The op count is charged per direction and aborts mid-operator.
- **`/inspect` returns the adapter's exact bytes as text.** Soma's `models.adapter` has a `CHECK`
  recomputing sha256 over that text; a re-serialisation hashes differently and the row is refused.
- **The `tb.` prefix is reserved.** A single-key object whose key starts `tb.` and is not an operator
  is a refusal, not an object literal — otherwise a typo reaches the graph as a malformed input.
- **`tests/fixtures/*.onnx` are committed build output**, regenerated deterministically by
  `make-model.py`; the ants observation is a real worst case (128×128, turn 600, 90 ants) and the
  op-count assertions in `tests/ants_adapter.rs` are measurements, not guesses.
- **FLOPs are measured at the shapes the adapter actually produced**, never at a declared input
  shape — a declared shape is a claim, what the adapter feeds is a fact.
