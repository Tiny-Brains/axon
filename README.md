# Axon — the Model Loader

The one process in TinyBrains that is not Orion. It holds ONNX weights resident, evaluates each
competitor's adapter under an operation count of its own keeping, runs the graph, holds every seat
to the turn clock, and answers **one call per turn for a whole wave**. A second instance of the
same binary stands beside Soma and is the platform's only data path for admission.

The specification is [`design/v2/04-model-loader.md`](../design/v2/04-model-loader.md). Where this
repository and that document disagree, the document is right and this is a bug.

```
cargo test                                             # everything
cargo test --test ants_adapter -- --nocapture          # what a real adapter costs
cargo run                                              # the dialect version and its digest
```

## What is built

All six calls, against a real ONNX model. 51 tests.

**Three additions on 8 September 2026, at layer 08 §13's asking**, each one a thing admission
cannot do without: `/inspect` returns the adapter's exact bytes as text, because `models.adapter`
stores them under a `CHECK` that recomputes the hash and only this process holds them; a release URL
answering `4xx` is now `ASSET_MISSING` with `fault: model` rather than a retryable `FETCH_FAILED`,
so a competitor who forgot to attach `adapter.json` is told so instead of being retried three times
and timed out; and `/validate` sets `over_budget`, which is what lets admission say
`ADAPTER_OVER_BUDGET` rather than sending someone whose adapter is merely expensive back to the
dialect specification.

| | |
|---|---|
| `dialect/` | the adapter dialect: an opaque tensor, the evaluator, the operation count, the twenty-one operators, and `evaluator_digest` over the dialect rather than the binary |
| `onnx_meta.rs` | the static facts `/inspect` reports, read straight out of the ONNX protobuf — no schema crate |
| `model.rs` | the ONNX session, whether it batches, and one run under a deadline |
| `residency.rs` | holds, LRU, the memory budget, the crash backstop |
| `store.rs` | fetch by hash. **Three implementations**: a directory, an HTTP base, and **S3/R2 signed with SigV4** — layer 07 §8.1, and the only one a fleet can use, since the admission instance and every replica must share one store and across hosts there is no shared volume |
| `server.rs` | the six calls, blocking and threaded |

```
AXON_MODE=replica AXON_STORE_DIR=/var/lib/axon AXON_BIND=127.0.0.1:9090 axon

# Or on S3/R2, which is what a deployment uses. The bucket is checked FIRST, before
# AXON_STORE_DIR: a deployment that names one means it, and falling back to a directory
# because a variable was missing would give this replica its own empty store, silently.
AXON_MODE=replica AXON_BIND=127.0.0.1:9090 \
  AXON_STORE_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
  AXON_STORE_S3_BUCKET=tinybrains-models \
  AXON_STORE_S3_REGION=auto \
  AXON_STORE_S3_ACCESS_KEY=... AXON_STORE_S3_SECRET_KEY=... axon
cargo run --release --example dump-fixtures -- /tmp/store    # seed a store from the test fixtures
```

## What is not built yet

S3/R2 with SigV4 (layer 07), asynchronous loading — `/load` is synchronous, so `/resident` never
reports `loading` — and a GPU pool for the `large` class. The stub in
[`design/v2/03-spike/stub-loader/`](../design/v2/03-spike/stub-loader/) answers the same API and is
what Kalam's wave loop was built against; axon answers it for real.

## Three things worth knowing before reading the code

**It is not built on `datalogic-rs`, and that was not the plan.** Layer 04 said "wrap or fork" the
engine Orion evaluates workflow logic with. Wrapping does not work: 5.4 has no evaluation hook, no
step budget and no fuel, its `CustomOperator` trait reaches the `tb.*` operators but not the core
ones where an adapter actually spends, and its trace API materialises a JSON copy of the context
per node. Forking pins the platform's fairness rule to internals the crate says may change. And the
dialect is not JSONLogic anyway — it is a fixed subset plus twenty operators over a value JSON does
not have. The cost of owning it is paid in `tests/differential.rs`, which runs the core subset
through both engines and requires agreement.

**There is one deliberate divergence, and it is asserted in both directions.** `{"==": [0, null]}`
is `false` here and `true` in `datalogic-rs`. JavaScript and the JSONLogic specification say false.
The wave-turn spike found the quirk the hard way: a join written against a path that does not
resolve silently selects the falsy elements and looks correct for exactly as long as the value it
is compared against is zero. An adapter is the worst place to rediscover that.

**The dialect has no arithmetic on tensors**, and that is load-bearing rather than an omission.
Computation belongs in the graph, where the FLOP cap prices it. Three budgets on three axes: the
FLOP cap prices thinking, the operation count prices marshalling, and the compressed-size metric
`S` prices knowledge. An adapter that could multiply matrices would collapse the first into the
second.

## What the counter measured

Against a real worst-case Ants observation — 128×128, 90 ants, 670 water runs — dumped from the
spike engine (`tests/fixtures/ants-observation.json`):

| | ops |
|---|---|
| a reference six-plane adapter, `in` | 197,272 |
| the same, `out` | 1,265 |
| deriving visibility, unrolled kernel | 217,269 |
| deriving visibility, with `tb.dilate` | 32,777 |
| measured cost | 1.8 ms per million operations |

That is what raised `adapter_ops_max` from 200,000 to **1,000,000** (`PROTOCOL.md` §8.3, decision
6) and what added the twentieth operator. The full argument is in layer 04 §4.4.

## What the loader costs

One `/play` for a whole wave, over HTTP, real model and real observation: **~1.0 ms a seat**,
linear in the wave. At K=32 that is 66 ms against a `turn_ms` of 1000 — with the engine and Orion,
16% of the turn. Layer 03's provisional K=16 is conservative by a factor of two, for a Micro-class
model; the classes go to 64 MiB, which is why K belongs to a deployment rather than to a constant.

**Batching is worth less than it sounds, and the measurement is in layer 04 §3.2.** A group runs as
one inference only if the graph declares a dynamic leading dimension, and the benefit shrinks as
the work per board grows: 2.05× at a 32×32 board, 1.11× at Ants' full 128×128, where one board
already saturates the cores. Every seat sees a different observation, so every seat costs its own
forward pass either way — batching amortises overhead, it does not avoid arithmetic.
