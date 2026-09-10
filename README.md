# axon

Axon loads and runs competitors' ONNX models for TinyBrains. It is a Rust HTTP service with two
roles: an admission process that checks and mirrors submissions, and a replica process that
holds models in memory and returns actions for Kalam's match waves.

## The name

An **axon** carries signals away from a neuron's cell body. Here it connects the platform's game
observations to a model and brings back actions; [Soma](https://github.com/Tiny-Brains/soma) takes
its name from the cell body.

## Scope

**It owns**

- Model residency, reference-counted holds, memory limits, and idle eviction.
- The declarative adapter dialect, tensor conversion, and deterministic operation accounting.
- ONNX execution and per-seat deadline handling.
- Asset fetching, hash verification, graph inspection, adapter validation, and mirroring.
- Directory, HTTP, and SigV4 S3-compatible model-store implementations.

**It does not**

- Decide admission or promotion; [Jodi](https://github.com/Tiny-Brains/jodi) interprets its results.
- Understand game rules; [Ants](https://github.com/Tiny-Brains/ants) supplies observations and action semantics.
- Schedule matches or persist results; [Kalam](https://github.com/Tiny-Brains/kalam) owns the wave.
- Connect to Postgres or maintain the roster and ratings.

## Where it sits

```text
[Kalam] -- wave HTTP --> [Axon: replica]   -- read hashes --> [model store]
[Jodi]  -- admission --> [Axon: admission] -- mirror bytes -> [same store]
                                  |
                             fetch allowlisted assets
```

| Direction | Party | Over | What moves |
|---|---|---|---|
| called by | Kalam | Replica-local HTTP | Load holds, observations, actions, and unloads |
| called by | Jodi | Admission HTTP | Asset URLs, expected hashes, inspection and validation results |
| reads | Shared model store | Directory, HTTP, or SigV4 | Weights and adapters addressed by hash |
| writes | Shared model store | Directory or SigV4 | Verified admission assets |
| calls | Allowed release hosts | HTTP transport with host checks | Submission bytes, admission role only |

Admission and replicas must use the same store: a verified model is useful only if every replica
can retrieve its bytes. See the [system map](https://github.com/Tiny-Brains/devops#where-it-sits).

## Interface

Requests and replies are JSON; [src/api.rs](src/api.rs) contains their serialized type definitions.
Wrong-role calls return `404 NO_SUCH_CALL`. Set AXON_AUTH_TOKEN to require bearer authentication.

| Method and path | Role | Work performed | Main cost |
|---|---|---|---|
| POST /load | Both | Acquire model holds; admission also verifies and mirrors URL assets | Fetching, compilation, resident memory |
| POST /play | Replica | Evaluate adapters and models for one wave turn | Adapter operations and ONNX execution |
| POST /unload | Both | Release holds idempotently | Residency bookkeeping |
| GET /resident | Both | List resident hashes and memory usage | Residency lookup |
| POST /inspect | Admission | Read graph facts and the adapter's exact text | Metadata and size inspection |
| POST /validate | Admission | Exercise reference observations against the adapter and graph | One validation run per case |
| GET /healthz | Both | Report liveness | No model inference |

Pass the game's `budget_ops` and `deadline_ms` explicitly on play and validation calls.
A load refusal's `fault` distinguishes `model` from `loader`; callers should branch on that field
rather than maintain their own list of reason strings. Play results retain the caller's seat reference.

Dialect version 1 and `evaluator_digest` identify adapter semantics. The digest covers dialect
implementation files rather than the whole binary; print the current values without starting a server:

```sh
cargo run -- --dialect
```

## Run it, test it

Run commands from this repository's root. No Orion or database is required.

- Stable Rust and Cargo; Cargo.toml does not declare a minimum compiler version.
- ONNX Runtime is supplied through the ort dependency's download-binaries feature.
- Initial dependency installation needs network access; HTTP tests need permission to bind loopback.

Start a replica with a fixture store in a temporary directory:

```sh
AXON_DEMO_STORE=$(mktemp -d)
cargo run --release --example dump-fixtures -- "$AXON_DEMO_STORE"
AXON_MODE=replica AXON_STORE_DIR="$AXON_DEMO_STORE" AXON_BIND=127.0.0.1:9090 cargo run --release
```

In another terminal, check the unauthenticated local instance and run the tests:

```sh
curl --fail --silent --show-error http://127.0.0.1:9090/healthz
cargo test
```

The suite has 52 tests: 49 exercise local behavior and three live S3 checks return early unless
AXON_S3_LIVE is set. To exercise those checks, configure the S3 variables below and run
`AXON_S3_LIVE=1 cargo test --test s3_live -- --nocapture` against a disposable test bucket.
[tests/differential.rs](tests/differential.rs) compares the shared JSONLogic subset with datalogic-rs
and pins the intentional equality difference; it is the adapter/workflow compatibility check.

## What a deployment owes it

[src/config.rs](src/config.rs) defines environment parsing and defaults. S3 takes precedence over
a directory, which takes precedence over an HTTP store; incomplete S3 configuration fails startup.

| Variable | Purpose | Missing or incorrect value |
|---|---|---|
| AXON_MODE | replica or admission | Defaults to replica; other values fail startup |
| AXON_BIND | Listener address | Uses the loopback default |
| AXON_AUTH_TOKEN | Secret bearer credential | Empty or absent disables authentication |
| AXON_STORE_S3_BUCKET, AXON_STORE_S3_ENDPOINT | Shared S3 store location | Endpoint is required when a bucket is named |
| AXON_STORE_S3_ACCESS_KEY, AXON_STORE_S3_SECRET_KEY | Secret store credentials | Required for S3; bad credentials cause store failures |
| AXON_STORE_S3_REGION | Signing region | Defaults to auto; must match the store |
| AXON_STORE_DIR, AXON_STORE_URL | Alternative directory or HTTP store | No selected store fails startup; HTTP is read-only |
| AXON_FETCH_ALLOW_HOSTS | Admission asset host allowlist | Admission defaults to GitHub hosts; replica always uses an empty list |
| AXON_MEMORY_BUDGET_BYTES, AXON_IDLE_TTL_S | Residency capacity and eviction | Defaults apply; insufficient capacity refuses holds |
| AXON_MAX_WEIGHTS_BYTES, AXON_MAX_ADAPTER_BYTES | Asset size backstops | Defaults apply; oversized assets are refused |
| AXON_THREADS, AXON_MAX_IN_FLIGHT | Session threads and concurrent calls | Defaults apply; poor sizing constrains throughput |

Capacity settings are tuning policy. Game budgets belong to the cartridge and arrive in requests.
Replica URL rejection limits submitted asset fetching; it does not prevent access to its configured
remote store or replace deployment network isolation.

## Layout

```text
src/main.rs        service entry point and --dialect command
src/api.rs         request and response types
src/config.rs      environment configuration and store selection
src/residency.rs   holds, eviction, and memory accounting
src/server/        mod.rs   the service type, /load, /unload, /resident
                   play.rs  the turn: adapters in, grouped inference, adapters out
                   admission.rs  /inspect and /validate
                   http.rs  routing, role checks, and authentication
src/model/         mod.rs   ONNX sessions, deadlines, and batchability
                   meta.rs  static graph inspection and the size metric
src/store/         mod.rs   keys, the trait, directory and HTTP stores
                   s3.rs    S3-compatible store and the SigV4 chain
src/dialect/       evaluator, tensor operators, budget, and semantic digest
tests/             dialect, differential, ONNX, adapter, and live S3 checks
examples/          fixture-store generator
Dockerfile         service image build
```

## What must stay true

- **The evaluator counts adapter work itself.** Dialect tests check deterministic charging and immediate budget refusal.
- **Tensor arithmetic belongs in the model graph.** The dialect exposes conversion and arrangement operations without becoming an alternative inference engine.
- **Tensors remain inside Axon.** The HTTP contract carries game JSON and model hashes rather than tensor payloads.
- **Admission and replica roles remain distinct.** Routing and asset checks refuse the other role's work.
- **Infrastructure faults stay distinguishable from model faults.** Store and end-to-end tests prevent an unavailable service from becoming a competitor rejection.
- **Semantic changes change the evaluator digest.** Digest tests guard the identity used to detect admission/play skew.

## Status

**10 September 2026.** All six model calls and health routing are implemented, including SigV4
storage and admission mirroring. `cargo test` passes the 49 local tests; the three S3 tests require
an explicit live-store run and have not been exercised against a store. Loading is synchronous, the
resident loading list remains empty, and no GPU pool or cross-architecture counter conformance run
is provided.

The source was reorganised on 10 September 2026 — `server/`, `model/` and `store/` are directories
now, `onnx_meta` moved to `model::meta`, and `AXON_ADAPTER_THREADS` is gone because nothing read it.
The evaluator digest is unchanged, which is the property that mattered.

## More

- Local references: [wire types](src/api.rs), [configuration](src/config.rs), and [dialect tests](tests/dialect.rs).
- Design docs: [`docs/design.md`](docs/design.md) (the six calls, the turn clock, residency) and [`docs/dialect.md`](docs/dialect.md) (the adapter dialect — the normative reference).
- [The competitor guide](https://github.com/Tiny-Brains/docs) — the reader-facing half: the rules, the model format, the adapter dialect, submitting, ranking and seasons. The platform section is the high-level design for someone new to the codebase.
- Related repositories: [Jodi](https://github.com/Tiny-Brains/jodi), [Kalam](https://github.com/Tiny-Brains/kalam), [Ants](https://github.com/Tiny-Brains/ants), [DevOps](https://github.com/Tiny-Brains/devops).
- Apache-2.0: see [LICENSE](LICENSE).
