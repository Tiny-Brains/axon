# Design — the loader

Axon is a Rust service, not an Orion package. It holds models resident, applies each seat's adapter
under an operation budget, runs the ONNX graph, holds every seat to the turn clock, and answers one
`/play` call per turn for a whole wave. **One instance runs beside every Kalam replica; one runs
beside Soma and serves admission** — the same binary, two roles by configuration.

This page is the seam: the six calls, the turn clock, residency and refusal, fetch-by-hash, the
admission instance, and the errors. The adapter dialect is [`dialect.md`](dialect.md).

Who calls it and when is
[devops/docs/architecture.md](https://github.com/Tiny-Brains/devops/blob/main/docs/architecture.md);
the wave that drives `/play` is
[kalam/docs/design.md](https://github.com/Tiny-Brains/kalam/blob/main/docs/design.md); the walk that
drives `/inspect` and `/validate` is
[jodi/docs/admission.md](https://github.com/Tiny-Brains/jodi/blob/main/docs/admission.md).

## 1. What this page fixes

| Settled here | Left to |
|---|---|
| the six calls: paths, request and reply shapes, and which of them the two instances serve | 07 how many instances run and where |
| the adapter dialect: the two programs, the JSONLogic subset, the tensor operators, and what a tensor *is* to a program | a published operator reference in `docs/` |
| the operation count and its unit, exactly enough to reimplement | 06 the number per game — `budgets.adapter_ops_max`, decision 6, from the counter spike |
| the turn clock: who owns it, what it covers, what a row that misses it answers | 03 the strike and the forfeit that follow |
| residency, holds, and the two-class refusal split that the schema's release and fail statements need | 03 the barrier's task list; 07 the memory budget per node |
| the fetch by hash, the object-store layout, and which instance may reach the internet | 07 the credentials and the bucket |
| the admission instance: how it differs, and the sequence admission drives it through | 08 the workflow, its timeout, and the verdict statement |
| the dialect version, the evaluator digest, and what each one is for | 08 the re-validation sweep that a dialect change fires |
| the errors, in one table, with the fault attributed to the model or to the loader | 03 what Kalam does with each |
| every number, in one configuration block, provisional and labelled | 07 the deployed values |

Two things this page deliberately does **not** fix. The engine's side of a turn is the cartridge's, and
the wave loop that calls all of this is Kalam's; this document says what the loader answers, not
when it is asked.

---

## 2. What it revises in the original design

An earlier document, *the model runner*, specified this service before three decisions were taken.
It has been retired; the parts of it that were never revised — the batcher, the runtime and the
telemetry — are §10.1 to §10.3 below. What its other sections got wrong, and what replaced them:

| The model-runner document said | Now | Why |
|---|---|---|
| three calls — `/load`, `/play`, `/unload` | **six** — those three, plus `/resident`, `/inspect` and `/validate` | `/inspect` and `/validate` were §7's "not in this scope"; finding 12.6 made the loader admission's only data path. `/resident` is decision 34 |
| §2.1 `/load` takes `weights_url` and `adapter_url`, and §9 decision 2 recommends "the caller passes URLs" | **`/load` takes hashes.** URLs are accepted **only** by the admission instance, and refused by a replica's | finding 12.3: the bytes live in the platform's own object store under a key that *is* the hash. Resolving a key from a hash is content addressing, not platform knowledge — the thing §9 decision 2 was protecting against. A replica that cannot be told a URL cannot be made to fetch arbitrary bytes |
| §3.2 fetches from a GitHub allowlist | **the admission instance fetches from GitHub; replicas fetch from the object store, by hash** | finding 12.3 (b). GitHub is touched once per submission |
| §3.3 the adapter is "JSONLogic in both directions … under a static instruction budget" | **the dialect is §4 here**: a JSONLogic subset plus twenty-five tensor operators, tensors opaque to the program, counted at run time rather than bounded statically | finding 4 rejected the static bound (option B) as astronomically loose over data-sized `map`s. The count is a run-time count, checked on every call |
| §3.3 "the exact encoding is decision 3" — flat `{dtype, shape, data}` or nested arrays | **neither: a program never sees a tensor's elements.** Decision 3 is dissolved, not answered | [`dialect.md`](dialect.md) §2. It is what makes the count exact and keeps a 128² plane out of the interpreter's value representation |
| §2.2 the deadline is "the cartridge's `turn_ms`", enforced somewhere | **the loader owns `turn_ms`** across adapter-in, inference and adapter-out, and reports `timed_out` per row | finding 4, clock option 3 |
| §3.1 holds lapse on `idle_ttl_s`, and `/play` is the keep-alive | **kept, and demoted to the crash backstop.** A wave holds its models for its life and releases them at wave end | the hold is now explicit in Kalam's task list; the TTL exists only for the wave that dies |
| §5 `MODEL_RUNNER_TOKEN`, `fetch_allow_hosts` on every instance | §11 here; `fetch_allow_hosts` is **empty on a replica** | the property is worth having: only one instance in the system can reach the internet |
| §9 decisions 1, 3, 5 | 1 is decision 6 and still open; 3 is dissolved ([`dialect.md`](dialect.md) §2); 5 (`wait_ms`) is **kept**, and the trial no longer needs it — a trial is an ordinary row | finding 6: the trial is a match, not a special path |

The name changed too: that document called it the model runner, the architecture calls it the model
loader, and the binary is `axon`. All three mean this process. This page uses **Axon** where it means
the running instance and "the loader" where it means the role.

---

## 3. The seam: six calls

JSON in, JSON out, over HTTP on loopback. Every request carries `authorization: Bearer <token>`.
Kalam reaches its own instance through the `model-loader` connector at `max_retries: 0`, because a
retried `/play` would replay a turn; Soma's admission workflow reaches the admission instance the
same way.

| Method | Path | Does | Rate | Replica | Admission |
|---|---|---|---|---|---|
| `POST` | `/load` | make a set of models resident, one hold each | once per wave | hashes only | hashes **or** URLs |
| `POST` | `/play` | one turn for a whole wave: observations in, actions out | **once per turn — the only hot call** | yes | no |
| `POST` | `/unload` | drop one hold each | once per wave | yes | yes |
| `GET` | `/resident` | which weights hashes have a session built | once per claim | yes | — |
| `POST` | `/inspect` | static facts about a resident graph | once per submission | — | yes |
| `POST` | `/validate` | run the adapter against the graph on reference observations | once per submission | — | yes |
| `GET` | `/healthz` | ready to serve | on boot, and per orchestrator probe | yes | yes |

A **model** on this seam is the pair `(weights_hash, adapter_hash)`, as it was. Two pairs sharing a
`weights_hash` share one ONNX session; that is internal, and it is why `/resident` answers weights
hashes rather than pairs.

### 3.1 `POST /load`

```jsonc
// in
{ "models": [ { "weights_hash": "sha256:…", "adapter_hash": "sha256:…" } ],
  "idle_ttl_s": 900,        // the crash backstop; a hold lapses after this long untouched
  "wait_ms": 0 }            // 0: answer at once. >0: block up to this long for the set to settle

// out
{ "models": [ { "weights_hash": "sha256:…", "adapter_hash": "sha256:…", "state": "resident" },
              { "weights_hash": "sha256:…", "adapter_hash": "sha256:…",
                "state": "refused", "reason": "MEMORY", "fault": "loader" } ],
  "evaluator_digest": "sha256:…", "dialect_version": 1 }
```

- **The hash is the identity and now also the address.** The loader derives the object key from the
  hash (§7), fetches, hashes what it received, and refuses `HASH_MISMATCH` if it differs — which on
  a replica means the object store is corrupt, not that the competitor lied, since admission already
  verified these bytes. On the admission instance the same check is the real gate (§8).
- **Batched, because a wave loads its whole model set in one call.** `state` is `resident`,
  `loading` or `refused`. `loading` means come back; the fetch continues and a later wave finds it
  warm. Kalam's barrier treats `loading` as not yet holdable and releases the row, the same as
  `MEMORY`.
- **Each load adds one hold** on each model named. A hold is what keeps a model resident (§6).
- **Every refusal carries `fault`**, `"model"` or `"loader"`, which is the whole of what Kalam
  needs to choose between [soma/docs/schema.md](https://github.com/Tiny-Brains/soma/blob/main/docs/schema.md) §4.4's release statement and its fail statement. The reason word
  is for the competitor and for the log; the `fault` field is for the workflow. A reason added later
  therefore costs no workflow change — which is the point of having both.

### 3.2 `POST /play`

```jsonc
// in — one row per live seat in the wave, in the caller's order
{ "rows": [ { "weights_hash": "sha256:…", "adapter_hash": "sha256:…",
              "ref": { "…": "opaque; echoed back untouched" },
              "observation": { "size": [64,96], "mine": [[12,30],[13,30],[41,77]],
                               "foes": [[12,33,1]], "food": [[11,31]],
                               "hills": [[20,20,0]],
                               "water": { "rle": [0,812,1,6,0,4110] } } } ],
  "deadline_ms": 1000,      // the cartridge's turn_ms; the call answers by then regardless
  "budget_ops": 200000 }    // the game's budgets.adapter_ops_max, per row, per direction

// out — same length, same order
{ "rows": [ { "action": ["N", "-", "E"], "ops": 41283, "elapsed_ms": 12, "ref": { … } },
            { "error": "TIMED_OUT", "elapsed_ms": 1000, "ref": { … } } ],
  "evaluator_digest": "sha256:…", "dialect_version": 1 }
```

- **The observation and the action are opaque.** The observation is whatever the engine's `observe`
  emitted for that seat; the action is whatever the adapter's `out` program produced. The root may
  be any JSON value ([ants/docs/protocol.md](https://github.com/Tiny-Brains/ants/blob/main/docs/protocol.md) §3). The loader validates neither against a
  game schema, because it has none and must not: `len(action) == len(mine)` is the engine's rule to
  enforce, and an action that breaks it is rejected by the engine, not here.
- **`budget_ops` is a parameter, not a setting.** The number is the game's, published in its
  manifest; passing it per call is what keeps the loader game-agnostic. The same number goes to
  `/validate`, and admission is not a real gate unless they agree.
- **Rows are grouped by `weights_hash`**, and a group runs as one batched inference when — and
  only when — **the graph declares a dynamic leading dimension on every input and every output**.
  Shape agreement between rows is *not* sufficient, which the build found the direct way: two seats
  with the same observation produce identical feeds and group happily, and then the graph refuses
  them because it was exported with a leading dimension of exactly `1`. Batchability is the
  competitor's authoring choice, it is visible statically, and it is decided once at load rather
  than discovered per call by a failed run. A graph that does not declare it is its own group of
  one, which costs that competitor throughput and no one else anything.
- **What batching is worth, measured 8 September 2026** (64 seats, the same Micro-class trunk,
  one authored ragged and one authored dense):

  | board | one inference per seat | one for the wave | speedup | MFLOP per board |
  |---|---|---|---|---|
  | 32×32 | 10.7 ms | 5.2 ms | **2.05×** | 13.5 |
  | 64×64 | 23.5 ms | 14.4 ms | **1.63×** | 54.1 |
  | 128×128 | 67.3 ms | 60.7 ms | **1.11×** | 216.3 |

  **The benefit shrinks as the work per board grows**, because batching amortises per-call overhead
  and improves core utilisation — it does not avoid arithmetic. Every seat sees a different
  observation, so every seat costs its own forward pass either way. the platform design §7's "a popular
  model in a wave of 64 matches costs one call per turn, not 64" is exactly true about *calls* and
  should not be read as a claim about FLOPs; at Ants' full board with a Micro model, one board
  already saturates the cores and the wave's batching is worth 11%. It is worth having and it is
  not what makes a wave the right shape — that is one engine call per turn, one round trip, and
  residency held for the wave's life.
- **Errors are per row, never per call.** A call-level status is reserved for a malformed body
  (`400`), a bad token (`401`), a body over the ceiling (`413`) and an instance not ready (`503`).
- **`ref` is echoed verbatim and never interpreted.** Optional, opaque, any JSON value, returned
  on the row it arrived on — success or error. *Added 7 September 2026 by the wave-turn spike*
  (the wave-turn spike), which found that without it **Kalam cannot
  count strikes at all**: accumulating a number per seat across turns means joining "this row timed
  out", which is positional in this reply, to "this is the seat it belongs to", which is positional
  in the engine's views — and an Orion workflow has no way to join two arrays. With the ref echoed,
  the error and the seat's identity are in the same object and one `map` produces both the actions
  and the updated counters. It costs the loader a field it copies; it is the difference between
  the platform design §7's five-strikes rule being implementable and not.
- **`ops` is reported per row** even on success. It is what makes a budget number tunable from
  production rather than from an argument, and Kalam logs it.

Size, for the record: an Ants view is about 2 KB, so a wave of eight two-player matches is 16 rows
and about 32 KB per call. Tensors never appear on the wire — that is the property the whole
placement exists for.

> **A gap the wave found, 8 September 2026.** A `/play` row reply carries `error` but **no
> `fault`**, while `/load`'s model state carries both (§6). Layer Kalam [`dialect.md`](dialect.md) §7 branches on `fault` to
> attribute a mid-play failure to the model and fail that row at once — and it has no field to
> branch on, so Kalam treats every per-row error as a missed clock and a strike ([`dialect.md`](dialect.md) §5's rule, which
> covers the case). Either add `fault` to the row reply and let [`dialect.md`](dialect.md) §7 work as written, or delete
> [`dialect.md`](dialect.md) §7 and say that strikes are the only mid-play mechanism. The second is closer to what is built
> and to what a competitor can reason about; the first is what the layer says.

### 3.3 `POST /unload`

```jsonc
// in
{ "models": [ { "weights_hash": "sha256:…", "adapter_hash": "sha256:…" } ] }
// out
{ "models": [ { "weights_hash": "sha256:…", "adapter_hash": "sha256:…", "state": "released" } ] }
```

`released` or `not_held`. Unload means "I no longer need it", not "free it now": it removes one
hold, and a model at zero holds becomes *evictable* but is dropped only when the memory budget needs
the space. A wave that loads what the last wave released pays nothing. `not_held` is not an error —
a hold that lapsed under `idle_ttl_s` answers it and the caller proceeds — so unloading is
idempotent by construction.

### 3.4 `GET /resident` — decision 34

```jsonc
// out
{ "weights": ["sha256:…", "sha256:…"],     // a session is built and warm
  "loading": ["sha256:…"],                 // fetching or building; not yet useful for affinity
  "adapters": ["sha256:…"],                // compiled
  "memory_bytes": 3_221_225_472, "memory_budget_bytes": 6_442_450_944 }
```

The sixth call, and the one the architecture's five do not name. [soma/docs/schema.md](https://github.com/Tiny-Brains/soma/blob/main/docs/schema.md) §4.2's claim takes
`$2 — resident weights hashes, text[]` and orders candidate rows by whether this replica already
holds their models; a workflow run has nowhere else to learn that list. The alternative — Kalam
remembering what it asked for — duplicates state that drifts the moment the loader evicts, and
drifts silently, since nothing would ever correct it.

**It is advisory and is allowed to be stale.** A hash listed here may be evicted before the claim
statement runs. Nothing breaks: affinity is an optimisation of the *fill*, and the residency barrier
(§6) is where correctness lives. `loading` is reported separately and deliberately excluded from
what Kalam passes to the claim, so a wave is not filled with rows whose models are still cold.

`GET` with no body, so it is a cheap first task of every claim occurrence and costs nothing when the
loader is warm.

### 3.5 `POST /inspect` — admission only

Static facts about a graph already made resident by `/load`. It needs no adapter and runs no
inference.

```jsonc
// in
{ "weights_hash": "sha256:…", "adapter_hash": "sha256:…" }

// out
{ "params": 41728,
  "opset": 17,
  "ops": ["Conv", "Relu", "Add", "MaxPool", "Reshape", "Gemm"],
  "unsupported_ops": [],
  "size_metric_bytes": 35_812,          // S = zstd-19(initializers) + zstd-19(adapter)
  "weights_zstd_bytes": 34_101, "adapter_zstd_bytes": 1_711,
  "weights_raw_bytes": 167_912, "adapter_raw_bytes": 8_204,
  "inputs":  [ { "name": "board",  "dtype": "int8",    "shape": [1, 6, "H", "W"] } ],
  "outputs": [ { "name": "policy", "dtype": "float32", "shape": ["N", 5] } ],
  "adapter": "{\"dialect\":1,\"in\":{…}}",   // the exact bytes, as text
  "evaluator_digest": "sha256:…", "dialect_version": 1 }
```

- **`adapter` is the exact document, byte for byte** — added 8 September 2026 at admission §13's
  asking. `models.adapter` stores the release asset's text under a `CHECK` that recomputes sha256
  over it, and this process is the only one that holds those bytes; a re-serialisation hashes
  differently and the constraint refuses the row. It is the good failure mode and not one to
  discover in production, so the reply carries text and never a parsed document.

- **`size_metric_bytes` is the platform design §5's `S`**, both terms, computed here because this is the
  only process that holds both artifacts. The class table is *not* applied here: the loader reports
  the number and admission classifies, so a threshold change is a platform decision and not a
  redeploy of this binary.
- **`ops` and `unsupported_ops` are reported, not judged**, against an allowlist the caller passes
  or, absent one, the loader's configured default. Same principle: facts here, policy there.
- **There is no FLOP number in this reply**, and its absence is deliberate — see §3.6.

### 3.6 `POST /validate` — admission only

```jsonc
// in
{ "weights_hash": "sha256:…", "adapter_hash": "sha256:…",
  "budget_ops": 200000,
  "observations": [ { … }, { … } ],     // the game's reference set, worst case included
  "deadline_ms": 5000 }

// out
{ "ok": true,
  "over_budget": null,                  // when ok is false: too expensive, or wrong
  "cases": [ { "ops_in": 137_402, "ops_out": 912, "elapsed_ms": 47,
               "inputs": [ { "name": "board", "dtype": "int8", "shape": [1,6,128,128] } ],
               "flops": 3.1e8,
               "action_shape": "array[180] of string" } ],
  "ops_max": 138_314,
  "flops_max": 3.1e8,
  "evaluator_digest": "sha256:…", "dialect_version": 1 }
```

For each observation: run `in`, check the tensors it produced against the graph's declared inputs by
name, dtype and shape, run the graph, run `out` on its outputs, and count both directions. `ok` is
false with a `reason` and the failing case index on any of: the program does not compile, a
direction exceeds `budget_ops`, a required input is missing or mis-shaped, the graph fails, or `out`
errors.

**`over_budget` separates "too expensive" from "wrong"** — admission §13's third ask, added
8 September 2026. A budget overrun answers `ADAPTER_FAILED` with `over_budget: true` and a
malformed program answers `ADAPTER_INVALID` with `false`; admission turns the first into
`ADAPTER_OVER_BUDGET`. Collapsing them would tell a competitor whose adapter merely costs too much
to go and re-read the dialect specification.

**The FLOP number comes from here, not from `/inspect`, and that is the point.** FLOPs are measured
on the graph as actually run, at **the shapes the adapter produced from the reference observation** —
not at a shape the competitor declared. A declared input shape is a claim; what the adapter feeds is
a fact, and a graph whose real input is eight times the declared one would otherwise pass a cap it
does not respect. Admission applies `budgets.flop_caps` for the class to `flops_max`.

**A requirement this places on admission and on the game:** the reference set must contain a
worst-case observation — the largest map, the most units — or admission is theatre. The budget is
checked per call at play, so an adapter validated only against a small sample and then struck every
turn in a real match has been admitted by a gate that did not test it. `schema/ants/`'s three worked
examples are the seed of that set; whether they include a worst case is the cartridge's to answer.

### 3.7 `GET /healthz`

`200` once the configured store is reachable and the evaluator has compiled its own self-test;
`503` before. `kalam/docker-entrypoint.sh` already probes exactly this path and refuses to load the
package until it answers, so a replica with no loader never claims a match it cannot play.

---

## 4. The adapter dialect

The dialect is its own page: [`dialect.md`](dialect.md). It carries the two programs and the object
rule, why a tensor is opaque to an adapter program, the twenty-five operators, the run-time
operation count, `dialect_version` versus `evaluator_digest`, and what the dialect deliberately
cannot do.

---

## 5. The turn clock

**The loader owns `turn_ms`** — finding 4, clock option 3 — and enforces it across all three legs of
a row: `in`, inference, `out`. Kalam passes it as `deadline_ms` and does no timing of its own.

- **The deadline is per call, and the call always answers on time.** Every row that has not produced
  an action when it expires comes back `TIMED_OUT` with its `elapsed_ms`; the rest carry actions.
- **Inference is terminable.** A run is started with a termination flag the deadline sets, so a slow
  graph is cut off rather than being waited on.
- **The adapter needs no wall-clock guard of its own** — the operation budget is its bound, and it is
  the deterministic one. A runaway program hits `budget_ops` and stops; a program that is merely slow
  on a loaded machine is caught by the call deadline like anything else.
- **A wave of K rows shares the deadline**, so a replica must be sized such that K rows of adapter
  and inference fit well inside `turn_ms` with the machine's cores. This is a sizing rule, not a
  fairness rule: the platform design §7 is explicit that **fairness is the static cap, not the clock**, and
  the clock is a safety net. Per-row `elapsed_ms` is reported on every reply precisely so a
  systematically starved replica is visible in the logs before it is visible in the ladder.

What follows a timeout is Kalam's: the action becomes the no-op, the seat takes a strike, five
strikes forfeit, and a forfeited seat plays no-ops until the engine ends the match (decision 16).

---

## 6. Residency, holds and refusal

The table is the model-runner document's residency table's, unchanged in mechanism:

```
/load      hold += 1              refused if it cannot fit and nothing is evictable
/play      last_touched = now     for every model a row names
/unload    hold -= 1
expiry     hold  = 0              when now − last_touched > idle_ttl_s
eviction   LRU over hold == 0     only under the memory budget, never a model with holds
```

**No eviction happens underneath a live match.** A model with a hold is never evicted whatever the
pressure; the budget is enforced at `/load`, by refusing. **A crashed replica cannot pin memory
forever**: its holds lapse `idle_ttl_s` after its last `/play`. The wave now holds its models
for its whole life and releases them at wave end, so the TTL is the crash backstop only, and it is
set above the longest match rather than above a turn.

**The refusal split — finding 7c, and the two statements it feeds.** [soma/docs/schema.md](https://github.com/Tiny-Brains/soma/blob/main/docs/schema.md) §4.4 has exactly two
outcomes for a row the barrier cannot start, and the `fault` field chooses between them:

| `fault` | Reasons | Kalam does | Layer 01 statement |
|---|---|---|---|
| `loader` | `MEMORY`, `FETCH_FAILED`, `STORE_UNAVAILABLE`, and `loading` | return the row to `pending`, raise `refusals`, clear the token, **spend no attempt**; `failed` with `UNLOADABLE` at the ceiling | [`dialect.md`](dialect.md) §4 release |
| `model` | `HASH_MISMATCH`, `GRAPH_INVALID`, `ADAPTER_INVALID`, `TOO_LARGE`, `URL_NOT_ACCEPTED` | `failed` at once, with the seat and the reason | [`dialect.md`](dialect.md) §4 fail |

The split is the design's fairness rule at the residency boundary: **a competitor is never charged
for the loader's problem, and the loader never absorbs the competitor's.** `FETCH_FAILED` sits on the
loader's side deliberately — an object-store hiccup is transient and a row that was paired against
verified bytes has done nothing wrong. `HASH_MISMATCH` on a replica means the store is corrupt rather
than the model is bad, and is nonetheless a `model` fault, because the row can never be played and
failing it with a named reason is better than looping it through `refusals` to the same end.

---

## 7. Fetch by hash, the mirror, and who may reach the internet

**The object store is the only place a replica reads bytes from**, and the key is the hash:

```
weights/sha256/<hex>            the ONNX file, exactly the bytes admitted
adapters/sha256/<hex>           the adapter document, exactly the bytes admitted
replays/<match_id>/<token>.json Kalam's, [soma/docs/schema.md](https://github.com/Tiny-Brains/soma/blob/main/docs/schema.md) §4.6 — not this process's
```

The loader holds a **read-only credential scoped to the two prefixes** and constructs the key from
the hash. Three consequences, each of which is why this is better than passing URLs:

1. **A replica cannot be told where to fetch from.** `/load` on a replica refuses a URL outright
   (`URL_NOT_ACCEPTED`), so a compromised or confused workflow cannot make it pull arbitrary bytes.
2. **`fetch_allow_hosts` is empty on a replica**, so the only instance in the system that can reach
   the public internet is the one beside Soma. That is a property worth having and it costs nothing.
3. **Kalam's `kalam-blobs` connector stays at `presign_get: false`** (the open work), a smaller
   grant than it would need to sign reads for the loader.

**The mirror** is the admission instance's, and it happens exactly once per submission (§8). A
competitor deleting their GitHub release afterwards breaks nothing — not their replays, not their
matches, not their rating — which is finding 12.3's real argument.

---

## 8. The admission instance

The same binary with `mode = "admission"`. It differs in four ways and no others:

1. **`/load` accepts URLs.** `{ weights_hash, adapter_hash, weights_url, adapter_url }`: fetch from
   the allowlisted hosts, hash what arrived, refuse `HASH_MISMATCH` if either differs from the hash
   the competitor declared. This is the real gate — on a replica the same check only catches a
   corrupt store.
2. **A successful `/load` mirrors**, writing both artifacts to their hash keys before answering
   `resident`. Idempotent: a key that exists with the right bytes is left alone. Mirroring inside
   `load` rather than as a call of its own means there is no state in which a model is admitted and
   its bytes are not in the store.
3. **`/inspect` and `/validate` are served**; on a replica they answer `404`.
4. **`/play` is refused.** Admission does not play; a trial is an ordinary match row that Kalam
   claims first (finding 6, decision 7), and this instance is not on that path.

The sequence admission drives, all four calls on the same resident pair:

```
POST /load       { hashes, urls }        → resident, or refused with a reason for the competitor
POST /inspect    { hashes }              → params, opset, ops, S, declared IO
POST /validate   { hashes, budget_ops, observations }
                                         → ok, ops_max, flops_max, actual input shapes
POST /unload     { hashes }              → released
```

Admission records the verdict, the class from `size_metric_bytes`, the FLOP check from `flops_max`,
and `evaluator_digest` on the `models` row, then moves the version `testing → verified`
(the schema §3.2). What this page owes admission is that **every refusal carries a reason word a
competitor can act on** — the whole vocabulary is §9 — and that the four calls are individually
retryable, since the workflow's timeout covers verification only (finding 6d).

**Adapter size.** [soma/docs/schema.md](https://github.com/Tiny-Brains/soma/blob/main/docs/schema.md) §3.2 leaves the cap here: **4 MiB raw**, refused as `TOO_LARGE` before the
document is parsed. It is a denial-of-service guard and not the fairness rule — the fairness rule is
`S`, which counts the adapter's compressed bytes against the class budget, so a Nano entry cannot
carry more than a few kilobytes of adapter whatever this cap says. It is also the number
`models.adapter text` is sized against.

---

## 9. Errors, in one table

| Word | Where | `fault` | Means |
|---|---|---|---|
| `MEMORY` | `/load` | loader | the budget is full and nothing is evictable |
| `FETCH_FAILED` | `/load` | loader | the object store or the release host did not answer — a `5xx`, a timeout, a connection failure |
| `ASSET_MISSING` | `/load` | model | a release URL answered `4xx`: the asset is not attached under its canonical name, or the repository is private |
| `STORE_UNAVAILABLE` | `/load` | loader | no credential, no bucket, or the store refused |
| `HASH_MISMATCH` | `/load` | model | the bytes do not hash to the declared hash |
| `GRAPH_INVALID` | `/load` | model | the ONNX file parses but a session cannot be built |
| `ADAPTER_INVALID` | `/load`, `/validate` | model | not the dialect, an unknown operator, an unimplemented `dialect` version, or over budget on the reference set |
| `TOO_LARGE` | `/load` | model | the adapter is over §8's raw cap, or the weights over the configured ceiling |
| `URL_NOT_ACCEPTED` | `/load` | model | a URL was passed to a replica instance |
| `NOT_RESIDENT` | `/play` row | — | the row names a model this instance does not hold |
| `TIMED_OUT` | `/play` row | — | the row did not answer within `deadline_ms` |
| `ADAPTER_FAILED` | `/play` row | — | the transform errored, or exceeded `budget_ops` (`over_budget: true`) |
| `INFERENCE_FAILED` | `/play` row | — | the runtime failed on this row |
| `SHAPE_MISMATCH` | `/validate` | model | the adapter produced a tensor the graph will not take |

Call-level status codes are `400` malformed, `401` bad token, `404` a call this instance does not
serve, `413` over the body ceiling, `503` not ready. Nothing else: a row's failure is a row's.

---

## 10. Batching, determinism, and what is not required to be reproducible

**Batching** is the model-runner document (§10.1), and it is designed rather than lucky: the schema
[`dialect.md`](dialect.md) §2's claim fills a wave with rows sharing the first row's models, so the group by `weights_hash`
is large by construction. That is finding 9–11's economics restored — *one inference per distinct
model per turn, per replica*.

**Three kinds of reproducibility, and only one is required.**

| | Required? | Why |
|---|---|---|
| the **operation count** across machines | **yes, exactly** | it is the competitor-facing budget; a count that differed by machine would mean a model admitted on one node and struck on another. Integer arithmetic over a deterministic walk; the spike proves it |
| the **evaluator's float arithmetic** | no | `normalise` produces `float32`, and nothing downstream audits it |
| the **graph's output** across machines or batch sizes | no | batching changes reduction order and the answer may differ in the last bits. The platform design settled this early: replays store the **action stream**, so the audit re-simulates the *game* against the cartridge digest, not the inference. Game state is integer-only; the model is not in that loop |

This is worth stating because the instinct on reading "determinism law" is to demand it everywhere,
and demanding it of the inference would cost the batching that pays for the whole competition.

---

### 10.1 Batcher

The three subsections here are carried unchanged from the model-runner document, which is otherwise
superseded by this page.

Within one `/play`, rows sharing a `weights_hash` become one inference with a leading batch
dimension, and the answers are scattered back to their rows. Coalescing across *concurrent* calls —
two waves in flight naming the same model — is deliberately out of scope; within-call batching
carries the economics.

> **Measured afterwards, and worth stating against the original claim.** Batching is worth **1.11x**
> at Ants' full board, not the order of magnitude the design argued for, because every seat costs
> its own forward pass either way. The decision survives the correction; the argument for it was
> overstated.

### 10.2 Runtime

ONNX Runtime, CPU execution provider, deterministic options, one session per resident weights hash,
a thread pool sized by config. The per-call deadline is enforced here. **Fairness is the static FLOP
cap checked at admission, not the clock**; the deadline is a safety net. A GPU pool for the `large`
class is a configuration value that is left empty today.

### 10.3 Telemetry

Load latency, residency hits and misses, evictions, per-model inference time, rows and batches per
`/play`, timeouts. Emitted as structured logs. Nothing reads them over HTTP.

---

## 11. Configuration — one block, every number provisional

One block, as the clocks §9 does it. Nothing that names a host, a budget or a secret is in code.

| Key | Meaning | Provisional |
|---|---|---|
| `mode` | `replica` or `admission` | — |
| `bind` | loopback address | `127.0.0.1:9090` |
| `auth_token` | the bearer every call must present | from the environment |
| `memory_budget_bytes` | resident weights and sessions; the refusal line at `/load` | sized to the node; **6 GiB** locally |
| `max_weights_bytes` | the raw ONNX ceiling, refused as `TOO_LARGE` | `96 MiB` — above the `large` class's 64 MiB compressed |
| `max_adapter_bytes` | the raw adapter ceiling (§8) | `4 MiB` |
| `default_idle_ttl_s` | when a `/load` names none; the crash backstop | `900` — above the longest match, not above a turn |
| `threads` | the ONNX Runtime intra-op pool | cores − 1 |
| `adapter_threads` | rows evaluated in parallel within one `/play` | cores |
| `max_in_flight` | concurrent `/play` calls | `1` on a replica, `4` on admission |
| `store_endpoint`, `store_bucket`, `store_prefix` | the object store and the two prefixes of §7 | from the environment |
| `store_credential` | read-only on a replica, read-write on admission | from the environment |
| `fetch_allow_hosts` | where a URL may point | **empty on a replica**; `github.com`, `objects.githubusercontent.com` on admission |
| `default_op_allowlist` | the ONNX operators `/inspect` reports against absent a caller's list | ~40 ops, the platform design §5 |
| `gpu_pool` | sessions for the `large` class | empty until M4 |

`budget_ops` and `deadline_ms` are **not** here: they are the game's, they arrive per call, and
putting them in this file would put a game's rules in a game-agnostic process.

---

## 12. Decisions taken here

Decisions **6** (the operation budget, measured at 1,000,000 rather than argued) and **34** (the
resident call), with the unnumbered calls this service forced — a tensor is opaque to a program, the
operator set has no arithmetic, the count is a run-time count, `dialect_version` and
`evaluator_digest` are different things, `/load` takes hashes while only admission takes URLs, the
mirror happens inside admission's `/load`, `fault: model | loader` on every refusal, FLOPs measured
at the shapes the adapter actually produced, and the 4 MiB adapter cap — are recorded with their
reasoning in
[devops/docs/decisions.md](https://github.com/Tiny-Brains/devops/blob/main/docs/decisions.md) §3,
under *The loader*.

---

## 13. What the build must prove, and what the build will find

> **BUILT, 8 September 2026 — and so is the rest of the loader.** `axon/` now answers this whole
> section against a real ONNX model: `/healthz`, `/load` (fetch by hash, verify, build, hold, and
> mirror on admission), `/play` (adapter → batched graph → adapter, per-row errors, the deadline),
> `/unload`, `/resident`, `/inspect` and `/validate`. 39 tests. A real Micro-class model plays a
> real worst-case Ants observation in **~1 ms a seat**, and the numbers K is bounded by are in §13.1.
>
> The counter is built and the five checks below are
> `axon/tests/dialect.rs` and `axon/tests/ants_adapter.rs`. What it found, beyond the number: the
> `tb.` prefix had to be reserved (a typo evaluated to an object literal and reached the graph as a
> malformed input); the dialect needed a twentieth operator; and the reference adapter's `to_json`
> rendered integers as floats, which would have changed the canonical form of every action and its
> audit hash — caught by the differential test against `datalogic-rs`, not by any test written for
> it.

The counter spike is this page's, and [`dialect.md`](dialect.md) §4 is what it tests:

1. **The count is identical on two machines** for the same adapter and the same observation —
   integer equality, not a tolerance. The one property §10 says is required.
2. **What the reference Ants adapter actually costs**, in `ops_in` and `ops_out`, on the worst-case
   observation. That number, plus headroom, is decision 6.
3. **A visibility-deriving adapter fits** — the game protocol §8.3 names it as the concrete case, since
   deriving the view mask from ant positions every turn is the expensive thing a real adapter wants
   to do.
4. **Over budget aborts mid-operator** and answers `ADAPTER_FAILED` rather than running to
   completion and reporting afterwards — otherwise the budget bounds the *report* and not the work.
5. **The evaluator's own deadline fires below the call deadline**, so a row that is merely slow is
   `TIMED_OUT` with the other rows unaffected.

What the wave-turn spike needs from this page, and can have before any of it is built: **a stub
loader answering §3.1, §3.2, §3.3 and §3.4 with random legal actions**. That is the whole point of
settling the API first — the stub is written against a shape that will not move, and the numbers it
measures are about Orion and the engine rather than about the loader. The stub was written first
and the real loader answers the same calls, so nothing built on it moved.

### 13.1 What the real loader costs, and what it means for K

Measured over HTTP against `axon`, a real 6,653-parameter conv model and the real worst-case Ants
observation, one `/play` for a whole wave:

| K | seats | one call | per seat |
|---|---|---|---|
| 8 | 16 | 16.5 ms | 1.03 ms |
| 16 | 32 | 32.8 ms | 1.03 ms |
| 32 | 64 | 65.6 ms | 1.02 ms |

Against `turn_ms = 1000`, and the engine-and-Orion cost the wave-turn spike measured:

| K | engine + Orion | loader | turn | of the budget |
|---|---|---|---|---|
| 8 | 23.9 ms | 16.5 ms | 40 ms | 4% |
| 16 | 46.1 ms | 32.8 ms | 79 ms | 8% |
| 32 | 96.1 ms | 65.6 ms | 162 ms | 16% |

**So K=32 spends a sixth of the turn budget on a Micro-class model**, and Kalam's provisional
K=16 is conservative by a factor of two or more. The caveat is the one the table cannot show: this
is a *Micro* model, and the classes go to `large` at 64 MiB. A model twenty times this size turns
16% into most of the budget, which is the argument for K being a deployment number rather than a
constant — deployment's, set against the class mix a replica actually serves.

---

## 14. Open questions

1. **Is twenty-one operators the right set?** It was nineteen until a measurement said otherwise and
   twenty until the build said otherwise, which is not a reassuring sample size. It is derived from
   one game. Tron needs `scatter` and
   `rle_expand` and little else; Planet Wars is all small dense vectors and may want nothing but
   `tb.tensor`. The honest answer is that the set will grow, that growth is a `dialect_version` bump
   and a re-validation sweep, and that the first outside cartridge (the build) is the real test.
2. **`tb.normalise` is on the arithmetic line.** [`dialect.md`](dialect.md) §6 admits it because a fixed affine cannot encode
   a policy. If a second operator ever needs the same argument, the line has moved and should be
   re-drawn deliberately rather than by precedent.
3. **Should `/validate` also run the graph a second time at a small shape** to catch a model whose
   FLOP cost is superlinear in the map? The cap is checked at one worst case, which is the right
   case, but it says nothing about behaviour between shapes. Cheap to add; unclear it buys anything
   while one preset per wave fixes the shape anyway.
4. ~~**The per-call deadline and K.**~~ **Answered, 8 September 2026, and the per-call deadline
   stays.** Measured: K=32 is 66 ms of loader against a `turn_ms` of 1000 — 16% of the turn with the
   engine and Orion included (§13.1). A per-row deadline with an admission-control queue would be
   more machinery and a harder fairness story to buy nothing. The caveat is the class mix: this is a
   Micro model and the classes go to 64 MiB, so re-ask it the first time a `large` entry plays.
5. **`ops` reported on every row costs a little bandwidth and buys tuning.** Keep it until the log
   volume argues otherwise; it is the only way `adapter_ops_max` is ever set from evidence.
6. **Whether the admission instance should hold anything resident at all** between calls. Four calls
   on one resident pair is simple, but it means admission's memory budget is sized for concurrent
   submissions rather than for one. `max_in_flight: 4` is a guess.
