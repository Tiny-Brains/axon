# The adapter dialect

An adapter is a competitor's declarative transform, submitted with the model: it maps the game's
observation JSON into the model's input tensors, and the model's output tensors back into the
game's action JSON. **It is data, not code** — evaluated by this service under an operation budget
it counts itself, never as an Orion definition.

This page is the normative reference: the two programs, the operator set, how operations are
counted, and what the dialect deliberately cannot do. The competitor-facing introduction is
[the book](https://github.com/Tiny-Brains/docs).

The evaluator is `src/dialect/`; `dialect_version` and `evaluator_digest` are what a submission is
validated against and what a re-validation sweep fires on.

---


The competitor's `adapter.json`, in full:

```jsonc
{ "dialect": 1,
  "in":  <program: observation → { input_name: tensor }>,
  "out": <program: { output_name: tensor } → action> }
```

`dialect` is declared by the competitor and checked; a document declaring a version this evaluator
does not implement is `ADAPTER_INVALID` at admission and never reaches play (§5).

## 1. Two programs, and the object rule

Each program is one JSONLogic expression evaluated against a document. For `in`, the document is the
observation exactly as the engine emitted it. For `out`, it is `{ output_name: tensor }` — so
`{"var": "policy"}` is the graph's `policy` output.

JSONLogic has no object constructor, and both programs must produce one, so the dialect states the
rule that most implementations leave to inference: **an object whose single key is a known operator
is an operation; any other object is a literal whose values are evaluated and whose keys are not.**
That is one sentence, it is unambiguous, and it is what lets `in` name its tensors:

```jsonc
{ "board":  { "tb.stack": [ [ …six planes… ], 0, "int8" ] },
  "n_ants": { "tb.tensor": [ [ { "tb.len": { "var": "mine" } } ], [1], "int32" ] } }
```

## 2. A tensor is opaque to a program

the model-runner document's decision 3 asked whether a program sees a tensor as flat
`{dtype, shape, data}` or as nested arrays. **Neither. A program never sees a tensor's elements at
all.** A tensor is an opaque value produced by an operator and consumed by an operator; the only
things a program may ask of one are its shape and its dtype. The bridges in both directions are
explicit operators: `tb.scatter` and friends turn JSON into a tensor, `tb.argmax` and `tb.to_list`
turn a tensor into JSON.

Three things follow, and together they are the reason for the choice:

1. **The count is exact and cheap.** Every operator knows how many elements it read and produced, so
   the budget is arithmetic rather than instrumentation of a general interpreter walking a 131,072-
   element array.
2. **A 128² plane never enters the interpreter's value representation.** No `map` over 16,384 JSON
   numbers, no boxing, no allocation the count would have to price.
3. **The dialect's surface is a list**, which is what makes `dialect_version` mean something and the
   re-validation sweep (finding 5) tractable.

The cost is real and worth stating plainly: **a competitor can only compose the operators the
platform ships.** An adapter that wants a transform not in §3 cannot express it, and the answer is
to ask for the operator rather than to work around it. That is a narrower dialect than "JSONLogic
over tensors" sounded like, and it is a deliberate narrowing — see §6.

## 3. The operators

**Twenty-five in all: the twenty-one below that make or reshape tensors, plus the four metadata
helpers at the end of this section.** `TENSOR_OPS` in `src/dialect/ops.rs` is the list, and its
order is fixed because `digest.rs` hashes it — a reordering that changed `evaluator_digest` without
changing the dialect's meaning would fire a re-validation sweep for nothing.

Names are namespaced `tb.` so nothing collides with a JSONLogic core operator, and **the
prefix is reserved**: a single-key object whose key begins `tb.` and is not an operator is refused
rather than falling through to the literal rule, so `tb.scattr` is a refusal at the operator that
does not exist instead of an object handed to the graph as an input that is not a tensor. Shapes
are JSON integer lists; dtypes are `int8`, `uint8`, `int16`, `int32`, `float32`.

**Making a tensor**

| Operator | Signature | Notes |
|---|---|---|
| `tb.zeros` | `(shape, dtype) → T` | |
| `tb.full` | `(shape, dtype, value) → T` | |
| `tb.tensor` | `(values, shape, dtype) → T` | from a flat JSON list; the small-tensor escape hatch |
| `tb.scatter` | `(points, shape, dtype, value?) → T` | `points` is a list of index lists; `[r,c]` writes `value` (default 1), `[r,c,v]` writes `v`. The natural encoding of `mine`, `food`, `foes` |
| `tb.rle_expand` | `(runs, shape, dtype) → T` | `[v0,n0, v1,n1, …]` row-major; the natural encoding of `water` |
| `tb.one_hot` | `(indices, depth, dtype, axis?) → T` | |
| `tb.range` | `(n) → list` | JSON, not a tensor |

**Shaping**

| Operator | Signature | Notes |
|---|---|---|
| `tb.stack` | `(tensors, axis, dtype?) → T` | the plane stack |
| `tb.concat` | `(tensors, axis) → T` | |
| `tb.unstack` | `(T, axis) → [T]` | |
| `tb.reshape` | `(T, shape) → T` | metadata only; costs 1 |
| `tb.transpose` | `(T, perm) → T` | |
| `tb.pad` | `(T, before, after, value) → T` | |
| `tb.crop` | `(T, offset, shape) → T` | |

**Values**

| Operator | Signature | Notes |
|---|---|---|
| `tb.cast` | `(T, dtype) → T` | saturating |
| `tb.normalise` | `(T, mean, scale) → T` | `(x − mean) × scale`, producing `float32` |

**Reading a tensor back to JSON**

| Operator | Signature | Notes |
|---|---|---|
| `tb.argmax` | `(T, axis) → list` | the inverse of `one_hot`, and how a policy head becomes moves |
| `tb.gather` | `(T, indices, axis) → T` | the inverse of `scatter` |
| `tb.dilate` | `(T, radius2) → T` | mark every cell within a fixed radius of a non-zero cell, on the last two dimensions, wrapping. **Added 8 September 2026 by measurement** — see below |
| `tb.to_list` | `(T) → nested list` | the general escape; expensive by the count, and meant to be |

**Reading a value that is not the document**

| Operator | Signature | Notes |
|---|---|---|
| `tb.get` | `(value, path) → value` | **Added 8 September 2026 by the build.** `var` reads the *document*; there was no way to read a field out of an expression's result, and that gap blocks any nested accumulation — `reduce`'s seed is the only channel from outer scope into a body, so a body must carry its loop invariants in the accumulator and then cannot drop them again on the way out. Deriving a visibility mask and computing flat indices from a map width both die exactly there. Cost 1, no arithmetic: it moves information, it does not compute |

**Metadata and small helpers**, none of which touch elements: `tb.shape(T) → list`,
`tb.dtype(T) → string`, `tb.len(list) → int`, `tb.at(list, i) → value`.

The JSONLogic core is available in full — `var`, `missing`, `if`, the comparisons, the booleans, the
arithmetic, `map`, `filter`, `reduce`, `all`/`some`/`none`, `merge`, `in`, `cat`, `substr` — over
JSON values only. It is total, so a program always terminates; what it can do is spend, and spending
is what the count prices.

An `out` program, whole, for Ants — a policy of shape `[n_ants, 5]` becoming a move per ant:

```jsonc
{ "map": [ { "tb.argmax": [ { "var": "policy" }, 1 ] },
           { "tb.at": [ ["N", "E", "S", "W", "-"], { "var": "" } ] } ] }
```

## 4. The operation count

The unit, stated exactly enough to reimplement:

1. **Every JSONLogic node evaluated costs 1.** Applied, not written: a `map` body over 180 ants
   costs 180 times the body, and the same program over 3 ants costs 3 times.
2. **Every tensor operator costs `1 + max(elements read, elements produced)`.** Reading and
   producing are counted separately and the larger is charged, so `tb.zeros` on a 128² plane costs
   16,385 and `tb.argmax` over `[180, 5]` costs 901. `tb.reshape` reads and produces nothing — it is
   a view — and costs 1, as do the metadata operators.
3. **Literals cost 1** however large. A lookup table in the program is priced by `S` — its
   compressed bytes count toward the weight budget ([the weight classes](https://github.com/Tiny-Brains/docs)) — which is
   the right axis: an op cap prices computation, and `S` prices knowledge.
4. **The two directions are counted separately** and each is checked against `budget_ops`. A row
   reports `ops_in` and `ops_out` at validate and their sum at play.
5. **Over budget aborts that direction immediately**, mid-operator, and answers `ADAPTER_FAILED`
   with `over_budget: true`. It is a `caller_input` error in Orion's vocabulary and so is **never
   retried** — which is the whole reason the count exists rather than a timeout.

**It is a run-time count, not a static bound.** Finding 4 rejected the static bound: a worst case
over nested `map`s on data-sized lists is astronomically loose, so it either rejects real adapters or
needs a budget so large it prices nothing. The consequence is honest and must be said: an adapter can
pass admission and still exceed at play on an observation the reference set did not cover. That row
answers `ADAPTER_FAILED`, and Kalam turns it into a strike like a missed clock — the competitor's
own logic, on their own input, at their own cost.

**The count must be identical on two machines.** It is integer arithmetic over a deterministic walk,
so this is a property to test rather than to hope for; it is exactly what the counter spike
checks (§13). Note what is *not* claimed: the evaluator's float arithmetic need not be reproducible
across machines, and neither need the graph's (§10).

**On the number — measured 8 September 2026, and the prediction held.** The counter is built
(`axon/src/dialect/`) and `axon/tests/ants_adapter.rs` runs a reference adapter against a real
worst-case observation dumped from the spike engine: a 128×128 map at turn 600, 90 ants, 670 water
runs, 2,437 bytes.

| | ops | against 200,000 |
|---|---|---|
| the reference six-plane adapter, `in` | **197,272** | fits with **1.01×** headroom |
| the same adapter, `out` | 1,265 | — |
| deriving visibility, unrolled kernel | 217,269 | **does not fit** |
| deriving visibility, with `tb.dilate` | 32,777 | fits |

So `200000` is not wrong so much as exactly, uselessly, just enough: it accommodates the plainest
adapter anyone would write and nothing else. the game protocol §8.3 asks that a visibility-deriving
adapter fit *with headroom*, and at 200,000 it does not fit at all.

**Recommended: `adapter_ops_max = 1000000`.** A realistic eight-plane adapter with a visibility
plane is about 300,000, so a million is a little over three times the richest thing anyone has
written, and the cost is affordable: measured at **1.8 ms per million operations**, a million-op
budget is 1.8 ms a seat and 58 ms for 32 seats run serially, against a `turn_ms` of 1000. The
budget is a fairness rule rather than a performance one, and at this price there is no reason to
make it tight.

**Why `tb.dilate` exists.** It was added because the measurement argued for it, and the cost
argument is the weaker half. Without it the only expressible visibility mask is a 317-pair kernel
unrolled per ant — seven times the cost, and *wrong*: Ants maps wrap, the unrolled coordinates run
off the edges, `tb.scatter` drops them, and the mask is quietly missing its borders. A competitor
cannot fix that inside the dialect. The modulo needs the map's size; the map's size is not in scope
where the ant is; and the `reduce`-seed trick that would carry it there cannot give the points back
again, because nothing projects a field out of a computed value. The mask is **not expressible
correctly** without the operator. `tb.dilate` moves information outward by a fixed geometry and
cannot encode a policy, so it sits on the same side of §6's line as `scatter` and `rle_expand`.

## 5. The dialect version and the evaluator digest

Two identifiers, and the difference between them earns its keep.

**`dialect_version`** is an integer, declared by the competitor in `adapter.json` and implemented by
the evaluator. It changes when the operator table or the counting rules change — a new operator, a
changed cost, a changed semantic. It is the compatibility contract.

**`evaluator_digest`** is `sha256` over the **canonical description of the dialect**: the ordered
operator table with each operator's name, arity, dtypes and cost rule, the counting rules of §4,
and the JSONLogic feature set. It is *not* a hash of the binary. So a rebuild of `axon` — a
dependency bump, a performance fix, a new endpoint — reports the same digest, and only a real change
to what an adapter means moves it.

**The build added a fourth part to that description, and it is the part most likely to be
forgotten.** An adapter can observe semantic choices that no operator name implies, so the digest
covers those too — as built (`axon/src/dialect/digest.rs`), eleven of them: loose equality follows
JavaScript; `null` compares equal to `null` only, which is the one deliberate divergence from
`datalogic-rs`; truthiness is JSONLogic's; a single-key object whose key is a known operator is an
operation and nothing else is; `map` scope is the element only; `reduce`'s seed is the one channel
from outer scope; narrowing casts saturate; `argmax` returns the first maximum; a scatter index out
of bounds is dropped rather than refused; division by zero is `null`; and the expression depth cap
is 64. Change any one of them and every admitted adapter's *meaning* may have changed while its
program is byte-identical — which is exactly the case the re-validation sweep exists for, and
exactly the case a digest over the operator table alone would miss.

That is the property finding 5's re-validation sweep needs. Layer 01 records `models.evaluator_digest`
on every admitted version; admission sweeps when it changes. If the digest tracked the binary, every
patch release would re-validate the whole roster and the sweep would be noise. Both identifiers are
reported on **every** reply from every call, so a row, a manifest and a log line can all say which
evaluator produced them.

## 6. What the dialect deliberately cannot do

**There is no tensor arithmetic beyond `cast` and `normalise`.** No add, no multiply, no matmul, no
convolution. This is the single most important constraint in §4 and it is not an omission.

**The argument changed on 10 September 2026 and the rule did not** — worth stating, because the
justification used to be borrowed and is now intrinsic. It used to read: computation belongs in the
graph because the graph is where the FLOP cap prices it. There is no FLOP cap any more
(decision 46), and the rule stands anyway, on §4's own cost model.

**A tensor operator costs `1 + max(elements read, elements produced)`. That is a marshalling price,
and arithmetic does not obey it.** `tb.scatter` and `tb.stack` do work proportional to the elements
they touch, so the rule bounds them exactly. A `tb.matmul` of two 128×128 tensors would read 32,768
elements and produce 16,384 — a charge of 32,769 for four million multiplies. Every arithmetic
operator is under-priced by the counting rule *by the ratio of its arithmetic to its data*, and that
ratio is unbounded. Admitting one would not stretch the budget; it would mean the budget had stopped
measuring anything.

Two consequences follow, and both used to be the FLOP cap's job:

- **A second, unpriced model.** An adapter that could multiply matrices would sit in front of the
  priced one, and a Nano entry could carry a Small policy in its adapter — now caught by `S` rather
  than by a compute cap, since the adapter's exact bytes are half of what `S` compresses.
- **It is against the competitor's own interest anyway.** The dialect is a tree-walking interpreter
  over JSON; the graph runs in ORT, compiled, threaded and vectorised. Since the turn deadline is
  now the fairness control (axon/docs/design.md §10.2), arithmetic in the adapter spends the
  competitor's own deadline share at roughly a thousand times the cost of spending it in the graph.

The line to hold when an operator is requested is unchanged: **does it move or reshape information,
or does it compute with it?** `scatter`, `one_hot`, `rle_expand`, `stack`, `transpose` move.
`normalise` is on the line and is allowed because a fixed affine per tensor cannot encode a policy,
and because its work *is* proportional to its elements. Anything whose cost grows faster than the
data it touches is refused, and the answer is "put it in your ONNX graph".

The line to hold when an operator is requested: **does it move or reshape information, or does it
compute with it?** `scatter`, `one_hot`, `rle_expand`, `stack`, `transpose` move. `normalise` is on
the line and is allowed because a fixed affine per tensor cannot encode a policy. Anything with a
learnable-looking shape is refused, and the answer is "put it in your ONNX graph".

---

