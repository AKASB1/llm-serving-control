# Contracts, version 1

This document is the binding, language-neutral definition of the data formats, metric definitions, and policy semantics of this project. Other projects (for example `gpu-cluster-scheduler`) follow this document rather than importing the Go packages. Changes are deliberate, versioned, and logged.

Version: **1** · Status: defined by this repository.

## 1. Conventions

- **Time.** Policies and controllers read time only from an injected clock (virtual in the simulator, monotonic wall clock in the live service). They never sleep and never read the wall clock directly. Trace timestamps are seconds (decimal, float) from the start of the trace. Durations in code are integer nanoseconds (`time.Duration`).
- **Randomness.** Every component draws from its own stream, derived deterministically from the run seed and the component's name (Go: PCG from `math/rand/v2`, seeded from FNV-1a(name) mixed with the seed by SplitMix64). The trace a seed produces depends only on the generator configuration and the seed, never on which policies run, so every policy sees identical traffic (common random numbers).
- **Units.** Tokens are integers. Time is seconds. GPU cost is GPU-seconds or GPU-hours. There is no currency.
- **Determinism.** Given the same inputs (configuration, trace, seed), every policy and the simulator produce identical outputs. Ties are broken by replica ID (lexicographic) or by an explicit RNG draw, never by map iteration order.

## 2. Trace schema v1

A trace is a CSV file with exactly this header and seven columns:

```text
request_id,arrival_s,model,prompt_tokens,output_tokens,prefix_group,slo_class
```

| Column | Type | Rule |
|---|---|---|
| `request_id` | string | non-empty, unique within the trace, no comma |
| `arrival_s` | decimal | finite, ≥ 0; seconds since the start of the trace |
| `model` | string | non-empty |
| `prompt_tokens` | integer | ≥ 1 |
| `output_tokens` | integer | ≥ 1 (the number of tokens the backend will generate; the control plane does not see it) |
| `prefix_group` | string | may be empty; requests with the same group share a prompt prefix |
| `slo_class` | string | may be empty (→ the default class of the experiment) |

Rows are sorted by `arrival_s` (non-decreasing). Line endings LF or CRLF. A loader must reject: a missing or different header (line 1), a wrong field count, an unparsable or out-of-range number, an empty required field, a duplicate `request_id`, and a row whose `arrival_s` is smaller than the previous row's — each error names the 1-based line number. A file with a header and no rows is a valid empty trace; a zero-byte file is invalid.

**Manifest.** Beside `<name>.csv` lies `<name>.manifest.json`:

```json
{
  "schema_version": 1,
  "generator": {"name": "loadgen", "version": 1, "params": {"...": "..."}},
  "seed": 7,
  "requests": 12345,
  "duration_s": 600,
  "content_sha256": "<hex SHA-256 of the CSV bytes>"
}
```

A replayer verifies `schema_version` and `content_sha256` before use.

## 3. Request lifecycle and metric definitions

Arrival is the moment a request reaches the router. Each request ends in exactly one terminal state:

| State | Meaning |
|---|---|
| `completed` | all output tokens delivered |
| `rejected` | refused by admission or by the router (never dispatched, or dispatched then refused) |
| `failed` | a replica error after the first token, a non-retryable error, or retries exhausted |
| `timed_out` | waited in the router queue longer than the router-queue timeout |
| `unfinished` | still in progress when the run ended |

Per request (completed requests only, except where noted):

- **TTFT** = first-token delivery time − arrival. Includes router queueing, retries, replica queueing, and prefill.
- **TPOT** = (completion time − first-token time) / (output_tokens − 1), defined for output_tokens ≥ 2.
- **E2E** = completion time − arrival.
- **Router queue time** = time spent in the router queue (0 if dispatched immediately; summed over retries).
- **Replica queue time** = from arrival at the (final) replica to the start of the first engine iteration that includes the request.
- **ITL** (gaps between consecutive tokens of a request) is aggregated into one per-run histogram (10 log-spaced buckets per decade from 0.1 ms to 1000 s), not reported per request. Reported ITL quantiles are the upper bound of the bucket that holds the quantile (up to about 26 % above the exact value; +Inf in the overflow bucket).

**SLO attainment.** A request meets its SLO if and only if it completed, TTFT ≤ its class's TTFT target, TPOT ≤ its class's TPOT target (vacuously true when output_tokens < 2), and E2E ≤ its class's E2E target when one is set. Rejected, failed, timed-out, and unfinished requests are violations and stay in the denominator.

Per run:

- **Violation rate** = 1 − (requests meeting their SLO / all requests in the trace).
- **Goodput** = requests meeting their SLO / run duration (s).
- **Latency percentiles** are over completed requests only and are always reported next to the rejection, failure, timeout, and unfinished rates. Percentile definition: nearest-rank on the sorted sample (`ceil(p·n)`-th smallest).
- **GPU cost** = Σ over replicas of GPUs × (termination time − provisioning start), with replicas alive at the end of the run charged until the end. Warm-up and draining are paid. Reported in GPU-hours.
- **Utilization** = the fraction of a replica's serving time (ready or draining, until a crash) with a non-empty running batch, aggregated over replicas as Σ busy time / Σ serving time. **KV occupancy** = KV tokens in use / KV capacity, time-averaged the same way, reported separately.
- **Run duration** = the trace duration plus a fixed drain horizon, identical for every policy in an experiment; requests not finished by then are `unfinished`.

**Objective** (tuning and summary score):

```text
J = α · P95(TTFT)/T_ttft + β · P95(TPOT)/T_tpot + γ · GPU-hours/(GPU budget · run hours) + δ · violation rate
```

The latency terms are request-weighted averages over (model, SLO class) groups of P95(group)/target(group); the violation term is the overall violation rate (request-weighted by construction); the GPU term is global. A group with no completed request contributes the ratio of its timeout (the run duration) to its target, so starving a group is never rewarded. The weights live in the committed experiment configuration and are identical for every policy. Tail percentiles are the penalty terms; means are never used in J.

## 4. Replica snapshots (scraped state)

The control plane sees replica state only through snapshots published every scrape interval (default 1 s) and through its own router-local counters.

| Field | Kind | Meaning |
|---|---|---|
| `replica_id` | label | |
| `at` | time | when the snapshot was taken |
| `running`, `waiting` | gauge | sequences in the running batch / FCFS waiting queue |
| `waiting_prompt_tokens` | gauge | prompt tokens not yet prefilled |
| `kv_used_tokens`, `kv_capacity_tokens` | gauge | KV cache occupancy |
| `preemptions`, `completed_requests`, `generated_tokens` | counter | cumulative |
| `busy_time` | counter | cumulative time with a non-empty running batch |
| `prefix_hits`, `prefix_queries` | counter | prefix-cache hits and lookups (0 when the backend has no prefix cache) |
| `remaining_decode_tokens` | oracle only | Σ (output − generated) over the replica's sequences; needs output lengths, so only the simulator fills it, only in the fresh state given to oracle policies; scraped snapshots carry 0 |

## 5. Policy semantics

All policies share these rules: a name; a JSON parameter object (`{"name": "...", "params": {...}}`); explicit inputs (a read-only view) plus private internal state; deterministic given their RNG stream; time read only from the view or an injected clock; no dependency on the simulator or on HTTP; constructed by one factory used by the simulator, the benchmark, and the live service.

### 5.1 Router

Input: the request (`id`, `model`, `prompt_tokens`, optional `max_output_tokens`, `prefix_group`, `slo_class`; not the output length) and a view of the **eligible** replicas (ready, healthy, serving the model; sorted by ID). Each replica view carries: class prior (GPUs, nominal prefill and decode rates, max sequences, KV capacity), router-local in-flight count (fresh), dispatches since the latest snapshot, the latest snapshot (stale), and the replica's SLO error. Oracle policies additionally receive the true current state; they are labelled oracles and cannot run live.

Output: dispatch to one replica of the view. When the view is empty the dispatcher queues the request without calling the router. Routers may receive outcome feedback (first token, completion, failure with latencies measured from dispatch).

### 5.2 Admission

Input: the same request and view as the router plus the router-queue length. Output: admit, queue, or reject. Rejected requests are violations.

### 5.3 Dispatcher (shared runtime, not a policy)

Order of operations for an arriving request: admission → (if admitted) router → dispatch. Queued requests wait FIFO per model and are retried whenever a replica becomes eligible, a snapshot arrives, or a request finishes; a request that waits longer than the router-queue timeout is `timed_out`. On a replica error **before the first token**, the request is retried on another eligible replica up to the retry limit; after the first token it fails. Health: a replica is ejected after K consecutive errors (or a failed active probe) for a backoff that doubles on each consecutive ejection up to a cap, then reinstated on probation; one success resets the backoff.

### 5.4 Scaler

Input, per model: replica counts by lifecycle state (ready, starting = provisioning + loading, draining), minimum and maximum, expected warm-up, windowed signals (arrival rate, completion rate, output-token rate, in-flight averaged over samples, queue depth, utilization, KV usage, service rate per busy replica, mean E2E, mean service time), and the SLO error and violation rate. Output: a desired count. Scale-in drains replicas (no new traffic, in-flight work finishes within a grace period); it never kills in-flight work while the grace period lasts.

### 5.5 Allocator

Input: per model the desired and current counts, minimum and maximum, GPUs per replica, arrival rate, mean service time (E2E minus router and replica queue time: how long a request holds a batch slot), slots per replica (default: the class's maximum running sequences), TTFT target, SLO error; and the GPU budget. Output: granted counts with a reason. Guarantees: minimums are honoured first; no model exceeds its maximum; Σ granted × GPUs ≤ budget whenever the minimums fit; deterministic.

### 5.6 SLO controller

Input: per finished request (model, class, replica, completion state, TTFT, TPOT). Output over a sliding window: per model and per replica `ttft_ratio = P95(TTFT/T_ttft)`, `tpot_ratio = P95(TPOT/T_tpot)`, the violation rate, and the **error** `max(ttft_ratio, tpot_ratio) − 1` (positive = missing targets). Buffers are bounded. The error feeds scaling (`slo_feedback`), allocation, and routing weights.

## 6. Live data plane (HTTP)

The live control plane accepts the OpenAI-compatible subset `POST /v1/chat/completions` and `POST /v1/completions` and routes by the body's `model`. Semantics shared with the simulator:

- A request is retried on another replica only if nothing has reached the client yet: the client receives the response headers and body only after the first body byte has arrived from a replica. After that, a replica error truncates the stream and the request is `failed`.
- Admission rejection → `429 Too Many Requests` with `Retry-After`; router-queue timeout → `503` with `Retry-After`; retries exhausted → `502`. A replica's `4xx` is relayed unchanged and not retried.
- When the client disconnects, the upstream request is cancelled and the request leaves the dispatcher (no health penalty).
- Replica state comes from scraping `<endpoint>/metrics` (vLLM or SGLang names, see `internal/adapters`) and from `GET <endpoint>/health` probes.
- Optional request header `X-SLO-Class` selects the SLO class. The mock backend additionally reads `mock_prompt_tokens` and `mock_output_tokens` from the body.
- Admin: `GET /admin/replicas`, `POST /admin/replicas` (`{"id","model","endpoint","class"}`), `DELETE /admin/replicas/{id}`; metrics at `GET /metrics` (Prometheus text format, `lsc_*` names).

## 7. Versioning

A change to any table or rule above increments the version and is logged in the repository's decision log. Readers reject an unknown `schema_version`.
