# Simulator model and assumptions

Every result produced with this simulator is **simulated, with assumed parameters**. The simulator is a model of LLM serving replicas, not a measurement of any real system. Nothing here describes the measured behaviour of vLLM, SGLang, or any GPU.

Code: `internal/sim` (event loop, lifecycle, failures, accounting) and `internal/sim/engine` (one replica's scheduler). The control plane on top (`internal/controller`, the policies) is the same code the live service uses.

## 1. Event loop

- Discrete-event with a virtual clock (`time.Duration` nanoseconds since the start of the trace). No sleeps, no wall clock, no goroutines inside a run. Each run is deterministic given its configuration, trace, and seed; independent runs execute in parallel in the benchmark.
- Events are ordered by (time, insertion sequence). Trace arrivals are merged with the heap; on a tie the heap event goes first.
- The run lasts the trace duration plus a fixed **drain horizon** (default 60 s, identical for every policy). Requests still in progress at the end are `unfinished` (violations). Replicas alive at the end are paid until the end.
- Periodic events: scrape (default every 1 s), dispatcher tick (0.5 s: health reinstatement, router-queue timeouts), control step (5 s: scalers and allocator), active probes (2 s).
- Router→replica network latency is zero (assumed).

## 2. Replica engine (iteration-level continuous batching)

A replica is one instance of one model on a given number of GPUs. State: a FCFS waiting queue and a running set (admission order). Each iteration, at time t:

1. **Decode.** Every running sequence whose prompt is complete decodes one token. If the KV growth would exceed capacity, the **most recently admitted** sequence is preempted by **recompute**: its KV is freed, it goes back to the **head** of the waiting queue, and when re-admitted it must prefill prompt + already generated tokens again. Preemptions are counted.
2. **Continue prefills.** The remaining token budget (`max_batched_tokens − n_decode`) is given, in admission order, to admitted sequences whose prompt is only partly processed (chunked prefill).
3. **Admit.** While budget remains, `running < max_num_seqs`, and the KV cache can hold the **whole prompt** of the sequence at the head of the waiting queue, that sequence is admitted and receives a prompt chunk. The head is never skipped (FCFS).
4. **Iteration time** `= t_base + c_seq·n_decode + c_ctx·context_tokens + c_pf·prefill_tokens`, where `context_tokens` is the KV held by the scheduled sequences before the iteration.
5. At the end of the iteration every decoding sequence emits one token; a sequence whose prompt became complete emits its **first token**; later tokens come one per iteration. A sequence that has emitted all `output_tokens` completes and frees its KV.

Simplifications (binding, documented):
- KV is counted in tokens (no block granularity, no fragmentation).
- Reserving KV for the whole prompt at admission is a simplification of vLLM's per-step block allocation; it rules out the prefill deadlock that per-step allocation resolves with extra preemptions.
- Attention cost of a prefill chunk over itself is folded into `c_pf` (no quadratic term).
- Requests with `prompt + output > max_model_len` are refused by the replica (non-retryable failure); generators cap lengths so this does not happen in the benchmark.
- The engine knows `output_tokens` only to decide when a sequence finishes (the end-of-sequence token); the control plane never sees it.

The engine has no event loop of its own: `Start(now)` returns the next iteration's duration and `Finish(now)` applies it. The simulator calls them at virtual times; a mock backend can call them at real times.

### Prefix cache (optional, off in the main experiment)

When a scenario sets `prefix_cache_groups` > 0, each replica keeps an LRU of that many prefix groups. A request whose trace row carries a `prefix_group` shares its first `prefix_tokens` prompt tokens with the group (capped at prompt − 1). On admission (first time only, not after a preemption) the engine looks the group up: on a hit those tokens need no prefill (they still count in the sequence's KV and in the attention context); on a miss the group is inserted when the sequence's prefill completes, evicting the least recently used group. Simplifications: the cached prefix does not occupy separate KV capacity, and recompute after a preemption does not consult the cache. Only `configs/prefix.json` enables it.

## 3. Lifecycle

`provisioning → loading → ready → draining → terminated`; a crashed replica becomes `failed`. Only `ready` replicas get new traffic. Delays per class: provisioning (node and pod start-up, assumed 30 s) and loading (weight load and warm-up, assumed 60–150 s by model size). Draining finishes in-flight work; after the **grace period** (default 180 s) the remaining requests are aborted and retried elsewhere if they have not produced a token yet (otherwise they fail). A replica drained before it became ready is cancelled immediately. Initial replicas of a scenario are ready at time 0 and paid from time 0.

**GPU cost** = Σ GPUs × (end − provisioning start): provisioning, loading, and draining are paid. A crashed replica is paid until the executor notices the crash (default 10 s after it).

## 4. Visibility

The control plane sees replica state only through:
- **snapshots** published every scrape interval (running, waiting, waiting prompt tokens, KV used/capacity, cumulative preemptions, completions, generated tokens, busy time) — stale by up to one interval;
- its own **router-local counters** (in-flight per replica, dispatches since the last snapshot), always fresh.

**Stale-snapshot herding is expected and not fixed silently**: a policy that ranks replicas by the scraped queue depth (`queue_aware`) sends every request between two scrapes to the replica that looked emptiest. The mitigation is a separate, documented variant (`queue_aware_corrected`, which adds router-local dispatches since the snapshot). Only oracle policies (`oracle_jsq`) see the true current state, and they are labelled oracles.

## 5. Failures

- `crash(replica, t)`: the engine loses all state. In-flight requests see an error after `fail_delay_s` (default 0: connection reset). Requests dispatched to the dead replica before the router notices fail after `connect_timeout_s` (1 s).
- The router notices through **consecutive errors** (passive ejection after 3 in a row) or a failed **active probe** (every 2 s, 1 s timeout). Ejection backoff starts at 5 s and doubles per consecutive ejection up to 60 s; after it, the replica is reinstated on probation (one failure re-ejects it).
- **Retries** are bounded (2 by default) and happen only before the first token; a request that already streamed tokens fails.
- The executor (the simulated cluster) marks the replica `failed` after `executor_detect_s` (10 s); the control loop then sees one replica fewer and may provision a replacement.

## 6. Replica classes (`configs/classes.json`)

Parameters derive from public specification-sheet numbers and the model architecture, with **assumed** efficiency factors. Derivation (`internal/sim/classes.go`), with aggregate values = per-GPU value × GPUs × tensor-parallel efficiency:

```text
t_base = weight bytes / (memory bandwidth · eff_mem_bw) + iteration overhead
c_ctx  = KV bytes per token / (memory bandwidth · eff_mem_bw)
c_pf   = 2 · parameters / (peak FLOPS · eff_flops)
c_seq  = c_pf + per-sequence overhead
KV capacity = (GPUs · memory · mem_util − weights − activation reserve · GPUs) / KV bytes per token
```

| Input | Value | Kind and source |
|---|---|---|
| H100 SXM5 80GB: memory, bandwidth, dense BF16 | 80 GB, 3350 GB/s, 989 TFLOPS | datasheet ("NVIDIA H100 Tensor Core GPU" datasheet), transcribed |
| A100 SXM4 80GB: memory, bandwidth, dense BF16 | 80 GB, 2039 GB/s, 312 TFLOPS | datasheet ("NVIDIA A100 Tensor Core GPU" datasheet), transcribed |
| V100 SXM2 32GB: memory, bandwidth, FP16 tensor | 32 GB, 900 GB/s, 125 TFLOPS | datasheet ("NVIDIA Tesla V100 GPU Accelerator" datasheet), transcribed |
| 8B model: parameters, KV bytes/token | 8.03 B, 131072 B (32 layers × 8 KV heads × 128 dim × 2 (K,V) × 2 B) | architecture of a public 8B grouped-query-attention model |
| 70B model: parameters, KV bytes/token | 70.6 B, 327680 B (80 layers × 8 × 128 × 2 × 2 B) | architecture of a public 70B grouped-query-attention model |
| Weights | 2 bytes per parameter (BF16/FP16) | assumed |
| `eff_mem_bw`, `eff_flops` | 0.7, 0.5 | assumed |
| `tp_efficiency` (4-GPU tensor parallel) | 0.85 | assumed |
| `mem_util`, activation reserve | 0.9, 2 GB per GPU | assumed (typical serving-engine settings) |
| iteration overhead, per-sequence overhead | 4 ms (6 ms V100, 70B), 20 µs (30 µs V100) | assumed |
| `max_num_seqs`, `max_batched_tokens`, `max_model_len` | 128, 2048, 16384 | assumed serving-engine settings |
| provisioning / loading delay | 30 s / 60 s (8B), 75 s (8B on V100), 150 s (70B) | assumed |
| GB | 10⁹ bytes | convention |

Derived values (printed by `go test ./internal/sim -run TestDeriveClasses -v`):

| Class | GPUs | t_base | c_seq | c_ctx | c_pf | KV tokens | nominal prefill tok/s | nominal decode tok/s (128 seqs, 1k ctx) |
|---|---|---|---|---|---|---|---|---|
| h100-8b (and h100-8b-code) | 1 | 10.85 ms | 52.5 µs | 0.0559 µs | 32.5 µs | 411529 | 26473 | 5142 |
| a100-8b | 1 | 15.25 ms | 122.9 µs | 0.0918 µs | 102.9 µs | 411529 | 9058 | 2975 |
| v100-8b | 1 | 31.49 ms | 287.0 µs | 0.2081 µs | 257.0 µs | 81939 | 3672 | 1340 |
| h100x4-70b | 4 | 23.71 ms | 104.0 µs | 0.0411 µs | 84.0 µs | 423583 | 10465 | 3018 |

The linear model sums memory and compute time instead of taking a roofline maximum; with the efficiency factors this is a deliberate, simple approximation of the regression-style iteration models used by serving simulators. It has not been fitted to any measurement.

## 7. Calibration protocol (written, not executed)

To fit the parameters to a real deployment (one model, one GPU type, one serving-engine version):

1. **t_base, c_seq, c_ctx**: run decode-only batches (prompts already cached or 1-token prompts) at batch sizes {1, 2, 4, …, max_num_seqs} and context lengths {128, 1k, 4k, 16k}; record per-iteration time from the engine's step timer (or `vllm:inter_token_latency_seconds`); fit `t = t_base + c_seq·n + c_ctx·Σctx` by least squares; report R² and residuals.
2. **c_pf**: run prefill-only iterations with chunk sizes {256, 512, 1024, 2048}; fit the slope against prefill tokens with the decode terms fixed.
3. **KV capacity**: read the engine's reported number of KV blocks × block size at start-up; check against `kv_cache_usage_perc` under load.
4. **Preemption**: drive a long-output workload into KV saturation; compare preemption counts and the preempt-recompute latency penalty against the model.
5. **Lifecycle delays**: time pod scheduling → container start → weights loaded → first successful health check, 10 repetitions each, cold and warm image cache.
6. **Validation**: replay a held-out trace at 50 %, 70 %, 90 % load; compare TTFT/TPOT P50/P95 and throughput against the simulator with the fitted parameters; accept when errors are within a stated band (for example 10 % on P50, 20 % on P95).

## 8. Closed-form check

`internal/sim/pk_test.go`: one replica, batch size one (`max_num_seqs = 1`), Poisson arrivals at ρ ≈ 0.67, lognormal lengths. The replica is then an M/G/1 FCFS queue, and the mean replica queue wait must match Pollaczek-Khinchine, `W_q = λ·E[S²] / (2·(1 − λ·E[S]))`, with E[S] and E[S²] taken from the realized service times. Tolerance: pooled over 10 seeds (≈ 20 000 requests each) within 5 %; every seed within 25 %. (With exponential service times this reduces to the M/M/1 value ρ/(μ − λ).)

## 9. Invariant tests

`internal/sim/sim_test.go` and `internal/sim/engine/engine_test.go`: every request ends in exactly one terminal state (the simulator panics on a second one); time never goes backwards; KV occupancy (used + reserved) never exceeds capacity, checked after every event; GPU-seconds equal the integral of GPUs held by live replicas, computed independently from event-by-event observation; preemption sends the most recently admitted sequence back to the head and it still emits exactly `output_tokens` tokens with one first token; a drained replica receives no new requests and loses none; a crash produces retries before the first token and failures after it; identical inputs give identical records.
