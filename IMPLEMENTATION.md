# Implementation plan

## Control-plane boundary

Token generation stays in the serving backends, which expose replicas and metrics; the control plane decides where requests go and how much capacity each model receives. The evaluation runs against a discrete-event simulator of serving replicas ([docs/simulator.md](docs/simulator.md)). The policy code is written once and is meant to run unchanged in front of real or mock backends.

## Core entities

### Backend

A serving backend exposes health, model inventory, replica endpoints, queue depth, and latency metrics. Interface: `internal/backends.Backend` (vLLM and SGLang adapters in `internal/adapters`).

### Replica

Model, class (hardware), lifecycle state (provisioning, loading, ready, draining, terminated, failed), router health, router-local in-flight count (`internal/registry`); scraped state (running, waiting, waiting prompt tokens, KV usage, counters) in `internal/metrics.ReplicaSnapshot`.

### Routing policy

Implemented in `internal/routing`: random, round robin, least outstanding requests, power of two choices, latency-aware (peak EWMA), and queue-aware (scraped queue depth and KV usage) with its corrected variant (stale-snapshot mitigation). Also capacity-weighted (smooth weighted round-robin from measured service rates) and an oracle (true remaining work) for reference.

### Autoscaler

Implemented in `internal/autoscaling`: static, threshold with cooldown, target tracking with stabilization, SLO feedback (dead band, hysteresis, integral term), and predictive (Holt forecast one warm-up ahead, Little's law). Inputs: queue depth, arrival rate, service rate, utilization, SLO error from windowed signals (`internal/metrics.Window`, `internal/slo.Controller`).

## Metrics

Defined in [docs/contracts.md](docs/contracts.md) §3 and computed in `internal/metrics`: time to first token, time per output token, inter-token latency (histogram), end-to-end latency, router and replica queue time, and tokens/s. Also replica utilization, KV occupancy, the SLO violation rate (rejected, failed, timed-out, unfinished requests included), goodput, and GPU-hours (warm-up and draining paid).

## Delivery order

Revised: traffic replay and a simulated backend come before real adapters, and the policy benchmarks before Kubernetes.

- [x] 1. backend registry
- [x] 2. static routing
- [x] 3. traffic replay / load generator and simulated backend
- [x] 4. metrics ingestion
- [x] 5. health-aware routing
- [x] 6. routing-policy benchmark
- [x] 7. replica autoscaler (hysteresis, cooldown)
- [x] 8. multi-model capacity allocation
- [x] 9. SLO controller and scaling / capacity benchmark
- [ ] 10. Kubernetes adapter and real-backend adapters

## Decision problem

- Routing (online): pick a replica for each request from health, router-local in-flight, scraped queue depth and KV usage, and recent latency.
- Capacity (periodic): choose replicas per model under a GPU budget.
- Scaling (control): add or remove replicas from arrival rate, service rate, and SLO error.
- Objective: `J = α·P95(TTFT)/T_ttft + β·P95(TPOT)/T_tpot + γ·GPU-hours/(budget·duration) + δ·violation rate`, weights fixed per experiment (`configs/experiment.json`).
- Uncertainty: arrival rate, request and output length, service rate. Tail targets (P95, P99) are penalties, not averages.

## Evaluation and acceptance

Workloads: steady, bursty, and diurnal arrival; short-heavy and long-heavy request mixes; one model, and three models sharing a GPU budget; heterogeneous replicas; a crash; sustained overload. Protocol and results: [benchmarks/README.md](benchmarks/README.md).

- [x] one command replays fixed traces against nine routing policies and seven scaling configurations (`go run ./cmd/benchmark`)
- [x] assumptions for the simulated backend (iteration-time model, KV cache, preemption, warm-up and scale delays, failures) are written down ([docs/simulator.md](docs/simulator.md))
- [x] the README has an architecture diagram and result figures produced by committed scripts, and a result table copied from `docs/figures/results_summary.md`

## Cross-project contracts

This project defines **version 1** of the trace schema, the metric definitions, and the policy interfaces in [docs/contracts.md](docs/contracts.md) (language-neutral). `gpu-cluster-scheduler` is expected to adopt them; replica placement is where the two projects meet. This project depends on no other project.

## Implementation checkpoint

Done and tested: trace schema v1 and generators. The discrete-event simulator (continuous batching, chunked prefill, KV capacity, preemption by recompute, lifecycle, scrapes, crashes, drain, GPU-hour accounting) has invariant tests and a Pollaczek–Khinchine closed-form check. Nine routing policies and health handling (passive ejection with backoff, active probes, retries before the first token). Windowed signals, the SLO controller, five scalers, three allocators, and three admission policies; the shared dispatcher and control loop. The benchmark has frozen tuning, 20 evaluation seeds, confidence intervals, paired comparisons, oracle references, and a sensitivity check. Figures and tables come from committed results, and CI runs the checks.

Also done and tested (live path and extensions): an OpenAI-compatible proxy over the shared dispatcher. It passes SSE through, retries only before the first byte, cancels upstream requests when clients disconnect, returns 429 with Retry-After, and has probes, scraping, Prometheus metrics, and an admin API. A mock backend running the engine in real time, a process executor, and an end-to-end demo script (PowerShell and bash). The vLLM and SGLang metric adapters are tested against fixtures written from their sources (never against a live server) and have a fuzzed parser. Converters for the Azure 2023 and BurstGPT traces, and replay results on Azure and BurstGPT excerpts. Rankings under other objective weights, recomputed from the per-run metrics. A prefix-cache model and a `prefix_affinity` router with its own evaluation, and a JSON-file replica store with restart recovery.

Not done: Kubernetes executor and manifests, real-backend runs (vLLM/SGLang on a GPU), gRPC admin API, PostgreSQL/Redis stores, OpenTelemetry.
