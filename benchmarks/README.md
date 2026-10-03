# Benchmarks

Every number in this directory is **simulated, with assumed parameters** (see [docs/simulator.md](../docs/simulator.md)). The benchmark compares control-plane policies on identical replayed traffic against simulated replicas. It says nothing about production performance or about real vLLM or SGLang servers.

## Reproduce

```bash
go run ./cmd/benchmark -tune
go run ./cmd/benchmark
python scripts/plot_results.py
```

1. `-tune` runs every parametrised policy on the **tuning seeds** (101–105) with the same random-search budget (12 configurations, search seed 20261001; configuration 0 is the policy's defaults) and writes `configs/tuned/tuned.json` and `configs/tuned/tuning_log.csv`. These files are committed **before** the evaluation; the evaluation never re-tunes. (14 min 18 s on the machine in the manifest, shared with unrelated jobs.)
2. The evaluation runs every policy of every family on the **evaluation seeds** (1–20) plus the sensitivity variants (seeds 1–10), writes one row per run to `results/runs.csv`, and the aggregates beside it. (14 min 19 s and 9 min 59 s in two runs on the same shared machine; the second run, on the final commit, produced the committed files.)
3. The plotting script reads only the committed result files and writes `docs/figures/*.png` and `docs/figures/results_table.md`.

`go run ./cmd/benchmark -quick` runs a smoke subset (three scenarios at a quarter of their duration, seeds 1–3, including a replay of the committed sample trace `configs/traces/sample.csv`) into `outputs/quick/` in a few seconds; CI runs it on every push.

Traces are generated from the committed generator configuration into `outputs/traces/` (ignored by git) and replayed from those files; each run row records the trace's SHA-256 prefix. The manifest (`results/manifest.json`) records the commit, the SHA-256 of the configuration files (experiment, classes, tuned configuration, sample trace), the commit the tuning ran on, the seeds, the Go version, the platform, the hardware, and the measured per-replica capacities. Result files contain no timestamps, so a rerun on the same commit is byte-identical (tested in `cmd/benchmark`).

## Protocol (fixed before the first evaluation run)

**Objective.** `J = 1·P95(TTFT)/T_ttft + 1·P95(TPOT)/T_tpot + 1·GPU-hours/(GPU budget·run hours) + 5·violation rate` (definitions in [docs/contracts.md](../docs/contracts.md) §3). Targets: `interactive` TTFT 2 s, TPOT 100 ms; `batch` TTFT 20 s, TPOT 250 ms. Lower J is better. The weights were set before any run and are the same for every policy.

**Seeds.** Tuning {101…105}; evaluation {1…20}; sensitivity {1…10} (a subset of the evaluation seeds, never used for tuning).

**One dimension at a time.** Routing scenarios fix the scaler (static at the initial count); scaling scenarios fix the router (the default chosen on the tuning seeds: the non-oracle router with the lowest tuning J); allocation fixes both (router as above, scaler = the adaptive scaler with the lowest tuning J); overload fixes the router and a static scaler.

**Statistics.** Mean and two-sided 95 % Student-t interval over seeds. Paired difference in J against the family baseline on common seeds with its t-interval; verdict "better"/"worse" only when that interval excludes zero, otherwise "no difference". Per-seed wins/ties/losses with a tie band of 1 % of the baseline's J. Baselines: routing `least_outstanding`, scaling `threshold_cooldown` (the baseline named in the original project plan), allocation `static_partition`, overload `none`.

**Capacity reference.** Arrival rates are fractions of each class's saturation throughput under each length mix, measured by a probe in the simulator (one replica, every request queued at once, completions counted from 120 s to 600 s) and recorded in the manifest.

**Length mixes.** short-heavy: prompt lognormal (median 256, σ 1.0, 8–4096), output lognormal (median 128, σ 1.2, 1–2048). long-heavy: prompt lognormal (median 2048, σ 0.7, 128–12288), output lognormal (median 256, σ 1.2, 1–4096). Prompt + output ≤ 16384.

## Scenarios

| ID | Family | Replicas | Traffic | Duration |
|---|---|---|---|---|
| r-steady70-short / -long | routing | 4 × h100-8b, static | Poisson at 70 % of capacity | 600 s |
| r-steady90-short / -long | routing | 4 × h100-8b | Poisson at 90 % | 600 s |
| r-bursty-short / -long | routing | 4 × h100-8b | MMPP on/off (2.0× for ~15 s, 0.5× for ~30 s), mean 70 % | 600 s |
| r-hetero | routing | 2 × h100-8b, 2 × a100-8b, 2 × v100-8b | Poisson at 80 % of the summed capacity | 600 s |
| r-crash | routing | 4 × h100-8b | Poisson 70 %; replica `h100-8b-001` crashes at 300 s | 600 s |
| s-diurnal-short / -long | scaling | h100-8b, 5 at start, 1–16 | diurnal (thinning), amplitude 0.8, period 1800 s, mean 70 % of 6 replicas | 1800 s |
| s-bursty-short / -long | scaling | same | MMPP with state rates 2.03× (mean sojourn 120 s) and 0.49× (240 s) the mean (configured multipliers 2.5 and 0.6, renormalised to mean 1), mean 70 % of 6 replicas | 1800 s |
| c-3model | capacity | chat-8b (h100-8b), code-8b (h100-8b-code), chat-70b (4 × H100); budget 20 GPUs, below the sum of the peaks | diurnal per model, phases shifted by a third of a period | 1800 s |
| o-overload | overload | 4 × h100-8b, static | 60 % → 130 % (300–600 s) → 60 %; 70 % interactive, 30 % batch | 900 s |

All runs add a 60 s drain horizon; requests unfinished by then are violations.

**Policies.** Routing: `random`, `round_robin`, `least_outstanding`, `power_of_two`, `latency_aware`, `queue_aware`, `queue_aware_corrected`, `capacity_weighted`, `oracle_jsq` (oracle). Scaling: `static_min` (1 replica), `static_tuned` (count chosen on the tuning seeds), `static_peak_oracle` (peak rate known in hindsight, sized at 80 % utilisation; an oracle reference, not a bound: `predictive` beats it on s-diurnal-short), `threshold_cooldown`, `target_tracking`, `slo_feedback`, `predictive`. Allocation: `static_partition`, `proportional_demand`, `marginal_gain`. Admission: `none`, `queue_cap`, `predicted_ttft_shed`.

**Sensitivity.** On `r-steady90-short` and `s-bursty-short`: warm-up ×0.5 and ×2, scrape interval 0.25 s and 5 s, lighter (σ 0.8) and heavier (σ 1.6) output tail (same absolute arrival rate, so the heavier tail is also more load). Reported: Kendall τ between the base and variant rankings of non-oracle policies by mean J, whether the top policy stays on top, and how many per-policy verdicts against the baseline stay the same.

## Results

Condensed (mean J and verdict against the family baseline); every metric with its interval and win/tie/loss counts: [docs/figures/results_table.md](../docs/figures/results_table.md). Raw rows: `results/runs.csv`; aggregates: `results/aggregates.csv`; paired comparisons: `results/paired.csv`; sensitivity: `results/sensitivity.csv`; provenance: `results/manifest.json`.

<!-- results-summary:start -->
<!-- generated by scripts/plot_results.py from benchmarks/results; do not edit by hand -->
Mean J over 20 evaluation seeds (lower is better). ✓ / ✗ / = : better / worse / no difference than the family baseline (paired 95% interval of ΔJ). Oracles are references, not deployable. Simulated, assumed parameters.

**routing** (baseline `least_outstanding`)

| policy | r-steady70-short | r-steady70-long | r-steady90-short | r-steady90-long | r-bursty-short | r-bursty-long | r-hetero | r-crash |
|---|---|---|---|---|---|---|---|---|
| random | 1.282 ✗ | 1.569 ✗ | 1.393 ✗ | 1.974 ✗ | 10.863 ✗ | 3.889 ✗ | 126.6 ✗ | 1.340 ✗ |
| round_robin | 1.261 ✓ | 1.488 ✓ | 1.343 ✗ | 1.781 = | 10.790 ✗ | 3.478 ✗ | 126.6 ✗ | 1.320 ✓ |
| least_outstanding (baseline) | 1.264 | 1.497 | 1.341 | 1.745 | 10.600 | 3.329 | 2.190 | 1.323 |
| power_of_two | 1.267 ✗ | 1.518 ✗ | 1.345 ✗ | 1.770 ✗ | 10.638 = | 3.362 = | 2.365 ✗ | 1.325 ✗ |
| latency_aware | 1.274 ✗ | 1.531 ✗ | 1.364 ✗ | 1.840 ✗ | 10.843 ✗ | 3.923 ✗ | 1.658 ✓ | 1.332 ✗ |
| queue_aware | 1.494 ✗ | 1.886 ✗ | 1.771 ✗ | 2.391 ✗ | 11.182 ✗ | 4.436 ✗ | 3.430 ✗ | 1.522 ✗ |
| queue_aware_corrected | 1.258 ✓ | 1.475 ✓ | 1.334 ✓ | 1.720 ✓ | 10.659 ✗ | 3.295 = | 2.092 ✓ | 1.316 ✓ |
| capacity_weighted | 1.286 ✗ | 1.529 ✗ | 2.629 ✗ | 2.654 = | 11.230 ✗ | 3.658 ✗ | 2.561 ✗ | 1.664 ✗ |
| oracle_jsq (oracle) | 1.279 ✗ | 1.523 ✗ | 1.362 ✗ | 1.762 ✗ | 10.696 ✗ | 3.485 ✗ | 1.529 ✓ | 1.329 ✗ |

**scaling** (baseline `threshold_cooldown`)

| policy | s-diurnal-short | s-diurnal-long | s-bursty-short | s-bursty-long |
|---|---|---|---|---|
| static_min | 580.1 ✗ | 576.2 ✗ | 654.9 ✗ | 642.0 ✗ |
| static_tuned | 0.922 ✓ | 1.097 = | 3.257 ✓ | 1.905 ✓ |
| static_peak_oracle (oracle) | 0.878 ✓ | 1.080 ✓ | 3.257 ✓ | 1.905 ✓ |
| threshold_cooldown (baseline) | 0.967 | 1.106 | 15.713 | 20.867 |
| target_tracking | 1.099 ✗ | 1.144 ✗ | 4.111 ✓ | 3.243 ✓ |
| slo_feedback | 19.371 ✗ | 19.440 ✗ | 54.123 ✗ | 98.565 ✗ |
| predictive | 0.836 ✓ | 1.211 = | 39.224 ✗ | 46.621 ✗ |

**capacity** (baseline `static_partition`)

| policy | c-3model |
|---|---|
| static_partition (baseline) | 7.311 |
| proportional_demand | 5.072 ✓ |
| marginal_gain | 1.687 ✓ |

**overload** (baseline `none`)

| policy | o-overload |
|---|---|
| none (baseline) | 32.261 |
| queue_cap | 2.435 ✓ |
| predicted_ttft_shed | 2.361 ✓ |
<!-- results-summary:end -->

Figures: [routing](../docs/figures/routing_J.png), [scaling](../docs/figures/scaling_tradeoff.png), [capacity](../docs/figures/capacity.png), [overload](../docs/figures/overload.png), [sensitivity](../docs/figures/sensitivity.png).

**Sensitivity** (`results/sensitivity.csv`): routing on r-steady90-short keeps its ranking under warm-up ×0.5/×2 (τ = 1, trivially: warm-up does not matter for a static fleet without crashes, and those runs are identical to the base runs), scrape 0.25 s (τ = 1) and 5 s (τ = 0.93), and the lighter tail (τ = 0.786); the heavier tail gives τ = 0.5 and `least_outstanding` takes the top spot from `queue_aware_corrected`. Scaling on s-bursty-short keeps `static_tuned` on top in five of six variants (τ 0.87–1); with the heavier tail `target_tracking` leads. The conclusion that survives everything: stale-snapshot routing (`queue_aware`) and latency-error scaling (`slo_feedback`) lose; the fine ordering among the good policies depends on the output-length tail.

**Development note on `marginal_gain`.** A tuning dry run (tuning seeds only, before the frozen tuning and before any evaluation run) showed `marginal_gain` far behind the other allocators. The cause was a defect in its inputs: the control loop passed the mean E2E latency (which includes queueing) as the service time, and scenarios used 32 slots per replica while the engine runs up to 128 sequences, so every model looked unstable and the greedy gains collapsed to zero. Both inputs were corrected (service time = E2E minus router and replica queue time; slots default to the class's `max_num_seqs`) before the frozen tuning. The other allocators do not use these inputs.

## Replayed public traffic

`configs/replay.json` replays two committed excerpts of the Azure LLM inference trace 2023 (conversation service, first 300 s, 1445 requests; coding service, first 600 s, 1482 requests; CC-BY 4.0, see [configs/traces/README.md](../configs/traces/README.md)) against one simulated `a100-8b` and one simulated `v100-8b` replica, with the routing configurations frozen by the main tuning. The cluster was chosen before the run from a hand estimate of the excerpts' offered load (token counts against the classes' nominal prefill and decode rates: conversation about 70 % of the pair's capacity; coding about 40 % on average with bursts above capacity). Labelled: replayed traffic, simulated backend. Reproduce: `go run ./cmd/benchmark -config configs/replay.json -out benchmarks/results-replay`.

<!-- replay-summary:start -->
<!-- generated by scripts/plot_results.py from benchmarks/results; do not edit by hand -->
Mean J over 20 evaluation seeds (lower is better). ✓ / ✗ / = : better / worse / no difference than the family baseline (paired 95% interval of ΔJ). Oracles are references, not deployable. Simulated, assumed parameters.

**routing** (baseline `least_outstanding`)

| policy | replay-azure-conv | replay-azure-code |
|---|---|---|
| random | 41.929 ✗ | 61.697 ✗ |
| round_robin | 40.230 ✗ | 63.123 ✗ |
| least_outstanding (baseline) | 3.418 | 37.637 |
| power_of_two | 3.524 ✗ | 38.457 ✗ |
| latency_aware | 2.741 ✓ | 31.921 ✓ |
| queue_aware | 2.741 ✓ | 40.810 ✗ |
| queue_aware_corrected | 2.477 ✓ | 37.681 ✗ |
| capacity_weighted | 2.912 ✓ | 33.872 ✓ |
| oracle_jsq (oracle) | 2.265 ✓ | 31.260 ✓ |
<!-- replay-summary:end -->

Deterministic routers see the same trace on every seed, so their intervals are zero (only `random` and `power_of_two` vary) and any nonzero difference gets a ✓ or ✗ verdict; read the W/T/L ties in the full table for differences inside 1 %. The coding excerpt overloads the pair for every router (prefill bursts on the V100 drive P95 TPOT to about 570 ms); this is reported as measured, not reshaped.

## Prefix caching

`configs/prefix.json` enables the prefix-cache model (8 groups per replica) on four `h100-8b` replicas: prompts lognormal around 1.5k tokens, 32 prefix groups shared by 90 % of requests with 1024-token prefixes, Poisson at 90 % of the no-cache capacity. Routers: `round_robin`, `least_outstanding` (baseline), `power_of_two`, `queue_aware_corrected`, `prefix_affinity`; tuned on the tuning seeds with the same budget ([configs/tuned-prefix/](../configs/tuned-prefix/)), then evaluated on 20 seeds. Reproduce: `go run ./cmd/benchmark -config configs/prefix.json -tune -tuned configs/tuned-prefix` (committed) and `go run ./cmd/benchmark -config configs/prefix.json -tuned configs/tuned-prefix -out benchmarks/results-prefix`.

<!-- prefix-summary:start -->
<!-- generated by scripts/plot_results.py from benchmarks/results; do not edit by hand -->
Mean J over 20 evaluation seeds (lower is better). ✓ / ✗ / = : better / worse / no difference than the family baseline (paired 95% interval of ΔJ). Oracles are references, not deployable. Simulated, assumed parameters.

**routing** (baseline `least_outstanding`)

| policy | p-prefix |
|---|---|
| round_robin | 1.437 ✓ |
| least_outstanding (baseline) | 1.444 |
| power_of_two | 1.455 ✗ |
| queue_aware_corrected | 1.430 ✓ |
| prefix_affinity | 1.352 ✓ |
<!-- prefix-summary:end -->

## Decision cost

<!-- decision-cost:start -->
<!-- generated by scripts/plot_results.py from benchmarks/results/microbench.txt; do not edit by hand -->
Time per routing decision in ns (`go test -bench`, real CPU time of the policy code; Intel Core i9-14900KF, Windows 11, go1.27.1; minimum of 5 runs, measured while unrelated jobs shared the machine). `Route`: the policy alone on a prepared view. `Dispatch`: the full dispatcher path (eligible-view construction from the registry, snapshot lookup, policy, completion).

| benchmark | policy | 8 replicas | 64 replicas | 512 replicas |
|---|---|---|---|---|
| Route | random | 8 | 7 | 8 |
| Route | round_robin | 23 | 25 | 22 |
| Route | least_outstanding | 11 | 78 | 569 |
| Route | power_of_two | 21 | 21 | 24 |
| Route | latency_aware | 242 | 1,958 | 20,096 |
| Route | queue_aware | 98 | 690 | 7,145 |
| Route | queue_aware_corrected | 96 | 857 | 6,812 |
| Route | capacity_weighted | 298 | 2,147 | 18,851 |
| Route | oracle_jsq | 54 | 674 | 4,754 |
| Route | prefix_affinity | 279 | 3,159 | 23,898 |
| Dispatch | least_outstanding | 1,757 | 18,034 | 148,123 |
| Dispatch | power_of_two | 1,877 | 16,952 | 152,528 |
| Dispatch | queue_aware_corrected | 2,063 | 17,217 | 177,167 |
| Dispatch | latency_aware | 1,982 | 17,257 | 180,638 |
<!-- decision-cost:end -->

The full dispatcher path grows linearly with the replica count because every decision lists and sorts the model's replicas from the registry (about 0.29 µs per replica on this machine: 148 µs at 512 replicas). The policy itself costs at most about a sixth of that path; `prefix_affinity` (per-call maps and ring lookup), `latency_aware`, and `capacity_weighted` (floating-point work per replica) are the most expensive policies.

## TODO

- Calibrate the simulator against a real GPU or serving engine. Today it is a model with assumed parameters, and the calibration protocol in docs/simulator.md is written, not executed.
- Model network latency between router and replicas and more than one router (zero latency and one router so far), and evaluate prefix caching in the other scenarios (it is enabled only in its own scenario).
- Check the rankings under other objective weights. J is dominated by P95 TTFT ratios when a policy lets queues build; a policy can trade GPU-hours for tail latency, and J's weights decide that trade. Read the per-metric columns, not J alone.
- Widen the tuning search: it used 5 seeds and 12 configurations per policy, and a larger search could change close rankings.
- Add capacity scenarios: the capacity scenario is a single configuration (three models, one budget), so its conclusions are narrower than the routing ones.
