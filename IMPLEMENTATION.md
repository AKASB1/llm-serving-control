# Implementation plan

## Control-plane boundary

This project does not implement token generation. Serving backends expose replicas and metrics; the control plane decides where requests go and how much capacity each model receives.

## Core entities

### Backend

A serving backend exposes health, model inventory, replica endpoints, queue depth, and latency metrics.

### Replica

Track model, GPU assignment, concurrency, queue length, recent TTFT, recent token throughput, and health.

### Routing policy

Start with:

1. round robin
2. least outstanding requests
3. latency-aware routing
4. queue-aware routing
5. weighted routing by measured capacity

### Autoscaler

Use queue depth, arrival rate, service rate, and SLO error. Add hysteresis and cooldowns before more advanced policies.

## Metrics

At minimum:

- time to first token
- inter-token latency
- end-to-end latency
- request queue time
- tokens/s
- replica utilization
- SLO violation rate
- GPU-hours per model

## Delivery order

1. backend registry
2. static routing
3. metrics ingestion
4. health-aware routing
5. traffic replay/load generator
6. replica autoscaler
7. multi-model capacity allocation
8. SLO controller
9. Kubernetes adapter
10. policy benchmarks
