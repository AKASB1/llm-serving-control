# LLM Serving Control Plane

A control plane for multi-model LLM serving. It manages routing, replicas, autoscaling, capacity, and latency targets while leaving token generation to serving backends such as vLLM or SGLang.

**Status:** implementation scaffold.

## Scope

- backend and model registry
- request routing
- replica health and load tracking
- scale-out / scale-in decisions
- capacity allocation across models
- SLO-aware routing
- GPU cost model
- admission and overload control
- traffic replay and load generation
- adapters for multiple serving backends

## Proposed stack

Go · gRPC/HTTP · Prometheus · PostgreSQL/Redis · vLLM/SGLang adapters · Kubernetes

## Control loop

```text
Requests
   │
   ▼
Router ───────────────► Serving Replicas
   │                         │
   ▼                         ▼
Metrics Collector ◄──── latency / load
   │
   ▼
SLO Controller
   │
   ├──► replica scaling
   ├──► capacity allocation
   └──► routing weights
```

## Repository layout

```text
cmd/control-plane/
internal/
  registry/
  routing/
  metrics/
  autoscaling/
  capacity/
  slo/
  backends/
  store/
loadgen/
deploy/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [vllm-project/vllm](https://github.com/vllm-project/vllm) — high-throughput LLM serving backend
- [sgl-project/sglang](https://github.com/sgl-project/sglang) — LLM serving runtime and scheduling
- [kserve/kserve](https://github.com/kserve/kserve) — model-serving control-plane patterns on Kubernetes
- [ray-project/ray](https://github.com/ray-project/ray) — Ray Serve and distributed serving patterns

## License

MIT
