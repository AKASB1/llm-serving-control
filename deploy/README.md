# Deployment

Everything here runs on one machine and binds to 127.0.0.1.

## What exists

- **Live control plane** (`cmd/control-plane`): OpenAI-compatible proxy on `127.0.0.1:18080` (configurable) in front of the replicas listed in its JSON configuration (`configs/live.json`), with the shared dispatcher, health tracker, admission, routing, and optionally the control loop with the process executor. Admin API and `/metrics` as described in [docs/contracts.md](../docs/contracts.md) §6. `POST /admin/shutdown` stops it and the replicas it started.
- **Mock backends** (`cmd/mock-backend`): one process per replica (ports from `18100`), the simulator's engine in real time, placeholder tokens, vLLM-named metrics.
- **Process executor**: starts additional mock backends for scale-out (ports from `18110` in the demo) and stops drained ones.
- **Replica store**: set `store_path` in the configuration to persist replicas registered through the admin API.
- **Demo**: `bash scripts/demo.sh` or `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1` builds both binaries into `outputs/bin/`, starts two mock backends and the control plane, streams a request, kills a backend (ejection and retries), triggers a scale-out, and shuts everything down. Logs go to `outputs/demo/`.

## Planned (not built, not tested)

- Kubernetes: an executor using the Deployment scale subresource and manifests under `deploy/k8s/`, validated with kubeconform.
- Running the proxy in front of real vLLM or SGLang servers: the metric adapters exist and are tested against fixtures written from those projects' sources, but have never been run against a live server.
- gRPC admin API, PostgreSQL/Redis stores, OpenTelemetry tracing.
