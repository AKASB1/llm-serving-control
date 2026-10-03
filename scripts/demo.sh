#!/usr/bin/env bash
# End-to-end demo of the live data plane against mock backends (simulated
# timing, placeholder tokens). Run from the repository root:
#
#   bash scripts/demo.sh
#
# Steps: build; start two mock backends and the control plane (127.0.0.1:18080,
# 18100, 18101); stream one request; kill a backend and watch ejection and
# retries; send a burst that makes the scaler start a third backend through the
# process executor; shut everything down. Logs go to outputs/demo/.
set -euo pipefail
cd "$(dirname "$0")/.."

EXE=""
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) EXE=".exe" ;; esac
OUT=outputs/demo
CP=http://127.0.0.1:18080
mkdir -p outputs/bin "$OUT"
rm -f "$OUT"/*.log "$OUT"/*.out

echo "== build"
go build -o "outputs/bin/mock-backend$EXE" ./cmd/mock-backend
go build -o "outputs/bin/control-plane$EXE" ./cmd/control-plane

pids=()
cleanup() {
  curl -s -m 5 -X POST "$CP/admin/shutdown" >/dev/null 2>&1 || true
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

wait_http() { # url, attempts of 100 ms
  for _ in $(seq 1 "${2:-100}"); do
    if curl -sf -m 1 "$1" >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done
  echo "timed out waiting for $1" >&2
  return 1
}

echo "== start mock backends and the control plane"
"outputs/bin/mock-backend$EXE" -listen 127.0.0.1:18100 -model chat-8b -class h100-8b 2>"$OUT/mock-0.log" &
pids+=($!)
"outputs/bin/mock-backend$EXE" -listen 127.0.0.1:18101 -model chat-8b -class h100-8b 2>"$OUT/mock-1.log" &
pids+=($!)
wait_http http://127.0.0.1:18100/health
wait_http http://127.0.0.1:18101/health
"outputs/bin/control-plane$EXE" -config configs/live.json 2>"$OUT/control-plane.log" &
pids+=($!)
wait_http "$CP/health"

echo "== 1. one streamed request (server-sent events through the proxy)"
curl -s -m 30 -N -X POST "$CP/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"chat-8b","stream":true,"messages":[{"role":"user","content":"Hello"}],"mock_output_tokens":8}' >"$OUT/stream.out"
grep -c '^data: ' "$OUT/stream.out" | sed 's/^/data events received: /'
head -n 2 "$OUT/stream.out"
tail -n 2 "$OUT/stream.out"

echo "== 2. kill mock-0; requests keep succeeding (passive ejection, retries before the first byte)"
kill "${pids[0]}"
for i in 1 2 3 4 5 6; do
  curl -s -m 30 -o /dev/null -w "request $i: HTTP %{http_code} via %header{x-replica}\n" -X POST "$CP/v1/completions" \
    -H 'Content-Type: application/json' -d '{"model":"chat-8b","prompt":"Hello there","max_tokens":4}'
done
curl -s -m 5 "$CP/admin/replicas"
echo
curl -s -m 5 "$CP/metrics" | grep -E '^lsc_(retries_total|upstream_errors_total|replica_healthy)' || true

echo "== 3. a burst of 30 long streams; the threshold scaler adds a replica through the process executor"
burst=()
for i in $(seq 1 30); do
  curl -s -m 120 -N -X POST "$CP/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"chat-8b","stream":true,"mock_prompt_tokens":200,"mock_output_tokens":600}' >"$OUT/burst-$i.out" &
  burst+=($!)
done
scaled=no
for _ in $(seq 1 120); do
  if curl -s -m 2 "$CP/admin/replicas" | grep -q '"id":"h100-8b-p01"[^}]*"state":"ready"'; then scaled=yes; break; fi
  sleep 0.5
done
echo "scale-out observed: $scaled"
curl -s -m 5 "$CP/admin/replicas"
echo
for p in "${burst[@]}"; do wait "$p" || true; done
complete=$(grep -l 'data: \[DONE\]' "$OUT"/burst-*.out | wc -l | tr -d ' ')
echo "burst streams completed: $complete/30"

echo "== 4. control-plane metrics"
curl -s -m 5 "$CP/metrics" | grep -E '^lsc_requests_total' || true

echo "== shut down"
curl -s -m 5 -X POST "$CP/admin/shutdown" >/dev/null || true
sleep 1
[ "$scaled" = yes ] && [ "$complete" = 30 ] && echo "demo OK" || { echo "demo FAILED"; exit 1; }
