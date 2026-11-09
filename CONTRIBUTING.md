# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Go 1.23 or later (the version in `go.mod`); the module uses the standard library only.
- Python 3 with matplotlib, only to redraw the figures and the generated tables.
- `go test -race` needs cgo and a C compiler (CI runs it on Linux).

## Checks

Run these before opening a pull request; CI runs the same steps (`.github/workflows/ci.yml`):

```bash
go build ./...
go vet ./...
gofmt -l .
go test ./...
go test -race ./...
go run ./cmd/benchmark -quick
```

`gofmt -l .` must print nothing.

## Results

Every result in the README and in `benchmarks/` comes from a committed result file. Policy results are simulated with assumed parameters; the decision-cost microbenchmarks measure the real CPU time of the policy code. To regenerate the main results:

```bash
go run ./cmd/benchmark -tune
go run ./cmd/benchmark
python scripts/plot_results.py
```

The replay, prefix, BurstGPT, and reweighting results have their own commands in [benchmarks/README.md](benchmarks/README.md).

A change to code or configuration that the benchmark reads (`cmd/benchmark`, the packages it imports, `configs/`) can change results. Such a change needs the documented re-run. Re-tune if a tuned quantity can change, and run the full evaluation once on the committed change. Then regenerate the figures and tables, and commit the result files with their manifests in the same pull request. The README's highlights and compact results table are written by hand: update them from `docs/figures/results_summary.md` and `docs/figures/results_table.md`. Tuning uses the tuning seeds only; the evaluation seeds are never used to choose a parameter.

## Figures and diagrams

`python scripts/plot_results.py` redraws the result figures in `docs/figures/` from the committed results. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs neither.

## Style

- Policy and simulator tests pass explicit virtual times; only the live-path tests (proxy, executor, and the mock backend in `internal/sim/live`) wait on the wall clock.
- Policies do not import the simulator or `net/http` and do not call `time.Now`.
- Documentation states what was measured, under which assumptions, and names the result file.
