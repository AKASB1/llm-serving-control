// Command benchmark runs the simulated policy evaluation.
//
//	go run ./cmd/benchmark -tune     # tuning seeds only; writes configs/tuned/
//	go run ./cmd/benchmark           # full evaluation on the evaluation seeds
//	go run ./cmd/benchmark -quick    # small smoke subset (outputs/quick)
//
// Every result is simulated with assumed parameters (docs/simulator.md).
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/sim"
)

type options struct {
	config   string
	tunedDir string
	out      string
	traces   string
	quick    bool
	tune     bool
	workers  int
	hardware string
}

func main() {
	var o options
	flag.StringVar(&o.config, "config", "configs/experiment.json", "experiment configuration")
	flag.StringVar(&o.tunedDir, "tuned", "configs/tuned", "directory of the frozen tuned configuration")
	flag.StringVar(&o.out, "out", "", "result directory (default benchmarks/results, or outputs/quick with -quick)")
	flag.StringVar(&o.traces, "traces", "outputs/traces", "directory for generated traces")
	flag.BoolVar(&o.quick, "quick", false, "run the small smoke subset")
	flag.BoolVar(&o.tune, "tune", false, "tune on the tuning seeds and write the frozen configuration")
	flag.IntVar(&o.workers, "workers", runtime.NumCPU(), "parallel runs")
	flag.StringVar(&o.hardware, "hardware", "", "hardware description for the manifest (default: detected)")
	flag.Parse()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
	if err := run(o); err != nil {
		slog.Error("benchmark failed", "err", err)
		os.Exit(1)
	}
}

func run(o options) error {
	start := time.Now()
	exp, _, err := LoadExperiment(o.config)
	if err != nil {
		return err
	}
	env, cfgHash, err := newEnv(exp, o)
	if err != nil {
		return err
	}
	commit, dirty := gitCommit()
	if err := env.probeCapacity(o.workers); err != nil {
		return err
	}
	slog.Info("capacity probe", "req_per_s", fmtCapacity(env.capacity))
	if o.tune {
		t, log, err := env.tune(o.workers, commit, cfgHash)
		if err != nil {
			return err
		}
		if err := writeTuned(o.tunedDir, t, log); err != nil {
			return err
		}
		slog.Info("tuning done", "default_router", t.DefaultRouter, "default_scaler", t.DefaultScaler, "elapsed", time.Since(start).Round(time.Second))
		return nil
	}
	t, err := LoadTuned(filepath.Join(o.tunedDir, "tuned.json"))
	switch {
	case err == nil:
		env.tuned = t
	case o.quick && errors.Is(err, os.ErrNotExist):
		slog.Warn("no tuned configuration; quick run uses policy defaults")
		env.tuned = defaultTuned()
	default:
		return fmt.Errorf("tuned configuration missing (run -tune first): %w", err)
	}
	jobs, err := env.jobs(o.quick)
	if err != nil {
		return err
	}
	slog.Info("running", "jobs", len(jobs), "workers", o.workers, "quick", o.quick)
	rows, err := env.runAll(jobs, o.workers)
	if err != nil {
		return err
	}
	out := o.out
	if out == "" {
		out = "benchmarks/results"
		if o.quick {
			out = "outputs/quick"
		}
	}
	hw := o.hardware
	if hw == "" {
		hw = detectHardware()
	}
	man := manifest{
		Label: "simulated, assumed parameters", Mode: modeName(o.quick), Commit: commit, Dirty: dirty,
		ConfigSHA256: cfgHash, TunedCommit: env.tuned.Commit, TunedConfigSHA256: env.tuned.ConfigSHA256,
		GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, Hardware: hw,
		Seeds: seedsOf(exp, o.quick), Workers: o.workers, Runs: len(rows), Capacity: env.capacity,
		DefaultRouter: env.tuned.DefaultRouter, DefaultScaler: env.tuned.DefaultScaler,
	}
	if err := writeResults(out, exp, rows, man); err != nil {
		return err
	}
	slog.Info("done", "runs", len(rows), "out", out, "elapsed", time.Since(start).Round(time.Second))
	return nil
}

func newEnv(exp *Experiment, o options) (*Env, string, error) {
	classes, err := sim.LoadClasses(exp.ClassesFile)
	if err != nil {
		return nil, "", err
	}
	targets, err := exp.Targets()
	if err != nil {
		return nil, "", err
	}
	files := []string{o.config, exp.ClassesFile}
	for _, sc := range exp.Scenarios {
		if sc.TracePath != "" {
			files = append(files, sc.TracePath)
		}
	}
	if !o.tune {
		if p := filepath.Join(o.tunedDir, "tuned.json"); fileExists(p) {
			files = append(files, p)
		}
	}
	h, err := hashFiles(files...)
	if err != nil {
		return nil, "", err
	}
	env := &Env{exp: exp, classes: classes, targets: targets, traceDir: o.traces, durScale: 1, tuned: defaultTuned()}
	if o.quick {
		env.durScale = exp.Quick.DurationScale
	}
	return env, h, nil
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

func modeName(quick bool) string {
	if quick {
		return "quick"
	}
	return "full"
}

// jobs lists the evaluation runs (and the sensitivity runs in full mode).
func (env *Env) jobs(quick bool) ([]Job, error) {
	exp := env.exp
	var jobs []Job
	add := func(sc *Scenario, v Variant, seeds []uint64) error {
		f := exp.Family(sc.Family)
		for _, label := range f.Policies {
			ps, err := env.policies(sc, label)
			if err != nil {
				return err
			}
			for _, seed := range seeds {
				jobs = append(jobs, Job{Scenario: sc, Variant: v, Label: label, Seed: seed, Policy: ps})
			}
		}
		return nil
	}
	base := Variant{ID: "base"}
	if quick {
		for _, id := range exp.Quick.Scenarios {
			if err := add(exp.Scenario(id), base, exp.Seeds.Quick); err != nil {
				return nil, err
			}
		}
		return jobs, nil
	}
	for _, f := range exp.Families {
		for _, sc := range exp.ScenariosOf(f.Name) {
			if sc.Family == "sample" {
				continue // the synthetic sample trace belongs to the quick path
			}
			if err := add(sc, base, exp.Seeds.Evaluation); err != nil {
				return nil, err
			}
		}
	}
	for _, id := range exp.Sensitivity.Scenarios {
		for _, v := range exp.Sensitivity.Variants {
			if err := add(exp.Scenario(id), v, exp.Seeds.Sensitivity); err != nil {
				return nil, err
			}
		}
	}
	return jobs, nil
}

type manifest struct {
	Label             string              `json:"label"`
	Mode              string              `json:"mode"`
	Commit            string              `json:"commit"`
	Dirty             bool                `json:"dirty"`
	ConfigSHA256      string              `json:"config_sha256"`
	TunedCommit       string              `json:"tuned_commit"`
	TunedConfigSHA256 string              `json:"tuned_config_sha256"`
	GoVersion         string              `json:"go_version"`
	Platform          string              `json:"platform"`
	Hardware          string              `json:"hardware"`
	Seeds             map[string][]uint64 `json:"seeds"`
	Workers           int                 `json:"workers"`
	Runs              int                 `json:"runs"`
	DefaultRouter     string              `json:"default_router"`
	DefaultScaler     string              `json:"default_scaler"`
	Capacity          map[string]float64  `json:"capacity_req_per_s"`
}

func seedsOf(exp *Experiment, quick bool) map[string][]uint64 {
	if quick {
		return map[string][]uint64{"quick": exp.Seeds.Quick}
	}
	return map[string][]uint64{"tuning": exp.Seeds.Tuning, "evaluation": exp.Seeds.Evaluation, "sensitivity": exp.Seeds.Sensitivity}
}

func gitCommit() (string, bool) {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown", false
	}
	st, _ := exec.Command("git", "status", "--porcelain", "--untracked-files=no").Output()
	return strings.TrimSpace(string(out)), len(strings.TrimSpace(string(st))) > 0
}

// detectHardware returns "<CPU>, <n> logical CPUs, <RAM> GB RAM, <OS>" using
// only the standard library and OS tools (hw_windows.go, hw_other.go).
func detectHardware() string {
	cpu := cpuName()
	if cpu == "" {
		cpu = "unknown CPU"
	}
	hw := fmt.Sprintf("%s, %d logical CPUs", cpu, runtime.NumCPU())
	if gb := totalRAMGB(); gb > 0 {
		hw += fmt.Sprintf(", %.0f GB RAM", gb)
	}
	return hw + ", " + osName()
}

func fmtCapacity(c map[string]float64) string {
	b, _ := json.Marshal(c)
	return string(b)
}
