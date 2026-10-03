package main

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/autoscaling"
	"github.com/AKASB1/llm-serving-control/internal/capacity"
	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/sim"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/trace"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

// Env is everything a run needs besides its job.
type Env struct {
	exp      *Experiment
	classes  map[string]sim.Class
	targets  slo.Targets
	capacity map[string]float64
	tuned    *Tuned
	traceDir string
	durScale float64
}

// policySet is the full policy configuration of one run.
type policySet struct {
	Router    controller.PolicySpec
	Scalers   map[string]controller.PolicySpec
	Allocator *controller.PolicySpec
	Admission controller.PolicySpec
}

// Job is one simulation run.
type Job struct {
	Scenario *Scenario
	Variant  Variant
	Label    string
	Seed     uint64
	Policy   policySet
}

// Row is one run's result.
type Row struct {
	Scenario  string
	Variant   string
	Family    string
	Policy    string
	Seed      uint64
	TraceSHA  string
	Ejections int
	Summary   metrics.Summary
}

func scaleDur(d time.Duration, f float64) time.Duration { return time.Duration(float64(d)*f + 0.5) }

// tracePath returns where a scenario/variant/seed trace lives. Variants that
// do not change the traffic share the base trace.
func (env *Env) tracePath(sc *Scenario, v Variant, seed uint64) string {
	if sc.TracePath != "" {
		return sc.TracePath
	}
	name := sc.ID
	if v.OutputSigma > 0 {
		name += "~sigma" + strconv.FormatFloat(v.OutputSigma, 'g', -1, 64)
	}
	if env.durScale != 1 {
		name += "~x" + strconv.FormatFloat(env.durScale, 'g', -1, 64)
	}
	return filepath.Join(env.traceDir, name, fmt.Sprintf("seed-%d.csv", seed))
}

// ensureTrace generates the trace file if it is missing or stale.
func (env *Env) ensureTrace(sc *Scenario, v Variant, seed uint64) error {
	if sc.TracePath != "" {
		return nil
	}
	p := env.tracePath(sc, v, seed)
	cfg := env.exp.traceConfig(sc, v, env.capacity, env.durScale)
	want, err := loadgen.Manifest(cfg, seed)
	if err != nil {
		return err
	}
	if _, m, err := trace.LoadFiles(p); err == nil && string(m.Generator.Params) == string(want.Generator.Params) && m.Seed == seed {
		return nil
	}
	reqs, err := loadgen.Generate(cfg, seed)
	if err != nil {
		return fmt.Errorf("%s seed %d: %w", sc.ID, seed, err)
	}
	_, err = trace.WriteFiles(p, reqs, want)
	return err
}

func (env *Env) staticCount(sm ScenarioModel) int {
	n := 0
	for _, ic := range sm.Initial {
		n += ic.Count
	}
	return n
}

func staticSpec(n int) controller.PolicySpec {
	return controller.PolicySpec{Name: "static", Params: json.RawMessage(fmt.Sprintf(`{"replicas":%d}`, n))}
}

// peakCount is the static_peak oracle's count for one model: enough replicas
// for the known peak rate at the configured utilisation.
func (env *Env) peakCount(sc *Scenario, sm ScenarioModel) int {
	a := sm.Arrival.arrival(refRate(sm, sc.Mix, env.capacity))
	c := env.capacity[capKey(sm.Class, sc.Mix)]
	n := int(math.Ceil(peakRate(a) / (c * env.exp.PeakUtilization)))
	return max(sm.Min, min(n, sm.Max))
}

// policies returns the policy set of a family label in a scenario.
func (env *Env) policies(sc *Scenario, label string) (policySet, error) {
	ps := policySet{
		Router:    env.tuned.spec("routing", env.tuned.DefaultRouter),
		Scalers:   map[string]controller.PolicySpec{},
		Admission: controller.PolicySpec{Name: "none"},
	}
	for _, sm := range sc.Models {
		ps.Scalers[sm.Model] = staticSpec(env.staticCount(sm))
	}
	switch sc.Family {
	case "routing", "sample":
		ps.Router = env.tuned.spec("routing", label)
	case "scaling":
		for _, sm := range sc.Models {
			switch label {
			case "static_min":
				ps.Scalers[sm.Model] = staticSpec(sm.Min)
			case "static_tuned":
				ps.Scalers[sm.Model] = env.tuned.spec("scaling", "static")
			case "static_peak_oracle":
				ps.Scalers[sm.Model] = staticSpec(env.peakCount(sc, sm))
			default:
				ps.Scalers[sm.Model] = env.tuned.spec("scaling", label)
			}
		}
	case "capacity":
		for _, sm := range sc.Models {
			ps.Scalers[sm.Model] = env.tuned.spec("scaling", env.tuned.DefaultScaler)
		}
		a := env.tuned.spec("capacity", label)
		ps.Allocator = &a
	case "overload":
		ps.Admission = env.tuned.spec("overload", label)
	default:
		return ps, fmt.Errorf("unknown family %q", sc.Family)
	}
	return ps, nil
}

// runJob executes one simulation and summarises it.
func (env *Env) runJob(j Job) (Row, error) {
	sc := j.Scenario
	reqs, man, err := trace.LoadFiles(env.tracePath(sc, j.Variant, j.Seed))
	if err != nil {
		return Row{}, err
	}
	classes := classesFor(env.classes, j.Variant)
	infos := sim.ClassInfos(classes)
	reg := registry.New()
	router, err := controller.NewRouter(j.Policy.Router, j.Seed)
	if err != nil {
		return Row{}, err
	}
	adm, err := controller.NewAdmission(j.Policy.Admission, env.targets)
	if err != nil {
		return Row{}, err
	}
	disp := controller.NewDispatcher(env.exp.Dispatch, reg, infos, router, adm, controller.NewHealth(env.exp.Health))
	scalers := map[string]autoscaling.Scaler{}
	var models []controller.ModelConfig
	var initial []sim.InitialReplica
	for _, sm := range sc.Models {
		s, err := controller.NewScaler(j.Policy.Scalers[sm.Model])
		if err != nil {
			return Row{}, err
		}
		scalers[sm.Model] = s
		models = append(models, controller.ModelConfig{Model: sm.Model, Class: sm.Class, Min: sm.Min, Max: sm.Max, Slots: sm.Slots})
		for _, ic := range sm.Initial {
			initial = append(initial, sim.InitialReplica{Model: sm.Model, Class: ic.Class, Count: ic.Count})
		}
	}
	var alloc capacity.Allocator
	if j.Policy.Allocator != nil {
		if alloc, err = controller.NewAllocator(*j.Policy.Allocator); err != nil {
			return Row{}, err
		}
	}
	warm := map[string]time.Duration{}
	for k, c := range classes {
		warm[k] = c.Provision + c.Load
	}
	loop, err := controller.NewLoop(controller.LoopConfig{Models: models, WindowS: env.exp.WindowS, BudgetGPUs: sc.BudgetGPUs},
		reg, disp, infos, warm, env.targets, scalers, alloc)
	if err != nil {
		return Row{}, err
	}
	cfg := env.exp.Sim
	cfg.DurationS = man.DurationS
	if sc.TracePath != "" {
		cfg.DurationS = sc.DurationS
	}
	cfg.BudgetGPUs = float64(sc.BudgetGPUs)
	cfg.Initial = initial
	cfg.Crashes = nil
	for _, c := range sc.Crashes {
		cfg.Crashes = append(cfg.Crashes, sim.Crash{AtS: c.AtS * env.durScale, Replica: c.Replica})
	}
	if j.Variant.ScrapeIntervalS > 0 {
		cfg.ScrapeIntervalS = j.Variant.ScrapeIntervalS
	}
	cfg.PrefixTokens, cfg.PrefixCacheGroups = sc.PrefixTokens, sc.PrefixCacheGroups
	s, err := sim.New(cfg, classes, reg, disp, loop, env.targets, env.exp.Weights, reqs, sim.Hooks{})
	if err != nil {
		return Row{}, err
	}
	res := s.Run()
	return Row{
		Scenario: sc.ID, Variant: j.Variant.ID, Family: sc.Family, Policy: j.Label, Seed: j.Seed,
		TraceSHA: man.ContentSHA256[:12], Ejections: res.Ejections,
		Summary: metrics.Summarize(res.Records, env.targets, env.exp.Weights, res.Accounting),
	}, nil
}

// runAll generates the traces the jobs need and runs the jobs on `workers`
// goroutines. Rows come back in job order, so the output does not depend on
// scheduling.
func (env *Env) runAll(jobs []Job, workers int) ([]Row, error) {
	type key struct {
		sc   *Scenario
		v    Variant
		seed uint64
	}
	seen := map[string]bool{}
	var keys []key
	for _, j := range jobs {
		p := env.tracePath(j.Scenario, j.Variant, j.Seed)
		if !seen[p] {
			seen[p] = true
			keys = append(keys, key{j.Scenario, j.Variant, j.Seed})
		}
	}
	if err := parallel(len(keys), workers, func(i int) error {
		return env.ensureTrace(keys[i].sc, keys[i].v, keys[i].seed)
	}); err != nil {
		return nil, err
	}
	rows := make([]Row, len(jobs))
	err := parallel(len(jobs), workers, func(i int) error {
		r, err := env.runJob(jobs[i])
		if err != nil {
			return fmt.Errorf("%s/%s/%s seed %d: %w", jobs[i].Scenario.ID, jobs[i].Variant.ID, jobs[i].Label, jobs[i].Seed, err)
		}
		rows[i] = r
		return nil
	})
	return rows, err
}

// parallel runs fn(0..n-1) on up to `workers` goroutines and returns the
// first error.
func parallel(n, workers int, fn func(i int) error) error {
	if workers < 1 {
		workers = 1
	}
	next := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var first error
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				if err := fn(i); err != nil {
					mu.Lock()
					if first == nil {
						first = err
					}
					mu.Unlock()
				}
			}
		}()
	}
	for i := 0; i < n; i++ {
		mu.Lock()
		stop := first != nil
		mu.Unlock()
		if stop {
			break
		}
		next <- i
	}
	close(next)
	wg.Wait()
	return first
}

// probeCapacity measures the saturation throughput (completed requests/s) of
// one replica of each needed class under each needed mix: every request
// arrives at once, and completions are counted after a warm-up.
func (env *Env) probeCapacity(workers int) error {
	need := map[string][2]string{}
	for _, sc := range env.exp.Scenarios {
		if sc.TracePath != "" {
			continue
		}
		for _, sm := range sc.Models {
			need[capKey(sm.Class, sc.Mix)] = [2]string{sm.Class, sc.Mix}
			for _, ic := range sm.Initial {
				need[capKey(ic.Class, sc.Mix)] = [2]string{ic.Class, sc.Mix}
			}
		}
	}
	keys := make([]string, 0, len(need))
	for k := range need {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	vals := make([]float64, len(keys))
	err := parallel(len(keys), workers, func(i int) error {
		v, err := env.probeOne(need[keys[i]][0], need[keys[i]][1])
		vals[i] = v
		return err
	})
	if err != nil {
		return err
	}
	env.capacity = map[string]float64{}
	for i, k := range keys {
		env.capacity[k] = vals[i]
	}
	return nil
}

func (env *Env) probeOne(class, mixName string) (float64, error) {
	p := env.exp.Probe
	mix := env.exp.Mixes[mixName]
	cls := env.classes[class]
	model := cls.Spec.Model
	gen := loadgen.Config{Name: "probe", DurationS: 1, Streams: []loadgen.Stream{{
		Model: model, Arrival: loadgen.Arrival{Kind: "poisson", Rate: float64(p.Requests)},
		Prompt: mix.Prompt, Output: mix.Output, MaxContext: mix.MaxContext,
	}}}
	reqs, err := loadgen.Generate(gen, 0)
	if err != nil {
		return 0, err
	}
	reg := registry.New()
	router, _ := controller.NewRouter(controller.PolicySpec{Name: "least_outstanding"}, 0)
	dc := env.exp.Dispatch
	dc.RouterQueueTimeoutS = 0
	disp := controller.NewDispatcher(dc, reg, sim.ClassInfos(env.classes), router, nil, controller.NewHealth(env.exp.Health))
	cfg := env.exp.Sim
	cfg.DurationS, cfg.DrainHorizonS = p.DurationS, 0
	cfg.Initial = []sim.InitialReplica{{Model: model, Class: class, Count: 1}}
	cfg.BudgetGPUs = float64(cls.Spec.GPUs)
	s, err := sim.New(cfg, env.classes, reg, disp, nil, env.targets, env.exp.Weights, reqs, sim.Hooks{})
	if err != nil {
		return 0, err
	}
	res := s.Run()
	n := 0
	warm := time.Duration(p.WarmupS * 1e9)
	for _, r := range res.Records {
		if r.State == metrics.Completed && r.Completion >= warm {
			n++
		}
	}
	unfinished := 0
	for _, r := range res.Records {
		if r.State != metrics.Completed {
			unfinished++
		}
	}
	if unfinished == 0 {
		return 0, fmt.Errorf("capacity probe %s/%s ran dry: raise capacity_probe.requests", class, mixName)
	}
	return float64(n) / (p.DurationS - p.WarmupS), nil
}
