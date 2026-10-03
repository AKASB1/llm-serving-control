package sim

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/trace"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

func testClasses(t *testing.T) map[string]Class {
	t.Helper()
	cs, err := LoadClasses("../../configs/classes.json")
	if err != nil {
		t.Fatal(err)
	}
	return cs
}

func testTargets(t *testing.T) slo.Targets {
	ts, err := slo.NewTargets("interactive", map[string]slo.TargetJSON{
		"interactive": {TTFTs: 2, TPOTs: 0.1}, "batch": {TTFTs: 20, TPOTs: 0.25},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

func shortTrace(t *testing.T, model string, rate, dur float64, seed uint64) []trace.Request {
	t.Helper()
	cfg := loadgen.Config{Name: "t", DurationS: dur, Streams: []loadgen.Stream{{
		Model:   model,
		Arrival: loadgen.Arrival{Kind: "poisson", Rate: rate},
		Prompt:  loadgen.Dist{Kind: "lognormal", Median: 256, Sigma: 1.0, Min: 8, Max: 4096},
		Output:  loadgen.Dist{Kind: "lognormal", Median: 128, Sigma: 1.2, Min: 1, Max: 2048},
	}}}
	reqs, err := loadgen.Generate(cfg, seed)
	if err != nil {
		t.Fatal(err)
	}
	return reqs
}

type harness struct {
	sim  *Sim
	reg  *registry.Registry
	disp *controller.Dispatcher
}

func build(t *testing.T, cfg Config, classes map[string]Class, router routing.Router, ctl Control, reqs []trace.Request, hooks Hooks) harness {
	t.Helper()
	reg := registry.New()
	disp := controller.NewDispatcher(controller.DefaultDispatch(), reg, ClassInfos(classes), router, nil, controller.NewHealth(controller.DefaultHealth()))
	s, err := New(cfg, classes, reg, disp, ctl, testTargets(t), metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, reqs, hooks)
	if err != nil {
		t.Fatal(err)
	}
	return harness{sim: s, reg: reg, disp: disp}
}

func baseConfig(dur float64, initial ...InitialReplica) Config {
	c := DefaultConfig()
	c.DurationS = dur
	c.Initial = initial
	gpus := 0
	for _, i := range initial {
		gpus += i.Count
	}
	c.BudgetGPUs = float64(gpus)
	return c
}

// invariantHooks checks monotone time, KV ≤ capacity, and integrates the
// GPUs held by live replicas between events.
type invariantHooks struct {
	t        *testing.T
	last     time.Duration
	gpus     int
	integral float64
}

func (h *invariantHooks) hooks() Hooks {
	return Hooks{AfterEvent: func(s *Sim) {
		now := s.Now()
		if now < h.last {
			h.t.Fatalf("time went backwards: %v -> %v", h.last, now)
		}
		h.integral += float64(h.gpus) * (now - h.last).Seconds()
		h.last = now
		h.gpus = s.LiveGPUs()
		if id := s.KVViolation(); id != "" {
			h.t.Fatalf("KV occupancy above capacity on %s at %v", id, now)
		}
	}}
}

func (h *invariantHooks) finish(end time.Duration) float64 {
	return h.integral + float64(h.gpus)*(end-h.last).Seconds()
}

func countStates(recs []metrics.RequestRecord) map[metrics.Terminal]int {
	m := map[metrics.Terminal]int{}
	for _, r := range recs {
		m[r.State]++
	}
	return m
}

func TestRunInvariantsAndAccounting(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 20, 120, 1)
	ih := &invariantHooks{t: t}
	ih.gpus = 2
	h := build(t, baseConfig(120, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 2}), cs, routing.NewLeastOutstanding(), nil, reqs, ih.hooks())
	res := h.sim.Run()
	if len(res.Records) != len(reqs) {
		t.Fatalf("records %d != requests %d", len(res.Records), len(reqs))
	}
	st := countStates(res.Records)
	if st[metrics.Completed] != len(reqs) {
		t.Fatalf("states %v: at 20 req/s two replicas must complete everything", st)
	}
	for _, r := range res.Records {
		if r.TTFT() <= 0 || r.E2E() < r.TTFT() || r.ReplicaQueue < 0 || r.FirstToken < r.Arrival {
			t.Fatalf("inconsistent record %+v", r)
		}
	}
	gpuInt := ih.finish(res.Accounting.RunDuration)
	if math.Abs(gpuInt-res.Accounting.GPUSeconds) > 1e-6 {
		t.Fatalf("GPU-seconds %v != integral of live GPUs %v", res.Accounting.GPUSeconds, gpuInt)
	}
	if want := 2 * 180.0; math.Abs(res.Accounting.GPUSeconds-want) > 1e-6 {
		t.Fatalf("GPU-seconds %v, want %v", res.Accounting.GPUSeconds, want)
	}
	s := metrics.Summarize(res.Records, testTargets(t), metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, res.Accounting)
	if s.Utilization <= 0 || s.Utilization > 1 || s.KVOccupancy <= 0 || s.KVOccupancy > 1 {
		t.Fatalf("utilization %v kv %v", s.Utilization, s.KVOccupancy)
	}
}

func TestDeterministicRuns(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 40, 60, 3)
	run := func() string {
		h := build(t, baseConfig(60, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 3}), cs,
			routing.NewPowerOfTwo(rngFor(7)), nil, reqs, Hooks{})
		res := h.sim.Run()
		var b strings.Builder
		for _, r := range res.Records {
			fmt.Fprintf(&b, "%+v\n", r)
		}
		s := metrics.Summarize(res.Records, testTargets(t), metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, res.Accounting)
		b.WriteString(strings.Join(s.Values(), ","))
		return b.String()
	}
	if a, b := run(), run(); a != b {
		t.Fatal("same seed and configuration produced different results")
	}
}

func TestPreemptionUnderKVPressure(t *testing.T) {
	cs := testClasses(t)
	cfg := loadgen.Config{Name: "long", DurationS: 120, Streams: []loadgen.Stream{{
		Model:   "chat-8b",
		Arrival: loadgen.Arrival{Kind: "poisson", Rate: 3},
		Prompt:  loadgen.Dist{Kind: "lognormal", Median: 3000, Sigma: 0.5, Min: 100, Max: 12000},
		Output:  loadgen.Dist{Kind: "lognormal", Median: 800, Sigma: 0.8, Min: 1, Max: 4000},
	}}}
	reqs, _ := loadgen.Generate(cfg, 2)
	ih := &invariantHooks{t: t, gpus: 1}
	h := build(t, baseConfig(120, InitialReplica{Model: "chat-8b", Class: "v100-8b", Count: 1}), cs, routing.NewLeastOutstanding(), nil, reqs, ih.hooks())
	res := h.sim.Run()
	pre := 0
	for _, r := range res.Records {
		pre += r.Preemptions
	}
	if pre == 0 {
		t.Fatal("expected preemptions on the small-KV class")
	}
	st := countStates(res.Records)
	if st[metrics.Failed]+st[metrics.Rejected] != 0 {
		t.Fatalf("preemption must not fail requests: %v", st)
	}
}

// crashControl is a static control loop used to check crash bookkeeping.
func TestCrashFailsRetriesAndStopsBilling(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 25, 100, 4)
	cfg := baseConfig(100, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 3})
	cfg.Crashes = []Crash{{AtS: 50, Replica: "h100-8b-001"}}
	ih := &invariantHooks{t: t, gpus: 3}
	h := build(t, cfg, cs, routing.NewRoundRobin(), nil, reqs, ih.hooks())
	res := h.sim.Run()
	st := countStates(res.Records)
	retried, failedAfterFirst := 0, 0
	for _, r := range res.Records {
		if r.Retries > 0 {
			retried++
		}
		if r.State == metrics.Failed {
			if !r.HasFirstToken {
				t.Fatalf("failed before first token without exhausting retries: %+v", r)
			}
			failedAfterFirst++
		}
	}
	if retried == 0 || failedAfterFirst == 0 {
		t.Fatalf("crash should cause retries (%d) and failures after first token (%d)", retried, failedAfterFirst)
	}
	if st[metrics.Completed]+st[metrics.Failed] != len(reqs) {
		t.Fatalf("states %v", st)
	}
	var crashed ReplicaRecord
	for _, r := range res.Replicas {
		if r.ID == "h100-8b-001" {
			crashed = r
		}
	}
	if crashed.Final != registry.Failed || crashed.EndAt != 60*time.Second {
		t.Fatalf("crashed replica record %+v (executor detects after 10 s)", crashed)
	}
	if res.Ejections == 0 {
		t.Fatal("the router should have ejected the crashed replica")
	}
	// Serving time: two survivors serve the whole 160 s run, the crashed
	// replica only until the crash at 50 s (no replacement without a loop).
	if got := res.Accounting.ReadySeconds; math.Abs(got-370) > 1e-9 {
		t.Fatalf("serving seconds %v, want 370", got)
	}
	if math.Abs(ih.finish(res.Accounting.RunDuration)-res.Accounting.GPUSeconds) > 1e-6 {
		t.Fatal("GPU accounting does not match the integral after a crash")
	}
}

// drainAt drains one replica at a fixed time.
type drainAt struct {
	at   time.Duration
	id   string
	done bool
}

func (d *drainAt) ObserveArrival(time.Duration, string)                           {}
func (d *drainAt) ObserveFinish(time.Duration, metrics.RequestRecord)             {}
func (d *drainAt) ObserveSnapshot(time.Duration, string, metrics.ReplicaSnapshot) {}
func (d *drainAt) Step(now time.Duration) []controller.ScaleCommand {
	if d.done || now < d.at {
		return nil
	}
	d.done = true
	return []controller.ScaleCommand{{Kind: controller.Drain, ReplicaID: d.id}}
}

func TestDrainLosesNoRequest(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 20, 100, 5)
	ih := &invariantHooks{t: t, gpus: 2}
	h := build(t, baseConfig(100, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 2}), cs,
		routing.NewLeastOutstanding(), &drainAt{at: 30 * time.Second, id: "h100-8b-000"}, reqs, ih.hooks())
	res := h.sim.Run()
	st := countStates(res.Records)
	if st[metrics.Completed] != len(reqs) {
		t.Fatalf("scale-in lost requests: %v", st)
	}
	r0 := res.Replicas[0]
	if r0.Final != registry.Terminated || r0.EndAt <= 30*time.Second || r0.EndAt > 30*time.Second+3*time.Minute {
		t.Fatalf("drained replica %+v", r0)
	}
	for _, r := range res.Records {
		if r.Replica == "h100-8b-000" && r.Arrival > 30*time.Second {
			t.Fatalf("request %s sent to a draining replica", r.ID)
		}
	}
	if math.Abs(ih.finish(res.Accounting.RunDuration)-res.Accounting.GPUSeconds) > 1e-6 {
		t.Fatal("GPU accounting does not match the integral after a drain")
	}
}
