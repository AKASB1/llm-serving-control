package sim

import (
	"math"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/autoscaling"
	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

func warmups(cs map[string]Class) map[string]time.Duration {
	out := map[string]time.Duration{}
	for k, c := range cs {
		out[k] = c.Provision + c.Load
	}
	return out
}

func TestControlLoopScalesOutAndInWithoutLoss(t *testing.T) {
	cs := testClasses(t)
	cfg := loadgen.Config{Name: "diurnal", DurationS: 1200, Streams: []loadgen.Stream{{
		Model:   "chat-8b",
		Arrival: loadgen.Arrival{Kind: "diurnal", Rate: 30, Amplitude: 0.9, PeriodS: 1200, Phase: 0.75},
		Prompt:  loadgen.Dist{Kind: "lognormal", Median: 256, Sigma: 1.0, Min: 8, Max: 4096},
		Output:  loadgen.Dist{Kind: "lognormal", Median: 128, Sigma: 1.2, Min: 1, Max: 2048},
	}}}
	reqs, err := loadgen.Generate(cfg, 77)
	if err != nil {
		t.Fatal(err)
	}
	sc := baseConfig(1200, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 1})
	sc.BudgetGPUs = 8
	reg := registry.New()
	disp := controller.NewDispatcher(controller.DefaultDispatch(), reg, ClassInfos(cs), router(t, "least_outstanding"), nil, controller.NewHealth(controller.DefaultHealth()))
	scaler := autoscaling.NewTargetTracking(autoscaling.TargetTrackingParams{Target: 16, Tolerance: 0.1, StabilizationDownS: 60, MaxStepUp: 4, MaxStepDown: 1})
	loop, err := controller.NewLoop(controller.LoopConfig{Models: []controller.ModelConfig{{Model: "chat-8b", Class: "h100-8b", Min: 1, Max: 8, Slots: 32}}, WindowS: 30},
		reg, disp, ClassInfos(cs), warmups(cs), testTargets(t), map[string]autoscaling.Scaler{"chat-8b": scaler}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ih := &invariantHooks{t: t, gpus: 1}
	s, err := New(sc, cs, reg, disp, loop, testTargets(t), metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, reqs, ih.hooks())
	if err != nil {
		t.Fatal(err)
	}
	res := s.Run()
	st := countStates(res.Records)
	if st[metrics.Failed] != 0 || st[metrics.Rejected] != 0 {
		t.Fatalf("scaling lost requests: %v", st)
	}
	if res.Accounting.PeakReplicas < 3 {
		t.Fatalf("no scale-out: peak %d", res.Accounting.PeakReplicas)
	}
	ready := map[string]time.Duration{}
	terminated := 0
	for _, r := range res.Replicas {
		ready[r.ID] = r.ReadyAt
		if r.Final == registry.Terminated {
			terminated++
			if r.ProvisionAt > 0 && r.ReadyAt >= 0 && r.ReadyAt-r.ProvisionAt != 90*time.Second {
				t.Fatalf("warm-up of %s is %v, want 90 s", r.ID, r.ReadyAt-r.ProvisionAt)
			}
		}
	}
	if terminated == 0 {
		t.Fatal("no scale-in")
	}
	for _, r := range res.Records {
		if r.HasFirstToken && r.FirstToken < ready[r.Replica] {
			t.Fatalf("request %s served by %s before it was ready", r.ID, r.Replica)
		}
	}
	if math.Abs(ih.finish(res.Accounting.RunDuration)-res.Accounting.GPUSeconds) > 1e-6 {
		t.Fatal("GPU accounting does not match the integral under scaling")
	}
	if len(loop.Log) == 0 || len(res.ScaleLog) == 0 {
		t.Fatal("no decisions logged")
	}
	t.Logf("peak %d replicas, %d scale actions, states %v, GPU-h %.3f", res.Accounting.PeakReplicas, len(res.ScaleLog), st, res.Accounting.GPUSeconds/3600)
}
