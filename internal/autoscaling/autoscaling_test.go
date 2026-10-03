package autoscaling

import (
	"math"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
)

const s = time.Second

// mv builds a view at time t with `ready` replicas and total in-flight.
func mv(t time.Duration, ready, starting int, inflight float64) ModelView {
	return ModelView{Now: t, Model: "m", Ready: ready, Starting: starting, Min: 1, Max: 10,
		Signals: metrics.Signals{InFlight: inflight}}
}

func TestStaticClamps(t *testing.T) {
	if d := NewStatic(StaticParams{Replicas: 20}).Desired(mv(0, 2, 0, 0)); d.Replicas != 10 {
		t.Fatalf("%+v", d)
	}
}

func TestThresholdCooldown(t *testing.T) {
	p := NewThresholdCooldown(ThresholdParams{Upper: 10, Lower: 2, Step: 1, CooldownUpS: 30, CooldownDownS: 60})
	if d := p.Desired(mv(0, 2, 0, 30)); d.Replicas != 3 {
		t.Fatalf("scale up: %+v", d)
	}
	// Still hot 10 s later but cooling down.
	if d := p.Desired(mv(10*s, 3, 0, 45)); d.Replicas != 3 {
		t.Fatalf("cooldown ignored: %+v", d)
	}
	// After the up cooldown it may act again, but not while a replica is starting.
	if d := p.Desired(mv(40*s, 3, 1, 45)); d.Replicas != 4 {
		t.Fatalf("starting replica counts toward current: %+v", d)
	}
	if d := p.Desired(mv(40*s, 3, 0, 45)); d.Replicas != 4 {
		t.Fatalf("second scale up: %+v", d)
	}
	// Cold now, but the down cooldown (60 s after the last action) holds.
	if d := p.Desired(mv(70*s, 4, 0, 1)); d.Replicas != 4 {
		t.Fatalf("down cooldown ignored: %+v", d)
	}
	if d := p.Desired(mv(101*s, 4, 0, 1)); d.Replicas != 3 {
		t.Fatalf("scale down: %+v", d)
	}
	// Between the thresholds: hold.
	if d := p.Desired(mv(300*s, 3, 0, 15)); d.Replicas != 3 {
		t.Fatalf("hold: %+v", d)
	}
}

func TestTargetTrackingToleranceAndStabilization(t *testing.T) {
	p := NewTargetTracking(TargetTrackingParams{Target: 10, Tolerance: 0.1, StabilizationDownS: 300, MaxStepUp: 10, MaxStepDown: 2})
	// 4 replicas at 10.5 per replica: within tolerance.
	if d := p.Desired(mv(0, 4, 0, 42)); d.Replicas != 4 {
		t.Fatalf("tolerance: %+v", d)
	}
	// 4 replicas at 20 per replica → 8.
	if d := p.Desired(mv(10*s, 4, 0, 80)); d.Replicas != 8 {
		t.Fatalf("scale up: %+v", d)
	}
	// Load drops at 60 s: raw 2, but the window remembers 8 → hold at 8.
	if d := p.Desired(mv(60*s, 8, 0, 20)); d.Replicas != 8 {
		t.Fatalf("stabilization ignored: %+v", d)
	}
	// After the 8 recommendation leaves the 300 s window: step down by at most 2.
	if d := p.Desired(mv(311*s, 8, 0, 20)); d.Replicas != 6 {
		t.Fatalf("stabilized scale down with step limit: %+v", d)
	}
}

func TestSLOFeedbackDeadBandHysteresisHold(t *testing.T) {
	p := NewSLOFeedback(SLOFeedbackParams{Kp: 0.5, Ki: 0, DeadBand: 0.1, UpThreshold: 0, DownThreshold: 0.5, IntegralMax: 10, HoldS: 30, MaxStepUp: 4, MaxStepDown: 1})
	v := mv(0, 4, 0, 0)
	v.SLOError = 0.05
	if d := p.Desired(v); d.Replicas != 4 {
		t.Fatalf("dead band: %+v", d)
	}
	v.Now, v.SLOError = 10*s, 1.0 // P95 twice the target: u = 0.5 → +2
	if d := p.Desired(v); d.Replicas != 6 {
		t.Fatalf("scale up: %+v", d)
	}
	v.Now, v.Ready = 20*s, 6
	if d := p.Desired(v); d.Replicas != 6 {
		t.Fatalf("hold time ignored: %+v", d)
	}
	// Moderately fast (error −0.3): inside the hysteresis gap → hold.
	v.Now, v.SLOError = 100*s, -0.3
	if d := p.Desired(v); d.Replicas != 6 {
		t.Fatalf("hysteresis: %+v", d)
	}
	v.Now, v.SLOError = 140*s, -0.8
	if d := p.Desired(v); d.Replicas != 5 {
		t.Fatalf("scale down by one: %+v", d)
	}
}

func TestSLOFeedbackIntegralAccumulates(t *testing.T) {
	p := NewSLOFeedback(SLOFeedbackParams{Kp: 0, Ki: 0.1, DeadBand: 0.05, UpThreshold: 0, DownThreshold: 0.5, IntegralMax: 100, HoldS: 0, MaxStepUp: 10})
	v := mv(0, 2, 0, 0)
	v.SLOError = 0.2
	p.Desired(v) // first call: dt = 0, integral 0 → u = 0 → hold
	v.Now = 10 * s
	if d := p.Desired(v); d.Replicas != 3 { // integral 2 → u = 0.2 → ceil(0.4) = 1
		t.Fatalf("integral action: %+v", d)
	}
}

func TestPredictiveLeadTimeIsWarmUp(t *testing.T) {
	p := NewPredictive(PredictiveParams{Alpha: 1, Beta: 1, Target: 10, Headroom: 1, PriorE2ES: 2})
	v := mv(0, 2, 0, 0)
	v.WarmUp = 60 * s
	v.Signals.ArrivalRate = 10
	p.Desired(v)
	v.Now, v.Signals.ArrivalRate = 10*s, 20 // trend 1 req/s per s
	d := p.Desired(v)
	if f := p.Forecast("m", 60*s); math.Abs(f-80) > 1e-9 {
		t.Fatalf("forecast at +warm-up = %v, want 80", f)
	}
	// 80 req/s × 2 s (prior latency) / 10 per replica = 16 → clamped to 10.
	if d.Replicas != 10 {
		t.Fatalf("%+v", d)
	}
	v.Now, v.Signals.ArrivalRate = 20*s, 20
	v.Signals.Completions, v.Signals.MeanE2E = 100, 0.5
	d = p.Desired(v)
	// level 20, trend 0 → 20 × 0.5 / 10 = 1, but at most one replica less per step.
	if d.Replicas != 1 {
		t.Fatalf("Little's law sizing: %+v", d)
	}
}
