package autoscaling

import (
	"fmt"
	"math"
	"time"
)

func secs(s float64) time.Duration { return time.Duration(s*1e9 + 0.5) }

// concurrency is the router-local in-flight per ready replica (time-averaged
// over the signal window).
func concurrency(v ModelView) float64 {
	return v.Signals.InFlight / math.Max(float64(v.Ready), 1)
}

// StaticParams configure `static`.
type StaticParams struct {
	Replicas int `json:"replicas"`
}

// Static always wants a fixed count.
type Static struct{ p StaticParams }

// NewStatic returns the `static` scaler.
func NewStatic(p StaticParams) *Static { return &Static{p: p} }

// Name implements Scaler.
func (s *Static) Name() string { return "static" }

// Desired implements Scaler.
func (s *Static) Desired(v ModelView) Decision {
	return Decision{Replicas: Clamp(s.p.Replicas, v.Min, v.Max), Reason: "static"}
}

// ThresholdParams configure `threshold_cooldown`.
type ThresholdParams struct {
	// Upper and Lower bound the in-flight requests per ready replica.
	Upper float64 `json:"upper"`
	Lower float64 `json:"lower"`
	Step  int     `json:"step"`
	// Cooldowns after any scale action before the next up/down action.
	CooldownUpS   float64 `json:"cooldown_up_s"`
	CooldownDownS float64 `json:"cooldown_down_s"`
}

// DefaultThreshold returns the defaults.
func DefaultThreshold() ThresholdParams {
	return ThresholdParams{Upper: 48, Lower: 16, Step: 1, CooldownUpS: 30, CooldownDownS: 120}
}

// ThresholdCooldown is the README baseline: add Step replicas when the
// concurrency per replica exceeds Upper, remove Step when it falls below
// Lower, and wait out a cooldown after every action.
type ThresholdCooldown struct {
	p    ThresholdParams
	last map[string]time.Duration
}

// NewThresholdCooldown returns the `threshold_cooldown` scaler.
func NewThresholdCooldown(p ThresholdParams) *ThresholdCooldown {
	if p.Step < 1 {
		p.Step = 1
	}
	return &ThresholdCooldown{p: p, last: map[string]time.Duration{}}
}

// Name implements Scaler.
func (s *ThresholdCooldown) Name() string { return "threshold_cooldown" }

// Desired implements Scaler.
func (s *ThresholdCooldown) Desired(v ModelView) Decision {
	cur := v.Current()
	c := concurrency(v)
	last, acted := s.last[v.Model]
	since := v.Now - last
	switch {
	case c > s.p.Upper && v.Starting == 0:
		if acted && since < secs(s.p.CooldownUpS) {
			return Decision{Replicas: cur, Reason: "up: cooling down"}
		}
		n := Clamp(cur+s.p.Step, v.Min, v.Max)
		if n != cur {
			s.last[v.Model] = v.Now
		}
		return Decision{Replicas: n, Reason: fmt.Sprintf("concurrency %.1f > %.1f", c, s.p.Upper)}
	case c < s.p.Lower:
		if acted && since < secs(s.p.CooldownDownS) {
			return Decision{Replicas: cur, Reason: "down: cooling down"}
		}
		n := Clamp(cur-s.p.Step, v.Min, v.Max)
		if n != cur {
			s.last[v.Model] = v.Now
		}
		return Decision{Replicas: n, Reason: fmt.Sprintf("concurrency %.1f < %.1f", c, s.p.Lower)}
	}
	return Decision{Replicas: Clamp(cur, v.Min, v.Max), Reason: "within thresholds"}
}

// TargetTrackingParams configure `target_tracking`.
type TargetTrackingParams struct {
	// Target is the desired in-flight requests per ready replica.
	Target    float64 `json:"target"`
	Tolerance float64 `json:"tolerance"`
	// Stabilization windows: scale-down follows the maximum recommendation
	// over the last StabilizationDownS, scale-up the minimum over the last
	// StabilizationUpS.
	StabilizationDownS float64 `json:"stabilization_down_s"`
	StabilizationUpS   float64 `json:"stabilization_up_s"`
	MaxStepUp          int     `json:"max_step_up"`
	MaxStepDown        int     `json:"max_step_down"`
}

// DefaultTargetTracking returns the defaults (modelled on the Kubernetes
// HPA: 10 % tolerance, 300 s scale-down stabilization).
func DefaultTargetTracking() TargetTrackingParams {
	return TargetTrackingParams{Target: 32, Tolerance: 0.1, StabilizationDownS: 300, StabilizationUpS: 0, MaxStepUp: 4, MaxStepDown: 1}
}

type rec struct {
	at time.Duration
	n  int
}

// TargetTracking is an HPA-like scaler: desired = ceil(ready · metric/target)
// unless the ratio is within the tolerance, then stabilised.
type TargetTracking struct {
	p    TargetTrackingParams
	hist map[string][]rec
}

// NewTargetTracking returns the `target_tracking` scaler.
func NewTargetTracking(p TargetTrackingParams) *TargetTracking {
	if p.Target <= 0 {
		p.Target = DefaultTargetTracking().Target
	}
	return &TargetTracking{p: p, hist: map[string][]rec{}}
}

// Name implements Scaler.
func (s *TargetTracking) Name() string { return "target_tracking" }

// Desired implements Scaler.
func (s *TargetTracking) Desired(v ModelView) Decision {
	cur := v.Current()
	ready := max(v.Ready, 1)
	ratio := concurrency(v) / s.p.Target
	raw := cur
	if math.Abs(ratio-1) > s.p.Tolerance {
		raw = int(math.Ceil(float64(ready) * ratio))
	}
	raw = Clamp(raw, v.Min, v.Max)
	h := append(s.hist[v.Model], rec{v.Now, raw})
	keep := secs(math.Max(s.p.StabilizationDownS, s.p.StabilizationUpS))
	for len(h) > 0 && h[0].at < v.Now-keep {
		h = h[1:]
	}
	s.hist[v.Model] = h
	want := raw
	if raw < cur {
		// Scale down only to the highest recommendation in the window.
		for _, r := range h {
			if r.at >= v.Now-secs(s.p.StabilizationDownS) && r.n > want {
				want = r.n
			}
		}
		want = min(want, cur)
		if s.p.MaxStepDown > 0 {
			want = max(want, cur-s.p.MaxStepDown)
		}
	} else if raw > cur {
		for _, r := range h {
			if r.at >= v.Now-secs(s.p.StabilizationUpS) && r.n < want {
				want = r.n
			}
		}
		want = max(want, cur)
		if s.p.MaxStepUp > 0 {
			want = min(want, cur+s.p.MaxStepUp)
		}
	}
	return Decision{Replicas: Clamp(want, v.Min, v.Max), Reason: fmt.Sprintf("concurrency/target %.2f, raw %d", ratio, raw)}
}

// SLOFeedbackParams configure `slo_feedback`.
type SLOFeedbackParams struct {
	Kp float64 `json:"kp"`
	Ki float64 `json:"ki"`
	// DeadBand: errors in [−DeadBand, +DeadBand] neither act nor integrate.
	DeadBand float64 `json:"dead_band"`
	// Hysteresis: scale up only above UpThreshold, down only below
	// −DownThreshold (asymmetric).
	UpThreshold   float64 `json:"up_threshold"`
	DownThreshold float64 `json:"down_threshold"`
	// IntegralMax clamps the integral (anti-windup).
	IntegralMax float64 `json:"integral_max"`
	// HoldS is the minimum time between two actions.
	HoldS       float64 `json:"hold_s"`
	MaxStepUp   int     `json:"max_step_up"`
	MaxStepDown int     `json:"max_step_down"`
}

// DefaultSLOFeedback returns the defaults.
func DefaultSLOFeedback() SLOFeedbackParams {
	return SLOFeedbackParams{Kp: 0.5, Ki: 0.02, DeadBand: 0.1, UpThreshold: 0.0, DownThreshold: 0.5, IntegralMax: 20, HoldS: 30, MaxStepUp: 4, MaxStepDown: 1}
}

type pi struct {
	integral float64
	lastAt   time.Duration
	lastAct  time.Duration
	acted    bool
	seen     bool
}

// SLOFeedback is error-driven: e = the SLO controller's error (P95 latency
// over target minus one). Outside the dead band, a PI term u = Kp·e + Ki·∫e
// scales the current count; hysteresis thresholds and a hold time keep it
// from oscillating.
type SLOFeedback struct {
	p     SLOFeedbackParams
	state map[string]*pi
}

// NewSLOFeedback returns the `slo_feedback` scaler.
func NewSLOFeedback(p SLOFeedbackParams) *SLOFeedback {
	return &SLOFeedback{p: p, state: map[string]*pi{}}
}

// Name implements Scaler.
func (s *SLOFeedback) Name() string { return "slo_feedback" }

// Desired implements Scaler.
func (s *SLOFeedback) Desired(v ModelView) Decision {
	st := s.state[v.Model]
	if st == nil {
		st = &pi{}
		s.state[v.Model] = st
	}
	cur := v.Current()
	e := v.SLOError
	dt := 0.0
	if st.seen {
		dt = (v.Now - st.lastAt).Seconds()
	}
	st.lastAt, st.seen = v.Now, true
	if math.Abs(e) <= s.p.DeadBand {
		return Decision{Replicas: Clamp(cur, v.Min, v.Max), Reason: fmt.Sprintf("error %.2f in dead band", e)}
	}
	st.integral = math.Max(-s.p.IntegralMax, math.Min(s.p.IntegralMax, st.integral+e*dt))
	u := s.p.Kp*e + s.p.Ki*st.integral
	if st.acted && v.Now-st.lastAct < secs(s.p.HoldS) {
		return Decision{Replicas: Clamp(cur, v.Min, v.Max), Reason: "holding"}
	}
	want := cur
	switch {
	case e > s.p.UpThreshold && u > 0 && v.Starting == 0:
		want = cur + max(1, int(math.Ceil(float64(max(cur, 1))*u)))
		if s.p.MaxStepUp > 0 {
			want = min(want, cur+s.p.MaxStepUp)
		}
	case e < -s.p.DownThreshold && u < 0:
		want = cur - max(1, int(math.Floor(float64(cur)*-u)))
		if s.p.MaxStepDown > 0 {
			want = max(want, cur-s.p.MaxStepDown)
		}
	}
	want = Clamp(want, v.Min, v.Max)
	if want != cur {
		st.acted, st.lastAct = true, v.Now
		if want < cur {
			st.integral = 0 // reset after a scale-in so it does not keep pushing down
		}
	}
	return Decision{Replicas: want, Reason: fmt.Sprintf("error %.2f, u %.2f", e, u)}
}

// PredictiveParams configure `predictive`.
type PredictiveParams struct {
	// Holt's linear smoothing of the arrival rate.
	Alpha float64 `json:"alpha"`
	Beta  float64 `json:"beta"`
	// Target is the desired in-flight requests per ready replica.
	Target float64 `json:"target"`
	// Headroom multiplies the forecast demand.
	Headroom float64 `json:"headroom"`
	// PriorE2ES is the request latency assumed before completions exist.
	PriorE2ES   float64 `json:"prior_e2e_s"`
	MaxStepDown int     `json:"max_step_down"`
}

// DefaultPredictive returns the defaults.
func DefaultPredictive() PredictiveParams {
	return PredictiveParams{Alpha: 0.5, Beta: 0.2, Target: 32, Headroom: 1.1, PriorE2ES: 5, MaxStepDown: 1}
}

type holt struct {
	level, trend float64
	at           time.Duration
	seen         bool
}

// Predictive forecasts the arrival rate one warm-up ahead (Holt's linear
// trend) and sizes the model by Little's law: replicas = forecast rate ×
// mean latency × headroom / target concurrency per replica.
type Predictive struct {
	p PredictiveParams
	h map[string]*holt
}

// NewPredictive returns the `predictive` scaler.
func NewPredictive(p PredictiveParams) *Predictive {
	if p.Target <= 0 {
		p.Target = DefaultPredictive().Target
	}
	return &Predictive{p: p, h: map[string]*holt{}}
}

// Name implements Scaler.
func (s *Predictive) Name() string { return "predictive" }

// Forecast returns the smoothed rate forecast lead seconds ahead.
func (s *Predictive) Forecast(model string, lead time.Duration) float64 {
	h := s.h[model]
	if h == nil {
		return 0
	}
	return math.Max(0, h.level+h.trend*lead.Seconds())
}

// Desired implements Scaler.
func (s *Predictive) Desired(v ModelView) Decision {
	h := s.h[v.Model]
	if h == nil {
		h = &holt{}
		s.h[v.Model] = h
	}
	x := v.Signals.ArrivalRate
	if !h.seen {
		h.level, h.trend, h.at, h.seen = x, 0, v.Now, true
	} else if dt := (v.Now - h.at).Seconds(); dt > 0 {
		prev := h.level
		h.level = s.p.Alpha*x + (1-s.p.Alpha)*(h.level+h.trend*dt)
		h.trend = s.p.Beta*(h.level-prev)/dt + (1-s.p.Beta)*h.trend
		h.at = v.Now
	}
	lat := v.Signals.MeanE2E
	if v.Signals.Completions == 0 || lat <= 0 {
		lat = s.p.PriorE2ES
	}
	f := s.Forecast(v.Model, v.WarmUp)
	want := int(math.Ceil(f * lat * s.p.Headroom / s.p.Target))
	cur := v.Current()
	if want < cur && s.p.MaxStepDown > 0 {
		want = max(want, cur-s.p.MaxStepDown)
	}
	return Decision{Replicas: Clamp(want, v.Min, v.Max), Reason: fmt.Sprintf("forecast %.1f req/s at +%v, latency %.2fs", f, v.WarmUp, lat)}
}
