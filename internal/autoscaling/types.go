package autoscaling

import (
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
)

// ModelView is the read-only input of one scaling decision for one model.
type ModelView struct {
	Now   time.Duration
	Model string
	// Replica counts by lifecycle state.
	Ready    int
	Starting int // provisioning + loading
	Draining int
	// Limits.
	Min, Max int
	// WarmUp is the expected time from a provisioning request to ready.
	WarmUp  time.Duration
	Signals metrics.Signals
	// SLOError is the model's windowed SLO error (see package slo); positive
	// means targets are being missed.
	SLOError float64
	// ViolationRate is the windowed fraction of finished requests that missed
	// their SLO.
	ViolationRate float64
}

// Current is the number of replicas that are or will become ready.
func (v ModelView) Current() int { return v.Ready + v.Starting }

// Decision is a scaler's desired replica count and the reason.
type Decision struct {
	Replicas int
	Reason   string
}

// Scaler returns the desired replica count for one model. Implementations are
// deterministic, keep their own state (cooldowns, windows, integrals) keyed by
// model, and read time only from the view.
type Scaler interface {
	Name() string
	Desired(v ModelView) Decision
}

// Clamp bounds n to [min, max].
func Clamp(n, minimum, maximum int) int {
	if n < minimum {
		return minimum
	}
	if maximum > 0 && n > maximum {
		return maximum
	}
	return n
}
