// Package controller holds the runtime shared by the simulator and the live
// service: the dispatcher (admission → routing → router queue → retries), the
// passive/active health tracker, the periodic control loop (SLO controller →
// scalers → allocator), and the policy factory. It never imports the
// simulator or net/http and reads time only from its callers.
package controller

import (
	"sort"
	"time"
)

// HealthConfig parameterises passive ejection and active probing.
type HealthConfig struct {
	// ConsecutiveErrors ejects a replica after this many errors in a row.
	ConsecutiveErrors int `json:"consecutive_errors"`
	// BaseBackoff is the first ejection's duration; it doubles per
	// consecutive ejection up to MaxBackoff.
	BaseBackoffS float64 `json:"base_backoff_s"`
	MaxBackoffS  float64 `json:"max_backoff_s"`
	// ProbeIntervalS and ProbeTimeoutS drive active probes (0 disables).
	ProbeIntervalS float64 `json:"probe_interval_s"`
	ProbeTimeoutS  float64 `json:"probe_timeout_s"`
}

// DefaultHealth returns the documented defaults.
func DefaultHealth() HealthConfig {
	return HealthConfig{ConsecutiveErrors: 3, BaseBackoffS: 5, MaxBackoffS: 60, ProbeIntervalS: 2, ProbeTimeoutS: 1}
}

func secs(s float64) time.Duration { return time.Duration(s*1e9 + 0.5) }

type replicaHealth struct {
	consecutive  int
	ejections    int // consecutive ejections without a success in between
	ejected      bool
	probation    bool
	ejectedUntil time.Duration
}

// Health tracks passive (request outcome) and active (probe) health. An
// ejected replica is reinstated on probation when its backoff expires; one
// failure on probation ejects it again with a doubled backoff, one success
// clears the history.
type Health struct {
	cfg HealthConfig
	m   map[string]*replicaHealth
	// Ejections counts every ejection, for reporting.
	Ejections int
}

// NewHealth returns a tracker.
func NewHealth(cfg HealthConfig) *Health {
	if cfg.ConsecutiveErrors < 1 {
		cfg.ConsecutiveErrors = 1
	}
	return &Health{cfg: cfg, m: map[string]*replicaHealth{}}
}

func (h *Health) get(id string) *replicaHealth {
	r := h.m[id]
	if r == nil {
		r = &replicaHealth{}
		h.m[id] = r
	}
	return r
}

// Healthy reports whether the replica may receive traffic.
func (h *Health) Healthy(id string) bool {
	r := h.m[id]
	return r == nil || !r.ejected
}

// Success records a successful request (or probe).
func (h *Health) Success(id string, _ time.Duration) {
	r := h.get(id)
	r.consecutive = 0
	if r.probation {
		r.probation = false
	}
	if !r.ejected {
		r.ejections = 0
	}
}

// Failure records a failed request; it returns true if the replica was
// ejected by this failure.
func (h *Health) Failure(id string, now time.Duration) bool {
	r := h.get(id)
	if r.ejected {
		return false
	}
	r.consecutive++
	if r.probation || r.consecutive >= h.cfg.ConsecutiveErrors {
		h.eject(r, now)
		return true
	}
	return false
}

// ProbeFailure records a failed active probe: the replica is ejected at once.
func (h *Health) ProbeFailure(id string, now time.Duration) bool {
	r := h.get(id)
	if r.ejected {
		return false
	}
	h.eject(r, now)
	return true
}

func (h *Health) eject(r *replicaHealth, now time.Duration) {
	r.ejections++
	b := h.cfg.BaseBackoffS
	for i := 1; i < r.ejections && b < h.cfg.MaxBackoffS; i++ {
		b *= 2
	}
	if b > h.cfg.MaxBackoffS {
		b = h.cfg.MaxBackoffS
	}
	r.ejected = true
	r.probation = false
	r.consecutive = 0
	r.ejectedUntil = now + secs(b)
	h.Ejections++
}

// Backoff returns the current ejection end time of an ejected replica.
func (h *Health) EjectedUntil(id string) (time.Duration, bool) {
	r := h.m[id]
	if r == nil || !r.ejected {
		return 0, false
	}
	return r.ejectedUntil, true
}

// Tick reinstates replicas whose backoff has expired (on probation) and
// returns their IDs, sorted.
func (h *Health) Tick(now time.Duration) []string {
	var out []string
	for id, r := range h.m {
		if r.ejected && now >= r.ejectedUntil {
			r.ejected = false
			r.probation = true
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// Forget drops a replica's history (it was removed).
func (h *Health) Forget(id string) { delete(h.m, id) }
