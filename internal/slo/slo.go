// Package slo defines latency targets, the per-request attainment rule, and the
// SLO controller that turns observed latency into error signals.
package slo

import (
	"fmt"
	"sort"
	"time"
)

// Target is the latency target of one SLO class. E2E == 0 means no E2E target.
type Target struct {
	TTFT time.Duration
	TPOT time.Duration
	E2E  time.Duration
}

// TargetJSON is the configuration form of a Target (seconds).
type TargetJSON struct {
	TTFTs float64 `json:"ttft_s"`
	TPOTs float64 `json:"tpot_s"`
	E2Es  float64 `json:"e2e_s,omitempty"`
}

// Target converts the JSON form.
func (t TargetJSON) Target() Target {
	sec := func(s float64) time.Duration { return time.Duration(s*1e9 + 0.5) }
	return Target{TTFT: sec(t.TTFTs), TPOT: sec(t.TPOTs), E2E: sec(t.E2Es)}
}

// Targets maps SLO class names to targets. Requests with an empty or unknown
// class use the Default class.
type Targets struct {
	Default string
	Classes map[string]Target
}

// NewTargets builds Targets from JSON forms and validates them.
func NewTargets(def string, classes map[string]TargetJSON) (Targets, error) {
	t := Targets{Default: def, Classes: map[string]Target{}}
	for name, c := range classes {
		tg := c.Target()
		if tg.TTFT <= 0 || tg.TPOT <= 0 || tg.E2E < 0 {
			return Targets{}, fmt.Errorf("slo class %q: TTFT and TPOT targets must be positive", name)
		}
		t.Classes[name] = tg
	}
	if _, ok := t.Classes[def]; !ok {
		return Targets{}, fmt.Errorf("slo: default class %q not defined", def)
	}
	return t, nil
}

// For returns the class name actually used and its target.
func (t Targets) For(class string) (string, Target) {
	if tg, ok := t.Classes[class]; ok {
		return class, tg
	}
	return t.Default, t.Classes[t.Default]
}

// ClassNames returns the class names sorted.
func (t Targets) ClassNames() []string {
	out := make([]string, 0, len(t.Classes))
	for k := range t.Classes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Met is the attainment rule of contract v1: a request meets its SLO only if
// it completed, TTFT ≤ target, TPOT ≤ target (only defined for
// outputTokens ≥ 2; vacuously true otherwise), and E2E ≤ target when an E2E
// target is set. Rejected, failed, timed-out, and unfinished requests never
// meet their SLO.
func Met(completed bool, ttft, tpot, e2e time.Duration, outputTokens int, t Target) bool {
	if !completed {
		return false
	}
	if ttft > t.TTFT {
		return false
	}
	if outputTokens >= 2 && tpot > t.TPOT {
		return false
	}
	if t.E2E > 0 && e2e > t.E2E {
		return false
	}
	return true
}
