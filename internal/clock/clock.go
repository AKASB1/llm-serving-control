// Package clock provides the injected time source used by every policy,
// controller, and simulator component. Time is a time.Duration measured from
// the clock's epoch (the start of a trace in simulation, process start in the
// live service). Only this package reads the wall clock.
package clock

import (
	"sync"
	"time"
)

// Clock returns the current time as the duration since the clock's epoch.
type Clock interface {
	Now() time.Duration
}

// Real is a wall clock backed by Go's monotonic clock reading.
type Real struct {
	start time.Time
}

// NewReal returns a Real clock whose epoch is the moment of the call.
func NewReal() *Real { return &Real{start: time.Now()} }

// Now returns the monotonic time elapsed since the epoch.
func (r *Real) Now() time.Duration { return time.Since(r.start) }

// Fake is a manually advanced clock for tests and for the simulator's
// virtual time. It is safe for concurrent use.
type Fake struct {
	mu  sync.Mutex
	now time.Duration
}

// NewFake returns a Fake clock set to t.
func NewFake(t time.Duration) *Fake { return &Fake{now: t} }

// Now returns the current fake time.
func (f *Fake) Now() time.Duration {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

// Set moves the clock to t. Moving backwards panics: time never goes back.
func (f *Fake) Set(t time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if t < f.now {
		panic("clock: time moved backwards")
	}
	f.now = t
}

// Advance moves the clock forward by d (d must be non-negative).
func (f *Fake) Advance(d time.Duration) {
	if d < 0 {
		panic("clock: negative advance")
	}
	f.mu.Lock()
	f.now += d
	f.mu.Unlock()
}

// Seconds converts a trace timestamp in seconds to a Duration, rounding to
// the nearest nanosecond.
func Seconds(s float64) time.Duration {
	return time.Duration(s*1e9 + 0.5)
}
