package slo

import (
	"math"
	"sort"
	"time"
)

type obs struct {
	at        time.Duration
	ttftRatio float64
	tpotRatio float64 // NaN when undefined (output < 2 or not completed)
	completed bool
	met       bool
}

// ring is a bounded buffer of the most recent observations.
type ring struct {
	buf  []obs
	next int
	full bool
}

func (r *ring) add(o obs) {
	r.buf[r.next] = o
	r.next = (r.next + 1) % len(r.buf)
	if r.next == 0 {
		r.full = true
	}
}

func (r *ring) each(fn func(obs)) {
	n := r.next
	if r.full {
		n = len(r.buf)
	}
	for i := 0; i < n; i++ {
		fn(r.buf[i])
	}
}

// Error is the SLO controller's output for one model or replica.
type Error struct {
	// TTFTRatio and TPOTRatio are P95 of observed/target over completed
	// requests in the window (NaN when there are none).
	TTFTRatio float64
	TPOTRatio float64
	// ViolationRate is over every finished request in the window (rejected,
	// failed, and timed-out requests count as violations).
	ViolationRate float64
	// Value = max(TTFTRatio, TPOTRatio) − 1: positive when targets are being
	// missed. It is 1 when requests finished but none completed, 0 when the
	// window is empty.
	Value     float64
	N         int
	Completed int
}

// Controller turns observed latency against targets into windowed error
// signals per model and per replica. Buffers are bounded: each key keeps at
// most Capacity recent observations, and only those inside the window count.
type Controller struct {
	targets  Targets
	window   time.Duration
	capacity int
	models   map[string]*ring
	replicas map[string]*ring
	scratch  []float64
}

// NewController returns a controller with the given window and per-key
// buffer capacity.
func NewController(t Targets, window time.Duration, capacity int) *Controller {
	if capacity < 16 {
		capacity = 16
	}
	return &Controller{targets: t, window: window, capacity: capacity, models: map[string]*ring{}, replicas: map[string]*ring{}}
}

// Window returns the sliding-window length.
func (c *Controller) Window() time.Duration { return c.window }

// Observe records one finished request. replica may be empty (never
// dispatched).
func (c *Controller) Observe(now time.Duration, model, replica, class string, completed bool, ttft, tpot time.Duration, outputTokens int, met bool) {
	_, tg := c.targets.For(class)
	o := obs{at: now, completed: completed, met: met, ttftRatio: math.NaN(), tpotRatio: math.NaN()}
	if completed {
		o.ttftRatio = ttft.Seconds() / tg.TTFT.Seconds()
		if outputTokens >= 2 {
			o.tpotRatio = tpot.Seconds() / tg.TPOT.Seconds()
		}
	}
	c.key(c.models, model).add(o)
	if replica != "" {
		c.key(c.replicas, replica).add(o)
	}
}

func (c *Controller) key(m map[string]*ring, k string) *ring {
	r := m[k]
	if r == nil {
		r = &ring{buf: make([]obs, c.capacity)}
		m[k] = r
	}
	return r
}

// Model returns the windowed error of a model.
func (c *Controller) Model(model string, now time.Duration) Error {
	return c.eval(c.models[model], now)
}

// Replica returns the windowed error of a replica.
func (c *Controller) Replica(id string, now time.Duration) Error { return c.eval(c.replicas[id], now) }

// Forget drops a replica's buffer.
func (c *Controller) Forget(id string) { delete(c.replicas, id) }

func (c *Controller) eval(r *ring, now time.Duration) Error {
	e := Error{TTFTRatio: math.NaN(), TPOTRatio: math.NaN()}
	if r == nil {
		return e
	}
	from := now - c.window
	ttft := c.scratch[:0]
	var tpot []float64
	violations := 0
	r.each(func(o obs) {
		if o.at <= from || o.at > now {
			return
		}
		e.N++
		if !o.met {
			violations++
		}
		if o.completed {
			e.Completed++
			ttft = append(ttft, o.ttftRatio)
			if !math.IsNaN(o.tpotRatio) {
				tpot = append(tpot, o.tpotRatio)
			}
		}
	})
	c.scratch = ttft[:0]
	if e.N == 0 {
		return e
	}
	e.ViolationRate = float64(violations) / float64(e.N)
	if e.Completed == 0 {
		e.Value = 1
		return e
	}
	e.TTFTRatio = p95(ttft)
	worst := e.TTFTRatio
	if len(tpot) > 0 {
		e.TPOTRatio = p95(tpot)
		worst = math.Max(worst, e.TPOTRatio)
	}
	e.Value = worst - 1
	return e
}

// p95 is the nearest-rank 95th percentile (sorts xs).
func p95(xs []float64) float64 {
	sort.Float64s(xs)
	k := int(math.Ceil(0.95*float64(len(xs)))) - 1
	if k < 0 {
		k = 0
	}
	return xs[k]
}
