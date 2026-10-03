package metrics

import "time"

type bucket struct {
	start       time.Duration
	arrivals    int
	completions int
	outTokens   int64
	e2eSum      float64
	svcSum      float64
	inflightSum float64
	inflightN   int
	busy        float64 // Σ busy-time deltas of ready replicas (s)
	serving     float64 // Σ elapsed time of ready replicas between snapshots (s)
	readySum    float64
	readyN      int
}

// Window aggregates one model's signals over a sliding window in fixed time
// buckets, so memory is bounded by window/bucket regardless of traffic.
type Window struct {
	bucket  time.Duration
	buckets []bucket
	// Latest gauges (not windowed).
	queueDepth float64
	kvUsage    float64
}

// NewWindow returns a window of length w made of n buckets (n ≥ 1).
func NewWindow(w time.Duration, n int) *Window {
	if n < 1 {
		n = 1
	}
	return &Window{bucket: w / time.Duration(n), buckets: make([]bucket, n)}
}

// Length returns the window length.
func (w *Window) Length() time.Duration { return w.bucket * time.Duration(len(w.buckets)) }

func (w *Window) at(now time.Duration) *bucket {
	start := now - now%w.bucket
	b := &w.buckets[int((now/w.bucket)%time.Duration(len(w.buckets)))]
	if b.start != start {
		*b = bucket{start: start}
	}
	return b
}

// Arrival records a request reaching the router.
func (w *Window) Arrival(now time.Duration) { w.at(now).arrivals++ }

// Completion records a completed request: its end-to-end latency and its
// service time (E2E minus router and replica queueing: the time it held a
// slot in a running batch).
func (w *Window) Completion(now time.Duration, outputTokens int, e2e, service time.Duration) {
	b := w.at(now)
	b.completions++
	b.outTokens += int64(outputTokens)
	b.e2eSum += e2e.Seconds()
	b.svcSum += service.Seconds()
}

// Sample records the router-local in-flight total and the ready replica count.
func (w *Window) Sample(now time.Duration, inflight, ready int) {
	b := w.at(now)
	b.inflightSum += float64(inflight)
	b.inflightN++
	b.readySum += float64(ready)
	b.readyN++
}

// Busy records a ready replica's busy-time and elapsed-time deltas between two
// of its snapshots.
func (w *Window) Busy(now time.Duration, busy, elapsed time.Duration) {
	b := w.at(now)
	b.busy += busy.Seconds()
	b.serving += elapsed.Seconds()
}

// Gauges sets the latest queue depth and mean KV usage.
func (w *Window) Gauges(queueDepth, kvUsage float64) {
	w.queueDepth, w.kvUsage = queueDepth, kvUsage
}

// Signals summarises the buckets that overlap (now − window, now].
func (w *Window) Signals(now time.Duration) Signals {
	length := w.Length()
	s := Signals{At: now, Window: length, QueueDepth: w.queueDepth, KVUsage: w.kvUsage}
	from := now - length
	var arrivals, completions, inflightN, readyN int
	var out int64
	var e2e, svc, inflight, busy, serving, ready float64
	minStart := now
	for i := range w.buckets {
		b := &w.buckets[i]
		if b.start <= from || b.start > now {
			continue // outside the window (or a stale bucket from an earlier cycle)
		}
		minStart = min(minStart, b.start)
		arrivals += b.arrivals
		completions += b.completions
		out += b.outTokens
		e2e += b.e2eSum
		svc += b.svcSum
		inflight += b.inflightSum
		inflightN += b.inflightN
		busy += b.busy
		serving += b.serving
		ready += b.readySum
		readyN += b.readyN
	}
	// The covered span runs from the oldest bucket in the window to now.
	span := (now - minStart).Seconds()
	if span <= 0 {
		return s
	}
	s.ArrivalRate = float64(arrivals) / span
	s.CompletionRate = float64(completions) / span
	s.OutputTokenRate = float64(out) / span
	s.Completions = completions
	if completions > 0 {
		s.MeanE2E = e2e / float64(completions)
		s.MeanServiceTime = svc / float64(completions)
	}
	if inflightN > 0 {
		s.InFlight = inflight / float64(inflightN)
	}
	if readyN > 0 {
		s.ReadyReplicas = ready / float64(readyN)
	}
	if serving > 0 {
		s.Utilization = busy / serving
	}
	if s.ReadyReplicas > 0 && s.Utilization > 0 {
		s.ServiceRate = s.CompletionRate / (s.ReadyReplicas * s.Utilization)
	}
	return s
}
