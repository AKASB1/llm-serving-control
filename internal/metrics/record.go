package metrics

import (
	"math"
	"sort"
	"time"
)

// Terminal is a request's final state (docs/contracts.md §3).
type Terminal int

// Terminal states. Every request ends in exactly one of them.
const (
	Completed Terminal = iota
	Rejected
	Failed
	TimedOut
	Unfinished
)

var terminalNames = [...]string{"completed", "rejected", "failed", "timed_out", "unfinished"}

func (t Terminal) String() string {
	if t < 0 || int(t) >= len(terminalNames) {
		return "unknown"
	}
	return terminalNames[t]
}

// RequestRecord is the per-request outcome of a run.
type RequestRecord struct {
	ID           string
	Model        string
	SLOClass     string // resolved class (default applied)
	PromptTokens int
	OutputTokens int
	Arrival      time.Duration
	// FirstToken is valid when HasFirstToken; Completion when State == Completed.
	FirstToken    time.Duration
	HasFirstToken bool
	Completion    time.Duration
	RouterQueue   time.Duration
	ReplicaQueue  time.Duration
	Replica       string
	Retries       int
	Preemptions   int
	State         Terminal
	Met           bool
}

// TTFT is first-token time minus arrival.
func (r RequestRecord) TTFT() time.Duration { return r.FirstToken - r.Arrival }

// E2E is completion time minus arrival.
func (r RequestRecord) E2E() time.Duration { return r.Completion - r.Arrival }

// TPOT is (completion − first token)/(output − 1); 0 for single-token outputs.
func (r RequestRecord) TPOT() time.Duration {
	if r.OutputTokens < 2 {
		return 0
	}
	return (r.Completion - r.FirstToken) / time.Duration(r.OutputTokens-1)
}

// Percentile returns the nearest-rank p-quantile (0 < p ≤ 1) of xs, sorting
// xs in place. It returns NaN for an empty slice.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	if !sort.Float64sAreSorted(xs) {
		sort.Float64s(xs)
	}
	k := int(math.Ceil(p*float64(len(xs)))) - 1
	if k < 0 {
		k = 0
	}
	if k >= len(xs) {
		k = len(xs) - 1
	}
	return xs[k]
}

// Histogram is a fixed log-bucketed histogram of durations in seconds
// (10 buckets per decade from 1e-4 s to 1e3 s, plus under- and overflow).
type Histogram struct {
	Counts [72]int64
	N      int64
	Sum    float64
}

const histLo, histPerDecade = -4.0, 10.0

func histIndex(s float64) int {
	if s <= 0 {
		return 0
	}
	i := int(math.Floor((math.Log10(s)-histLo)*histPerDecade)) + 1
	if i < 0 {
		return 0
	}
	if i > 71 {
		return 71
	}
	return i
}

// Add records one observation.
func (h *Histogram) Add(d time.Duration) {
	s := d.Seconds()
	h.Counts[histIndex(s)]++
	h.N++
	h.Sum += s
}

// Quantile returns the upper bound of the bucket holding the p-quantile
// (buckets are 10 per decade, so the value can exceed the exact quantile by
// up to about 26 %); +Inf when it falls in the overflow bucket (≥ 1000 s).
func (h *Histogram) Quantile(p float64) float64 {
	if h.N == 0 {
		return math.NaN()
	}
	target := int64(math.Ceil(p * float64(h.N)))
	var c int64
	for i, n := range h.Counts {
		c += n
		if c >= target {
			switch i {
			case 0:
				return math.Pow(10, histLo)
			case len(h.Counts) - 1:
				return math.Inf(1)
			}
			return math.Pow(10, histLo+float64(i)/histPerDecade)
		}
	}
	return math.Inf(1)
}
