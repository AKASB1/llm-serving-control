package routing

import (
	"math"
	"math/rand/v2"
	"time"
)

// LatencyAwareParams configure `latency_aware`.
type LatencyAwareParams struct {
	// TauS is the EWMA time constant (s) for both smoothing and idle decay.
	TauS float64 `json:"tau_s"`
	// PriorS is the latency assumed for a replica without samples (s).
	PriorS float64 `json:"prior_s"`
	// Explore is the probability of a uniformly random pick.
	Explore float64 `json:"explore"`
}

// DefaultLatencyAware returns the defaults.
func DefaultLatencyAware() LatencyAwareParams {
	return LatencyAwareParams{TauS: 10, PriorS: 0.2, Explore: 0}
}

type ewma struct {
	value float64
	at    time.Duration
	seen  bool
}

// LatencyAwarePolicy is peak-EWMA routing: cost = smoothed dispatch-to-first-
// token latency × (in-flight + 1). A sample above the current value replaces
// it immediately (peak sensitivity); lower samples are blended with weight
// exp(−Δt/τ). With no new samples the value decays toward zero at the same
// rate, so a replica that stopped receiving traffic is retried and never
// starves.
type LatencyAwarePolicy struct {
	p   LatencyAwareParams
	rng *rand.Rand
	m   map[string]*ewma
}

// NewLatencyAware returns the `latency_aware` policy.
func NewLatencyAware(p LatencyAwareParams, r *rand.Rand) *LatencyAwarePolicy {
	if p.TauS <= 0 {
		p.TauS = DefaultLatencyAware().TauS
	}
	return &LatencyAwarePolicy{p: p, rng: r, m: map[string]*ewma{}}
}

// Name implements Router.
func (p *LatencyAwarePolicy) Name() string { return "latency_aware" }

func (p *LatencyAwarePolicy) decayed(e *ewma, now time.Duration) float64 {
	if !e.seen {
		return p.p.PriorS
	}
	dt := (now - e.at).Seconds()
	if dt <= 0 {
		return e.value
	}
	return e.value * math.Exp(-dt/p.p.TauS)
}

// Observe implements Observer (first-token latencies only).
func (p *LatencyAwarePolicy) Observe(o Outcome) {
	if o.Kind != FirstToken {
		return
	}
	e := p.m[o.ReplicaID]
	if e == nil {
		e = &ewma{}
		p.m[o.ReplicaID] = e
	}
	x := o.Latency.Seconds()
	if !e.seen || x > p.decayed(e, o.Now) {
		e.value = x
	} else {
		w := math.Exp(-(o.Now - e.at).Seconds() / p.p.TauS)
		e.value = w*e.value + (1-w)*x
	}
	e.at, e.seen = o.Now, true
}

// Route implements Router.
func (p *LatencyAwarePolicy) Route(_ Request, v View) Decision {
	if p.p.Explore > 0 && p.rng.Float64() < p.p.Explore {
		return Decision{Action: Dispatch, ReplicaID: v.Replicas[p.rng.IntN(len(v.Replicas))].ID, Reason: "explore"}
	}
	best, bestCost := 0, math.Inf(1)
	for i, r := range v.Replicas {
		e := p.m[r.ID]
		lat := p.p.PriorS
		if e != nil {
			lat = p.decayed(e, v.Now)
		}
		cost := lat * float64(r.InFlight+1)
		if cost < bestCost || (cost == bestCost && r.InFlight < v.Replicas[best].InFlight) {
			best, bestCost = i, cost
		}
	}
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[best].ID}
}

// QueueAwareParams configure `queue_aware` and `queue_aware_corrected`.
type QueueAwareParams struct {
	// Score = waiting + RunWeight·running/max_num_seqs + KVWeight·kv_usage
	// (+ SentWeight·dispatches since the snapshot, corrected variant only).
	RunWeight  float64 `json:"run_weight"`
	KVWeight   float64 `json:"kv_weight"`
	SentWeight float64 `json:"sent_weight"`
}

// DefaultQueueAware returns the defaults.
func DefaultQueueAware() QueueAwareParams {
	return QueueAwareParams{RunWeight: 1, KVWeight: 1, SentWeight: 1}
}

// QueueAwarePolicy ranks replicas by scraped queue depth and KV usage. It uses
// only the snapshot, so between two scrapes every request goes to the
// replica that looked emptiest (stale-snapshot herding, shown on purpose).
// The corrected variant adds the router-local dispatches since the snapshot
// as the documented mitigation. Replicas without a snapshot are scored by
// router-local in-flight.
type QueueAwarePolicy struct {
	p         QueueAwareParams
	corrected bool
}

// NewQueueAware returns `queue_aware` (corrected=false) or
// `queue_aware_corrected` (corrected=true).
func NewQueueAware(p QueueAwareParams, corrected bool) *QueueAwarePolicy {
	return &QueueAwarePolicy{p: p, corrected: corrected}
}

// Name implements Router.
func (p *QueueAwarePolicy) Name() string {
	if p.corrected {
		return "queue_aware_corrected"
	}
	return "queue_aware"
}

func (p *QueueAwarePolicy) score(r ReplicaView) float64 {
	if !r.HasScrape {
		return float64(r.InFlight)
	}
	maxSeqs := float64(max(r.Class.MaxNumSeqs, 1))
	s := float64(r.Scraped.Waiting) + p.p.RunWeight*float64(r.Scraped.Running)/maxSeqs + p.p.KVWeight*r.Scraped.KVUsage()
	if p.corrected {
		s += p.p.SentWeight * float64(r.SentSinceScrape)
	}
	return s
}

// Route implements Router.
func (p *QueueAwarePolicy) Route(_ Request, v View) Decision {
	best, bestScore := 0, math.Inf(1)
	for i, r := range v.Replicas {
		if s := p.score(r); s < bestScore {
			best, bestScore = i, s
		}
	}
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[best].ID}
}

// CapacityWeightedParams configure `capacity_weighted`.
type CapacityWeightedParams struct {
	// PriorWeight is the pseudo-count of the class prior in the rate blend.
	PriorWeight float64 `json:"prior_weight"`
	// EWMA weight of a new measured rate sample (0..1].
	SampleWeight float64 `json:"sample_weight"`
	// SLOGain scales weights down by 1/(1 + SLOGain·max(0, SLO error)).
	SLOGain float64 `json:"slo_gain"`
}

// DefaultCapacityWeighted returns the defaults.
func DefaultCapacityWeighted() CapacityWeightedParams {
	return CapacityWeightedParams{PriorWeight: 5, SampleWeight: 0.2, SLOGain: 0}
}

type rateEst struct {
	lastGen  int64
	lastBusy time.Duration
	lastAt   time.Duration
	rate     float64
	n        float64
	current  float64 // smooth weighted round-robin state
}

// CapacityWeightedPolicy is smooth weighted round-robin (each pick: add every
// weight to its running total, take the largest, subtract the sum). Weights
// are service-rate estimates: generated tokens per busy second from snapshot
// deltas, blended with the class prior (nominal decode tokens/s) by pseudo-
// counts, optionally scaled down for replicas missing their SLO.
type CapacityWeightedPolicy struct {
	p CapacityWeightedParams
	m map[string]*rateEst
}

// NewCapacityWeighted returns the `capacity_weighted` policy.
func NewCapacityWeighted(p CapacityWeightedParams) *CapacityWeightedPolicy {
	if p.SampleWeight <= 0 || p.SampleWeight > 1 {
		p.SampleWeight = DefaultCapacityWeighted().SampleWeight
	}
	return &CapacityWeightedPolicy{p: p, m: map[string]*rateEst{}}
}

// Name implements Router.
func (p *CapacityWeightedPolicy) Name() string { return "capacity_weighted" }

func (p *CapacityWeightedPolicy) weight(r ReplicaView) float64 {
	e := p.m[r.ID]
	if e == nil {
		e = &rateEst{lastAt: -1}
		p.m[r.ID] = e
	}
	if r.HasScrape && r.Scraped.At != e.lastAt {
		dBusy := (r.Scraped.BusyTime - e.lastBusy).Seconds()
		dGen := float64(r.Scraped.GeneratedTokens - e.lastGen)
		if e.lastAt >= 0 && dBusy > 0.05 && dGen > 0 {
			x := dGen / dBusy
			if e.n == 0 {
				e.rate = x
			} else {
				e.rate = (1-p.p.SampleWeight)*e.rate + p.p.SampleWeight*x
			}
			e.n++
		}
		e.lastGen, e.lastBusy, e.lastAt = r.Scraped.GeneratedTokens, r.Scraped.BusyTime, r.Scraped.At
	}
	prior := r.Class.DecodeTokensPerSec
	if prior <= 0 {
		prior = 1
	}
	w := prior
	if e.n > 0 {
		w = (prior*p.p.PriorWeight + e.rate*e.n) / (p.p.PriorWeight + e.n)
	}
	if p.p.SLOGain > 0 && r.SLOError > 0 {
		w /= 1 + p.p.SLOGain*r.SLOError
	}
	return w
}

// Route implements Router.
func (p *CapacityWeightedPolicy) Route(_ Request, v View) Decision {
	total := 0.0
	best := -1
	for i, r := range v.Replicas {
		w := p.weight(r)
		e := p.m[r.ID]
		e.current += w
		total += w
		if best < 0 || e.current > p.m[v.Replicas[best].ID].current {
			best = i
		}
	}
	p.m[v.Replicas[best].ID].current -= total
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[best].ID}
}

// OracleJSQPolicy joins the replica with the least remaining work according
// to the true current state (ReplicaView.Fresh): un-prefilled prompt tokens
// over the class prefill rate plus remaining output tokens over the class
// decode rate. It knows output lengths and has no staleness: an oracle,
// labelled as such, usable only in the simulator.
type OracleJSQPolicy struct{}

// NewOracleJSQ returns the `oracle_jsq` policy.
func NewOracleJSQ() *OracleJSQPolicy { return &OracleJSQPolicy{} }

// Name implements Router.
func (p *OracleJSQPolicy) Name() string { return "oracle_jsq" }

// IsOracle implements Oracle.
func (p *OracleJSQPolicy) IsOracle() bool { return true }

// Route implements Router.
func (p *OracleJSQPolicy) Route(req Request, v View) Decision {
	best, bestCost := 0, math.Inf(1)
	for i, r := range v.Replicas {
		var cost float64
		if r.Fresh != nil {
			pf := math.Max(r.Class.PrefillTokensPerSec, 1)
			dc := math.Max(r.Class.DecodeTokensPerSec, 1)
			cost = float64(r.Fresh.WaitingPromptTokens+req.PromptTokens)/pf + float64(r.Fresh.RemainingDecodeTokens)/dc
		} else {
			cost = math.Inf(1)
		}
		if cost < bestCost {
			best, bestCost = i, cost
		}
	}
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[best].ID}
}
