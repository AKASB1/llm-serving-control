package admission

import (
	"fmt"
	"math"

	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/slo"
)

// None admits everything (requests queue in the router only when no replica
// is eligible).
type None struct{}

// Name implements Policy.
func (None) Name() string { return "none" }

// Admit implements Policy.
func (None) Admit(routing.Request, routing.View) Decision { return Decision{Action: routing.Dispatch} }

// QueueCapParams configure `queue_cap`.
type QueueCapParams struct {
	// MaxInFlightPerReplica caps router-local concurrency per replica.
	MaxInFlightPerReplica int `json:"max_inflight_per_replica"`
	// MaxQueue bounds the router queue (per model); beyond it requests are
	// rejected.
	MaxQueue int `json:"max_queue"`
}

// DefaultQueueCap returns the defaults.
func DefaultQueueCap() QueueCapParams {
	return QueueCapParams{MaxInFlightPerReplica: 96, MaxQueue: 256}
}

// QueueCap holds requests in the router queue while every eligible replica is
// at its concurrency cap and rejects them when that queue is full.
type QueueCap struct{ p QueueCapParams }

// NewQueueCap returns the `queue_cap` policy.
func NewQueueCap(p QueueCapParams) *QueueCap { return &QueueCap{p: p} }

// Name implements Policy.
func (q *QueueCap) Name() string { return "queue_cap" }

// Admit implements Policy.
func (q *QueueCap) Admit(_ routing.Request, v routing.View) Decision {
	free := false
	for _, r := range v.Replicas {
		if r.InFlight < q.p.MaxInFlightPerReplica {
			free = true
			break
		}
	}
	if free {
		return Decision{Action: routing.Dispatch}
	}
	if v.RouterQueueLen >= q.p.MaxQueue {
		return Decision{Action: routing.Reject, Reason: fmt.Sprintf("router queue full (%d)", v.RouterQueueLen)}
	}
	return Decision{Action: routing.Queue, Reason: "all replicas at the concurrency cap"}
}

// PredictedTTFTParams configure `predicted_ttft_shed`.
type PredictedTTFTParams struct {
	// Factors maps SLO class → shed threshold as a multiple of the default
	// class's TTFT target. A lower factor sheds that class earlier; the
	// defaults shed batch before interactive.
	Factors map[string]float64 `json:"factors"`
	// DefaultFactor applies to classes missing from Factors.
	DefaultFactor float64 `json:"default_factor"`
	// PrefillShare is the fraction of an iteration's time available to
	// prefill under load (decodes take the rest); it scales the class
	// prefill rate.
	PrefillShare float64 `json:"prefill_share"`
}

// DefaultPredictedTTFT returns the defaults.
func DefaultPredictedTTFT() PredictedTTFTParams {
	return PredictedTTFTParams{Factors: map[string]float64{"interactive": 1.0, "batch": 0.5}, DefaultFactor: 1.0, PrefillShare: 0.5}
}

// PredictedTTFT sheds a request when its predicted TTFT on the best replica
// exceeds its class threshold. Prediction: prompt tokens ahead of it (the
// scraped un-prefilled tokens plus, for dispatches since the scrape, its own
// prompt size per dispatch) plus its own prompt, over the class prefill rate
// scaled by PrefillShare. It never queues except when no replica is eligible.
type PredictedTTFT struct {
	p    PredictedTTFTParams
	base float64 // default class TTFT target (s)
}

// NewPredictedTTFT returns the `predicted_ttft_shed` policy.
func NewPredictedTTFT(p PredictedTTFTParams, targets slo.Targets) *PredictedTTFT {
	if p.PrefillShare <= 0 || p.PrefillShare > 1 {
		p.PrefillShare = DefaultPredictedTTFT().PrefillShare
	}
	if p.DefaultFactor <= 0 {
		p.DefaultFactor = 1
	}
	_, tg := targets.For("")
	return &PredictedTTFT{p: p, base: tg.TTFT.Seconds()}
}

// Name implements Policy.
func (a *PredictedTTFT) Name() string { return "predicted_ttft_shed" }

// Predict returns the predicted TTFT (s) of req on replica r.
func (a *PredictedTTFT) Predict(req routing.Request, r routing.ReplicaView) float64 {
	rate := math.Max(r.Class.PrefillTokensPerSec*a.p.PrefillShare, 1)
	ahead := 0
	if r.HasScrape {
		ahead = r.Scraped.WaitingPromptTokens + r.SentSinceScrape*req.PromptTokens
	} else {
		ahead = r.InFlight * req.PromptTokens
	}
	return float64(ahead+req.PromptTokens) / rate
}

// Admit implements Policy.
func (a *PredictedTTFT) Admit(req routing.Request, v routing.View) Decision {
	if len(v.Replicas) == 0 {
		return Decision{Action: routing.Queue, Reason: "no eligible replica"}
	}
	best := math.Inf(1)
	for _, r := range v.Replicas {
		best = math.Min(best, a.Predict(req, r))
	}
	f, ok := a.p.Factors[req.SLOClass]
	if !ok {
		f = a.p.DefaultFactor
	}
	if limit := f * a.base; best > limit {
		return Decision{Action: routing.Reject, Reason: fmt.Sprintf("predicted TTFT %.2fs > %.2fs", best, limit)}
	}
	return Decision{Action: routing.Dispatch}
}
