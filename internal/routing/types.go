package routing

import (
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

// Request is what the router knows about one request. The output length is
// not known to the control plane (only an optional client-side cap).
type Request struct {
	ID              string
	Model           string
	PromptTokens    int
	MaxOutputTokens int // 0 when the client sets no cap
	PrefixGroup     string
	SLOClass        string
}

// ReplicaView is the read-only, per-request view of one eligible replica.
type ReplicaView struct {
	ID    string
	Class registry.ClassInfo
	// InFlight is the router-local count of requests dispatched to this
	// replica and not yet finished. Always fresh.
	InFlight int
	// SentSinceScrape counts dispatches since the latest snapshot arrived.
	SentSinceScrape int
	// Scraped is the latest snapshot (stale by up to one scrape interval);
	// HasScrape is false until the first snapshot arrives.
	Scraped   metrics.ReplicaSnapshot
	HasScrape bool
	// SLOError is the replica's windowed SLO error from the SLO controller
	// (0 when unknown; positive means the replica is missing its targets).
	SLOError float64
	// Fresh is the replica's true current state. Only the simulator fills it,
	// and only for policies that declare themselves oracles.
	Fresh *metrics.ReplicaSnapshot
}

// View is the input of one routing or admission decision.
type View struct {
	Now   time.Duration
	Model string
	// Replicas holds the eligible replicas (ready, healthy, serving Model),
	// sorted by ID. It may be empty.
	Replicas []ReplicaView
	// RouterQueueLen is the number of requests of this model waiting in the
	// router queue.
	RouterQueueLen int
}

// Action is the outcome kind of a routing or admission decision.
type Action int

// Decision outcomes.
const (
	Dispatch Action = iota // send to Decision.ReplicaID (routing) / admit (admission)
	Queue                  // hold in the router queue
	Reject                 // refuse the request (a violation)
)

func (a Action) String() string {
	switch a {
	case Dispatch:
		return "dispatch"
	case Queue:
		return "queue"
	case Reject:
		return "reject"
	}
	return "unknown"
}

// Decision is a router's answer for one request.
type Decision struct {
	Action    Action
	ReplicaID string
	Reason    string
}

// Router picks a replica for one request. Implementations are deterministic
// given their RNG stream, never read the wall clock, and never import the
// simulator or net/http.
type Router interface {
	Name() string
	// Route is called with a non-empty v.Replicas; the dispatcher handles the
	// empty case by queueing.
	Route(req Request, v View) Decision
}

// OutcomeKind classifies feedback events.
type OutcomeKind int

// Feedback kinds.
const (
	FirstToken OutcomeKind = iota
	Completed
	Failed
)

// Outcome is feedback about a request previously dispatched by the router.
type Outcome struct {
	ReplicaID string
	Now       time.Duration
	Kind      OutcomeKind
	// Latency is dispatch→first token (FirstToken), dispatch→completion
	// (Completed), or dispatch→error (Failed).
	Latency      time.Duration
	OutputTokens int
}

// Observer is implemented by routers that learn from outcomes.
type Observer interface {
	Observe(o Outcome)
}

// Oracle is implemented by routers that need the true replica state
// (ReplicaView.Fresh). They are labelled oracles and cannot run live.
type Oracle interface {
	IsOracle() bool
}
