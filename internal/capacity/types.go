package capacity

import "time"

// ModelDemand describes one model in an allocation round.
type ModelDemand struct {
	Model          string
	Desired        int // the scaler's desired count
	Current        int // ready + starting
	Min, Max       int
	GPUsPerReplica int
	// ArrivalRate (requests/s) and MeanServiceTime (s: E2E minus queueing,
	// the time a request holds a batch slot) feed the queueing approximation
	// of marginal_gain.
	ArrivalRate     float64
	MeanServiceTime float64
	// SlotsPerReplica is the effective number of concurrent requests a replica
	// serves without queueing (used as servers per replica).
	SlotsPerReplica int
	// TTFTTarget is the model's binding TTFT target.
	TTFTTarget time.Duration
	SLOError   float64
}

// Input is one allocation round.
type Input struct {
	Now        time.Duration
	BudgetGPUs int
	// Models is sorted by Model.
	Models []ModelDemand
}

// Grant is the allocator's answer for one model.
type Grant struct {
	Model    string
	Replicas int
	Reason   string
}

// Allocator turns desired counts into granted counts under the GPU budget.
// Contract: sum(Replicas × GPUsPerReplica) ≤ BudgetGPUs whenever the sum of
// minimums fits; every model gets at least Min (minimums are honoured first,
// even when they alone exceed the budget, and the reason says so); no model
// exceeds Max; the result is deterministic for a given input; output order
// follows Input.Models.
type Allocator interface {
	Name() string
	Allocate(in Input) []Grant
}
