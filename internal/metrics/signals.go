package metrics

import "time"

// Signals are the windowed, per-model inputs of scalers and allocators. All
// rates are per second over the window that ends at At.
type Signals struct {
	At     time.Duration
	Window time.Duration
	// ArrivalRate counts requests reaching the router (admitted or not).
	ArrivalRate float64
	// CompletionRate counts requests completed.
	CompletionRate float64
	// OutputTokenRate counts output tokens of completed requests.
	OutputTokenRate float64
	// InFlight is the time-average of the router-local in-flight total.
	InFlight float64
	// QueueDepth is the latest scraped waiting count summed over ready
	// replicas plus the router queue length.
	QueueDepth float64
	// Utilization is the mean fraction of ready-replica time with a non-empty
	// running batch, from snapshot BusyTime deltas.
	Utilization float64
	// KVUsage is the mean KV occupancy over ready replicas (latest snapshots).
	KVUsage float64
	// ServiceRate is completions per busy-replica-second:
	// CompletionRate / (mean ready replicas × Utilization). 0 when unknown.
	ServiceRate float64
	// MeanE2E is the mean end-to-end latency of completions in the window (s).
	MeanE2E float64
	// MeanServiceTime is the mean of E2E minus router and replica queue time
	// of completions in the window (s): how long a request held a batch slot.
	MeanServiceTime float64
	// ReadyReplicas is the time-average number of ready replicas.
	ReadyReplicas float64
	// Completions is the number of completions in the window.
	Completions int
}
