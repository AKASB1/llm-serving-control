package metrics

import "time"

// ReplicaSnapshot is what a replica publishes every scrape interval. Gauges
// describe the instant At; counters are cumulative since the replica started.
// The control plane never sees replica state except through snapshots and its
// own router-local counters, so a snapshot can be up to one scrape interval old.
type ReplicaSnapshot struct {
	ReplicaID string
	At        time.Duration
	// Gauges.
	Running             int // sequences in the running batch
	Waiting             int // sequences in the replica's FCFS waiting queue
	WaitingPromptTokens int // prompt tokens not yet prefilled (waiting + partially prefilled)
	KVUsedTokens        int
	KVCapacityTokens    int
	// Counters.
	Preemptions       int64
	CompletedRequests int64
	GeneratedTokens   int64
	BusyTime          time.Duration // time with a non-empty running batch
	PrefixHits        int64         // prefix-cache hits
	PrefixQueries     int64         // prefix-cache lookups
	// RemainingDecodeTokens is oracle-only knowledge (it needs output
	// lengths): the simulator fills it only in the fresh state given to
	// oracle policies; scraped snapshots always carry 0.
	RemainingDecodeTokens int
}

// KVUsage is the KV-cache occupancy in [0, 1].
func (s ReplicaSnapshot) KVUsage() float64 {
	if s.KVCapacityTokens <= 0 {
		return 0
	}
	return float64(s.KVUsedTokens) / float64(s.KVCapacityTokens)
}
