package routing

import "math/rand/v2"

// RandomPolicy picks a uniformly random eligible replica.
type RandomPolicy struct{ rng *rand.Rand }

// NewRandom returns the `random` policy.
func NewRandom(r *rand.Rand) *RandomPolicy { return &RandomPolicy{rng: r} }

// Name implements Router.
func (p *RandomPolicy) Name() string { return "random" }

// Route implements Router.
func (p *RandomPolicy) Route(_ Request, v View) Decision {
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[p.rng.IntN(len(v.Replicas))].ID}
}

// RoundRobinPolicy cycles over the eligible replicas (per model) in ID order.
type RoundRobinPolicy struct{ next map[string]int }

// NewRoundRobin returns the `round_robin` policy.
func NewRoundRobin() *RoundRobinPolicy { return &RoundRobinPolicy{next: map[string]int{}} }

// Name implements Router.
func (p *RoundRobinPolicy) Name() string { return "round_robin" }

// Route implements Router.
func (p *RoundRobinPolicy) Route(_ Request, v View) Decision {
	i := p.next[v.Model] % len(v.Replicas)
	p.next[v.Model] = i + 1
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[i].ID}
}

// LeastOutstandingPolicy picks the replica with the fewest router-local
// in-flight requests; ties go to the lowest ID.
type LeastOutstandingPolicy struct{}

// NewLeastOutstanding returns the `least_outstanding` policy.
func NewLeastOutstanding() *LeastOutstandingPolicy { return &LeastOutstandingPolicy{} }

// Name implements Router.
func (p *LeastOutstandingPolicy) Name() string { return "least_outstanding" }

// Route implements Router.
func (p *LeastOutstandingPolicy) Route(_ Request, v View) Decision {
	best := 0
	for i := 1; i < len(v.Replicas); i++ {
		if v.Replicas[i].InFlight < v.Replicas[best].InFlight {
			best = i
		}
	}
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[best].ID}
}

// PowerOfTwoPolicy samples two distinct eligible replicas uniformly at random
// and picks the one with fewer in-flight requests (ties: the first sampled).
type PowerOfTwoPolicy struct{ rng *rand.Rand }

// NewPowerOfTwo returns the `power_of_two` policy.
func NewPowerOfTwo(r *rand.Rand) *PowerOfTwoPolicy { return &PowerOfTwoPolicy{rng: r} }

// Name implements Router.
func (p *PowerOfTwoPolicy) Name() string { return "power_of_two" }

// Route implements Router.
func (p *PowerOfTwoPolicy) Route(_ Request, v View) Decision {
	n := len(v.Replicas)
	if n == 1 {
		return Decision{Action: Dispatch, ReplicaID: v.Replicas[0].ID}
	}
	a := p.rng.IntN(n)
	b := p.rng.IntN(n - 1)
	if b >= a {
		b++
	}
	if v.Replicas[b].InFlight < v.Replicas[a].InFlight {
		a = b
	}
	return Decision{Action: Dispatch, ReplicaID: v.Replicas[a].ID}
}
