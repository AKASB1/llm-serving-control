package capacity

import (
	"fmt"
	"math"
	"sort"
)

// floorMins grants every model its minimum and returns the remaining budget
// (negative when the minimums alone exceed it).
func floorMins(in Input) ([]Grant, int) {
	out := make([]Grant, len(in.Models))
	left := in.BudgetGPUs
	for i, m := range in.Models {
		out[i] = Grant{Model: m.Model, Replicas: m.Min, Reason: "minimum"}
		left -= m.Min * gpus(m)
	}
	return out, left
}

func gpus(m ModelDemand) int { return max(m.GPUsPerReplica, 1) }

// capOf is the most a model can be granted: its desired count within
// [Min, Max].
func capOf(m ModelDemand) int {
	c := max(m.Desired, m.Min)
	if m.Max > 0 && c > m.Max {
		c = m.Max
	}
	return c
}

// fits reports whether every model's capped desire fits the budget; if so the
// desired counts are granted unchanged.
func fits(in Input) ([]Grant, bool) {
	total := 0
	out := make([]Grant, len(in.Models))
	for i, m := range in.Models {
		c := capOf(m)
		total += c * gpus(m)
		out[i] = Grant{Model: m.Model, Replicas: c, Reason: "desired fits the budget"}
	}
	return out, total <= in.BudgetGPUs
}

// StaticPartitionParams configure `static_partition`.
type StaticPartitionParams struct {
	// Shares maps model → fraction of the GPU budget; models without a share
	// split the unassigned fraction equally.
	Shares map[string]float64 `json:"shares"`
}

// StaticPartition gives every model its minimum and then a fixed slice of the
// remaining budget; a model may use up to its slice (its desired count,
// capped) and never lends the rest.
type StaticPartition struct{ p StaticPartitionParams }

// NewStaticPartition returns the `static_partition` allocator.
func NewStaticPartition(p StaticPartitionParams) *StaticPartition { return &StaticPartition{p: p} }

// Name implements Allocator.
func (a *StaticPartition) Name() string { return "static_partition" }

// Allocate implements Allocator.
func (a *StaticPartition) Allocate(in Input) []Grant {
	out, left := floorMins(in)
	if left <= 0 {
		return out
	}
	assigned, unassigned := 0.0, 0
	for _, m := range in.Models {
		if s, ok := a.p.Shares[m.Model]; ok {
			assigned += s
		} else {
			unassigned++
		}
	}
	rest := 0.0
	if unassigned > 0 {
		rest = math.Max(0, 1-assigned) / float64(unassigned)
	}
	for i, m := range in.Models {
		share, ok := a.p.Shares[m.Model]
		if !ok {
			share = rest
		}
		slice := int(math.Floor(share*float64(left)+1e-9)) / gpus(m)
		extra := min(capOf(m)-m.Min, slice)
		if extra > 0 {
			out[i] = Grant{Model: m.Model, Replicas: m.Min + extra, Reason: fmt.Sprintf("minimum + partition of %d replicas", slice)}
		} else if capOf(m) > m.Min {
			out[i].Reason = fmt.Sprintf("minimum (partition of %d extra replicas)", slice)
		}
	}
	return out
}

// ProportionalDemand grants the desired counts when they fit; otherwise it
// gives every model its minimum and splits the remaining GPUs in proportion
// to the GPUs each model still wants (largest remainder, ties by model name).
type ProportionalDemand struct{}

// NewProportionalDemand returns the `proportional_demand` allocator.
func NewProportionalDemand() *ProportionalDemand { return &ProportionalDemand{} }

// Name implements Allocator.
func (a *ProportionalDemand) Name() string { return "proportional_demand" }

// Allocate implements Allocator.
func (a *ProportionalDemand) Allocate(in Input) []Grant {
	if out, ok := fits(in); ok {
		return out
	}
	out, left := floorMins(in)
	if left <= 0 {
		return out
	}
	want := make([]float64, len(in.Models))
	total := 0.0
	for i, m := range in.Models {
		want[i] = float64((capOf(m) - m.Min) * gpus(m))
		total += want[i]
	}
	if total == 0 {
		return out
	}
	type rem struct {
		i    int
		frac float64
	}
	var rems []rem
	used := 0
	for i, m := range in.Models {
		share := float64(left) * want[i] / total
		n := int(math.Floor(share / float64(gpus(m))))
		n = min(n, capOf(m)-m.Min)
		out[i].Replicas += n
		used += n * gpus(m)
		rems = append(rems, rem{i, share/float64(gpus(m)) - float64(n)})
		out[i].Reason = fmt.Sprintf("proportional share of %d spare GPUs", left)
	}
	sort.SliceStable(rems, func(x, y int) bool {
		if rems[x].frac != rems[y].frac {
			return rems[x].frac > rems[y].frac
		}
		return in.Models[rems[x].i].Model < in.Models[rems[y].i].Model
	})
	for _, r := range rems {
		m := in.Models[r.i]
		if out[r.i].Replicas < capOf(m) && used+gpus(m) <= left {
			out[r.i].Replicas++
			used += gpus(m)
		}
	}
	return out
}

// MarginalGainParams configure `marginal_gain`.
type MarginalGainParams struct {
	// WaitFraction: a request is counted as violating when its queueing delay
	// exceeds WaitFraction × the TTFT target.
	WaitFraction float64 `json:"wait_fraction"`
	// PriorServiceS is the mean service time assumed before measurements.
	PriorServiceS float64 `json:"prior_service_s"`
	// MinGain stops the greedy loop when the best gain (violations/s per GPU)
	// is below it.
	MinGain float64 `json:"min_gain"`
}

// DefaultMarginalGain returns the defaults.
func DefaultMarginalGain() MarginalGainParams {
	return MarginalGainParams{WaitFraction: 0.5, PriorServiceS: 5, MinGain: 0}
}

// MarginalGain grants the desired counts when they fit; otherwise it starts
// from the minimums and repeatedly adds replicas to the model with the largest
// estimated reduction in SLO violations per second per GPU, up to the desired
// counts. Because the estimate is not concave (below its stability threshold
// a model gains nothing from one more replica), each step looks ahead: for
// every model it takes the best average gain over adding k = 1..K replicas
// and adds that whole chunk. The estimate treats each model as an M/M/c queue with c =
// replicas × slots servers and service rate 1/mean service time per server:
// violations/s ≈ λ · P(wait > WaitFraction · TTFT target), with the Erlang-C
// waiting-time tail P(W > t) = C(c, λ/μ) · exp(−(cμ − λ)t) (1 when unstable).
type MarginalGain struct{ p MarginalGainParams }

// NewMarginalGain returns the `marginal_gain` allocator.
func NewMarginalGain(p MarginalGainParams) *MarginalGain {
	if p.PriorServiceS <= 0 {
		p.PriorServiceS = DefaultMarginalGain().PriorServiceS
	}
	return &MarginalGain{p: p}
}

// Name implements Allocator.
func (a *MarginalGain) Name() string { return "marginal_gain" }

// Violations estimates SLO violations per second of model m with n replicas.
func (a *MarginalGain) Violations(m ModelDemand, n int) float64 {
	if m.ArrivalRate <= 0 {
		return 0
	}
	if n <= 0 {
		return m.ArrivalRate
	}
	s := m.MeanServiceTime
	if s <= 0 {
		s = a.p.PriorServiceS
	}
	mu := 1 / s
	c := n * max(m.SlotsPerReplica, 1)
	lam := m.ArrivalRate
	if lam >= float64(c)*mu {
		return lam
	}
	t := a.p.WaitFraction * m.TTFTTarget.Seconds()
	return lam * ErlangC(c, lam/mu) * math.Exp(-(float64(c)*mu-lam)*t)
}

// ErlangC is the probability of waiting in an M/M/c queue with offered load
// a = λ/μ (a < c), computed through the stable Erlang-B recursion.
func ErlangC(c int, a float64) float64 {
	if a <= 0 {
		return 0
	}
	if a >= float64(c) {
		return 1
	}
	b := 1.0
	for k := 1; k <= c; k++ {
		b = a * b / (float64(k) + a*b)
	}
	rho := a / float64(c)
	return b / (1 - rho*(1-b))
}

// Allocate implements Allocator.
func (a *MarginalGain) Allocate(in Input) []Grant {
	if out, ok := fits(in); ok {
		return out
	}
	out, left := floorMins(in)
	for left > 0 {
		best, bestK, bestGain := -1, 0, 0.0
		for i, m := range in.Models {
			n := out[i].Replicas
			base := a.Violations(m, n)
			for k := 1; n+k <= capOf(m) && k*gpus(m) <= left; k++ {
				g := (base - a.Violations(m, n+k)) / float64(k*gpus(m))
				if best < 0 || g > bestGain {
					best, bestK, bestGain = i, k, g
				}
			}
		}
		if best < 0 || bestGain < a.p.MinGain {
			break
		}
		out[best].Replicas += bestK
		out[best].Reason = "greedy marginal gain"
		left -= bestK * gpus(in.Models[best])
	}
	return out
}
