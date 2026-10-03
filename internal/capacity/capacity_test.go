package capacity

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/rng"
)

func allocators() []Allocator {
	return []Allocator{
		NewStaticPartition(StaticPartitionParams{Shares: map[string]float64{"m0": 0.5}}),
		NewProportionalDemand(),
		NewMarginalGain(DefaultMarginalGain()),
	}
}

func randomInput(seed uint64) Input {
	r := rng.Stream(seed, "capacity-property")
	n := 1 + r.IntN(5)
	in := Input{BudgetGPUs: r.IntN(40)}
	for i := 0; i < n; i++ {
		g := []int{1, 1, 2, 4}[r.IntN(4)]
		mn := r.IntN(3)
		mx := mn + r.IntN(10)
		in.Models = append(in.Models, ModelDemand{
			Model: fmt.Sprintf("m%d", i), Desired: r.IntN(14), Current: r.IntN(10), Min: mn, Max: mx,
			GPUsPerReplica: g, ArrivalRate: r.Float64() * 50, MeanServiceTime: 0.5 + r.Float64()*10,
			SlotsPerReplica: 1 + r.IntN(64), TTFTTarget: 2 * time.Second,
		})
	}
	sort.Slice(in.Models, func(a, b int) bool { return in.Models[a].Model < in.Models[b].Model })
	return in
}

// Property test on 2000 random inputs per allocator: the budget is never
// exceeded when the minimums fit, minimums are honoured, no model exceeds
// its maximum, output order follows the input, and the result is
// deterministic.
func TestAllocatorProperties(t *testing.T) {
	for _, a := range allocators() {
		for seed := uint64(0); seed < 2000; seed++ {
			in := randomInput(seed)
			g1 := a.Allocate(in)
			g2 := a.Allocate(in)
			minGPUs, used := 0, 0
			if len(g1) != len(in.Models) {
				t.Fatalf("%s seed %d: %d grants for %d models", a.Name(), seed, len(g1), len(in.Models))
			}
			for i, m := range in.Models {
				g := g1[i]
				if g.Model != m.Model {
					t.Fatalf("%s seed %d: order", a.Name(), seed)
				}
				if g != g2[i] {
					t.Fatalf("%s seed %d: not deterministic", a.Name(), seed)
				}
				if g.Replicas < m.Min {
					t.Fatalf("%s seed %d: %s below minimum (%d < %d)", a.Name(), seed, m.Model, g.Replicas, m.Min)
				}
				if m.Max > 0 && g.Replicas > max(m.Max, m.Min) {
					t.Fatalf("%s seed %d: %s above maximum", a.Name(), seed, m.Model)
				}
				minGPUs += m.Min * gpus(m)
				used += g.Replicas * gpus(m)
			}
			if minGPUs <= in.BudgetGPUs && used > in.BudgetGPUs {
				t.Fatalf("%s seed %d: budget exceeded (%d > %d)", a.Name(), seed, used, in.BudgetGPUs)
			}
			if minGPUs > in.BudgetGPUs && used != minGPUs {
				t.Fatalf("%s seed %d: minimums exceed the budget, grant must be exactly the minimums", a.Name(), seed)
			}
		}
	}
}

func three(budget int) Input {
	return Input{BudgetGPUs: budget, Models: []ModelDemand{
		{Model: "a", Desired: 6, Min: 1, Max: 10, GPUsPerReplica: 1, ArrivalRate: 60, MeanServiceTime: 2, SlotsPerReplica: 16, TTFTTarget: 2 * time.Second},
		{Model: "b", Desired: 2, Min: 1, Max: 10, GPUsPerReplica: 1, ArrivalRate: 1, MeanServiceTime: 2, SlotsPerReplica: 16, TTFTTarget: 2 * time.Second},
		{Model: "c", Desired: 2, Min: 1, Max: 4, GPUsPerReplica: 4, ArrivalRate: 5, MeanServiceTime: 4, SlotsPerReplica: 16, TTFTTarget: 2 * time.Second},
	}}
}

func grants(gs []Grant) []int {
	out := make([]int, len(gs))
	for i, g := range gs {
		out[i] = g.Replicas
	}
	return out
}

func TestDesiredFitsIsGrantedByDemandAllocators(t *testing.T) {
	for _, a := range []Allocator{NewProportionalDemand(), NewMarginalGain(DefaultMarginalGain())} {
		if g := grants(a.Allocate(three(16))); g[0] != 6 || g[1] != 2 || g[2] != 2 {
			t.Fatalf("%s: %v", a.Name(), g)
		}
	}
}

func TestStaticPartitionNeverLends(t *testing.T) {
	a := NewStaticPartition(StaticPartitionParams{Shares: map[string]float64{"a": 0.25, "b": 0.25, "c": 0.5}})
	// Budget 16: minimums use 6, 10 spare: slices a=2, b=2 replicas, c=5 GPUs
	// (1 replica). a wants 6 but gets 1+2 although b leaves a GPU unused.
	if g := grants(a.Allocate(three(16))); g[0] != 3 || g[1] != 2 || g[2] != 2 {
		t.Fatalf("%v", g)
	}
}

func TestProportionalSplitUnderPressure(t *testing.T) {
	// Budget 10: minimums use 1+1+4 = 6, 4 spare GPUs, wanted a=5, b=1, c=4 GPUs.
	// Proportional floors give a 2; remainders (b 0.4, c 0.4 → b by name) give
	// b 1; c's 4-GPU replica does not fit; the last GPU goes to a.
	g := grants(NewProportionalDemand().Allocate(three(10)))
	if g[0] != 4 || g[1] != 2 || g[2] != 1 {
		t.Fatalf("%v", g)
	}
}

func TestMarginalGainFavoursTheSavableModel(t *testing.T) {
	a := NewMarginalGain(DefaultMarginalGain())
	in := three(10)
	in.Models[0].ArrivalRate = 10 // a: 8 req/s per replica → 2 replicas make it stable
	g := grants(a.Allocate(in))
	if g[0] < 2 || g[2] != 1 {
		t.Fatalf("%v", g)
	}
	m := in.Models[0]
	if a.Violations(m, 1) <= a.Violations(m, 3) || a.Violations(m, 3) < 0 {
		t.Fatal("violations must fall with replicas")
	}
}

func TestMarginalGainLooksAheadPastTheStabilityThreshold(t *testing.T) {
	// x needs 3 replicas (8 req/s each) before any violation is removed; y
	// gains a little from one more replica. With 3 spare GPUs a one-step
	// greedy would feed y; the lookahead gives x its two replicas.
	in := Input{BudgetGPUs: 5, Models: []ModelDemand{
		{Model: "x", Desired: 3, Min: 1, Max: 5, GPUsPerReplica: 1, ArrivalRate: 20, MeanServiceTime: 1, SlotsPerReplica: 8, TTFTTarget: 2 * time.Second},
		{Model: "y", Desired: 3, Min: 1, Max: 5, GPUsPerReplica: 1, ArrivalRate: 7.5, MeanServiceTime: 1, SlotsPerReplica: 8, TTFTTarget: 2 * time.Second},
	}}
	g := grants(NewMarginalGain(DefaultMarginalGain()).Allocate(in))
	if g[0] != 3 {
		t.Fatalf("%v", g)
	}
}

func TestErlangC(t *testing.T) {
	// M/M/1: P(wait) = ρ.
	if c := ErlangC(1, 0.7); math.Abs(c-0.7) > 1e-12 {
		t.Fatalf("C(1, 0.7) = %v", c)
	}
	// Known value: C(2, 1) = 1/3.
	if c := ErlangC(2, 1); math.Abs(c-1.0/3) > 1e-12 {
		t.Fatalf("C(2, 1) = %v", c)
	}
	if ErlangC(3, 3) != 1 || ErlangC(3, 0) != 0 {
		t.Fatal("edges")
	}
}
