package sim

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/rng"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

func rngFor(seed uint64) *rand.Rand { return rng.Stream(seed, "test-router") }

// TestPollaczekKhinchine is the closed-form check of the simulator: one
// replica with batch size one under Poisson arrivals is an M/G/1 FCFS queue,
// so the mean queue wait must match Pollaczek-Khinchine,
// W_q = λ·E[S²] / (2·(1 − λ·E[S])), computed from the realized service times.
// Tolerance: pooled over 10 seeds within 5 %, every seed within 25 %.
func TestPollaczekKhinchine(t *testing.T) {
	p := engine.Params{
		MaxNumSeqs: 1, MaxBatchedTokens: 4096, KVCapacityTokens: 100000, MaxModelLen: 50000,
		TBase: 0.004, CSeq: 0.0005, CCtx: 1e-6, CPf: 2e-5,
	}
	cls := map[string]Class{"mg1": {
		Spec:   HardwareSpec{Name: "mg1", GPUs: 1},
		Params: p,
		Info:   registry.ClassInfo{Name: "mg1", GPUs: 1, MaxNumSeqs: 1, KVCapacityTokens: p.KVCapacityTokens},
	}}
	const lambda, dur = 5.0, 4000.0
	var simSum, pkSum float64
	for seed := uint64(1); seed <= 10; seed++ {
		cfg := loadgen.Config{Name: "mg1", DurationS: dur, Streams: []loadgen.Stream{{
			Model:   "m",
			Arrival: loadgen.Arrival{Kind: "poisson", Rate: lambda},
			Prompt:  loadgen.Dist{Kind: "lognormal", Median: 200, Sigma: 0.5, Min: 1, Max: 4000},
			Output:  loadgen.Dist{Kind: "lognormal", Median: 20, Sigma: 0.8, Min: 1, Max: 500},
		}}}
		reqs, err := loadgen.Generate(cfg, seed)
		if err != nil {
			t.Fatal(err)
		}
		sc := baseConfig(dur, InitialReplica{Model: "m", Class: "mg1", Count: 1})
		sc.DrainHorizonS = 600
		h := build(t, sc, cls, routing.NewLeastOutstanding(), nil, reqs, Hooks{})
		res := h.sim.Run()
		var n, es, es2, wq float64
		for _, r := range res.Records {
			if r.RouterQueue != 0 {
				t.Fatalf("router queueing in an M/G/1 run: %+v", r)
			}
			s := (r.Completion - r.Arrival - r.ReplicaQueue).Seconds()
			n++
			es += s
			es2 += s * s
			wq += r.ReplicaQueue.Seconds()
		}
		if int(n) != len(reqs) {
			t.Fatalf("seed %d: %d of %d requests completed", seed, int(n), len(reqs))
		}
		es, es2, wq = es/n, es2/n, wq/n
		rho := lambda * es
		pk := lambda * es2 / (2 * (1 - rho))
		rel := math.Abs(wq-pk) / pk
		t.Logf("seed %d: n=%d rho=%.3f E[S]=%.4f sim Wq=%.4f P-K Wq=%.4f rel.err=%.3f", seed, int(n), rho, es, wq, pk, rel)
		if rho < 0.5 || rho > 0.9 {
			t.Fatalf("test misconfigured: rho = %.3f", rho)
		}
		if rel > 0.25 {
			t.Errorf("seed %d: relative error %.3f > 0.25", seed, rel)
		}
		simSum += wq
		pkSum += pk
	}
	if rel := math.Abs(simSum-pkSum) / pkSum; rel > 0.05 {
		t.Fatalf("pooled relative error %.4f > 0.05", rel)
	}
}
