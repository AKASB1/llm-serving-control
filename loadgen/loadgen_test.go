package loadgen

import (
	"math"
	"sort"
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/rng"
	"github.com/AKASB1/llm-serving-control/internal/trace"
)

var shortMix = Stream{
	Prompt: Dist{Kind: "lognormal", Median: 256, Sigma: 1.0, Min: 8, Max: 4096},
	Output: Dist{Kind: "lognormal", Median: 128, Sigma: 1.2, Min: 1, Max: 2048},
}

func interarrivalStats(ts []float64) (mean, cv float64) {
	n := len(ts) - 1
	var s, s2 float64
	for i := 1; i < len(ts); i++ {
		d := ts[i] - ts[i-1]
		s += d
		s2 += d * d
	}
	mean = s / float64(n)
	v := s2/float64(n) - mean*mean
	return mean, math.Sqrt(v) / mean
}

func within(t *testing.T, name string, got, want, relTol float64) {
	t.Helper()
	if math.Abs(got-want) > relTol*math.Abs(want) {
		t.Errorf("%s = %.4f, want %.4f ± %.1f%%", name, got, want, relTol*100)
	}
}

// Seeds used for the statistical tests; every seed must pass.
var statSeeds = []uint64{1, 2, 3}

func TestPoissonRateAndCV(t *testing.T) {
	a := Arrival{Kind: "poisson", Rate: 50}
	for _, seed := range statSeeds {
		ts := a.Times(rng.Stream(seed, "t"), 2000)
		within(t, "poisson rate", float64(len(ts))/2000, 50, 0.02)
		_, cv := interarrivalStats(ts)
		within(t, "poisson cv", cv, 1, 0.03)
	}
}

func TestGammaRenewalRateAndCV(t *testing.T) {
	a := Arrival{Kind: "gamma", Rate: 20, CV: 3}
	for _, seed := range statSeeds {
		ts := a.Times(rng.Stream(seed, "t"), 5000)
		within(t, "gamma rate", float64(len(ts))/5000, 20, 0.06)
		_, cv := interarrivalStats(ts)
		within(t, "gamma cv", cv, 3, 0.10)
	}
}

func TestMMPPRateAndBurstiness(t *testing.T) {
	a := Arrival{Kind: "mmpp", Rates: []float64{60, 5}, MeanSojournS: []float64{10, 30}}
	if got := a.MeanRate(); math.Abs(got-18.75) > 1e-12 {
		t.Fatalf("MeanRate = %v", got)
	}
	for _, seed := range statSeeds {
		ts := a.Times(rng.Stream(seed, "t"), 20000)
		within(t, "mmpp rate", float64(len(ts))/20000, 18.75, 0.08)
		if _, cv := interarrivalStats(ts); cv <= 1.2 {
			t.Errorf("mmpp cv = %.3f, want > 1.2 (bursty)", cv)
		}
	}
}

func TestDiurnalRateAndShape(t *testing.T) {
	a := Arrival{Kind: "diurnal", Rate: 20, Amplitude: 0.5, PeriodS: 1000}
	for _, seed := range statSeeds {
		ts := a.Times(rng.Stream(seed, "t"), 10000)
		within(t, "diurnal rate", float64(len(ts))/10000, 20, 0.02)
		// Peak quarter (phase 0.125..0.375) vs trough quarter (0.625..0.875).
		var peak, trough int
		for _, x := range ts {
			ph := math.Mod(x, 1000) / 1000
			switch {
			case ph >= 0.125 && ph < 0.375:
				peak++
			case ph >= 0.625 && ph < 0.875:
				trough++
			}
		}
		// Expected ratio (1 + 0.5·0.9)/(1 − 0.5·0.9) ≈ 2.6.
		if r := float64(peak) / float64(trough); r < 2.3 || r > 2.9 {
			t.Errorf("peak/trough = %.2f", r)
		}
	}
}

func TestLengthMixMediansAndTails(t *testing.T) {
	r := rng.Stream(9, "len")
	sample := func(d Dist, n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = d.Sample(r)
		}
		sort.Ints(out)
		return out
	}
	p := sample(shortMix.Prompt, 50000)
	within(t, "prompt median", float64(p[len(p)/2]), 256, 0.04)
	light := Dist{Kind: "lognormal", Median: 128, Sigma: 0.8, Min: 1, Max: 100000}
	heavy := Dist{Kind: "lognormal", Median: 128, Sigma: 1.6, Min: 1, Max: 100000}
	l, h := sample(light, 50000), sample(heavy, 50000)
	tail := func(x []int) float64 { return float64(x[len(x)*99/100]) / float64(x[len(x)/2]) }
	if tail(h) <= 2*tail(l) {
		t.Errorf("heavy tail P99/median %.1f not well above light %.1f", tail(h), tail(l))
	}
	// Theoretical P99/median of a lognormal is exp(2.326·sigma).
	within(t, "heavy P99/median", tail(h), math.Exp(2.326*1.6), 0.10)
	pa := Dist{Kind: "pareto", Xm: 10, Alpha: 1.5, Min: 1, Max: 1 << 30}
	x := sample(pa, 50000)
	within(t, "pareto median", float64(x[len(x)/2]), 10*math.Pow(2, 1/1.5), 0.05)
	for _, v := range sample(Dist{Kind: "lognormal", Median: 100, Sigma: 3, Min: 5, Max: 50}, 1000) {
		if v < 5 || v > 50 {
			t.Fatalf("clip violated: %d", v)
		}
	}
}

func twoModelConfig() Config {
	a := shortMix
	a.Model = "alpha"
	a.Arrival = Arrival{Kind: "poisson", Rate: 10}
	a.Classes = []ClassShare{{"interactive", 0.7}, {"batch", 0.3}}
	a.PrefixGroups, a.PrefixShare = 4, 0.5
	a.MaxContext = 4200
	b := shortMix
	b.Model = "beta"
	b.Arrival = Arrival{Kind: "diurnal", Rate: 5, Amplitude: 0.8, PeriodS: 300, Phase: 0.5}
	return Config{Name: "two", DurationS: 600, Streams: []Stream{a, b}}
}

func filter(reqs []trace.Request, model string) []trace.Request {
	var out []trace.Request
	for _, r := range reqs {
		if r.Model == model {
			out = append(out, r)
		}
	}
	return out
}

func TestGenerateDeterministicAndIndependent(t *testing.T) {
	cfg := twoModelConfig()
	a1, err := Generate(cfg, 42)
	if err != nil {
		t.Fatal(err)
	}
	a2, _ := Generate(cfg, 42)
	if len(a1) != len(a2) {
		t.Fatal("non-deterministic length")
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Fatalf("row %d differs", i)
		}
	}
	b, _ := Generate(cfg, 43)
	if len(b) == len(a1) && b[0] == a1[0] {
		t.Fatal("different seeds gave the same trace")
	}
	// Removing the beta stream must not change alpha's requests.
	solo := cfg
	solo.Streams = cfg.Streams[:1]
	s, _ := Generate(solo, 42)
	fa := filter(a1, "alpha")
	if len(s) != len(fa) {
		t.Fatalf("alpha has %d requests alone, %d with beta", len(s), len(fa))
	}
	for i := range s {
		if s[i] != fa[i] {
			t.Fatalf("alpha row %d changed when beta was added", i)
		}
	}
}

func TestGenerateClassesPrefixesAndContext(t *testing.T) {
	cfg := twoModelConfig()
	cfg.DurationS = 3000
	reqs, err := Generate(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	alpha := filter(reqs, "alpha")
	var inter, pref int
	for _, r := range alpha {
		if r.SLOClass == "interactive" {
			inter++
		}
		if r.PrefixGroup != "" {
			pref++
		}
		if r.PromptTokens+r.OutputTokens > 4200 {
			t.Fatalf("max_context violated: %+v", r)
		}
	}
	within(t, "interactive share", float64(inter)/float64(len(alpha)), 0.7, 0.03)
	within(t, "prefix share", float64(pref)/float64(len(alpha)), 0.5, 0.05)
	for _, r := range filter(reqs, "beta") {
		if r.SLOClass != "" || r.PrefixGroup != "" {
			t.Fatalf("beta must have empty class and prefix: %+v", r)
		}
	}
	if err := trace.Validate(reqs); err != nil {
		t.Fatal(err)
	}
}

func TestConfigValidation(t *testing.T) {
	bad := []Config{
		{DurationS: 0, Streams: twoModelConfig().Streams},
		{DurationS: 10},
		{DurationS: 10, Streams: []Stream{{Model: "m", Arrival: Arrival{Kind: "poisson"}, Prompt: shortMix.Prompt, Output: shortMix.Output}}},
		{DurationS: 10, Streams: []Stream{{Model: "m", Arrival: Arrival{Kind: "mmpp", Rates: []float64{1}}, Prompt: shortMix.Prompt, Output: shortMix.Output}}},
		{DurationS: 10, Streams: []Stream{{Model: "m", Arrival: Arrival{Kind: "diurnal", Rate: 1, Amplitude: 1, PeriodS: 1}, Prompt: shortMix.Prompt, Output: shortMix.Output}}},
		{DurationS: 10, Streams: []Stream{{Model: "m", Arrival: Arrival{Kind: "poisson", Rate: 1}, Prompt: Dist{Kind: "zipf", Min: 1, Max: 2}, Output: shortMix.Output}}},
	}
	for i, c := range bad {
		if err := c.Validate(); err == nil {
			t.Errorf("config %d should be invalid", i)
		}
	}
}

func TestScaled(t *testing.T) {
	a := Arrival{Kind: "mmpp", Rates: []float64{10, 2}, MeanSojournS: []float64{1, 1}}
	b := a.Scaled(2)
	if b.Rates[0] != 20 || a.Rates[0] != 10 || b.MeanRate() != 12 {
		t.Fatalf("Scaled: %+v / %+v", a, b)
	}
}

func TestPiecewiseRates(t *testing.T) {
	a := Arrival{Kind: "piecewise", Rates: []float64{10, 40, 10}, StartsS: []float64{0, 1000, 2000}}
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, seed := range statSeeds {
		ts := a.Times(rng.Stream(seed, "t"), 3000)
		var seg [3]int
		for _, x := range ts {
			seg[int(x/1000)]++
		}
		within(t, "segment 0 rate", float64(seg[0])/1000, 10, 0.08)
		within(t, "segment 1 rate", float64(seg[1])/1000, 40, 0.05)
		within(t, "segment 2 rate", float64(seg[2])/1000, 10, 0.08)
	}
	if (Arrival{Kind: "piecewise", Rates: []float64{1}, StartsS: []float64{5}}).Validate() == nil {
		t.Fatal("must start at 0")
	}
}
