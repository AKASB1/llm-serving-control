package metrics

import (
	"math"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/slo"
)

func TestPercentileNearestRank(t *testing.T) {
	xs := []float64{5, 1, 4, 2, 3}
	cases := map[float64]float64{0.2: 1, 0.5: 3, 0.95: 5, 1: 5, 0.01: 1}
	for p, want := range cases {
		if got := Percentile(append([]float64(nil), xs...), p); got != want {
			t.Errorf("P%v = %v, want %v", p, got, want)
		}
	}
	if !math.IsNaN(Percentile(nil, 0.5)) {
		t.Error("empty percentile must be NaN")
	}
}

func TestHistogramQuantile(t *testing.T) {
	var h Histogram
	for range 90 {
		h.Add(10 * time.Millisecond)
	}
	for range 10 {
		h.Add(2 * time.Second)
	}
	if q := h.Quantile(0.5); q < 0.01 || q > 0.0126 {
		t.Errorf("P50 bucket bound %v", q)
	}
	if q := h.Quantile(0.99); q < 2 || q > 2.52 {
		t.Errorf("P99 bucket bound %v", q)
	}
	var o Histogram
	o.Add(2000 * time.Second)
	if q := o.Quantile(0.5); !math.IsInf(q, 1) {
		t.Errorf("overflow bucket must report +Inf, got %v", q)
	}
}

func targets(t *testing.T) slo.Targets {
	ts, err := slo.NewTargets("interactive", map[string]slo.TargetJSON{
		"interactive": {TTFTs: 1, TPOTs: 0.1},
		"batch":       {TTFTs: 10, TPOTs: 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

func rec(id, class string, st Terminal, ttft, tpot time.Duration, out int, met bool) RequestRecord {
	r := RequestRecord{ID: id, Model: "m", SLOClass: class, OutputTokens: out, State: st, Met: met, Arrival: time.Second}
	if st == Completed {
		r.HasFirstToken = true
		r.FirstToken = r.Arrival + ttft
		r.Completion = r.FirstToken + tpot*time.Duration(out-1)
	}
	return r
}

func TestSummarizeCountsViolationsAndJ(t *testing.T) {
	recs := []RequestRecord{
		rec("a", "interactive", Completed, 500*time.Millisecond, 50*time.Millisecond, 11, true),
		rec("b", "interactive", Completed, 2*time.Second, 50*time.Millisecond, 11, false),
		rec("c", "interactive", Rejected, 0, 0, 5, false),
		rec("d", "batch", Failed, 0, 0, 5, false),
	}
	acc := Accounting{RunDuration: 100 * time.Second, BudgetGPUs: 2, GPUSeconds: 100}
	w := Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}
	s := Summarize(recs, targets(t), w, acc)
	if s.Requests != 4 || s.Completed != 2 || s.Rejected != 1 || s.Failed != 1 {
		t.Fatalf("counts %+v", s)
	}
	if s.ViolationRate != 0.75 {
		t.Fatalf("violation rate %v (rejected and failed must stay in the denominator)", s.ViolationRate)
	}
	if s.Goodput != 0.01 {
		t.Fatalf("goodput %v", s.Goodput)
	}
	// interactive group: n=3, P95 TTFT = 2 s → 2/1; P95 TPOT = 0.05 → 0.5.
	// batch group: n=1, no completions → 100/10 and 100/0.5.
	wantTTFT := (3*2.0 + 1*10.0) / 4
	wantTPOT := (3*0.5 + 1*200.0) / 4
	if math.Abs(s.JTTFT-wantTTFT) > 1e-9 || math.Abs(s.JTPOT-wantTPOT) > 1e-9 {
		t.Fatalf("J terms %v %v, want %v %v", s.JTTFT, s.JTPOT, wantTTFT, wantTPOT)
	}
	if s.JGPU != 0.5 {
		t.Fatalf("GPU term %v", s.JGPU)
	}
	if want := wantTTFT + wantTPOT + 0.5 + 5*0.75; math.Abs(s.J-want) > 1e-9 {
		t.Fatalf("J = %v, want %v", s.J, want)
	}
	if s.TTFTP95 != 2 {
		t.Fatalf("TTFT P95 over completed only: %v", s.TTFTP95)
	}
	if len(s.Values()) != len(Columns()) {
		t.Fatal("Values/Columns length mismatch")
	}
}

func TestRecordTPOT(t *testing.T) {
	r := RequestRecord{OutputTokens: 1, FirstToken: time.Second, Completion: time.Second}
	if r.TPOT() != 0 {
		t.Fatal("single-token TPOT must be 0")
	}
	r = RequestRecord{OutputTokens: 5, FirstToken: time.Second, Completion: 3 * time.Second}
	if r.TPOT() != 500*time.Millisecond {
		t.Fatalf("TPOT %v", r.TPOT())
	}
}
