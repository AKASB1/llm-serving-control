package sim

import (
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

// Under sustained overload, predicted_ttft_shed rejects requests (batch class
// first); rejected requests stay in the SLO denominator.
func TestOverloadWithAndWithoutAdmission(t *testing.T) {
	cs := testClasses(t)
	cfg := loadgen.Config{Name: "overload", DurationS: 120, Streams: []loadgen.Stream{{
		Model:   "chat-8b",
		Arrival: loadgen.Arrival{Kind: "poisson", Rate: 60},
		Prompt:  loadgen.Dist{Kind: "lognormal", Median: 256, Sigma: 1.0, Min: 8, Max: 4096},
		Output:  loadgen.Dist{Kind: "lognormal", Median: 128, Sigma: 1.2, Min: 1, Max: 2048},
		Classes: []loadgen.ClassShare{{Class: "interactive", Share: 0.7}, {Class: "batch", Share: 0.3}},
	}}}
	reqs, _ := loadgen.Generate(cfg, 11)
	run := func(adm string) (metrics.Summary, []metrics.RequestRecord) {
		reg := registry.New()
		pol, err := controller.NewAdmission(controller.PolicySpec{Name: adm}, testTargets(t))
		if err != nil {
			t.Fatal(err)
		}
		disp := controller.NewDispatcher(controller.DefaultDispatch(), reg, ClassInfos(cs), router(t, "least_outstanding"), pol, controller.NewHealth(controller.DefaultHealth()))
		s, err := New(baseConfig(120, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 2}), cs, reg, disp, nil, testTargets(t),
			metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, reqs, Hooks{})
		if err != nil {
			t.Fatal(err)
		}
		res := s.Run()
		return metrics.Summarize(res.Records, testTargets(t), metrics.Weights{Alpha: 1, Beta: 1, Gamma: 1, Delta: 5}, res.Accounting), res.Records
	}
	none, _ := run("none")
	shed, recs := run("predicted_ttft_shed")
	if none.Rejected != 0 || shed.Rejected == 0 {
		t.Fatalf("rejections: none %d, shed %d", none.Rejected, shed.Rejected)
	}
	byClass := map[string][2]int{}
	met := 0
	for _, r := range recs {
		c := byClass[r.SLOClass]
		c[0]++
		if r.State == metrics.Rejected {
			c[1]++
			if r.Met {
				t.Fatal("a rejected request met its SLO")
			}
		}
		if r.Met {
			met++
		}
		byClass[r.SLOClass] = c
	}
	if want := 1 - float64(met)/float64(len(recs)); shed.ViolationRate != want {
		t.Fatalf("violation rate %v, want %v (rejections in the denominator)", shed.ViolationRate, want)
	}
	bi, ii := byClass["batch"], byClass["interactive"]
	if float64(bi[1])/float64(bi[0]) <= float64(ii[1])/float64(ii[0]) {
		t.Fatalf("batch must be shed first: batch %v interactive %v", bi, ii)
	}
	t.Logf("none: viol %.3f P95 TTFT %.2fs | shed: viol %.3f P95 TTFT %.2fs rejected %d (batch %d/%d, interactive %d/%d)",
		none.ViolationRate, none.TTFTP95, shed.ViolationRate, shed.TTFTP95, shed.Rejected, bi[1], bi[0], ii[1], ii[0])
}
