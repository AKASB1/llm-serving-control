package sim

import (
	"sort"
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/routing"
)

func router(t *testing.T, name string) routing.Router {
	t.Helper()
	r, err := controller.NewRouter(controller.PolicySpec{Name: name}, 99)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// Every routing policy runs end to end on a heterogeneous cluster, and every
// request ends in a terminal state.
func TestEveryRouterRunsOnHeterogeneousCluster(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 30, 60, 99)
	for _, name := range controller.RouterNames() {
		h := build(t, baseConfig(60,
			InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 2},
			InitialReplica{Model: "chat-8b", Class: "a100-8b", Count: 1},
			InitialReplica{Model: "chat-8b", Class: "v100-8b", Count: 1}), cs, router(t, name), nil, reqs, Hooks{})
		res := h.sim.Run()
		st := countStates(res.Records)
		if st[metrics.Completed]+st[metrics.Unfinished] != len(reqs) || st[metrics.Completed] == 0 {
			t.Fatalf("%s: states %v", name, st)
		}
	}
}

// sameReplicaShare is the fraction of consecutive arrivals sent to the same
// replica.
func sameReplicaShare(recs []metrics.RequestRecord) float64 {
	rs := append([]metrics.RequestRecord(nil), recs...)
	sort.Slice(rs, func(i, j int) bool { return rs[i].Arrival < rs[j].Arrival })
	same := 0
	for i := 1; i < len(rs); i++ {
		if rs[i].Replica == rs[i-1].Replica {
			same++
		}
	}
	return float64(same) / float64(len(rs)-1)
}

// Stale-snapshot herding is shown, not silently fixed: with a 5 s scrape
// interval queue_aware sends runs of consecutive requests to one replica; the
// corrected variant spreads them.
func TestStaleSnapshotHerding(t *testing.T) {
	cs := testClasses(t)
	reqs := shortTrace(t, "chat-8b", 40, 60, 98)
	cfg := baseConfig(60, InitialReplica{Model: "chat-8b", Class: "h100-8b", Count: 4})
	cfg.ScrapeIntervalS = 5
	plain := build(t, cfg, cs, router(t, "queue_aware"), nil, reqs, Hooks{}).sim.Run()
	corr := build(t, cfg, cs, router(t, "queue_aware_corrected"), nil, reqs, Hooks{}).sim.Run()
	sp, sc := sameReplicaShare(plain.Records), sameReplicaShare(corr.Records)
	t.Logf("same-replica share: queue_aware %.2f, queue_aware_corrected %.2f (uniform: 0.25)", sp, sc)
	if sp < 0.8 || sc > 0.5 {
		t.Fatalf("herding not visible or not mitigated: %.2f vs %.2f", sp, sc)
	}
}
