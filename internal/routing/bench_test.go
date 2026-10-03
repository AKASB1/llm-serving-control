package routing

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

// benchView builds a view of n eligible replicas with scraped snapshots and
// fresh state, as the dispatcher would pass it.
func benchView(n int) View {
	r := rand.New(rand.NewPCG(1, 1))
	v := View{Now: 100 * time.Second, Model: "m"}
	for i := 0; i < n; i++ {
		snap := metrics.ReplicaSnapshot{
			ReplicaID: fmt.Sprintf("r%04d", i), At: 99 * time.Second, Running: r.IntN(128), Waiting: r.IntN(20),
			WaitingPromptTokens: r.IntN(20000), KVUsedTokens: r.IntN(400000), KVCapacityTokens: 411529,
			GeneratedTokens: int64(r.IntN(1e6)), BusyTime: time.Duration(r.IntN(90)) * time.Second,
		}
		fresh := snap
		v.Replicas = append(v.Replicas, ReplicaView{
			ID: snap.ReplicaID, InFlight: r.IntN(150), SentSinceScrape: r.IntN(5), Scraped: snap, HasScrape: true,
			Class: registry.ClassInfo{MaxNumSeqs: 128, PrefillTokensPerSec: 26000, DecodeTokensPerSec: 5000, KVCapacityTokens: 411529},
			Fresh: &fresh,
		})
	}
	return v
}

// BenchmarkRoute measures one routing decision per policy at 8, 64, and 512
// eligible replicas (the policy's own work; view construction is measured by
// BenchmarkDispatch in package controller).
func BenchmarkRoute(b *testing.B) {
	policies := []func() Router{
		func() Router { return NewRandom(rand.New(rand.NewPCG(1, 2))) },
		func() Router { return NewRoundRobin() },
		func() Router { return NewLeastOutstanding() },
		func() Router { return NewPowerOfTwo(rand.New(rand.NewPCG(1, 2))) },
		func() Router { return NewLatencyAware(DefaultLatencyAware(), rand.New(rand.NewPCG(1, 2))) },
		func() Router { return NewQueueAware(DefaultQueueAware(), false) },
		func() Router { return NewQueueAware(DefaultQueueAware(), true) },
		func() Router { return NewCapacityWeighted(DefaultCapacityWeighted()) },
		func() Router { return NewOracleJSQ() },
		func() Router { return NewPrefixAffinity(DefaultPrefixAffinity()) },
	}
	for _, n := range []int{8, 64, 512} {
		v := benchView(n)
		for _, mk := range policies {
			p := mk()
			if o, ok := p.(Observer); ok {
				for _, rv := range v.Replicas {
					o.Observe(Outcome{ReplicaID: rv.ID, Now: 90 * time.Second, Kind: FirstToken, Latency: 200 * time.Millisecond})
				}
			}
			req := Request{ID: "x", Model: "m", PromptTokens: 512, PrefixGroup: "g7"}
			b.Run(fmt.Sprintf("%s/replicas=%d", p.Name(), n), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					_ = p.Route(req, v)
				}
			})
		}
	}
}
