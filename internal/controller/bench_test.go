package controller

import (
	"fmt"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
)

// BenchmarkDispatch measures a full routing decision through the dispatcher
// (eligible-view construction from the registry, snapshots, and the policy)
// at 8, 64, and 512 replicas, followed by the completion that releases it.
func BenchmarkDispatch(b *testing.B) {
	for _, n := range []int{8, 64, 512} {
		for _, name := range []string{"least_outstanding", "power_of_two", "queue_aware_corrected", "latency_aware"} {
			reg := registry.New()
			classes := map[string]registry.ClassInfo{"c": {Name: "c", MaxNumSeqs: 128, PrefillTokensPerSec: 26000, DecodeTokensPerSec: 5000}}
			for i := 0; i < n; i++ {
				_ = reg.Add(registry.Replica{ID: fmt.Sprintf("r%04d", i), Model: "m", Endpoint: "e", Class: "c", State: registry.Ready, Healthy: true})
			}
			router, _ := NewRouter(PolicySpec{Name: name}, 1)
			d := NewDispatcher(DefaultDispatch(), reg, classes, router, nil, NewHealth(DefaultHealth()))
			for i := 0; i < n; i++ {
				d.Snapshot(0, metrics.ReplicaSnapshot{ReplicaID: fmt.Sprintf("r%04d", i), At: 0, Running: i % 50, Waiting: i % 7, KVCapacityTokens: 1000})
			}
			b.Run(fmt.Sprintf("%s/replicas=%d", name, n), func(b *testing.B) {
				b.ReportAllocs()
				now := time.Second
				for i := 0; i < b.N; i++ {
					id := "q"
					cmds := d.Arrive(now, routing.Request{ID: id, Model: "m", PromptTokens: 512})
					if len(cmds) != 1 || cmds[0].Kind != Send {
						b.Fatal("not dispatched")
					}
					d.FirstToken(now, id)
					d.Complete(now, id, 10)
				}
			})
		}
	}
}
