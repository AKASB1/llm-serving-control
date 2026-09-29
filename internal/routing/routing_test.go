package routing

import (
	"testing"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

func TestLeastOutstandingHealthy(t *testing.T) {
	replicas := []registry.Replica{
		{ID: "a", Healthy: true, Outstanding: 3},
		{ID: "b", Healthy: true, Outstanding: 1},
		{ID: "c", Healthy: false, Outstanding: 0},
	}
	chosen, ok := LeastOutstanding(replicas)
	if !ok || chosen.ID != "b" { t.Fatalf("unexpected choice: %+v", chosen) }
}
