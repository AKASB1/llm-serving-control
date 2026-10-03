package backends

import (
	"context"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

// Backend lists the replicas a serving backend exposes.
type Backend interface {
	Replicas() ([]registry.Replica, error)
}

// Scraper reads one replica's metrics endpoint into a snapshot. The vLLM and
// SGLang adapters (internal/adapters) implement it; so does the mock backend's
// vLLM-named endpoint.
type Scraper interface {
	Scrape(ctx context.Context, endpoint, replicaID string) (metrics.ReplicaSnapshot, error)
}

// Static is a fixed replica list (from configuration).
type Static []registry.Replica

// Replicas implements Backend.
func (s Static) Replicas() ([]registry.Replica, error) {
	return append([]registry.Replica(nil), s...), nil
}
