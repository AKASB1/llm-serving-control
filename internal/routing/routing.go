package routing

import (
	"sort"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

func LeastOutstanding(replicas []registry.Replica) (registry.Replica, bool) {
	healthy := make([]registry.Replica, 0, len(replicas))
	for _, replica := range replicas {
		if replica.Healthy { healthy = append(healthy, replica) }
	}
	if len(healthy) == 0 { return registry.Replica{}, false }
	sort.Slice(healthy, func(i, j int) bool {
		if healthy[i].Outstanding == healthy[j].Outstanding { return healthy[i].ID < healthy[j].ID }
		return healthy[i].Outstanding < healthy[j].Outstanding
	})
	return healthy[0], true
}
