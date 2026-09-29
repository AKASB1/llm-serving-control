package loadgen

import "github.com/AKASB1/llm-serving-control/internal/registry"

func ModelRequests(model string, count int, r *registry.Registry) int {
	if count < 0 || len(r.ForModel(model)) == 0 { return 0 }
	return count
}
