package backends

import "github.com/AKASB1/llm-serving-control/internal/registry"

type Backend interface {
	Replicas() ([]registry.Replica, error)
}
