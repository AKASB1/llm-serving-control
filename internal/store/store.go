package store

import "github.com/AKASB1/llm-serving-control/internal/registry"

type ReplicaStore interface {
	Save(registry.Replica) error
	ForModel(string) []registry.Replica
}
