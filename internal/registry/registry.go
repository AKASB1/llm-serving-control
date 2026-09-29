package registry

import "errors"

type Replica struct {
	ID string
	Model string
	Endpoint string
	Healthy bool
	Outstanding int
}

type Registry struct { replicas map[string]Replica }

func New() *Registry { return &Registry{replicas: make(map[string]Replica)} }

func (r *Registry) Add(replica Replica) error {
	if replica.ID == "" || replica.Model == "" || replica.Endpoint == "" { return errors.New("replica fields required") }
	if _, exists := r.replicas[replica.ID]; exists { return errors.New("duplicate replica") }
	r.replicas[replica.ID] = replica
	return nil
}

func (r *Registry) ForModel(model string) []Replica {
	var result []Replica
	for _, replica := range r.replicas {
		if replica.Model == model { result = append(result, replica) }
	}
	return result
}
