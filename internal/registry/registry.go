// Package registry holds the control plane's view of serving replicas: which
// model each replica serves, its hardware class, its lifecycle state, its
// health as seen by the router, and the router-local in-flight count.
package registry

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// State is a replica's lifecycle state.
type State int

// Lifecycle: Provisioning → Loading → Ready → Draining → Terminated. Failed is
// terminal for a replica that crashed.
const (
	Provisioning State = iota
	Loading
	Ready
	Draining
	Terminated
	Failed
)

var stateNames = [...]string{"provisioning", "loading", "ready", "draining", "terminated", "failed"}

func (s State) String() string {
	if s < 0 || int(s) >= len(stateNames) {
		return fmt.Sprintf("state(%d)", int(s))
	}
	return stateNames[s]
}

// Live reports whether the replica still holds GPUs (it is paid for).
func (s State) Live() bool { return s <= Draining }

// Replica is one instance of one model on a given number of GPUs.
type Replica struct {
	ID       string
	Model    string
	Endpoint string
	Class    string
	GPUs     int
	State    State
	// Healthy is the router's health verdict (false while ejected).
	Healthy bool
	// Outstanding is the router-local in-flight count: requests dispatched to
	// this replica that have not completed or failed. Always fresh.
	Outstanding int
}

// Eligible reports whether the replica may receive new requests.
func (r Replica) Eligible() bool { return r.State == Ready && r.Healthy }

// Registry is a concurrency-safe replica table. Every listing is sorted by ID
// so that callers are deterministic.
type Registry struct {
	mu       sync.RWMutex
	replicas map[string]Replica
}

// New returns an empty registry.
func New() *Registry { return &Registry{replicas: make(map[string]Replica)} }

// Errors returned by the registry.
var (
	ErrInvalid   = errors.New("replica fields required")
	ErrDuplicate = errors.New("duplicate replica")
	ErrNotFound  = errors.New("replica not found")
)

// Add registers a new replica. ID, Model, and Endpoint are required.
func (r *Registry) Add(replica Replica) error {
	if replica.ID == "" || replica.Model == "" || replica.Endpoint == "" {
		return ErrInvalid
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.replicas[replica.ID]; exists {
		return ErrDuplicate
	}
	r.replicas[replica.ID] = replica
	return nil
}

// Remove deletes a replica.
func (r *Registry) Remove(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.replicas[id]; !ok {
		return ErrNotFound
	}
	delete(r.replicas, id)
	return nil
}

// Get returns a copy of one replica.
func (r *Registry) Get(id string) (Replica, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rep, ok := r.replicas[id]
	return rep, ok
}

// Update applies fn to a replica under the write lock.
func (r *Registry) Update(id string, fn func(*Replica)) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	rep, ok := r.replicas[id]
	if !ok {
		return ErrNotFound
	}
	fn(&rep)
	r.replicas[id] = rep
	return nil
}

// ForModel returns the replicas of one model, sorted by ID.
func (r *Registry) ForModel(model string) []Replica {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var result []Replica
	for _, rep := range r.replicas {
		if rep.Model == model {
			result = append(result, rep)
		}
	}
	sortByID(result)
	return result
}

// All returns every replica, sorted by ID.
func (r *Registry) All() []Replica {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]Replica, 0, len(r.replicas))
	for _, rep := range r.replicas {
		result = append(result, rep)
	}
	sortByID(result)
	return result
}

// Models returns the distinct model names, sorted.
func (r *Registry) Models() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	seen := map[string]bool{}
	for _, rep := range r.replicas {
		seen[rep.Model] = true
	}
	out := make([]string, 0, len(seen))
	for m := range seen {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

func sortByID(rs []Replica) {
	sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
}
