// Package store persists the replica registry so that a restarted control
// plane recovers the replicas registered through the admin API.
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/AKASB1/llm-serving-control/internal/registry"
)

// ReplicaStore persists replica registrations.
type ReplicaStore interface {
	Save(registry.Replica) error
	Delete(id string) error
	ForModel(string) []registry.Replica
	All() []registry.Replica
}

// Memory is an in-process store (no persistence).
type Memory struct {
	mu sync.Mutex
	m  map[string]registry.Replica
}

// NewMemory returns an empty in-memory store.
func NewMemory() *Memory { return &Memory{m: map[string]registry.Replica{}} }

// Save implements ReplicaStore.
func (s *Memory) Save(r registry.Replica) error {
	if r.ID == "" {
		return registry.ErrInvalid
	}
	s.mu.Lock()
	s.m[r.ID] = r
	s.mu.Unlock()
	return nil
}

// Delete implements ReplicaStore.
func (s *Memory) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.m[id]; !ok {
		return registry.ErrNotFound
	}
	delete(s.m, id)
	return nil
}

// All implements ReplicaStore (sorted by ID).
func (s *Memory) All() []registry.Replica {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]registry.Replica, 0, len(s.m))
	for _, r := range s.m {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// ForModel implements ReplicaStore.
func (s *Memory) ForModel(model string) []registry.Replica {
	var out []registry.Replica
	for _, r := range s.All() {
		if r.Model == model {
			out = append(out, r)
		}
	}
	return out
}

// fileFormat is the JSON snapshot on disk.
type fileFormat struct {
	Version  int            `json:"version"`
	Replicas []replicaEntry `json:"replicas"`
}

type replicaEntry struct {
	ID       string `json:"id"`
	Model    string `json:"model"`
	Endpoint string `json:"endpoint"`
	Class    string `json:"class,omitempty"`
	GPUs     int    `json:"gpus,omitempty"`
}

// File is a JSON-snapshot store: every change rewrites the whole snapshot to a
// temporary file and renames it over the old one, so a crash leaves either the
// old or the new snapshot. Only registrations are persisted (ID, model,
// endpoint, class, GPUs); health, lifecycle, and in-flight counts are runtime
// state and start fresh after a restart.
type File struct {
	path string
	mem  *Memory
	wmu  sync.Mutex
}

// OpenFile loads the snapshot at path (an absent file is an empty store).
func OpenFile(path string) (*File, error) {
	f := &File{path: path, mem: NewMemory()}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return f, nil
	}
	if err != nil {
		return nil, err
	}
	var ff fileFormat
	if err := json.Unmarshal(data, &ff); err != nil {
		return nil, fmt.Errorf("store: %s: %w", path, err)
	}
	if ff.Version != 1 {
		return nil, fmt.Errorf("store: %s: unsupported version %d", path, ff.Version)
	}
	for _, e := range ff.Replicas {
		if err := f.mem.Save(registry.Replica{ID: e.ID, Model: e.Model, Endpoint: e.Endpoint, Class: e.Class, GPUs: e.GPUs}); err != nil {
			return nil, fmt.Errorf("store: %s: %w", path, err)
		}
	}
	return f, nil
}

func (f *File) flush() error {
	ff := fileFormat{Version: 1}
	for _, r := range f.mem.All() {
		ff.Replicas = append(ff.Replicas, replicaEntry{ID: r.ID, Model: r.Model, Endpoint: r.Endpoint, Class: r.Class, GPUs: r.GPUs})
	}
	data, err := json.MarshalIndent(ff, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(f.path), 0o755); err != nil {
		return err
	}
	tmp := f.path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, f.path)
}

// Save implements ReplicaStore.
func (f *File) Save(r registry.Replica) error {
	f.wmu.Lock()
	defer f.wmu.Unlock()
	if err := f.mem.Save(r); err != nil {
		return err
	}
	return f.flush()
}

// Delete implements ReplicaStore.
func (f *File) Delete(id string) error {
	f.wmu.Lock()
	defer f.wmu.Unlock()
	if err := f.mem.Delete(id); err != nil {
		return err
	}
	return f.flush()
}

// All implements ReplicaStore.
func (f *File) All() []registry.Replica { return f.mem.All() }

// ForModel implements ReplicaStore.
func (f *File) ForModel(model string) []registry.Replica { return f.mem.ForModel(model) }
