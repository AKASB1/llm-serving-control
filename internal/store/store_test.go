package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/registry"
)

func TestFileStoreRestartRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "replicas.json")
	s, err := OpenFile(path)
	if err != nil || len(s.All()) != 0 {
		t.Fatalf("fresh store: %v %v", s.All(), err)
	}
	for _, r := range []registry.Replica{
		{ID: "b", Model: "m1", Endpoint: "http://b", Class: "h100-8b", GPUs: 1, State: registry.Ready, Healthy: true, Outstanding: 7},
		{ID: "a", Model: "m1", Endpoint: "http://a"},
		{ID: "c", Model: "m2", Endpoint: "http://c"},
	} {
		if err := s.Save(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Delete("c"); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete("c"); !errors.Is(err, registry.ErrNotFound) {
		t.Fatalf("double delete: %v", err)
	}
	// "Restart": a new store over the same file sees the registrations only.
	s2, err := OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	got := s2.ForModel("m1")
	if len(got) != 2 || got[0].ID != "a" || got[1].ID != "b" || got[1].Class != "h100-8b" || got[1].Endpoint != "http://b" {
		t.Fatalf("recovered %+v", got)
	}
	if got[1].Outstanding != 0 || got[1].Healthy {
		t.Fatalf("runtime state must not be persisted: %+v", got[1])
	}
	if len(s2.ForModel("m2")) != 0 {
		t.Fatal("deleted replica came back")
	}
	if _, err := os.Stat(path + ".tmp"); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("temporary file left behind")
	}
}

func TestFileStoreRejectsCorruptSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replicas.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenFile(path); err == nil {
		t.Fatal("corrupt snapshot accepted")
	}
	if err := os.WriteFile(path, []byte(`{"version":2,"replicas":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenFile(path); err == nil {
		t.Fatal("unknown version accepted")
	}
}

func TestMemoryStoreValidates(t *testing.T) {
	m := NewMemory()
	if err := m.Save(registry.Replica{}); !errors.Is(err, registry.ErrInvalid) {
		t.Fatalf("empty ID: %v", err)
	}
	var _ ReplicaStore = m
	var _ ReplicaStore = (*File)(nil)
}
