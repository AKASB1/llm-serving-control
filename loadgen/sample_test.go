package loadgen

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/trace"
)

// The committed sample trace must load, verify against its manifest, and be
// reproduced byte for byte by its generator configuration and seed.
func TestCommittedSampleTraceIsReproducible(t *testing.T) {
	const csvPath = "../configs/traces/sample.csv"
	reqs, m, err := trace.LoadFiles(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) == 0 {
		t.Fatal("sample trace is empty")
	}
	data, err := os.ReadFile("../configs/traces/sample.gen.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	gen, err := Generate(cfg, m.Seed)
	if err != nil {
		t.Fatal(err)
	}
	_, sum, err := trace.Encode(gen)
	if err != nil {
		t.Fatal(err)
	}
	if sum != m.ContentSHA256 {
		t.Fatalf("regenerated hash %s != committed %s", sum, m.ContentSHA256)
	}
	if fi, _ := os.Stat(csvPath); fi.Size() >= 100*1024 {
		t.Fatalf("sample trace is %d bytes, must stay below 100 KB", fi.Size())
	}
}
