// Command tracegen writes a schema v1 trace and its manifest from a loadgen
// JSON configuration and a seed.
//
//	go run ./cmd/tracegen -config configs/traces/sample.gen.json -seed 1 -out outputs/traces/sample.csv
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/AKASB1/llm-serving-control/internal/trace"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

func main() {
	cfgPath := flag.String("config", "", "loadgen JSON configuration")
	seed := flag.Uint64("seed", 1, "generator seed")
	out := flag.String("out", "", "output CSV path (manifest is written beside it)")
	flag.Parse()
	if err := run(*cfgPath, *seed, *out); err != nil {
		slog.Error("tracegen failed", "err", err)
		os.Exit(1)
	}
}

func run(cfgPath string, seed uint64, out string) error {
	if cfgPath == "" || out == "" {
		return fmt.Errorf("-config and -out are required")
	}
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	var cfg loadgen.Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("%s: %w", cfgPath, err)
	}
	reqs, err := loadgen.Generate(cfg, seed)
	if err != nil {
		return err
	}
	m, err := loadgen.Manifest(cfg, seed)
	if err != nil {
		return err
	}
	m, err = trace.WriteFiles(out, reqs, m)
	if err != nil {
		return err
	}
	slog.Info("trace written", "path", out, "requests", m.Requests, "sha256", m.ContentSHA256)
	return nil
}
