// Command traceconv converts a public LLM-serving trace to schema v1 (with a
// manifest that records the source file's SHA-256 and the conversion).
//
//	go run ./cmd/traceconv -format azure -in outputs/real-traces/AzureLLMInferenceTrace_conv.csv \
//	    -model chat-8b -start-s 0 -duration-s 300 -out configs/traces/azure-conv-300s.csv
//
// Supported formats: "azure" (Azure LLM inference trace 2023) and
// "burstgpt" (BurstGPT). Both datasets are CC-BY 4.0; cite them when you use
// converted traces.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/AKASB1/llm-serving-control/internal/trace"
)

type params struct {
	Format       string             `json:"format"`
	Source       string             `json:"source"`
	SourceSHA256 string             `json:"source_sha256"`
	Model        string             `json:"model,omitempty"`
	ModelMap     map[string]string  `json:"model_map,omitempty"`
	StartS       float64            `json:"start_s"`
	DurationS    float64            `json:"duration_s"`
	MaxContext   int                `json:"max_context"`
	Stats        trace.ConvertStats `json:"stats"`
}

func main() {
	var p params
	in := flag.String("in", "", "source trace file")
	out := flag.String("out", "", "output CSV (manifest written beside it)")
	modelMap := flag.String("model-map", "", "BurstGPT model renames, e.g. ChatGPT=chat-8b,GPT-4=chat-70b")
	flag.StringVar(&p.Format, "format", "azure", "source format: azure or burstgpt")
	flag.StringVar(&p.Model, "model", "", "model name for every request (Azure) or the default (BurstGPT)")
	flag.Float64Var(&p.StartS, "start-s", 0, "window start, seconds after the first source request")
	flag.Float64Var(&p.DurationS, "duration-s", 0, "window length in seconds (0: to the end)")
	flag.IntVar(&p.MaxContext, "max-context", 16384, "cap on prompt+output tokens (0: none)")
	flag.Parse()
	if err := run(&p, *in, *out, *modelMap); err != nil {
		slog.Error("traceconv failed", "err", err)
		os.Exit(1)
	}
}

func run(p *params, in, out, modelMap string) error {
	if in == "" || out == "" {
		return fmt.Errorf("-in and -out are required")
	}
	data, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	p.Source, p.SourceSHA256 = filepath.Base(in), hex.EncodeToString(sum[:])
	if modelMap != "" {
		p.ModelMap = map[string]string{}
		for _, kv := range strings.Split(modelMap, ",") {
			k, v, ok := strings.Cut(kv, "=")
			if !ok {
				return fmt.Errorf("bad -model-map entry %q", kv)
			}
			p.ModelMap[k] = v
		}
	}
	o := trace.ConvertOptions{Model: p.Model, ModelMap: p.ModelMap, StartS: p.StartS, DurationS: p.DurationS, MaxContext: p.MaxContext}
	var reqs []trace.Request
	switch p.Format {
	case "azure":
		reqs, p.Stats, err = trace.FromAzure(strings.NewReader(string(data)), o)
	case "burstgpt":
		reqs, p.Stats, err = trace.FromBurstGPT(strings.NewReader(string(data)), o)
	default:
		err = fmt.Errorf("unknown format %q", p.Format)
	}
	if err != nil {
		return err
	}
	raw, err := json.Marshal(p)
	if err != nil {
		return err
	}
	dur := p.DurationS
	if dur <= 0 && len(reqs) > 0 {
		dur = reqs[len(reqs)-1].ArrivalS
	}
	m, err := trace.WriteFiles(out, reqs, trace.Manifest{Generator: trace.Generator{Name: "traceconv", Version: 1, Params: raw}, DurationS: dur})
	if err != nil {
		return err
	}
	slog.Info("trace converted", "out", out, "requests", m.Requests, "dropped", p.Stats.Dropped, "truncated", p.Stats.Truncated)
	return nil
}
