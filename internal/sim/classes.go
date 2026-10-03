package sim

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
)

// HardwareSpec describes a replica class from public numbers (specification
// sheets and model architecture) plus assumed efficiency factors. Derive turns
// it into engine parameters; docs/simulator.md documents every field as
// "datasheet", "architecture", or "assumed".
type HardwareSpec struct {
	Name  string `json:"name"`
	Model string `json:"model"`
	// Datasheet values, per GPU.
	GPU        string  `json:"gpu"`
	GPUs       int     `json:"gpus"`
	GPUMemGB   float64 `json:"gpu_mem_gb"`
	MemBWGBs   float64 `json:"mem_bw_gb_s"`
	PeakTFLOPS float64 `json:"peak_tflops"`
	// Model architecture.
	ParamsB         float64 `json:"params_b"`
	BytesPerParam   float64 `json:"bytes_per_param"`
	KVBytesPerToken float64 `json:"kv_bytes_per_token"`
	// Assumed efficiency and overhead factors.
	EffMemBW            float64 `json:"eff_mem_bw"`
	EffFLOPS            float64 `json:"eff_flops"`
	TPEfficiency        float64 `json:"tp_efficiency"`
	MemUtil             float64 `json:"mem_util"`
	ActivationReserveGB float64 `json:"activation_reserve_gb"`
	IterOverheadMs      float64 `json:"iter_overhead_ms"`
	PerSeqOverheadUs    float64 `json:"per_seq_overhead_us"`
	// Assumed serving-engine settings.
	MaxNumSeqs       int `json:"max_num_seqs"`
	MaxBatchedTokens int `json:"max_batched_tokens"`
	MaxModelLen      int `json:"max_model_len"`
	// Assumed lifecycle delays.
	ProvisionS float64 `json:"provision_s"`
	LoadS      float64 `json:"load_s"`
}

// Class is a derived replica class.
type Class struct {
	Spec      HardwareSpec
	Params    engine.Params
	Info      registry.ClassInfo
	Provision time.Duration
	Load      time.Duration
}

// Derive computes engine parameters with a roofline-style linear model:
//
//	t_base = weight bytes / (aggregate bandwidth · eff_mem_bw) + iteration overhead
//	c_ctx  = KV bytes per token / (aggregate bandwidth · eff_mem_bw)
//	c_pf   = 2 · params / (aggregate FLOPS · eff_flops)
//	c_seq  = c_pf + per-sequence overhead
//	KV capacity = (GPUs · memory · mem_util − weights − activation reserve) / KV bytes per token
//
// where aggregate = per-GPU value × GPUs × tp_efficiency (tp_efficiency = 1
// for one GPU).
func (h HardwareSpec) Derive() (Class, error) {
	if h.GPUs < 1 || h.MemBWGBs <= 0 || h.PeakTFLOPS <= 0 || h.ParamsB <= 0 || h.KVBytesPerToken <= 0 {
		return Class{}, fmt.Errorf("class %q: incomplete hardware spec", h.Name)
	}
	tp := 1.0
	if h.GPUs > 1 {
		tp = h.TPEfficiency
	}
	g := float64(h.GPUs)
	bw := h.MemBWGBs * 1e9 * g * tp * h.EffMemBW
	flops := h.PeakTFLOPS * 1e12 * g * tp * h.EffFLOPS
	weights := h.ParamsB * 1e9 * h.BytesPerParam
	perTok := 2 * h.ParamsB * 1e9 / flops
	kvBytes := g*h.GPUMemGB*1e9*h.MemUtil - weights - h.ActivationReserveGB*1e9*g
	kvTokens := int(math.Floor(kvBytes / h.KVBytesPerToken))
	p := engine.Params{
		MaxNumSeqs:       h.MaxNumSeqs,
		MaxBatchedTokens: h.MaxBatchedTokens,
		KVCapacityTokens: kvTokens,
		MaxModelLen:      h.MaxModelLen,
		TBase:            weights/bw + h.IterOverheadMs/1e3,
		CSeq:             perTok + h.PerSeqOverheadUs/1e6,
		CCtx:             h.KVBytesPerToken / bw,
		CPf:              perTok,
	}
	if err := p.Validate(); err != nil {
		return Class{}, fmt.Errorf("class %q: %w (kv tokens %d)", h.Name, err, kvTokens)
	}
	// Nominal rates for the control plane's class prior: prefill tokens/s at a
	// full chunk, aggregate decode tokens/s at a full batch with 1k context.
	pfIter := p.IterationTime(0, 0, p.MaxBatchedTokens).Seconds()
	dcIter := p.IterationTime(p.MaxNumSeqs, p.MaxNumSeqs*1024, 0).Seconds()
	info := registry.ClassInfo{
		Name: h.Name, Model: h.Model, GPUs: h.GPUs,
		PrefillTokensPerSec: float64(p.MaxBatchedTokens) / pfIter,
		DecodeTokensPerSec:  float64(p.MaxNumSeqs) / dcIter,
		MaxNumSeqs:          p.MaxNumSeqs,
		KVCapacityTokens:    p.KVCapacityTokens,
	}
	return Class{
		Spec: h, Params: p, Info: info,
		Provision: time.Duration(h.ProvisionS * 1e9), Load: time.Duration(h.LoadS * 1e9),
	}, nil
}

// LoadClasses reads a JSON array of hardware specs and derives each class.
func LoadClasses(path string) (map[string]Class, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var specs []HardwareSpec
	if err := json.Unmarshal(data, &specs); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return DeriveAll(specs)
}

// DeriveAll derives a set of classes, rejecting duplicate names.
func DeriveAll(specs []HardwareSpec) (map[string]Class, error) {
	out := map[string]Class{}
	for _, s := range specs {
		if _, dup := out[s.Name]; dup {
			return nil, fmt.Errorf("duplicate class %q", s.Name)
		}
		c, err := s.Derive()
		if err != nil {
			return nil, err
		}
		out[s.Name] = c
	}
	return out, nil
}

// ClassInfos returns the control-plane priors of a class set.
func ClassInfos(cs map[string]Class) map[string]registry.ClassInfo {
	out := make(map[string]registry.ClassInfo, len(cs))
	for k, c := range cs {
		out[k] = c.Info
	}
	return out
}

// ClassNames returns sorted class names.
func ClassNames(cs map[string]Class) []string {
	out := make([]string, 0, len(cs))
	for k := range cs {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
