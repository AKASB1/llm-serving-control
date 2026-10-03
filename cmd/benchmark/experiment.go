package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/sim"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/loadgen"
)

// Experiment is the committed benchmark configuration (configs/experiment.json).
type Experiment struct {
	Weights metrics.Weights `json:"weights"`
	SLO     struct {
		Default string                    `json:"default"`
		Classes map[string]slo.TargetJSON `json:"classes"`
	} `json:"slo"`
	Seeds struct {
		Tuning      []uint64 `json:"tuning"`
		Evaluation  []uint64 `json:"evaluation"`
		Sensitivity []uint64 `json:"sensitivity"`
		Quick       []uint64 `json:"quick"`
	} `json:"seeds"`
	ClassesFile string                    `json:"classes_file"`
	Sim         sim.Config                `json:"sim"`
	Dispatch    controller.DispatchConfig `json:"dispatch"`
	Health      controller.HealthConfig   `json:"health"`
	WindowS     float64                   `json:"window_s"`
	// PeakUtilization sizes the static_peak oracle: ceil(peak rate /
	// (capacity per replica × PeakUtilization)).
	PeakUtilization float64         `json:"peak_utilization"`
	TieBand         float64         `json:"tie_band"`
	Probe           ProbeConfig     `json:"capacity_probe"`
	Mixes           map[string]Mix  `json:"mixes"`
	Families        []Family        `json:"families"`
	Scenarios       []Scenario      `json:"scenarios"`
	Sensitivity     SensitivityConf `json:"sensitivity"`
	Quick           QuickConf       `json:"quick"`
	Tuning          TuningConf      `json:"tuning"`
}

// ProbeConfig sizes the saturation-throughput probe.
type ProbeConfig struct {
	Requests  int     `json:"requests"`
	DurationS float64 `json:"duration_s"`
	WarmupS   float64 `json:"warmup_s"`
}

// Mix is a request-length mix.
type Mix struct {
	Prompt     loadgen.Dist `json:"prompt"`
	Output     loadgen.Dist `json:"output"`
	MaxContext int          `json:"max_context"`
}

// Family groups scenarios that vary one dimension.
type Family struct {
	Name     string   `json:"name"`
	Policies []string `json:"policies"`
	Baseline string   `json:"baseline"`
	// Oracles are policy labels reported as bounds, never as deployable.
	Oracles []string `json:"oracles"`
}

// ClassCount is a number of ready-at-start replicas of a class.
type ClassCount struct {
	Class string `json:"class"`
	Count int    `json:"count"`
}

// ArrivalSpec is an arrival process relative to a reference rate R.
type ArrivalSpec struct {
	Kind string `json:"kind"`
	// mmpp: state rates ∝ RateMultipliers, normalised so the long-run mean is
	// R; piecewise: segment rates = RateMultipliers × R.
	RateMultipliers []float64 `json:"rate_multipliers,omitempty"`
	MeanSojournS    []float64 `json:"mean_sojourn_s,omitempty"`
	StartsS         []float64 `json:"starts_s,omitempty"`
	CV              float64   `json:"cv,omitempty"`
	Amplitude       float64   `json:"amplitude,omitempty"`
	PeriodS         float64   `json:"period_s,omitempty"`
	Phase           float64   `json:"phase,omitempty"`
}

// ScenarioModel is one model's traffic and replicas in a scenario.
type ScenarioModel struct {
	Model string `json:"model"`
	// Class is used for replicas added by the control loop and for the
	// capacity reference when LoadReplicas > 0.
	Class   string       `json:"class"`
	Initial []ClassCount `json:"initial"`
	Min     int          `json:"min"`
	Max     int          `json:"max"`
	Slots   int          `json:"slots"`
	// Reference rate R = Load × (LoadReplicas × capacity(Class) if
	// LoadReplicas > 0, else Σ initial count × capacity(class)).
	Load         float64              `json:"load"`
	LoadReplicas float64              `json:"load_replicas,omitempty"`
	Arrival      ArrivalSpec          `json:"arrival"`
	Classes      []loadgen.ClassShare `json:"slo_classes,omitempty"`
	// PrefixGroups and PrefixShare give requests shared-prefix groups.
	PrefixGroups int     `json:"prefix_groups,omitempty"`
	PrefixShare  float64 `json:"prefix_share,omitempty"`
}

// Scenario is one benchmark scenario.
type Scenario struct {
	ID         string          `json:"id"`
	Family     string          `json:"family"`
	DurationS  float64         `json:"duration_s"`
	Mix        string          `json:"mix"`
	BudgetGPUs int             `json:"budget_gpus"`
	Models     []ScenarioModel `json:"models"`
	Crashes    []sim.Crash     `json:"crashes,omitempty"`
	// TracePath replays a committed trace instead of generating one.
	TracePath string `json:"trace_path,omitempty"`
	// PrefixTokens is the shared-prefix length; PrefixCacheGroups > 0
	// enables each replica's prefix cache (LRU over that many groups).
	PrefixTokens      int `json:"prefix_tokens,omitempty"`
	PrefixCacheGroups int `json:"prefix_cache_groups,omitempty"`
}

// Variant changes assumptions for the sensitivity check.
type Variant struct {
	ID              string  `json:"id"`
	WarmupScale     float64 `json:"warmup_scale,omitempty"`
	ScrapeIntervalS float64 `json:"scrape_interval_s,omitempty"`
	OutputSigma     float64 `json:"output_sigma,omitempty"`
}

// SensitivityConf lists the sensitivity scenarios and variants.
type SensitivityConf struct {
	Scenarios []string  `json:"scenarios"`
	Variants  []Variant `json:"variants"`
}

// QuickConf is the -quick subset.
type QuickConf struct {
	Scenarios     []string `json:"scenarios"`
	DurationScale float64  `json:"duration_scale"`
}

// TuningConf fixes the search budget.
type TuningConf struct {
	Configs    int    `json:"configs"`
	SearchSeed uint64 `json:"search_seed"`
}

// LoadExperiment reads and validates the experiment and returns it with the
// SHA-256 of the bytes that define it (experiment, classes).
func LoadExperiment(path string) (*Experiment, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var e Experiment
	if err := dec.Decode(&e); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	if err := e.validate(); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return &e, data, nil
}

func (e *Experiment) validate() error {
	if len(e.Seeds.Evaluation) < 10 {
		return fmt.Errorf("need at least 10 evaluation seeds")
	}
	tune := map[uint64]bool{}
	for _, s := range e.Seeds.Tuning {
		tune[s] = true
	}
	for _, s := range append(append([]uint64{}, e.Seeds.Evaluation...), e.Seeds.Sensitivity...) {
		if tune[s] {
			return fmt.Errorf("seed %d is both a tuning and an evaluation seed", s)
		}
	}
	fams := map[string]bool{}
	for _, f := range e.Families {
		fams[f.Name] = true
	}
	ids := map[string]bool{}
	for _, sc := range e.Scenarios {
		if ids[sc.ID] {
			return fmt.Errorf("duplicate scenario %q", sc.ID)
		}
		ids[sc.ID] = true
		if !fams[sc.Family] {
			return fmt.Errorf("scenario %q: unknown family %q", sc.ID, sc.Family)
		}
		if _, ok := e.Mixes[sc.Mix]; !ok && sc.TracePath == "" {
			return fmt.Errorf("scenario %q: unknown mix %q", sc.ID, sc.Mix)
		}
	}
	for _, id := range append(append([]string{}, e.Sensitivity.Scenarios...), e.Quick.Scenarios...) {
		if !ids[id] {
			return fmt.Errorf("unknown scenario %q in sensitivity/quick", id)
		}
	}
	if e.Tuning.Configs < 1 {
		return fmt.Errorf("tuning.configs must be >= 1")
	}
	return nil
}

// Targets returns the SLO targets.
func (e *Experiment) Targets() (slo.Targets, error) {
	return slo.NewTargets(e.SLO.Default, e.SLO.Classes)
}

// Family returns a family by name.
func (e *Experiment) Family(name string) Family {
	for _, f := range e.Families {
		if f.Name == name {
			return f
		}
	}
	return Family{}
}

// Scenario returns a scenario by ID.
func (e *Experiment) Scenario(id string) *Scenario {
	for i := range e.Scenarios {
		if e.Scenarios[i].ID == id {
			return &e.Scenarios[i]
		}
	}
	return nil
}

// ScenariosOf returns a family's scenarios in file order.
func (e *Experiment) ScenariosOf(family string) []*Scenario {
	var out []*Scenario
	for i := range e.Scenarios {
		if e.Scenarios[i].Family == family {
			out = append(out, &e.Scenarios[i])
		}
	}
	return out
}

// mixFor returns the scenario's length mix with a variant applied.
func (e *Experiment) mixFor(sc *Scenario, v Variant) Mix {
	m := e.Mixes[sc.Mix]
	if v.OutputSigma > 0 {
		m.Output.Sigma = v.OutputSigma
	}
	return m
}

// refRate returns R for one model of a scenario (capacities in req/s keyed
// by class/mix).
func refRate(sm ScenarioModel, mix string, capacity map[string]float64) float64 {
	if sm.LoadReplicas > 0 {
		return sm.Load * sm.LoadReplicas * capacity[capKey(sm.Class, mix)]
	}
	r := 0.0
	for _, ic := range sm.Initial {
		r += float64(ic.Count) * capacity[capKey(ic.Class, mix)]
	}
	return sm.Load * r
}

func capKey(class, mix string) string { return class + "/" + mix }

// arrival builds the loadgen arrival with reference rate r.
func (a ArrivalSpec) arrival(r float64) loadgen.Arrival {
	switch a.Kind {
	case "mmpp":
		num, den := 0.0, 0.0
		for i, m := range a.RateMultipliers {
			num += m * a.MeanSojournS[i]
			den += a.MeanSojournS[i]
		}
		scale := r / (num / den)
		rates := make([]float64, len(a.RateMultipliers))
		for i, m := range a.RateMultipliers {
			rates[i] = m * scale
		}
		return loadgen.Arrival{Kind: "mmpp", Rates: rates, MeanSojournS: a.MeanSojournS}
	case "piecewise":
		rates := make([]float64, len(a.RateMultipliers))
		for i, m := range a.RateMultipliers {
			rates[i] = m * r
		}
		return loadgen.Arrival{Kind: "piecewise", Rates: rates, StartsS: a.StartsS}
	case "gamma":
		return loadgen.Arrival{Kind: "gamma", Rate: r, CV: a.CV}
	case "diurnal":
		return loadgen.Arrival{Kind: "diurnal", Rate: r, Amplitude: a.Amplitude, PeriodS: a.PeriodS, Phase: a.Phase}
	}
	return loadgen.Arrival{Kind: "poisson", Rate: r}
}

// peakRate returns the largest instantaneous rate of the process.
func peakRate(a loadgen.Arrival) float64 {
	switch a.Kind {
	case "mmpp", "piecewise":
		p := 0.0
		for _, x := range a.Rates {
			p = math.Max(p, x)
		}
		return p
	case "diurnal":
		return a.Rate * (1 + a.Amplitude)
	}
	return a.Rate
}

// traceConfig builds the loadgen configuration of a scenario under a variant.
func (e *Experiment) traceConfig(sc *Scenario, v Variant, capacity map[string]float64, durScale float64) loadgen.Config {
	mix := e.mixFor(sc, v)
	cfg := loadgen.Config{Name: sc.ID, DurationS: sc.DurationS * durScale}
	for _, sm := range sc.Models {
		cfg.Streams = append(cfg.Streams, loadgen.Stream{
			Model: sm.Model, Arrival: sm.Arrival.arrival(refRate(sm, sc.Mix, capacity)),
			Prompt: mix.Prompt, Output: mix.Output, MaxContext: mix.MaxContext, Classes: sm.Classes,
			PrefixGroups: sm.PrefixGroups, PrefixShare: sm.PrefixShare,
		})
	}
	return cfg
}

// classesFor returns the replica classes with a variant's warm-up scaling.
func classesFor(base map[string]sim.Class, v Variant) map[string]sim.Class {
	if v.WarmupScale <= 0 || v.WarmupScale == 1 {
		return base
	}
	out := make(map[string]sim.Class, len(base))
	for k, c := range base {
		c.Provision = scaleDur(c.Provision, v.WarmupScale)
		c.Load = scaleDur(c.Load, v.WarmupScale)
		out[k] = c
	}
	return out
}

func hashFiles(paths ...string) (string, error) {
	h := sha256.New()
	sorted := append([]string(nil), paths...)
	sort.Strings(sorted)
	for _, p := range sorted {
		data, err := os.ReadFile(p)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(h, "%s\n%d\n", filepath.ToSlash(p), len(data))
		h.Write(data)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
