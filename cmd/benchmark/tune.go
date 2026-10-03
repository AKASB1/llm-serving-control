package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/rng"
)

// Tuned is the frozen result of tuning (configs/tuned/tuned.json).
type Tuned struct {
	Commit        string                           `json:"commit"`
	ConfigSHA256  string                           `json:"config_sha256"`
	TuningSeeds   []uint64                         `json:"tuning_seeds"`
	Configs       int                              `json:"configs_per_policy"`
	DefaultRouter string                           `json:"default_router"`
	DefaultScaler string                           `json:"default_scaler"`
	Policies      map[string]controller.PolicySpec `json:"policies"`
}

// spec returns the tuned spec for family/label, or the policy's defaults.
func (t *Tuned) spec(family, label string) controller.PolicySpec {
	if s, ok := t.Policies[family+"/"+label]; ok {
		return s
	}
	return controller.PolicySpec{Name: label}
}

func defaultTuned() *Tuned {
	return &Tuned{DefaultRouter: "least_outstanding", DefaultScaler: "target_tracking", Policies: map[string]controller.PolicySpec{}}
}

// LoadTuned reads the frozen tuned configuration.
func LoadTuned(path string) (*Tuned, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t := &Tuned{}
	if err := json.Unmarshal(data, t); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// sampler draws one parameter object for a policy; the policy's family is
// passed for policies whose space depends on the scenario (static count,
// allocator shares).
type sampler func(r *rand.Rand, env *Env, family string) map[string]any

func logU(r *rand.Rand, lo, hi float64) float64 {
	return math.Exp(math.Log(lo) + r.Float64()*(math.Log(hi)-math.Log(lo)))
}
func u(r *rand.Rand, lo, hi float64) float64 { return lo + r.Float64()*(hi-lo) }
func intIn(r *rand.Rand, lo, hi int) int     { return lo + r.IntN(hi-lo+1) }
func round3(x float64) float64               { return math.Round(x*1000) / 1000 }

// searchSpaces maps policy name → sampler. Every parametrised policy gets
// the same number of configurations; config 0 is always its defaults.
var searchSpaces = map[string]sampler{
	"latency_aware": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		ex := 0.0
		if r.IntN(2) == 1 {
			ex = u(r, 0, 0.1)
		}
		return map[string]any{"tau_s": round3(logU(r, 1, 60)), "prior_s": round3(logU(r, 0.05, 2)), "explore": round3(ex)}
	},
	"queue_aware": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"run_weight": round3(u(r, 0, 4)), "kv_weight": round3(u(r, 0, 8))}
	},
	"queue_aware_corrected": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"run_weight": round3(u(r, 0, 4)), "kv_weight": round3(u(r, 0, 8)), "sent_weight": round3(u(r, 0.2, 3))}
	},
	"prefix_affinity": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"virtual_nodes": intIn(r, 20, 200), "load_factor": round3(u(r, 1.05, 2.5))}
	},
	"capacity_weighted": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		g := 0.0
		if r.IntN(2) == 1 {
			g = u(r, 0, 5)
		}
		return map[string]any{"prior_weight": round3(logU(r, 0.5, 50)), "sample_weight": round3(u(r, 0.05, 1)), "slo_gain": round3(g)}
	},
	"static": func(r *rand.Rand, env *Env, family string) map[string]any {
		hi := 1
		for _, sc := range env.exp.ScenariosOf(family) {
			for _, sm := range sc.Models {
				hi = max(hi, sm.Max)
			}
		}
		return map[string]any{"replicas": intIn(r, 1, hi)}
	},
	"threshold_cooldown": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		up := u(r, 16, 96)
		return map[string]any{"upper": round3(up), "lower": round3(u(r, 2, up/2)), "step": intIn(r, 1, 2),
			"cooldown_up_s": round3(u(r, 10, 120)), "cooldown_down_s": round3(u(r, 30, 600))}
	},
	"target_tracking": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"target": round3(u(r, 8, 96)), "tolerance": round3(u(r, 0, 0.3)), "stabilization_down_s": round3(u(r, 0, 600)),
			"stabilization_up_s": round3(u(r, 0, 60)), "max_step_up": intIn(r, 1, 8), "max_step_down": intIn(r, 1, 4)}
	},
	"slo_feedback": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"kp": round3(logU(r, 0.1, 2)), "ki": round3(u(r, 0, 0.1)), "dead_band": round3(u(r, 0, 0.3)),
			"up_threshold": round3(u(r, 0, 0.5)), "down_threshold": round3(u(r, 0.2, 0.9)), "integral_max": 20.0,
			"hold_s": round3(u(r, 0, 120)), "max_step_up": intIn(r, 1, 8), "max_step_down": intIn(r, 1, 2)}
	},
	"predictive": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"alpha": round3(u(r, 0.1, 0.9)), "beta": round3(u(r, 0, 0.5)), "target": round3(u(r, 8, 96)),
			"headroom": round3(u(r, 1, 1.5)), "prior_e2e_s": round3(logU(r, 1, 20)), "max_step_down": intIn(r, 1, 4)}
	},
	"static_partition": func(r *rand.Rand, env *Env, family string) map[string]any {
		var models []string
		for _, sc := range env.exp.ScenariosOf(family) {
			for _, sm := range sc.Models {
				models = append(models, sm.Model)
			}
		}
		sort.Strings(models)
		w := make([]float64, len(models))
		tot := 0.0
		for i := range w {
			w[i] = r.ExpFloat64() // Dirichlet(1, …, 1)
			tot += w[i]
		}
		shares := map[string]any{}
		for i, m := range models {
			shares[m] = round3(w[i] / tot)
		}
		return map[string]any{"shares": shares}
	},
	"marginal_gain": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		mg := 0.0
		if r.IntN(2) == 1 {
			mg = u(r, 0, 0.01)
		}
		return map[string]any{"wait_fraction": round3(u(r, 0.1, 1)), "prior_service_s": round3(logU(r, 1, 20)), "min_gain": mg}
	},
	"queue_cap": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		return map[string]any{"max_inflight_per_replica": intIn(r, 16, 256), "max_queue": intIn(r, 0, 1024)}
	},
	"predicted_ttft_shed": func(r *rand.Rand, _ *Env, _ string) map[string]any {
		fi := u(r, 0.5, 3)
		return map[string]any{"factors": map[string]any{"interactive": round3(fi), "batch": round3(u(r, 0.1, fi))},
			"default_factor": round3(fi), "prefill_share": round3(u(r, 0.2, 1))}
	},
}

// policyName maps a family label to the policy it tunes ("" = not tuned).
func policyName(family, label string) string {
	switch {
	case family == "scaling" && label == "static_tuned":
		return "static"
	case family == "scaling" && (label == "static_min" || label == "static_peak_oracle"):
		return ""
	}
	return label
}

// candidates returns the configurations tried for one policy: its defaults
// (or, for static, a first random draw), then random draws up to the budget.
// Parameterless policies get exactly one (empty) configuration.
func (env *Env) candidates(family, name string) []controller.PolicySpec {
	s, ok := searchSpaces[name]
	if !ok {
		return []controller.PolicySpec{{Name: name}}
	}
	r := rng.Stream(env.exp.Tuning.SearchSeed, "tune/"+family+"/"+name)
	var out []controller.PolicySpec
	seen := map[string]bool{}
	if name != "static" && name != "static_partition" {
		out = append(out, controller.PolicySpec{Name: name})
		seen["{}"] = true
	}
	for tries := 0; len(out) < env.exp.Tuning.Configs && tries < 1000; tries++ {
		raw, _ := json.Marshal(s(r, env, family))
		if seen[string(raw)] {
			continue
		}
		seen[string(raw)] = true
		out = append(out, controller.PolicySpec{Name: name, Params: raw})
	}
	return out
}

type tuneResult struct {
	Family, Label string
	Spec          controller.PolicySpec
	MeanJ         float64
	Runs          int
}

// tune runs the tuning protocol family by family, freezing each family's
// choices before the next family uses them as defaults.
func (env *Env) tune(workers int, commit, cfgHash string) (*Tuned, []tuneResult, error) {
	t := defaultTuned()
	t.Commit, t.ConfigSHA256 = commit, cfgHash
	t.TuningSeeds, t.Configs = env.exp.Seeds.Tuning, env.exp.Tuning.Configs
	env.tuned = t
	var log []tuneResult
	for _, fam := range []string{"routing", "scaling", "capacity", "overload"} {
		f := env.exp.Family(fam)
		scs := env.exp.ScenariosOf(fam)
		if len(scs) == 0 {
			continue
		}
		best := map[string]tuneResult{}
		for _, label := range f.Policies {
			name := policyName(fam, label)
			if name == "" {
				continue
			}
			cands := env.candidates(fam, name)
			var jobs []Job
			for ci, c := range cands {
				for _, sc := range scs {
					ps, err := env.policies(sc, label)
					if err != nil {
						return nil, nil, err
					}
					setPolicy(&ps, fam, sc, c)
					for _, seed := range env.exp.Seeds.Tuning {
						jobs = append(jobs, Job{Scenario: sc, Variant: Variant{ID: "base"}, Label: fmt.Sprintf("%s#%d", label, ci), Seed: seed, Policy: ps})
					}
				}
			}
			rows, err := env.runAll(jobs, workers)
			if err != nil {
				return nil, nil, err
			}
			per := len(scs) * len(env.exp.Seeds.Tuning)
			for ci, c := range cands {
				sum := 0.0
				for _, r := range rows[ci*per : (ci+1)*per] {
					sum += r.Summary.J
				}
				res := tuneResult{Family: fam, Label: label, Spec: c, MeanJ: sum / float64(per), Runs: per}
				log = append(log, res)
				if b, ok := best[label]; !ok || res.MeanJ < b.MeanJ {
					best[label] = res
				}
			}
			t.Policies[fam+"/"+name] = best[label].Spec
			slog.Info("tuned", "family", fam, "policy", label, "configs", len(cands), "best_mean_J", fmt.Sprintf("%.4f", best[label].MeanJ))
		}
		switch fam {
		case "routing":
			t.DefaultRouter = pickBest(best, f.Oracles, nil)
		case "scaling":
			// Allocation needs time-varying desired counts: static variants are excluded.
			t.DefaultScaler = policyName(fam, pickBest(best, f.Oracles, func(l string) bool { return !strings.HasPrefix(l, "static") }))
		}
	}
	return t, log, nil
}

// setPolicy puts candidate c into the slot the family varies.
func setPolicy(ps *policySet, family string, sc *Scenario, c controller.PolicySpec) {
	switch family {
	case "routing":
		ps.Router = c
	case "scaling":
		for _, sm := range sc.Models {
			ps.Scalers[sm.Model] = c
		}
	case "capacity":
		ps.Allocator = &c
	case "overload":
		ps.Admission = c
	}
}

// pickBest returns the label with the lowest tuning J, excluding oracles and
// labels rejected by keep; ties by label.
func pickBest(best map[string]tuneResult, oracles []string, keep func(string) bool) string {
	labels := make([]string, 0, len(best))
	for l := range best {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	out, bestJ := "", math.Inf(1)
	for _, l := range labels {
		if contains(oracles, l) || (keep != nil && !keep(l)) {
			continue
		}
		if best[l].MeanJ < bestJ {
			out, bestJ = l, best[l].MeanJ
		}
	}
	return out
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

func writeTuned(dir string, t *Tuned, log []tuneResult) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "tuned.json"), append(data, '\n'), 0o644); err != nil {
		return err
	}
	var b strings.Builder
	b.WriteString("family,policy,config,runs,mean_J,params\n")
	for i, r := range log {
		_ = i
		params := strings.ReplaceAll(string(r.Spec.Params), "\"", "'")
		if params == "" {
			params = "{}"
		}
		fmt.Fprintf(&b, "%s,%s,%s,%d,%.6f,\"%s\"\n", r.Family, r.Label, r.Spec.Name, r.Runs, r.MeanJ, params)
	}
	return os.WriteFile(filepath.Join(dir, "tuning_log.csv"), []byte(b.String()), 0o644)
}
