package main

// Reweighting: the rankings of the frozen configurations under other
// objective weights, recomputed from committed per-run metrics. Nothing is
// simulated or tuned again; J' = alpha·j_ttft + beta·j_tpot + gamma·j_gpu +
// delta·violation_rate is rebuilt from the runs.csv columns.

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
)

// ReweightConfig fixes the weight vectors and the GPU-cost cap rule before
// anything is computed (configs/reweight.json).
type ReweightConfig struct {
	Runs    string        `json:"runs"`    // committed per-run metrics, e.g. benchmarks/results/runs.csv
	Variant string        `json:"variant"` // rows of this variant only
	Weights []NamedWeight `json:"weights"` // the first entry is the reference ranking
	GPUCap  struct {
		Family  string      `json:"family"`  // family whose cap view is computed
		Weights NamedWeight `json:"weights"` // objective among the feasible policies
	} `json:"gpu_cap"`
}

// NamedWeight is one weight vector of the objective.
type NamedWeight struct {
	Name string `json:"name"`
	metrics.Weights
}

type runMetrics struct {
	scenario, family, policy       string
	seed                           uint64
	j, jTTFT, jTPOT, jGPU, viol, g float64
}

func (r runMetrics) objective(w metrics.Weights) float64 {
	return w.Alpha*r.jTTFT + w.Beta*r.jTPOT + w.Gamma*r.jGPU + w.Delta*r.viol
}

// LoadReweightConfig reads and checks a reweighting configuration.
func LoadReweightConfig(path string) (*ReweightConfig, []byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var c ReweightConfig
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	if c.Runs == "" || c.Variant == "" || len(c.Weights) == 0 {
		return nil, nil, fmt.Errorf("%s: runs, variant and at least one weight vector are required", path)
	}
	seen := map[string]bool{}
	for _, w := range c.Weights {
		if w.Name == "" || seen[w.Name] {
			return nil, nil, fmt.Errorf("%s: weight names must be unique and non-empty", path)
		}
		seen[w.Name] = true
	}
	return &c, raw, nil
}

func readRunMetrics(path, variant string) ([]runMetrics, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(recs) < 2 {
		return nil, fmt.Errorf("%s: no rows", path)
	}
	col := map[string]int{}
	for i, h := range recs[0] {
		col[h] = i
	}
	for _, h := range []string{"scenario", "variant", "family", "policy", "seed", "J", "j_ttft", "j_tpot", "j_gpu", "violation_rate", "gpu_hours"} {
		if _, ok := col[h]; !ok {
			return nil, fmt.Errorf("%s: missing column %q", path, h)
		}
	}
	var out []runMetrics
	for li, rec := range recs[1:] {
		if rec[col["variant"]] != variant {
			continue
		}
		num := func(h string) (float64, error) {
			v, err := strconv.ParseFloat(rec[col[h]], 64)
			if err != nil {
				return 0, fmt.Errorf("%s line %d: %s: %w", path, li+2, h, err)
			}
			return v, nil
		}
		seed, err := strconv.ParseUint(rec[col["seed"]], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%s line %d: seed: %w", path, li+2, err)
		}
		r := runMetrics{scenario: rec[col["scenario"]], family: rec[col["family"]], policy: rec[col["policy"]], seed: seed}
		for _, p := range []struct {
			h string
			v *float64
		}{{"J", &r.j}, {"j_ttft", &r.jTTFT}, {"j_tpot", &r.jTPOT}, {"j_gpu", &r.jGPU}, {"violation_rate", &r.viol}, {"gpu_hours", &r.g}} {
			if *p.v, err = num(p.h); err != nil {
				return nil, err
			}
		}
		out = append(out, r)
	}
	return out, nil
}

type reweightKey struct{ scenario, policy string }

// policyStats is the paired comparison of one policy with its family baseline.
type policyStats struct {
	mean, half         float64
	diff, diffHalf     float64
	w, t, l            int
	verdict            string
	oracle, isBaseline bool
}

// compare reproduces the published verdict rule (output.go) for objective f.
func compare(byKey map[reweightKey][]runMetrics, sc, policy, base string, band float64, f func(runMetrics) float64) policyStats {
	var st policyStats
	var xs []float64
	for _, r := range byKey[reweightKey{sc, policy}] {
		xs = append(xs, f(r))
	}
	st.mean, st.half = meanCI(xs)
	st.isBaseline = policy == base
	st.verdict, st.diff, st.diffHalf = "", math.NaN(), math.NaN()
	if st.isBaseline || base == "" {
		return st
	}
	bj := map[uint64]float64{}
	for _, r := range byKey[reweightKey{sc, base}] {
		bj[r.seed] = f(r)
	}
	var diffs, pj, bjs []float64
	for _, r := range byKey[reweightKey{sc, policy}] {
		if v, ok := bj[r.seed]; ok {
			diffs = append(diffs, f(r)-v)
			pj = append(pj, f(r))
			bjs = append(bjs, v)
		}
	}
	st.diff, st.diffHalf = meanCI(diffs)
	st.w, st.t, st.l = wtl(pj, bjs, band)
	st.verdict = "no_difference"
	switch {
	case st.diff+st.diffHalf < 0:
		st.verdict = "better"
	case st.diff-st.diffHalf > 0:
		st.verdict = "worse"
	}
	return st
}

// runReweight writes rankings.csv, summary.csv, gpu_cap.csv and manifest.json
// into out. It refuses to run if J at the reference weights does not
// reproduce the committed J column (to the CSV's printed precision).
func runReweight(exp *Experiment, cfgPath, out string) error {
	cfg, raw, err := LoadReweightConfig(cfgPath)
	if err != nil {
		return err
	}
	rows, err := readRunMetrics(cfg.Runs, cfg.Variant)
	if err != nil {
		return err
	}
	src, err := os.ReadFile(cfg.Runs)
	if err != nil {
		return err
	}
	var maxDiff float64
	for _, r := range rows {
		d := math.Abs(r.objective(exp.Weights) - r.j)
		maxDiff = math.Max(maxDiff, d/math.Max(1, math.Abs(r.j)))
	}
	if maxDiff > reproduceTol {
		return fmt.Errorf("J at the experiment weights differs from the committed J by %g (relative) > %g", maxDiff, reproduceTol)
	}
	if ref := cfg.Weights[0]; ref.Weights != exp.Weights {
		return fmt.Errorf("the first weight vector %q is the reference and must equal the experiment weights %+v", ref.Name, exp.Weights)
	}
	if len(rows) == 0 {
		return fmt.Errorf("%s: no rows of variant %q", cfg.Runs, cfg.Variant)
	}
	byKey := map[reweightKey][]runMetrics{}
	var scenarios []string
	famOf := map[string]string{}
	for _, r := range rows {
		k := reweightKey{r.scenario, r.policy}
		if _, ok := famOf[r.scenario]; !ok {
			famOf[r.scenario] = r.family
			scenarios = append(scenarios, r.scenario)
		}
		fam := exp.Family(r.family)
		if fam.Name == "" {
			return fmt.Errorf("%s: family %q of scenario %s is not in the experiment configuration", cfg.Runs, r.family, r.scenario)
		}
		if !contains(fam.Policies, r.policy) {
			return fmt.Errorf("%s: policy %q of scenario %s is not in family %q", cfg.Runs, r.policy, r.scenario, fam.Name)
		}
		byKey[k] = append(byKey[k], r)
	}
	if cfg.GPUCap.Family != "" && exp.Family(cfg.GPUCap.Family).Name == "" {
		return fmt.Errorf("gpu_cap family %q is not in the experiment configuration", cfg.GPUCap.Family)
	}
	var rk, sm strings.Builder
	rk.WriteString("weights,alpha,beta,gamma,delta,scenario,family,policy,oracle,n,mean_J,ci95,rank,baseline,mean_diff_J,ci95_diff,wins,ties,losses,verdict\n")
	sm.WriteString("weights,scenario,family,policies,kendall_tau_vs_reference,top_policy,top_policy_reference,top_same,verdicts_same,verdicts_total\n")
	type ref struct {
		means    []float64
		verdicts map[string]string
		top      string
	}
	refs := map[string]ref{}
	for wi, nw := range cfg.Weights {
		w := nw.Weights
		f := func(r runMetrics) float64 { return r.objective(w) }
		for _, sc := range scenarios {
			fam := exp.Family(famOf[sc])
			var labels []string
			for _, p := range fam.Policies {
				if len(byKey[reweightKey{sc, p}]) > 0 {
					labels = append(labels, p)
				}
			}
			var deploy []string
			var means []float64
			stats := map[string]policyStats{}
			for _, p := range labels {
				st := compare(byKey, sc, p, fam.Baseline, exp.TieBand, f)
				st.oracle = contains(fam.Oracles, p)
				stats[p] = st
				if !st.oracle {
					deploy = append(deploy, p)
					means = append(means, st.mean)
				}
			}
			rank := map[string]int{}
			for i, idx := range rankOrder(means) {
				rank[deploy[idx]] = i + 1
			}
			for _, p := range labels {
				st := stats[p]
				rs := "-"
				if r, ok := rank[p]; ok {
					rs = strconv.Itoa(r)
				}
				n := len(byKey[reweightKey{sc, p}])
				fmt.Fprintf(&rk, "%s,%s,%s,%s,%s,%s,%s,%s,%t,%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%s\n",
					nw.Name, ff(w.Alpha), ff(w.Beta), ff(w.Gamma), ff(w.Delta), sc, fam.Name, p, st.oracle, n,
					ff(st.mean), ff(st.half), rs, fam.Baseline, ff(st.diff), ff(st.diffHalf), st.w, st.t, st.l, dash(st.verdict))
			}
			top := ""
			if len(means) > 0 {
				top = deploy[rankOrder(means)[0]]
			}
			verdicts := map[string]string{}
			for _, p := range deploy {
				if v := stats[p].verdict; v != "" {
					verdicts[p] = v
				}
			}
			if wi == 0 {
				refs[sc] = ref{means: means, verdicts: verdicts, top: top}
			}
			r := refs[sc]
			same, total := 0, 0
			for p, v := range verdicts {
				total++
				if r.verdicts[p] == v {
					same++
				}
			}
			fmt.Fprintf(&sm, "%s,%s,%s,%d,%s,%s,%s,%t,%d,%d\n", nw.Name, sc, fam.Name, len(deploy), ff(kendallTau(r.means, means)), top, r.top, top != "" && top == r.top, same, total)
		}
	}
	var gc strings.Builder
	gc.WriteString("scenario,policy,oracle,n,mean_gpu_hours,cap_gpu_hours,feasible,weights,mean_J,ci95,rank\n")
	if cfg.GPUCap.Family != "" {
		fam := exp.Family(cfg.GPUCap.Family)
		w := cfg.GPUCap.Weights.Weights
		f := func(r runMetrics) float64 { return r.objective(w) }
		g := func(r runMetrics) float64 { return r.g }
		for _, sc := range scenarios {
			if famOf[sc] != fam.Name {
				continue
			}
			capGPU := compare(byKey, sc, fam.Baseline, fam.Baseline, exp.TieBand, g).mean
			type cand struct {
				p        string
				gh, j, h float64
				oracle   bool
				feasible bool
			}
			var cs []cand
			var feas []int
			var feasJ []float64
			for _, p := range fam.Policies {
				if len(byKey[reweightKey{sc, p}]) == 0 {
					continue
				}
				gh := compare(byKey, sc, p, fam.Baseline, exp.TieBand, g).mean
				st := compare(byKey, sc, p, fam.Baseline, exp.TieBand, f)
				c := cand{p: p, gh: gh, j: st.mean, h: st.half, oracle: contains(fam.Oracles, p), feasible: gh <= capGPU}
				if c.feasible && !c.oracle {
					feas = append(feas, len(cs))
					feasJ = append(feasJ, c.j)
				}
				cs = append(cs, c)
			}
			rank := map[int]int{}
			for i, idx := range rankOrder(feasJ) {
				rank[feas[idx]] = i + 1
			}
			for i, c := range cs {
				rs := "-"
				if r, ok := rank[i]; ok {
					rs = strconv.Itoa(r)
				}
				fmt.Fprintf(&gc, "%s,%s,%t,%d,%s,%s,%t,%s,%s,%s,%s\n", sc, c.p, c.oracle, len(byKey[reweightKey{sc, c.p}]),
					ff(c.gh), ff(capGPU), c.feasible, cfg.GPUCap.Weights.Name, ff(c.j), ff(c.h), rs)
			}
		}
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	for name, body := range map[string]string{"rankings.csv": rk.String(), "summary.csv": sm.String(), "gpu_cap.csv": gc.String()} {
		if err := os.WriteFile(filepath.Join(out, name), []byte(body), 0o644); err != nil {
			return err
		}
	}
	commit, dirty := gitCommit()
	cs, ss := sha256.Sum256(raw), sha256.Sum256(src)
	man := map[string]any{
		"label":               "simulated, assumed parameters; recomputed from committed per-run metrics (no new runs)",
		"commit":              commit,
		"dirty":               dirty,
		"reweight_config":     filepath.ToSlash(cfgPath),
		"reweight_config_sha": hex.EncodeToString(cs[:]),
		"runs":                filepath.ToSlash(cfg.Runs),
		"runs_sha256":         hex.EncodeToString(ss[:]),
		"variant":             cfg.Variant,
		"rows":                len(rows),
		"reference_weights":   exp.Weights,
		"max_rel_diff_J":      maxDiff,
		"reproduce_tolerance": reproduceTol,
		"go_version":          runtime.Version(),
	}
	data, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "manifest.json"), append(data, '\n'), 0o644)
}

// reproduceTol bounds |J(reference weights) − committed J| / max(1, |J|):
// runs.csv prints 8 significant digits per column, so rebuilt J differs from
// the printed J by a few units in the 8th digit at most.
const reproduceTol = 1e-6

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
