package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
)

// aggregated metrics: column name in runs.csv → extractor.
var aggMetrics = []struct {
	name string
	get  func(s metrics.Summary) float64
}{
	{"J", func(s metrics.Summary) float64 { return s.J }},
	{"violation_rate", func(s metrics.Summary) float64 { return s.ViolationRate }},
	{"goodput", func(s metrics.Summary) float64 { return s.Goodput }},
	{"ttft_p95", func(s metrics.Summary) float64 { return s.TTFTP95 }},
	{"ttft_p99", func(s metrics.Summary) float64 { return s.TTFTP99 }},
	{"tpot_p95", func(s metrics.Summary) float64 { return s.TPOTP95 }},
	{"e2e_p95", func(s metrics.Summary) float64 { return s.E2EP95 }},
	{"gpu_hours", func(s metrics.Summary) float64 { return s.GPUHours }},
	{"utilization", func(s metrics.Summary) float64 { return s.Utilization }},
	{"kv_occupancy", func(s metrics.Summary) float64 { return s.KVOccupancy }},
	{"rejected_rate", func(s metrics.Summary) float64 { return ratio(s.Rejected, s.Requests) }},
	{"failed_rate", func(s metrics.Summary) float64 { return ratio(s.Failed, s.Requests) }},
	{"timed_out_rate", func(s metrics.Summary) float64 { return ratio(s.TimedOut, s.Requests) }},
	{"unfinished_rate", func(s metrics.Summary) float64 { return ratio(s.Unfinished, s.Requests) }},
	{"retries", func(s metrics.Summary) float64 { return float64(s.Retries) }},
	{"preemptions", func(s metrics.Summary) float64 { return float64(s.Preemptions) }},
	{"scale_actions", func(s metrics.Summary) float64 { return float64(s.ScaleActions) }},
	{"router_q_p95", func(s metrics.Summary) float64 { return s.RouterQP95 }},
	{"replica_q_p95", func(s metrics.Summary) float64 { return s.ReplicaQP95 }},
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func ff(x float64) string {
	if math.IsNaN(x) {
		return "NaN"
	}
	return strconv.FormatFloat(x, 'g', 8, 64)
}

type groupKey struct{ scenario, variant, policy string }

type group struct {
	family string
	seeds  []uint64
	rows   []Row
}

// groups collects rows by (scenario, variant, policy) in first-seen order.
func groups(rows []Row) ([]groupKey, map[groupKey]*group) {
	var order []groupKey
	m := map[groupKey]*group{}
	for _, r := range rows {
		k := groupKey{r.Scenario, r.Variant, r.Policy}
		g := m[k]
		if g == nil {
			g = &group{family: r.Family}
			m[k] = g
			order = append(order, k)
		}
		g.seeds = append(g.seeds, r.Seed)
		g.rows = append(g.rows, r)
	}
	return order, m
}

func writeResults(dir string, exp *Experiment, rows []Row, man manifest) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// runs.csv
	var b strings.Builder
	b.WriteString("scenario,variant,family,policy,seed,trace_sha,ejections," + strings.Join(metrics.Columns(), ",") + "\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "%s,%s,%s,%s,%d,%s,%d,%s\n", r.Scenario, r.Variant, r.Family, r.Policy, r.Seed, r.TraceSHA, r.Ejections, strings.Join(r.Summary.Values(), ","))
	}
	if err := os.WriteFile(filepath.Join(dir, "runs.csv"), []byte(b.String()), 0o644); err != nil {
		return err
	}
	order, gm := groups(rows)
	// aggregates.csv
	b.Reset()
	b.WriteString("scenario,variant,family,policy,oracle,n,metric,mean,ci95\n")
	for _, k := range order {
		g := gm[k]
		oracle := contains(exp.Family(g.family).Oracles, k.policy)
		for _, m := range aggMetrics {
			xs := make([]float64, 0, len(g.rows))
			for _, r := range g.rows {
				if v := m.get(r.Summary); !math.IsNaN(v) {
					xs = append(xs, v)
				}
			}
			mean, half := meanCI(xs)
			fmt.Fprintf(&b, "%s,%s,%s,%s,%t,%d,%s,%s,%s\n", k.scenario, k.variant, g.family, k.policy, oracle, len(xs), m.name, ff(mean), ff(half))
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "aggregates.csv"), []byte(b.String()), 0o644); err != nil {
		return err
	}
	// paired.csv: J differences against the family baseline on common seeds.
	type verdictKey struct{ scenario, variant, policy string }
	verdicts := map[verdictKey]string{}
	b.Reset()
	b.WriteString("scenario,variant,family,policy,baseline,n,mean_diff_J,ci95,wins,ties,losses,verdict\n")
	for _, k := range order {
		g := gm[k]
		base := exp.Family(g.family).Baseline
		if base == "" || k.policy == base {
			continue
		}
		bg := gm[groupKey{k.scenario, k.variant, base}]
		if bg == nil {
			continue
		}
		bj := map[uint64]float64{}
		for _, r := range bg.rows {
			bj[r.Seed] = r.Summary.J
		}
		var diffs, pj, bjs []float64
		for _, r := range g.rows {
			if v, ok := bj[r.Seed]; ok {
				diffs = append(diffs, r.Summary.J-v)
				pj = append(pj, r.Summary.J)
				bjs = append(bjs, v)
			}
		}
		mean, half := meanCI(diffs)
		w, t, l := wtl(pj, bjs, exp.TieBand)
		verdict := "no_difference"
		switch {
		case mean+half < 0:
			verdict = "better"
		case mean-half > 0:
			verdict = "worse"
		}
		verdicts[verdictKey{k.scenario, k.variant, k.policy}] = verdict
		fmt.Fprintf(&b, "%s,%s,%s,%s,%s,%d,%s,%s,%d,%d,%d,%s\n", k.scenario, k.variant, g.family, k.policy, base, len(diffs), ff(mean), ff(half), w, t, l, verdict)
	}
	if err := os.WriteFile(filepath.Join(dir, "paired.csv"), []byte(b.String()), 0o644); err != nil {
		return err
	}
	// sensitivity.csv: does the ranking (and each policy's verdict against the
	// baseline) survive each assumption change?
	b.Reset()
	b.WriteString("scenario,variant,policies,kendall_tau,top_policy,top_policy_base,top_same,verdicts_same,verdicts_total\n")
	meanJ := func(sc, v, p string) float64 {
		g := gm[groupKey{sc, v, p}]
		if g == nil {
			return math.NaN()
		}
		var xs []float64
		for _, r := range g.rows {
			xs = append(xs, r.Summary.J)
		}
		m, _ := meanCI(xs)
		return m
	}
	for _, id := range exp.Sensitivity.Scenarios {
		sc := exp.Scenario(id)
		f := exp.Family(sc.Family)
		var labels []string
		for _, l := range f.Policies {
			if !contains(f.Oracles, l) {
				labels = append(labels, l)
			}
		}
		for _, v := range exp.Sensitivity.Variants {
			var x, y []float64
			for _, l := range labels {
				x = append(x, meanJ(id, "base", l))
				y = append(y, meanJ(id, v.ID, l))
			}
			if anyNaN(x) || anyNaN(y) {
				continue
			}
			tb, tv := labels[rankOrder(x)[0]], labels[rankOrder(y)[0]]
			same, total := 0, 0
			for _, l := range labels {
				if l == f.Baseline {
					continue
				}
				vb, okb := verdicts[verdictKey{id, "base", l}]
				vv, okv := verdicts[verdictKey{id, v.ID, l}]
				if okb && okv {
					total++
					if vb == vv {
						same++
					}
				}
			}
			fmt.Fprintf(&b, "%s,%s,%d,%s,%s,%s,%t,%d,%d\n", id, v.ID, len(labels), ff(kendallTau(x, y)), tv, tb, tb == tv, same, total)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "sensitivity.csv"), []byte(b.String()), 0o644); err != nil {
		return err
	}
	data, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(data, '\n'), 0o644)
}

func anyNaN(xs []float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) {
			return true
		}
	}
	return false
}
