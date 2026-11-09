package main

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func writeReweightConfig(t *testing.T, runs string, weights string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "reweight.json")
	body := `{"runs": "` + filepath.ToSlash(runs) + `", "variant": "base", "weights": [` + weights + `],
	  "gpu_cap": {"family": "scaling", "weights": {"name": "no_gpu", "alpha": 1, "beta": 1, "gamma": 0, "delta": 5}}}`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func readCSVFile(t *testing.T, path string) []map[string]string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]string
	for _, r := range recs[1:] {
		m := map[string]string{}
		for i, h := range recs[0] {
			m[h] = r[i]
		}
		out = append(out, m)
	}
	return out
}

// At the experiment's own weights the recomputation reproduces the committed
// per-policy means (aggregates.csv) and verdicts (paired.csv).
func TestReweightReproducesCommittedResults(t *testing.T) {
	exp, _, err := LoadExperiment("../../configs/experiment.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg := writeReweightConfig(t, "../../benchmarks/results/runs.csv", `{"name": "default", "alpha": 1, "beta": 1, "gamma": 1, "delta": 5}`)
	out := t.TempDir()
	if err := runReweight(exp, cfg, out); err != nil {
		t.Fatal(err)
	}
	means := map[string]float64{}
	for _, r := range readCSVFile(t, "../../benchmarks/results/aggregates.csv") {
		if r["variant"] == "base" && r["metric"] == "J" {
			v, _ := strconv.ParseFloat(r["mean"], 64)
			means[r["scenario"]+"/"+r["policy"]] = v
		}
	}
	paired := map[string]map[string]string{}
	for _, r := range readCSVFile(t, "../../benchmarks/results/paired.csv") {
		if r["variant"] == "base" {
			paired[r["scenario"]+"/"+r["policy"]] = r
		}
	}
	rows := readCSVFile(t, filepath.Join(out, "rankings.csv"))
	if len(rows) != len(means) {
		t.Fatalf("%d ranking rows, %d committed means", len(rows), len(means))
	}
	for _, r := range rows {
		k := r["scenario"] + "/" + r["policy"]
		got, _ := strconv.ParseFloat(r["mean_J"], 64)
		if want := means[k]; math.Abs(got-want) > reproduceTol*math.Max(1, math.Abs(want)) {
			t.Errorf("%s: mean J %v, committed %v", k, got, want)
		}
		pr, ok := paired[k]
		if !ok {
			continue
		}
		for _, c := range [][2]string{{"verdict", "verdict"}, {"wins", "wins"}, {"ties", "ties"}, {"losses", "losses"}} {
			if r[c[0]] != pr[c[1]] {
				t.Errorf("%s: %s %s, committed %s", k, c[0], r[c[0]], pr[c[1]])
			}
		}
		for _, c := range [][2]string{{"mean_diff_J", "mean_diff_J"}, {"ci95_diff", "ci95"}} {
			got, _ := strconv.ParseFloat(r[c[0]], 64)
			want, _ := strconv.ParseFloat(pr[c[1]], 64)
			if math.Abs(got-want) > 1e-5*math.Max(1, math.Abs(want)) {
				t.Errorf("%s: %s %v, committed %v", k, c[0], got, want)
			}
		}
	}
	if n := len(paired); n == 0 {
		t.Fatal("no paired rows")
	}
	for _, r := range readCSVFile(t, filepath.Join(out, "summary.csv")) {
		if r["kendall_tau_vs_reference"] != "1" || r["top_same"] != "true" {
			t.Errorf("reference ranking must agree with itself: %v", r)
		}
	}
}

// The run refuses inputs that would give a silently wrong or empty result:
// weights that do not rebuild the committed J, a first (reference) vector
// other than the experiment's weights, an unknown configuration key, and runs
// whose family is not in the experiment configuration.
func TestReweightRefusesBadInputs(t *testing.T) {
	load := func(path string) *Experiment {
		exp, _, err := LoadExperiment(path)
		if err != nil {
			t.Fatal(err)
		}
		return exp
	}
	def := `{"name": "default", "alpha": 1, "beta": 1, "gamma": 1, "delta": 5}`
	runs := "../../benchmarks/results/runs.csv"
	exp := load("../../configs/experiment.json")
	exp.Weights.Delta = 1
	if err := runReweight(exp, writeReweightConfig(t, runs, def), t.TempDir()); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Errorf("mismatched experiment weights: got %v", err)
	}
	exp = load("../../configs/experiment.json")
	cfg := writeReweightConfig(t, runs, `{"name": "equal", "alpha": 1, "beta": 1, "gamma": 1, "delta": 1}, `+def)
	if err := runReweight(exp, cfg, t.TempDir()); err == nil || !strings.Contains(err.Error(), "reference") {
		t.Errorf("first vector is not the experiment's: got %v", err)
	}
	cfg = writeReweightConfig(t, runs, `{"name": "default", "alpha": 1, "beta": 1, "gama": 1, "delta": 5}`)
	if err := runReweight(exp, cfg, t.TempDir()); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Errorf("misspelt weight key: got %v", err)
	}
	if err := runReweight(load("../../configs/replay.json"), writeReweightConfig(t, runs, def), t.TempDir()); err == nil || !strings.Contains(err.Error(), "not in the experiment configuration") {
		t.Errorf("runs of families the experiment does not define: got %v", err)
	}
}

// A hand-built case: under gamma = 3 the cheap policy overtakes the fast one;
// a noisy policy gets one win, one tie, and one loss and no verdict; the oracle
// is compared but never ranked; the GPU cap keeps only policies at or below
// the baseline's GPU-hours.
func TestReweightHandCase(t *testing.T) {
	dir := t.TempDir()
	runs := filepath.Join(dir, "runs.csv")
	// J = j_ttft + j_tpot + j_gpu + 5·violation_rate at the reference weights.
	body := "scenario,variant,family,policy,seed,J,j_ttft,j_tpot,j_gpu,violation_rate,gpu_hours\n" +
		"s-x,base,scaling,threshold_cooldown,1,3,1,1,1,0,10\n" +
		"s-x,base,scaling,threshold_cooldown,2,3.1,1.1,1,1,0,10\n" +
		"s-x,base,scaling,threshold_cooldown,3,2.9,0.9,1,1,0,10\n" +
		"s-x,base,scaling,predictive,1,2.6,0.5,0.5,1.6,0,16\n" +
		"s-x,base,scaling,predictive,2,2.7,0.6,0.5,1.6,0,16\n" +
		"s-x,base,scaling,predictive,3,2.5,0.4,0.5,1.6,0,16\n" +
		"s-x,base,scaling,static_min,1,3.4,1.5,1.2,0.7,0,7\n" +
		"s-x,base,scaling,static_min,2,3.5,1.6,1.2,0.7,0,7\n" +
		"s-x,base,scaling,static_min,3,3.3,1.4,1.2,0.7,0,7\n" +
		"s-x,base,scaling,slo_feedback,1,3.02,1.02,1,1,0,9\n" +
		"s-x,base,scaling,slo_feedback,2,2.9,0.9,1,1,0,9\n" +
		"s-x,base,scaling,slo_feedback,3,3.2,1.2,1,1,0,9\n" +
		"s-x,base,scaling,static_peak_oracle,1,2,0.4,0.4,1.2,0,12\n" +
		"s-x,base,scaling,static_peak_oracle,2,2,0.4,0.4,1.2,0,12\n" +
		"s-x,base,scaling,static_peak_oracle,3,2,0.4,0.4,1.2,0,12\n"
	if err := os.WriteFile(runs, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	exp := &Experiment{TieBand: 0.01, Families: []Family{{Name: "scaling", Baseline: "threshold_cooldown",
		Policies: []string{"static_min", "threshold_cooldown", "predictive", "slo_feedback", "static_peak_oracle"},
		Oracles:  []string{"static_peak_oracle"}}}}
	exp.Weights.Alpha, exp.Weights.Beta, exp.Weights.Gamma, exp.Weights.Delta = 1, 1, 1, 5
	cfg := writeReweightConfig(t, runs, `{"name": "default", "alpha": 1, "beta": 1, "gamma": 1, "delta": 5},
	  {"name": "gpu_x3", "alpha": 1, "beta": 1, "gamma": 3, "delta": 5}`)
	out := filepath.Join(dir, "out")
	if err := runReweight(exp, cfg, out); err != nil {
		t.Fatal(err)
	}
	rank := map[string]string{}
	for _, r := range readCSVFile(t, filepath.Join(out, "rankings.csv")) {
		rank[r["weights"]+"/"+r["policy"]] = r["rank"] + "/" + r["verdict"] + "/" + r["wins"] + r["ties"] + r["losses"]
	}
	// default means: predictive 2.6 < threshold_cooldown 3 < slo_feedback 3.04 < static_min 3.4 (oracle 2)
	// gpu_x3 means:  static_min 4.8 < threshold_cooldown 5 < slo_feedback 5.04 < predictive 5.8 (oracle 4.4)
	// slo_feedback against the baseline per seed: +0.02 (inside the 1 % band), −0.2, +0.3
	want := map[string]string{
		"default/predictive": "1/better/300", "default/threshold_cooldown": "2/-/000", "default/slo_feedback": "3/no_difference/111",
		"default/static_min": "4/worse/003", "default/static_peak_oracle": "-/better/300",
		"gpu_x3/static_min": "1/better/300", "gpu_x3/threshold_cooldown": "2/-/000", "gpu_x3/slo_feedback": "3/no_difference/111",
		"gpu_x3/predictive": "4/worse/003", "gpu_x3/static_peak_oracle": "-/better/300",
	}
	for k, v := range want {
		if rank[k] != v {
			t.Errorf("%s: got %s, want %s", k, rank[k], v)
		}
	}
	// Kendall tau-b over the four deployable policies: 1 concordant pair, 5 discordant;
	// verdicts that stay the same: slo_feedback only, of three non-baseline policies.
	for _, r := range readCSVFile(t, filepath.Join(out, "summary.csv")) {
		if r["weights"] == "gpu_x3" && (r["policies"] != "4" || r["kendall_tau_vs_reference"] != "-0.66666667" || r["top_policy"] != "static_min" ||
			r["top_same"] != "false" || r["verdicts_same"] != "1" || r["verdicts_total"] != "3") {
			t.Errorf("gpu_x3 summary %v", r)
		}
	}
	// cap = 10 GPU-hours: predictive (16) and the oracle (12) are infeasible; with gamma = 0,
	// threshold_cooldown (2) ranks before slo_feedback (2.04) and static_min (2.7).
	cap := map[string]string{}
	for _, r := range readCSVFile(t, filepath.Join(out, "gpu_cap.csv")) {
		cap[r["policy"]] = r["feasible"] + "/" + r["rank"]
	}
	if cap["predictive"] != "false/-" || cap["static_peak_oracle"] != "false/-" || cap["threshold_cooldown"] != "true/1" ||
		cap["slo_feedback"] != "true/2" || cap["static_min"] != "true/3" {
		t.Fatalf("gpu cap view %v", cap)
	}
}
