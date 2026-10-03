package main

import (
	"os"
	"path/filepath"
	"testing"
)

// tinyEnv loads the committed experiment and shrinks it to two short
// scenarios so the determinism tests run in seconds.
func tinyEnv(t *testing.T, policies []string) (*Env, []Job) {
	t.Helper()
	exp, _, err := LoadExperiment("../../configs/experiment.json")
	if err != nil {
		t.Fatal(err)
	}
	exp.ClassesFile = "../../configs/classes.json"
	for i := range exp.Scenarios {
		if exp.Scenarios[i].TracePath != "" {
			exp.Scenarios[i].TracePath = "../../" + exp.Scenarios[i].TracePath
		}
	}
	exp.Probe = ProbeConfig{Requests: 3000, DurationS: 120, WarmupS: 30}
	o := options{config: "../../configs/experiment.json", traces: filepath.Join(t.TempDir(), "traces"), tunedDir: t.TempDir()}
	env, _, err := newEnv(exp, o)
	if err != nil {
		t.Fatal(err)
	}
	env.durScale = 0.1
	if err := env.probeCapacity(4); err != nil {
		t.Fatal(err)
	}
	var jobs []Job
	for _, id := range []string{"r-bursty-short", "r-crash"} {
		sc := exp.Scenario(id)
		for _, label := range policies {
			ps, err := env.policies(sc, label)
			if err != nil {
				t.Fatal(err)
			}
			for _, seed := range []uint64{1, 2} {
				jobs = append(jobs, Job{Scenario: sc, Variant: Variant{ID: "base"}, Label: label, Seed: seed, Policy: ps})
			}
		}
	}
	return env, jobs
}

func readAll(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, f := range []string{"runs.csv", "aggregates.csv", "paired.csv", "sensitivity.csv", "manifest.json"} {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			t.Fatal(err)
		}
		out[f] = string(b)
	}
	return out
}

// Same seeds and configuration give byte-identical result files, whatever
// the number of parallel workers (this is also the runner's concurrency test:
// run it with -count=20).
func TestBenchmarkDeterministicAcrossRunsAndWorkers(t *testing.T) {
	all := []string{"random", "least_outstanding", "power_of_two", "latency_aware", "queue_aware", "oracle_jsq"}
	env, jobs := tinyEnv(t, all)
	r1, err := env.runAll(jobs, 1)
	if err != nil {
		t.Fatal(err)
	}
	r8, err := env.runAll(jobs, 8)
	if err != nil {
		t.Fatal(err)
	}
	man := manifest{Label: "test", Capacity: env.capacity}
	d1, d8 := t.TempDir(), t.TempDir()
	if err := writeResults(d1, env.exp, r1, man); err != nil {
		t.Fatal(err)
	}
	if err := writeResults(d8, env.exp, r8, man); err != nil {
		t.Fatal(err)
	}
	f1, f8 := readAll(t, d1), readAll(t, d8)
	for name := range f1 {
		if f1[name] != f8[name] {
			t.Fatalf("%s differs between 1 and 8 workers", name)
		}
	}
	// Different seeds give different results.
	if r1[0].Summary.Values()[0] == r1[1].Summary.Values()[0] && r1[0].TraceSHA == r1[1].TraceSHA {
		t.Fatal("seeds 1 and 2 produced the same trace")
	}
}

// Adding or removing a policy does not change the trace a seed produces.
func TestTraceIndependentOfPolicies(t *testing.T) {
	envA, jobsA := tinyEnv(t, []string{"random", "least_outstanding"})
	rowsA, err := envA.runAll(jobsA, 4)
	if err != nil {
		t.Fatal(err)
	}
	envB, jobsB := tinyEnv(t, []string{"least_outstanding"})
	rowsB, err := envB.runAll(jobsB, 4)
	if err != nil {
		t.Fatal(err)
	}
	sha := map[string]string{}
	for _, r := range rowsA {
		k := r.Scenario + "/" + string(rune('0'+r.Seed))
		if prev, ok := sha[k]; ok && prev != r.TraceSHA {
			t.Fatalf("policies saw different traces for %s", k)
		}
		sha[k] = r.TraceSHA
	}
	for _, r := range rowsB {
		if sha[r.Scenario+"/"+string(rune('0'+r.Seed))] != r.TraceSHA {
			t.Fatalf("removing a policy changed the trace of %s seed %d", r.Scenario, r.Seed)
		}
		// The shared policy's results are identical too.
		for _, a := range rowsA {
			if a.Policy == r.Policy && a.Scenario == r.Scenario && a.Seed == r.Seed && a.Summary != r.Summary {
				t.Fatalf("least_outstanding result changed when another policy was removed")
			}
		}
	}
}

func TestSearchBudgetIsEqual(t *testing.T) {
	exp, _, err := LoadExperiment("../../configs/experiment.json")
	if err != nil {
		t.Fatal(err)
	}
	env := &Env{exp: exp}
	for name := range searchSpaces {
		fam := "routing"
		switch name {
		case "static", "threshold_cooldown", "target_tracking", "slo_feedback", "predictive":
			fam = "scaling"
		case "static_partition", "marginal_gain":
			fam = "capacity"
		case "queue_cap", "predicted_ttft_shed":
			fam = "overload"
		}
		c := env.candidates(fam, name)
		if len(c) != exp.Tuning.Configs {
			t.Fatalf("%s gets %d configurations, want %d", name, len(c), exp.Tuning.Configs)
		}
		again := env.candidates(fam, name)
		for i := range c {
			if string(c[i].Params) != string(again[i].Params) {
				t.Fatalf("%s: search is not deterministic", name)
			}
		}
	}
	if c := env.candidates("routing", "least_outstanding"); len(c) != 1 {
		t.Fatal("parameterless policies get exactly one configuration")
	}
}
