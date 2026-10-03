package slo

import (
	"testing"
	"time"
)

func TestMetRule(t *testing.T) {
	tg := Target{TTFT: time.Second, TPOT: 100 * time.Millisecond}
	cases := []struct {
		name      string
		completed bool
		ttft      time.Duration
		tpot      time.Duration
		e2e       time.Duration
		out       int
		want      bool
	}{
		{"ok", true, time.Second, 100 * time.Millisecond, 9 * time.Second, 50, true},
		{"not completed", false, 0, 0, 0, 50, false},
		{"ttft over", true, time.Second + 1, 0, 0, 50, false},
		{"tpot over", true, 0, 100*time.Millisecond + 1, 0, 50, false},
		{"tpot ignored for 1 token", true, 0, time.Hour, 0, 1, true},
	}
	for _, c := range cases {
		if got := Met(c.completed, c.ttft, c.tpot, c.e2e, c.out, tg); got != c.want {
			t.Errorf("%s: got %v", c.name, got)
		}
	}
	tg.E2E = 5 * time.Second
	if Met(true, 0, 0, 5*time.Second+1, 10, tg) {
		t.Error("e2e over target must violate")
	}
}

func TestTargetsDefaultAndValidation(t *testing.T) {
	ts, err := NewTargets("interactive", map[string]TargetJSON{
		"interactive": {TTFTs: 2, TPOTs: 0.1},
		"batch":       {TTFTs: 20, TPOTs: 0.25},
	})
	if err != nil {
		t.Fatal(err)
	}
	if name, tg := ts.For(""); name != "interactive" || tg.TTFT != 2*time.Second {
		t.Fatalf("For(\"\") = %s %v", name, tg)
	}
	if name, _ := ts.For("batch"); name != "batch" {
		t.Fatal("batch class")
	}
	if _, err := NewTargets("x", map[string]TargetJSON{"y": {TTFTs: 1, TPOTs: 1}}); err == nil {
		t.Fatal("missing default must fail")
	}
	if _, err := NewTargets("y", map[string]TargetJSON{"y": {TTFTs: 0, TPOTs: 1}}); err == nil {
		t.Fatal("zero TTFT must fail")
	}
}
