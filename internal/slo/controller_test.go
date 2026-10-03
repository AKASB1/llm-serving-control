package slo

import (
	"math"
	"testing"
	"time"
)

func TestControllerErrorSignal(t *testing.T) {
	ts, _ := NewTargets("i", map[string]TargetJSON{"i": {TTFTs: 1, TPOTs: 0.1}, "b": {TTFTs: 10, TPOTs: 1}})
	c := NewController(ts, 30*time.Second, 100)
	if e := c.Model("m", 0); e.N != 0 || e.Value != 0 {
		t.Fatalf("empty: %+v", e)
	}
	// 19 fast interactive requests and one at 3× the TTFT target.
	for i := 0; i < 19; i++ {
		c.Observe(time.Duration(i)*time.Second, "m", "r1", "i", true, 500*time.Millisecond, 50*time.Millisecond, 10, true)
	}
	c.Observe(19*time.Second, "m", "r2", "i", true, 3*time.Second, 50*time.Millisecond, 10, false)
	e := c.Model("m", 20*time.Second)
	// Nearest-rank P95 of 20 values is the 19th smallest: 0.5.
	if e.TTFTRatio != 0.5 || e.TPOTRatio != 0.5 || math.Abs(e.Value+0.5) > 1e-9 || e.ViolationRate != 0.05 {
		t.Fatalf("model error %+v", e)
	}
	if r := c.Replica("r2", 20*time.Second); r.TTFTRatio != 3 || r.Value != 2 {
		t.Fatalf("replica error %+v", r)
	}
	// Batch class is normalised by its own target.
	c.Observe(21*time.Second, "m2", "", "b", true, 5*time.Second, 500*time.Millisecond, 10, true)
	if e := c.Model("m2", 21*time.Second); e.TTFTRatio != 0.5 {
		t.Fatalf("class normalisation %+v", e)
	}
	// Rejections count as violations; with no completions the error saturates at 1.
	c.Observe(22*time.Second, "m3", "", "i", false, 0, 0, 10, false)
	if e := c.Model("m3", 22*time.Second); e.Value != 1 || e.ViolationRate != 1 {
		t.Fatalf("rejections %+v", e)
	}
	// Window expiry.
	if e := c.Model("m", 100*time.Second); e.N != 0 {
		t.Fatalf("expired observations counted: %+v", e)
	}
}

func TestControllerBufferIsBounded(t *testing.T) {
	ts, _ := NewTargets("i", map[string]TargetJSON{"i": {TTFTs: 1, TPOTs: 0.1}})
	c := NewController(ts, time.Hour, 16)
	for i := 0; i < 1000; i++ {
		c.Observe(time.Duration(i)*time.Millisecond, "m", "", "i", true, time.Second, 0, 1, true)
	}
	if e := c.Model("m", time.Second); e.N != 16 {
		t.Fatalf("buffer holds %d, want 16", e.N)
	}
}
