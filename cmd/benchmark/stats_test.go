package main

import (
	"math"
	"testing"
)

func TestTQuantileKnownValues(t *testing.T) {
	// Two-sided 95 % critical values from standard t tables.
	cases := map[float64]float64{1: 12.706, 4: 2.776, 9: 2.262, 19: 2.093, 30: 2.042, 1000: 1.962}
	for df, want := range cases {
		if got := tQuantile(0.975, df); math.Abs(got-want) > 0.001 {
			t.Errorf("t(0.975, %v) = %.4f, want %.3f", df, got, want)
		}
	}
}

func TestMeanCI(t *testing.T) {
	m, h := meanCI([]float64{1, 2, 3, 4, 5})
	// sd = 1.5811, t(0.975, 4) = 2.7764 → half = 2.7764·1.5811/√5 = 1.9632
	if m != 3 || math.Abs(h-1.9632) > 1e-3 {
		t.Fatalf("mean %v half %v", m, h)
	}
	if _, h := meanCI([]float64{7}); !math.IsNaN(h) {
		t.Fatal("n=1 half-width must be NaN")
	}
}

func TestWinTieLoss(t *testing.T) {
	w, ti, l := wtl([]float64{0.9, 1.005, 1.2, 1}, []float64{1, 1, 1, 1}, 0.01)
	if w != 1 || ti != 2 || l != 1 {
		t.Fatalf("w/t/l %d/%d/%d", w, ti, l)
	}
}

func TestKendallTau(t *testing.T) {
	if k := kendallTau([]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}); k != 1 {
		t.Fatalf("identical order %v", k)
	}
	if k := kendallTau([]float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}); k != -1 {
		t.Fatalf("reversed order %v", k)
	}
	if k := kendallTau([]float64{1, 2, 3}, []float64{1, 3, 2}); math.Abs(k-1.0/3) > 1e-12 {
		t.Fatalf("one swap %v", k)
	}
}
