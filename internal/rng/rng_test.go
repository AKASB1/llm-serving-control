package rng

import "testing"

func draw(seed uint64, name string, n int) []uint64 {
	r := Stream(seed, name)
	out := make([]uint64, n)
	for i := range out {
		out[i] = r.Uint64()
	}
	return out
}

func TestStreamDeterministic(t *testing.T) {
	a, b := draw(7, "arrivals", 16), draw(7, "arrivals", 16)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("draw %d differs: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestStreamsDiffer(t *testing.T) {
	same := func(x, y []uint64) bool {
		for i := range x {
			if x[i] != y[i] {
				return false
			}
		}
		return true
	}
	if same(draw(7, "arrivals", 8), draw(8, "arrivals", 8)) {
		t.Fatal("different seeds produced the same stream")
	}
	if same(draw(7, "arrivals", 8), draw(7, "lengths", 8)) {
		t.Fatal("different components produced the same stream")
	}
}

func TestStreamUniformity(t *testing.T) {
	// Coarse sanity check: the mean of 100k uniforms is within 0.01 of 0.5.
	r := Stream(1, "uniformity")
	sum := 0.0
	const n = 100000
	for range n {
		sum += r.Float64()
	}
	if m := sum / n; m < 0.49 || m > 0.51 {
		t.Fatalf("mean %v", m)
	}
}
