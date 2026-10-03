package sim

import (
	"math"
	"testing"
)

// TestDeriveClasses checks the derivation against hand-computed values for
// h100-8b and logs the derived table (used in docs/simulator.md).
func TestDeriveClasses(t *testing.T) {
	cs := testClasses(t)
	for _, name := range ClassNames(cs) {
		c := cs[name]
		p := c.Params
		t.Logf("%-13s gpus=%d t_base=%.2fms c_seq=%.1fus c_ctx=%.4fus c_pf=%.1fus kv=%d tokens prefill=%.0f tok/s decode=%.0f tok/s",
			name, c.Spec.GPUs, p.TBase*1e3, p.CSeq*1e6, p.CCtx*1e6, p.CPf*1e6, p.KVCapacityTokens, c.Info.PrefillTokensPerSec, c.Info.DecodeTokensPerSec)
	}
	h := cs["h100-8b"].Params
	// weights 8.03e9·2 B over 3350e9·0.7 B/s, plus 4 ms overhead.
	if want := 16.06e9/(3350e9*0.7) + 0.004; math.Abs(h.TBase-want) > 1e-12 {
		t.Fatalf("t_base %v want %v", h.TBase, want)
	}
	if want := 2 * 8.03e9 / (989e12 * 0.5); math.Abs(h.CPf-want) > 1e-15 {
		t.Fatalf("c_pf %v want %v", h.CPf, want)
	}
	if want := int(math.Floor((80e9*0.9 - 16.06e9 - 2e9) / 131072)); h.KVCapacityTokens != want {
		t.Fatalf("kv %d want %d", h.KVCapacityTokens, want)
	}
	// Ordering sanity: the older GPU is slower and smaller.
	v, a := cs["v100-8b"].Params, cs["a100-8b"].Params
	if !(v.TBase > a.TBase && a.TBase > h.TBase && v.KVCapacityTokens < h.KVCapacityTokens) {
		t.Fatal("class ordering")
	}
	if _, err := (HardwareSpec{Name: "x"}).Derive(); err == nil {
		t.Fatal("incomplete spec must fail")
	}
}
