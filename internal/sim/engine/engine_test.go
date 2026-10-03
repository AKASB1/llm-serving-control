package engine

import (
	"fmt"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/rng"
)

func params() Params {
	return Params{
		MaxNumSeqs: 4, MaxBatchedTokens: 2048, KVCapacityTokens: 100000, MaxModelLen: 32768,
		TBase: 0.01, CSeq: 0.001, CCtx: 1e-6, CPf: 1e-5,
	}
}

// run drives the engine to completion, checking the KV invariant after every
// step, and returns the events in order.
func run(t *testing.T, e *Engine, now time.Duration) ([]Event, time.Duration) {
	t.Helper()
	var all []Event
	for e.HasWork() {
		d, ok := e.Start(now)
		if !ok {
			t.Fatal("engine has work but cannot schedule")
		}
		checkKV(t, e)
		now += d
		all = append(all, e.Finish(now)...)
		checkKV(t, e)
	}
	return all, now
}

func checkKV(t *testing.T, e *Engine) {
	t.Helper()
	st := e.State()
	if st.KVUsed < 0 || st.KVReserved < 0 || st.KVUsed+st.KVReserved > st.KVCapacity {
		t.Fatalf("KV invariant violated: used %d reserved %d cap %d", st.KVUsed, st.KVReserved, st.KVCapacity)
	}
}

func TestSingleSequenceTimeline(t *testing.T) {
	p := params()
	e := New(p)
	s := &Seq{ID: "a", Prompt: 100, Output: 3}
	if err := e.Add(s, 0); err != nil {
		t.Fatal(err)
	}
	d1, _ := e.Start(0)
	if want := p.IterationTime(0, 0, 100); d1 != want {
		t.Fatalf("prefill iteration %v, want %v", d1, want)
	}
	ev := e.Finish(d1)
	if len(ev) != 1 || !ev[0].First || ev[0].Done {
		t.Fatalf("first token event %+v", ev)
	}
	if s.FirstScheduledAt != 0 || s.KV() != 100 {
		t.Fatalf("after prefill: scheduled %v kv %d", s.FirstScheduledAt, s.KV())
	}
	d2, _ := e.Start(d1)
	if want := p.IterationTime(1, 100, 0); d2 != want {
		t.Fatalf("decode iteration %v, want %v", d2, want)
	}
	e.Finish(d1 + d2)
	d3, _ := e.Start(d1 + d2)
	ev = e.Finish(d1 + d2 + d3)
	if len(ev) != 1 || !ev[0].Done {
		t.Fatalf("last token %+v", ev)
	}
	if st := e.State(); st.KVUsed != 0 || st.Running != 0 || st.Completed != 1 || st.Generated != 3 {
		t.Fatalf("final state %+v", st)
	}
	if _, ok := e.Start(d1 + d2 + d3); ok {
		t.Fatal("idle engine scheduled an iteration")
	}
}

func TestChunkedPrefillFirstTokenAtPromptCompletion(t *testing.T) {
	e := New(params())
	s := &Seq{ID: "long", Prompt: 5000, Output: 2}
	_ = e.Add(s, 0)
	events, _ := run(t, e, 0)
	// 2048 + 2048 + 904 → first token after the third iteration, then one decode.
	if e.State().Iterations != 4 {
		t.Fatalf("iterations = %d, want 4", e.State().Iterations)
	}
	if len(events) != 2 || !events[0].First || !events[1].Done {
		t.Fatalf("events %+v", events)
	}
}

func TestDecodesFirstThenPromptChunks(t *testing.T) {
	p := params()
	p.MaxBatchedTokens = 100
	e := New(p)
	a := &Seq{ID: "a", Prompt: 10, Output: 50}
	_ = e.Add(a, 0)
	d, _ := e.Start(0)
	e.Finish(d) // a is now decoding
	b := &Seq{ID: "b", Prompt: 1000, Output: 5}
	_ = e.Add(b, d)
	d2, _ := e.Start(d)
	// One decode token for a, then 99 prompt tokens of b.
	if want := p.IterationTime(1, 10, 99); d2 != want {
		t.Fatalf("mixed iteration %v, want %v", d2, want)
	}
	ev := e.Finish(d + d2)
	if len(ev) != 1 || ev[0].Seq != a {
		t.Fatalf("only a should emit: %+v", ev)
	}
	if b.FirstScheduledAt != d {
		t.Fatalf("b first scheduled at %v, want %v", b.FirstScheduledAt, d)
	}
}

func TestMaxNumSeqsAndFCFS(t *testing.T) {
	e := New(params()) // MaxNumSeqs 4
	var seqs []*Seq
	for i := range 6 {
		s := &Seq{ID: fmt.Sprint(i), Prompt: 10, Output: 100}
		seqs = append(seqs, s)
		_ = e.Add(s, 0)
	}
	e.Start(0)
	if st := e.State(); st.Running != 4 || st.Waiting != 2 {
		t.Fatalf("running %d waiting %d", st.Running, st.Waiting)
	}
	for i, s := range seqs {
		if (s.FirstScheduledAt == 0) != (i < 4) {
			t.Fatalf("seq %d scheduled=%v: FCFS order broken", i, s.FirstScheduledAt == 0)
		}
	}
}

func TestPreemptionByRecompute(t *testing.T) {
	p := params()
	p.KVCapacityTokens = 300
	p.MaxModelLen = 300
	e := New(p)
	a := &Seq{ID: "a", Prompt: 100, Output: 120}
	b := &Seq{ID: "b", Prompt: 100, Output: 120}
	_ = e.Add(a, 0)
	_ = e.Add(b, 0)
	events, _ := run(t, e, 0)
	if b.Preemptions == 0 {
		t.Fatal("the most recently admitted sequence (b) must be preempted")
	}
	if a.Preemptions != 0 {
		t.Fatal("a was admitted first and must not be preempted while b runs")
	}
	tokens := map[string]int{}
	firsts := map[string]int{}
	for _, ev := range events {
		if ev.Kind == Token {
			tokens[ev.Seq.ID]++
			if ev.First {
				firsts[ev.Seq.ID]++
			}
		}
	}
	if tokens["a"] != 120 || tokens["b"] != 120 || firsts["a"] != 1 || firsts["b"] != 1 {
		t.Fatalf("tokens %v firsts %v", tokens, firsts)
	}
	if e.State().Preemptions != int64(b.Preemptions) {
		t.Fatal("preemption counter mismatch")
	}
}

func TestTooLongIsRefused(t *testing.T) {
	e := New(params())
	if err := e.Add(&Seq{ID: "x", Prompt: 30000, Output: 3000}, 0); err != ErrTooLong {
		t.Fatalf("err = %v", err)
	}
}

func TestAbortAndReset(t *testing.T) {
	e := New(params())
	a := &Seq{ID: "a", Prompt: 10, Output: 100}
	b := &Seq{ID: "b", Prompt: 10, Output: 100}
	c := &Seq{ID: "c", Prompt: 10, Output: 100}
	for _, s := range []*Seq{a, b, c} {
		_ = e.Add(s, 0)
	}
	d, _ := e.Start(0)
	e.Finish(d)
	if !e.Abort(b) || e.Abort(b) {
		t.Fatal("abort of running sequence")
	}
	if st := e.State(); st.Running != 2 || st.KVUsed != a.KV()+c.KV() {
		t.Fatalf("after abort %+v", st)
	}
	out := e.Reset(d)
	if len(out) != 2 || e.HasWork() || e.State().KVUsed != 0 {
		t.Fatalf("reset returned %d, state %+v", len(out), e.State())
	}
}

// Random workloads under tight KV: every sequence emits exactly Output tokens
// with exactly one first token, KV never exceeds capacity, and everything
// completes.
func TestRandomizedInvariants(t *testing.T) {
	for seed := uint64(1); seed <= 30; seed++ {
		r := rng.Stream(seed, "engine-invariants")
		p := Params{
			MaxNumSeqs: 1 + r.IntN(16), KVCapacityTokens: 500 + r.IntN(4000),
			TBase: 0.005, CSeq: 1e-4, CCtx: 1e-7, CPf: 1e-5,
		}
		p.MaxBatchedTokens = p.MaxNumSeqs + r.IntN(512)
		p.MaxModelLen = p.KVCapacityTokens
		e := New(p)
		n := 20 + r.IntN(60)
		want := map[string]int{}
		now := time.Duration(0)
		added := 0
		got := map[string]int{}
		first := map[string]int{}
		for added < n || e.HasWork() {
			if added < n && r.IntN(3) == 0 {
				pr := 1 + r.IntN(p.MaxModelLen/2)
				out := 1 + r.IntN(p.MaxModelLen-pr)
				s := &Seq{ID: fmt.Sprint(added), Prompt: pr, Output: out}
				if err := e.Add(s, now); err != nil {
					t.Fatal(err)
				}
				want[s.ID] = out
				added++
			}
			if d, ok := e.Start(now); ok {
				checkKV(t, e)
				if d <= 0 {
					t.Fatal("non-positive iteration time")
				}
				now += d
				for _, ev := range e.Finish(now) {
					if ev.Kind == Token {
						got[ev.Seq.ID]++
						if ev.First {
							first[ev.Seq.ID]++
						}
					}
				}
				checkKV(t, e)
			}
		}
		for id, w := range want {
			if got[id] != w || first[id] != 1 {
				t.Fatalf("seed %d seq %s: %d tokens (%d first), want %d", seed, id, got[id], first[id], w)
			}
		}
		if st := e.State(); st.KVUsed != 0 || st.KVReserved != 0 || st.Completed != int64(n) {
			t.Fatalf("seed %d final %+v", seed, st)
		}
	}
}

// The preempted sequence goes back to the head of a non-empty waiting queue,
// ahead of a sequence that has been waiting longer.
func TestPreemptedSequenceGoesToTheHead(t *testing.T) {
	p := params()
	p.KVCapacityTokens, p.MaxModelLen, p.MaxNumSeqs = 300, 300, 2
	e := New(p)
	a := &Seq{ID: "a", Prompt: 100, Output: 120}
	b := &Seq{ID: "b", Prompt: 100, Output: 120}
	c := &Seq{ID: "c", Prompt: 10, Output: 5}
	for _, s := range []*Seq{a, b, c} {
		_ = e.Add(s, 0)
	}
	now := time.Duration(0)
	for b.Preemptions == 0 {
		d, ok := e.Start(now)
		if !ok {
			t.Fatal("stalled before the preemption")
		}
		if b.Preemptions > 0 {
			break // preempted while planning this iteration
		}
		now += d
		e.Finish(now)
	}
	if len(e.waiting) != 2 || e.waiting[0] != b || e.waiting[1] != c {
		t.Fatalf("waiting order after preemption: %v", ids(e.waiting))
	}
}

func ids(ss []*Seq) []string {
	var out []string
	for _, s := range ss {
		out = append(out, s.ID)
	}
	return out
}

func TestPrefixCacheHitSkipsPrefixPrefill(t *testing.T) {
	p := params()
	p.PrefixCacheGroups = 1
	e := New(p)
	a := &Seq{ID: "a", Prompt: 1000, Output: 2, PrefixGroup: "g", PrefixTokens: 800}
	_ = e.Add(a, 0)
	d1, _ := e.Start(0)
	e.Finish(d1)
	now := d1
	for e.HasWork() {
		d, _ := e.Start(now)
		now += d
		e.Finish(now)
	}
	b := &Seq{ID: "b", Prompt: 1000, Output: 2, PrefixGroup: "g", PrefixTokens: 800}
	_ = e.Add(b, now)
	d2, _ := e.Start(now)
	// Hit: only 200 prompt tokens are prefilled, with 800 cached tokens of context.
	if want := p.IterationTime(0, 800, 200); d2 != want || d1 != p.IterationTime(0, 0, 1000) {
		t.Fatalf("iteration times: miss %v, hit %v (want %v)", d1, d2, want)
	}
	e.Finish(now + d2)
	if h, q := e.PrefixStats(); h != 1 || q != 2 {
		t.Fatalf("hits/queries %d/%d", h, q)
	}
}

func TestPrefixCacheLRUEviction(t *testing.T) {
	p := params()
	p.PrefixCacheGroups = 1
	e := New(p)
	now := time.Duration(0)
	for _, g := range []string{"A", "B", "A"} {
		_ = e.Add(&Seq{ID: g, Prompt: 100, Output: 1, PrefixGroup: g, PrefixTokens: 50}, now)
		for e.HasWork() {
			d, _ := e.Start(now)
			now += d
			e.Finish(now)
		}
	}
	if h, q := e.PrefixStats(); h != 0 || q != 3 {
		t.Fatalf("B must evict A: hits/queries %d/%d", h, q)
	}
}

func TestRandomizedInvariantsWithPrefixCache(t *testing.T) {
	for seed := uint64(1); seed <= 20; seed++ {
		r := rng.Stream(seed, "engine-prefix")
		p := Params{MaxNumSeqs: 1 + r.IntN(12), KVCapacityTokens: 2000 + r.IntN(4000), TBase: 0.005, CSeq: 1e-4, CCtx: 1e-7, CPf: 1e-5, PrefixCacheGroups: 1 + r.IntN(4)}
		p.MaxBatchedTokens = p.MaxNumSeqs + r.IntN(512)
		p.MaxModelLen = p.KVCapacityTokens
		e := New(p)
		want, got := map[string]int{}, map[string]int{}
		now := time.Duration(0)
		for i := 0; i < 60; i++ {
			pr := 2 + r.IntN(p.MaxModelLen/3)
			s := &Seq{ID: fmt.Sprint(i), Prompt: pr, Output: 1 + r.IntN(p.MaxModelLen/3), PrefixGroup: fmt.Sprint("g", r.IntN(6)), PrefixTokens: r.IntN(pr + 5)}
			_ = e.Add(s, now)
			want[s.ID] = s.Output
		}
		for e.HasWork() {
			d, ok := e.Start(now)
			if !ok {
				t.Fatal("stalled")
			}
			checkKV(t, e)
			now += d
			for _, ev := range e.Finish(now) {
				if ev.Kind == Token {
					got[ev.Seq.ID]++
				}
			}
			checkKV(t, e)
		}
		for id, w := range want {
			if got[id] != w {
				t.Fatalf("seed %d seq %s: %d tokens, want %d", seed, id, got[id], w)
			}
		}
		if st := e.State(); st.KVUsed != 0 || st.KVReserved != 0 {
			t.Fatalf("seed %d: KV not released %+v", seed, st)
		}
	}
}
