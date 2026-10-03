package routing

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

func view(rs ...ReplicaView) View { return View{Model: "m", Replicas: rs} }

func rv(id string, inflight int) ReplicaView {
	return ReplicaView{ID: id, InFlight: inflight, Class: registry.ClassInfo{MaxNumSeqs: 128, DecodeTokensPerSec: 1000, PrefillTokensPerSec: 10000}}
}

func testRNG() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

// Ported from the scaffold's TestLeastOutstandingHealthy: the scaffold
// function filtered unhealthy replicas itself; health filtering now happens in
// the dispatcher (only eligible replicas reach the view), so the unhealthy
// replica "c" is simply not in the view. Same expectation: "b".
func TestLeastOutstandingHealthy(t *testing.T) {
	d := NewLeastOutstanding().Route(Request{}, view(rv("a", 3), rv("b", 1)))
	if d.ReplicaID != "b" || d.Action != Dispatch {
		t.Fatalf("unexpected choice: %+v", d)
	}
}

func TestLeastOutstandingTiesGoToLowestID(t *testing.T) {
	d := NewLeastOutstanding().Route(Request{}, view(rv("a", 2), rv("b", 1), rv("c", 1)))
	if d.ReplicaID != "b" {
		t.Fatalf("tie: %+v", d)
	}
}

func TestRoundRobinCycles(t *testing.T) {
	p := NewRoundRobin()
	v := view(rv("a", 0), rv("b", 0), rv("c", 0))
	var got []string
	for range 4 {
		got = append(got, p.Route(Request{}, v).ReplicaID)
	}
	if got[0] != "a" || got[1] != "b" || got[2] != "c" || got[3] != "a" {
		t.Fatalf("order %v", got)
	}
}

func TestRandomIsUniformAndSeeded(t *testing.T) {
	v := view(rv("a", 0), rv("b", 0), rv("c", 0))
	count := map[string]int{}
	p := NewRandom(testRNG())
	for range 3000 {
		count[p.Route(Request{}, v).ReplicaID]++
	}
	for id, n := range count {
		if n < 900 || n > 1100 {
			t.Fatalf("%s picked %d of 3000", id, n)
		}
	}
	a, b := NewRandom(testRNG()), NewRandom(testRNG())
	for range 50 {
		if a.Route(Request{}, v) != b.Route(Request{}, v) {
			t.Fatal("same seed, different picks")
		}
	}
}

func TestPowerOfTwoNeverPicksTheWorst(t *testing.T) {
	p := NewPowerOfTwo(testRNG())
	v := view(rv("a", 0), rv("b", 5), rv("c", 10))
	count := map[string]int{}
	for range 3000 {
		count[p.Route(Request{}, v).ReplicaID]++
	}
	if count["c"] != 0 {
		t.Fatalf("picked the most loaded replica %d times", count["c"])
	}
	// a wins whenever sampled (2/3 of pairs), b otherwise.
	if count["a"] < 1850 || count["a"] > 2150 {
		t.Fatalf("a picked %d of 3000, want about 2000", count["a"])
	}
	two := view(rv("x", 3), rv("y", 1))
	for range 20 {
		if p.Route(Request{}, two).ReplicaID != "y" {
			t.Fatal("with two replicas the less loaded one must win")
		}
	}
}

func TestLatencyAwarePeakEWMAAndNoStarvation(t *testing.T) {
	p := NewLatencyAware(LatencyAwareParams{TauS: 10, PriorS: 0.2}, testRNG())
	sec := time.Second
	// a reports a slow first token: peak sensitivity replaces the value.
	p.Observe(Outcome{ReplicaID: "a", Now: 1 * sec, Kind: FirstToken, Latency: 5 * sec})
	p.Observe(Outcome{ReplicaID: "b", Now: 1 * sec, Kind: FirstToken, Latency: 200 * time.Millisecond})
	v := view(rv("a", 0), rv("b", 2))
	v.Now = 1 * sec
	if d := p.Route(Request{}, v); d.ReplicaID != "b" {
		t.Fatalf("slow replica chosen: %+v", d)
	}
	// A lower sample is blended, not adopted.
	p.Observe(Outcome{ReplicaID: "a", Now: 2 * sec, Kind: FirstToken, Latency: 100 * time.Millisecond})
	if e := p.m["a"].value; e < 1 {
		t.Fatalf("lower sample adopted immediately: %v", e)
	}
	// After a long idle period a's cost decays below b's: it gets traffic again.
	v.Now = 60 * sec
	p.Observe(Outcome{ReplicaID: "b", Now: 60 * sec, Kind: FirstToken, Latency: 300 * time.Millisecond})
	if d := p.Route(Request{}, v); d.ReplicaID != "a" {
		t.Fatalf("idle replica starved: %+v", d)
	}
}

func scraped(id string, waiting, running int, kv float64, sent int) ReplicaView {
	r := rv(id, waiting+running)
	r.HasScrape = true
	r.SentSinceScrape = sent
	r.Scraped = metrics.ReplicaSnapshot{ReplicaID: id, Waiting: waiting, Running: running, KVUsedTokens: int(kv * 1000), KVCapacityTokens: 1000}
	return r
}

func TestQueueAwareHerdsAndCorrectedSpreads(t *testing.T) {
	plain := NewQueueAware(DefaultQueueAware(), false)
	corr := NewQueueAware(DefaultQueueAware(), true)
	// a looks emptiest in the snapshot; the router has since sent it 5 requests.
	v := view(scraped("a", 0, 10, 0.1, 5), scraped("b", 1, 10, 0.1, 0), scraped("c", 2, 10, 0.1, 0))
	if d := plain.Route(Request{}, v); d.ReplicaID != "a" {
		t.Fatalf("queue_aware must follow the stale snapshot: %+v", d)
	}
	if d := corr.Route(Request{}, v); d.ReplicaID != "b" {
		t.Fatalf("corrected variant must count dispatches since the scrape: %+v", d)
	}
	if plain.Name() != "queue_aware" || corr.Name() != "queue_aware_corrected" {
		t.Fatal("names")
	}
	// Without snapshots both fall back to router-local in-flight.
	if d := plain.Route(Request{}, view(rv("a", 4), rv("b", 2))); d.ReplicaID != "b" {
		t.Fatalf("fallback %+v", d)
	}
}

func TestCapacityWeightedSmoothWRR(t *testing.T) {
	p := NewCapacityWeighted(CapacityWeightedParams{PriorWeight: 5, SampleWeight: 0.2})
	fast, slow := rv("fast", 0), rv("slow", 0)
	fast.Class.DecodeTokensPerSec, slow.Class.DecodeTokensPerSec = 3000, 1000
	v := view(fast, slow)
	count := map[string]int{}
	var seq []string
	for range 400 {
		id := p.Route(Request{}, v).ReplicaID
		count[id]++
		if len(seq) < 4 {
			seq = append(seq, id)
		}
	}
	if count["fast"] != 300 || count["slow"] != 100 {
		t.Fatalf("counts %v, want 300/100", count)
	}
	// Smooth: the slow replica is interleaved, not batched at the end.
	if seq[0] != "fast" || seq[1] != "fast" || seq[2] != "slow" || seq[3] != "fast" {
		t.Fatalf("sequence %v", seq)
	}
}

func TestCapacityWeightedLearnsMeasuredRate(t *testing.T) {
	p := NewCapacityWeighted(CapacityWeightedParams{PriorWeight: 1, SampleWeight: 1})
	a, b := rv("a", 0), rv("b", 0)
	snap := func(r ReplicaView, at time.Duration, gen int64, busy time.Duration) ReplicaView {
		r.HasScrape = true
		r.Scraped = metrics.ReplicaSnapshot{ReplicaID: r.ID, At: at, GeneratedTokens: gen, BusyTime: busy}
		return r
	}
	// Same prior; a measures 4000 tok/s, b 1000 tok/s over many scrapes.
	for i := 0; i <= 50; i++ {
		at := time.Duration(i) * time.Second
		p.weight(snap(a, at, int64(i)*4000, at))
		p.weight(snap(b, at, int64(i)*1000, at))
	}
	wa := p.weight(snap(a, 50*time.Second, 200000, 50*time.Second))
	wb := p.weight(snap(b, 50*time.Second, 50000, 50*time.Second))
	if wa < 3*wb {
		t.Fatalf("measured rates not reflected: %v vs %v", wa, wb)
	}
}

func TestOracleJSQUsesFreshWork(t *testing.T) {
	p := NewOracleJSQ()
	if !p.IsOracle() {
		t.Fatal("oracle must declare itself")
	}
	a, b := rv("a", 0), rv("b", 9)
	a.Fresh = &metrics.ReplicaSnapshot{WaitingPromptTokens: 20000, RemainingDecodeTokens: 5000}
	b.Fresh = &metrics.ReplicaSnapshot{WaitingPromptTokens: 0, RemainingDecodeTokens: 1000}
	if d := p.Route(Request{PromptTokens: 100}, view(a, b)); d.ReplicaID != "b" {
		t.Fatalf("oracle ignored true work: %+v", d)
	}
}

func TestPrefixAffinityStickyAndBounded(t *testing.T) {
	p := NewPrefixAffinity(PrefixAffinityParams{VirtualNodes: 50, LoadFactor: 1.25})
	v := view(rv("a", 0), rv("b", 0), rv("c", 0), rv("d", 0))
	// The same group always lands on the same replica while loads are equal.
	first := p.Route(Request{PrefixGroup: "g1"}, v).ReplicaID
	for i := 0; i < 10; i++ {
		if got := p.Route(Request{PrefixGroup: "g1"}, v).ReplicaID; got != first {
			t.Fatalf("group moved from %s to %s", first, got)
		}
	}
	// Groups spread over replicas.
	seen := map[string]bool{}
	for i := 0; i < 64; i++ {
		seen[p.Route(Request{PrefixGroup: fmt.Sprint("grp", i)}, v).ReplicaID] = true
	}
	if len(seen) != 4 {
		t.Fatalf("64 groups used only %d replicas", len(seen))
	}
	// Load bound: with the home replica far above the average, the group moves.
	loaded := view(rv("a", 0), rv("b", 0), rv("c", 0), rv("d", 0))
	for i := range loaded.Replicas {
		if loaded.Replicas[i].ID == first {
			loaded.Replicas[i].InFlight = 20
		}
	}
	if got := p.Route(Request{PrefixGroup: "g1"}, loaded).ReplicaID; got == first {
		t.Fatal("bounded load ignored")
	}
	// No group: least outstanding.
	if got := p.Route(Request{}, view(rv("a", 3), rv("b", 1))).ReplicaID; got != "b" {
		t.Fatalf("fallback chose %s", got)
	}
}
