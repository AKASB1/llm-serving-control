package admission

import (
	"testing"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/slo"
)

func rep(id string, inflight int) routing.ReplicaView {
	return routing.ReplicaView{ID: id, InFlight: inflight, Class: registry.ClassInfo{PrefillTokensPerSec: 10000, MaxNumSeqs: 128}}
}

func TestQueueCap(t *testing.T) {
	q := NewQueueCap(QueueCapParams{MaxInFlightPerReplica: 4, MaxQueue: 2})
	v := routing.View{Replicas: []routing.ReplicaView{rep("a", 4), rep("b", 3)}}
	if d := q.Admit(routing.Request{}, v); d.Action != routing.Dispatch {
		t.Fatalf("free slot: %+v", d)
	}
	v.Replicas[1].InFlight = 4
	if d := q.Admit(routing.Request{}, v); d.Action != routing.Queue {
		t.Fatalf("all at cap: %+v", d)
	}
	v.RouterQueueLen = 2
	if d := q.Admit(routing.Request{}, v); d.Action != routing.Reject {
		t.Fatalf("queue full: %+v", d)
	}
}

func TestPredictedTTFTShedsBatchFirst(t *testing.T) {
	ts, _ := slo.NewTargets("interactive", map[string]slo.TargetJSON{"interactive": {TTFTs: 2, TPOTs: 0.1}, "batch": {TTFTs: 20, TPOTs: 0.25}})
	a := NewPredictedTTFT(DefaultPredictedTTFT(), ts)
	r := rep("a", 0)
	r.HasScrape = true
	// 7000 prompt tokens ahead at 10000·0.5 tok/s: predicted (7000+1000)/5000 = 1.6 s.
	r.Scraped = metrics.ReplicaSnapshot{WaitingPromptTokens: 7000}
	v := routing.View{Replicas: []routing.ReplicaView{r}}
	if p := a.Predict(routing.Request{PromptTokens: 1000}, r); p != 1.6 {
		t.Fatalf("prediction %v", p)
	}
	inter := routing.Request{PromptTokens: 1000, SLOClass: "interactive"}
	batch := routing.Request{PromptTokens: 1000, SLOClass: "batch"}
	if d := a.Admit(inter, v); d.Action != routing.Dispatch {
		t.Fatalf("interactive must pass at 1.6 s < 2 s: %+v", d)
	}
	if d := a.Admit(batch, v); d.Action != routing.Reject {
		t.Fatalf("batch must be shed at 1.6 s > 1 s: %+v", d)
	}
	// Dispatches since the scrape count against the prediction.
	v.Replicas[0].SentSinceScrape = 3
	if d := a.Admit(inter, v); d.Action != routing.Reject {
		t.Fatalf("interactive must be shed at 2.2 s: %+v", d)
	}
	if d := a.Admit(inter, routing.View{}); d.Action != routing.Queue {
		t.Fatalf("no replica: %+v", d)
	}
}
