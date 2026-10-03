package metrics

import (
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/slo"
)

// Weights are the objective weights of J (docs/contracts.md §3).
type Weights struct {
	Alpha float64 `json:"alpha"`
	Beta  float64 `json:"beta"`
	Gamma float64 `json:"gamma"`
	Delta float64 `json:"delta"`
}

// Accounting is the cluster-side input of a run summary.
type Accounting struct {
	RunDuration time.Duration
	BudgetGPUs  float64
	// GPUSeconds is Σ GPUs × (termination − provisioning start).
	GPUSeconds float64
	// ReadySeconds, BusySeconds: Σ over replicas of serving time (ready or
	// draining, and alive) and of the part of it with a non-empty running batch.
	ReadySeconds float64
	BusySeconds  float64
	// KVTokenSeconds and KVCapacityTokenSeconds integrate KV occupancy and
	// capacity over serving time.
	KVTokenSeconds         float64
	KVCapacityTokenSeconds float64
	ITL                    *Histogram
	ScaleActions           int
	PeakReplicas           int
}

// Summary is one run's result row.
type Summary struct {
	Requests      int
	Completed     int
	Rejected      int
	Failed        int
	TimedOut      int
	Unfinished    int
	ViolationRate float64
	Goodput       float64 // SLO-meeting completions per second
	J             float64
	JTTFT, JTPOT  float64 // the latency terms before weighting
	JGPU          float64
	TTFTP50       float64
	TTFTP95       float64
	TTFTP99       float64
	TPOTP50       float64
	TPOTP95       float64
	TPOTP99       float64
	E2EP95        float64
	ITLP50        float64
	ITLP99        float64
	RouterQMean   float64
	RouterQP95    float64
	ReplicaQMean  float64
	ReplicaQP95   float64
	Retries       int
	Preemptions   int
	GPUHours      float64
	Utilization   float64
	KVOccupancy   float64
	ScaleActions  int
	PeakReplicas  int
}

type groupKey struct{ model, class string }

// Summarize computes a run summary. Latency percentiles are over completed
// requests; J's latency terms are request-weighted averages over (model, SLO
// class) groups of P95(group)/target(group); a group without completions
// contributes RunDuration/target (starvation is never rewarded).
func Summarize(recs []RequestRecord, targets slo.Targets, w Weights, acc Accounting) Summary {
	var s Summary
	s.Requests = len(recs)
	var ttft, tpot, e2e, rq, pq []float64
	groups := map[groupKey]*struct {
		n          int
		ttft, tpot []float64
	}{}
	met := 0
	for _, r := range recs {
		switch r.State {
		case Completed:
			s.Completed++
		case Rejected:
			s.Rejected++
		case Failed:
			s.Failed++
		case TimedOut:
			s.TimedOut++
		case Unfinished:
			s.Unfinished++
		}
		if r.Met {
			met++
		}
		s.Retries += r.Retries
		s.Preemptions += r.Preemptions
		k := groupKey{r.Model, r.SLOClass}
		g := groups[k]
		if g == nil {
			g = &struct {
				n          int
				ttft, tpot []float64
			}{}
			groups[k] = g
		}
		g.n++
		if r.State != Completed {
			continue
		}
		ttft = append(ttft, r.TTFT().Seconds())
		e2e = append(e2e, r.E2E().Seconds())
		rq = append(rq, r.RouterQueue.Seconds())
		pq = append(pq, r.ReplicaQueue.Seconds())
		g.ttft = append(g.ttft, r.TTFT().Seconds())
		if r.OutputTokens >= 2 {
			tpot = append(tpot, r.TPOT().Seconds())
			g.tpot = append(g.tpot, r.TPOT().Seconds())
		}
	}
	runS := acc.RunDuration.Seconds()
	if s.Requests > 0 {
		s.ViolationRate = 1 - float64(met)/float64(s.Requests)
	}
	if runS > 0 {
		s.Goodput = float64(met) / runS
	}
	s.TTFTP50, s.TTFTP95, s.TTFTP99 = Percentile(ttft, 0.5), Percentile(ttft, 0.95), Percentile(ttft, 0.99)
	s.TPOTP50, s.TPOTP95, s.TPOTP99 = Percentile(tpot, 0.5), Percentile(tpot, 0.95), Percentile(tpot, 0.99)
	s.E2EP95 = Percentile(e2e, 0.95)
	s.RouterQMean, s.RouterQP95 = mean(rq), Percentile(rq, 0.95)
	s.ReplicaQMean, s.ReplicaQP95 = mean(pq), Percentile(pq, 0.95)
	if acc.ITL != nil {
		s.ITLP50, s.ITLP99 = acc.ITL.Quantile(0.5), acc.ITL.Quantile(0.99)
	} else {
		s.ITLP50, s.ITLP99 = math.NaN(), math.NaN()
	}

	// J latency terms, request-weighted over groups, iterated in sorted order.
	keys := make([]groupKey, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].model != keys[j].model {
			return keys[i].model < keys[j].model
		}
		return keys[i].class < keys[j].class
	})
	var wTTFT, wTPOT float64
	for _, k := range keys {
		g := groups[k]
		_, tg := targets.For(k.class)
		ft := runS / tg.TTFT.Seconds()
		if len(g.ttft) > 0 {
			ft = Percentile(g.ttft, 0.95) / tg.TTFT.Seconds()
		}
		fp := runS / tg.TPOT.Seconds()
		if len(g.tpot) > 0 {
			fp = Percentile(g.tpot, 0.95) / tg.TPOT.Seconds()
		} else if len(g.ttft) > 0 {
			fp = 0 // only single-token outputs completed: TPOT undefined, no penalty
		}
		wTTFT += float64(g.n) * ft
		wTPOT += float64(g.n) * fp
	}
	if s.Requests > 0 {
		s.JTTFT = wTTFT / float64(s.Requests)
		s.JTPOT = wTPOT / float64(s.Requests)
	}
	s.GPUHours = acc.GPUSeconds / 3600
	if acc.BudgetGPUs > 0 && runS > 0 {
		s.JGPU = acc.GPUSeconds / (acc.BudgetGPUs * runS)
	}
	s.J = w.Alpha*s.JTTFT + w.Beta*s.JTPOT + w.Gamma*s.JGPU + w.Delta*s.ViolationRate
	if acc.ReadySeconds > 0 {
		s.Utilization = acc.BusySeconds / acc.ReadySeconds
	}
	if acc.KVCapacityTokenSeconds > 0 {
		s.KVOccupancy = acc.KVTokenSeconds / acc.KVCapacityTokenSeconds
	}
	s.ScaleActions = acc.ScaleActions
	s.PeakReplicas = acc.PeakReplicas
	return s
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	t := 0.0
	for _, x := range xs {
		t += x
	}
	return t / float64(len(xs))
}

// Columns returns the CSV column names of a Summary, in Values order.
func Columns() []string {
	return []string{
		"requests", "completed", "rejected", "failed", "timed_out", "unfinished",
		"violation_rate", "goodput", "J", "j_ttft", "j_tpot", "j_gpu",
		"ttft_p50", "ttft_p95", "ttft_p99", "tpot_p50", "tpot_p95", "tpot_p99", "e2e_p95",
		"itl_p50", "itl_p99", "router_q_mean", "router_q_p95", "replica_q_mean", "replica_q_p95",
		"retries", "preemptions", "gpu_hours", "utilization", "kv_occupancy", "scale_actions", "peak_replicas",
	}
}

// Values formats the summary for CSV with fixed precision (byte-stable).
func (s Summary) Values() []string {
	i := func(v int) string { return strconv.Itoa(v) }
	f := func(v float64) string {
		if math.IsNaN(v) {
			return "NaN"
		}
		return strconv.FormatFloat(v, 'g', 8, 64)
	}
	return []string{
		i(s.Requests), i(s.Completed), i(s.Rejected), i(s.Failed), i(s.TimedOut), i(s.Unfinished),
		f(s.ViolationRate), f(s.Goodput), f(s.J), f(s.JTTFT), f(s.JTPOT), f(s.JGPU),
		f(s.TTFTP50), f(s.TTFTP95), f(s.TTFTP99), f(s.TPOTP50), f(s.TPOTP95), f(s.TPOTP99), f(s.E2EP95),
		f(s.ITLP50), f(s.ITLP99), f(s.RouterQMean), f(s.RouterQP95), f(s.ReplicaQMean), f(s.ReplicaQP95),
		i(s.Retries), i(s.Preemptions), f(s.GPUHours), f(s.Utilization), f(s.KVOccupancy), i(s.ScaleActions), i(s.PeakReplicas),
	}
}
