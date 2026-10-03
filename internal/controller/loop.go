package controller

import (
	"fmt"
	"sort"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/autoscaling"
	"github.com/AKASB1/llm-serving-control/internal/capacity"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/slo"
)

// ModelConfig describes one model under control.
type ModelConfig struct {
	Model string `json:"model"`
	// Class is the replica class used for new replicas of the model.
	Class string `json:"class"`
	Min   int    `json:"min"`
	Max   int    `json:"max"`
	// Slots is the number of concurrent requests per replica used by the
	// allocator's queueing approximation; 0 means the class's max_num_seqs.
	Slots int `json:"slots"`
}

// LoopConfig parameterises the control loop.
type LoopConfig struct {
	Models     []ModelConfig `json:"models"`
	WindowS    float64       `json:"window_s"`
	BudgetGPUs int           `json:"budget_gpus"`
}

// StepLog records one control decision for one model.
type StepLog struct {
	At       time.Duration
	Model    string
	Desired  int
	Granted  int
	Current  int
	Reason   string
	SLOError float64
}

// Loop is the periodic control loop: windowed signals and the SLO controller
// feed one scaler per model; an optional allocator arbitrates the desired
// counts under the GPU budget; the difference to the current replicas becomes
// provision and drain commands.
type Loop struct {
	cfg      LoopConfig
	reg      *registry.Registry
	disp     *Dispatcher
	classes  map[string]registry.ClassInfo
	warmup   map[string]time.Duration
	targets  slo.Targets
	slo      *slo.Controller
	windows  map[string]*metrics.Window
	scalers  map[string]autoscaling.Scaler
	alloc    capacity.Allocator
	lastSnap map[string]metrics.ReplicaSnapshot
	sampled  map[string]time.Duration
	repErr   map[string]float64
	models   []ModelConfig
	// Log is every decision in order.
	Log []StepLog
}

// NewLoop builds a loop. scalers maps model → scaler; alloc may be nil;
// warmup maps class → expected provisioning + loading time.
func NewLoop(cfg LoopConfig, reg *registry.Registry, disp *Dispatcher, classes map[string]registry.ClassInfo,
	warmup map[string]time.Duration, targets slo.Targets, scalers map[string]autoscaling.Scaler, alloc capacity.Allocator) (*Loop, error) {
	if cfg.WindowS <= 0 {
		cfg.WindowS = 30
	}
	l := &Loop{
		cfg: cfg, reg: reg, disp: disp, classes: classes, warmup: warmup, targets: targets,
		slo:     slo.NewController(targets, secs(cfg.WindowS), 4096),
		windows: map[string]*metrics.Window{}, scalers: scalers, alloc: alloc,
		lastSnap: map[string]metrics.ReplicaSnapshot{}, sampled: map[string]time.Duration{}, repErr: map[string]float64{},
	}
	for _, m := range cfg.Models {
		if m.Slots <= 0 {
			m.Slots = classes[m.Class].MaxNumSeqs
		}
		l.models = append(l.models, m)
	}
	sort.Slice(l.models, func(i, j int) bool { return l.models[i].Model < l.models[j].Model })
	for _, m := range l.models {
		if _, ok := classes[m.Class]; !ok {
			return nil, fmt.Errorf("loop: model %q uses unknown class %q", m.Model, m.Class)
		}
		if scalers[m.Model] == nil {
			return nil, fmt.Errorf("loop: no scaler for model %q", m.Model)
		}

		l.windows[m.Model] = metrics.NewWindow(secs(cfg.WindowS), 30)
	}
	disp.SLOError = func(id string) float64 { return l.repErr[id] }
	return l, nil
}

// SLO returns the SLO controller.
func (l *Loop) SLO() *slo.Controller { return l.slo }

// ObserveArrival records an arrival at the router.
func (l *Loop) ObserveArrival(now time.Duration, model string) {
	if w := l.windows[model]; w != nil {
		w.Arrival(now)
	}
}

// ObserveFinish records a finished request (any terminal state).
func (l *Loop) ObserveFinish(now time.Duration, rec metrics.RequestRecord) {
	completed := rec.State == metrics.Completed
	if w := l.windows[rec.Model]; w != nil && completed {
		w.Completion(now, rec.OutputTokens, rec.E2E(), rec.E2E()-rec.RouterQueue-rec.ReplicaQueue)
	}
	var ttft, tpot time.Duration
	if completed {
		ttft, tpot = rec.TTFT(), rec.TPOT()
	}
	l.slo.Observe(now, rec.Model, rec.Replica, rec.SLOClass, completed, ttft, tpot, rec.OutputTokens, rec.Met)
}

// ObserveSnapshot ingests a scraped snapshot (busy-time deltas → utilization).
func (l *Loop) ObserveSnapshot(now time.Duration, model string, s metrics.ReplicaSnapshot) {
	w := l.windows[model]
	if w == nil {
		return
	}
	if prev, ok := l.lastSnap[s.ReplicaID]; ok && s.At > prev.At {
		w.Busy(now, s.BusyTime-prev.BusyTime, s.At-prev.At)
	}
	l.lastSnap[s.ReplicaID] = s
	l.sample(now, model)
}

func (l *Loop) sample(now time.Duration, model string) {
	if t, ok := l.sampled[model]; ok && t == now {
		return
	}
	l.sampled[model] = now
	ready := 0
	for _, r := range l.reg.ForModel(model) {
		if r.State == registry.Ready {
			ready++
		}
	}
	l.windows[model].Sample(now, l.disp.TotalInFlight(model), ready)
}

type counts struct {
	ready, starting, draining int
	readyIDs, startingIDs     []registry.Replica
}

func (l *Loop) count(model string) counts {
	var c counts
	for _, r := range l.reg.ForModel(model) {
		switch r.State {
		case registry.Ready:
			c.ready++
			c.readyIDs = append(c.readyIDs, r)
		case registry.Provisioning, registry.Loading:
			c.starting++
			c.startingIDs = append(c.startingIDs, r)
		case registry.Draining:
			c.draining++
		}
	}
	return c
}

// Step runs one control round at time now.
func (l *Loop) Step(now time.Duration) []ScaleCommand {
	desired := map[string]int{}
	reasons := map[string]string{}
	cs := map[string]counts{}
	views := map[string]autoscaling.ModelView{}
	for _, m := range l.models {
		l.sample(now, m.Model)
		c := l.count(m.Model)
		cs[m.Model] = c
		sig := l.windows[m.Model].Signals(now)
		sig.QueueDepth, sig.KVUsage = l.gauges(m.Model, c)
		e := l.slo.Model(m.Model, now)
		v := autoscaling.ModelView{
			Now: now, Model: m.Model, Ready: c.ready, Starting: c.starting, Draining: c.draining,
			Min: m.Min, Max: m.Max, WarmUp: l.warmup[m.Class], Signals: sig, SLOError: e.Value, ViolationRate: e.ViolationRate,
		}
		views[m.Model] = v
		d := l.scalers[m.Model].Desired(v)
		desired[m.Model] = autoscaling.Clamp(d.Replicas, m.Min, m.Max)
		reasons[m.Model] = l.scalers[m.Model].Name() + ": " + d.Reason
	}
	granted := desired
	if l.alloc != nil {
		in := capacity.Input{Now: now, BudgetGPUs: l.cfg.BudgetGPUs}
		for _, m := range l.models {
			v := views[m.Model]
			_, tg := l.targets.For("")
			svc := v.Signals.MeanServiceTime
			if v.Signals.Completions == 0 {
				svc = 0
			}
			in.Models = append(in.Models, capacity.ModelDemand{
				Model: m.Model, Desired: desired[m.Model], Current: v.Current(), Min: m.Min, Max: m.Max,
				GPUsPerReplica: l.classes[m.Class].GPUs, ArrivalRate: v.Signals.ArrivalRate, MeanServiceTime: svc,
				SlotsPerReplica: m.Slots, TTFTTarget: tg.TTFT, SLOError: v.SLOError,
			})
		}
		granted = map[string]int{}
		for _, g := range l.alloc.Allocate(in) {
			granted[g.Model] = g.Replicas
			if g.Replicas != desired[g.Model] {
				reasons[g.Model] += fmt.Sprintf("; %s granted %d of %d: %s", l.alloc.Name(), g.Replicas, desired[g.Model], g.Reason)
			}
		}
	}
	var cmds []ScaleCommand
	for _, m := range l.models {
		c := cs[m.Model]
		cur := c.ready + c.starting
		want := granted[m.Model]
		l.Log = append(l.Log, StepLog{At: now, Model: m.Model, Desired: desired[m.Model], Granted: want, Current: cur, Reason: reasons[m.Model], SLOError: views[m.Model].SLOError})
		switch {
		case want > cur:
			for i := 0; i < want-cur; i++ {
				cmds = append(cmds, ScaleCommand{Kind: Provision, Model: m.Model, Class: m.Class, Reason: reasons[m.Model]})
			}
		case want < cur:
			cmds = append(cmds, l.scaleIn(m.Model, c, cur-want, reasons[m.Model])...)
		}
	}
	// Refresh per-replica SLO errors for routing weights.
	for _, r := range l.reg.All() {
		l.repErr[r.ID] = l.slo.Replica(r.ID, now).Value
	}
	return cmds
}

// scaleIn cancels starting replicas first (newest first), then drains ready
// replicas with the fewest in-flight requests (ties: newest).
func (l *Loop) scaleIn(model string, c counts, n int, reason string) []ScaleCommand {
	var cmds []ScaleCommand
	starting := c.startingIDs
	sort.Slice(starting, func(i, j int) bool { return starting[i].ID > starting[j].ID })
	for _, r := range starting {
		if n == 0 {
			return cmds
		}
		cmds = append(cmds, ScaleCommand{Kind: Drain, Model: model, ReplicaID: r.ID, Reason: reason})
		n--
	}
	ready := c.readyIDs
	sort.Slice(ready, func(i, j int) bool {
		a, b := l.disp.InFlight(ready[i].ID), l.disp.InFlight(ready[j].ID)
		if a != b {
			return a < b
		}
		return ready[i].ID > ready[j].ID
	})
	for _, r := range ready {
		if n == 0 {
			break
		}
		cmds = append(cmds, ScaleCommand{Kind: Drain, Model: model, ReplicaID: r.ID, Reason: reason})
		n--
	}
	return cmds
}

// gauges returns the latest scraped waiting count plus the router queue, and
// the mean KV usage over ready replicas.
func (l *Loop) gauges(model string, c counts) (float64, float64) {
	q := float64(l.disp.QueueLen(model))
	kv, n := 0.0, 0
	for _, r := range c.readyIDs {
		if s, ok := l.disp.LatestSnapshot(r.ID); ok {
			q += float64(s.Waiting)
			kv += s.KVUsage()
			n++
		}
	}
	if n > 0 {
		kv /= float64(n)
	}
	return q, kv
}
