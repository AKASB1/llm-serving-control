// Package sim is the deterministic discrete-event simulator: a virtual clock,
// an event heap, simulated replicas (package engine) with a lifecycle
// (provisioning → loading → ready → draining → terminated, or failed), scrape
// snapshots, failure injection, and per-request records. The control plane
// (package controller and the policies) runs unchanged on top of it. One run
// is single-threaded and deterministic.
package sim

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/trace"
)

// InitialReplica asks for Count replicas of Class serving Model that are
// ready at time 0 (pre-warmed; paid from time 0).
type InitialReplica struct {
	Model string `json:"model"`
	Class string `json:"class"`
	Count int    `json:"count"`
}

// Crash kills a replica (by ID) at a time.
type Crash struct {
	AtS     float64 `json:"at_s"`
	Replica string  `json:"replica"`
}

// Config is one simulation run's environment (the policies come from the
// control plane passed to Run).
type Config struct {
	DurationS        float64 `json:"duration_s"`
	DrainHorizonS    float64 `json:"drain_horizon_s"`
	ScrapeIntervalS  float64 `json:"scrape_interval_s"`
	TickIntervalS    float64 `json:"tick_interval_s"`
	ControlIntervalS float64 `json:"control_interval_s"`
	// FailDelayS: in-flight requests on a crashed replica see an error after
	// this delay (0: connection reset). ConnectTimeoutS: a request sent to a
	// dead replica errors after this delay. ExecutorDetectS: the executor
	// marks a crashed replica failed (and stops paying for it) after this.
	FailDelayS      float64 `json:"fail_delay_s"`
	ConnectTimeoutS float64 `json:"connect_timeout_s"`
	ExecutorDetectS float64 `json:"executor_detect_s"`
	// DrainGraceS bounds how long a draining replica may finish in-flight work.
	DrainGraceS float64 `json:"drain_grace_s"`
	// ProbeIntervalS > 0 enables active health probes; a probe of a dead
	// replica fails after ProbeTimeoutS.
	ProbeIntervalS float64 `json:"probe_interval_s"`
	ProbeTimeoutS  float64 `json:"probe_timeout_s"`
	// PrefixTokens is the shared-prefix length of requests that carry a
	// prefix group; PrefixCacheGroups > 0 enables every replica's prefix
	// cache with that many groups (overriding the class).
	PrefixTokens      int              `json:"prefix_tokens,omitempty"`
	PrefixCacheGroups int              `json:"prefix_cache_groups,omitempty"`
	BudgetGPUs        float64          `json:"budget_gpus"`
	Initial           []InitialReplica `json:"initial"`
	Crashes           []Crash          `json:"crashes,omitempty"`
}

// DefaultConfig returns the documented defaults (durations to be set).
func DefaultConfig() Config {
	return Config{
		DrainHorizonS: 60, ScrapeIntervalS: 1, TickIntervalS: 0.5, ControlIntervalS: 5,
		FailDelayS: 0, ConnectTimeoutS: 1, ExecutorDetectS: 10, DrainGraceS: 180,
		ProbeIntervalS: 2, ProbeTimeoutS: 1,
	}
}

// Control is the periodic control loop driven by the simulator (nil means
// static replicas).
type Control interface {
	ObserveArrival(now time.Duration, model string)
	ObserveFinish(now time.Duration, rec metrics.RequestRecord)
	ObserveSnapshot(now time.Duration, model string, s metrics.ReplicaSnapshot)
	Step(now time.Duration) []controller.ScaleCommand
}

// Hooks lets tests observe the run (all optional).
type Hooks struct {
	// AfterEvent runs after every processed event.
	AfterEvent func(s *Sim)
}

// ReplicaRecord is the lifecycle of one replica in a run.
type ReplicaRecord struct {
	ID          string
	Model       string
	Class       string
	GPUs        int
	ProvisionAt time.Duration
	ReadyAt     time.Duration // -1 if never ready
	EndAt       time.Duration // termination (or the run end while alive)
	Final       registry.State
}

// Result is the outcome of one run.
type Result struct {
	Records    []metrics.RequestRecord
	Accounting metrics.Accounting
	Replicas   []ReplicaRecord
	ScaleLog   []ScaleEvent
	Ejections  int
	Events     int64
}

// ScaleEvent is a logged scale action.
type ScaleEvent struct {
	At  time.Duration
	Cmd controller.ScaleCommand
}

type evKind uint8

const (
	evIterDone evKind = iota
	evScrape
	evTick
	evControl
	evProbe
	evProbeFail
	evProvisioned
	evLoaded
	evCrash
	evCrashDetected
	evReqError
	evDrainDeadline
)

type event struct {
	at   time.Duration
	seq  uint64
	kind evKind
	rep  *replica
	req  *request
	gen  uint64
}

type eventHeap []event

func (h eventHeap) less(i, j int) bool {
	if h[i].at != h[j].at {
		return h[i].at < h[j].at
	}
	return h[i].seq < h[j].seq
}

func (h *eventHeap) push(e event) {
	*h = append(*h, e)
	i := len(*h) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !h.less(i, p) {
			break
		}
		(*h)[i], (*h)[p] = (*h)[p], (*h)[i]
		i = p
	}
}

func (h *eventHeap) pop() event {
	old := *h
	top := old[0]
	n := len(old) - 1
	old[0] = old[n]
	*h = old[:n]
	i := 0
	for {
		l, r, m := 2*i+1, 2*i+2, i
		if l < n && h.less(l, m) {
			m = l
		}
		if r < n && h.less(r, m) {
			m = r
		}
		if m == i {
			break
		}
		(*h)[i], (*h)[m] = (*h)[m], (*h)[i]
		i = m
	}
	return top
}

type replica struct {
	id, model, class string
	gpus             int
	cls              Class
	eng              *engine.Engine
	state            registry.State
	crashed          bool
	iterActive       bool
	gen              uint64
	provisionAt      time.Duration
	readyAt          time.Duration
	servingSince     time.Duration // start of the current serving (ready/draining) span
	servingTime      time.Duration
	endAt            time.Duration
	inflight         map[*request]struct{}
}

type request struct {
	tr        trace.Request
	rec       metrics.RequestRecord
	seq       *engine.Seq
	rep       *replica
	lastToken time.Duration
	finished  bool
	attempt   int
}

// Sim is one simulation run. Construct with New, then call Run once.
type Sim struct {
	cfg      Config
	classes  map[string]Class
	reg      *registry.Registry
	disp     *controller.Dispatcher
	ctl      Control
	targets  slo.Targets
	weights  metrics.Weights
	hooks    Hooks
	now      time.Duration
	end      time.Duration
	heap     eventHeap
	seq      uint64
	reps     map[string]*replica
	repOrder []*replica
	reqs     []*request
	byID     map[string]*request
	itl      metrics.Histogram
	counter  map[string]int
	scaleLog []ScaleEvent
	events   int64
	finished int
}

// New prepares a run. The dispatcher must have been built over reg with the
// class priors of classes; ctl may be nil.
func New(cfg Config, classes map[string]Class, reg *registry.Registry, disp *controller.Dispatcher, ctl Control,
	targets slo.Targets, w metrics.Weights, reqs []trace.Request, hooks Hooks) (*Sim, error) {
	if cfg.DurationS <= 0 || cfg.ScrapeIntervalS <= 0 || cfg.TickIntervalS <= 0 {
		return nil, fmt.Errorf("sim: duration, scrape and tick intervals must be positive")
	}
	s := &Sim{
		cfg: cfg, classes: classes, reg: reg, disp: disp, ctl: ctl, targets: targets, weights: w, hooks: hooks,
		reps: map[string]*replica{}, byID: make(map[string]*request, len(reqs)), counter: map[string]int{},
		end: sec(cfg.DurationS + cfg.DrainHorizonS),
	}
	for _, tr := range reqs {
		class, _ := targets.For(tr.SLOClass)
		r := &request{tr: tr, rec: metrics.RequestRecord{
			ID: tr.ID, Model: tr.Model, SLOClass: class, PromptTokens: tr.PromptTokens,
			OutputTokens: tr.OutputTokens, Arrival: sec(tr.ArrivalS), State: metrics.Unfinished,
		}}
		s.reqs = append(s.reqs, r)
		s.byID[tr.ID] = r
	}
	disp.Fresh = s.fresh
	for _, ir := range cfg.Initial {
		if _, ok := classes[ir.Class]; !ok {
			return nil, fmt.Errorf("sim: unknown class %q", ir.Class)
		}
		for i := 0; i < ir.Count; i++ {
			rep := s.addReplica(ir.Model, ir.Class, 0)
			s.setState(rep, registry.Ready)
		}
	}
	for _, c := range cfg.Crashes {
		rep := s.reps[c.Replica]
		if rep == nil {
			return nil, fmt.Errorf("sim: crash of unknown replica %q", c.Replica)
		}
		s.schedule(sec(c.AtS), evCrash, rep, nil)
	}
	return s, nil
}

func sec(x float64) time.Duration { return time.Duration(x*1e9 + 0.5) }

func (s *Sim) schedule(at time.Duration, k evKind, rep *replica, req *request) {
	s.seq++
	e := event{at: at, seq: s.seq, kind: k, rep: rep, req: req}
	if rep != nil {
		e.gen = rep.gen
	}
	s.heap.push(e)
}

// Now returns the virtual time.
func (s *Sim) Now() time.Duration { return s.now }

func (s *Sim) addReplica(model, class string, now time.Duration) *replica {
	cls := s.classes[class]
	if s.cfg.PrefixCacheGroups > 0 {
		cls.Params.PrefixCacheGroups = s.cfg.PrefixCacheGroups
	}
	id := fmt.Sprintf("%s-%03d", class, s.counter[class])
	s.counter[class]++
	rep := &replica{
		id: id, model: model, class: class, gpus: cls.Spec.GPUs, cls: cls, eng: engine.New(cls.Params),
		state: registry.Provisioning, provisionAt: now, readyAt: -1, endAt: -1, inflight: map[*request]struct{}{},
	}
	s.reps[id] = rep
	s.repOrder = append(s.repOrder, rep)
	_ = s.reg.Add(registry.Replica{ID: id, Model: model, Endpoint: "sim://" + id, Class: class, GPUs: rep.gpus, State: registry.Provisioning, Healthy: true})
	return rep
}

func (s *Sim) setState(rep *replica, st registry.State) {
	serving := func(x registry.State) bool { return x == registry.Ready || x == registry.Draining }
	if serving(rep.state) && !serving(st) && !rep.crashed { // a crash already stopped the clock
		rep.servingTime += s.now - rep.servingSince
	}
	if !serving(rep.state) && serving(st) {
		rep.servingSince = s.now
	}
	if st == registry.Ready && rep.readyAt < 0 {
		rep.readyAt = s.now
	}
	if !st.Live() && rep.endAt < 0 {
		rep.endAt = s.now
	}
	rep.state = st
	if !st.Live() {
		_ = s.reg.Remove(rep.id)
		s.disp.Forget(rep.id)
		return
	}
	_ = s.reg.Update(rep.id, func(r *registry.Replica) { r.State = st })
}

func (s *Sim) fresh(id string) (metrics.ReplicaSnapshot, bool) {
	rep := s.reps[id]
	if rep == nil || rep.crashed {
		return metrics.ReplicaSnapshot{}, false
	}
	snap := s.snapshot(rep)
	snap.RemainingDecodeTokens = rep.eng.State().RemainingDecode
	return snap, true
}

// snapshot is what a scrape publishes (no oracle-only fields).
func (s *Sim) snapshot(rep *replica) metrics.ReplicaSnapshot {
	st := rep.eng.State()
	return metrics.ReplicaSnapshot{
		ReplicaID: rep.id, At: s.now, Running: st.Running, Waiting: st.Waiting,
		WaitingPromptTokens: st.WaitingPromptTokens, KVUsedTokens: st.KVUsed, KVCapacityTokens: st.KVCapacity,
		Preemptions: st.Preemptions, CompletedRequests: st.Completed, GeneratedTokens: st.Generated, BusyTime: st.Busy,
		PrefixHits: st.PrefixHits, PrefixQueries: st.PrefixQueries,
	}
}

// Run executes the simulation to the end of the drain horizon.
func (s *Sim) Run() Result {
	s.schedule(0, evScrape, nil, nil)
	s.schedule(sec(s.cfg.TickIntervalS), evTick, nil, nil)
	if s.ctl != nil && s.cfg.ControlIntervalS > 0 {
		s.schedule(sec(s.cfg.ControlIntervalS), evControl, nil, nil)
	}
	if s.cfg.ProbeIntervalS > 0 {
		s.schedule(sec(s.cfg.ProbeIntervalS), evProbe, nil, nil)
	}
	next := 0
	for {
		var e event
		haveHeap := len(s.heap) > 0
		haveArr := next < len(s.reqs) && s.reqs[next].rec.Arrival < s.end
		if !haveHeap && !haveArr {
			break
		}
		if haveArr && (!haveHeap || s.reqs[next].rec.Arrival < s.heap[0].at) {
			r := s.reqs[next]
			next++
			s.advance(r.rec.Arrival)
			s.arrive(r)
		} else {
			e = s.heap.pop()
			if e.at >= s.end {
				break
			}
			s.advance(e.at)
			s.handle(e)
		}
		s.events++
		if s.hooks.AfterEvent != nil {
			s.hooks.AfterEvent(s)
		}
	}
	s.advance(s.end)
	return s.finish()
}

func (s *Sim) advance(t time.Duration) {
	if t < s.now {
		panic(fmt.Sprintf("sim: time went backwards: %v -> %v", s.now, t))
	}
	s.now = t
}

func (s *Sim) arrive(r *request) {
	if s.ctl != nil {
		s.ctl.ObserveArrival(s.now, r.tr.Model)
	}
	s.exec(s.disp.Arrive(s.now, routing.Request{
		ID: r.tr.ID, Model: r.tr.Model, PromptTokens: r.tr.PromptTokens,
		PrefixGroup: r.tr.PrefixGroup, SLOClass: r.tr.SLOClass,
	}))
}

// exec carries out dispatcher commands.
func (s *Sim) exec(cmds []controller.Command) {
	for _, c := range cmds {
		r := s.byID[c.ReqID]
		if r == nil || r.finished {
			panic(fmt.Sprintf("sim: command %v for unknown or finished request %s", c.Kind, c.ReqID))
		}
		r.rec.RouterQueue = c.RouterQueue
		switch c.Kind {
		case controller.Send:
			s.send(r, s.reps[c.ReplicaID], c.Attempt)
		case controller.Reject:
			s.terminate(r, metrics.Rejected)
		case controller.Fail:
			if c.Attempt > 0 {
				r.rec.Retries = c.Attempt - 1
			}
			s.terminate(r, metrics.Failed)
		case controller.TimeOut:
			s.terminate(r, metrics.TimedOut)
		}
	}
}

func (s *Sim) send(r *request, rep *replica, attempt int) {
	r.attempt = attempt
	r.rec.Retries = attempt - 1
	r.rep = rep
	r.rec.Replica = rep.id
	if rep.crashed || rep.state != registry.Ready {
		// The connection fails after the connect timeout.
		s.schedule(s.now+sec(s.cfg.ConnectTimeoutS), evReqError, rep, r)
		return
	}
	r.seq = &engine.Seq{ID: r.tr.ID, Prompt: r.tr.PromptTokens, Output: r.tr.OutputTokens, Tag: r}
	if r.tr.PrefixGroup != "" {
		r.seq.PrefixGroup, r.seq.PrefixTokens = r.tr.PrefixGroup, s.cfg.PrefixTokens
	}
	if err := rep.eng.Add(r.seq, s.now); err != nil {
		// Non-retryable (prompt+output beyond the model length).
		r.seq = nil
		s.exec(s.disp.Error(s.now, r.tr.ID, false, err.Error()))
		return
	}
	rep.inflight[r] = struct{}{}
	s.kick(rep)
}

func (s *Sim) kick(rep *replica) {
	if rep.iterActive || rep.crashed {
		return
	}
	if d, ok := rep.eng.Start(s.now); ok {
		rep.iterActive = true
		s.schedule(s.now+d, evIterDone, rep, nil)
	}
}

func (s *Sim) terminate(r *request, st metrics.Terminal) {
	if r.finished {
		panic("sim: request terminated twice: " + r.tr.ID)
	}
	r.finished = true
	r.rec.State = st
	if st == metrics.Completed {
		_, tg := s.targets.For(r.rec.SLOClass)
		r.rec.Met = slo.Met(true, r.rec.TTFT(), r.rec.TPOT(), r.rec.E2E(), r.rec.OutputTokens, tg)
	}
	s.finished++
	if s.ctl != nil {
		s.ctl.ObserveFinish(s.now, r.rec)
	}
}

func (s *Sim) handle(e event) {
	rep := e.rep
	if rep != nil && e.gen != rep.gen && (e.kind == evIterDone) {
		return // stale iteration of a crashed engine
	}
	switch e.kind {
	case evIterDone:
		s.iterDone(rep)
	case evScrape:
		for _, r := range s.repOrder {
			if !r.crashed && (r.state == registry.Ready || r.state == registry.Draining) {
				snap := s.snapshot(r)
				if s.ctl != nil {
					s.ctl.ObserveSnapshot(s.now, r.model, snap)
				}
				s.exec(s.disp.Snapshot(s.now, snap))
			}
		}
		s.schedule(s.now+sec(s.cfg.ScrapeIntervalS), evScrape, nil, nil)
	case evTick:
		s.exec(s.disp.Tick(s.now))
		s.schedule(s.now+sec(s.cfg.TickIntervalS), evTick, nil, nil)
	case evControl:
		for _, c := range s.ctl.Step(s.now) {
			s.applyScale(c)
		}
		s.schedule(s.now+sec(s.cfg.ControlIntervalS), evControl, nil, nil)
	case evProbe:
		for _, r := range s.repOrder {
			if r.state == registry.Ready && r.crashed {
				s.schedule(s.now+sec(s.cfg.ProbeTimeoutS), evProbeFail, r, nil)
			}
		}
		s.schedule(s.now+sec(s.cfg.ProbeIntervalS), evProbe, nil, nil)
	case evProbeFail:
		s.exec(s.disp.ProbeResult(s.now, rep.id, false))
	case evProvisioned:
		if rep.state == registry.Provisioning {
			s.setState(rep, registry.Loading)
			s.schedule(s.now+rep.cls.Load, evLoaded, rep, nil)
		}
	case evLoaded:
		if rep.state == registry.Loading {
			s.setState(rep, registry.Ready)
			s.exec(s.disp.Drain(s.now))
		}
	case evCrash:
		s.crash(rep)
	case evCrashDetected:
		if rep.state.Live() {
			s.setState(rep, registry.Failed)
		}
	case evReqError:
		r := e.req
		if !r.finished && r.rep == rep && r.seq == nil {
			s.exec(s.disp.Error(s.now, r.tr.ID, true, "replica unavailable"))
		}
	case evDrainDeadline:
		if rep.state == registry.Draining && !rep.crashed {
			s.abortAll(rep, "drain grace period expired")
			s.setState(rep, registry.Terminated)
		}
	}
}

func (s *Sim) iterDone(rep *replica) {
	rep.iterActive = false
	for _, ev := range rep.eng.Finish(s.now) {
		r := ev.Seq.Tag.(*request)
		if ev.Kind == engine.Preempted {
			r.rec.Preemptions++
			continue
		}
		if ev.First {
			r.rec.FirstToken = s.now
			r.rec.HasFirstToken = true
			s.disp.FirstToken(s.now, r.tr.ID)
		} else {
			s.itl.Add(s.now - r.lastToken)
		}
		r.lastToken = s.now
		if ev.Done {
			r.rec.Completion = s.now
			r.rec.ReplicaQueue = r.seq.FirstScheduledAt - r.seq.EnqueuedAt
			delete(rep.inflight, r)
			s.terminate(r, metrics.Completed)
			s.exec(s.disp.Complete(s.now, r.tr.ID, r.tr.OutputTokens))
		}
	}
	s.kick(rep)
	s.maybeRetire(rep)
}

// maybeRetire terminates a draining replica once it is idle.
func (s *Sim) maybeRetire(rep *replica) {
	if rep.state == registry.Draining && len(rep.inflight) == 0 && !rep.eng.HasWork() {
		s.setState(rep, registry.Terminated)
	}
}

func (s *Sim) crash(rep *replica) {
	if rep.crashed || !rep.state.Live() {
		return
	}
	rep.crashed = true
	if rep.state == registry.Ready || rep.state == registry.Draining {
		rep.servingTime += s.now - rep.servingSince // a dead replica serves nothing
	}
	rep.gen++
	rep.iterActive = false
	rep.eng.Reset(s.now)
	victims := s.sortedInflight(rep)
	rep.inflight = map[*request]struct{}{}
	for _, r := range victims {
		r.seq = nil
		s.schedule(s.now+sec(s.cfg.FailDelayS), evReqError, rep, r)
	}
	s.schedule(s.now+sec(s.cfg.ExecutorDetectS), evCrashDetected, rep, nil)
}

func (s *Sim) sortedInflight(rep *replica) []*request {
	out := make([]*request, 0, len(rep.inflight))
	for r := range rep.inflight {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].tr.ID < out[j].tr.ID })
	return out
}

// abortAll ends every request on a replica that is being shut down (the
// iteration in progress, if any, is discarded).
func (s *Sim) abortAll(rep *replica, reason string) {
	rep.gen++
	rep.iterActive = false
	rep.eng.Reset(s.now)
	victims := s.sortedInflight(rep)
	rep.inflight = map[*request]struct{}{}
	for _, r := range victims {
		r.seq = nil
		s.exec(s.disp.Error(s.now, r.tr.ID, true, reason))
	}
}

// Apply executes a scale command at the current virtual time; it makes the
// simulated cluster an executor.Executor.
func (s *Sim) Apply(_ context.Context, c controller.ScaleCommand) error {
	s.applyScale(c)
	return nil
}

func (s *Sim) applyScale(c controller.ScaleCommand) {
	switch c.Kind {
	case controller.Provision:
		if _, ok := s.classes[c.Class]; !ok {
			return
		}
		rep := s.addReplica(c.Model, c.Class, s.now)
		c.ReplicaID = rep.id
		s.schedule(s.now+rep.cls.Provision, evProvisioned, rep, nil)
	case controller.Drain:
		rep := s.reps[c.ReplicaID]
		if rep == nil || !rep.state.Live() || rep.state == registry.Draining {
			return
		}
		if rep.state != registry.Ready {
			// Not serving yet: cancel the start-up immediately.
			s.setState(rep, registry.Terminated)
		} else {
			s.setState(rep, registry.Draining)
			s.schedule(s.now+sec(s.cfg.DrainGraceS), evDrainDeadline, rep, nil)
			s.maybeRetire(rep)
		}
	}
	s.scaleLog = append(s.scaleLog, ScaleEvent{At: s.now, Cmd: c})
}

func (s *Sim) finish() Result {
	var acc metrics.Accounting
	acc.RunDuration = s.end
	acc.BudgetGPUs = s.cfg.BudgetGPUs
	acc.ITL = &s.itl
	acc.ScaleActions = len(s.scaleLog)
	var recs []ReplicaRecord
	for _, rep := range s.repOrder {
		end := rep.endAt
		if end < 0 {
			end = s.end
		}
		serving := rep.servingTime
		if (rep.state == registry.Ready || rep.state == registry.Draining) && !rep.crashed {
			serving += s.end - rep.servingSince
		}
		st := rep.eng.State()
		acc.GPUSeconds += float64(rep.gpus) * (end - rep.provisionAt).Seconds()
		acc.ReadySeconds += serving.Seconds()
		acc.BusySeconds += st.Busy.Seconds()
		acc.KVTokenSeconds += st.KVIntegral
		acc.KVCapacityTokenSeconds += float64(st.KVCapacity) * serving.Seconds()
		recs = append(recs, ReplicaRecord{ID: rep.id, Model: rep.model, Class: rep.class, GPUs: rep.gpus,
			ProvisionAt: rep.provisionAt, ReadyAt: rep.readyAt, EndAt: end, Final: rep.state})
	}
	acc.PeakReplicas = s.peakLive()
	out := Result{Accounting: acc, Replicas: recs, ScaleLog: s.scaleLog, Ejections: s.disp.Health().Ejections, Events: s.events}
	out.Records = make([]metrics.RequestRecord, len(s.reqs))
	for i, r := range s.reqs {
		out.Records[i] = r.rec
	}
	return out
}

// peakLive returns the maximum number of simultaneously live replicas.
func (s *Sim) peakLive() int {
	type pt struct {
		t time.Duration
		d int
	}
	var pts []pt
	for _, rep := range s.repOrder {
		pts = append(pts, pt{rep.provisionAt, 1})
		end := rep.endAt
		if end < 0 {
			end = s.end
		}
		pts = append(pts, pt{end, -1})
	}
	sort.Slice(pts, func(i, j int) bool {
		if pts[i].t != pts[j].t {
			return pts[i].t < pts[j].t
		}
		return pts[i].d < pts[j].d
	})
	cur, peak := 0, 0
	for _, p := range pts {
		cur += p.d
		if cur > peak {
			peak = cur
		}
	}
	return peak
}

// LiveGPUs returns the GPUs held by live replicas now (for invariant tests).
func (s *Sim) LiveGPUs() int {
	n := 0
	for _, rep := range s.repOrder {
		if rep.state.Live() {
			n += rep.gpus
		}
	}
	return n
}

// KVViolation reports a replica whose KV occupancy exceeds its capacity.
func (s *Sim) KVViolation() string {
	for _, rep := range s.repOrder {
		st := rep.eng.State()
		if st.KVUsed+st.KVReserved > st.KVCapacity || st.KVUsed < 0 {
			return rep.id
		}
	}
	return ""
}

// Finished returns the number of requests in a terminal state so far.
func (s *Sim) Finished() int { return s.finished }
