package controller

import (
	"sort"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/admission"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
)

// DispatchConfig parameterises the router runtime.
type DispatchConfig struct {
	// RouterQueueTimeoutS: a request waiting longer in the router queue
	// (one stint) is timed out.
	RouterQueueTimeoutS float64 `json:"router_queue_timeout_s"`
	// MaxRetries bounds re-dispatches after replica errors before the first
	// token.
	MaxRetries int `json:"max_retries"`
}

// DefaultDispatch returns the documented defaults.
func DefaultDispatch() DispatchConfig {
	return DispatchConfig{RouterQueueTimeoutS: 60, MaxRetries: 2}
}

// CmdKind is what the caller must do with a request.
type CmdKind int

// Command kinds.
const (
	Send    CmdKind = iota // forward the request to ReplicaID
	Reject                 // refuse the request
	Fail                   // fail the request (retries exhausted or after first token)
	TimeOut                // the request timed out in the router queue
)

// Command is an instruction from the dispatcher to its host (simulator or
// HTTP proxy).
type Command struct {
	Kind      CmdKind
	ReqID     string
	ReplicaID string
	Reason    string
	// RouterQueue is the request's cumulative router-queue time so far;
	// Attempt is the dispatch number (1 = first try).
	RouterQueue time.Duration
	Attempt     int
}

type pending struct {
	req          routing.Request
	arrival      time.Duration
	queued       bool
	queuedAt     time.Duration
	routerQueue  time.Duration
	attempts     int
	replica      string
	dispatchedAt time.Duration
	firstToken   bool
	excluded     []string
}

type snapState struct {
	snap      metrics.ReplicaSnapshot
	sentSince int
}

// Dispatcher is the router runtime. It is not safe for concurrent use; the
// live proxy serialises calls with a mutex, the simulator is single-threaded.
type Dispatcher struct {
	cfg      DispatchConfig
	reg      *registry.Registry
	classes  map[string]registry.ClassInfo
	router   routing.Router
	observer routing.Observer
	oracle   bool
	adm      admission.Policy
	health   *Health

	// Fresh returns a replica's true state; only the simulator sets it, and
	// it is used only for oracle routers.
	Fresh func(replicaID string) (metrics.ReplicaSnapshot, bool)
	// SLOError returns a replica's windowed SLO error (optional).
	SLOError func(replicaID string) float64

	reqs     map[string]*pending
	queues   map[string][]*pending
	inflight map[string]int
	snaps    map[string]*snapState
	buf      []routing.ReplicaView
}

// NewDispatcher wires a dispatcher. adm may be nil (admit everything).
func NewDispatcher(cfg DispatchConfig, reg *registry.Registry, classes map[string]registry.ClassInfo,
	router routing.Router, adm admission.Policy, health *Health) *Dispatcher {
	d := &Dispatcher{
		cfg: cfg, reg: reg, classes: classes, router: router, adm: adm, health: health,
		reqs: map[string]*pending{}, queues: map[string][]*pending{},
		inflight: map[string]int{}, snaps: map[string]*snapState{},
	}
	if o, ok := router.(routing.Observer); ok {
		d.observer = o
	}
	if o, ok := router.(routing.Oracle); ok && o.IsOracle() {
		d.oracle = true
	}
	return d
}

// IsOracle reports whether the router needs the simulator's true state.
func (d *Dispatcher) IsOracle() bool { return d.oracle }

// Health returns the health tracker.
func (d *Dispatcher) Health() *Health { return d.health }

// InFlight returns the router-local in-flight count of a replica.
func (d *Dispatcher) InFlight(replicaID string) int { return d.inflight[replicaID] }

// TotalInFlight returns the router-local in-flight total of a model.
func (d *Dispatcher) TotalInFlight(model string) int {
	n := 0
	for _, r := range d.reg.ForModel(model) {
		n += d.inflight[r.ID]
	}
	return n
}

// QueueLen returns the router-queue length of a model.
func (d *Dispatcher) QueueLen(model string) int { return len(d.queues[model]) }

// LatestSnapshot returns the latest snapshot received from a replica.
func (d *Dispatcher) LatestSnapshot(replicaID string) (metrics.ReplicaSnapshot, bool) {
	s := d.snaps[replicaID]
	if s == nil {
		return metrics.ReplicaSnapshot{}, false
	}
	return s.snap, true
}

// view builds the eligible-replica view. The returned slice is reused across
// calls; policies must not retain it.
func (d *Dispatcher) view(now time.Duration, model string, exclude []string) routing.View {
	d.buf = d.buf[:0]
	reps := d.reg.ForModel(model)
	for pass := 0; pass < 2; pass++ {
		for _, r := range reps {
			if r.State != registry.Ready || !d.health.Healthy(r.ID) {
				continue
			}
			if pass == 0 && contains(exclude, r.ID) {
				continue
			}
			rv := routing.ReplicaView{ID: r.ID, Class: d.classes[r.Class], InFlight: d.inflight[r.ID]}
			if s := d.snaps[r.ID]; s != nil {
				rv.Scraped, rv.HasScrape, rv.SentSinceScrape = s.snap, true, s.sentSince
			}
			if d.SLOError != nil {
				rv.SLOError = d.SLOError(r.ID)
			}
			if d.oracle && d.Fresh != nil {
				if f, ok := d.Fresh(r.ID); ok {
					fc := f
					rv.Fresh = &fc
				}
			}
			d.buf = append(d.buf, rv)
		}
		// Exclusion (replicas that already failed this request) applies only
		// while another eligible replica remains.
		if len(d.buf) > 0 || len(exclude) == 0 {
			break
		}
	}
	return routing.View{Now: now, Model: model, Replicas: d.buf, RouterQueueLen: len(d.queues[model])}
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// Arrive handles a new request at time now.
func (d *Dispatcher) Arrive(now time.Duration, req routing.Request) []Command {
	p := &pending{req: req, arrival: now}
	d.reqs[req.ID] = p
	var cmds []Command
	if len(d.queues[req.Model]) > 0 {
		cmds = d.drainModel(now, req.Model, cmds)
	}
	v := d.view(now, req.Model, nil)
	verdict := admission.Decision{Action: routing.Dispatch}
	if d.adm != nil {
		verdict = d.adm.Admit(req, v)
	}
	switch {
	case verdict.Action == routing.Reject:
		delete(d.reqs, req.ID)
		return append(cmds, Command{Kind: Reject, ReqID: req.ID, Reason: "admission: " + verdict.Reason})
	case verdict.Action == routing.Queue || len(d.queues[req.Model]) > 0 || len(v.Replicas) == 0:
		d.enqueue(now, p, false)
		return cmds
	}
	return append(cmds, d.dispatch(now, p, v))
}

func (d *Dispatcher) enqueue(now time.Duration, p *pending, head bool) {
	p.queued, p.queuedAt = true, now
	q := d.queues[p.req.Model]
	if head {
		q = append([]*pending{p}, q...)
	} else {
		q = append(q, p)
	}
	d.queues[p.req.Model] = q
}

// dispatch routes p over the non-empty view v.
func (d *Dispatcher) dispatch(now time.Duration, p *pending, v routing.View) Command {
	dec := d.router.Route(p.req, v)
	if dec.Action == routing.Reject {
		delete(d.reqs, p.req.ID)
		return Command{Kind: Reject, ReqID: p.req.ID, Reason: "router: " + dec.Reason, RouterQueue: p.routerQueue}
	}
	id := dec.ReplicaID
	if dec.Action != routing.Dispatch || !inView(v, id) {
		// A router may only pick from the view; fall back deterministically.
		id = v.Replicas[0].ID
	}
	p.attempts++
	p.replica = id
	p.dispatchedAt = now
	d.inflight[id]++
	if s := d.snaps[id]; s != nil {
		s.sentSince++
	}
	return Command{Kind: Send, ReqID: p.req.ID, ReplicaID: id, RouterQueue: p.routerQueue, Attempt: p.attempts}
}

func inView(v routing.View, id string) bool {
	for _, r := range v.Replicas {
		if r.ID == id {
			return true
		}
	}
	return false
}

// drainModel dispatches queued requests of one model in FIFO order while
// admission and eligibility allow.
func (d *Dispatcher) drainModel(now time.Duration, model string, cmds []Command) []Command {
	for len(d.queues[model]) > 0 {
		p := d.queues[model][0]
		v := d.view(now, model, p.excluded)
		if len(v.Replicas) == 0 {
			break
		}
		if d.adm != nil && p.attempts == 0 {
			verdict := d.adm.Admit(p.req, v)
			if verdict.Action == routing.Queue {
				break
			}
			if verdict.Action == routing.Reject {
				d.popHead(now, model, p)
				delete(d.reqs, p.req.ID)
				cmds = append(cmds, Command{Kind: Reject, ReqID: p.req.ID, Reason: "admission: " + verdict.Reason, RouterQueue: p.routerQueue})
				continue
			}
		}
		d.popHead(now, model, p)
		v = d.view(now, model, p.excluded) // admission may not change state, but rebuild for safety
		cmds = append(cmds, d.dispatch(now, p, v))
	}
	return cmds
}

func (d *Dispatcher) popHead(now time.Duration, model string, p *pending) {
	d.queues[model] = d.queues[model][1:]
	if len(d.queues[model]) == 0 {
		delete(d.queues, model)
	}
	p.queued = false
	p.routerQueue += now - p.queuedAt
}

// Drain retries queued requests of every model (call after a replica became
// eligible or capacity was freed).
func (d *Dispatcher) Drain(now time.Duration) []Command {
	if len(d.queues) == 0 {
		return nil
	}
	models := make([]string, 0, len(d.queues))
	for m := range d.queues {
		models = append(models, m)
	}
	sort.Strings(models)
	var cmds []Command
	for _, m := range models {
		cmds = d.drainModel(now, m, cmds)
	}
	return cmds
}

// FirstToken records the first token of a dispatched request.
func (d *Dispatcher) FirstToken(now time.Duration, reqID string) {
	p := d.reqs[reqID]
	if p == nil || p.firstToken {
		return
	}
	p.firstToken = true
	d.health.Success(p.replica, now)
	if d.observer != nil {
		d.observer.Observe(routing.Outcome{ReplicaID: p.replica, Now: now, Kind: routing.FirstToken, Latency: now - p.dispatchedAt})
	}
}

// Complete records a completed request and drains the queues.
func (d *Dispatcher) Complete(now time.Duration, reqID string, outputTokens int) []Command {
	p := d.reqs[reqID]
	if p == nil {
		return nil
	}
	d.release(p)
	if d.observer != nil {
		d.observer.Observe(routing.Outcome{ReplicaID: p.replica, Now: now, Kind: routing.Completed, Latency: now - p.dispatchedAt, OutputTokens: outputTokens})
	}
	delete(d.reqs, reqID)
	return d.Drain(now)
}

func (d *Dispatcher) release(p *pending) {
	if p.replica != "" {
		d.inflight[p.replica]--
		if d.inflight[p.replica] <= 0 {
			delete(d.inflight, p.replica)
		}
	}
}

// Error records a replica error for a dispatched request. Before the first
// token and within the retry limit the request is re-dispatched (or queued at
// the head when no other replica is eligible); otherwise it fails.
func (d *Dispatcher) Error(now time.Duration, reqID string, retryable bool, reason string) []Command {
	p := d.reqs[reqID]
	if p == nil {
		return nil
	}
	d.release(p)
	failed := p.replica
	if d.health.Failure(failed, now) {
		d.markHealth(failed, false)
	}
	if d.observer != nil {
		d.observer.Observe(routing.Outcome{ReplicaID: failed, Now: now, Kind: routing.Failed, Latency: now - p.dispatchedAt})
	}
	var cmds []Command
	if retryable && !p.firstToken && p.attempts <= d.cfg.MaxRetries {
		p.excluded = append(p.excluded, failed)
		p.replica = ""
		v := d.view(now, p.req.Model, p.excluded)
		if len(v.Replicas) == 0 {
			d.enqueue(now, p, true)
		} else {
			cmds = append(cmds, d.dispatch(now, p, v))
		}
	} else {
		delete(d.reqs, reqID)
		cmds = append(cmds, Command{Kind: Fail, ReqID: reqID, ReplicaID: failed, Reason: reason, RouterQueue: p.routerQueue, Attempt: p.attempts})
	}
	return append(cmds, d.Drain(now)...)
}

// Cancel drops a request (client went away) without health consequences.
func (d *Dispatcher) Cancel(now time.Duration, reqID string) []Command {
	p := d.reqs[reqID]
	if p == nil {
		return nil
	}
	if p.queued {
		q := d.queues[p.req.Model]
		for i, x := range q {
			if x == p {
				d.queues[p.req.Model] = append(q[:i], q[i+1:]...)
				break
			}
		}
		if len(d.queues[p.req.Model]) == 0 {
			delete(d.queues, p.req.Model)
		}
	} else {
		d.release(p)
	}
	delete(d.reqs, reqID)
	return d.Drain(now)
}

// Snapshot ingests a replica snapshot and drains the queues.
func (d *Dispatcher) Snapshot(now time.Duration, s metrics.ReplicaSnapshot) []Command {
	d.snaps[s.ReplicaID] = &snapState{snap: s}
	return d.Drain(now)
}

// ProbeResult ingests an active health probe.
func (d *Dispatcher) ProbeResult(now time.Duration, replicaID string, ok bool) []Command {
	if ok {
		return nil
	}
	if d.health.ProbeFailure(replicaID, now) {
		d.markHealth(replicaID, false)
	}
	return nil
}

// Tick reinstates replicas whose ejection expired, times out queued requests,
// and drains the queues.
func (d *Dispatcher) Tick(now time.Duration) []Command {
	for _, id := range d.health.Tick(now) {
		d.markHealth(id, true)
	}
	var cmds []Command
	timeout := secs(d.cfg.RouterQueueTimeoutS)
	models := make([]string, 0, len(d.queues))
	for m := range d.queues {
		models = append(models, m)
	}
	sort.Strings(models)
	for _, m := range models {
		keep := d.queues[m][:0]
		for _, p := range d.queues[m] {
			if timeout > 0 && now-p.queuedAt >= timeout {
				p.routerQueue += now - p.queuedAt
				delete(d.reqs, p.req.ID)
				cmds = append(cmds, Command{Kind: TimeOut, ReqID: p.req.ID, RouterQueue: p.routerQueue, Attempt: p.attempts})
				continue
			}
			keep = append(keep, p)
		}
		if len(keep) == 0 {
			delete(d.queues, m)
		} else {
			d.queues[m] = keep
		}
	}
	return append(cmds, d.Drain(now)...)
}

// Forget drops all state about a removed replica.
func (d *Dispatcher) Forget(replicaID string) {
	d.health.Forget(replicaID)
	delete(d.snaps, replicaID)
}

// Pending returns the number of requests the dispatcher still tracks.
func (d *Dispatcher) Pending() int { return len(d.reqs) }

func (d *Dispatcher) markHealth(id string, healthy bool) {
	_ = d.reg.Update(id, func(r *registry.Replica) { r.Healthy = healthy })
}
