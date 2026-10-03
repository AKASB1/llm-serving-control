// Package proxy is the live data plane: an OpenAI-compatible HTTP front end
// (/v1/chat/completions, /v1/completions) that routes by the request's model
// through the same dispatcher, health tracker, admission, routing, and control
// loop code the simulator uses. It passes server-sent events through,
// retries a request on another replica only before the first byte reaches the
// client, cancels the upstream request when the client goes away, answers
// rejections with 429 and Retry-After, probes and scrapes replicas, exposes
// Prometheus metrics, and has admin endpoints to register, remove, and list
// replicas.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/backends"
	"github.com/AKASB1/llm-serving-control/internal/clock"
	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/executor"
	"github.com/AKASB1/llm-serving-control/internal/metrics"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/store"
)

// Config parameterises the proxy's loops and responses.
type Config struct {
	ProbeInterval   time.Duration
	ProbeTimeout    time.Duration
	ScrapeInterval  time.Duration
	TickInterval    time.Duration
	ControlInterval time.Duration
	// RetryAfter is sent with 429 (rejected) and 503 (timed out) responses.
	RetryAfter time.Duration
	// MaxBody bounds request bodies (bytes).
	MaxBody int64
}

// DefaultConfig returns the documented defaults.
func DefaultConfig() Config {
	return Config{ProbeInterval: 2 * time.Second, ProbeTimeout: time.Second, ScrapeInterval: time.Second,
		TickInterval: 200 * time.Millisecond, ControlInterval: 5 * time.Second, RetryAfter: time.Second, MaxBody: 16 << 20}
}

// Options are the optional collaborators.
type Options struct {
	Loop     *controller.Loop
	Executor executor.Executor
	Scraper  backends.Scraper
	Client   *http.Client
	Logger   *slog.Logger
	// Store persists replicas registered or removed through the admin API.
	Store store.ReplicaStore
}

// Server is the live control plane front end.
type Server struct {
	cfg     Config
	clk     clock.Clock
	reg     *registry.Registry
	disp    *controller.Dispatcher
	targets slo.Targets
	opt     Options
	mu      sync.Mutex // serialises every dispatcher and loop call
	waiters map[string]chan controller.Command
	seq     atomic.Uint64
	m       *promMetrics
}

// New wires a server. Oracle routers are refused: they need the simulator's
// true replica state.
func New(cfg Config, clk clock.Clock, reg *registry.Registry, disp *controller.Dispatcher, targets slo.Targets, opt Options) (*Server, error) {
	if disp.IsOracle() {
		return nil, errors.New("proxy: oracle routing policies need the simulator's true state and cannot run live")
	}
	if opt.Client == nil {
		opt.Client = &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 60 * time.Second, MaxIdleConnsPerHost: 64}}
	}
	if opt.Logger == nil {
		opt.Logger = slog.Default()
	}
	if cfg.MaxBody <= 0 {
		cfg.MaxBody = DefaultConfig().MaxBody
	}
	return &Server{cfg: cfg, clk: clk, reg: reg, disp: disp, targets: targets, opt: opt,
		waiters: map[string]chan controller.Command{}, m: newMetrics()}, nil
}

// Handler returns the HTTP API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.complete)
	mux.HandleFunc("POST /v1/completions", s.complete)
	mux.HandleFunc("GET /metrics", s.metrics)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok\n") })
	mux.HandleFunc("GET /admin/replicas", s.listReplicas)
	mux.HandleFunc("POST /admin/replicas", s.addReplica)
	mux.HandleFunc("DELETE /admin/replicas/{id}", s.removeReplica)
	return mux
}

// deliver hands dispatcher commands to the waiting request handlers.
// Caller holds s.mu.
func (s *Server) deliver(cmds []controller.Command) {
	for _, c := range cmds {
		if ch := s.waiters[c.ReqID]; ch != nil {
			select {
			case ch <- c:
			default:
				s.opt.Logger.Error("dropped dispatcher command: waiter full", "req", c.ReqID)
			}
		}
	}
}

// call runs fn under the dispatcher lock and delivers its commands.
func (s *Server) call(fn func(now time.Duration) []controller.Command) {
	s.mu.Lock()
	s.deliver(fn(s.clk.Now()))
	s.mu.Unlock()
}

type requestBody struct {
	Model            string                     `json:"model"`
	Stream           bool                       `json:"stream"`
	MaxTokens        int                        `json:"max_tokens"`
	Prompt           json.RawMessage            `json:"prompt"`
	Messages         []struct{ Content string } `json:"messages"`
	MockPromptTokens int                        `json:"mock_prompt_tokens"`
}

func (q requestBody) promptTokens() int {
	if q.MockPromptTokens > 0 {
		return q.MockPromptTokens
	}
	n := len(q.Prompt)
	for _, m := range q.Messages {
		n += len(m.Content)
	}
	return max(1, n/4)
}

func jsonError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}})
}

// attempt outcomes.
type outcome int

const (
	done            outcome = iota // response fully relayed
	failedBefore                   // replica error before the first byte reached the client
	failedAfter                    // replica error after the first byte (stream truncated)
	clientGone                     // the client went away
	clientErrorDone                // upstream 4xx relayed to the client
)

type request struct {
	id         string
	model      string
	class      string
	arrival    time.Duration
	firstToken time.Duration
	hasFirst   bool
	tokens     int
	retries    int
}

func (s *Server) complete(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, s.cfg.MaxBody))
	if err != nil {
		jsonError(w, http.StatusBadRequest, "cannot read body")
		return
	}
	var q requestBody
	if err := json.Unmarshal(body, &q); err != nil || q.Model == "" {
		jsonError(w, http.StatusBadRequest, "body must be JSON with a model")
		return
	}
	rq := &request{id: fmt.Sprintf("px-%d", s.seq.Add(1)), model: q.Model, class: r.Header.Get("X-SLO-Class")}
	ch := make(chan controller.Command, 4)
	s.mu.Lock()
	s.waiters[rq.id] = ch
	rq.arrival = s.clk.Now()
	if s.opt.Loop != nil {
		s.opt.Loop.ObserveArrival(rq.arrival, q.Model)
	}
	s.deliver(s.disp.Arrive(rq.arrival, routing.Request{ID: rq.id, Model: q.Model, PromptTokens: q.promptTokens(), MaxOutputTokens: q.MaxTokens, SLOClass: rq.class}))
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.waiters, rq.id)
		s.mu.Unlock()
	}()
	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			s.call(func(now time.Duration) []controller.Command { return s.disp.Cancel(now, rq.id) })
			s.finish(rq, metrics.Failed, "canceled")
			return
		case c := <-ch:
			switch c.Kind {
			case controller.Reject:
				w.Header().Set("Retry-After", strconv.Itoa(int(max(1, s.cfg.RetryAfter.Seconds()))))
				jsonError(w, http.StatusTooManyRequests, c.Reason)
				s.finish(rq, metrics.Rejected, "rejected")
				return
			case controller.TimeOut:
				w.Header().Set("Retry-After", strconv.Itoa(int(max(1, s.cfg.RetryAfter.Seconds()))))
				jsonError(w, http.StatusServiceUnavailable, "timed out waiting for a replica")
				s.finish(rq, metrics.TimedOut, "timed_out")
				return
			case controller.Fail:
				if !rq.hasFirst {
					jsonError(w, http.StatusBadGateway, c.Reason)
				}
				s.finish(rq, metrics.Failed, "failed")
				return
			case controller.Send:
				rep, ok := s.reg.Get(c.ReplicaID)
				if c.Attempt > 1 {
					rq.retries++
					s.m.inc("lsc_retries_total", "Requests re-dispatched after a replica error before the first byte.", "model", rq.model)
				}
				var res outcome
				var reason string
				if !ok {
					res, reason = failedBefore, "replica removed"
				} else {
					res, reason = s.forward(ctx, w, r, rq, rep, body, q.Stream)
				}
				switch res {
				case done, clientErrorDone:
					s.call(func(now time.Duration) []controller.Command {
						return s.disp.Complete(now, rq.id, rq.tokens)
					})
					if res == done {
						s.finish(rq, metrics.Completed, "completed")
					} else {
						s.finish(rq, metrics.Failed, "client_error")
					}
					return
				case clientGone:
					s.call(func(now time.Duration) []controller.Command { return s.disp.Cancel(now, rq.id) })
					s.finish(rq, metrics.Failed, "canceled")
					return
				case failedBefore, failedAfter:
					s.m.inc("lsc_upstream_errors_total", "Replica errors seen by the proxy.", "replica", c.ReplicaID)
					s.call(func(now time.Duration) []controller.Command {
						return s.disp.Error(now, rq.id, true, reason)
					})
					// The dispatcher answers with a new Send (retry) or Fail.
				}
			}
		}
	}
}

// forward relays one attempt. Headers and body reach the client only once
// the first body byte has arrived from the replica, so every failure up to
// that point can still be retried elsewhere.
func (s *Server) forward(ctx context.Context, w http.ResponseWriter, r *http.Request, rq *request, rep registry.Replica, body []byte, stream bool) (outcome, string) {
	up, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(rep.Endpoint, "/")+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		return failedBefore, err.Error()
	}
	up.Header.Set("Content-Type", "application/json")
	if a := r.Header.Get("Authorization"); a != "" {
		up.Header.Set("Authorization", a)
	}
	resp, err := s.opt.Client.Do(up)
	if err != nil {
		if ctx.Err() != nil {
			return clientGone, "client went away"
		}
		return failedBefore, err.Error()
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
		return failedBefore, fmt.Sprintf("replica status %d", resp.StatusCode)
	}
	buf := make([]byte, 32<<10)
	wrote := false
	fl, _ := w.(http.Flusher)
	start := func() {
		for _, h := range []string{"Content-Type", "Cache-Control"} {
			if v := resp.Header.Get(h); v != "" {
				w.Header().Set(h, v)
			}
		}
		w.Header().Set("X-Replica", rep.ID)
		w.WriteHeader(resp.StatusCode)
		wrote = true
	}
	if resp.StatusCode >= 400 {
		start()
		_, _ = io.Copy(w, resp.Body)
		return clientErrorDone, ""
	}
	if !stream {
		all, err := io.ReadAll(resp.Body)
		if err != nil {
			if ctx.Err() != nil {
				return clientGone, "client went away"
			}
			return failedBefore, err.Error()
		}
		s.firstToken(rq)
		var parsed struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			}
		}
		if json.Unmarshal(all, &parsed) == nil {
			rq.tokens = parsed.Usage.CompletionTokens
		}
		start()
		_, _ = w.Write(all)
		return done, ""
	}
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if !wrote {
				s.firstToken(rq)
				start()
			}
			rq.tokens += bytes.Count(buf[:n], []byte("data: ")) - bytes.Count(buf[:n], []byte("data: [DONE]"))
			if _, werr := w.Write(buf[:n]); werr != nil {
				return clientGone, "client went away"
			}
			if fl != nil {
				fl.Flush()
			}
		}
		if errors.Is(err, io.EOF) {
			if !wrote {
				return failedBefore, "empty response"
			}
			return done, ""
		}
		if err != nil {
			if ctx.Err() != nil {
				return clientGone, "client went away"
			}
			if !wrote {
				return failedBefore, err.Error()
			}
			return failedAfter, err.Error()
		}
	}
}

func (s *Server) firstToken(rq *request) {
	s.mu.Lock()
	rq.firstToken, rq.hasFirst = s.clk.Now(), true
	s.disp.FirstToken(rq.firstToken, rq.id)
	s.mu.Unlock()
}

// finish records the outcome in the proxy metrics and the control loop.
func (s *Server) finish(rq *request, st metrics.Terminal, label string) {
	now := s.clk.Now()
	s.m.inc("lsc_requests_total", "Requests by model and outcome.", "model", rq.model, "outcome", label)
	rec := metrics.RequestRecord{ID: rq.id, Model: rq.model, Arrival: rq.arrival, State: st, OutputTokens: max(rq.tokens, 1),
		Retries: rq.retries, FirstToken: rq.firstToken, HasFirstToken: rq.hasFirst}
	class, tg := s.targets.For(rq.class)
	rec.SLOClass = class
	if st == metrics.Completed {
		rec.Completion = now
		rec.Met = slo.Met(true, rec.TTFT(), rec.TPOT(), rec.E2E(), rec.OutputTokens, tg)
		s.m.observe("lsc_e2e_seconds", "End-to-end latency of completed requests.", rec.E2E().Seconds(), "model", rq.model)
	}
	if rq.hasFirst {
		s.m.observe("lsc_ttft_seconds", "Time to first byte of the response body.", rec.TTFT().Seconds(), "model", rq.model)
	}
	if s.opt.Loop != nil && label != "canceled" && label != "client_error" {
		s.mu.Lock()
		s.opt.Loop.ObserveFinish(now, rec)
		s.mu.Unlock()
	}
}

func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	var gs []gauge
	s.mu.Lock()
	reps := s.reg.All()
	for _, rep := range reps {
		gs = append(gs, gauge{"lsc_replica_inflight", "Router-local in-flight requests per replica.", labelString("replica", rep.ID, "model", rep.Model), float64(s.disp.InFlight(rep.ID))})
	}
	for _, rep := range reps {
		h := 0.0
		if s.disp.Health().Healthy(rep.ID) {
			h = 1
		}
		gs = append(gs, gauge{"lsc_replica_healthy", "1 if the replica is not ejected.", labelString("replica", rep.ID), h})
	}
	for _, rep := range reps {
		gs = append(gs, gauge{"lsc_replica_ready", "1 if the replica is in the ready state.", labelString("replica", rep.ID, "state", rep.State.String()), b2f(rep.State == registry.Ready)})
	}
	for _, m := range s.reg.Models() {
		gs = append(gs, gauge{"lsc_router_queue_length", "Requests waiting in the router queue.", labelString("model", m), float64(s.disp.QueueLen(m))})
	}
	s.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.m.write(w, gs)
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

type replicaJSON struct {
	ID       string `json:"id"`
	Model    string `json:"model"`
	Endpoint string `json:"endpoint"`
	Class    string `json:"class,omitempty"`
	State    string `json:"state,omitempty"`
	Healthy  bool   `json:"healthy"`
	InFlight int    `json:"inflight"`
}

func (s *Server) listReplicas(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	var out []replicaJSON
	for _, r := range s.reg.All() {
		out = append(out, replicaJSON{ID: r.ID, Model: r.Model, Endpoint: r.Endpoint, Class: r.Class, State: r.State.String(),
			Healthy: s.disp.Health().Healthy(r.ID), InFlight: s.disp.InFlight(r.ID)})
	}
	s.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (s *Server) addReplica(w http.ResponseWriter, r *http.Request) {
	var in replicaJSON
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&in); err != nil {
		jsonError(w, http.StatusBadRequest, "bad JSON")
		return
	}
	rep := registry.Replica{ID: in.ID, Model: in.Model, Endpoint: in.Endpoint, Class: in.Class, GPUs: 1, State: registry.Ready, Healthy: true}
	err := s.reg.Add(rep)
	if err == nil && s.opt.Store != nil {
		if serr := s.opt.Store.Save(rep); serr != nil {
			_ = s.reg.Remove(rep.ID)
			jsonError(w, http.StatusInternalServerError, "store: "+serr.Error())
			return
		}
	}
	switch {
	case errors.Is(err, registry.ErrDuplicate):
		jsonError(w, http.StatusConflict, err.Error())
		return
	case err != nil:
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.call(func(now time.Duration) []controller.Command { return s.disp.Drain(now) })
	s.opt.Logger.Info("replica registered", "id", in.ID, "model", in.Model, "endpoint", in.Endpoint)
	w.WriteHeader(http.StatusCreated)
}

func (s *Server) removeReplica(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.Lock()
	err := s.reg.Remove(id)
	if err == nil {
		s.disp.Forget(id)
	}
	s.mu.Unlock()
	if err != nil {
		jsonError(w, http.StatusNotFound, err.Error())
		return
	}
	if s.opt.Store != nil {
		_ = s.opt.Store.Delete(id) // replicas from the static configuration are not in the store
	}
	s.opt.Logger.Info("replica removed", "id", id)
	w.WriteHeader(http.StatusNoContent)
}

// Run drives the periodic loops (dispatcher tick, active probes, scrapes,
// control steps) until ctx is cancelled.
func (s *Server) Run(ctx context.Context) {
	var wg sync.WaitGroup
	every := func(d time.Duration, fn func(context.Context)) {
		if d <= 0 {
			return
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			t := time.NewTicker(d)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					fn(ctx)
				}
			}
		}()
	}
	every(s.cfg.TickInterval, func(context.Context) {
		s.call(func(now time.Duration) []controller.Command { return s.disp.Tick(now) })
	})
	every(s.cfg.ProbeInterval, s.probeAll)
	if s.opt.Scraper != nil {
		every(s.cfg.ScrapeInterval, s.scrapeAll)
	}
	if s.opt.Loop != nil && s.opt.Executor != nil {
		every(s.cfg.ControlInterval, s.controlStep)
	}
	wg.Wait()
}

func (s *Server) ready() []registry.Replica {
	var out []registry.Replica
	for _, r := range s.reg.All() {
		if r.State == registry.Ready {
			out = append(out, r)
		}
	}
	return out
}

// probeAll sends GET /health to every ready replica.
func (s *Server) probeAll(ctx context.Context) {
	for _, rep := range s.ready() {
		pctx, cancel := context.WithTimeout(ctx, s.cfg.ProbeTimeout)
		req, _ := http.NewRequestWithContext(pctx, http.MethodGet, strings.TrimRight(rep.Endpoint, "/")+"/health", nil)
		resp, err := s.opt.Client.Do(req)
		ok := err == nil && resp.StatusCode == http.StatusOK
		if resp != nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
		cancel()
		if ctx.Err() != nil {
			return
		}
		if !ok {
			s.call(func(now time.Duration) []controller.Command { return s.disp.ProbeResult(now, rep.ID, false) })
		}
	}
}

// scrapeAll reads every ready replica's metrics into a snapshot.
func (s *Server) scrapeAll(ctx context.Context) {
	for _, rep := range s.ready() {
		snap, err := s.opt.Scraper.Scrape(ctx, rep.Endpoint, rep.ID)
		if err != nil {
			continue
		}
		s.mu.Lock()
		now := s.clk.Now()
		snap.At = now
		if s.opt.Loop != nil {
			s.opt.Loop.ObserveSnapshot(now, rep.Model, snap)
		}
		s.deliver(s.disp.Snapshot(now, snap))
		s.mu.Unlock()
	}
}

func (s *Server) controlStep(ctx context.Context) {
	s.mu.Lock()
	cmds := s.opt.Loop.Step(s.clk.Now())
	s.mu.Unlock()
	for _, c := range cmds {
		if err := s.opt.Executor.Apply(ctx, c); err != nil {
			s.opt.Logger.Error("scale command failed", "kind", c.Kind.String(), "model", c.Model, "replica", c.ReplicaID, "err", err)
			continue
		}
		s.opt.Logger.Info("scale command", "kind", c.Kind.String(), "model", c.Model, "class", c.Class, "replica", c.ReplicaID, "reason", c.Reason)
	}
}

// Counter exposes a proxy counter (for tests and diagnostics).
func (s *Server) Counter(name string, kv ...string) float64 { return s.m.counter(name, kv...) }

// InFlight returns a replica's router-local in-flight count (thread-safe).
func (s *Server) InFlight(id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.disp.InFlight(id)
}
