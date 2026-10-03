package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/adapters"
	"github.com/AKASB1/llm-serving-control/internal/admission"
	"github.com/AKASB1/llm-serving-control/internal/clock"
	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
	"github.com/AKASB1/llm-serving-control/internal/sim/live"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/store"
)

// mockBackend starts a live mock replica that runs as fast as possible
// (fake clock, no real sleeps) and counts the requests it receives.
func mockBackend(t *testing.T) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	fake := clock.NewFake(0)
	b, err := live.New(live.Options{Model: "m", Clock: fake, Sleep: fake.Advance, Params: engine.Params{
		MaxNumSeqs: 16, MaxBatchedTokens: 1024, KVCapacityTokens: 100000, MaxModelLen: 16384, TBase: 0.01, CSeq: 0.001, CPf: 1e-5}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go b.Run(ctx)
	var n atomic.Int64
	h := b.Handler()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			n.Add(1)
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(func() { srv.Close(); cancel() })
	return srv, &n
}

type rig struct {
	srv  *Server
	http *httptest.Server
	reg  *registry.Registry
}

func newRig(t *testing.T, adm admission.Policy, cfg Config, endpoints ...string) rig {
	t.Helper()
	reg := registry.New()
	for i, ep := range endpoints {
		id := string(rune('a' + i))
		if err := reg.Add(registry.Replica{ID: id, Model: "m", Endpoint: ep, GPUs: 1, State: registry.Ready, Healthy: true}); err != nil {
			t.Fatal(err)
		}
	}
	router, _ := controller.NewRouter(controller.PolicySpec{Name: "least_outstanding"}, 1)
	disp := controller.NewDispatcher(controller.DispatchConfig{RouterQueueTimeoutS: 5, MaxRetries: 2}, reg, nil, router, adm,
		controller.NewHealth(controller.HealthConfig{ConsecutiveErrors: 3, BaseBackoffS: 60, MaxBackoffS: 60}))
	targets, _ := slo.NewTargets("interactive", map[string]slo.TargetJSON{"interactive": {TTFTs: 2, TPOTs: 0.1}})
	s, err := New(cfg, clock.NewReal(), reg, disp, targets, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()
	hs := httptest.NewServer(s.Handler())
	t.Cleanup(func() { hs.Close(); cancel(); <-done })
	return rig{srv: s, http: hs, reg: reg}
}

func quietConfig() Config {
	c := DefaultConfig()
	c.ProbeInterval, c.ScrapeInterval, c.ControlInterval = 0, 0, 0
	c.TickInterval = 20 * time.Millisecond
	return c
}

func postJSON(t *testing.T, ctx context.Context, url, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func sseData(t *testing.T, r io.Reader) []string {
	t.Helper()
	var out []string
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		if l := sc.Text(); strings.HasPrefix(l, "data: ") {
			out = append(out, strings.TrimPrefix(l, "data: "))
		}
	}
	return out
}

const streamBody = `{"model":"m","stream":true,"messages":[{"role":"user","content":"hello"}],"mock_prompt_tokens":20,"mock_output_tokens":5}`

// poll waits until cond holds or fails after timeout.
func poll(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(msg)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestStreamingPassThrough(t *testing.T) {
	b, _ := mockBackend(t)
	r := newRig(t, nil, quietConfig(), b.URL)
	resp := postJSON(t, context.Background(), r.http.URL+"/v1/chat/completions", streamBody)
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("X-Replica") != "a" || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status %d replica %q type %q", resp.StatusCode, resp.Header.Get("X-Replica"), resp.Header.Get("Content-Type"))
	}
	data := sseData(t, resp.Body)
	if len(data) != 6 || data[5] != "[DONE]" || !strings.Contains(data[0], `"content":"tok "`) {
		t.Fatalf("relayed stream: %v", data)
	}
	poll(t, 5*time.Second, func() bool { return r.srv.Counter("lsc_requests_total", "model", "m", "outcome", "completed") == 1 }, "completion not recorded")
}

func TestRetryBeforeFirstByte(t *testing.T) {
	fail := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(500) }))
	defer fail.Close()
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close() // connection refused
	good, n := mockBackend(t)
	r := newRig(t, nil, quietConfig(), fail.URL, deadURL, good.URL)
	resp := postJSON(t, context.Background(), r.http.URL+"/v1/chat/completions", streamBody)
	data := sseData(t, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("X-Replica") != "c" || len(data) != 6 || n.Load() != 1 {
		t.Fatalf("status %d replica %q chunks %d upstream requests %d", resp.StatusCode, resp.Header.Get("X-Replica"), len(data), n.Load())
	}
	if got := r.srv.Counter("lsc_retries_total", "model", "m"); got != 2 {
		t.Fatalf("retries %v, want 2 (500, then connection refused)", got)
	}
}

func TestNoRetryAfterFirstByte(t *testing.T) {
	breaks := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close() // abrupt failure after the first byte
		}
	}))
	defer breaks.Close()
	good, n := mockBackend(t)
	r := newRig(t, nil, quietConfig(), breaks.URL, good.URL)
	resp := postJSON(t, context.Background(), r.http.URL+"/v1/chat/completions", streamBody)
	data := sseData(t, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || len(data) != 1 || n.Load() != 0 {
		t.Fatalf("status %d, %d chunks, %d requests reached the second replica (want truncated stream, no retry)", resp.StatusCode, len(data), n.Load())
	}
	poll(t, 5*time.Second, func() bool { return r.srv.Counter("lsc_requests_total", "model", "m", "outcome", "failed") == 1 }, "failure not recorded")
}

func TestClientCancelReachesUpstream(t *testing.T) {
	cancelled := make(chan struct{})
	block := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer block.Close()
	r := newRig(t, nil, quietConfig(), block.URL)
	ctx, cancel := context.WithCancel(context.Background())
	resp := postJSON(t, ctx, r.http.URL+"/v1/chat/completions", streamBody)
	if _, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	cancel()
	resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream request not cancelled after the client went away")
	}
	poll(t, 5*time.Second, func() bool {
		return r.srv.Counter("lsc_requests_total", "model", "m", "outcome", "canceled") == 1 && r.srv.InFlight("a") == 0
	}, "cancellation not recorded or in-flight not released")
}

func TestAdmissionRejectsWith429(t *testing.T) {
	release := make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-release:
		case <-r.Context().Done():
		}
		_, _ = io.WriteString(w, `{"usage":{"completion_tokens":1}}`)
	}))
	defer slow.Close()
	defer close(release)
	adm := admission.NewQueueCap(admission.QueueCapParams{MaxInFlightPerReplica: 1, MaxQueue: 0})
	r := newRig(t, adm, quietConfig(), slow.URL)
	go func() {
		resp, err := http.Post(r.http.URL+"/v1/completions", "application/json", strings.NewReader(`{"model":"m","prompt":"x"}`))
		if err == nil {
			resp.Body.Close()
		}
	}()
	poll(t, 5*time.Second, func() bool { return r.srv.InFlight("a") == 1 }, "first request not dispatched")
	resp := postJSON(t, context.Background(), r.http.URL+"/v1/completions", `{"model":"m","prompt":"y"}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests || resp.Header.Get("Retry-After") == "" {
		t.Fatalf("status %d, Retry-After %q", resp.StatusCode, resp.Header.Get("Retry-After"))
	}
}

func TestActiveProbeEjectsReplica(t *testing.T) {
	sick := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(503)
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer sick.Close()
	cfg := quietConfig()
	cfg.ProbeInterval, cfg.ProbeTimeout = 20*time.Millisecond, time.Second
	r := newRig(t, nil, cfg, sick.URL)
	poll(t, 5*time.Second, func() bool {
		r.srv.mu.Lock()
		defer r.srv.mu.Unlock()
		return !r.srv.disp.Health().Healthy("a")
	}, "failing probe did not eject the replica")
}

func TestPassiveEjection(t *testing.T) {
	var errs atomic.Int64
	broken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		errs.Add(1)
		w.WriteHeader(500)
	}))
	defer broken.Close()
	good, _ := mockBackend(t)
	r := newRig(t, nil, quietConfig(), broken.URL, good.URL)
	for i := 0; i < 8; i++ {
		resp := postJSON(t, context.Background(), r.http.URL+"/v1/completions", `{"model":"m","prompt":"abcd","max_tokens":2}`)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("request %d: status %d", i, resp.StatusCode)
		}
	}
	if errs.Load() != 3 {
		t.Fatalf("broken replica saw %d requests, want 3 (ejected after 3 consecutive errors)", errs.Load())
	}
}

func TestAdminAndMetrics(t *testing.T) {
	good, _ := mockBackend(t)
	r := newRig(t, nil, quietConfig())
	add := `{"id":"x","model":"m","endpoint":"` + good.URL + `"}`
	if resp := postJSON(t, context.Background(), r.http.URL+"/admin/replicas", add); resp.StatusCode != 201 {
		t.Fatalf("add: %d", resp.StatusCode)
	}
	if resp := postJSON(t, context.Background(), r.http.URL+"/admin/replicas", add); resp.StatusCode != 409 {
		t.Fatalf("duplicate add: %d", resp.StatusCode)
	}
	resp := postJSON(t, context.Background(), r.http.URL+"/v1/completions", `{"model":"m","prompt":"abcd","max_tokens":2}`)
	resp.Body.Close()
	lr, _ := http.Get(r.http.URL + "/admin/replicas")
	var list []replicaJSON
	_ = json.NewDecoder(lr.Body).Decode(&list)
	lr.Body.Close()
	if len(list) != 1 || list[0].ID != "x" || !list[0].Healthy || list[0].State != "ready" {
		t.Fatalf("list: %+v", list)
	}
	mr, _ := http.Get(r.http.URL + "/metrics")
	text, _ := io.ReadAll(mr.Body)
	mr.Body.Close()
	samples, err := adapters.Parse(string(text))
	if err != nil || len(samples) == 0 || !strings.Contains(string(text), `lsc_requests_total{model="m",outcome="completed"} 1`) {
		t.Fatalf("metrics (%v):\n%s", err, text)
	}
	req, _ := http.NewRequest(http.MethodDelete, r.http.URL+"/admin/replicas/x", nil)
	dr, _ := http.DefaultClient.Do(req)
	if dr.StatusCode != 204 || len(r.reg.All()) != 0 {
		t.Fatalf("delete: %d", dr.StatusCode)
	}
}

func TestOracleRefused(t *testing.T) {
	reg := registry.New()
	router, _ := controller.NewRouter(controller.PolicySpec{Name: "oracle_jsq"}, 1)
	disp := controller.NewDispatcher(controller.DefaultDispatch(), reg, nil, router, nil, controller.NewHealth(controller.DefaultHealth()))
	if _, err := New(DefaultConfig(), clock.NewReal(), reg, disp, slo.Targets{}, Options{}); err == nil {
		t.Fatal("oracle router accepted by the live proxy")
	}
}

// Replicas registered through the admin API survive a restart when a file
// store is configured.
func TestAdminRegistrationsSurviveRestart(t *testing.T) {
	path := t.TempDir() + "/replicas.json"
	start := func() (*httptest.Server, *registry.Registry) {
		fs, err := store.OpenFile(path)
		if err != nil {
			t.Fatal(err)
		}
		reg := registry.New()
		for _, r := range fs.All() {
			r.State, r.Healthy = registry.Ready, true
			_ = reg.Add(r)
		}
		router, _ := controller.NewRouter(controller.PolicySpec{Name: "least_outstanding"}, 1)
		disp := controller.NewDispatcher(controller.DefaultDispatch(), reg, nil, router, nil, controller.NewHealth(controller.DefaultHealth()))
		s, err := New(quietConfig(), clock.NewReal(), reg, disp, slo.Targets{}, Options{Store: fs})
		if err != nil {
			t.Fatal(err)
		}
		hs := httptest.NewServer(s.Handler())
		t.Cleanup(hs.Close)
		return hs, reg
	}
	hs, _ := start()
	for _, id := range []string{"x", "y"} {
		resp := postJSON(t, context.Background(), hs.URL+"/admin/replicas", `{"id":"`+id+`","model":"m","endpoint":"http://127.0.0.1:1"}`)
		resp.Body.Close()
	}
	req, _ := http.NewRequest(http.MethodDelete, hs.URL+"/admin/replicas/x", nil)
	if resp, err := http.DefaultClient.Do(req); err != nil || resp.StatusCode != 204 {
		t.Fatalf("delete: %v", err)
	}
	hs.Close()
	_, reg := start() // restart over the same file
	all := reg.All()
	if len(all) != 1 || all[0].ID != "y" || all[0].Endpoint != "http://127.0.0.1:1" {
		t.Fatalf("recovered %+v", all)
	}
}
