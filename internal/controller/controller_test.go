package controller

import (
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/admission"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
)

const s = time.Second

func TestHealthEjectBackoffReinstate(t *testing.T) {
	h := NewHealth(HealthConfig{ConsecutiveErrors: 3, BaseBackoffS: 5, MaxBackoffS: 12})
	if h.Failure("a", 1*s) || h.Failure("a", 2*s) {
		t.Fatal("ejected before the threshold")
	}
	h.Success("a", 3*s) // resets the streak
	h.Failure("a", 4*s)
	h.Failure("a", 5*s)
	if !h.Failure("a", 6*s) || h.Healthy("a") {
		t.Fatal("third consecutive failure must eject")
	}
	if until, _ := h.EjectedUntil("a"); until != 11*s {
		t.Fatalf("first backoff ends at %v, want 11s", until)
	}
	if got := h.Tick(10 * s); len(got) != 0 {
		t.Fatal("reinstated early")
	}
	if got := h.Tick(11 * s); len(got) != 1 || !h.Healthy("a") {
		t.Fatal("not reinstated after backoff")
	}
	// On probation one failure re-ejects with a doubled backoff (10 s).
	if !h.Failure("a", 12*s) {
		t.Fatal("probation failure must eject")
	}
	if until, _ := h.EjectedUntil("a"); until != 22*s {
		t.Fatalf("second backoff ends at %v, want 22s", until)
	}
	h.Tick(22 * s)
	h.Failure("a", 23*s) // third ejection: 20 s capped at 12 s
	if until, _ := h.EjectedUntil("a"); until != 35*s {
		t.Fatalf("capped backoff ends at %v, want 35s", until)
	}
	h.Tick(35 * s)
	h.Success("a", 36*s) // success on probation clears history
	h.Failure("a", 37*s)
	h.Failure("a", 38*s)
	h.Failure("a", 39*s)
	if until, _ := h.EjectedUntil("a"); until != 44*s {
		t.Fatalf("history not cleared: ejected until %v", until)
	}
	if h.Ejections != 4 {
		t.Fatalf("ejections %d", h.Ejections)
	}
}

func TestProbeFailureEjects(t *testing.T) {
	h := NewHealth(DefaultHealth())
	if !h.ProbeFailure("x", 0) || h.Healthy("x") || h.ProbeFailure("x", s) {
		t.Fatal("probe failure semantics")
	}
}

func newTestDispatcher(t *testing.T, adm admission.Policy, ids ...string) (*Dispatcher, *registry.Registry) {
	t.Helper()
	reg := registry.New()
	for _, id := range ids {
		if err := reg.Add(registry.Replica{ID: id, Model: "m", Endpoint: "e", State: registry.Ready, Healthy: true}); err != nil {
			t.Fatal(err)
		}
	}
	cfg := DispatchConfig{RouterQueueTimeoutS: 30, MaxRetries: 2}
	return NewDispatcher(cfg, reg, nil, routing.NewLeastOutstanding(), adm, NewHealth(DefaultHealth())), reg
}

func req(id string) routing.Request { return routing.Request{ID: id, Model: "m", PromptTokens: 10} }

func TestDispatchQueueAndDrainOnReady(t *testing.T) {
	d, reg := newTestDispatcher(t, nil)
	_ = reg.Add(registry.Replica{ID: "a", Model: "m", Endpoint: "e", State: registry.Loading, Healthy: true})
	if cmds := d.Arrive(0, req("r1")); len(cmds) != 0 || d.QueueLen("m") != 1 {
		t.Fatalf("must queue without eligible replicas: %+v", cmds)
	}
	_ = reg.Update("a", func(r *registry.Replica) { r.State = registry.Ready })
	cmds := d.Drain(3 * s)
	if len(cmds) != 1 || cmds[0].Kind != Send || cmds[0].ReplicaID != "a" || cmds[0].RouterQueue != 3*s {
		t.Fatalf("drain: %+v", cmds)
	}
}

func TestRouterQueueTimeout(t *testing.T) {
	d, _ := newTestDispatcher(t, nil)
	d.Arrive(0, req("r1"))
	if cmds := d.Tick(29 * s); len(cmds) != 0 {
		t.Fatal("timed out early")
	}
	cmds := d.Tick(30 * s)
	if len(cmds) != 1 || cmds[0].Kind != TimeOut || cmds[0].RouterQueue != 30*s || d.Pending() != 0 {
		t.Fatalf("timeout: %+v", cmds)
	}
}

func TestRetryBeforeFirstTokenOnly(t *testing.T) {
	d, _ := newTestDispatcher(t, nil, "a", "b")
	c := d.Arrive(0, req("r1"))[0]
	if c.ReplicaID != "a" {
		t.Fatalf("first pick %+v", c)
	}
	// Error before the first token: retried on the other replica.
	c2 := d.Error(s, "r1", true, "reset")
	if len(c2) != 1 || c2[0].Kind != Send || c2[0].ReplicaID != "b" || c2[0].Attempt != 2 {
		t.Fatalf("retry: %+v", c2)
	}
	// Error after the first token: fails.
	d.FirstToken(2*s, "r1")
	c3 := d.Error(3*s, "r1", true, "reset")
	if len(c3) != 1 || c3[0].Kind != Fail {
		t.Fatalf("after first token: %+v", c3)
	}
	if d.InFlight("a") != 0 || d.InFlight("b") != 0 {
		t.Fatal("in-flight not released")
	}
}

func TestRetriesAreBounded(t *testing.T) {
	d, _ := newTestDispatcher(t, nil, "a", "b", "c", "d")
	d.Arrive(0, req("r1"))
	var last []Command
	for i := 0; i < 3; i++ {
		last = d.Error(time.Duration(i)*s, "r1", true, "boom")
	}
	if len(last) != 1 || last[0].Kind != Fail || last[0].Attempt != 3 {
		t.Fatalf("after 2 retries the third error must fail: %+v", last)
	}
}

func TestNonRetryableFailsImmediately(t *testing.T) {
	d, _ := newTestDispatcher(t, nil, "a", "b")
	d.Arrive(0, req("r1"))
	if c := d.Error(s, "r1", false, "too long"); len(c) != 1 || c[0].Kind != Fail {
		t.Fatalf("%+v", c)
	}
}

func TestPassiveEjectionRemovesReplicaFromView(t *testing.T) {
	d, reg := newTestDispatcher(t, nil, "a", "b")
	// least_outstanding with ties to the lowest ID: x→a, y→b, z→a.
	for _, id := range []string{"x", "y", "z"} {
		d.Arrive(0, req(id))
	}
	d.Error(s, "x", false, "boom")
	d.Error(s, "z", false, "boom")
	if !d.health.Healthy("a") {
		t.Fatal("ejected after two errors")
	}
	if c := d.Arrive(s, req("w")); c[0].ReplicaID != "a" {
		t.Fatalf("w should go to a (0 vs 1 in flight): %+v", c)
	}
	d.Error(s, "w", false, "boom") // third consecutive error on a
	if ra, _ := reg.Get("a"); ra.Healthy || d.health.Healthy("a") {
		t.Fatal("a should be ejected and marked unhealthy in the registry")
	}
	for i := 0; i < 5; i++ {
		c := d.Arrive(2*s, req("n"+string(rune('0'+i))))
		if c[0].ReplicaID != "b" {
			t.Fatalf("ejected replica received traffic: %+v", c)
		}
	}
	d.Tick(6 * s) // ejected at 1 s with a 5 s backoff → reinstated on probation
	if ra, _ := reg.Get("a"); !d.health.Healthy("a") || !ra.Healthy {
		t.Fatal("not reinstated")
	}
}

type rejectAll struct{}

func (rejectAll) Name() string { return "reject" }
func (rejectAll) Admit(routing.Request, routing.View) admission.Decision {
	return admission.Decision{Action: routing.Reject, Reason: "full"}
}

func TestAdmissionReject(t *testing.T) {
	d, _ := newTestDispatcher(t, rejectAll{}, "a")
	c := d.Arrive(0, req("r1"))
	if len(c) != 1 || c[0].Kind != Reject || d.Pending() != 0 {
		t.Fatalf("%+v", c)
	}
}

func TestFactoryBuildsEveryRouterAndRejectsUnknownParams(t *testing.T) {
	for _, n := range RouterNames() {
		r, err := NewRouter(PolicySpec{Name: n}, 1)
		if err != nil || r.Name() != n {
			t.Fatalf("%s: %v", n, err)
		}
	}
	if _, err := NewRouter(PolicySpec{Name: "latency_aware", Params: []byte(`{"tau":1}`)}, 1); err == nil {
		t.Fatal("unknown parameter accepted")
	}
	if _, err := NewRouter(PolicySpec{Name: "nope"}, 1); err == nil {
		t.Fatal("unknown policy accepted")
	}
}
