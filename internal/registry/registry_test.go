package registry

import (
	"errors"
	"testing"
)

func TestAddValidatesAndRejectsDuplicates(t *testing.T) {
	r := New()
	if err := r.Add(Replica{ID: "a", Model: "m"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing endpoint: %v", err)
	}
	if err := r.Add(Replica{ID: "a", Model: "m", Endpoint: "x"}); err != nil {
		t.Fatal(err)
	}
	if err := r.Add(Replica{ID: "a", Model: "m", Endpoint: "y"}); !errors.Is(err, ErrDuplicate) {
		t.Fatalf("duplicate: %v", err)
	}
}

func TestForModelIsSortedAndFiltered(t *testing.T) {
	r := New()
	for _, id := range []string{"c", "a", "d", "b"} {
		model := "m1"
		if id == "d" {
			model = "m2"
		}
		if err := r.Add(Replica{ID: id, Model: model, Endpoint: "e-" + id}); err != nil {
			t.Fatal(err)
		}
	}
	got := r.ForModel("m1")
	if len(got) != 3 || got[0].ID != "a" || got[1].ID != "b" || got[2].ID != "c" {
		t.Fatalf("ForModel = %+v", got)
	}
	if ms := r.Models(); len(ms) != 2 || ms[0] != "m1" || ms[1] != "m2" {
		t.Fatalf("Models = %v", ms)
	}
}

func TestUpdateRemoveAndEligibility(t *testing.T) {
	r := New()
	_ = r.Add(Replica{ID: "a", Model: "m", Endpoint: "e", State: Loading, Healthy: true})
	rep, _ := r.Get("a")
	if rep.Eligible() {
		t.Fatal("loading replica must not be eligible")
	}
	if err := r.Update("a", func(x *Replica) { x.State = Ready; x.Outstanding = 2 }); err != nil {
		t.Fatal(err)
	}
	rep, _ = r.Get("a")
	if !rep.Eligible() || rep.Outstanding != 2 {
		t.Fatalf("after update: %+v", rep)
	}
	if err := r.Update("zz", func(*Replica) {}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("update missing: %v", err)
	}
	if err := r.Remove("a"); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Get("a"); ok {
		t.Fatal("removed replica still present")
	}
}

func TestStateStrings(t *testing.T) {
	if Ready.String() != "ready" || Failed.String() != "failed" || State(99).String() != "state(99)" {
		t.Fatal("state names")
	}
	if !Draining.Live() || Terminated.Live() || Failed.Live() {
		t.Fatal("Live()")
	}
}
