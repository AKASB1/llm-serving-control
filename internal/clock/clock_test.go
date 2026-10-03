package clock

import (
	"testing"
	"time"
)

func TestFakeAdvanceAndSet(t *testing.T) {
	f := NewFake(time.Second)
	f.Advance(500 * time.Millisecond)
	if got := f.Now(); got != 1500*time.Millisecond {
		t.Fatalf("Now = %v", got)
	}
	f.Set(2 * time.Second)
	if got := f.Now(); got != 2*time.Second {
		t.Fatalf("Now = %v", got)
	}
}

func TestFakeRejectsBackwards(t *testing.T) {
	f := NewFake(time.Second)
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when moving backwards")
		}
	}()
	f.Set(0)
}

func TestRealIsMonotonic(t *testing.T) {
	r := NewReal()
	a := r.Now()
	b := r.Now()
	if b < a || a < 0 {
		t.Fatalf("not monotonic: %v then %v", a, b)
	}
}

func TestSeconds(t *testing.T) {
	if got := Seconds(1.25); got != 1250*time.Millisecond {
		t.Fatalf("Seconds(1.25) = %v", got)
	}
	if got := Seconds(0.000000001); got != time.Nanosecond {
		t.Fatalf("Seconds(1e-9) = %v", got)
	}
}
