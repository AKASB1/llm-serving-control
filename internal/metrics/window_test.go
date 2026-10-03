package metrics

import (
	"math"
	"testing"
	"time"
)

func TestWindowRatesAndExpiry(t *testing.T) {
	w := NewWindow(10*time.Second, 10)
	for i := 0; i < 100; i++ { // 10 per second for 10 s
		at := time.Duration(i) * 100 * time.Millisecond
		w.Arrival(at)
		w.Completion(at, 50, 2*time.Second, 1500*time.Millisecond)
	}
	s := w.Signals(10*time.Second - 1)
	if math.Abs(s.ArrivalRate-10) > 0.2 || s.Completions != 100 || math.Abs(s.MeanE2E-2) > 1e-9 || math.Abs(s.MeanServiceTime-1.5) > 1e-9 || math.Abs(s.OutputTokenRate-500) > 10 {
		t.Fatalf("signals %+v", s)
	}
	// 30 s later with no traffic the window is empty.
	if s := w.Signals(40 * time.Second); s.ArrivalRate != 0 || s.Completions != 0 {
		t.Fatalf("expired data counted: %+v", s)
	}
}

func TestWindowUtilizationAndServiceRate(t *testing.T) {
	w := NewWindow(10*time.Second, 5)
	for i := 1; i <= 9; i++ {
		at := time.Duration(i) * time.Second
		w.Sample(at, 8, 2)
		w.Busy(at, 500*time.Millisecond, time.Second) // each of 2 replicas half busy
		w.Busy(at, 500*time.Millisecond, time.Second)
		w.Completion(at, 10, time.Second, time.Second)
	}
	s := w.Signals(9 * time.Second)
	if math.Abs(s.Utilization-0.5) > 1e-9 || s.InFlight != 8 || s.ReadyReplicas != 2 {
		t.Fatalf("%+v", s)
	}
	if math.Abs(s.ServiceRate-s.CompletionRate/(2*0.5)) > 1e-9 {
		t.Fatalf("service rate %+v", s)
	}
}

func TestWindowMemoryIsBounded(t *testing.T) {
	w := NewWindow(time.Second, 4)
	for i := 0; i < 100000; i++ {
		w.Arrival(time.Duration(i) * time.Millisecond)
	}
	if len(w.buckets) != 4 {
		t.Fatal("bucket count changed")
	}
}
