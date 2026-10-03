package proxy

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
)

// promMetrics is a minimal Prometheus text-format exposition (standard library
// only): labelled counters and histograms with sorted, stable output.
type promMetrics struct {
	mu       sync.Mutex
	counters map[string]map[string]float64 // name → label string → value
	hists    map[string]map[string]*histo
	help     map[string]string
}

type histo struct {
	counts []uint64
	sum    float64
	n      uint64
}

var latencyBuckets = []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}

func newMetrics() *promMetrics {
	return &promMetrics{counters: map[string]map[string]float64{}, hists: map[string]map[string]*histo{}, help: map[string]string{}}
}

func labelString(kv ...string) string {
	var parts []string
	for i := 0; i+1 < len(kv); i += 2 {
		v := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(kv[i+1])
		parts = append(parts, fmt.Sprintf("%s=\"%s\"", kv[i], v))
	}
	return strings.Join(parts, ",")
}

func (m *promMetrics) inc(name, help string, kv ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.help[name] = help
	if m.counters[name] == nil {
		m.counters[name] = map[string]float64{}
	}
	m.counters[name][labelString(kv...)]++
}

func (m *promMetrics) observe(name, help string, v float64, kv ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.help[name] = help
	if m.hists[name] == nil {
		m.hists[name] = map[string]*histo{}
	}
	l := labelString(kv...)
	h := m.hists[name][l]
	if h == nil {
		h = &histo{counts: make([]uint64, len(latencyBuckets))}
		m.hists[name][l] = h
	}
	for i, b := range latencyBuckets {
		if v <= b {
			h.counts[i]++
			break
		}
	}
	h.sum += v
	h.n++
}

// counter returns a counter's value (for tests).
func (m *promMetrics) counter(name string, kv ...string) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.counters[name][labelString(kv...)]
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func braces(l string) string {
	if l == "" {
		return ""
	}
	return "{" + l + "}"
}

func fnum(v float64) string {
	switch {
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsNaN(v):
		return "NaN"
	}
	return fmt.Sprintf("%g", v)
}

// gauge is a sampled value written alongside the stored metrics.
type gauge struct {
	name, help, labels string
	value              float64
}

func (m *promMetrics) write(w io.Writer, gauges []gauge) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b strings.Builder
	for _, name := range sortedKeys(m.counters) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, m.help[name], name)
		for _, l := range sortedKeys(m.counters[name]) {
			fmt.Fprintf(&b, "%s%s %s\n", name, braces(l), fnum(m.counters[name][l]))
		}
	}
	for _, name := range sortedKeys(m.hists) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s histogram\n", name, m.help[name], name)
		for _, l := range sortedKeys(m.hists[name]) {
			h := m.hists[name][l]
			sep := ""
			if l != "" {
				sep = ","
			}
			var c uint64
			for i, bound := range latencyBuckets {
				c += h.counts[i]
				fmt.Fprintf(&b, "%s_bucket{%s%sle=\"%s\"} %d\n", name, l, sep, fnum(bound), c)
			}
			fmt.Fprintf(&b, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, l, sep, h.n)
			fmt.Fprintf(&b, "%s_sum%s %s\n%s_count%s %d\n", name, braces(l), fnum(h.sum), name, braces(l), h.n)
		}
	}
	seen := map[string]bool{}
	for _, g := range gauges {
		if !seen[g.name] {
			fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n", g.name, g.help, g.name)
			seen[g.name] = true
		}
		fmt.Fprintf(&b, "%s%s %s\n", g.name, braces(g.labels), fnum(g.value))
	}
	_, _ = io.WriteString(w, b.String())
}
