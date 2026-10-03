package adapters

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func fixture(t *testing.T, name string) Family {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Parse(string(b))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return Index(s)
}

func TestParseFormats(t *testing.T) {
	in := strings.Join([]string{
		"# HELP a doc",
		"a 1",
		`b{x="1",y="q\"uote\\n",z="new\nline"} -2.5e3 1700000000`,
		`c{le="+Inf"} +Inf`,
		"d NaN",
		"",
		"e{} 4",
		"bad line here",
		"f 5",
	}, "\n")
	s, err := Parse(in)
	var pe *ParseError
	if !errors.As(err, &pe) || pe.Line != 8 {
		t.Fatalf("want error on line 8, got %v", err)
	}
	if len(s) != 6 {
		t.Fatalf("parsed %d samples: %+v", len(s), s)
	}
	if s[1].Labels["y"] != `q"uote\n` || s[1].Labels["z"] != "new\nline" || s[1].Value != -2500 {
		t.Fatalf("labels/escapes: %+v", s[1])
	}
	if !math.IsInf(s[2].Value, 1) || !math.IsNaN(s[3].Value) || s[5].Name != "f" {
		t.Fatalf("special values: %+v", s)
	}
}

func TestVLLMMapping(t *testing.T) {
	snap := VLLM(fixture(t, "vllm_metrics.txt"), "r1", 5*time.Second)
	if snap.Running != 12 || snap.Waiting != 3 || snap.Preemptions != 7 || snap.GeneratedTokens != 123456 {
		t.Fatalf("gauges/counters: %+v", snap)
	}
	if snap.CompletedRequests != 1000 {
		t.Fatalf("request_success must sum over finished_reason: %d", snap.CompletedRequests)
	}
	if math.Abs(snap.KVUsage()-0.4321) > 1e-4 || snap.ReplicaID != "r1" || snap.At != 5*time.Second {
		t.Fatalf("kv/identity: %+v", snap)
	}
}

func TestVLLMRenamedAndMissingMetrics(t *testing.T) {
	snap := VLLM(fixture(t, "vllm_old_metrics.txt"), "r", 0)
	if snap.Running != 5 || snap.Waiting != 0 || math.Abs(snap.KVUsage()-0.25) > 1e-9 || snap.Preemptions != 2 {
		t.Fatalf("fallbacks: %+v", snap)
	}
	empty := VLLM(Family{}, "r", 0)
	if empty.KVCapacityTokens != 0 || empty.Running != 0 {
		t.Fatalf("missing metrics must stay zero: %+v", empty)
	}
}

func TestSGLangMapping(t *testing.T) {
	f := fixture(t, "sglang_metrics.txt")
	snap := SGLang(f, "s1", 0)
	if snap.Running != 20 || snap.Waiting != 4 || snap.KVUsedTokens != 250000 || snap.KVCapacityTokens != 400000 ||
		snap.CompletedRequests != 512 || snap.GeneratedTokens != 65536 || snap.Preemptions != 3 {
		t.Fatalf("%+v", snap)
	}
	delete(f, "sglang:kv_used_tokens")
	if snap := SGLang(f, "s1", 0); math.Abs(snap.KVUsage()-0.61) > 1e-4 {
		t.Fatalf("token_usage fallback: %+v", snap)
	}
	if _, err := ByName("tgi"); err == nil {
		t.Fatal("unknown format accepted")
	}
}

func TestScraper(t *testing.T) {
	body, _ := os.ReadFile("testdata/vllm_metrics.txt")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	sc := &Scraper{Client: srv.Client(), Mapping: VLLM, Now: func() time.Duration { return 7 * time.Second }}
	snap, err := sc.Scrape(context.Background(), srv.URL, "r1")
	if err != nil || snap.Running != 12 || snap.At != 7*time.Second {
		t.Fatalf("%+v %v", snap, err)
	}
	if _, err := sc.Scrape(context.Background(), srv.URL+"/nope", "r1"); err == nil {
		t.Fatal("404 accepted")
	}
}

// format renders samples back to text for the round-trip property.
func format(s []Sample) string {
	var b strings.Builder
	for _, x := range s {
		b.WriteString(x.Name)
		if len(x.Labels) > 0 {
			b.WriteByte('{')
			first := true
			for k, v := range x.Labels {
				if !first {
					b.WriteByte(',')
				}
				first = false
				v = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(v)
				fmt.Fprintf(&b, "%s=\"%s\"", k, v)
			}
			b.WriteByte('}')
		}
		b.WriteByte(' ')
		b.WriteString(strconv.FormatFloat(x.Value, 'g', -1, 64))
		b.WriteByte('\n')
	}
	return b.String()
}

// FuzzParse: the parser never panics, and whatever it accepts survives a
// format/parse round trip unchanged.
func FuzzParse(f *testing.F) {
	for _, name := range []string{"vllm_metrics.txt", "sglang_metrics.txt", "vllm_old_metrics.txt"} {
		b, _ := os.ReadFile("testdata/" + name)
		f.Add(string(b))
	}
	f.Fuzz(func(t *testing.T, in string) {
		s, _ := Parse(in)
		again, err := Parse(format(s))
		if err != nil {
			t.Fatalf("re-parse of formatted output failed: %v", err)
		}
		if len(again) != len(s) {
			t.Fatalf("round trip changed the sample count: %d → %d", len(s), len(again))
		}
		for i := range s {
			a, b := s[i], again[i]
			if a.Name != b.Name || len(a.Labels) != len(b.Labels) || !(a.Value == b.Value || (math.IsNaN(a.Value) && math.IsNaN(b.Value))) {
				t.Fatalf("round trip changed sample %d: %+v → %+v", i, a, b)
			}
			for k, v := range a.Labels {
				if b.Labels[k] != v {
					t.Fatalf("label %q changed: %q → %q", k, v, b.Labels[k])
				}
			}
		}
	})
}
