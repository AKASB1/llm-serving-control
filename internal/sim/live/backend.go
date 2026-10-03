// Package live runs the simulator's replica engine in real time behind an
// OpenAI-compatible HTTP API, as a mock serving backend for the live data
// plane. It generates placeholder tokens; its timing comes from the same
// iteration model as the simulator, scaled by a speed-up factor. It is a mock,
// not a model server, and its metrics only borrow vLLM's names.
package live

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/clock"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
)

// Options configure a mock backend.
type Options struct {
	Model  string
	Params engine.Params
	// Speedup is virtual seconds per real second (1 = real time).
	Speedup float64
	// Clock is the real clock (clock.NewReal() by default).
	Clock clock.Clock
	// Sleep waits for a real duration (time.Sleep by default); tests may
	// replace it.
	Sleep func(time.Duration)
	// DefaultMaxTokens is the output length when a request sets no limit.
	DefaultMaxTokens int
}

type tok struct{ first, done bool }

type stream struct {
	ch       chan tok
	arrived  time.Duration // virtual
	finished bool
}

// Backend is one mock replica.
type Backend struct {
	opt     Options
	mu      sync.Mutex
	eng     *engine.Engine
	streams map[*engine.Seq]*stream
	aborts  []*engine.Seq
	wake    chan struct{}
	nextID  atomic.Uint64

	promptTotal, genTotal, success int64
	ttft, e2e                      hist
}

// New returns a backend; call Run to start its engine loop.
func New(o Options) (*Backend, error) {
	if err := o.Params.Validate(); err != nil {
		return nil, err
	}
	if o.Speedup <= 0 {
		o.Speedup = 1
	}
	if o.Clock == nil {
		o.Clock = clock.NewReal()
	}
	if o.Sleep == nil {
		o.Sleep = time.Sleep
	}
	if o.DefaultMaxTokens <= 0 {
		o.DefaultMaxTokens = 128
	}
	return &Backend{opt: o, eng: engine.New(o.Params), streams: map[*engine.Seq]*stream{}, wake: make(chan struct{}, 1),
		ttft: newHist(), e2e: newHist()}, nil
}

// vnow is the engine's virtual time.
func (b *Backend) vnow() time.Duration {
	return time.Duration(float64(b.opt.Clock.Now()) * b.opt.Speedup)
}

func (b *Backend) poke() {
	select {
	case b.wake <- struct{}{}:
	default:
	}
}

// Run drives the engine until ctx is cancelled.
func (b *Backend) Run(ctx context.Context) {
	for {
		b.mu.Lock()
		b.applyAborts()
		now := b.vnow()
		d, ok := b.eng.Start(now)
		b.mu.Unlock()
		if !ok {
			select {
			case <-b.wake:
				continue
			case <-ctx.Done():
				return
			}
		}
		b.opt.Sleep(time.Duration(float64(d) / b.opt.Speedup))
		b.mu.Lock()
		end := now + d
		for _, ev := range b.eng.Finish(end) {
			st := b.streams[ev.Seq]
			if st == nil || ev.Kind != engine.Token {
				continue
			}
			b.genTotal++
			if ev.First {
				b.ttft.add((end - st.arrived).Seconds())
			}
			if ev.Done {
				b.success++
				b.e2e.add((end - st.arrived).Seconds())
				st.finished = true
				delete(b.streams, ev.Seq)
			}
			st.ch <- tok{first: ev.First, done: ev.Done} // buffered to Output+1: never blocks
		}
		b.mu.Unlock()
		if ctx.Err() != nil {
			return
		}
	}
}

// applyAborts removes cancelled sequences between iterations (the engine
// forbids Abort during one). Caller holds b.mu.
func (b *Backend) applyAborts() {
	for _, s := range b.aborts {
		if st := b.streams[s]; st != nil && !st.finished {
			b.eng.Abort(s)
			delete(b.streams, s)
		}
	}
	b.aborts = b.aborts[:0]
}

// Handler returns the HTTP API.
func (b *Backend) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) { b.complete(w, r, true) })
	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) { b.complete(w, r, false) })
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok\n") })
	mux.HandleFunc("GET /metrics", b.metrics)
	return mux
}

// Request is the subset of the OpenAI request body the mock reads. The
// mock_* fields let tests and demos fix the token counts exactly.
type Request struct {
	Model            string                     `json:"model"`
	Stream           bool                       `json:"stream"`
	MaxTokens        int                        `json:"max_tokens"`
	Prompt           json.RawMessage            `json:"prompt"`
	Messages         []struct{ Content string } `json:"messages"`
	MockPromptTokens int                        `json:"mock_prompt_tokens"`
	MockOutputTokens int                        `json:"mock_output_tokens"`
}

// PromptTokens estimates the prompt length (4 characters per token) unless
// mock_prompt_tokens is set.
func (q Request) PromptTokens() int {
	if q.MockPromptTokens > 0 {
		return q.MockPromptTokens
	}
	n := len(q.Prompt)
	for _, m := range q.Messages {
		n += len(m.Content)
	}
	return max(1, n/4)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}})
}

func (b *Backend) complete(w http.ResponseWriter, r *http.Request, chat bool) {
	var q Request
	if err := json.NewDecoder(io.LimitReader(r.Body, 16<<20)).Decode(&q); err != nil {
		writeErr(w, http.StatusBadRequest, "bad JSON: "+err.Error())
		return
	}
	// Reading the body to EOF lets net/http watch the connection, so the
	// request context is cancelled when the client goes away.
	_, _ = io.Copy(io.Discard, io.LimitReader(r.Body, 1<<20))
	if q.Model != b.opt.Model {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q is not served here", q.Model))
		return
	}
	out := q.MockOutputTokens
	if out <= 0 {
		out = q.MaxTokens
	}
	if out <= 0 {
		out = b.opt.DefaultMaxTokens
	}
	id := fmt.Sprintf("cmpl-%d", b.nextID.Add(1))
	seq := &engine.Seq{ID: id, Prompt: q.PromptTokens(), Output: out}
	st := &stream{ch: make(chan tok, out+1)}
	b.mu.Lock()
	st.arrived = b.vnow()
	if err := b.eng.Add(seq, st.arrived); err != nil {
		b.mu.Unlock()
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	b.streams[seq] = st
	b.promptTotal += int64(seq.Prompt)
	b.mu.Unlock()
	b.poke()

	cancel := func() {
		b.mu.Lock()
		b.aborts = append(b.aborts, seq)
		b.mu.Unlock()
		b.poke()
	}
	object, field := "text_completion", "text"
	if chat {
		object, field = "chat.completion.chunk", "delta"
	}
	if q.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		if fl != nil {
			fl.Flush() // send the headers now; the first token may take a while
		}
		n := 0
		for {
			select {
			case t := <-st.ch:
				n++
				choice := map[string]any{"index": 0}
				if chat {
					choice[field] = map[string]any{"content": "tok "}
				} else {
					choice[field] = "tok "
				}
				if t.done {
					choice["finish_reason"] = "length"
				}
				chunk, _ := json.Marshal(map[string]any{"id": id, "object": object, "model": q.Model, "choices": []any{choice}})
				if _, err := fmt.Fprintf(w, "data: %s\n\n", chunk); err != nil {
					cancel()
					return
				}
				if t.done {
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
					if fl != nil {
						fl.Flush()
					}
					return
				}
				if fl != nil {
					fl.Flush()
				}
			case <-r.Context().Done():
				cancel()
				return
			}
		}
	}
	n := 0
	for n < out {
		select {
		case <-st.ch:
			n++
		case <-r.Context().Done():
			cancel()
			return
		}
	}
	text := strings.Repeat("tok ", out)
	resp := map[string]any{"id": id, "model": q.Model, "usage": map[string]int{"prompt_tokens": seq.Prompt, "completion_tokens": out, "total_tokens": seq.Prompt + out}}
	if chat {
		resp["object"] = "chat.completion"
		resp["choices"] = []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": text}, "finish_reason": "length"}}
	} else {
		resp["object"] = "text_completion"
		resp["choices"] = []any{map[string]any{"index": 0, "text": text, "finish_reason": "length"}}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// metrics writes vLLM-named metrics (times in virtual seconds).
func (b *Backend) metrics(w http.ResponseWriter, _ *http.Request) {
	b.mu.Lock()
	st := b.eng.State()
	pt, gt, sc := b.promptTotal, b.genTotal, b.success
	ttft, e2e := b.ttft.copy(), b.e2e.copy()
	b.mu.Unlock()
	lbl := fmt.Sprintf("model_name=%q", b.opt.Model)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	var sb strings.Builder
	gauge := func(name string, v float64) {
		fmt.Fprintf(&sb, "# TYPE %s gauge\n%s{%s} %g\n", name, name, lbl, v)
	}
	counter := func(name string, v int64, extra string) {
		fmt.Fprintf(&sb, "# TYPE %s counter\n%s{%s%s} %d\n", name, name, lbl, extra, v)
	}
	sb.WriteString("# mock backend (simulator engine in real time); vLLM metric names, not vLLM\n")
	gauge("vllm:num_requests_running", float64(st.Running))
	gauge("vllm:num_requests_waiting", float64(st.Waiting))
	gauge("vllm:kv_cache_usage_perc", float64(st.KVUsed)/float64(max(st.KVCapacity, 1)))
	counter("vllm:num_preemptions_total", st.Preemptions, "")
	counter("vllm:prompt_tokens_total", pt, "")
	counter("vllm:generation_tokens_total", gt, "")
	counter("vllm:request_success_total", sc, `,finished_reason="length"`)
	counter("vllm:prefix_cache_hits_total", st.PrefixHits, "")
	counter("vllm:prefix_cache_queries_total", st.PrefixQueries, "")
	ttft.write(&sb, "vllm:time_to_first_token_seconds", lbl)
	e2e.write(&sb, "vllm:e2e_request_latency_seconds", lbl)
	_, _ = io.WriteString(w, sb.String())
}

// hist is a cumulative Prometheus histogram.
type hist struct {
	bounds []float64
	counts []int64
	sum    float64
	n      int64
}

func newHist() hist {
	return hist{bounds: []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}, counts: make([]int64, 12)}
}

func (h *hist) add(x float64) {
	i := sort.SearchFloat64s(h.bounds, x)
	if i < len(h.counts) {
		h.counts[i]++
	}
	h.sum += x
	h.n++
}

func (h hist) copy() hist {
	c := h
	c.counts = append([]int64(nil), h.counts...)
	return c
}

func (h hist) write(sb *strings.Builder, name, lbl string) {
	fmt.Fprintf(sb, "# TYPE %s histogram\n", name)
	var c int64
	for i, bnd := range h.bounds {
		c += h.counts[i]
		fmt.Fprintf(sb, "%s_bucket{%s,le=\"%g\"} %d\n", name, lbl, bnd, c)
	}
	fmt.Fprintf(sb, "%s_bucket{%s,le=\"+Inf\"} %d\n%s_sum{%s} %s\n%s_count{%s} %d\n", name, lbl, h.n, name, lbl, fmtF(h.sum), name, lbl, h.n)
}

func fmtF(x float64) string {
	if math.IsNaN(x) {
		return "NaN"
	}
	return fmt.Sprintf("%g", x)
}
