package live

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/adapters"
	"github.com/AKASB1/llm-serving-control/internal/clock"
	"github.com/AKASB1/llm-serving-control/internal/sim/engine"
)

func params() engine.Params {
	return engine.Params{MaxNumSeqs: 8, MaxBatchedTokens: 512, KVCapacityTokens: 100000, MaxModelLen: 16384,
		TBase: 0.01, CSeq: 0.001, CCtx: 1e-6, CPf: 1e-5}
}

// start runs a backend whose sleep advances a fake clock; gate (if non-nil)
// makes every iteration wait for a value, so a test controls progress.
func start(t *testing.T, gate chan struct{}) (*Backend, *httptest.Server) {
	t.Helper()
	fake := clock.NewFake(0)
	b, err := New(Options{Model: "m", Params: params(), Clock: fake, Sleep: func(d time.Duration) {
		if gate != nil {
			<-gate
		}
		fake.Advance(d)
	}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { b.Run(ctx); close(done) }()
	srv := httptest.NewServer(b.Handler())
	t.Cleanup(func() {
		srv.Close()
		cancel()
		if gate != nil {
			close(gate)
		}
		<-done
	})
	return b, srv
}

func post(t *testing.T, ctx context.Context, url, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestStreamingChatCompletion(t *testing.T) {
	_, srv := start(t, nil)
	resp := post(t, context.Background(), srv.URL+"/v1/chat/completions",
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],"mock_prompt_tokens":40,"mock_output_tokens":5}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status %d, type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	var data []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			data = append(data, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(data) != 6 || data[5] != "[DONE]" {
		t.Fatalf("want 5 chunks and [DONE], got %d: %v", len(data), data)
	}
	var last struct {
		Choices []struct {
			Delta        struct{ Content string }
			FinishReason string `json:"finish_reason"`
		}
	}
	if err := json.Unmarshal([]byte(data[4]), &last); err != nil || last.Choices[0].FinishReason != "length" || last.Choices[0].Delta.Content != "tok " {
		t.Fatalf("last chunk %s (%v)", data[4], err)
	}
}

func TestNonStreamingCompletionAndErrors(t *testing.T) {
	b, srv := start(t, nil)
	resp := post(t, context.Background(), srv.URL+"/v1/completions", `{"model":"m","prompt":"abcdefgh","max_tokens":3}`)
	var out struct {
		Usage   map[string]int
		Choices []struct{ Text string }
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if out.Usage["completion_tokens"] != 3 || out.Usage["prompt_tokens"] != 2 || out.Choices[0].Text != "tok tok tok " {
		t.Fatalf("%+v", out)
	}
	if r := post(t, context.Background(), srv.URL+"/v1/completions", `{"model":"other","prompt":"x"}`); r.StatusCode != 404 {
		t.Fatalf("wrong model: %d", r.StatusCode)
	}
	if r := post(t, context.Background(), srv.URL+"/v1/completions", `{"model":"m","mock_prompt_tokens":20000,"mock_output_tokens":1}`); r.StatusCode != 400 {
		t.Fatalf("too long: %d", r.StatusCode)
	}
	// Metrics parse with the vLLM adapter and count the generated tokens.
	mr, _ := http.Get(srv.URL + "/metrics")
	body := new(strings.Builder)
	_, _ = bufio.NewReader(mr.Body).WriteTo(body)
	mr.Body.Close()
	samples, err := adapters.Parse(body.String())
	if err != nil {
		t.Fatalf("metrics do not parse: %v", err)
	}
	snap := adapters.VLLM(adapters.Index(samples), "r", 0)
	if snap.GeneratedTokens != 3 || snap.CompletedRequests != 1 || snap.Running != 0 {
		t.Fatalf("snapshot %+v", snap)
	}
	_ = b
}

// A client that disconnects mid-stream makes the backend abort the sequence.
func TestClientDisconnectAbortsSequence(t *testing.T) {
	gate := make(chan struct{})
	b, srv := start(t, gate)
	ctx, cancel := context.WithCancel(context.Background())
	resp := post(t, ctx, srv.URL+"/v1/chat/completions", `{"model":"m","stream":true,"mock_prompt_tokens":10,"mock_output_tokens":1000}`)
	br := bufio.NewReader(resp.Body)
	gate <- struct{}{} // prefill iteration → first token
	if line, err := br.ReadString('\n'); err != nil || !strings.HasPrefix(line, "data: ") {
		t.Fatalf("first chunk: %q %v", line, err)
	}
	cancel()
	resp.Body.Close()
	deadline := time.After(5 * time.Second)
	for {
		b.mu.Lock()
		running := b.eng.State().Running + b.eng.State().Waiting
		b.mu.Unlock()
		if running == 0 {
			return
		}
		select {
		case gate <- struct{}{}: // let iterations proceed until the abort is applied
		case <-time.After(10 * time.Millisecond): // engine idle: re-check
		case <-deadline:
			t.Fatal("sequence not aborted after the client went away")
		}
	}
}
