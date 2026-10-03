package adapters

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/metrics"
)

// kvScale expresses a fractional KV usage as tokens when a backend reports
// only the fraction: KVUsedTokens/KVCapacityTokens = usage, on a 10^4 scale.
const kvScale = 10000

// Mapping turns one backend's metric families into a snapshot.
type Mapping func(f Family, replicaID string, at time.Duration) metrics.ReplicaSnapshot

// VLLM maps vLLM metric names (checked against vllm/v1/metrics/loggers.py at
// commit 37d6174, 2026-10-01). Counters are exposed with the _total suffix.
// Older releases' names are fallbacks (gpu_cache_usage_perc for KV usage,
// counters without _total). Fields vLLM does not expose (busy time,
// un-prefilled prompt tokens) stay zero.
func VLLM(f Family, replicaID string, at time.Duration) metrics.ReplicaSnapshot {
	s := metrics.ReplicaSnapshot{ReplicaID: replicaID, At: at}
	if v, ok := f.Sum("vllm:num_requests_running"); ok {
		s.Running = int(v)
	}
	if v, ok := f.Sum("vllm:num_requests_waiting"); ok {
		s.Waiting = int(v)
	}
	if v, ok := f.Mean("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"); ok {
		s.KVCapacityTokens = kvScale
		s.KVUsedTokens = int(math.Round(clamp01(v) * kvScale))
	}
	if v, ok := f.Sum("vllm:num_preemptions_total", "vllm:num_preemptions"); ok {
		s.Preemptions = int64(v)
	}
	if v, ok := f.Sum("vllm:request_success_total", "vllm:request_success"); ok {
		s.CompletedRequests = int64(v)
	}
	if v, ok := f.Sum("vllm:generation_tokens_total", "vllm:generation_tokens"); ok {
		s.GeneratedTokens = int64(v)
	}
	if v, ok := f.Sum("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"); ok {
		s.PrefixHits = int64(v)
	}
	if v, ok := f.Sum("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"); ok {
		s.PrefixQueries = int64(v)
	}
	return s
}

// SGLang maps SGLang metric names (checked against
// python/sglang/srt/observability/metrics_collector.py at commit 3c4653b,
// 2026-09-29). KV usage prefers absolute token counts (kv_used_tokens over
// max_total_num_tokens) and falls back to the token_usage fraction;
// retracted requests count as preemptions.
func SGLang(f Family, replicaID string, at time.Duration) metrics.ReplicaSnapshot {
	s := metrics.ReplicaSnapshot{ReplicaID: replicaID, At: at}
	if v, ok := f.Sum("sglang:num_running_reqs"); ok {
		s.Running = int(v)
	}
	if v, ok := f.Sum("sglang:num_queue_reqs"); ok {
		s.Waiting = int(v)
	}
	used, okU := f.Sum("sglang:kv_used_tokens")
	capTok, okC := f.Sum("sglang:max_total_num_tokens")
	switch {
	case okU && okC && capTok > 0:
		s.KVUsedTokens, s.KVCapacityTokens = int(used), int(capTok)
	default:
		if v, ok := f.Mean("sglang:token_usage", "sglang:full_token_usage"); ok {
			s.KVCapacityTokens = kvScale
			s.KVUsedTokens = int(math.Round(clamp01(v) * kvScale))
		}
	}
	if v, ok := f.Sum("sglang:num_retracted_requests_total", "sglang:num_retracted_reqs"); ok {
		s.Preemptions = int64(v)
	}
	if v, ok := f.Sum("sglang:num_requests_total"); ok {
		s.CompletedRequests = int64(v)
	}
	if v, ok := f.Sum("sglang:generation_tokens_total"); ok {
		s.GeneratedTokens = int64(v)
	}
	return s
}

func clamp01(v float64) float64 { return math.Max(0, math.Min(1, v)) }

// ByName returns a mapping by backend name ("vllm" or "sglang").
func ByName(name string) (Mapping, error) {
	switch strings.ToLower(name) {
	case "vllm", "":
		return VLLM, nil
	case "sglang":
		return SGLang, nil
	}
	return nil, fmt.Errorf("unknown backend metrics format %q", name)
}

// Scraper fetches <endpoint>/metrics and maps it. It implements
// backends.Scraper.
type Scraper struct {
	Client  *http.Client
	Mapping Mapping
	// Now returns the control plane's clock reading for the snapshot.
	Now func() time.Duration
}

// Scrape implements backends.Scraper. Malformed lines are skipped (tolerant);
// a response with no parsable sample is an error.
func (s *Scraper) Scrape(ctx context.Context, endpoint, replicaID string) (metrics.ReplicaSnapshot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(endpoint, "/")+"/metrics", nil)
	if err != nil {
		return metrics.ReplicaSnapshot{}, err
	}
	resp, err := s.Client.Do(req)
	if err != nil {
		return metrics.ReplicaSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return metrics.ReplicaSnapshot{}, fmt.Errorf("scrape %s: status %d", replicaID, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return metrics.ReplicaSnapshot{}, err
	}
	samples, perr := Parse(string(body))
	if len(samples) == 0 {
		if perr == nil {
			perr = fmt.Errorf("no samples")
		}
		return metrics.ReplicaSnapshot{}, fmt.Errorf("scrape %s: %w", replicaID, perr)
	}
	return s.Mapping(Index(samples), replicaID, s.Now()), nil
}
