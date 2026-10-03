// Package engine models one serving replica: iteration-level continuous
// batching with a FCFS waiting queue, a per-iteration token budget, a KV-cache
// capacity in tokens, and preemption by recompute. It knows nothing about the
// event loop: a caller asks Start for the next iteration's duration and calls
// Finish when that much time has passed (virtual time in the simulator, real
// time in the mock backend).
package engine

import (
	"container/list"
	"errors"
	"fmt"
	"math"
	"time"
)

// Params are the engine parameters of one replica class.
type Params struct {
	MaxNumSeqs       int `json:"max_num_seqs"`
	MaxBatchedTokens int `json:"max_batched_tokens"`
	KVCapacityTokens int `json:"kv_capacity_tokens"`
	// MaxModelLen bounds prompt+output; longer requests are refused.
	MaxModelLen int `json:"max_model_len"`
	// PrefixCacheGroups is the capacity of the prefix cache in prefix groups
	// (LRU); 0 disables it.
	PrefixCacheGroups int `json:"prefix_cache_groups,omitempty"`
	// Iteration time = TBase + CSeq·n_decode + CCtx·context_tokens + CPf·prefill_tokens (seconds).
	TBase float64 `json:"t_base_s"`
	CSeq  float64 `json:"c_seq_s"`
	CCtx  float64 `json:"c_ctx_s"`
	CPf   float64 `json:"c_pf_s"`
}

// Validate checks the parameters.
func (p Params) Validate() error {
	switch {
	case p.MaxNumSeqs < 1:
		return errors.New("engine: max_num_seqs >= 1 required")
	case p.MaxBatchedTokens < p.MaxNumSeqs:
		return errors.New("engine: max_batched_tokens must be >= max_num_seqs")
	case p.MaxModelLen < 2 || p.MaxModelLen > p.KVCapacityTokens:
		return errors.New("engine: need 2 <= max_model_len <= kv_capacity_tokens")
	case p.TBase <= 0 || p.CSeq < 0 || p.CCtx < 0 || p.CPf < 0:
		return errors.New("engine: t_base > 0 and non-negative coefficients required")
	}
	return nil
}

// IterationTime returns the modelled duration of one iteration.
func (p Params) IterationTime(nDecode, contextTokens, prefillTokens int) time.Duration {
	s := p.TBase + p.CSeq*float64(nDecode) + p.CCtx*float64(contextTokens) + p.CPf*float64(prefillTokens)
	return time.Duration(math.Round(s * 1e9))
}

// Seq is one request inside the engine.
type Seq struct {
	ID     string
	Prompt int
	Output int
	// PrefixGroup and PrefixTokens describe a prompt prefix shared with other
	// requests of the same group; a prefix-cache hit skips its prefill.
	PrefixGroup  string
	PrefixTokens int
	// Tag is an opaque payload for the caller.
	Tag any

	prefillTarget int // tokens to prefill: Prompt, or Prompt+generated after a preemption
	prefilled     int
	generated     int
	kv            int // KV tokens held
	admitOrder    uint64
	running       bool
	waiting       bool

	// EnqueuedAt is when the request reached this engine; FirstScheduledAt is
	// the start of the first iteration that included it (-1 before).
	EnqueuedAt       time.Duration
	FirstScheduledAt time.Duration
	Preemptions      int
}

// Generated returns the number of output tokens emitted so far.
func (s *Seq) Generated() int { return s.generated }

// KV returns the KV tokens the sequence holds.
func (s *Seq) KV() int { return s.kv }

// EventKind classifies what happened to a sequence in an iteration.
type EventKind int

// Event kinds.
const (
	// Token: one output token emitted (First marks the first one; Done the last).
	Token EventKind = iota
	// Preempted: the sequence lost its KV and went back to the waiting queue.
	Preempted
)

// Event is one outcome of Finish (or a preemption during Start).
type Event struct {
	Seq   *Seq
	Kind  EventKind
	First bool
	Done  bool
}

type chunk struct {
	seq    *Seq
	tokens int
	decode bool
}

// Engine is one replica's scheduler state. It is not safe for concurrent use.
type Engine struct {
	p        Params
	waiting  []*Seq // FCFS; preempted sequences go back to the head
	running  []*Seq // admission order
	kvUsed   int
	reserved int // KV reserved for the not-yet-prefilled part of admitted prompts
	nextAdm  uint64

	inIter    bool
	iterStart time.Duration
	plan      []chunk
	preempted []Event

	// Counters.
	preemptions int64
	completed   int64
	generatedN  int64
	busy        time.Duration
	iterations  int64
	// KV occupancy integral (token·seconds), for the time-averaged occupancy.
	kvIntegral float64

	// Prefix cache: LRU of prefix groups (front = most recent).
	lru    *list.List
	lruIdx map[string]*list.Element
	pHits  int64
	pQuery int64
}

// New returns an engine. It panics on invalid parameters (validate first).
func New(p Params) *Engine {
	if err := p.Validate(); err != nil {
		panic(err)
	}
	return &Engine{p: p, lru: list.New(), lruIdx: map[string]*list.Element{}}
}

// Params returns the engine's parameters.
func (e *Engine) Params() Params { return e.p }

// ErrTooLong is returned by Add for requests longer than MaxModelLen.
var ErrTooLong = errors.New("engine: prompt+output exceeds max_model_len")

// Add enqueues a sequence at the tail of the waiting queue.
func (e *Engine) Add(s *Seq, now time.Duration) error {
	if s.Prompt < 1 || s.Output < 1 {
		return fmt.Errorf("engine: invalid lengths %d/%d", s.Prompt, s.Output)
	}
	if s.Prompt+s.Output > e.p.MaxModelLen {
		return ErrTooLong
	}
	s.prefillTarget = s.Prompt
	s.prefilled, s.generated, s.kv = 0, 0, 0
	s.EnqueuedAt = now
	s.FirstScheduledAt = -1
	s.waiting = true
	e.waiting = append(e.waiting, s)
	return nil
}

// Busy reports whether an iteration is in progress.
func (e *Engine) Busy() bool { return e.inIter }

// HasWork reports whether any sequence is waiting or running.
func (e *Engine) HasWork() bool { return len(e.waiting) > 0 || len(e.running) > 0 }

// Start plans the next iteration at time now and returns its duration. It
// returns ok == false when there is nothing to schedule. Preemptions decided
// while planning are returned by the following Finish.
func (e *Engine) Start(now time.Duration) (dur time.Duration, ok bool) {
	if e.inIter {
		panic("engine: Start during an iteration")
	}
	e.plan = e.plan[:0]
	budget := e.p.MaxBatchedTokens

	// Phase A: every decoding sequence decodes one token. If the KV growth
	// does not fit, preempt the most recently admitted sequence (recompute).
	for {
		nDecode := 0
		for _, s := range e.running {
			if s.prefilled == s.prefillTarget {
				nDecode++
			}
		}
		if e.kvUsed+e.reserved+nDecode <= e.p.KVCapacityTokens || len(e.running) == 0 {
			break
		}
		e.preemptLast()
	}
	nDecode, ctx := 0, 0
	for _, s := range e.running {
		if s.prefilled == s.prefillTarget {
			e.plan = append(e.plan, chunk{seq: s, tokens: 1, decode: true})
			ctx += s.kv
			nDecode++
		}
	}
	e.kvUsed += nDecode
	budget -= nDecode

	// Phase B: continue chunked prefills of admitted sequences (their KV is
	// reserved), in admission order.
	prefill := 0
	for _, s := range e.running {
		if budget == 0 {
			break
		}
		if rem := s.prefillTarget - s.prefilled; rem > 0 {
			c := min(rem, budget)
			e.plan = append(e.plan, chunk{seq: s, tokens: c})
			ctx += s.kv
			e.reserved -= c
			e.kvUsed += c
			budget -= c
			prefill += c
		}
	}

	// Phase C: admit from the head of the waiting queue while the sequence
	// limit, the token budget, and the KV cache (whole prompt) allow.
	for len(e.waiting) > 0 && budget > 0 && len(e.running) < e.p.MaxNumSeqs {
		s := e.waiting[0]
		if e.kvUsed+e.reserved+s.prefillTarget > e.p.KVCapacityTokens {
			break // FCFS: never skip the head
		}
		e.waiting = e.waiting[1:]
		s.waiting, s.running = false, true
		e.nextAdm++
		s.admitOrder = e.nextAdm
		if s.FirstScheduledAt < 0 {
			s.FirstScheduledAt = now
		}
		e.running = append(e.running, s)
		if skip := e.prefixHit(s); skip > 0 {
			// The cached prefix needs no prefill; its KV is still counted
			// for this sequence (compute is saved, memory is not).
			s.prefilled, s.kv = skip, skip
			e.kvUsed += skip
			ctx += skip
		}
		c := min(s.prefillTarget-s.prefilled, budget)
		e.plan = append(e.plan, chunk{seq: s, tokens: c})
		e.reserved += s.prefillTarget - s.prefilled - c
		e.kvUsed += c
		budget -= c
		prefill += c
	}
	if len(e.plan) == 0 {
		return 0, false
	}
	e.inIter = true
	e.iterStart = now
	return e.p.IterationTime(nDecode, ctx, prefill), true
}

func (e *Engine) preemptLast() {
	s := e.running[len(e.running)-1]
	e.running = e.running[:len(e.running)-1]
	e.kvUsed -= s.kv
	e.reserved -= s.prefillTarget - s.prefilled
	s.kv = 0
	s.prefilled = 0
	s.prefillTarget = s.Prompt + s.generated
	s.running, s.waiting = false, true
	s.Preemptions++
	e.preemptions++
	e.waiting = append([]*Seq{s}, e.waiting...)
	e.preempted = append(e.preempted, Event{Seq: s, Kind: Preempted})
}

// Finish applies the planned iteration at time now (the iteration's end) and
// returns the emitted tokens and preemptions.
func (e *Engine) Finish(now time.Duration) []Event {
	if !e.inIter {
		panic("engine: Finish without Start")
	}
	dur := now - e.iterStart
	e.busy += dur
	e.iterations++
	events := e.preempted
	e.preempted = nil
	anyDone := false
	for _, c := range e.plan {
		s := c.seq
		s.kv += c.tokens
		if !c.decode {
			s.prefilled += c.tokens
			if s.prefilled < s.prefillTarget {
				continue
			}
			if s.generated == 0 {
				e.prefixInsert(s)
			}
		}
		// A decode step, or the iteration that completes a prompt, emits one token.
		s.generated++
		e.generatedN++
		ev := Event{Seq: s, Kind: Token, First: s.generated == 1, Done: s.generated == s.Output}
		if ev.Done {
			anyDone = true
		}
		events = append(events, ev)
	}
	e.plan = e.plan[:0]
	// KV occupancy during the iteration, before completions free their KV.
	e.kvIntegral += float64(e.kvUsed) * dur.Seconds()
	if anyDone {
		keep := e.running[:0]
		for _, s := range e.running {
			if s.generated == s.Output {
				e.kvUsed -= s.kv
				s.running = false
				e.completed++
				continue
			}
			keep = append(keep, s)
		}
		for i := len(keep); i < len(e.running); i++ {
			e.running[i] = nil
		}
		e.running = keep
	}
	e.inIter = false
	return events
}

// Abort removes a sequence (client cancellation or drain timeout) and frees
// its KV. It must not be called during an iteration. It reports whether the
// sequence was present.
func (e *Engine) Abort(s *Seq) bool {
	if e.inIter {
		panic("engine: Abort during an iteration")
	}
	if s.waiting {
		for i, w := range e.waiting {
			if w == s {
				e.waiting = append(e.waiting[:i], e.waiting[i+1:]...)
				s.waiting = false
				return true
			}
		}
	}
	if s.running {
		for i, r := range e.running {
			if r == s {
				e.running = append(e.running[:i], e.running[i+1:]...)
				e.kvUsed -= s.kv
				e.reserved -= s.prefillTarget - s.prefilled
				s.kv = 0
				s.running = false
				return true
			}
		}
	}
	return false
}

// Reset drops every sequence (a crash or an aborted drain) at time now and
// returns them. An iteration in progress is discarded; the time it ran until
// now still counts as busy time and in the KV integral.
func (e *Engine) Reset(now time.Duration) []*Seq {
	if e.inIter && now > e.iterStart {
		d := now - e.iterStart
		e.busy += d
		e.kvIntegral += float64(e.kvUsed) * d.Seconds()
	}
	out := make([]*Seq, 0, len(e.running)+len(e.waiting))
	out = append(out, e.running...)
	out = append(out, e.waiting...)
	for _, s := range out {
		s.running, s.waiting, s.kv = false, false, 0
	}
	e.running, e.waiting, e.plan, e.preempted = nil, nil, nil, nil
	e.kvUsed, e.reserved = 0, 0
	e.inIter = false
	return out
}

// State is a point-in-time view of the engine.
type State struct {
	Running             int
	Waiting             int
	WaitingPromptTokens int
	KVUsed              int
	KVReserved          int
	KVCapacity          int
	Preemptions         int64
	Completed           int64
	Generated           int64
	Busy                time.Duration
	Iterations          int64
	KVIntegral          float64 // token·seconds
	// RemainingDecode is Σ (output − generated) over all sequences; it uses
	// output lengths, so only oracles may see it.
	RemainingDecode int
	PrefixHits      int64
	PrefixQueries   int64
}

// State returns counters and gauges.
func (e *Engine) State() State {
	wpt, rem := 0, 0
	for _, s := range e.waiting {
		wpt += s.prefillTarget - s.prefilled
		rem += s.Output - s.generated
	}
	for _, s := range e.running {
		wpt += s.prefillTarget - s.prefilled
		rem += s.Output - s.generated
	}
	return State{
		Running: len(e.running), Waiting: len(e.waiting), WaitingPromptTokens: wpt,
		KVUsed: e.kvUsed, KVReserved: e.reserved, KVCapacity: e.p.KVCapacityTokens,
		Preemptions: e.preemptions, Completed: e.completed, Generated: e.generatedN,
		Busy: e.busy, Iterations: e.iterations, KVIntegral: e.kvIntegral, RemainingDecode: rem,
		PrefixHits: e.pHits, PrefixQueries: e.pQuery,
	}
}

// prefixHit looks a newly admitted sequence up in the prefix cache and
// returns the number of prompt tokens whose prefill is skipped (0 on a miss,
// when the cache is disabled, or when the sequence was preempted before).
func (e *Engine) prefixHit(s *Seq) int {
	if e.p.PrefixCacheGroups <= 0 || s.PrefixGroup == "" || s.PrefixTokens <= 0 || s.generated > 0 || s.prefilled > 0 {
		return 0
	}
	e.pQuery++
	el, ok := e.lruIdx[s.PrefixGroup]
	if !ok {
		return 0
	}
	e.pHits++
	e.lru.MoveToFront(el)
	return min(s.PrefixTokens, s.Prompt-1)
}

// prefixInsert records a completed prefill of a sequence's prefix group.
func (e *Engine) prefixInsert(s *Seq) {
	if e.p.PrefixCacheGroups <= 0 || s.PrefixGroup == "" || s.PrefixTokens <= 0 {
		return
	}
	if el, ok := e.lruIdx[s.PrefixGroup]; ok {
		e.lru.MoveToFront(el)
		return
	}
	e.lruIdx[s.PrefixGroup] = e.lru.PushFront(s.PrefixGroup)
	for e.lru.Len() > e.p.PrefixCacheGroups {
		last := e.lru.Back()
		delete(e.lruIdx, last.Value.(string))
		e.lru.Remove(last)
	}
}

// PrefixStats returns cumulative prefix-cache hits and lookups.
func (e *Engine) PrefixStats() (hits, queries int64) { return e.pHits, e.pQuery }
