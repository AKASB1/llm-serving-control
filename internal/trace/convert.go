package trace

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ConvertOptions control the conversion of a public trace to schema v1.
type ConvertOptions struct {
	// Model replaces the source's model name (required for sources without
	// one; for BurstGPT, ModelMap takes precedence when it has an entry).
	Model    string
	ModelMap map[string]string
	// StartS and DurationS select a window of the source, in seconds from the
	// source's first request; arrivals are re-based to the window start.
	// DurationS <= 0 keeps everything after StartS.
	StartS    float64
	DurationS float64
	// MaxContext caps prompt + output; longer requests keep their prompt
	// (capped at MaxContext−1) and lose output tokens. 0 disables the cap.
	MaxContext int
	// IDPrefix prefixes request IDs (default "r").
	IDPrefix string
}

// ConvertStats reports what a conversion kept and changed.
type ConvertStats struct {
	Rows      int // data rows read
	Kept      int // rows inside the window
	Dropped   int // rows dropped (zero output tokens, i.e. failed requests)
	Truncated int // rows shortened to MaxContext
}

// FromAzure converts the Azure LLM inference trace (2023 format: header
// TIMESTAMP,ContextTokens,GeneratedTokens; timestamps "2006-01-02
// 15:04:05.9999999") to schema v1.
func FromAzure(r io.Reader, o ConvertOptions) ([]Request, ConvertStats, error) {
	if o.Model == "" {
		return nil, ConvertStats{}, errors.New("convert: Azure traces need a model name")
	}
	rows, header, err := readCSV(r)
	if err != nil {
		return nil, ConvertStats{}, err
	}
	ti, pi, oi := col(header, "TIMESTAMP"), col(header, "ContextTokens"), col(header, "GeneratedTokens")
	if ti < 0 || pi < 0 || oi < 0 {
		return nil, ConvertStats{}, errors.New("convert: Azure header must contain TIMESTAMP, ContextTokens, GeneratedTokens")
	}
	var raw []rawReq
	var t0 time.Time
	for i, rec := range rows {
		ts, err := time.Parse("2006-01-02 15:04:05.9999999", strings.TrimSpace(rec[ti]))
		if err != nil {
			return nil, ConvertStats{}, &LineError{Line: i + 2, Msg: "bad TIMESTAMP: " + err.Error()}
		}
		if i == 0 {
			t0 = ts
		}
		p, o1, err := ints(rec[pi], rec[oi])
		if err != nil {
			return nil, ConvertStats{}, &LineError{Line: i + 2, Msg: err.Error()}
		}
		raw = append(raw, rawReq{t: ts.Sub(t0).Seconds(), model: o.Model, prompt: p, output: o1})
	}
	return finish(raw, o)
}

// FromBurstGPT converts a BurstGPT trace (header with Timestamp, Model,
// Request tokens, Response tokens; timestamps in seconds) to schema v1.
// Rows with zero response tokens (failed requests in the source) are dropped.
func FromBurstGPT(r io.Reader, o ConvertOptions) ([]Request, ConvertStats, error) {
	rows, header, err := readCSV(r)
	if err != nil {
		return nil, ConvertStats{}, err
	}
	ti, mi, pi, oi := col(header, "Timestamp"), col(header, "Model"), col(header, "Request tokens"), col(header, "Response tokens")
	if ti < 0 || pi < 0 || oi < 0 {
		return nil, ConvertStats{}, errors.New("convert: BurstGPT header must contain Timestamp, Request tokens, Response tokens")
	}
	var raw []rawReq
	t0 := math.NaN()
	for i, rec := range rows {
		ts, err := strconv.ParseFloat(strings.TrimSpace(rec[ti]), 64)
		if err != nil || math.IsNaN(ts) || math.IsInf(ts, 0) {
			return nil, ConvertStats{}, &LineError{Line: i + 2, Msg: fmt.Sprintf("bad Timestamp %q", rec[ti])}
		}
		if math.IsNaN(t0) {
			t0 = ts
		}
		p, o1, err := ints(rec[pi], rec[oi])
		if err != nil {
			return nil, ConvertStats{}, &LineError{Line: i + 2, Msg: err.Error()}
		}
		model := o.Model
		if mi >= 0 {
			if m, ok := o.ModelMap[rec[mi]]; ok {
				model = m
			} else if model == "" {
				model = sanitize(rec[mi])
			}
		}
		if model == "" {
			return nil, ConvertStats{}, &LineError{Line: i + 2, Msg: "no model name (set Model or ModelMap)"}
		}
		raw = append(raw, rawReq{t: ts - t0, model: model, prompt: p, output: o1})
	}
	return finish(raw, o)
}

type rawReq struct {
	t              float64
	model          string
	prompt, output int
}

func readCSV(r io.Reader) ([][]string, []string, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	all, err := cr.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("convert: %w", err)
	}
	if len(all) == 0 {
		return nil, nil, &LineError{Line: 1, Msg: "empty input: missing header"}
	}
	header := all[0]
	if len(header) > 0 {
		header[0] = strings.TrimPrefix(header[0], string(rune(0xFEFF)))
	}
	for i, rec := range all[1:] {
		if len(rec) != len(header) {
			return nil, nil, &LineError{Line: i + 2, Msg: fmt.Sprintf("want %d fields, got %d", len(header), len(rec))}
		}
	}
	return all[1:], header, nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if strings.EqualFold(strings.TrimSpace(h), name) {
			return i
		}
	}
	return -1
}

func ints(a, b string) (int, int, error) {
	p, err := strconv.Atoi(strings.TrimSpace(a))
	if err != nil || p < 0 {
		return 0, 0, fmt.Errorf("bad prompt token count %q", a)
	}
	o, err := strconv.Atoi(strings.TrimSpace(b))
	if err != nil || o < 0 {
		return 0, 0, fmt.Errorf("bad output token count %q", b)
	}
	return p, o, nil
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r == ',' || r == '"' || r == '\n' || r == '\r' || r == ' ' {
			return '-'
		}
		return r
	}, strings.ToLower(s))
}

// finish windows, cleans, sorts, and numbers the requests.
func finish(raw []rawReq, o ConvertOptions) ([]Request, ConvertStats, error) {
	st := ConvertStats{Rows: len(raw)}
	prefix := o.IDPrefix
	if prefix == "" {
		prefix = "r"
	}
	sort.SliceStable(raw, func(i, j int) bool { return raw[i].t < raw[j].t })
	var out []Request
	for _, r := range raw {
		if r.t < o.StartS || (o.DurationS > 0 && r.t >= o.StartS+o.DurationS) {
			continue
		}
		st.Kept++
		if r.output == 0 {
			st.Dropped++
			continue
		}
		p, out1 := max(r.prompt, 1), r.output
		if o.MaxContext > 0 && p+out1 > o.MaxContext {
			st.Truncated++
			p = min(p, o.MaxContext-1)
			out1 = o.MaxContext - p
		}
		out = append(out, Request{
			ID:       fmt.Sprintf("%s%06d", prefix, len(out)),
			ArrivalS: math.Round((r.t-o.StartS)*1e6) / 1e6, Model: r.model, PromptTokens: p, OutputTokens: out1,
		})
	}
	return out, st, Validate(out)
}
