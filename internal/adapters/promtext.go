// Package adapters turns serving-backend metrics (Prometheus text format)
// into the control plane's replica snapshots. The vLLM and SGLang mappings are
// written from those projects' metric definitions (see docs/contracts.md and
// the fixtures); they have not been run against a live vLLM or SGLang server.
package adapters

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Sample is one Prometheus sample.
type Sample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// ParseError reports a malformed line.
type ParseError struct {
	Line int
	Msg  string
}

func (e *ParseError) Error() string { return fmt.Sprintf("promtext: line %d: %s", e.Line, e.Msg) }

// Parse reads the Prometheus text exposition format (version 0.0.4). Comment
// and blank lines are skipped. Malformed sample lines are skipped too; the
// first one is reported as the error, so callers may use the samples that
// did parse (tolerant scraping) or reject the input (strict use).
func Parse(text string) ([]Sample, error) {
	var out []Sample
	var first error
	for i, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		s, err := parseLine(trimmed)
		if err != nil {
			if first == nil {
				first = &ParseError{Line: i + 1, Msg: err.Error()}
			}
			continue
		}
		out = append(out, s)
	}
	return out, first
}

func isNameStart(c byte) bool {
	return c == '_' || c == ':' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isNameChar(c byte) bool { return isNameStart(c) || (c >= '0' && c <= '9') }

func parseLine(line string) (Sample, error) {
	i := 0
	if i >= len(line) || !isNameStart(line[i]) {
		return Sample{}, fmt.Errorf("bad metric name")
	}
	for i < len(line) && isNameChar(line[i]) {
		i++
	}
	s := Sample{Name: line[:i]}
	if i < len(line) && line[i] == '{' {
		labels, n, err := parseLabels(line[i+1:])
		if err != nil {
			return Sample{}, err
		}
		s.Labels = labels
		i += 1 + n
	}
	rest := strings.Fields(line[i:])
	if len(rest) < 1 || len(rest) > 2 {
		return Sample{}, fmt.Errorf("want value and optional timestamp")
	}
	v, err := parseValue(rest[0])
	if err != nil {
		return Sample{}, err
	}
	if len(rest) == 2 {
		if _, err := strconv.ParseInt(rest[1], 10, 64); err != nil {
			return Sample{}, fmt.Errorf("bad timestamp %q", rest[1])
		}
	}
	s.Value = v
	return s, nil
}

func parseValue(s string) (float64, error) {
	switch s {
	case "+Inf", "Inf":
		return math.Inf(1), nil
	case "-Inf":
		return math.Inf(-1), nil
	case "NaN":
		return math.NaN(), nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("bad value %q", s)
	}
	return v, nil
}

// parseLabels parses `name="value",...}` and returns the labels and the
// number of bytes consumed including the closing brace.
func parseLabels(s string) (map[string]string, int, error) {
	labels := map[string]string{}
	i := 0
	for {
		for i < len(s) && (s[i] == ' ' || s[i] == ',') {
			i++
		}
		if i >= len(s) {
			return nil, 0, fmt.Errorf("unterminated label set")
		}
		if s[i] == '}' {
			return labels, i + 1, nil
		}
		start := i
		if !isNameStart(s[i]) || s[i] == ':' {
			return nil, 0, fmt.Errorf("bad label name")
		}
		for i < len(s) && isNameChar(s[i]) && s[i] != ':' {
			i++
		}
		name := s[start:i]
		for i < len(s) && s[i] == ' ' {
			i++
		}
		if i+1 >= len(s) || s[i] != '=' || s[i+1] != '"' {
			return nil, 0, fmt.Errorf("label %q: want =\"", name)
		}
		i += 2
		var b strings.Builder
		closed := false
		for i < len(s) {
			c := s[i]
			if c == '\\' {
				if i+1 >= len(s) {
					return nil, 0, fmt.Errorf("label %q: dangling escape", name)
				}
				switch s[i+1] {
				case 'n':
					b.WriteByte('\n')
				case '\\':
					b.WriteByte('\\')
				case '"':
					b.WriteByte('"')
				default:
					return nil, 0, fmt.Errorf("label %q: bad escape", name)
				}
				i += 2
				continue
			}
			if c == '"' {
				closed = true
				i++
				break
			}
			b.WriteByte(c)
			i++
		}
		if !closed {
			return nil, 0, fmt.Errorf("label %q: unterminated value", name)
		}
		labels[name] = b.String()
	}
}

// Family indexes samples by metric name.
type Family map[string][]Sample

// Index groups samples by name.
func Index(samples []Sample) Family {
	f := Family{}
	for _, s := range samples {
		f[s.Name] = append(f[s.Name], s)
	}
	return f
}

// Sum returns the sum of the first present name's samples (names are tried in
// order, so renamed metrics can be listed as fallbacks).
func (f Family) Sum(names ...string) (float64, bool) {
	for _, n := range names {
		if ss, ok := f[n]; ok {
			t := 0.0
			for _, s := range ss {
				if !math.IsNaN(s.Value) {
					t += s.Value
				}
			}
			return t, true
		}
	}
	return 0, false
}

// Mean returns the mean of the first present name's samples.
func (f Family) Mean(names ...string) (float64, bool) {
	for _, n := range names {
		if ss, ok := f[n]; ok && len(ss) > 0 {
			t, k := 0.0, 0
			for _, s := range ss {
				if !math.IsNaN(s.Value) {
					t += s.Value
					k++
				}
			}
			if k == 0 {
				return 0, false
			}
			return t / float64(k), true
		}
	}
	return 0, false
}
