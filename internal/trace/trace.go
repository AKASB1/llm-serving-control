// Package trace implements trace schema v1 (docs/contracts.md §2): the CSV
// reader, writer, and validator, and the manifest that travels beside it.
package trace

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// SchemaVersion is the trace schema version this package reads and writes.
const SchemaVersion = 1

// Header is the exact CSV header of schema v1.
var Header = []string{"request_id", "arrival_s", "model", "prompt_tokens", "output_tokens", "prefix_group", "slo_class"}

// Request is one row of a trace.
type Request struct {
	ID           string
	ArrivalS     float64
	Model        string
	PromptTokens int
	OutputTokens int
	PrefixGroup  string
	SLOClass     string
}

// LineError reports a validation failure at a 1-based line of the CSV.
type LineError struct {
	Line int
	Msg  string
}

func (e *LineError) Error() string { return fmt.Sprintf("trace: line %d: %s", e.Line, e.Msg) }

// Read parses and validates a schema v1 trace.
func Read(r io.Reader) ([]Request, error) {
	cr := csv.NewReader(bufio.NewReader(r))
	cr.FieldsPerRecord = -1 // the count is checked here to report the line
	head, err := cr.Read()
	if errors.Is(err, io.EOF) {
		return nil, &LineError{Line: 1, Msg: "empty input: missing header"}
	}
	if err != nil {
		return nil, &LineError{Line: 1, Msg: err.Error()}
	}
	if len(head) > 0 {
		head[0] = strings.TrimPrefix(head[0], string(rune(0xFEFF)))
	}
	if strings.Join(head, ",") != strings.Join(Header, ",") {
		return nil, &LineError{Line: 1, Msg: fmt.Sprintf("header must be %q", strings.Join(Header, ","))}
	}
	var out []Request
	seen := map[string]int{}
	prev := 0.0
	for {
		rec, err := cr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			var pe *csv.ParseError
			if errors.As(err, &pe) {
				return nil, &LineError{Line: pe.Line, Msg: pe.Err.Error()}
			}
			return nil, &LineError{Line: 0, Msg: err.Error()}
		}
		line, _ := cr.FieldPos(0)
		if len(rec) != len(Header) {
			return nil, &LineError{Line: line, Msg: fmt.Sprintf("want %d fields, got %d", len(Header), len(rec))}
		}
		req, msg := parseRow(rec)
		if msg != "" {
			return nil, &LineError{Line: line, Msg: msg}
		}
		if first, dup := seen[req.ID]; dup {
			return nil, &LineError{Line: line, Msg: fmt.Sprintf("duplicate request_id %q (first on line %d)", req.ID, first)}
		}
		seen[req.ID] = line
		if req.ArrivalS < prev {
			return nil, &LineError{Line: line, Msg: fmt.Sprintf("arrival_s %v is before the previous row (%v): trace must be sorted", req.ArrivalS, prev)}
		}
		prev = req.ArrivalS
		out = append(out, req)
	}
	return out, nil
}

func parseRow(rec []string) (Request, string) {
	var r Request
	r.ID = rec[0]
	if r.ID == "" {
		return r, "empty request_id"
	}
	a, err := strconv.ParseFloat(rec[1], 64)
	if err != nil || math.IsNaN(a) || math.IsInf(a, 0) || a < 0 {
		return r, fmt.Sprintf("arrival_s %q must be a finite number >= 0", rec[1])
	}
	r.ArrivalS = a
	r.Model = rec[2]
	if r.Model == "" {
		return r, "empty model"
	}
	p, err := strconv.Atoi(rec[3])
	if err != nil || p < 1 {
		return r, fmt.Sprintf("prompt_tokens %q must be an integer >= 1", rec[3])
	}
	o, err := strconv.Atoi(rec[4])
	if err != nil || o < 1 {
		return r, fmt.Sprintf("output_tokens %q must be an integer >= 1", rec[4])
	}
	r.PromptTokens, r.OutputTokens = p, o
	r.PrefixGroup, r.SLOClass = rec[5], rec[6]
	return r, ""
}

// Validate applies the loader's rules to in-memory requests (line numbers
// count the header as line 1).
func Validate(reqs []Request) error {
	seen := map[string]bool{}
	prev := 0.0
	for i, r := range reqs {
		line := i + 2
		if r.ID == "" || strings.ContainsAny(r.ID, ",\"\r\n") {
			return &LineError{Line: line, Msg: "request_id must be non-empty and contain no comma, quote, or newline"}
		}
		if seen[r.ID] {
			return &LineError{Line: line, Msg: fmt.Sprintf("duplicate request_id %q", r.ID)}
		}
		seen[r.ID] = true
		if math.IsNaN(r.ArrivalS) || math.IsInf(r.ArrivalS, 0) || r.ArrivalS < 0 || r.ArrivalS < prev {
			return &LineError{Line: line, Msg: "arrival_s must be finite, >= 0, and sorted"}
		}
		prev = r.ArrivalS
		if r.Model == "" || strings.ContainsAny(r.Model, ",\"\r\n") {
			return &LineError{Line: line, Msg: "invalid model"}
		}
		if r.PromptTokens < 1 || r.OutputTokens < 1 {
			return &LineError{Line: line, Msg: "token counts must be >= 1"}
		}
		if strings.ContainsAny(r.PrefixGroup+r.SLOClass, ",\"\r\n") {
			return &LineError{Line: line, Msg: "prefix_group and slo_class must contain no comma, quote, or newline"}
		}
	}
	return nil
}

// Write validates reqs and writes them as schema v1 CSV (LF line endings,
// arrival_s with microsecond precision) so that output is byte-stable.
func Write(w io.Writer, reqs []Request) error {
	if err := Validate(reqs); err != nil {
		return err
	}
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(strings.Join(Header, ",") + "\n"); err != nil {
		return err
	}
	buf := make([]byte, 0, 128)
	for _, r := range reqs {
		buf = buf[:0]
		buf = append(buf, r.ID...)
		buf = append(buf, ',')
		buf = strconv.AppendFloat(buf, r.ArrivalS, 'f', 6, 64)
		buf = append(buf, ',')
		buf = append(buf, r.Model...)
		buf = append(buf, ',')
		buf = strconv.AppendInt(buf, int64(r.PromptTokens), 10)
		buf = append(buf, ',')
		buf = strconv.AppendInt(buf, int64(r.OutputTokens), 10)
		buf = append(buf, ',')
		buf = append(buf, r.PrefixGroup...)
		buf = append(buf, ',')
		buf = append(buf, r.SLOClass...)
		buf = append(buf, '\n')
		if _, err := bw.Write(buf); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// Generator identifies what produced a trace.
type Generator struct {
	Name    string          `json:"name"`
	Version int             `json:"version"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// Manifest is the JSON document stored beside a trace.
type Manifest struct {
	SchemaVersion int       `json:"schema_version"`
	Generator     Generator `json:"generator"`
	Seed          uint64    `json:"seed"`
	Requests      int       `json:"requests"`
	DurationS     float64   `json:"duration_s"`
	ContentSHA256 string    `json:"content_sha256"`
}

// ManifestPath returns the manifest path for a trace CSV path.
func ManifestPath(csvPath string) string {
	return strings.TrimSuffix(csvPath, filepath.Ext(csvPath)) + ".manifest.json"
}

// Encode returns the CSV bytes of reqs and their SHA-256 (hex).
func Encode(reqs []Request) ([]byte, string, error) {
	var b bytes.Buffer
	if err := Write(&b, reqs); err != nil {
		return nil, "", err
	}
	sum := sha256.Sum256(b.Bytes())
	return b.Bytes(), hex.EncodeToString(sum[:]), nil
}

// WriteFiles writes the CSV and its manifest. SchemaVersion, Requests, and
// ContentSHA256 are filled in here.
func WriteFiles(csvPath string, reqs []Request, m Manifest) (Manifest, error) {
	data, sum, err := Encode(reqs)
	if err != nil {
		return m, err
	}
	m.SchemaVersion = SchemaVersion
	m.Requests = len(reqs)
	m.ContentSHA256 = sum
	if err := os.MkdirAll(filepath.Dir(csvPath), 0o755); err != nil {
		return m, err
	}
	if err := os.WriteFile(csvPath, data, 0o644); err != nil {
		return m, err
	}
	mj, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return m, err
	}
	return m, os.WriteFile(ManifestPath(csvPath), append(mj, '\n'), 0o644)
}

// LoadFiles reads a trace and its manifest and verifies version and hash.
func LoadFiles(csvPath string) ([]Request, Manifest, error) {
	var m Manifest
	mj, err := os.ReadFile(ManifestPath(csvPath))
	if err != nil {
		return nil, m, fmt.Errorf("trace: manifest: %w", err)
	}
	if err := json.Unmarshal(mj, &m); err != nil {
		return nil, m, fmt.Errorf("trace: manifest: %w", err)
	}
	if m.SchemaVersion != SchemaVersion {
		return nil, m, fmt.Errorf("trace: unsupported schema_version %d", m.SchemaVersion)
	}
	data, err := os.ReadFile(csvPath)
	if err != nil {
		return nil, m, err
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != m.ContentSHA256 {
		return nil, m, fmt.Errorf("trace: content hash mismatch for %s: manifest %s, file %s", filepath.Base(csvPath), m.ContentSHA256, got)
	}
	reqs, err := Read(bytes.NewReader(data))
	if err != nil {
		return nil, m, err
	}
	if len(reqs) != m.Requests {
		return nil, m, fmt.Errorf("trace: manifest says %d requests, file has %d", m.Requests, len(reqs))
	}
	return reqs, m, nil
}
