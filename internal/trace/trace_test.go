package trace

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const head = "request_id,arrival_s,model,prompt_tokens,output_tokens,prefix_group,slo_class\n"

func TestReadValid(t *testing.T) {
	in := head +
		"r1,0.000000,m,10,5,,\n" +
		"r2,0.5,m,20,1,g1,batch\r\n" +
		"r3,0.5,m2,1,7,,interactive\n"
	reqs, err := Read(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 3 || reqs[1].PrefixGroup != "g1" || reqs[1].SLOClass != "batch" || reqs[2].Model != "m2" || reqs[1].ArrivalS != 0.5 {
		t.Fatalf("parsed %+v", reqs)
	}
}

func TestReadHeaderOnlyIsEmptyTrace(t *testing.T) {
	reqs, err := Read(strings.NewReader(head))
	if err != nil || len(reqs) != 0 {
		t.Fatalf("header-only: %v %v", reqs, err)
	}
}

func TestReadEmptyInput(t *testing.T) {
	_, err := Read(strings.NewReader(""))
	assertLine(t, err, 1, "missing header")
}

func TestReadUnsorted(t *testing.T) {
	in := head + "r1,1.0,m,10,5,,\nr2,0.9,m,10,5,,\n"
	_, err := Read(strings.NewReader(in))
	assertLine(t, err, 3, "sorted")
}

func TestReadMalformed(t *testing.T) {
	cases := []struct {
		name string
		body string
		line int
		frag string
	}{
		{"bad header", "id,arrival\n", 1, "header"},
		{"field count", head + "r1,0,m,1,1,\n", 2, "fields"},
		{"bad arrival", head + "r1,abc,m,1,1,,\n", 2, "arrival_s"},
		{"negative arrival", head + "r1,-1,m,1,1,,\n", 2, "arrival_s"},
		{"nan arrival", head + "r1,NaN,m,1,1,,\n", 2, "arrival_s"},
		{"zero prompt", head + "r1,0,m,0,1,,\n", 2, "prompt_tokens"},
		{"float output", head + "r1,0,m,1,1.5,,\n", 2, "output_tokens"},
		{"empty id", head + ",0,m,1,1,,\n", 2, "request_id"},
		{"empty model", head + "r1,0,,1,1,,\n", 2, "model"},
		{"duplicate id", head + "r1,0,m,1,1,,\nr2,0,m,1,1,,\nr1,1,m,1,1,,\n", 4, "duplicate"},
		{"bad quote", head + "r1,0,m,1,1,\"x,\n", 2, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Read(strings.NewReader(c.body))
			assertLine(t, err, c.line, c.frag)
		})
	}
}

func assertLine(t *testing.T, err error, line int, frag string) {
	t.Helper()
	var le *LineError
	if !errors.As(err, &le) {
		t.Fatalf("want LineError, got %v", err)
	}
	if le.Line != line {
		t.Fatalf("want line %d, got %d (%v)", line, le.Line, err)
	}
	if frag != "" && !strings.Contains(le.Msg, frag) {
		t.Fatalf("message %q lacks %q", le.Msg, frag)
	}
}

func TestWriteReadRoundTripAndStableBytes(t *testing.T) {
	reqs := []Request{
		{ID: "a", ArrivalS: 0.1234567, Model: "m", PromptTokens: 3, OutputTokens: 9},
		{ID: "b", ArrivalS: 2, Model: "m", PromptTokens: 1, OutputTokens: 1, PrefixGroup: "p", SLOClass: "batch"},
	}
	var b1, b2 bytes.Buffer
	if err := Write(&b1, reqs); err != nil {
		t.Fatal(err)
	}
	_ = Write(&b2, reqs)
	if b1.String() != b2.String() {
		t.Fatal("Write is not byte-stable")
	}
	back, err := Read(&b1)
	if err != nil {
		t.Fatal(err)
	}
	if back[0].ArrivalS != 0.123457 || back[1] != reqs[1] {
		t.Fatalf("round trip: %+v", back)
	}
}

func TestWriteRejectsInvalid(t *testing.T) {
	err := Write(&bytes.Buffer{}, []Request{{ID: "a", ArrivalS: 1, Model: "m", PromptTokens: 1, OutputTokens: 1}, {ID: "b", ArrivalS: 0.5, Model: "m", PromptTokens: 1, OutputTokens: 1}})
	assertLine(t, err, 3, "sorted")
	err = Write(&bytes.Buffer{}, []Request{{ID: "a,b", Model: "m", PromptTokens: 1, OutputTokens: 1}})
	assertLine(t, err, 2, "request_id")
}

func TestFilesAndManifestHash(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "t.csv")
	reqs := []Request{{ID: "a", ArrivalS: 0, Model: "m", PromptTokens: 1, OutputTokens: 2}}
	m, err := WriteFiles(p, reqs, Manifest{Seed: 3, DurationS: 10, Generator: Generator{Name: "test", Version: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if m.Requests != 1 || len(m.ContentSHA256) != 64 || m.SchemaVersion != 1 {
		t.Fatalf("manifest %+v", m)
	}
	got, m2, err := LoadFiles(p)
	if err != nil || len(got) != 1 || m2.Seed != 3 {
		t.Fatalf("load: %v %+v", err, m2)
	}
	// Tampering with the CSV must be detected.
	if err := os.WriteFile(p, []byte(head+"a,0.000000,m,1,3,,\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadFiles(p); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("tamper not detected: %v", err)
	}
}
