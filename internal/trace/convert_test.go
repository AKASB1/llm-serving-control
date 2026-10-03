package trace

import (
	"errors"
	"strings"
	"testing"
)

// Fixtures follow the documented formats of the public datasets.
const azureFixture = `TIMESTAMP,ContextTokens,GeneratedTokens
2023-11-16 18:15:46.6805900,374,44
2023-11-16 18:15:50.9951690,396,109
2023-11-16 18:16:46.6805900,20000,5000
2023-11-16 18:17:00.0000000,0,7
`

const burstFixture = `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
5,ChatGPT,472,18,490,Conversation log
45,ChatGPT,1087,230,1317,Conversation log
118,GPT-4,417,276,693,Conversation log
120,GPT-4,300,0,300,API log
`

func TestFromAzure(t *testing.T) {
	reqs, st, err := FromAzure(strings.NewReader(azureFixture), ConvertOptions{Model: "chat-8b", MaxContext: 16384})
	if err != nil {
		t.Fatal(err)
	}
	if st.Rows != 4 || st.Kept != 4 || st.Truncated != 1 || len(reqs) != 4 {
		t.Fatalf("stats %+v, %d requests", st, len(reqs))
	}
	if reqs[0].ArrivalS != 0 || reqs[1].ArrivalS != 4.314579 || reqs[1].PromptTokens != 396 || reqs[1].OutputTokens != 109 {
		t.Fatalf("row 1: %+v", reqs[1])
	}
	if reqs[2].PromptTokens != 16383 || reqs[2].OutputTokens != 1 {
		t.Fatalf("truncation: %+v", reqs[2])
	}
	if reqs[3].PromptTokens != 1 {
		t.Fatalf("zero prompt must become 1: %+v", reqs[3])
	}
	// Window [60 s, 120 s) holds the rows at 60.0 s and 73.3 s, re-based to 0.
	w, st, err := FromAzure(strings.NewReader(azureFixture), ConvertOptions{Model: "m", StartS: 60, DurationS: 60})
	if err != nil || len(w) != 2 || w[0].ArrivalS != 0 || w[1].ArrivalS != 13.31941 || st.Kept != 2 {
		t.Fatalf("window: %+v %+v %v", w, st, err)
	}
}

func TestFromAzureErrors(t *testing.T) {
	if _, _, err := FromAzure(strings.NewReader(azureFixture), ConvertOptions{}); err == nil {
		t.Fatal("model required")
	}
	_, _, err := FromAzure(strings.NewReader("TIMESTAMP,ContextTokens,GeneratedTokens\nnot-a-time,1,1\n"), ConvertOptions{Model: "m"})
	var le *LineError
	if !errors.As(err, &le) || le.Line != 2 {
		t.Fatalf("bad timestamp: %v", err)
	}
	if _, _, err := FromAzure(strings.NewReader("a,b\n1,2\n"), ConvertOptions{Model: "m"}); err == nil {
		t.Fatal("bad header accepted")
	}
	if _, _, err := FromAzure(strings.NewReader(""), ConvertOptions{Model: "m"}); err == nil {
		t.Fatal("empty input accepted")
	}
}

func TestFromBurstGPT(t *testing.T) {
	reqs, st, err := FromBurstGPT(strings.NewReader(burstFixture), ConvertOptions{ModelMap: map[string]string{"ChatGPT": "chat-8b", "GPT-4": "chat-70b"}})
	if err != nil {
		t.Fatal(err)
	}
	if st.Dropped != 1 || len(reqs) != 3 {
		t.Fatalf("failed rows must be dropped: %+v %d", st, len(reqs))
	}
	if reqs[0].ArrivalS != 0 || reqs[1].ArrivalS != 40 || reqs[2].Model != "chat-70b" || reqs[0].Model != "chat-8b" {
		t.Fatalf("%+v", reqs)
	}
	// Without a map the source name is sanitised.
	reqs, _, _ = FromBurstGPT(strings.NewReader(burstFixture), ConvertOptions{})
	if reqs[2].Model != "gpt-4" {
		t.Fatalf("sanitised model %q", reqs[2].Model)
	}
	// A single override wins over source names.
	reqs, _, _ = FromBurstGPT(strings.NewReader(burstFixture), ConvertOptions{Model: "x"})
	if reqs[2].Model != "x" {
		t.Fatalf("override %q", reqs[2].Model)
	}
}
