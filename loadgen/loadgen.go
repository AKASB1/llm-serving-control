// Package loadgen generates synthetic workloads in trace schema v1: arrival
// processes (Poisson, Markov-modulated on/off, gamma renewal, diurnal by
// thinning), prompt and output length distributions with heavy tails, SLO
// class mixes, prefix groups, and several models with shifting demand.
//
// A trace depends only on (Config, seed): every model stream draws from its
// own named RNG streams, so adding a stream or changing a policy never changes
// another stream's requests.
package loadgen

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/AKASB1/llm-serving-control/internal/rng"
	"github.com/AKASB1/llm-serving-control/internal/trace"
)

// Version is recorded in trace manifests; bump it when generation changes.
const Version = 1

// Dist is a token-length distribution, rounded to integers and clipped to
// [Min, Max].
type Dist struct {
	// Kind is "lognormal" (Median, Sigma), "pareto" (Xm, Alpha), or "constant" (Value).
	Kind   string  `json:"kind"`
	Median float64 `json:"median,omitempty"`
	Sigma  float64 `json:"sigma,omitempty"`
	Xm     float64 `json:"xm,omitempty"`
	Alpha  float64 `json:"alpha,omitempty"`
	Value  int     `json:"value,omitempty"`
	Min    int     `json:"min"`
	Max    int     `json:"max"`
}

// Validate checks the distribution parameters.
func (d Dist) Validate() error {
	if d.Min < 1 || d.Max < d.Min {
		return fmt.Errorf("dist: need 1 <= min <= max, got [%d, %d]", d.Min, d.Max)
	}
	switch d.Kind {
	case "lognormal":
		if d.Median <= 0 || d.Sigma < 0 {
			return errors.New("dist lognormal: median > 0 and sigma >= 0 required")
		}
	case "pareto":
		if d.Xm <= 0 || d.Alpha <= 0 {
			return errors.New("dist pareto: xm > 0 and alpha > 0 required")
		}
	case "constant":
		if d.Value < 1 {
			return errors.New("dist constant: value >= 1 required")
		}
	default:
		return fmt.Errorf("dist: unknown kind %q", d.Kind)
	}
	return nil
}

// Sample draws one length. It consumes exactly one normal or uniform draw so
// that changing a parameter keeps draws aligned across variants.
func (d Dist) Sample(r *rand.Rand) int {
	var x float64
	switch d.Kind {
	case "lognormal":
		x = math.Exp(math.Log(d.Median) + d.Sigma*r.NormFloat64())
	case "pareto":
		u := 1 - r.Float64() // (0, 1]
		x = d.Xm / math.Pow(u, 1/d.Alpha)
	default:
		x = float64(d.Value)
	}
	n := int(math.Round(x))
	if n < d.Min {
		n = d.Min
	}
	if n > d.Max {
		n = d.Max
	}
	return n
}

// Arrival describes an arrival process.
type Arrival struct {
	// Kind is "poisson" (Rate), "gamma" (Rate, CV), "mmpp" (Rates,
	// MeanSojournS: states visited cyclically, exponential sojourns), or
	// "diurnal" (Rate·(1 + Amplitude·sin(2π(t/PeriodS + Phase))), by
	// thinning), or "piecewise" (Poisson with rate Rates[i] from StartsS[i]
	// until the next start).
	Kind         string    `json:"kind"`
	Rate         float64   `json:"rate,omitempty"`
	CV           float64   `json:"cv,omitempty"`
	Rates        []float64 `json:"rates,omitempty"`
	MeanSojournS []float64 `json:"mean_sojourn_s,omitempty"`
	Amplitude    float64   `json:"amplitude,omitempty"`
	PeriodS      float64   `json:"period_s,omitempty"`
	Phase        float64   `json:"phase,omitempty"`
	StartsS      []float64 `json:"starts_s,omitempty"`
}

// Validate checks the arrival parameters.
func (a Arrival) Validate() error {
	switch a.Kind {
	case "poisson":
		if a.Rate <= 0 {
			return errors.New("poisson: rate > 0 required")
		}
	case "gamma":
		if a.Rate <= 0 || a.CV <= 0 {
			return errors.New("gamma: rate > 0 and cv > 0 required")
		}
	case "mmpp":
		if len(a.Rates) < 2 || len(a.Rates) != len(a.MeanSojournS) {
			return errors.New("mmpp: at least two states with rates and mean_sojourn_s required")
		}
		for i := range a.Rates {
			if a.Rates[i] < 0 || a.MeanSojournS[i] <= 0 {
				return errors.New("mmpp: rates >= 0 and sojourns > 0 required")
			}
		}
		if a.MeanRate() <= 0 {
			return errors.New("mmpp: mean rate must be positive")
		}
	case "diurnal":
		if a.Rate <= 0 || a.Amplitude < 0 || a.Amplitude >= 1 || a.PeriodS <= 0 {
			return errors.New("diurnal: rate > 0, 0 <= amplitude < 1, period_s > 0 required")
		}
	case "piecewise":
		if len(a.Rates) == 0 || len(a.Rates) != len(a.StartsS) || a.StartsS[0] != 0 {
			return errors.New("piecewise: rates and starts_s of equal length, starting at 0, required")
		}
		for i := range a.Rates {
			if a.Rates[i] < 0 || (i > 0 && a.StartsS[i] <= a.StartsS[i-1]) {
				return errors.New("piecewise: rates >= 0 and increasing starts required")
			}
		}
	default:
		return fmt.Errorf("arrival: unknown kind %q", a.Kind)
	}
	return nil
}

// MeanRate returns the long-run mean arrival rate (requests/s); for
// "piecewise" it is the largest segment rate (the peak).
func (a Arrival) MeanRate() float64 {
	if a.Kind == "piecewise" {
		m := 0.0
		for _, x := range a.Rates {
			m = math.Max(m, x)
		}
		return m
	}
	if a.Kind == "mmpp" {
		num, den := 0.0, 0.0
		for i := range a.Rates {
			num += a.Rates[i] * a.MeanSojournS[i]
			den += a.MeanSojournS[i]
		}
		return num / den
	}
	return a.Rate
}

// RateAt returns the instantaneous rate of a diurnal process (the mean rate
// for the other kinds).
func (a Arrival) RateAt(t float64) float64 {
	switch a.Kind {
	case "diurnal":
		return a.Rate * (1 + a.Amplitude*math.Sin(2*math.Pi*(t/a.PeriodS+a.Phase)))
	case "piecewise":
		r := a.Rates[0]
		for i, s := range a.StartsS {
			if t >= s {
				r = a.Rates[i]
			}
		}
		return r
	}
	return a.MeanRate()
}

// Scaled returns a copy with every rate multiplied by f.
func (a Arrival) Scaled(f float64) Arrival {
	b := a
	b.Rate *= f
	if a.Rates != nil {
		b.Rates = make([]float64, len(a.Rates))
		for i, r := range a.Rates {
			b.Rates[i] = r * f
		}
	}
	return b
}

// Times draws arrival times in [0, duration).
func (a Arrival) Times(r *rand.Rand, duration float64) []float64 {
	var out []float64
	switch a.Kind {
	case "poisson":
		for t := r.ExpFloat64() / a.Rate; t < duration; t += r.ExpFloat64() / a.Rate {
			out = append(out, t)
		}
	case "gamma":
		k := 1 / (a.CV * a.CV)
		theta := 1 / (a.Rate * k)
		for t := gamma(r, k) * theta; t < duration; t += gamma(r, k) * theta {
			out = append(out, t)
		}
	case "mmpp":
		// Start in a state drawn from the time-stationary distribution.
		total := 0.0
		for _, s := range a.MeanSojournS {
			total += s
		}
		u, state := r.Float64()*total, 0
		for i, s := range a.MeanSojournS {
			if u < s {
				state = i
				break
			}
			u -= s
		}
		t := 0.0
		for t < duration {
			end := t + r.ExpFloat64()*a.MeanSojournS[state]
			if rate := a.Rates[state]; rate > 0 {
				for x := t + r.ExpFloat64()/rate; x < end && x < duration; x += r.ExpFloat64() / rate {
					out = append(out, x)
				}
			}
			t = end
			state = (state + 1) % len(a.Rates)
		}
	case "diurnal":
		peak := a.Rate * (1 + a.Amplitude)
		for t := r.ExpFloat64() / peak; t < duration; t += r.ExpFloat64() / peak {
			if r.Float64()*peak < a.RateAt(t) {
				out = append(out, t)
			}
		}
	case "piecewise":
		peak := 0.0
		for _, x := range a.Rates {
			peak = math.Max(peak, x)
		}
		if peak == 0 {
			return nil
		}
		for t := r.ExpFloat64() / peak; t < duration; t += r.ExpFloat64() / peak {
			if r.Float64()*peak < a.RateAt(t) {
				out = append(out, t)
			}
		}
	}
	return out
}

// gamma draws Gamma(k, 1) (Marsaglia-Tsang, with the k < 1 boost).
func gamma(r *rand.Rand, k float64) float64 {
	if k < 1 {
		u := 1 - r.Float64()
		return gamma(r, k+1) * math.Pow(u, 1/k)
	}
	d := k - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := r.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := 1 - r.Float64()
		if math.Log(u) < 0.5*x*x+d-d*v+d*math.Log(v) {
			return d * v
		}
	}
}

// ClassShare is the probability of one SLO class within a stream.
type ClassShare struct {
	Class string  `json:"class"`
	Share float64 `json:"share"`
}

// Stream is the traffic of one model (or one model and class mix).
type Stream struct {
	// Name prefixes request IDs; defaults to Model.
	Name    string  `json:"name,omitempty"`
	Model   string  `json:"model"`
	Arrival Arrival `json:"arrival"`
	Prompt  Dist    `json:"prompt"`
	Output  Dist    `json:"output"`
	// Classes is the SLO-class mix; empty means every request has an empty
	// class (the experiment default).
	Classes []ClassShare `json:"classes,omitempty"`
	// PrefixGroups > 0 assigns each request, with probability PrefixShare, to
	// one of PrefixGroups shared-prefix groups (Zipf-like weights 1/(i+1)).
	PrefixGroups int     `json:"prefix_groups,omitempty"`
	PrefixShare  float64 `json:"prefix_share,omitempty"`
	// MaxContext caps prompt+output (the output is shortened); 0 = no cap.
	MaxContext int `json:"max_context,omitempty"`
}

// Config is a complete workload description.
type Config struct {
	Name      string   `json:"name"`
	DurationS float64  `json:"duration_s"`
	Streams   []Stream `json:"streams"`
}

// Validate checks the configuration.
func (c Config) Validate() error {
	if c.DurationS <= 0 {
		return errors.New("loadgen: duration_s > 0 required")
	}
	if len(c.Streams) == 0 {
		return errors.New("loadgen: at least one stream required")
	}
	names := map[string]bool{}
	for i, s := range c.Streams {
		if s.Model == "" {
			return fmt.Errorf("stream %d: model required", i)
		}
		n := s.name()
		if names[n] {
			return fmt.Errorf("stream %d: duplicate stream name %q", i, n)
		}
		names[n] = true
		if err := s.Arrival.Validate(); err != nil {
			return fmt.Errorf("stream %q: %w", n, err)
		}
		if err := s.Prompt.Validate(); err != nil {
			return fmt.Errorf("stream %q prompt: %w", n, err)
		}
		if err := s.Output.Validate(); err != nil {
			return fmt.Errorf("stream %q output: %w", n, err)
		}
		for _, cs := range s.Classes {
			if cs.Share < 0 {
				return fmt.Errorf("stream %q: negative class share", n)
			}
		}
		if s.PrefixGroups < 0 || s.PrefixShare < 0 || s.PrefixShare > 1 {
			return fmt.Errorf("stream %q: invalid prefix settings", n)
		}
		if s.MaxContext != 0 && s.MaxContext < s.Prompt.Max+1 {
			return fmt.Errorf("stream %q: max_context must exceed prompt max", n)
		}
	}
	return nil
}

func (s Stream) name() string {
	if s.Name != "" {
		return s.Name
	}
	return s.Model
}

// Generate produces the trace of cfg for seed, sorted by arrival time with
// arrival times rounded to the trace precision (microseconds).
func Generate(cfg Config, seed uint64) ([]trace.Request, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	type tagged struct {
		req    trace.Request
		stream int
		idx    int
	}
	var all []tagged
	for si, s := range cfg.Streams {
		name := s.name()
		ra := rng.Stream(seed, "loadgen/arrivals/"+name)
		rl := rng.Stream(seed, "loadgen/lengths/"+name)
		rc := rng.Stream(seed, "loadgen/classes/"+name)
		rp := rng.Stream(seed, "loadgen/prefix/"+name)
		times := s.Arrival.Times(ra, cfg.DurationS)
		classTotal := 0.0
		for _, c := range s.Classes {
			classTotal += c.Share
		}
		prefixTotal := 0.0
		for g := 0; g < s.PrefixGroups; g++ {
			prefixTotal += 1 / float64(g+1)
		}
		for i, t := range times {
			req := trace.Request{
				ID:           fmt.Sprintf("%s-%06d", name, i),
				ArrivalS:     math.Round(t*1e6) / 1e6,
				Model:        s.Model,
				PromptTokens: s.Prompt.Sample(rl),
				OutputTokens: s.Output.Sample(rl),
			}
			if s.MaxContext > 0 && req.PromptTokens+req.OutputTokens > s.MaxContext {
				req.OutputTokens = s.MaxContext - req.PromptTokens
			}
			if classTotal > 0 {
				u := rc.Float64() * classTotal
				req.SLOClass = s.Classes[len(s.Classes)-1].Class
				for _, c := range s.Classes {
					if u < c.Share {
						req.SLOClass = c.Class
						break
					}
					u -= c.Share
				}
			}
			if s.PrefixGroups > 0 {
				in, u := rp.Float64(), rp.Float64()*prefixTotal
				if in < s.PrefixShare {
					for g := 0; g < s.PrefixGroups; g++ {
						w := 1 / float64(g+1)
						if u < w || g == s.PrefixGroups-1 {
							req.PrefixGroup = fmt.Sprintf("%s-p%d", name, g)
							break
						}
						u -= w
					}
				}
			}
			all = append(all, tagged{req: req, stream: si, idx: i})
		}
	}
	sort.SliceStable(all, func(i, j int) bool {
		a, b := all[i], all[j]
		if a.req.ArrivalS != b.req.ArrivalS {
			return a.req.ArrivalS < b.req.ArrivalS
		}
		if a.stream != b.stream {
			return a.stream < b.stream
		}
		return a.idx < b.idx
	})
	out := make([]trace.Request, len(all))
	for i, x := range all {
		out[i] = x.req
	}
	return out, trace.Validate(out)
}

// Manifest returns the trace manifest for cfg and seed (hash filled in by
// trace.WriteFiles).
func Manifest(cfg Config, seed uint64) (trace.Manifest, error) {
	params, err := json.Marshal(cfg)
	if err != nil {
		return trace.Manifest{}, err
	}
	return trace.Manifest{
		Generator: trace.Generator{Name: "loadgen", Version: Version, Params: params},
		Seed:      seed,
		DurationS: cfg.DurationS,
	}, nil
}
