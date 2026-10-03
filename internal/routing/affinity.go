package routing

import (
	"hash/fnv"
	"math"
	"sort"
	"strconv"
	"strings"
)

// PrefixAffinityParams configure `prefix_affinity`.
type PrefixAffinityParams struct {
	// VirtualNodes per replica on the hash ring.
	VirtualNodes int `json:"virtual_nodes"`
	// LoadFactor c bounds every replica at ceil(c × average in-flight)
	// (consistent hashing with bounded loads).
	LoadFactor float64 `json:"load_factor"`
}

// DefaultPrefixAffinity returns the defaults.
func DefaultPrefixAffinity() PrefixAffinityParams {
	return PrefixAffinityParams{VirtualNodes: 100, LoadFactor: 1.25}
}

type ringPoint struct {
	hash uint64
	id   string
}

// PrefixAffinityPolicy sends requests of the same prefix group to the same
// replica, so that replica's prefix cache keeps the group: the group is hashed
// onto a ring of replica virtual nodes and walks clockwise to the first
// replica whose in-flight count stays within ceil(LoadFactor × average) after
// the dispatch. Requests without a prefix group go to the least-outstanding
// replica.
type PrefixAffinityPolicy struct {
	p       PrefixAffinityParams
	ringKey string
	ring    []ringPoint
}

// NewPrefixAffinity returns the `prefix_affinity` policy.
func NewPrefixAffinity(p PrefixAffinityParams) *PrefixAffinityPolicy {
	if p.VirtualNodes < 1 {
		p.VirtualNodes = DefaultPrefixAffinity().VirtualNodes
	}
	if p.LoadFactor < 1 {
		p.LoadFactor = 1
	}
	return &PrefixAffinityPolicy{p: p}
}

// Name implements Router.
func (p *PrefixAffinityPolicy) Name() string { return "prefix_affinity" }

func hash64(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	x := h.Sum64()
	// SplitMix64 finaliser: FNV alone clusters similar keys on the ring.
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// ringFor builds (and caches) the ring for the replicas in the view.
func (p *PrefixAffinityPolicy) ringFor(v View) []ringPoint {
	ids := make([]string, len(v.Replicas))
	for i, r := range v.Replicas {
		ids[i] = r.ID
	}
	key := strings.Join(ids, "\x00")
	if key == p.ringKey && p.ring != nil {
		return p.ring
	}
	ring := make([]ringPoint, 0, len(ids)*p.p.VirtualNodes)
	for _, id := range ids {
		for k := 0; k < p.p.VirtualNodes; k++ {
			ring = append(ring, ringPoint{hash: hash64(id + "#" + strconv.Itoa(k)), id: id})
		}
	}
	sort.Slice(ring, func(i, j int) bool {
		if ring[i].hash != ring[j].hash {
			return ring[i].hash < ring[j].hash
		}
		return ring[i].id < ring[j].id
	})
	p.ringKey, p.ring = key, ring
	return ring
}

// Route implements Router.
func (p *PrefixAffinityPolicy) Route(req Request, v View) Decision {
	if req.PrefixGroup == "" {
		return NewLeastOutstanding().Route(req, v)
	}
	total := 1
	inflight := make(map[string]int, len(v.Replicas))
	for _, r := range v.Replicas {
		total += r.InFlight
		inflight[r.ID] = r.InFlight
	}
	bound := int(math.Ceil(p.p.LoadFactor * float64(total) / float64(len(v.Replicas))))
	ring := p.ringFor(v)
	h := hash64(req.PrefixGroup)
	start := sort.Search(len(ring), func(i int) bool { return ring[i].hash >= h })
	tried := map[string]bool{}
	for k := 0; k < len(ring) && len(tried) < len(v.Replicas); k++ {
		pt := ring[(start+k)%len(ring)]
		if tried[pt.id] {
			continue
		}
		tried[pt.id] = true
		if inflight[pt.id]+1 <= bound {
			return Decision{Action: Dispatch, ReplicaID: pt.id}
		}
	}
	return NewLeastOutstanding().Route(req, v) // unreachable: some replica is at or below the average
}
