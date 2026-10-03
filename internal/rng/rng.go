// Package rng derives independent, reproducible random streams. Every
// component gets its own math/rand/v2 PCG stream from (run seed, component
// name), so adding or removing a component never shifts another component's
// numbers (common random numbers across policies).
package rng

import (
	"hash/fnv"
	"math/rand/v2"
)

// Stream returns a new deterministic generator for the named component.
func Stream(seed uint64, component string) *rand.Rand {
	h := fnv.New64a()
	_, _ = h.Write([]byte(component))
	c := h.Sum64()
	s1 := splitmix64(seed ^ c)
	s2 := splitmix64(s1 ^ 0x9e3779b97f4a7c15 ^ c)
	return rand.New(rand.NewPCG(s1, s2))
}

// splitmix64 is the SplitMix64 finaliser; it decorrelates nearby inputs.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
