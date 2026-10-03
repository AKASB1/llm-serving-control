package main

import (
	"math"
	"sort"
)

// meanCI returns the mean and the half-width of the two-sided 95 %
// Student-t confidence interval (NaN half-width for n < 2).
func meanCI(xs []float64) (mean, half float64) {
	n := len(xs)
	if n == 0 {
		return math.NaN(), math.NaN()
	}
	for _, x := range xs {
		mean += x
	}
	mean /= float64(n)
	if n < 2 {
		return mean, math.NaN()
	}
	var ss float64
	for _, x := range xs {
		ss += (x - mean) * (x - mean)
	}
	sd := math.Sqrt(ss / float64(n-1))
	return mean, tQuantile(0.975, float64(n-1)) * sd / math.Sqrt(float64(n))
}

// tQuantile returns the p-quantile of Student's t with df degrees of freedom
// (p > 0.5), by bisection on the CDF.
func tQuantile(p, df float64) float64 {
	lo, hi := 0.0, 1000.0
	for range 200 {
		mid := (lo + hi) / 2
		if tCDF(mid, df) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// tCDF is the CDF of Student's t for t ≥ 0.
func tCDF(t, df float64) float64 {
	x := df / (df + t*t)
	return 1 - 0.5*regIncBeta(df/2, 0.5, x)
}

// regIncBeta is the regularised incomplete beta function I_x(a, b)
// (continued fraction, Numerical Recipes betacf).
func regIncBeta(a, b, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	la, _ := math.Lgamma(a + b)
	lb, _ := math.Lgamma(a)
	lc, _ := math.Lgamma(b)
	front := math.Exp(la - lb - lc + a*math.Log(x) + b*math.Log(1-x))
	if x < (a+1)/(a+b+2) {
		return front * betacf(a, b, x) / a
	}
	return 1 - front*betacf(b, a, 1-x)/b
}

func betacf(a, b, x float64) float64 {
	const eps, tiny = 1e-15, 1e-300
	qab, qap, qam := a+b, a+1, a-1
	c, d := 1.0, 1-qab*x/qap
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1 / d
	h := d
	for m := 1; m <= 300; m++ {
		fm := float64(m)
		m2 := 2 * fm
		aa := fm * (b - fm) * x / ((qam + m2) * (a + m2))
		d = 1 + aa*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + aa/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		h *= d * c
		aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2))
		d = 1 + aa*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + aa/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}
	return h
}

// wtl counts wins, ties, and losses of policy against baseline over paired
// seeds (lower is better); a tie is |difference| ≤ band·|baseline|.
func wtl(policy, base []float64, band float64) (w, t, l int) {
	for i := range policy {
		d := policy[i] - base[i]
		switch {
		case math.Abs(d) <= band*math.Abs(base[i]):
			t++
		case d < 0:
			w++
		default:
			l++
		}
	}
	return
}

// kendallTau returns Kendall's tau-b between two score vectors over the same
// items (ties in either vector are handled by the tau-b correction).
func kendallTau(x, y []float64) float64 {
	var conc, disc, tx, ty float64
	for i := 0; i < len(x); i++ {
		for j := i + 1; j < len(x); j++ {
			dx, dy := x[i]-x[j], y[i]-y[j]
			switch {
			case dx == 0 && dy == 0:
			case dx == 0:
				tx++
			case dy == 0:
				ty++
			case (dx > 0) == (dy > 0):
				conc++
			default:
				disc++
			}
		}
	}
	den := math.Sqrt((conc + disc + tx) * (conc + disc + ty))
	if den == 0 {
		return math.NaN()
	}
	return (conc - disc) / den
}

// rankOrder returns item indices sorted by score ascending (ties by index).
func rankOrder(scores []float64) []int {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return scores[idx[a]] < scores[idx[b]] })
	return idx
}
