package slo

func Violated(latencyMs, targetMs float64) bool {
	return targetMs > 0 && latencyMs > targetMs
}
