package metrics

type Sample struct {
	ReplicaID string
	LatencyMs float64
	QueueDepth int
}

func MeanLatency(samples []Sample) float64 {
	if len(samples) == 0 { return 0 }
	total := 0.0
	for _, sample := range samples { total += sample.LatencyMs }
	return total / float64(len(samples))
}
