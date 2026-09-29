package autoscaling

func DesiredReplicas(queueDepth, targetPerReplica, minimum int) int {
	if targetPerReplica <= 0 || minimum < 1 { return minimum }
	desired := (queueDepth + targetPerReplica - 1) / targetPerReplica
	if desired < minimum { return minimum }
	return desired
}
