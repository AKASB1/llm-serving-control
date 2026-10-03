package controller

// ScaleKind is a lifecycle action requested by the control loop.
type ScaleKind int

// Scale actions.
const (
	Provision ScaleKind = iota // start a new replica of Model on Class
	Drain                      // stop sending traffic to ReplicaID and terminate it when idle
)

func (k ScaleKind) String() string {
	if k == Provision {
		return "provision"
	}
	return "drain"
}

// ScaleCommand is one action for the executor (simulated cluster or process
// executor).
type ScaleCommand struct {
	Kind      ScaleKind
	Model     string
	Class     string
	ReplicaID string
	Reason    string
}
