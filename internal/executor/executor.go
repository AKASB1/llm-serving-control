// Package executor carries out the control loop's scale commands. The
// simulator implements Executor directly (sim.Sim.Apply); Process starts and
// stops mock-backend processes on this machine.
package executor

import (
	"context"

	"github.com/AKASB1/llm-serving-control/internal/controller"
)

// Executor applies one scale command. Implementations update the replica
// registry as replicas move through their lifecycle.
type Executor interface {
	Apply(ctx context.Context, cmd controller.ScaleCommand) error
}
