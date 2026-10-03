// Package admission decides, before routing, whether a request is admitted,
// held in the router queue, or rejected.
package admission

import "github.com/AKASB1/llm-serving-control/internal/routing"

// Decision is an admission verdict; Action uses the routing constants
// (Dispatch = admit).
type Decision struct {
	Action routing.Action
	Reason string
}

// Policy decides admission for one request from the same view as the router.
// A rejected request is a violation (it stays in the SLO denominator).
type Policy interface {
	Name() string
	Admit(req routing.Request, v routing.View) Decision
}
