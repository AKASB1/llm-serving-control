package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/AKASB1/llm-serving-control/internal/admission"
	"github.com/AKASB1/llm-serving-control/internal/autoscaling"
	"github.com/AKASB1/llm-serving-control/internal/capacity"
	"github.com/AKASB1/llm-serving-control/internal/rng"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/slo"
)

// PolicySpec is the JSON form of a policy: {"name": "...", "params": {...}}.
type PolicySpec struct {
	Name   string          `json:"name"`
	Params json.RawMessage `json:"params,omitempty"`
}

// decode fills dst (pre-set to defaults) from raw, rejecting unknown fields.
func decode(raw json.RawMessage, dst any) error {
	if len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null" {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

// RouterNames lists the routing policies the factory knows, sorted.
func RouterNames() []string {
	names := []string{"random", "round_robin", "least_outstanding", "power_of_two", "latency_aware",
		"queue_aware", "queue_aware_corrected", "capacity_weighted", "oracle_jsq", "prefix_affinity"}
	sort.Strings(names)
	return names
}

// NewRouter builds a routing policy. Randomised policies draw from the
// stream "router/<name>" of seed.
func NewRouter(spec PolicySpec, seed uint64) (routing.Router, error) {
	r := rng.Stream(seed, "router/"+spec.Name)
	switch spec.Name {
	case "random":
		return routing.NewRandom(r), decode(spec.Params, &struct{}{})
	case "round_robin":
		return routing.NewRoundRobin(), decode(spec.Params, &struct{}{})
	case "least_outstanding":
		return routing.NewLeastOutstanding(), decode(spec.Params, &struct{}{})
	case "power_of_two":
		return routing.NewPowerOfTwo(r), decode(spec.Params, &struct{}{})
	case "latency_aware":
		p := routing.DefaultLatencyAware()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("latency_aware: %w", err)
		}
		return routing.NewLatencyAware(p, r), nil
	case "queue_aware", "queue_aware_corrected":
		p := routing.DefaultQueueAware()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("%s: %w", spec.Name, err)
		}
		return routing.NewQueueAware(p, spec.Name == "queue_aware_corrected"), nil
	case "capacity_weighted":
		p := routing.DefaultCapacityWeighted()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("capacity_weighted: %w", err)
		}
		return routing.NewCapacityWeighted(p), nil
	case "oracle_jsq":
		return routing.NewOracleJSQ(), decode(spec.Params, &struct{}{})
	case "prefix_affinity":
		p := routing.DefaultPrefixAffinity()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("prefix_affinity: %w", err)
		}
		return routing.NewPrefixAffinity(p), nil
	}
	return nil, fmt.Errorf("unknown routing policy %q", spec.Name)
}

// ScalerNames lists the scaling policies the factory knows, sorted.
func ScalerNames() []string {
	return []string{"predictive", "slo_feedback", "static", "target_tracking", "threshold_cooldown"}
}

// NewScaler builds a scaling policy.
func NewScaler(spec PolicySpec) (autoscaling.Scaler, error) {
	switch spec.Name {
	case "static":
		p := autoscaling.StaticParams{Replicas: 1}
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("static: %w", err)
		}
		return autoscaling.NewStatic(p), nil
	case "threshold_cooldown":
		p := autoscaling.DefaultThreshold()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("threshold_cooldown: %w", err)
		}
		return autoscaling.NewThresholdCooldown(p), nil
	case "target_tracking":
		p := autoscaling.DefaultTargetTracking()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("target_tracking: %w", err)
		}
		return autoscaling.NewTargetTracking(p), nil
	case "slo_feedback":
		p := autoscaling.DefaultSLOFeedback()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("slo_feedback: %w", err)
		}
		return autoscaling.NewSLOFeedback(p), nil
	case "predictive":
		p := autoscaling.DefaultPredictive()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("predictive: %w", err)
		}
		return autoscaling.NewPredictive(p), nil
	}
	return nil, fmt.Errorf("unknown scaling policy %q", spec.Name)
}

// AllocatorNames lists the capacity allocators the factory knows, sorted.
func AllocatorNames() []string {
	return []string{"marginal_gain", "proportional_demand", "static_partition"}
}

// NewAllocator builds a capacity allocator.
func NewAllocator(spec PolicySpec) (capacity.Allocator, error) {
	switch spec.Name {
	case "static_partition":
		var p capacity.StaticPartitionParams
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("static_partition: %w", err)
		}
		return capacity.NewStaticPartition(p), nil
	case "proportional_demand":
		return capacity.NewProportionalDemand(), decode(spec.Params, &struct{}{})
	case "marginal_gain":
		p := capacity.DefaultMarginalGain()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("marginal_gain: %w", err)
		}
		return capacity.NewMarginalGain(p), nil
	}
	return nil, fmt.Errorf("unknown allocator %q", spec.Name)
}

// AdmissionNames lists the admission policies the factory knows, sorted.
func AdmissionNames() []string { return []string{"none", "predicted_ttft_shed", "queue_cap"} }

// NewAdmission builds an admission policy.
func NewAdmission(spec PolicySpec, targets slo.Targets) (admission.Policy, error) {
	switch spec.Name {
	case "", "none":
		return admission.None{}, decode(spec.Params, &struct{}{})
	case "queue_cap":
		p := admission.DefaultQueueCap()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("queue_cap: %w", err)
		}
		return admission.NewQueueCap(p), nil
	case "predicted_ttft_shed":
		p := admission.DefaultPredictedTTFT()
		if err := decode(spec.Params, &p); err != nil {
			return nil, fmt.Errorf("predicted_ttft_shed: %w", err)
		}
		return admission.NewPredictedTTFT(p, targets), nil
	}
	return nil, fmt.Errorf("unknown admission policy %q", spec.Name)
}
