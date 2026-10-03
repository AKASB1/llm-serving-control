package registry

// ClassInfo is the control plane's static prior about a replica class: the
// hardware a replica of that class runs on and its nominal speed. Policies use
// it as a prior before measurements exist; the simulator derives its engine
// parameters from the same class table.
type ClassInfo struct {
	Name string `json:"name"`
	// Model is the model this class serves (a class is model + hardware).
	Model string `json:"model"`
	GPUs  int    `json:"gpus"`
	// PrefillTokensPerSec is the nominal prompt-processing rate.
	PrefillTokensPerSec float64 `json:"prefill_tokens_per_s"`
	// DecodeTokensPerSec is the nominal aggregate decode rate at a full batch.
	DecodeTokensPerSec float64 `json:"decode_tokens_per_s"`
	MaxNumSeqs         int     `json:"max_num_seqs"`
	KVCapacityTokens   int     `json:"kv_capacity_tokens"`
}
