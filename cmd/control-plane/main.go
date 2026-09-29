package main

import (
	"fmt"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
)

func main() {
	r := registry.New()
	if err := r.Add(registry.Replica{ID: "local", Model: "demo", Endpoint: "http://localhost:8000", Healthy: true}); err != nil { panic(err) }
	selected, _ := routing.LeastOutstanding(r.ForModel("demo"))
	fmt.Println(selected.Endpoint)
}
