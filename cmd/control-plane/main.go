// Command control-plane runs the live control plane: an OpenAI-compatible
// HTTP proxy in front of registered replicas, using the same dispatcher,
// policies, and control loop as the simulator.
//
//	go run ./cmd/control-plane -config configs/live.json         # serve until Ctrl-C or POST /admin/shutdown
//	go run ./cmd/control-plane -config configs/example.json -once # route one request in-process and print the endpoint
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/adapters"
	"github.com/AKASB1/llm-serving-control/internal/autoscaling"
	"github.com/AKASB1/llm-serving-control/internal/clock"
	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/executor"
	"github.com/AKASB1/llm-serving-control/internal/proxy"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/routing"
	"github.com/AKASB1/llm-serving-control/internal/sim"
	"github.com/AKASB1/llm-serving-control/internal/slo"
	"github.com/AKASB1/llm-serving-control/internal/store"
)

type replicaConf struct {
	ID       string `json:"id"`
	Model    string `json:"model"`
	Class    string `json:"class"`
	Endpoint string `json:"endpoint"`
}

type config struct {
	Listen      string `json:"listen"`
	Model       string `json:"model"` // default model for replicas without one
	ClassesFile string `json:"classes_file"`
	SLO         struct {
		Default string                    `json:"default"`
		Classes map[string]slo.TargetJSON `json:"classes"`
	} `json:"slo"`
	Routing       controller.PolicySpec      `json:"routing"`
	Admission     controller.PolicySpec      `json:"admission"`
	Dispatch      *controller.DispatchConfig `json:"dispatch"`
	Health        *controller.HealthConfig   `json:"health"`
	MetricsFormat string                     `json:"metrics_format"`
	Intervals     struct {
		TickS         float64 `json:"tick_s"`
		ProbeS        float64 `json:"probe_s"`
		ProbeTimeoutS float64 `json:"probe_timeout_s"`
		ScrapeS       float64 `json:"scrape_s"`
		ControlS      float64 `json:"control_s"`
	} `json:"intervals"`
	Replicas []replicaConf `json:"replicas"`
	Control  *struct {
		WindowS    float64                  `json:"window_s"`
		BudgetGPUs int                      `json:"budget_gpus"`
		Scaler     controller.PolicySpec    `json:"scaler"`
		Models     []controller.ModelConfig `json:"models"`
	} `json:"control"`
	Executor *executor.ProcessConfig `json:"executor"`
	Seed     uint64                  `json:"seed"`
	// StorePath persists replicas registered through the admin API (JSON).
	StorePath string `json:"store_path"`
}

func main() {
	path := flag.String("config", "configs/example.json", "configuration file")
	once := flag.Bool("once", false, "route one request in-process, print the chosen endpoint, and exit")
	flag.Parse()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
	if err := run(*path, *once); err != nil {
		slog.Error("control-plane failed", "err", err)
		os.Exit(1)
	}
}

func secs(s, def float64) time.Duration {
	if s <= 0 {
		s = def
	}
	return time.Duration(s * float64(time.Second))
}

func run(path string, once bool) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if cfg.SLO.Default == "" {
		cfg.SLO.Default = "interactive"
		cfg.SLO.Classes = map[string]slo.TargetJSON{"interactive": {TTFTs: 2, TPOTs: 0.1}}
	}
	targets, err := slo.NewTargets(cfg.SLO.Default, cfg.SLO.Classes)
	if err != nil {
		return err
	}
	var infos map[string]registry.ClassInfo
	if cfg.ClassesFile != "" {
		cs, err := sim.LoadClasses(cfg.ClassesFile)
		if err != nil {
			return err
		}
		infos = sim.ClassInfos(cs)
	}
	reg := registry.New()
	for _, r := range cfg.Replicas {
		if r.Model == "" {
			r.Model = cfg.Model
		}
		if err := reg.Add(registry.Replica{ID: r.ID, Model: r.Model, Endpoint: r.Endpoint, Class: r.Class, GPUs: 1, State: registry.Ready, Healthy: true}); err != nil {
			return fmt.Errorf("replica %q: %w", r.ID, err)
		}
	}
	var st store.ReplicaStore
	if cfg.StorePath != "" && !once {
		fs, err := store.OpenFile(cfg.StorePath)
		if err != nil {
			return err
		}
		for _, r := range fs.All() {
			r.State, r.Healthy = registry.Ready, true
			if err := reg.Add(r); err != nil && !errors.Is(err, registry.ErrDuplicate) {
				return fmt.Errorf("stored replica %q: %w", r.ID, err)
			}
		}
		st = fs
	}
	router, err := controller.NewRouter(cfg.Routing, cfg.Seed)
	if err != nil {
		return err
	}
	adm, err := controller.NewAdmission(cfg.Admission, targets)
	if err != nil {
		return err
	}
	dc, hc := controller.DefaultDispatch(), controller.DefaultHealth()
	if cfg.Dispatch != nil {
		dc = *cfg.Dispatch
	}
	if cfg.Health != nil {
		hc = *cfg.Health
	}
	disp := controller.NewDispatcher(dc, reg, infos, router, adm, controller.NewHealth(hc))
	if once {
		model := cfg.Model
		if model == "" && len(cfg.Replicas) > 0 {
			model = cfg.Replicas[0].Model
		}
		for _, c := range disp.Arrive(0, routing.Request{ID: "once", Model: model, PromptTokens: 16}) {
			if c.Kind != controller.Send {
				return fmt.Errorf("request not dispatched: %s", c.Reason)
			}
			rep, _ := reg.Get(c.ReplicaID)
			fmt.Println(rep.Endpoint)
		}
		return nil
	}
	return serve(cfg, reg, disp, infos, targets, st)
}

func serve(cfg config, reg *registry.Registry, disp *controller.Dispatcher, infos map[string]registry.ClassInfo, targets slo.Targets, st store.ReplicaStore) error {
	clk := clock.NewReal()
	mapping, err := adapters.ByName(cfg.MetricsFormat)
	if err != nil {
		return err
	}
	client := &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 60 * time.Second, MaxIdleConnsPerHost: 64}}
	opt := proxy.Options{Store: st, Client: client, Scraper: &adapters.Scraper{Client: &http.Client{Timeout: 2 * time.Second}, Mapping: mapping, Now: clk.Now}}
	var srv *proxy.Server
	var proc *executor.Process
	if cfg.Control != nil {
		scalers := map[string]autoscaling.Scaler{}
		warm := map[string]time.Duration{}
		for _, m := range cfg.Control.Models {
			s, err := controller.NewScaler(cfg.Control.Scaler)
			if err != nil {
				return err
			}
			scalers[m.Model] = s
			warm[m.Class] = time.Second
		}
		loop, err := controller.NewLoop(controller.LoopConfig{Models: cfg.Control.Models, WindowS: cfg.Control.WindowS, BudgetGPUs: cfg.Control.BudgetGPUs},
			reg, disp, infos, warm, targets, scalers, nil)
		if err != nil {
			return err
		}
		opt.Loop = loop
		if cfg.Executor != nil {
			proc = executor.NewProcess(*cfg.Executor, reg, func(id string) int { return srv.InFlight(id) }, slog.Default())
			opt.Executor = proc
		}
	}
	pc := proxy.Config{
		TickInterval: secs(cfg.Intervals.TickS, 0.2), ProbeInterval: secs(cfg.Intervals.ProbeS, 2),
		ProbeTimeout: secs(cfg.Intervals.ProbeTimeoutS, 1), ScrapeInterval: secs(cfg.Intervals.ScrapeS, 1),
		ControlInterval: secs(cfg.Intervals.ControlS, 5), RetryAfter: time.Second,
	}
	srv, err = proxy.New(pc, clk, reg, disp, targets, opt)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	mux := http.NewServeMux()
	mux.Handle("/", srv.Handler())
	mux.HandleFunc("POST /admin/shutdown", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		cancel()
	})
	listen := cfg.Listen
	if listen == "" {
		listen = "127.0.0.1:18080"
	}
	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	hs := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	done := make(chan struct{})
	go func() { srv.Run(ctx); close(done) }()
	go func() {
		<-ctx.Done()
		sctx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		_ = hs.Shutdown(sctx)
	}()
	slog.Info("control plane listening", "addr", ln.Addr().String(), "replicas", len(reg.All()), "routing", cfg.Routing.Name, "control", cfg.Control != nil)
	err = hs.Serve(ln)
	<-done
	if proc != nil {
		proc.StopAll()
	}
	slog.Info("control plane stopped")
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
