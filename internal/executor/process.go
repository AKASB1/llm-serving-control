package executor

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/registry"
)

// ProcessConfig configures the process executor.
type ProcessConfig struct {
	// Binary is the mock-backend executable.
	Binary      string  `json:"binary"`
	ClassesFile string  `json:"classes_file"`
	Speedup     float64 `json:"speedup"`
	Host        string  `json:"host"`
	// PortBase > 0 picks the first free port at or above it; 0 lets the OS
	// choose.
	PortBase int    `json:"port_base"`
	LogDir   string `json:"log_dir"`
	// ReadyTimeoutS bounds start-up (until /health answers);
	// DrainGraceS bounds a drain before the process is killed.
	ReadyTimeoutS float64 `json:"ready_timeout_s"`
	DrainGraceS   float64 `json:"drain_grace_s"`
}

// Process starts mock-backend processes on free local ports for Provision
// commands and stops them for Drain commands, moving each replica through
// provisioning → ready → draining → removed in the registry.
type Process struct {
	cfg      ProcessConfig
	reg      *registry.Registry
	inflight func(id string) int
	log      *slog.Logger
	client   *http.Client
	mu       sync.Mutex
	procs    map[string]*exec.Cmd
	logs     map[string]*os.File
	count    map[string]int
	wg       sync.WaitGroup
}

// NewProcess returns a process executor. inflight reports the router-local
// in-flight count of a replica (the caller makes it safe to call).
func NewProcess(cfg ProcessConfig, reg *registry.Registry, inflight func(id string) int, log *slog.Logger) *Process {
	if runtime.GOOS == "windows" && filepath.Ext(cfg.Binary) == "" {
		cfg.Binary += ".exe" // configurations name the binary without the extension
	}
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.Speedup <= 0 {
		cfg.Speedup = 1
	}
	if cfg.ReadyTimeoutS <= 0 {
		cfg.ReadyTimeoutS = 30
	}
	if cfg.DrainGraceS <= 0 {
		cfg.DrainGraceS = 30
	}
	if log == nil {
		log = slog.Default()
	}
	return &Process{cfg: cfg, reg: reg, inflight: inflight, log: log, client: &http.Client{Timeout: time.Second},
		procs: map[string]*exec.Cmd{}, logs: map[string]*os.File{}, count: map[string]int{}}
}

// Apply implements Executor. Start-up and drain complete asynchronously.
func (p *Process) Apply(ctx context.Context, cmd controller.ScaleCommand) error {
	switch cmd.Kind {
	case controller.Provision:
		return p.provision(ctx, cmd.Model, cmd.Class)
	case controller.Drain:
		return p.drain(ctx, cmd.ReplicaID)
	}
	return fmt.Errorf("executor: unknown command %v", cmd.Kind)
}

func freePort(host string, base int) (int, error) {
	if base > 0 {
		for port := base; port < base+1000 && port < 65536; port++ {
			if ln, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port))); err == nil {
				ln.Close()
				return port, nil
			}
		}
		return 0, fmt.Errorf("executor: no free port in [%d, %d)", base, base+1000)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

func (p *Process) provision(ctx context.Context, model, class string) error {
	p.mu.Lock() // serialise port selection between concurrent provisions
	port, err := freePort(p.cfg.Host, p.cfg.PortBase+len(p.procs))
	p.mu.Unlock()
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.count[class]++
	id := fmt.Sprintf("%s-p%02d", class, p.count[class])
	p.mu.Unlock()
	addr := net.JoinHostPort(p.cfg.Host, strconv.Itoa(port))
	c := exec.Command(p.cfg.Binary, "-listen", addr, "-model", model, "-class", class,
		"-classes", p.cfg.ClassesFile, "-speedup", strconv.FormatFloat(p.cfg.Speedup, 'g', -1, 64))
	var logFile *os.File
	if p.cfg.LogDir != "" {
		if err := os.MkdirAll(p.cfg.LogDir, 0o755); err != nil {
			return err
		}
		f, err := os.Create(filepath.Join(p.cfg.LogDir, id+".log"))
		if err != nil {
			return err
		}
		c.Stdout, c.Stderr = f, f
		logFile = f
	}
	if err := c.Start(); err != nil {
		if logFile != nil {
			logFile.Close()
		}
		return err
	}
	endpoint := "http://" + addr
	if err := p.reg.Add(registry.Replica{ID: id, Model: model, Endpoint: endpoint, Class: class, GPUs: 1, State: registry.Provisioning, Healthy: true}); err != nil {
		_ = c.Process.Kill()
		_ = c.Wait()
		if logFile != nil {
			logFile.Close()
		}
		return err
	}
	p.mu.Lock()
	p.procs[id] = c
	if logFile != nil {
		p.logs[id] = logFile
	}
	p.mu.Unlock()
	p.log.Info("replica process started", "id", id, "pid", c.Process.Pid, "endpoint", endpoint)
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		deadline := time.Now().Add(time.Duration(p.cfg.ReadyTimeoutS * float64(time.Second)))
		for time.Now().Before(deadline) && ctx.Err() == nil {
			if resp, err := p.client.Get(endpoint + "/health"); err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					_ = p.reg.Update(id, func(r *registry.Replica) { r.State = registry.Ready })
					p.log.Info("replica ready", "id", id)
					return
				}
			}
			time.Sleep(50 * time.Millisecond)
		}
		p.log.Error("replica did not become ready", "id", id)
		p.stop(id)
	}()
	return nil
}

func (p *Process) drain(ctx context.Context, id string) error {
	rep, ok := p.reg.Get(id)
	if !ok {
		return registry.ErrNotFound
	}
	p.mu.Lock()
	_, managed := p.procs[id]
	p.mu.Unlock()
	if !managed {
		return fmt.Errorf("executor: replica %s was not started by this executor", id)
	}
	if rep.State != registry.Ready {
		p.stop(id)
		return nil
	}
	_ = p.reg.Update(id, func(r *registry.Replica) { r.State = registry.Draining })
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		deadline := time.Now().Add(time.Duration(p.cfg.DrainGraceS * float64(time.Second)))
		for time.Now().Before(deadline) && ctx.Err() == nil && p.inflight(id) > 0 {
			time.Sleep(50 * time.Millisecond)
		}
		p.stop(id)
	}()
	return nil
}

// stop kills a replica's process and removes it from the registry.
func (p *Process) stop(id string) {
	p.mu.Lock()
	c, f := p.procs[id], p.logs[id]
	delete(p.procs, id)
	delete(p.logs, id)
	p.mu.Unlock()
	if c != nil {
		_ = c.Process.Kill()
		_ = c.Wait()
	}
	if f != nil {
		f.Close()
	}
	_ = p.reg.Remove(id)
	p.log.Info("replica process stopped", "id", id)
}

// StopAll stops every process this executor started and waits for its
// background work to end.
func (p *Process) StopAll() {
	p.mu.Lock()
	ids := make([]string, 0, len(p.procs))
	for id := range p.procs {
		ids = append(ids, id)
	}
	p.mu.Unlock()
	for _, id := range ids {
		p.stop(id)
	}
	p.wg.Wait()
}
