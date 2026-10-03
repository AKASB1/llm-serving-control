package executor

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/controller"
	"github.com/AKASB1/llm-serving-control/internal/registry"
	"github.com/AKASB1/llm-serving-control/internal/sim"
)

// The simulator is an Executor too.
var _ Executor = (*sim.Sim)(nil)

func waitFor(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(msg)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestProcessProvisionAndDrain builds the mock backend, provisions a replica
// process, waits until it is ready, drains it, and checks it is gone.
func TestProcessProvisionAndDrain(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and starts a process")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH: cannot build the mock backend")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "mock-backend")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command(goBin, "build", "-o", bin, "../../cmd/mock-backend")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	classes, _ := filepath.Abs("../../configs/classes.json")
	reg := registry.New()
	inflight := 0
	p := NewProcess(ProcessConfig{Binary: bin, ClassesFile: classes, LogDir: dir, ReadyTimeoutS: 20, DrainGraceS: 5}, reg,
		func(string) int { return inflight }, nil)
	defer p.StopAll()
	ctx := context.Background()
	if err := p.Apply(ctx, controller.ScaleCommand{Kind: controller.Provision, Model: "chat-8b", Class: "h100-8b"}); err != nil {
		t.Fatal(err)
	}
	reps := reg.All()
	if len(reps) != 1 || reps[0].State != registry.Provisioning {
		t.Fatalf("after provision: %+v", reps)
	}
	id := reps[0].ID
	waitFor(t, 20*time.Second, func() bool { r, _ := reg.Get(id); return r.State == registry.Ready }, "replica never became ready")
	if err := p.Apply(ctx, controller.ScaleCommand{Kind: controller.Drain, ReplicaID: id}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 10*time.Second, func() bool { _, ok := reg.Get(id); return !ok }, "drained replica not removed")
	if _, err := os.Stat(filepath.Join(dir, id+".log")); err != nil {
		t.Fatalf("no process log: %v", err)
	}
	if err := p.Apply(ctx, controller.ScaleCommand{Kind: controller.Drain, ReplicaID: "unknown"}); err == nil {
		t.Fatal("drain of an unknown replica accepted")
	}
}
