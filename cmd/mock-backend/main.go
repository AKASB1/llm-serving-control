// Command mock-backend serves an OpenAI-compatible API backed by the
// simulator's replica engine running in real time (placeholder tokens,
// simulated timing). It exposes /health and vLLM-named /metrics.
//
//	go run ./cmd/mock-backend -listen 127.0.0.1:18100 -model chat-8b -class h100-8b
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/AKASB1/llm-serving-control/internal/sim"
	"github.com/AKASB1/llm-serving-control/internal/sim/live"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:18100", "listen address")
	model := flag.String("model", "chat-8b", "model name served")
	class := flag.String("class", "h100-8b", "replica class (configs/classes.json)")
	classes := flag.String("classes", "configs/classes.json", "class definitions")
	speedup := flag.Float64("speedup", 1, "virtual seconds per real second")
	flag.Parse()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
	if err := run(*listen, *model, *class, *classes, *speedup); err != nil {
		slog.Error("mock-backend failed", "err", err)
		os.Exit(1)
	}
}

func run(listen, model, class, classesPath string, speedup float64) error {
	cs, err := sim.LoadClasses(classesPath)
	if err != nil {
		return err
	}
	c, ok := cs[class]
	if !ok {
		return errors.New("unknown class " + class)
	}
	b, err := live.New(live.Options{Model: model, Params: c.Params, Speedup: speedup})
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go b.Run(ctx)
	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: b.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()
	slog.Info("mock backend listening", "addr", ln.Addr().String(), "model", model, "class", class, "speedup", speedup)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
