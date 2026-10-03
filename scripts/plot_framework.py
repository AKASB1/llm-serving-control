"""Draw the framework figure (docs/figures/framework.png).

Usage: python scripts/plot_framework.py [--out docs/figures]

The figure shows the control loop and which code the simulator and the live
service share. It is a diagram, not a result.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SHARED_FILL, SHARED_EDGE = "#e3eefb", "#2a78d6"   # shared policy/runtime code
ENV_FILL, ENV_EDGE = "#fdebe3", "#eb6834"         # replaceable environment
DATA_FILL, DATA_EDGE = "#eeeeea", "#8a8984"       # data and evaluation


def box(ax, x, y, w, h, title, body, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + 0.12, y + h - 0.47, body, ha="left", va="top", fontsize=7.6, color=INK2, linespacing=1.35)


def arrow(ax, a, b, label="", rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=7.4, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.1, 6.2)
    ax.axis("off")

    box(ax, 0.2, 3.9, 2.75, 1.9, "Traffic", "trace schema v1 (CSV + manifest)\nloadgen: Poisson, MMPP, gamma,\ndiurnal, piecewise; length mixes\nreplay of committed traces", DATA_FILL, DATA_EDGE)
    box(ax, 3.2, 3.6, 3.4, 2.2, "Dispatcher (shared)", "admission: none · queue_cap ·\n  predicted_ttft_shed\nrouter: 9 policies (incl. oracle)\nrouter queue, timeouts, retries\n  before the first token\nhealth: ejection, backoff, probes", SHARED_FILL, SHARED_EDGE)
    box(ax, 7.2, 3.6, 2.9, 2.2, "Replicas (environment)", "simulator: engine with continuous\n  batching, KV cache, preemption,\n  lifecycle, crashes\nlive (planned): mock backend /\n  vLLM, SGLang adapters", ENV_FILL, ENV_EDGE)
    box(ax, 10.6, 3.9, 2.2, 1.9, "Executor", "sim cluster: provision,\nwarm-up, drain, failure\nlive (planned): process /\nKubernetes", ENV_FILL, ENV_EDGE)
    box(ax, 7.2, 0.6, 2.9, 2.2, "Metrics (shared)", "snapshots every scrape interval\nrouter-local in-flight (fresh)\nwindowed signals: arrival,\n  service rate, utilization,\n  queue depth, KV usage", SHARED_FILL, SHARED_EDGE)
    box(ax, 3.2, 0.6, 3.4, 2.2, "Control loop (shared)", "SLO controller: P95 error per\n  model and replica\nscalers: static, threshold_cooldown,\n  target_tracking, slo_feedback,\n  predictive\nallocator under a GPU budget", SHARED_FILL, SHARED_EDGE)
    box(ax, 0.2, 0.6, 2.75, 2.6, "Benchmark", "tuning seeds → frozen configs\nevaluation seeds → runs\nmean ± 95% t-CI, paired ΔJ,\nW/T/L, sensitivity\nfigures + table (simulated)", DATA_FILL, DATA_EDGE)

    arrow(ax, (2.95, 4.85), (3.2, 4.85))
    arrow(ax, (6.6, 4.85), (7.2, 4.85), "dispatch")
    arrow(ax, (10.6, 4.85), (10.1, 4.85), "lifecycle")
    arrow(ax, (8.65, 3.6), (8.65, 2.8), "  snapshots, completions", dy=0.0)
    arrow(ax, (7.2, 1.7), (6.6, 1.7), "signals")
    arrow(ax, (4.9, 2.8), (4.9, 3.6), "SLO error → routing weights", dy=0.0)
    # desired → granted replicas: routed below the boxes to the executor
    ax.plot([4.9, 4.9, 11.7], [0.6, 0.35, 0.35], color=INK2, lw=1.2, solid_capstyle="round")
    arrow(ax, (11.7, 0.35), (11.7, 3.9))
    ax.text(11.82, 2.1, "desired →\ngranted\nreplicas", fontsize=7.4, color=INK2, va="center")
    arrow(ax, (3.2, 1.7), (2.95, 1.7))

    ax.text(0.2, 6.05, "llm-serving-control: one policy code path for the simulator and the live service",
            fontsize=11, fontweight="bold", color=INK, va="top")
    ax.text(12.8, 0.02, "blue: shared policy and runtime code · orange: replaceable environment · gray: data and evaluation",
            fontsize=7.6, color=INK2, ha="right")
    path = os.path.join(args.out, "framework.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
