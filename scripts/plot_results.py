"""Draw the result figures and the README table from committed benchmark results.

Usage:
    python scripts/plot_results.py [--results benchmarks/results] [--out docs/figures]

Reads aggregates.csv, paired.csv, sensitivity.csv, and manifest.json written by
`go run ./cmd/benchmark`; writes PNG figures (each below 300 KB) and
results_table.md. Every number comes from those files; nothing is drawn by
hand. All results are simulated with assumed parameters.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Reference categorical palette (light): slot 1 blue for policies, slot 2
# orange for the named baseline; oracles are hollow gray. Text uses ink tokens.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def load(results):
    agg = defaultdict(dict)  # (scenario, variant, policy) -> metric -> (mean, ci, n)
    meta = {}
    with open(os.path.join(results, "aggregates.csv"), newline="") as f:
        for r in csv.DictReader(f):
            key = (r["scenario"], r["variant"], r["policy"])
            agg[key][r["metric"]] = (float(r["mean"]), float(r["ci95"]), int(r["n"]))
            meta[key] = (r["family"], r["oracle"] == "true")
    paired = {}
    with open(os.path.join(results, "paired.csv"), newline="") as f:
        for r in csv.DictReader(f):
            paired[(r["scenario"], r["variant"], r["policy"])] = r
    sens = []
    path = os.path.join(results, "sensitivity.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            sens = list(csv.DictReader(f))
    with open(os.path.join(results, "manifest.json")) as f:
        manifest = json.load(f)
    return agg, meta, paired, sens, manifest


def ordered(agg, meta, scenario, variant="base"):
    """Policies of a scenario in the order they appear in aggregates.csv."""
    return [k[2] for k in agg if k[0] == scenario and k[1] == variant]


def scenarios_of(meta, family):
    out = []
    for (sc, v, _), (fam, _) in meta.items():
        if fam == family and v == "base" and sc not in out:
            out.append(sc)
    return out


def finite(x):
    return x is not None and not math.isnan(x) and not math.isinf(x)


def step125(x, down):
    """The nearest value of the 1-2-5 series at or below (down) or at or above x."""
    e = math.floor(math.log10(x))
    series = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1, 2, 5)]
    return max(v for v in series if v <= x) if down else min(v for v in series if v >= x)


def dot_panel(ax, agg, meta, scenario, metric, baseline, log=False, pct=False, labels=True):
    pols = ordered(agg, meta, scenario)
    ys = list(range(len(pols)))[::-1]
    vals = [agg[(scenario, "base", p)][metric][0] for p in pols]
    vals = [v for v in vals if finite(v) and v > 0]
    if log and (not vals or max(vals) / min(vals) < 4):
        log = False  # narrow range: a linear axis reads better
    for y, p in zip(ys, pols):
        m, ci, _ = agg[(scenario, "base", p)][metric]
        if not finite(m):
            continue
        scale = 100 if pct else 1
        m, ci = m * scale, (ci * scale if finite(ci) else 0)
        oracle = meta[(scenario, "base", p)][1]
        color = GRAY if oracle else (ORANGE if p == baseline else BLUE)
        lo, hi = (max(m - ci, m * 0.5) if log else m - ci), m + ci
        if pct:
            lo, hi = max(lo, 0.0), min(hi, 100.0)
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=6, mfc=SURFACE if oracle else color, mec=color, mew=1.6, zorder=3)
    ax.set_yticks(ys)
    if labels:
        names = []
        for p in pols:
            tag = " (oracle)" if meta[(scenario, "base", p)][1] else (" (baseline)" if p == baseline else "")
            names.append(p + tag)
        ax.set_yticklabels(names)
    else:
        ax.set_yticklabels([])
    if log:
        ax.set_xscale("log")
        if max(vals) / min(vals) < 100:  # under two decades: 1-2-5 ticks, axis widened to the next tick on each side
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            lo_v, hi_v = ax.get_xlim()
            ax.set_xlim(step125(lo_v, down=True), step125(hi_v, down=False))
        else:  # wider: a label at every power of ten in view
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(pols) - 0.4)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})  # no tool version in the PNG
    plt.close(fig)
    size = os.path.getsize(path)
    if size >= 300 * 1024:
        raise SystemExit(f"{name} is {size} bytes (limit 300 KB)")
    print(f"wrote {path} ({size // 1024} KB)")


def fig_routing(agg, meta, out, baseline):
    scs = scenarios_of(meta, "routing")
    if not scs:
        return
    cols = min(2, len(scs))  # two columns keep the text legible in an 880 px README column
    rows = math.ceil(len(scs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.4 * cols, 6.6), 2.7 * rows + 1.0), squeeze=False)
    for i, sc in enumerate(scs):
        ax = axes[i // cols][i % cols]
        dot_panel(ax, agg, meta, sc, "J", baseline, log=True, labels=(i % cols == 0))
        ax.set_title(sc, loc="left")
    for j in range(len(scs), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Routing (simulated, assumed parameters): mean J ± 95% CI over 20 evaluation seeds\n(log axis where the range is wide), lower is better",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "routing_J.png")


def fig_family_metrics(agg, meta, out, family, baseline, metrics, fname, title):
    scs = scenarios_of(meta, family)
    if not scs:
        return
    if len(scs) == 1:  # one scenario: a 2 x 2 grid of metrics instead of a wide strip
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.4), squeeze=False)
        for j, (metric, label, log, pct) in enumerate(metrics):
            ax = axes[j // 2][j % 2]
            dot_panel(ax, agg, meta, scs[0], metric, baseline, log=log, pct=pct, labels=(j % 2 == 0))
            ax.set_title(label, loc="left")
        axes[0][0].set_ylabel(scs[0], color=INK, fontweight="bold")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        save(fig, out, fname)
        return
    fig, axes = plt.subplots(len(scs), len(metrics), figsize=(1.95 * len(metrics) + 1.8, 2.3 * len(scs) + 0.6), squeeze=False)
    for i, sc in enumerate(scs):
        for j, (metric, label, log, pct) in enumerate(metrics):
            ax = axes[i][j]
            dot_panel(ax, agg, meta, sc, metric, baseline, log=log, pct=pct, labels=(j == 0))
            if i == 0:
                ax.set_title(label, loc="left")
            if j == 0:
                ax.set_ylabel(sc, color=INK, fontweight="bold")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, fname)


# README versions: one scenario each, at most about 880 px wide, so that the text stays at 14 px or more
# and the titles at 16 px or more in GitHub's 880 px column
README_FONT = {"font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10}


def fig_readme_routing(agg, meta, out, scenario, baseline):
    if (scenario, "base", baseline) not in agg:
        return
    with plt.rc_context(README_FONT):
        fig, ax = plt.subplots(figsize=(7.4, 3.9))
        dot_panel(ax, agg, meta, scenario, "J", baseline, log=True)
        ax.set_title(f"{scenario}: mean J ± 95% CI over 20 evaluation seeds", loc="left")
        ax.set_xlabel("mean J (lower is better)")
        fig.suptitle("Routing (simulated, assumed parameters)", x=0.01, ha="left", fontsize=11.5, color=INK)
        fig.tight_layout()
        save(fig, out, "readme_routing_J.png")


def fig_readme_family(agg, meta, out, scenario, baseline, metrics, fname, title):
    if (scenario, "base", baseline) not in agg:
        return
    with plt.rc_context(README_FONT):
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), squeeze=False)
        for j, (metric, label, log, pct) in enumerate(metrics):
            ax = axes[j // 2][j % 2]
            dot_panel(ax, agg, meta, scenario, metric, baseline, log=log, pct=pct, labels=(j % 2 == 0))
            ax.set_title(label, loc="left")
        fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
        fig.tight_layout()
        save(fig, out, fname)


def fig_sensitivity(sens, out):
    if not sens:
        return
    scs = []
    for r in sens:
        if r["scenario"] not in scs:
            scs.append(r["scenario"])
    fig, axes = plt.subplots(len(scs), 1, figsize=(7.8, 2.6 * len(scs) + 0.6), squeeze=False)  # stacked: legible at 880 px
    for i, sc in enumerate(scs):
        ax = axes[i][0]
        rows = [r for r in sens if r["scenario"] == sc]
        ys = list(range(len(rows)))[::-1]
        for y, r in zip(ys, rows):
            tau = float(r["kendall_tau"])
            ax.plot([0, tau], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(tau, y, "o", ms=6, color=BLUE)
            note = "top policy kept" if r["top_same"] == "true" else f"top: {r['top_policy']}"
            ax.text(1.04, y, f"{note}; verdicts kept {r['verdicts_same']}/{r['verdicts_total']}", va="center", color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(ys)
        ax.set_yticklabels([r["variant"] for r in rows])
        ax.set_xlim(-1, 1)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{sc}: Kendall tau vs base ranking", loc="left")
    fig.suptitle("Sensitivity (simulated): does the policy ranking survive changed assumptions?", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, out, "sensitivity.png")


def fmt(m, ci, digits=2, pct=False):
    if not finite(m):
        return "n/a"
    if pct:
        m, ci = 100 * m, 100 * ci
    if not finite(ci):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {ci:.{digits}f}"


def table(agg, meta, paired, manifest, out, families):
    lines = [
        f"<!-- generated by scripts/plot_results.py from {SOURCE}; do not edit by hand -->",
        f"Simulated, assumed parameters. Commit `{manifest['commit'][:7]}`, config `{manifest['config_sha256'][:12]}`, "
        f"{len(manifest['seeds'].get('evaluation', []))} evaluation seeds, {manifest['go_version']}, {manifest['hardware']}. "
        "Mean ± 95% t-interval over seeds; W/T/L = per-seed wins/ties/losses in J against the family baseline (tie: |ΔJ| ≤ 1%); "
        "verdict from the paired 95% interval of ΔJ.",
        "",
    ]
    for family, baseline in families:
        for sc in scenarios_of(meta, family):
            lines += [f"**{sc}** ({family}, baseline `{baseline}`)", "",
                      "| policy | J | P95 TTFT (s) | P95 TPOT (ms) | violations (%) | rejected (%) | GPU-h | W/T/L | verdict |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for p in ordered(agg, meta, sc):
                a = agg[(sc, "base", p)]
                name = p + (" (oracle)" if meta[(sc, "base", p)][1] else "")
                pr = paired.get((sc, "base", p))
                wtl = f"{pr['wins']}/{pr['ties']}/{pr['losses']}" if pr else "—"
                verdict = pr["verdict"].replace("_", " ") if pr else "baseline"
                tpot = a["tpot_p95"]
                lines.append(f"| {name} | {fmt(*a['J'][:2], digits=3)} | {fmt(*a['ttft_p95'][:2])} | {fmt(tpot[0] * 1000, tpot[1] * 1000, 1)} | "
                             f"{fmt(*a['violation_rate'][:2], digits=1, pct=True)} | {fmt(*a['rejected_rate'][:2], digits=1, pct=True)} | "
                             f"{fmt(*a['gpu_hours'][:2])} | {wtl} | {verdict} |")
            lines.append("")
    path = os.path.join(out, "results_table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


VERDICT = {"better": "✓", "worse": "✗", "no_difference": "="}


def summary(agg, meta, paired, families):
    """Condensed tables: mean J per policy (rows) and scenario (columns), with
    the paired verdict against the family baseline."""
    lines = [f"<!-- generated by scripts/plot_results.py from {SOURCE}; do not edit by hand -->",
             "Mean J over 20 evaluation seeds (lower is better). ✓ / ✗ / = : better / worse / no difference "
             "than the family baseline (paired 95% interval of ΔJ). Oracles are references, not deployable. "
             "Simulated, assumed parameters.", ""]
    for family, baseline in families:
        scs = scenarios_of(meta, family)
        if not scs:
            continue
        pols = ordered(agg, meta, scs[0])
        lines.append(f"**{family}** (baseline `{baseline}`)")
        lines.append("")
        lines.append("| policy | " + " | ".join(scs) + " |")
        lines.append("|---|" + "---|" * len(scs))
        for p in pols:
            name = p + (" (oracle)" if meta[(scs[0], "base", p)][1] else "") + (" (baseline)" if p == baseline else "")
            cells = []
            for sc in scs:
                m = agg[(sc, "base", p)]["J"][0]
                pr = paired.get((sc, "base", p))
                mark = VERDICT.get(pr["verdict"], "") if pr else ""
                cells.append(f"{m:.3f} {mark}".strip() if m < 100 else f"{m:.1f} {mark}".strip())
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def decision_cost(path):
    """Table of ns/op from the committed microbenchmark output."""
    if not os.path.exists(path):
        return ""
    rows = defaultdict(dict)
    order = []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("Benchmark"):
            continue
        parts = line.split()
        name = parts[0].rsplit("-", 1)[0]  # drop the GOMAXPROCS suffix
        bench, policy, reps = name.split("/")
        key = (bench.replace("Benchmark", ""), policy)
        if key not in rows:
            order.append(key)
        n = reps.split("=")[1]
        v = float(parts[2])
        rows[key][n] = min(v, rows[key].get(n, v))  # minimum over repeated runs
    lines = ["<!-- generated by scripts/plot_results.py from benchmarks/results/microbench.txt; do not edit by hand -->",
             "Time per routing decision in ns (`go test -bench`, real CPU time of the policy code, not simulated; x86-64 desktop CPU "
             "with 32 logical CPUs, Windows 11, toolchain in `benchmarks/results/microbench.txt`; minimum of 5 runs, indicative only). "
             "`Route`: the policy alone on a prepared view. `Dispatch`: the full dispatcher path "
             "(eligible-view construction from the registry, snapshot lookup, policy, completion).", "",
             "| benchmark | policy | 8 replicas | 64 replicas | 512 replicas |", "|---|---|---|---|---|"]
    for key in order:
        r = rows[key]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(f"{r.get(n, float('nan')):,.0f}" for n in ("8", "64", "512")) + " |")
    return "\n".join(lines) + "\n"


def splice(path, marker, content):
    """Replace the text between <!-- marker:start --> and <!-- marker:end -->."""
    if not os.path.exists(path):
        return
    text = open(path, encoding="utf-8").read()
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    if start not in text or end not in text:
        return
    a, b = text.index(start) + len(start), text.index(end)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text[:a] + "\n" + content.rstrip("\n") + "\n" + text[b:])
    print(f"updated {path} [{marker}]")


# results folder named in the generated headers (set from --results)
SOURCE = "benchmarks/results"


def main():
    global SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--no-splice", action="store_true", help="do not update the README tables")
    ap.add_argument("--marker", default="results-summary", help="README marker that receives the summary table")
    args = ap.parse_args()
    SOURCE = args.results.replace("\\", "/").rstrip("/")
    os.makedirs(args.out, exist_ok=True)
    agg, meta, paired, sens, manifest = load(args.results)
    fig_routing(agg, meta, args.out, "least_outstanding")
    fig_family_metrics(agg, meta, args.out, "scaling", "threshold_cooldown",
                       [("J", "J (log)", True, False), ("violation_rate", "SLO violations (%)", False, True),
                        ("ttft_p95", "P95 TTFT (s, log)", True, False), ("gpu_hours", "GPU-hours", False, False)],
                       "scaling_tradeoff.png", "Scaling (simulated, assumed parameters): latency, violations, and cost per policy; mean ± 95% CI")
    fig_family_metrics(agg, meta, args.out, "capacity", "static_partition",
                       [("J", "J (log)", True, False), ("violation_rate", "SLO violations (%)", False, True),
                        ("ttft_p95", "P95 TTFT (s, log)", True, False), ("gpu_hours", "GPU-hours", False, False)],
                       "capacity.png", "Capacity allocation, three models under one GPU budget (simulated): mean ± 95% CI")
    fig_family_metrics(agg, meta, args.out, "overload", "none",
                       [("J", "J (log)", True, False), ("violation_rate", "SLO violations (%)", False, True),
                        ("rejected_rate", "rejected (%)", False, True), ("ttft_p95", "P95 TTFT (s, log)", True, False)],
                       "overload.png", "Overload with and without admission control (simulated): mean ± 95% CI")
    fig_sensitivity(sens, args.out)
    fig_readme_routing(agg, meta, args.out, "r-steady90-short", "least_outstanding")
    fig_readme_family(agg, meta, args.out, "s-bursty-short", "threshold_cooldown",
                      [("J", "J (log)", True, False), ("violation_rate", "SLO violations (%)", False, True),
                       ("ttft_p95", "P95 TTFT (s, log)", True, False), ("gpu_hours", "GPU-hours", False, False)],
                      "readme_scaling.png", "Scaling in s-bursty-short (simulated, assumed parameters): mean ± 95% CI")
    fams = [("routing", "least_outstanding"), ("scaling", "threshold_cooldown"), ("capacity", "static_partition"), ("overload", "none")]
    table(agg, meta, paired, manifest, args.out, fams + [("sample", "least_outstanding")])
    summ = summary(agg, meta, paired, fams)
    with open(os.path.join(args.out, "results_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summ + "\n")
    cost = decision_cost(os.path.join(args.results, "microbench.txt"))
    for readme in () if args.no_splice else ("README.md", os.path.join("benchmarks", "README.md")):
        splice(readme, args.marker, summ)
        if cost and args.marker == "results-summary":
            splice(readme, "decision-cost", cost)


if __name__ == "__main__":
    main()
