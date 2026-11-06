#!/usr/bin/env python3
"""Turn benchmark CSVs into charts + a summary table for the slides.
Needs matplotlib:  pip install matplotlib

  python3 plot.py                              uses every results/*.csv
  python3 plot.py results/cpu.csv results/gpu.csv

If the same measurement appears in several files, the later file wins - so you
can combine OpenMP numbers from your laptop with CUDA numbers from Colab.
Writes charts/*.png and charts/summary.md
"""
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE_NAMES = {(1280, 720): "HD", (1920, 1080): "Full HD", (3840, 2160): "4K", (7680, 4320): "8K"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: blue, orange, aqua, yellow
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "text.color": INK,
})


def load(paths):
    data = {}
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                for k in ("threads", "width", "height"):
                    r[k] = int(r[k])
                for k in ("blur_ms", "sobel_ms", "transfer_ms", "total_ms"):
                    r[k] = float(r[k])
                data[(r["version"], r["threads"], r["width"], r["height"])] = r
    return list(data.values())


def size_name(wh):
    return SIZE_NAMES.get(wh, f"{wh[0]}x{wh[1]}")


def pick(rows, wh, prefix):
    """Fastest row for this image size whose version starts with prefix."""
    cands = [r for r in rows if (r["width"], r["height"]) == wh and r["version"].startswith(prefix)]
    return min(cands, key=lambda r: r["total_ms"]) if cands else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"charts/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"charts/{name}")


def chart_versions(rows, sizes):
    """Grouped bars: total time of each version per image size (log scale)."""
    series = [("Serial", "serial"), ("OpenMP (best)", "openmp"),
              ("CUDA naive", "cuda_naive"), ("CUDA shared mem", "cuda_tiled")]
    series = [s for s in series if any(pick(rows, wh, s[1]) for wh in sizes)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(series)
    for i, (label, prefix) in enumerate(series):
        xs, ys = [], []
        for j, wh in enumerate(sizes):
            r = pick(rows, wh, prefix)
            if r:
                xs.append(j + (i - (len(series) - 1) / 2) * width)
                ys.append(r["total_ms"])
        ax.bar(xs, ys, width * 0.92, label=label, color=COLORS[i])
        if prefix != "serial":  # speedup over serial on top of each bar
            for x, y, wh in zip(xs, ys, sizes):
                s = pick(rows, wh, "serial")
                if s:
                    ax.text(x, y * 1.08, f"{s['total_ms'] / y:.0f}x", ha="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), [size_name(wh) for wh in sizes])
    ax.set_ylabel("Total time (ms, log scale) - lower is better")
    ax.set_title("Blur + edge detection: time per image")
    ax.legend(ncol=len(series), loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "1_time_by_version.png")


def openmp_curve(rows, wh, sched):
    serial = pick(rows, wh, "serial")
    pts = sorted((r["threads"], r["total_ms"]) for r in rows
                 if r["version"] == f"openmp_{sched}" and (r["width"], r["height"]) == wh)
    if not serial or not pts:
        return [], []
    return [t for t, _ in pts], [serial["total_ms"] / ms for _, ms in pts]


def chart_openmp(rows, sizes):
    """Speedup and efficiency vs threads (static schedule), one line per image size."""
    curves = [(wh, *openmp_curve(rows, wh, "static")) for wh in sizes]
    curves = [c for c in curves if c[1]]
    if not curves:
        return
    max_t = max(max(c[1]) for c in curves)
    for kind in ("speedup", "efficiency"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        if kind == "speedup":
            ax.plot([1, max_t], [1, max_t], "--", color=MUTED, lw=1.2, label="Ideal (linear)")
        else:
            ax.axhline(100, ls="--", color=MUTED, lw=1.2, label="Ideal (100%)")
        for i, (wh, ts, sp) in enumerate(curves):
            ys = sp if kind == "speedup" else [100 * s / t for s, t in zip(sp, ts)]
            ax.plot(ts, ys, "-o", lw=2, ms=6, color=COLORS[i], label=size_name(wh))
        ax.set_xticks(curves[0][1])
        ax.set_xlabel("OpenMP threads")
        if kind == "speedup":
            ax.set_ylabel("Speedup over serial (x)")
            ax.set_title("OpenMP speedup (static schedule)")
        else:
            ax.set_ylabel("Efficiency = speedup / threads (%)")
            ax.set_ylim(0, 115)
            ax.set_title("OpenMP efficiency (static schedule)")
        ax.legend()
        save(fig, f"{2 if kind == 'speedup' else 3}_openmp_{kind}.png")


def chart_schedule(rows, sizes):
    """static vs dynamic scheduling on the largest image."""
    wh = sizes[-1]
    lines = [(s, *openmp_curve(rows, wh, s)) for s in ("static", "dynamic")]
    lines = [line for line in lines if line[1]]
    if len(lines) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (sched, ts, sp) in enumerate(lines):
        ax.plot(ts, sp, "-o", lw=2, ms=6, color=COLORS[i], label=f"schedule({sched})")
    ax.set_xticks(lines[0][1])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Speedup over serial (x)")
    ax.set_title(f"Static vs dynamic scheduling ({size_name(wh)} image)")
    ax.legend()
    save(fig, "4_openmp_schedule.png")


def chart_gpu(rows, sizes):
    """Where the GPU time goes: copies vs the two kernels."""
    bars = []
    for wh in sizes:
        for prefix, label in (("cuda_naive", "naive"), ("cuda_tiled", "shared")):
            r = pick(rows, wh, prefix)
            if r:
                bars.append((f"{size_name(wh)}\n{label}", r))
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = range(len(bars))
    bottom = [0.0] * len(bars)
    for i, (key, label) in enumerate([("transfer_ms", "CPU <-> GPU copies"),
                                      ("blur_ms", "Blur kernel"), ("sobel_ms", "Sobel kernel")]):
        vals = [r[key] for _, r in bars]
        ax.bar(xs, vals, 0.6, bottom=bottom, color=COLORS[i], label=label, edgecolor="white", linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, (_, r) in zip(xs, bars):
        pct = 100 * r["transfer_ms"] / r["total_ms"]
        ax.text(x, r["total_ms"] * 1.02, f"{pct:.0f}% copying", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title("GPU time breakdown: copying data vs computing")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "5_gpu_breakdown.png")


def summary(rows, sizes):
    lines = ["| Image | Serial (ms) | Best OpenMP (ms) | OpenMP speedup | Efficiency | "
             "Serial fraction f (Karp-Flatt) | CUDA kernels only (ms) | CUDA incl. copies (ms) | "
             "GPU speedup (kernels / incl. copies) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for wh in sizes:
        s, o = pick(rows, wh, "serial"), pick(rows, wh, "openmp")
        g = pick(rows, wh, "cuda")
        cells = [size_name(wh), f"{s['total_ms']:.1f}" if s else "-"]
        if s and o:
            sp, p = s["total_ms"] / o["total_ms"], o["threads"]
            # Karp-Flatt metric: experimentally measured serial fraction (Amdahl's f)
            f = (1 / sp - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
            cells += [f"{o['total_ms']:.1f} ({p} threads)", f"{sp:.1f}x", f"{100 * sp / p:.0f}%", f"{f:.3f}"]
        else:
            cells += ["-"] * 4
        if g:
            kern = g["blur_ms"] + g["sobel_ms"]
            cells += [f"{kern:.2f}", f"{g['total_ms']:.2f}"]
            cells.append(f"{s['total_ms'] / kern:.0f}x / {s['total_ms'] / g['total_ms']:.0f}x" if s else "-")
        else:
            cells += ["-"] * 3
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open("charts/summary.md", "w") as f:
        f.write(text)
    print("charts/summary.md\n\n" + text)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/*.csv"))
    if not paths:
        sys.exit("No results/*.csv yet - run bench.sh first")
    rows = load(paths)
    sizes = sorted({(r["width"], r["height"]) for r in rows}, key=lambda wh: wh[0] * wh[1])
    os.makedirs("charts", exist_ok=True)
    chart_versions(rows, sizes)
    chart_openmp(rows, sizes)
    chart_schedule(rows, sizes)
    chart_gpu(rows, sizes)
    summary(rows, sizes)


if __name__ == "__main__":
    main()
