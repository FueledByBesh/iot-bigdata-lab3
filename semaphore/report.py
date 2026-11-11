"""Plot styling and result-table helpers shared by the experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path

import os

import matplotlib

if os.environ.get("LAB3_INTERACTIVE") != "1":   # visualize.py needs a real window backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
MODELS = ROOT / "models"

# Categorical slots, fixed order (validated palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Each controller keeps the same colour in every figure.
CTRL_COLOR = {
    "Fixed (Webster)": SERIES[0],
    "FLC (paper rules)": SERIES[1],
    "FNN-GA": SERIES[2],
    "FNN-GA naive": SERIES[2],
    "FNN-GA coordinated": SERIES[3],
    "FNN-GA retuned": SERIES[6],
}


def use_quick_dirs():
    """--quick writes to results/quick/ so it never overwrites the full-run outputs or models."""
    global FIG, TAB, MODELS
    q = RESULTS / "quick"
    FIG, TAB, MODELS = q / "figures", q / "tables", q / "models"


def setup():
    for d in (FIG, TAB, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "text.color": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    return path


def grouped_bars(ax, df, value, group, series, colors=None, err=None, fmt="{:.1f}", label_values=True):
    """df: long table; one bar group per `group`, one bar per `series`."""
    groups = list(dict.fromkeys(df[group]))
    sers = list(dict.fromkeys(df[series]))
    n = len(sers)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(sers):
        sub = df[df[series] == s].set_index(group).reindex(groups)
        col = (colors or {}).get(s, SERIES[i % len(SERIES)])
        yerr = sub[err].values if err else None
        bars = ax.bar(x + (i - (n - 1) / 2) * width, sub[value].values, width * 0.92,
                      color=col, label=s, yerr=yerr, error_kw={"ecolor": TEXT2, "elinewidth": 1, "capsize": 2},
                      edgecolor=SURFACE, linewidth=1)
        if label_values:
            for b, v in zip(bars, sub[value].values):
                if np.isfinite(v):
                    ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                                fontsize=7, color=TEXT2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="x", visible=False)


def write_table(df: pd.DataFrame, name: str, floatfmt=2):
    df.to_csv(TAB / f"{name}.csv", index=False)
    md = df.copy()
    for c in md.columns:
        if pd.api.types.is_float_dtype(md[c]):
            md[c] = md[c].map(lambda v: f"{v:.{floatfmt}f}" if pd.notna(v) else "")
    lines = ["| " + " | ".join(map(str, md.columns)) + " |",
             "| " + " | ".join("---" for _ in md.columns) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in md.values]
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")
    return df


def write_json(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    (TAB / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv))


METRIC_COLS = ["avg_queue", "avg_wait", "p95_wait", "avg_delay_all", "p95_delay_all",
               "worst_node_wait", "stops_per_trip", "throughput_vph", "blocked_green_s", "unfinished"]


def summarise(rows, keys):
    """Mean (and std of avg_wait) over seeds."""
    df = pd.DataFrame(rows)
    agg = df.groupby(keys, sort=False)[METRIC_COLS].mean().reset_index()
    agg["avg_wait_std"] = df.groupby(keys, sort=False)["avg_wait"].std().values
    agg["avg_queue_std"] = df.groupby(keys, sort=False)["avg_queue"].std().values
    return agg
