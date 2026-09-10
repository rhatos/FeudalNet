"""
plot_evaluation.py

Plot agent evaluation results as a ranked win-rate bar chart — one bar per
opponent, labelled with the win percentage:

    mayari          |=============================| 100%
    coacAI          |==========================|     88%

Styled to match plot_standings.py so the two figures sit together: bars are
coloured on a diverging scale centred on 50%, with a dashed reference line at
that mark. The OVERALL row is excluded — it aggregates opponents of very
different difficulty, so it is not comparable to the per-opponent bars.

Takes one or more *_summary.csv files written by evaluate_agent.py. Those
files use a VERTICAL layout (metrics down the rows, opponents across the
columns); the older horizontal layout is also accepted.

When several CSVs are given, opponents are merged across them and each
opponent keeps its single BEST result — the row with the highest win rate
(ties broken toward more games). This gives a best-of chart across runs, so
the per-opponent game counts (and the header) reflect only the winning run,
not the total played.

A merged run also writes the combined table back out as a single vertical
*_summary.csv (default: <first csv>_best_summary.csv, or --combined-out).
Each opponent's column is copied whole from its winning file — including the
steps_* statistics — so the result is a drop-in summary for the companion
scripts (plot_winrate_comparison.py, plot_steps_comparison.py) or for a later
run of this one. Context rows (checkpoint/map/...) are written only where all
the inputs agree.

Usage:
    python plot_evaluation.py results/eval_..._summary.csv
    python plot_evaluation.py results/run1_summary.csv results/run2_summary.csv
    python plot_evaluation.py results/*_summary.csv --out fig/eval.pdf
    python plot_evaluation.py results/*_summary.csv --combined-out best.csv
    python plot_evaluation.py results/..._summary.csv --highlight coacAI
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# Non-metric rows in the vertical summary layout (run metadata).
CONTEXT_KEYS = ("checkpoint", "map", "sides", "seed", "num_games")


def load_summary(path):
    """
    Read evaluate_agent.py's *_summary.csv in either layout.

    Returns (rows, context); rows have name/games/wins/losses/draws/win_pct.
    The OVERALL aggregate is dropped.
    """
    with open(path, newline="") as f:
        raw = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not raw:
        raise ValueError(f"{path} is empty")

    header = [h.strip() for h in raw[0]]
    rows, context = [], {}

    # --- Vertical layout: first cell 'metric', opponents across the top ---
    if header[0].lower() == "metric":
        names = [h.strip() for h in header[1:]]
        table = {}
        for r in raw[1:]:
            key  = r[0].strip()
            vals = [c.strip() for c in r[1:]]
            if key in CONTEXT_KEYS:
                context[key] = vals[0] if vals else ""
            else:
                table[key] = vals

        if "wins" not in table:
            raise ValueError(
                f"{path} has no 'wins' metric. Found: {sorted(table)[:12]}...")

        def num(metric, i, default=0.0):
            try:
                return float(table[metric][i])
            except (KeyError, IndexError, ValueError):
                return default

        for i, nm in enumerate(names):
            w, l, d = num("wins", i), num("losses", i), num("draws", i)
            games = num("games", i, w + l + d)
            if games <= 0:
                continue                                  # no data
            try:
                target = int(float(context.get("num_games", 0))
                             * num("repeats", i, 1)) or int(games)
            except (TypeError, ValueError):
                target = int(games)
            rows.append({"name": nm, "wins": int(w), "losses": int(l),
                         "draws": int(d), "games": int(games),
                         "target": target,
                         "std": num("win_pct_std", i, 0.0),
                         "win_pct": num("win_pct", i, 100.0 * w / games)})

    # --- Horizontal layout: one row per opponent ---
    elif "opponent" in header:
        for r in csv.DictReader(open(path, newline="")):
            try:
                w, l, d = int(r["wins"]), int(r["losses"]), int(r["draws"])
            except (KeyError, ValueError):
                continue
            games = int(r.get("games") or (w + l + d))
            if games <= 0:
                continue
            rows.append({"name": r["opponent"], "wins": w, "losses": l,
                         "draws": d, "games": games, "target": games,
                         "std": float(r.get("win_pct_std") or 0.0),
                         "win_pct": float(r.get("win_pct")
                                          or 100.0 * w / games)})
            for k in ("checkpoint", "map", "sides"):
                if r.get(k):
                    context[k] = r[k]
    else:
        raise ValueError(
            f"{path} does not look like an evaluation summary "
            f"(first cell is {header[0]!r}, expected 'metric' or 'opponent').\n"
            f"Pass the *_summary.csv written by evaluate_agent.py.")

    # Drop the aggregate — it mixes opponents of different difficulty
    dropped = [r["name"] for r in rows if r["name"].upper() == "OVERALL"]
    rows = [r for r in rows if r["name"].upper() != "OVERALL"]
    if dropped:
        print(f"  (excluding {', '.join(dropped)} from the chart)")

    if not rows:
        raise ValueError(f"No per-opponent rows found in {path}")
    return rows, context


def read_raw_columns(path):
    """
    Read a summary CSV as raw per-opponent metric columns, both layouts.

    Returns (metric_order, columns) where columns maps opponent name to an
    ordered {metric: string value} dict. Values are kept verbatim so the
    combined file preserves everything the source wrote (steps_* statistics,
    medians, ...), not just the win counts load_summary extracts.
    """
    with open(path, newline="") as f:
        raw = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    header = [h.strip() for h in raw[0]]
    metric_order, columns = [], {}

    if header[0].lower() == "metric":
        names = [h.strip() for h in header[1:]]
        for r in raw[1:]:
            key  = r[0].strip()
            if key in CONTEXT_KEYS:
                continue
            vals = [c.strip() for c in r[1:]]
            metric_order.append(key)
            for i, nm in enumerate(names):
                columns.setdefault(nm, {})[key] = (
                    vals[i] if i < len(vals) else "")
    elif "opponent" in header:
        for r in csv.DictReader(open(path, newline="")):
            nm = (r.get("opponent") or "").strip()
            if not nm:
                continue
            col = {}
            for k, v in r.items():
                k = k.strip()
                if k == "opponent" or k in CONTEXT_KEYS:
                    continue
                if k not in metric_order:
                    metric_order.append(k)
                col[k] = (v or "").strip()
            columns[nm] = col
    # Anything else already failed loudly in load_summary.
    return metric_order, columns


def write_combined(path, rows, columns, metric_order, context):
    """
    Write the merged best-per-opponent table as a vertical summary CSV that
    load_summary (and the companion scripts) can read straight back.

    Opponent columns are ordered by descending win rate, matching the chart.
    """
    names = [r["name"] for r in
             sorted(rows, key=lambda r: (-r["win_pct"], r["name"]))]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        for key in CONTEXT_KEYS:
            if context.get(key):
                w.writerow([key] + [context[key]] * len(names))
        for metric in metric_order:
            w.writerow([metric] + [columns[nm].get(metric, "")
                                   for nm in names])
    print(f"Wrote combined summary {path} "
          f"({len(names)} opponents, {len(metric_order)} metrics)")


def merge_best(paths):
    """
    Load several summary CSVs and keep, for each opponent, the single row
    with the highest win rate (ties broken toward the larger sample).

    Returns (rows, context, columns, metric_order): rows/context as
    load_summary gives them, plus each best opponent's full raw metric
    column (from its winning file) for writing a combined summary. Context
    keys are kept only where every file agrees — a merged chart over two
    maps has no single map to name in the header.
    """
    best, sources, columns = {}, {}, {}
    context, conflicting = {}, set()
    metric_order = []

    for path in paths:
        rows, ctx = load_summary(path)
        raw_order, raw_cols = read_raw_columns(path)
        for m in raw_order:
            if m not in metric_order:
                metric_order.append(m)
        print(f"Loaded {len(rows)} opponent(s) from {path}")
        for k, v in ctx.items():
            if k in conflicting or not v:
                continue
            if k in context and context[k] != v:
                del context[k]
                conflicting.add(k)
            elif k not in context:
                context[k] = v

        for r in rows:
            cur = best.get(r["name"])
            if cur is None or (r["win_pct"], r["games"]) > (cur["win_pct"],
                                                            cur["games"]):
                if cur is not None:
                    print(f"  {r['name']}: {r['win_pct']:.1f}% beats "
                          f"{cur['win_pct']:.1f}% from "
                          f"{os.path.basename(sources[r['name']])}")
                best[r["name"]] = r
                sources[r["name"]] = path
                columns[r["name"]] = raw_cols.get(r["name"], {})

    if len(paths) > 1:
        if conflicting:
            print(f"  ({', '.join(sorted(conflicting))} differ between files "
                  f"— omitted from the header)")
        print("Best result per opponent:")
        for nm in sorted(best, key=lambda n: -best[n]["win_pct"]):
            print(f"  {nm:<20s} {best[nm]['win_pct']:6.1f}%  "
                  f"({best[nm]['games']} games, "
                  f"{os.path.basename(sources[nm])})")

    return list(best.values()), context, columns, metric_order


def plot(rows, context, args):
    rows = sorted(rows, key=lambda r: (-r["win_pct"], r["name"]))
    names = [r["name"] for r in rows]
    pct   = np.array([r["win_pct"] for r in rows])
    std   = np.array([r.get("std", 0.0) for r in rows])
    n     = len(rows)

    # Diverging colours centred on an even record, matching plot_standings.py
    norm   = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    cmap   = matplotlib.colormaps["RdBu"]
    colors = [cmap(norm(p)) for p in pct]

    fig, ax = plt.subplots(figsize=(args.width, max(3.0, 0.42 * n + 1.8)),
                           dpi=args.dpi)

    pos = np.arange(n)[::-1]                     # best at the top
    ax.barh(pos, pct, height=0.78, color=colors,
            edgecolor="#222222", linewidth=0.9, zorder=3,
            xerr=std, error_kw=dict(ecolor="#444444", elinewidth=1.6,
                                    capsize=4, capthick=1.6, zorder=4))

    # Ring a named opponent if requested (e.g. the headline matchup)
    for i, nm in enumerate(names):
        if nm == args.highlight:
            ax.add_patch(Rectangle((0, pos[i] - 0.39), pct[i], 0.78,
                                   fill=False, edgecolor=args.highlight_color,
                                   linewidth=2.6, zorder=5))

    for i, r in enumerate(rows):
        # Completed/target games inside the bar, tucked against its end
        ax.text(max(pct[i] - 1.5, 2.0), pos[i],
                f"{r['games']}/{r['target']}", ha="right", va="center",
                fontsize=args.font_size - 1, color="white", alpha=0.85,
                fontfamily="monospace", zorder=4)
        # Win rate with seed-level spread to the right of the error bar
        ax.text(105.0, pos[i], f"{pct[i]:.0f}% \u00b1{std[i]:.0f}",
                ha="left", va="center", fontsize=args.font_size + 1,
                fontweight="bold", zorder=4)

    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=args.font_size, fontweight="bold",
                       fontfamily="monospace")
    # The axis runs past 100 to leave room for the percentage labels, so the
    # xlabel must be pinned to the centre of the 0-100 tick range rather than
    # the centre of the axes, or it reads as shifted to the right.
    x_max = 126.0
    ax.set_xlabel("Win Rate (%)", fontsize=args.font_size + 3,
                  fontweight="bold")
    ax.xaxis.set_label_coords(50.0 / x_max, -0.085)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100])

    ax.grid(axis="x", color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=args.font_size + 2)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    if args.show_even_line:
        ax.axvline(50, color="#999999", linewidth=1.6, linestyle="--", zorder=2)
        ax.text(51.0, -0.55, "50% WR", ha="left", va="bottom",
                fontsize=args.font_size, style="italic", color="#888888",
                zorder=4)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Header: overall progress and win total, e.g.
    # "14000/14000 games  •  12501 wins overall (89%)"
    total_games  = sum(r["games"] for r in rows)
    total_target = sum(r["target"] for r in rows)
    total_wins   = sum(r["wins"] for r in rows)
    overall_pct  = 100.0 * total_wins / max(total_games, 1)
    bits = [f"{total_games}/{total_target} games",
            f"{total_wins} wins overall ({overall_pct:.0f}%)"]
    if args.map_name:
        bits.insert(0, f"{args.map_name} Map")
    # Centred on the tick range for the same reason as the xlabel: the axes
    # run past 100 to fit the percentage labels.
    ax.set_title("   \u2022   ".join(bits), fontsize=args.font_size + 3,
                 style="italic", fontweight="bold", color="#555555", pad=14,
                 x=50.0 / x_max)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Plot evaluation win rate per opponent")
    p.add_argument("csv", nargs="+",
                   help="One or more *_summary.csv files from "
                        "evaluate_agent.py; with several, each opponent "
                        "keeps its best win rate across them")
    p.add_argument("--out", default=None,
                   help="Output image (.png/.pdf/.svg). Default: next to the CSV")
    p.add_argument("--highlight", default=None,
                   help="Opponent to ring with a highlight border")
    p.add_argument("--highlight-color", default="#f5b800")
    p.add_argument("--map-name", default=None,
                   help="Override the map name shown in the header")
    p.add_argument("--combined-out", default=None,
                   help="Where to write the merged best-per-opponent summary "
                        "CSV (only written when several CSVs are given). "
                        "Default: <first csv>_best_summary.csv")
    p.add_argument("--no-even-line", dest="show_even_line",
                   action="store_false",
                   help="Hide the 50%% win-rate reference line")
    p.add_argument("--width", type=float, default=11.0)
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    rows, context, columns, metric_order = merge_best(args.csv)
    if context.get("checkpoint"):
        print(f"  checkpoint: {context['checkpoint']}")

    if len(args.csv) > 1:
        combined = args.combined_out or (
            os.path.splitext(args.csv[0])[0] + "_best_summary.csv")
        os.makedirs(os.path.dirname(combined) or ".", exist_ok=True)
        write_combined(combined, rows, columns, metric_order, context)

    fig = plot(rows, context, args)
    out = args.out or (os.path.splitext(args.csv[0])[0]
                       + ("_best_winrate.png" if len(args.csv) > 1
                          else "_winrate.png"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
