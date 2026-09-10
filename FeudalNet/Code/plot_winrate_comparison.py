"""
plot_winrate_comparison.py

Compare win rates from a multi-opponent evaluation ("baseline") against
evaluations run against a single opponent each ("focused"), as paired bars.

The typical use: you evaluate one checkpoint across all opponents, then
fine-tune or re-evaluate against specific opponents individually, and want to
see per-opponent what changed.

Only opponents that appear in the individual CSVs are plotted — the baseline
supplies the comparison value, not the set of bars.

Both *_summary.csv and *_games.csv from evaluate_agent.py are accepted.

Usage:
    python plot_winrate_comparison.py \\
        --summary results/eval_tqlgxayp_all_summary.csv \\
        --individual results/vs_izanagi_summary.csv \\
                     results/vs_mayari_summary.csv \\
                     results/vs_droplet_summary.csv

    # Custom legend names, sorted by biggest change, vector output
    python plot_winrate_comparison.py \\
        --summary results/base_summary.csv \\
        --individual results/ft_*_summary.csv \\
        --label-baseline "before fine-tuning" \\
        --label-individual "after fine-tuning" \\
        --sort delta --out fig/comparison.pdf

When the CSVs carry repeat information (win_pct_std from --repeats runs, or
a `repeat` column in games CSVs), each bar gains an error-bar whisker showing
the standard deviation across repeats. CSVs without repeat information plot
exactly as before.

Each pair is annotated with the adaptation metric of the paper: the raw
delta in win-rate percentage points, and the headroom-normalised delta
    norm = (WR_post - WR_pre) / (100 - WR_pre)
which expresses the change as the fraction of the available headroom that
was recovered. The normalised value is undefined when the baseline is
already at 100% (no headroom) or when the opponent has no baseline row.
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_evaluation import load_summary

C_BASE = "#2166ac"   # blue — the multi-opponent baseline
C_INDIV = "#c1272d"  # red  — the focused, single-opponent runs

# Delta badges: strong text on a soft tint of the same hue
DELTA_STYLE = {
    "up":   {"text": "#166534", "fill": "#e7f6ec", "edge": "#8fd0a6"},
    "down": {"text": "#9b1c1c", "fill": "#fdeceb", "edge": "#eaa39f"},
    "flat": {"text": "#555555", "fill": "#f2f2f2", "edge": "#d0d0d0"},
}


def load_summary_stds(path):
    """
    Pull win_pct_std (and the repeats context) from a vertical summary CSV.
    Returns {opponent: std or None}. Layouts without the row give {}.
    """
    stds = {}
    try:
        with open(path, newline="") as f:
            raw = [r for r in csv.reader(f) if r and any(x.strip() for x in r)]
        if not raw or raw[0][0].strip().lower() != "metric":
            return stds
        names = [h.strip() for h in raw[0][1:]]
        for r in raw[1:]:
            if r[0].strip() == "win_pct_std":
                for n, v in zip(names, [x.strip() for x in r[1:]]):
                    try:
                        stds[n] = float(v)
                    except ValueError:
                        stds[n] = None
                break
    except Exception:                                        # noqa: BLE001
        pass
    return stds


def load_games_csv(path):
    """
    Aggregate a *_games.csv (one row per match) into per-opponent win rates.
    Accepted so either file evaluate_agent.py writes can be passed.
    """
    counts, context = {}, {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "result" not in rows[0]:
        raise ValueError("not a games CSV")
    for r in rows:
        opp = r.get("opponent")
        if not opp:
            continue
        c = counts.setdefault(opp, {"WIN": 0, "LOSS": 0, "DRAW": 0})
        res = (r.get("result") or "").upper()
        if res in c:
            c[res] += 1
        for k in ("checkpoint", "map", "sides"):
            if r.get(k):
                context[k] = r[k]

    # Per-repeat win rates -> std, when the repeat column is present
    rep_counts = {}
    for r in rows:
        opp, rep_id = r.get("opponent"), r.get("repeat")
        if opp and rep_id:
            b = rep_counts.setdefault(opp, {}).setdefault(
                rep_id, {"w": 0, "n": 0})
            b["n"] += 1
            if (r.get("result") or "").upper() == "WIN":
                b["w"] += 1
    stds = {}
    for opp, per in rep_counts.items():
        rates = [100.0 * b["w"] / b["n"] for b in per.values() if b["n"]]
        stds[opp] = (float(np.std(rates, ddof=1))
                     if len(rates) > 1 else None)

    out = []
    for opp, c in counts.items():
        n = c["WIN"] + c["LOSS"] + c["DRAW"]
        if n:
            out.append({"name": opp, "games": n, "wins": c["WIN"],
                        "losses": c["LOSS"], "draws": c["DRAW"],
                        "win_pct": 100.0 * c["WIN"] / n})
    return out, context, stds


def load_any(path):
    """Load a summary CSV, falling back to a games CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    try:
        rows, ctx = load_summary(path)
        return rows, ctx, load_summary_stds(path)
    except Exception as summary_err:                         # noqa: BLE001
        try:
            return load_games_csv(path)
        except Exception:                                    # noqa: BLE001
            raise ValueError(
                f"{path} is neither a summary nor a games CSV from "
                f"evaluate_agent.py.\n  summary parse said: {summary_err}"
            ) from None


def collect(args):
    """
    Build the comparison table.

    Returns (rows, baseline_ctx, indiv_ctx) where each row has the opponent,
    its baseline win rate (or None when the baseline never faced it) and its
    focused win rate.
    """
    base_rows, base_ctx, base_stds = load_any(args.summary)
    baseline = {r["name"]: r for r in base_rows}
    print(f"Baseline {args.summary}: {len(baseline)} opponent(s)")

    focused, indiv_ctx, indiv_stds = {}, {}, {}
    for path in args.individual:
        rows, ctx, stds = load_any(path)
        indiv_stds.update(stds)
        indiv_ctx.update({k: v for k, v in ctx.items() if v})
        if not rows:
            print(f"  WARNING: {path} has no opponent rows — skipped")
            continue
        if len(rows) > 1:
            print(f"  note: {os.path.basename(path)} contains "
                  f"{len(rows)} opponents — using all of them")
        for r in rows:
            if r["name"] in focused:
                print(f"  WARNING: {r['name']} appears in more than one "
                      f"individual CSV — keeping the later one ({path})")
            focused[r["name"]] = r
            print(f"  {os.path.basename(path)}: {r['name']} "
                  f"{r['win_pct']:.1f}% of {r['games']} games")

    if not focused:
        raise ValueError("No opponents found in the individual CSVs.")

    # Only opponents from the individual CSVs are plotted
    rows = []
    for name, f in focused.items():
        b = baseline.get(name)
        if b is None:
            print(f"  note: {name} is not in the baseline — bar shown "
                  f"without a comparison")
        delta = (f["win_pct"] - b["win_pct"]) if b else None
        headroom = (100.0 - b["win_pct"]) if b else None
        norm = (delta / headroom) if delta is not None \
            and headroom is not None and headroom > 1e-9 else None
        b_std = base_stds.get(name)
        i_std = indiv_stds.get(name)
        rows.append({
            "name": name,
            "base_pct": b["win_pct"] if b else None,
            "base_games": b["games"] if b else 0,
            "base_std": b_std,
            "indiv_pct": f["win_pct"],
            "indiv_games": f["games"],
            "indiv_std": i_std,
            "delta": delta,
            "norm": norm,
        })

    missing = sorted(set(baseline) - set(focused))
    if missing:
        print(f"Excluded {len(missing)} baseline opponent(s) with no "
              f"individual CSV: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))

    key = {
        "opponent": lambda r: r["name"],
        "baseline": lambda r: (-(r["base_pct"] if r["base_pct"] is not None
                                 else -1), r["name"]),
        "focused":  lambda r: (-r["indiv_pct"], r["name"]),
        "delta":    lambda r: (-(r["delta"] if r["delta"] is not None
                                 else -1e9), r["name"]),
        "norm":     lambda r: (-(r["norm"] if r["norm"] is not None
                                 else -1e9), r["name"]),
    }[args.sort]
    rows.sort(key=key)
    return rows, base_ctx, indiv_ctx


def plot(rows, args):
    n = len(rows)
    names = [r["name"] for r in rows]
    base = np.array([np.nan if r["base_pct"] is None else r["base_pct"]
                     for r in rows])
    indiv = np.array([r["indiv_pct"] for r in rows])

    horizontal = args.orientation == "horizontal"
    if horizontal:
        size = max(3.2, 0.62 * n + 1.8)
        fig, ax = plt.subplots(figsize=(args.width, size), dpi=args.dpi)
    else:
        per_cat = 1.7 if args.show_delta else 0.95
        size = max(6.0, per_cat * n + 2.4)
        fig, ax = plt.subplots(figsize=(size, args.height), dpi=args.dpi)

    idx = np.arange(n)
    w = 0.38
    common = dict(edgecolor="#222222", linewidth=1.0, zorder=3)

    if horizontal:
        pos = idx[::-1]
        ax.barh(pos + w / 2, np.nan_to_num(base), height=w, color=C_BASE,
                label=args.label_baseline, **common)
        ax.barh(pos - w / 2, indiv, height=w, color=C_INDIV,
                label=args.label_individual, **common)
    else:
        ax.bar(idx - w / 2, np.nan_to_num(base), width=w, color=C_BASE,
               label=args.label_baseline, **common)
        ax.bar(idx + w / 2, indiv, width=w, color=C_INDIV,
               label=args.label_individual, **common)

    # Error-bar whiskers where a std is known (asymmetrically clipped to
    # the valid 0-100 range, matching the live chart)
    def _hi(val, std):
        """Upper whisker extent (0 when there is none)."""
        return 0.0 if std is None else min(std, 100.0 - val)

    def _whisker(x, y, val, std, vertical):
        if std is None:
            return
        lo = min(std, val)
        err = [[lo], [_hi(val, std)]]
        kw = dict(fmt="none", ecolor="#111111", elinewidth=1.3,
                  capsize=3.0, capthick=1.3, zorder=6)
        if vertical:
            ax.errorbar(x, val, yerr=err, **kw)
        else:
            ax.errorbar(val, y, xerr=err, **kw)

    for i, r in enumerate(rows):
        if horizontal:
            if r["base_pct"] is not None:
                _whisker(None, pos[i] + w / 2, r["base_pct"],
                         r["base_std"], vertical=False)
            _whisker(None, pos[i] - w / 2, r["indiv_pct"],
                     r["indiv_std"], vertical=False)
        else:
            if r["base_pct"] is not None:
                _whisker(idx[i] - w / 2, None, r["base_pct"],
                         r["base_std"], vertical=True)
            _whisker(idx[i] + w / 2, None, r["indiv_pct"],
                     r["indiv_std"], vertical=True)

    # Value labels, and the change per pair
    for i, r in enumerate(rows):
        pairs = []
        if r["base_pct"] is not None:
            pairs.append((r["base_pct"], +1, r["base_std"]))
        pairs.append((r["indiv_pct"], -1, r["indiv_std"]))
        for val, side, std in pairs:
            lbl = f"{val:.0f}%" if float(val).is_integer() else f"{val:.1f}%"
            # Offset past the upper whisker tip so the error bar never
            # overlaps the percentage
            clear = _hi(val, std) + 1.5
            if horizontal:
                ax.text(val + clear, pos[i] + side * w / 2, lbl, ha="left",
                        va="center", fontsize=args.font_size - 1,
                        fontweight="bold", zorder=4)
            else:
                # Anchor each label at its bar centre but let it extend
                # away from the pair's seam, so near-equal heights can't
                # produce touching labels
                ax.text(idx[i] - side * w / 2, val + clear, lbl,
                        ha="right" if side > 0 else "left", va="bottom",
                        fontsize=args.font_size - 1,
                        fontweight="bold", zorder=4)

        # Change badge — a soft pill so it reads as an annotation rather than
        # competing with the bar labels for attention
        if args.show_delta and r["delta"] is not None:
            d = r["delta"]
            style = DELTA_STYLE["up" if d > 0.05 else
                                "down" if d < -0.05 else "flat"]
            glyph = "\u25b2" if d > 0.05 else "\u25bc" if d < -0.05 else "\u2014"
            mag = f"{abs(d):.0f}" if float(abs(d)).is_integer() \
                else f"{abs(d):.1f}"
            if abs(d) > 0.05:
                txt = f"{glyph} {mag} pts"
                if r["norm"] is not None:
                    txt += f"\n{r['norm']:+.2f} Norm. Pts"
                else:
                    txt += "\nNorm. Pts n/a"
            else:
                txt = glyph
            box = dict(boxstyle="round,pad=0.34", facecolor=style["fill"],
                       edgecolor=style["edge"], linewidth=1.0)
            # Offset clears the taller bar AND its percentage label, so the
            # badge sits on its own line rather than crowding the numbers
            if horizontal:
                # Fixed column rather than tracking each bar's length, so the
                # badges line up instead of stepping in and out
                ax.text(117.0, pos[i], txt, ha="left", va="center",
                        fontsize=args.font_size - 1, fontweight="bold",
                        color=style["text"], bbox=box, zorder=5)
            else:
                top = max((r["base_pct"] or 0.0)
                          + _hi(r["base_pct"] or 0.0, r["base_std"]),
                          r["indiv_pct"]
                          + _hi(r["indiv_pct"], r["indiv_std"]))
                ax.text(idx[i], top + 10.0, txt, ha="center", va="bottom",
                        fontsize=args.font_size - 1, fontweight="bold",
                        color=style["text"], bbox=box, zorder=5,
                        linespacing=1.25)

    # Room above the bars for the two-line delta badges
    lim = 138 if args.show_delta else 118
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(names, fontsize=args.font_size,
                           fontweight="bold", fontfamily="monospace")
        ax.set_xlim(0, lim)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Win Rate (%)", fontsize=args.font_size + 3,
                      fontweight="bold")
        ax.xaxis.set_label_coords(50.0 / lim, -0.085)
        ax.grid(axis="x", color="#cccccc", linewidth=1.0, linestyle="--",
                zorder=0)
        ax.set_ylim(-0.75, n - 0.25)
    else:
        ax.set_xticks(idx)
        ax.set_xticklabels(names, fontsize=args.font_size, rotation=30,
                           ha="right", fontweight="bold",
                           fontfamily="monospace")
        ax.set_ylim(0, lim)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Win Rate (%)", fontsize=args.font_size + 3,
                      fontweight="bold")
        ax.yaxis.set_label_coords(-0.06, 50.0 / lim)
        ax.grid(axis="y", color="#cccccc", linewidth=1.0, linestyle="--",
                zorder=0)
        pad = 0.45 if args.show_delta else 0.0
        ax.set_xlim(-0.7 - pad, n - 0.3 + pad)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x" if horizontal else "y",
                   labelsize=args.font_size + 1)
    for lbl in (ax.get_xticklabels() if horizontal else ax.get_yticklabels()):
        lbl.set_fontweight("bold")
    for s in ("top", "right") + (("left",) if horizontal else ()):
        ax.spines[s].set_visible(False)

    handles = [Patch(facecolor=C_BASE, edgecolor="#222222",
                     label=args.label_baseline),
               Patch(facecolor=C_INDIV, edgecolor="#222222",
                     label=args.label_individual)]
    if args.show_delta and args.legend_loc == "best":
        # The badge rows occupy the top of the axes; "best" tends to pick
        # exactly that corner, so lift the legend above the plot instead
        anchor = (0.0, 1.01) if horizontal else (1.0, 1.01)
        corner = "lower left" if horizontal else "lower right"
        ax.legend(handles=handles, fontsize=args.font_size,
                  loc=corner, bbox_to_anchor=anchor,
                  ncol=2, framealpha=0.95, edgecolor="#cccccc")
    else:
        ax.legend(handles=handles, fontsize=args.font_size,
                  loc=args.legend_loc, framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Compare multi-opponent baseline win rates against "
                    "single-opponent evaluations")
    p.add_argument("--summary", required=True,
                   help="Baseline *_summary.csv (the multi-opponent run)")
    p.add_argument("--individual", nargs="+", required=True,
                   help="One or more CSVs from single-opponent evaluations")
    p.add_argument("--label-baseline", default="all opponents")
    p.add_argument("--label-individual", default="single opponent")
    p.add_argument("--sort", choices=["opponent", "baseline", "focused",
                                      "delta", "norm"], default="baseline")
    p.add_argument("--orientation", choices=["vertical", "horizontal"],
                   default="vertical",
                   help="vertical = paired columns (default, as sketched)")
    p.add_argument("--no-delta", dest="show_delta", action="store_false",
                   help="Hide the change badge above each pair")
    p.add_argument("--out", default=None)
    p.add_argument("--legend-loc", default="best",
                   help="Legend position. 'best' (default) avoids the bars; "
                        "pass a matplotlib loc string to pin it.")
    p.add_argument("--width", type=float, default=11.0)
    p.add_argument("--height", type=float, default=6.0)
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    rows, base_ctx, indiv_ctx = collect(args)
    print(f"\nPlotting {len(rows)} opponent(s)")
    for r in rows:
        bs = "" if r.get("base_std") is None else f"\u00b1{r['base_std']:.1f}"
        xs = "" if r.get("indiv_std") is None else f"\u00b1{r['indiv_std']:.1f}"
        b = "n/a" if r["base_pct"] is None else f"{r['base_pct']:.1f}{bs}%"
        d = "" if r["delta"] is None else f"  ({r['delta']:+.1f} pts)"
        nrm = "" if r["norm"] is None else f"  [norm {r['norm']:+.2f}]"
        print(f"  {r['name']:<18} {b:>12} -> {r['indiv_pct']:>6.1f}{xs}%"
              f"{d}{nrm}")

    fig = plot(rows, args)
    out = args.out or (os.path.splitext(args.summary)[0] + "_comparison.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
