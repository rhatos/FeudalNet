"""
plot_steps_comparison.py

Compare episode lengths from a multi-opponent evaluation ("baseline") against
evaluations run against a single opponent each ("focused"), as paired bars with
standard-deviation error bars.

Companion to plot_winrate_comparison.py: same inputs, same layout, but the
measure is how long games took rather than how many were won. Episode length is
a decisiveness measure — shorter wins mean more dominant play, and longer losses
mean the agent survived further before being beaten.

    --outcome win    steps in games the agent WON  (default)
    --outcome loss   steps in games the agent LOST
    --outcome all    steps across every game
    --outcome draw   steps in drawn games

Only opponents present in the individual CSVs are plotted. Opponents with no
games of the requested outcome (a 100% win rate has no losses to measure) are
reported and skipped.

Both *_summary.csv and *_games.csv from evaluate_agent.py are accepted; from a
games CSV the mean and standard deviation are computed directly.

Usage:
    python plot_steps_comparison.py \\
        --summary results/eval_tqlgxayp_all_summary.csv \\
        --individual results/vs_izanagi_summary.csv \\
                     results/vs_droplet_summary.csv

    # Length of losses instead, sorted by the biggest change
    python plot_steps_comparison.py \\
        --summary results/base_summary.csv \\
        --individual results/vs_*.csv \\
        --outcome loss --sort delta
"""

import os
import csv
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_BASE = "#2166ac"   # blue — the multi-opponent baseline
C_INDIV = "#c1272d"  # red  — the focused, single-opponent runs

DELTA_STYLE = {
    "good": {"text": "#166534", "fill": "#e7f6ec", "edge": "#8fd0a6"},
    "bad":  {"text": "#9b1c1c", "fill": "#fdeceb", "edge": "#eaa39f"},
    "flat": {"text": "#555555", "fill": "#f2f2f2", "edge": "#d0d0d0"},
}

# Which direction counts as an improvement, per outcome.
#   win  — fewer steps to win is more dominant
#   loss — more steps before losing means it survived longer
#   all / draw — genuinely ambiguous, so left uncoloured
BETTER_BY_OUTCOME = {"win": "lower", "loss": "higher",
                     "all": "none", "draw": "none"}


def _stats_from_summary(path, outcome):
    """
    Read per-opponent step statistics from a *_summary.csv (either layout).

    Returns (rows, context); each row has name / mean / std / n.
    """
    with open(path, newline="") as f:
        raw = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not raw:
        raise ValueError(f"{path} is empty")

    header = [h.strip() for h in raw[0]]
    mean_key = f"steps_{outcome}_mean"
    std_key = f"steps_{outcome}_std"
    n_key = {"win": "wins", "loss": "losses", "draw": "draws",
             "all": "games"}[outcome]

    rows, context = [], {}

    # --- Vertical layout: 'metric' down the rows, opponents across the top ---
    if header[0].lower() == "metric":
        names = header[1:]
        table = {}
        for r in raw[1:]:
            key, vals = r[0].strip(), [c.strip() for c in r[1:]]
            if key in ("checkpoint", "map", "sides", "seed", "num_games"):
                context[key] = vals[0] if vals else ""
            else:
                table[key] = vals
        if mean_key not in table:
            raise ValueError(
                f"{path} has no '{mean_key}'. This summary predates the "
                f"episode-length statistics — re-run evaluate_agent.py, or "
                f"pass the matching *_games.csv instead.")

        def num(k, i):
            try:
                return float(table[k][i])
            except (KeyError, IndexError, ValueError):
                return None

        for i, nm in enumerate(names):
            if nm.upper() == "OVERALL":
                continue
            m = num(mean_key, i)
            if m is None:
                continue                      # no games of this outcome
            rows.append({"name": nm, "mean": m,
                         "std": num(std_key, i) or 0.0,
                         "n": int(num(n_key, i) or 0)})

    # --- Horizontal layout: one row per opponent ---
    elif "opponent" in header:
        # A games CSV also has an 'opponent' column, so reject it here rather
        # than returning an empty result — load_any then falls through to the
        # games parser, which computes the statistics from the raw matches.
        if mean_key not in header:
            raise ValueError(
                f"{path} has no '{mean_key}' column "
                f"(not a summary with step statistics)")
        for r in csv.DictReader(open(path, newline="")):
            nm = r.get("opponent", "")
            if not nm or nm.upper() == "OVERALL":
                continue
            try:
                m = float(r[mean_key])
            except (KeyError, ValueError):
                continue
            try:
                sd = float(r.get(std_key) or 0.0)
            except ValueError:
                sd = 0.0
            rows.append({"name": nm, "mean": m, "std": sd,
                         "n": int(float(r.get(n_key) or 0))})
            for k in ("checkpoint", "map", "sides"):
                if r.get(k):
                    context[k] = r[k]
    else:
        raise ValueError(
            f"{path} does not look like an evaluation summary "
            f"(first cell is {header[0]!r}, expected 'metric' or 'opponent').")

    return rows, context


def _stats_from_games(path, outcome):
    """Compute per-opponent step statistics directly from a *_games.csv."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw or "result" not in raw[0] or "steps" not in raw[0]:
        raise ValueError("not a games CSV")

    buckets, context = {}, {}
    for r in raw:
        nm = r.get("opponent")
        if not nm:
            continue
        res = (r.get("result") or "").upper()
        if outcome != "all" and res != outcome.upper():
            continue
        try:
            buckets.setdefault(nm, []).append(float(r["steps"]))
        except (TypeError, ValueError):
            continue
        for k in ("checkpoint", "map", "sides"):
            if r.get(k):
                context[k] = r[k]

    rows = []
    for nm, vals in buckets.items():
        a = np.asarray(vals, dtype=float)
        rows.append({"name": nm, "mean": float(a.mean()),
                     "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                     "n": a.size})
    return rows, context


def load_any(path, outcome):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    try:
        return _stats_from_summary(path, outcome)
    except Exception as summary_err:                         # noqa: BLE001
        try:
            return _stats_from_games(path, outcome)
        except Exception:                                    # noqa: BLE001
            raise ValueError(
                f"{path}: could not read step statistics.\n"
                f"  {summary_err}") from None


def collect(args):
    base_rows, base_ctx = load_any(args.summary, args.outcome)
    baseline = {r["name"]: r for r in base_rows}
    print(f"Baseline {args.summary}: {len(baseline)} opponent(s) with "
          f"{args.outcome} games")

    focused, indiv_ctx = {}, {}
    for path in args.individual:
        rows, ctx = load_any(path, args.outcome)
        indiv_ctx.update({k: v for k, v in ctx.items() if v})
        if not rows:
            print(f"  WARNING: {os.path.basename(path)} has no "
                  f"{args.outcome} games — skipped")
            continue
        for r in rows:
            if r["name"] in focused:
                print(f"  WARNING: {r['name']} appears in more than one "
                      f"individual CSV — keeping the later one")
            focused[r["name"]] = r
            print(f"  {os.path.basename(path)}: {r['name']} "
                  f"{r['mean']:.0f} ± {r['std']:.0f} steps "
                  f"over {r['n']} {args.outcome} game(s)")

    if not focused:
        raise ValueError(
            f"No opponent in the individual CSVs has any '{args.outcome}' "
            f"games. Try --outcome win, loss or all.")

    rows = []
    for name, f in focused.items():
        b = baseline.get(name)
        if b is None:
            print(f"  note: {name} has no {args.outcome} games in the "
                  f"baseline — bar shown without a comparison")
        rows.append({
            "name": name,
            "base_mean": b["mean"] if b else None,
            "base_std": b["std"] if b else 0.0,
            "base_n": b["n"] if b else 0,
            "indiv_mean": f["mean"], "indiv_std": f["std"], "indiv_n": f["n"],
            "delta": (f["mean"] - b["mean"]) if b else None,
        })

    missing = sorted(set(baseline) - set(focused))
    if missing:
        print(f"Excluded {len(missing)} baseline opponent(s) with no "
              f"individual CSV: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))

    key = {
        "opponent": lambda r: r["name"],
        "baseline": lambda r: (-(r["base_mean"] if r["base_mean"] is not None
                                 else -1), r["name"]),
        "focused":  lambda r: (-r["indiv_mean"], r["name"]),
        "delta":    lambda r: (-(r["delta"] if r["delta"] is not None
                                 else -1e9), r["name"]),
    }[args.sort]
    rows.sort(key=key)
    return rows, base_ctx, indiv_ctx


def plot(rows, args):
    n = len(rows)
    names = [r["name"] for r in rows]
    base = np.array([0.0 if r["base_mean"] is None else r["base_mean"]
                     for r in rows])
    base_sd = np.array([0.0 if r["base_mean"] is None else r["base_std"]
                        for r in rows])
    indiv = np.array([r["indiv_mean"] for r in rows])
    indiv_sd = np.array([r["indiv_std"] for r in rows])

    # Headroom for the error bars, value labels and change badges.
    # Badges share one row above every bar rather than tracking each pair,
    # so they line up and the band above the bars stays uncluttered.
    top = float(max((base + base_sd).max(), (indiv + indiv_sd).max()))
    badge_at = top * 1.13
    lim = (badge_at * 1.10) if args.show_delta else top * 1.20

    horizontal = args.orientation == "horizontal"
    if horizontal:
        size = max(3.2, 0.62 * n + 1.8)
        fig, ax = plt.subplots(figsize=(args.width, size), dpi=args.dpi)
    else:
        size = max(6.0, 0.95 * n + 2.4)
        fig, ax = plt.subplots(figsize=(size, args.height), dpi=args.dpi)

    idx = np.arange(n)
    w = 0.38
    common = dict(edgecolor="#222222", linewidth=1.0, zorder=3)
    ekw = dict(ecolor="#333333", capsize=4, elinewidth=1.2, capthick=1.2)

    if horizontal:
        pos = idx[::-1]
        ax.barh(pos + w / 2, base, height=w, color=C_BASE, xerr=base_sd,
                error_kw=ekw, label=args.label_baseline, **common)
        ax.barh(pos - w / 2, indiv, height=w, color=C_INDIV, xerr=indiv_sd,
                error_kw=ekw, label=args.label_individual, **common)
    else:
        ax.bar(idx - w / 2, base, width=w, color=C_BASE, yerr=base_sd,
               error_kw=ekw, label=args.label_baseline, **common)
        ax.bar(idx + w / 2, indiv, width=w, color=C_INDIV, yerr=indiv_sd,
               error_kw=ekw, label=args.label_individual, **common)

    better = (BETTER_BY_OUTCOME[args.outcome] if args.better == "auto"
              else args.better)

    for i, r in enumerate(rows):
        entries = []
        if r["base_mean"] is not None:
            entries.append((r["base_mean"], r["base_std"], +1))
        entries.append((r["indiv_mean"], r["indiv_std"], -1))

        for val, sd, side in entries:
            lbl = f"{val:,.0f}"
            if horizontal:
                ax.text(val + sd + lim * 0.015, pos[i] + side * w / 2, lbl,
                        ha="left", va="center", fontsize=args.font_size - 1,
                        fontweight="bold", zorder=4)
            else:
                ax.text(idx[i] - side * w / 2, val + sd + lim * 0.015, lbl,
                        ha="center", va="bottom",
                        fontsize=args.font_size - 1, fontweight="bold",
                        zorder=4)

        if args.show_delta and r["delta"] is not None:
            d = r["delta"]
            negligible = abs(d) < max(1.0, 0.005 * lim)
            if negligible or better == "none":
                kind = "flat" if negligible else "flat"
            else:
                improved = (d < 0) if better == "lower" else (d > 0)
                kind = "good" if improved else "bad"
            style = DELTA_STYLE[kind]
            glyph = "\u2014" if negligible else ("\u25b2" if d > 0 else "\u25bc")
            txt = glyph if negligible else f"{glyph} {abs(d):,.0f}"
            box = dict(boxstyle="round,pad=0.34", facecolor=style["fill"],
                       edgecolor=style["edge"], linewidth=1.0)
            if horizontal:
                ax.text(lim * 0.955, pos[i], txt, ha="right", va="center",
                        fontsize=args.font_size - 1, fontweight="bold",
                        color=style["text"], bbox=box, zorder=5)
            else:
                ax.text(idx[i], badge_at, txt, ha="center", va="center",
                        fontsize=args.font_size - 1, fontweight="bold",
                        color=style["text"], bbox=box, zorder=5)

    axis_label = {
        "win": "Steps to Win", "loss": "Steps to Loss",
        "draw": "Steps to Draw", "all": "Steps per Game",
    }[args.outcome]

    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(names, fontsize=args.font_size,
                           fontweight="bold", fontfamily="monospace")
        ax.set_xlim(0, lim)
        ax.set_xlabel(axis_label, fontsize=args.font_size + 3,
                      fontweight="bold")
        ax.xaxis.set_label_coords(0.5 * top / lim, -0.085)
        ax.grid(axis="x", color="#cccccc", linewidth=1.0, linestyle="--",
                zorder=0)
        ax.set_ylim(-0.75, n - 0.25)
    else:
        ax.set_xticks(idx)
        ax.set_xticklabels(names, fontsize=args.font_size, rotation=30,
                           ha="right", fontweight="bold",
                           fontfamily="monospace")
        ax.set_ylim(0, lim)
        ax.set_ylabel(axis_label, fontsize=args.font_size + 3,
                      fontweight="bold")
        # Clear of the tick labels, which are 4 digits wide for step counts
        ax.yaxis.set_label_coords(-0.135, 0.5 * top / lim)
        ax.grid(axis="y", color="#cccccc", linewidth=1.0, linestyle="--",
                zorder=0)
        ax.set_xlim(-0.7, n - 0.3)
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
    note = "error bars: ±1 s.d." if args.show_std_note else None
    if args.legend_loc == "outside":
        # Bars run from zero, so an inline legend has nowhere safe to sit
        ax.legend(handles=handles, fontsize=args.font_size,
                  loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
                  mode="expand", ncol=3, frameon=False, borderaxespad=0.0,
                  title=note, title_fontsize=args.font_size - 2)
    else:
        ax.legend(handles=handles, fontsize=args.font_size,
                  loc=args.legend_loc, framealpha=0.95, edgecolor="#cccccc",
                  title=note, title_fontsize=args.font_size - 2)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Compare episode lengths between a multi-opponent "
                    "baseline and single-opponent evaluations")
    p.add_argument("--summary", required=True,
                   help="Baseline *_summary.csv (the multi-opponent run)")
    p.add_argument("--individual", nargs="+", required=True,
                   help="One or more CSVs from single-opponent evaluations")
    p.add_argument("--outcome", choices=["win", "loss", "all", "draw"],
                   default="win",
                   help="Which games to measure (default: win)")
    p.add_argument("--better", choices=["auto", "lower", "higher", "none"],
                   default="auto",
                   help="Which direction the badges treat as an improvement. "
                        "auto: fewer steps for wins, more steps for losses, "
                        "neutral otherwise.")
    p.add_argument("--label-baseline", default="all opponents")
    p.add_argument("--label-individual", default="single opponent")
    p.add_argument("--sort", choices=["opponent", "baseline", "focused",
                                      "delta"], default="baseline")
    p.add_argument("--orientation", choices=["vertical", "horizontal"],
                   default="vertical")
    p.add_argument("--no-delta", dest="show_delta", action="store_false",
                   help="Hide the change badge above each pair")
    p.add_argument("--no-std-note", dest="show_std_note",
                   action="store_false",
                   help="Drop the '±1 s.d.' note from the legend")
    p.add_argument("--out", default=None)
    p.add_argument("--legend-loc", default="outside",
                   help="'outside' (default) places the legend above the "
                        "axes; any matplotlib loc string pins it inside")
    p.add_argument("--width", type=float, default=11.0)
    p.add_argument("--height", type=float, default=6.0)
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    rows, base_ctx, indiv_ctx = collect(args)
    print(f"\nPlotting {len(rows)} opponent(s) — steps to {args.outcome}")
    for r in rows:
        b = ("n/a" if r["base_mean"] is None
             else f"{r['base_mean']:,.0f} ± {r['base_std']:,.0f}")
        d = "" if r["delta"] is None else f"   ({r['delta']:+,.0f} steps)"
        print(f"  {r['name']:<18} {b:>16} -> "
              f"{r['indiv_mean']:,.0f} ± {r['indiv_std']:,.0f}{d}")

    fig = plot(rows, args)
    out = args.out or (os.path.splitext(args.summary)[0]
                       + f"_steps_{args.outcome}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
