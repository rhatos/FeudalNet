"""
plot_reward_curve.py

Fetch a run's history from wandb and plot any logged metric as a training
curve ending at a specified checkpoint, with the raw series drawn faintly
behind a time-weighted EMA (the style of smoothing wandb's UI applies).

The end point can be given either as a checkpoint file — the step is read from
the checkpoint's own `step` field, so the curve ends exactly where that model
was saved — or as an explicit --max-step.

Any metric can be plotted, several at once, and a metric may be given for
multiple runs to compare them on one axis.

Usage:
    # What can I plot?
    python plot_reward_curve.py --run-id tqlgxayp --list-metrics

    # Default metric, ending at a checkpoint
    python plot_reward_curve.py --run-id tqlgxayp \
        --checkpoint models/tqlgxayp_latest.pt

    # Any other logged metric
    python plot_reward_curve.py --run-id tqlgxayp --metric worker/entropy

    # Several metrics — one file each, or --overlay onto shared axes
    python plot_reward_curve.py --run-id tqlgxayp \
        --metric loss/value_worker loss/value_manager --overlay

    # Wildcards work too
    python plot_reward_curve.py --run-id tqlgxayp --metric "charts/episode_reward/*"

    # Compare runs
    python plot_reward_curve.py --run-id runA runB --metric eval/winrate --overlay

    # Cache once, replot offline
    python plot_reward_curve.py --run-id tqlgxayp --metric "*" --csv-out hist.csv
    python plot_reward_curve.py --from-csv hist.csv --metric worker/entropy
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smoothing_to_half_life(x, smoothing, base_frac):
    """
    Map a 0-1 smoothing slider onto a half-life measured in x units.

    The EMA is time-weighted, so its time constant lives in the same units as
    the x axis (training steps) rather than "number of samples". Tying it to
    the span of the run makes the visual amount of smoothing independent of
    how often the metric happened to be logged:

        half_life = base_frac * (s / (1 - s)) * x_range

    The odds-ratio term s/(1-s) gives a smooth, monotonic response across the
    whole slider: 0.5 -> base_frac of the run, 0.9 -> 9x that, and values near
    1 approach a flat line. base_frac defaults to 0.01, chosen so smoothing
    0.5 reproduces roughly what wandb's UI shows at the same setting. wandb
    does not publish its exact constant, so treat this as a close match rather
    than an exact reimplementation; use --half-life for precise control.
    """
    s = float(np.clip(smoothing, 0.0, 0.999))
    if s <= 0.0:
        return 0.0
    x_range = float(np.nanmax(x) - np.nanmin(x)) if len(x) > 1 else 1.0
    if x_range <= 0:
        x_range = 1.0
    return base_frac * (s / (1.0 - s)) * x_range


def time_weighted_ema(x, y, half_life):
    """
    Time-weighted exponential moving average with debiasing.

    A plain EMA assumes evenly spaced samples. Training histories are not
    evenly spaced (logging cadence varies, runs get resumed), so the weight
    of the previous value decays by the actual x-gap:

        decay_i = 0.5 ** (dx_i / half_life)

    i.e. the running average loses half its influence every `half_life` steps
    of x, regardless of how many samples fall in that interval. Dividing by
    the accumulated weight debiases the start of the series, so the curve
    begins at the first observation instead of being dragged toward zero.

    half_life <= 0 returns the raw series unchanged.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) == 0 or half_life <= 0:
        return y.copy()

    out = np.empty_like(y)
    last = 0.0          # running weighted sum
    accum = 0.0         # running weight, for debiasing
    prev_x = x[0]
    seeded = False
    for i, (xi, yi) in enumerate(zip(x, y)):
        if not np.isfinite(yi):
            out[i] = np.nan
            continue
        if not seeded:
            # Seed with the first finite sample. Without this the opening
            # gap is zero, so decay == 1 and that sample would carry no
            # weight into the running average at all.
            last, accum, seeded = yi, 1.0, True
        else:
            gap = max(xi - prev_x, 0.0)
            decay = 0.5 ** (gap / half_life)
            last = last * decay + (1.0 - decay) * yi
            accum = accum * decay + (1.0 - decay)
        out[i] = last / accum if accum > 0 else yi
        prev_x = xi
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def step_from_checkpoint(path):
    """Read the training step a checkpoint was saved at."""
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "step" not in ckpt:
        raise KeyError(
            f"{path} has no 'step' field (keys: {list(ckpt)[:8]}). "
            f"Pass --max-step instead.")
    return int(ckpt["step"])


def list_metrics(args):
    """Print every numeric key logged by the run, so you know what to ask for."""
    import wandb
    api = wandb.Api()
    path = (f"{args.entity}/{args.project}/{args.run_id[0]}" if args.entity
            else f"{args.project}/{args.run_id[0]}")
    run = api.run(path)
    keys = sorted(k for k in run.summary.keys() if not k.startswith("_"))
    print(f"{run.name} ({args.run_id[0]}) logs {len(keys)} keys:\n")
    groups = {}
    for k in keys:
        groups.setdefault(k.split("/")[0] if "/" in k else "(top level)",
                          []).append(k)
    for group, ks in groups.items():
        print(f"  {group}")
        for k in ks:
            print(f"      {k}")
    print("\nPass any of these to --metric (wildcards allowed, e.g. 'loss/*').")


def expand_metrics(patterns, available):
    """Resolve --metric patterns (exact names or globs) against logged keys."""
    import fnmatch
    out, seen = [], set()
    for pat in patterns:
        hits = ([pat] if pat in available
                else sorted(k for k in available if fnmatch.fnmatch(k, pat)))
        if not hits:
            near = sorted(k for k in available
                          if pat.split("/")[-1].lower() in k.lower())[:6]
            raise ValueError(
                f"No logged metric matches {pat!r}.\n"
                + (f"Did you mean: {', '.join(near)}?\n" if near else "")
                + "Run with --list-metrics to see everything available.")
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def fetch_history(args, metrics):
    """
    Pull history for one or more metrics from one or more runs.

    scan_history streams every logged row rather than the downsampled view
    history() returns, so raw curves keep their true spikiness. Metrics are
    logged at different cadences, so each series is collected independently
    and keeps only the rows where that metric is present.

    Returns {(run_label, metric): (x, y)}.
    """
    import wandb
    api = wandb.Api()
    series = {}

    for rid in args.run_id:
        path = (f"{args.entity}/{args.project}/{rid}" if args.entity
                else f"{args.project}/{rid}")
        print(f"Fetching {path} ...")
        run = api.run(path)
        print(f"  name: {run.name}   state: {run.state}")

        available = set(k for k in run.summary.keys() if not k.startswith("_"))
        wanted = expand_metrics(metrics, available) if available else metrics
        if len(wanted) > 1:
            print(f"  metrics: {', '.join(wanted)}")

        cols = {m: ([], []) for m in wanted}
        for row in run.scan_history(keys=None, page_size=10000):
            xv = row.get(args.x_key)
            if xv is None:
                continue
            for m in wanted:
                yv = row.get(m)
                if yv is None:
                    continue
                try:
                    fy = float(yv)
                except (TypeError, ValueError):
                    continue
                cols[m][0].append(float(xv))
                cols[m][1].append(fy)

        label = run.name if len(args.run_id) > 1 else None
        for m, (xs, ys) in cols.items():
            if not xs:
                print(f"  WARNING: no rows with both "
                      f"'{args.x_key}' and '{m}' — skipping")
                continue
            order = np.argsort(xs)
            series[(label or run.name, m)] = (np.asarray(xs)[order],
                                              np.asarray(ys)[order])

    if not series:
        raise ValueError(
            f"Nothing to plot. Check --x-key '{args.x_key}' and --metric, "
            f"or run with --list-metrics.")
    return series


def load_csv(path, args, metrics):
    """
    Load a cached history CSV. Any column other than the x key is a metric,
    so a cache written with --metric '*' can be replotted for any of them.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")

    cols = [c for c in rows[0].keys() if c and c != args.x_key]
    if args.x_key not in rows[0]:
        raise ValueError(
            f"{path} has no '{args.x_key}' column (columns: {list(rows[0])}).")

    wanted = expand_metrics(metrics, set(cols))
    series = {}
    for m in wanted:
        xs, ys = [], []
        for r in rows:
            xv, yv = r.get(args.x_key, ""), r.get(m, "")
            if xv == "" or yv == "" or yv is None:
                continue
            try:
                xs.append(float(xv)); ys.append(float(yv))
            except ValueError:
                continue
        if not xs:
            print(f"  WARNING: column '{m}' has no usable values — skipping")
            continue
        order = np.argsort(xs)
        series[(os.path.basename(path), m)] = (np.asarray(xs)[order],
                                               np.asarray(ys)[order])
    if not series:
        raise ValueError(f"No usable series for {metrics} in {path}")
    return series


def write_csv(series, path, x_key):
    """
    Cache fetched series to one CSV, aligned on the x key. Metrics logged at
    different cadences leave blanks rather than being force-interpolated.
    """
    metrics = [m for (_, m) in series]
    table = {}
    for (_, m), (xs, ys) in series.items():
        for xv, yv in zip(xs, ys):
            table.setdefault(xv, {})[m] = yv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([x_key] + metrics)
        for xv in sorted(table):
            w.writerow([xv] + [table[xv].get(m, "") for m in metrics])
    print(f"  history cached -> {path} "
          f"({len(table):,} rows x {len(metrics)} metrics)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def thousands(v, _pos):
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:g}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:g}k"
    return f"{v:g}"


# Colour cycle for overlays — distinguishable and colour-blind friendly
PALETTE = ["#2166ac", "#d6604d", "#1a9850", "#762a83", "#e08214",
           "#01665e", "#c51b7d", "#4d4d4d"]


def plot(series, smoothed, args):
    """
    series   : {(label, metric): (x, y)}
    smoothed : {(label, metric): (y_smooth, half_life)}
    """
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    keys = list(series)
    single = len(keys) == 1

    for i, key in enumerate(keys):
        label, metric = key
        x, y = series[key]
        ys, half_life = smoothed[key]
        color = args.color if (single and args.color) else PALETTE[i % len(PALETTE)]

        ax.plot(x, y, color=color, alpha=args.raw_alpha,
                linewidth=args.raw_width, zorder=2, solid_joinstyle="round",
                label="raw" if single else None)

        if single:
            leg = (f"EMA, half-life {half_life:,.0f} steps"
                   if args.half_life is not None
                   else f"EMA, smoothing {args.smoothing:g}")
        else:
            # With several series the metric (and run, when comparing runs)
            # identifies the line; the smoothing is noted once in the title.
            leg = metric if len(set(l for l, _ in keys)) == 1 else \
                f"{label} — {metric}"
        ax.plot(x, ys, color=color, linewidth=args.line_width, zorder=3,
                solid_capstyle="round", label=leg)

    ax.set_xlabel(args.x_label or args.x_key.replace("_", " "),
                  fontsize=args.font_size + 2, fontweight="bold")
    ylab = args.y_label
    if ylab is None:
        ylab = keys[0][1].split("/")[-1] if single else "value"
    ax.set_ylabel(ylab, fontsize=args.font_size + 2, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.tick_params(labelsize=args.font_size)
    ax.grid(color="#dddddd", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    title = args.title
    if title is None and not single:
        title = f"EMA smoothing {args.smoothing:g}"
    if title:
        ax.set_title(title, fontsize=args.font_size + 3, style="italic",
                     fontweight="bold", color="#555555", pad=12)

    if not args.legend:
        fig.tight_layout()
        return fig

    leg = ax.legend(fontsize=args.font_size, loc=args.legend_loc,
                    framealpha=0.92, edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.8)
    if single:
        # At legend scale the raw series is nearly invisible at its plotted
        # alpha, but matching the EMA exactly would make the two swatches
        # indistinguishable. Lift it part-way and keep it thinner.
        raw_handle, ema_handle = leg.legend_handles[0], leg.legend_handles[1]
        raw_handle.set_alpha(min(1.0, max(args.raw_alpha * 2.5, 0.45)))
        raw_handle.set_linewidth(1.2)
        ema_handle.set_alpha(1.0)
        ema_handle.set_linewidth(args.line_width)
    else:
        for h in leg.legend_handles:
            h.set_alpha(1.0)
            h.set_linewidth(args.line_width)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def safe_name(metric):
    return metric.replace("/", "_").replace("*", "all").replace(" ", "_")


def main():
    p = argparse.ArgumentParser(
        description="Plot wandb training curves up to a checkpoint")
    src = p.add_argument_group("source")
    src.add_argument("--run-id", nargs="+", default=None,
                     help="One or more wandb run ids. Several runs plot the "
                          "same metric for each, for comparison.")
    src.add_argument("--entity", default=None)
    src.add_argument("--project", default="fun-microrts")
    src.add_argument("--from-csv", default=None,
                     help="Plot from a cached history CSV instead of wandb")
    src.add_argument("--csv-out", default=None,
                     help="Also write the fetched history to this CSV")
    src.add_argument("--list-metrics", action="store_true",
                     help="List every metric the run logs, then exit")

    cut = p.add_argument_group("end point")
    cut.add_argument("--checkpoint", default=None,
                     help="End the curve at the step stored in this .pt file")
    cut.add_argument("--max-step", type=float, default=None,
                     help="End the curve at this step (overrides --checkpoint)")

    p.add_argument("--metric", nargs="+", default=["episode/total_reward"],
                   help="One or more metrics. Globs allowed, e.g. 'loss/*'. "
                        "Default: episode/total_reward")
    p.add_argument("--overlay", action="store_true",
                   help="Draw all metrics on one axes instead of one file each")
    p.add_argument("--x-key", default="global_step")
    p.add_argument("--smoothing", type=float, default=0.5,
                   help="Time-weighted EMA smoothing, 0-1 (default 0.5)")
    p.add_argument("--half-life", type=float, default=None,
                   help="Override the EMA half-life directly, in x units "
                        "(steps). Takes precedence over --smoothing.")
    p.add_argument("--smoothing-base", type=float, default=0.01,
                   help="Fraction of the run span that smoothing=0.5 "
                        "corresponds to (default 0.01)")

    style = p.add_argument_group("style")
    style.add_argument("--out", default=None,
                       help="Output path. With several metrics and no "
                            "--overlay this is used as a prefix.")
    style.add_argument("--title", default=None,
                       help="Optional title (omitted by default)")
    style.add_argument("--legend-loc", default="lower right",
                       help="Legend position (matplotlib loc string)")
    style.add_argument("--no-legend", dest="legend", action="store_false",
                       help="Hide the legend")
    style.add_argument("--x-label", default="Step")
    style.add_argument("--y-label", default=None)
    style.add_argument("--color", default="#2166ac",
                       help="Line colour for single-series plots; overlays "
                            "use a built-in palette")
    style.add_argument("--raw-alpha", type=float, default=0.22)
    style.add_argument("--raw-width", type=float, default=0.8)
    style.add_argument("--line-width", type=float, default=2.2)
    style.add_argument("--width", type=float, default=11.0)
    style.add_argument("--height", type=float, default=5.0)
    style.add_argument("--font-size", type=int, default=11)
    style.add_argument("--dpi", type=int, default=200)

    args = p.parse_args()

    if args.list_metrics:
        if not args.run_id:
            p.error("--list-metrics needs --run-id")
        list_metrics(args)
        return

    if not args.from_csv and not args.run_id:
        p.error("give --run-id (to fetch from wandb) or --from-csv")

    # --- Load ---
    if args.from_csv:
        series = load_csv(args.from_csv, args, args.metric)
    else:
        series = fetch_history(args, args.metric)
    for (label, metric), (x, _) in series.items():
        print(f"  {label} / {metric}: {len(x):,} points, "
              f"steps {x.min():,.0f} to {x.max():,.0f}")

    if args.csv_out and not args.from_csv:
        write_csv(series, args.csv_out, args.x_key)

    # --- Truncate at the checkpoint ---
    end = args.max_step
    if end is None and args.checkpoint:
        end = step_from_checkpoint(args.checkpoint)
        print(f"  checkpoint {os.path.basename(args.checkpoint)} "
              f"was saved at step {end:,}")

    if end is not None:
        trimmed = {}
        for key, (x, y) in series.items():
            keep = x <= end
            if keep.sum() == 0:
                print(f"  WARNING: {key[1]} has no data at or before "
                      f"step {end:,.0f} — skipping")
                continue
            trimmed[key] = (x[keep], y[keep])
            print(f"  {key[1]}: truncated at {end:,.0f} -> "
                  f"kept {int(keep.sum()):,}, dropped {int((~keep).sum()):,}")
        if not trimmed:
            raise ValueError(f"No series has data at or before step {end:,.0f}")
        series = trimmed

    # --- Smooth ---
    smoothed = {}
    for key, (x, y) in series.items():
        if args.half_life is not None:
            hl = args.half_life
        else:
            hl = smoothing_to_half_life(x, args.smoothing, args.smoothing_base)
        smoothed[key] = (time_weighted_ema(x, y, hl), hl)
    any_hl = next(iter(smoothed.values()))[1]
    print(f"  EMA half-life: {any_hl:,.0f} steps"
          + ("" if args.half_life is not None
             else f" (from smoothing {args.smoothing:g})"))

    # --- Plot: one figure, or one per metric ---
    single_run = len(set(l for l, _ in series)) == 1
    if args.overlay or len(series) == 1:
        fig = plot(series, smoothed, args)
        out = args.out or (
            f"{(args.run_id[0] if args.run_id else 'curve')}_"
            f"{safe_name(args.metric[0]) if len(series) == 1 else 'overlay'}.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")
    else:
        base, ext = os.path.splitext(args.out or "")
        ext = ext or ".png"
        for key in series:
            one_s = {key: series[key]}
            one_m = {key: smoothed[key]}
            fig = plot(one_s, one_m, args)
            tag = safe_name(key[1]) if single_run else \
                f"{safe_name(key[0])}_{safe_name(key[1])}"
            out = (f"{base}_{tag}{ext}" if base
                   else f"{args.run_id[0] if args.run_id else 'curve'}_{tag}{ext}")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
