"""
plot_standings.py

Plot round-robin tournament standings as a ranked points bar chart.

Bars are coloured on a diverging scale centred on the 50% win-rate mark
(points == games_played / 2), so participants that finished above an even
record are blue and those below are red. The scoring convention (win +1,
draw +0.5, loss +0) is stated in the header alongside the total number of
games and the map.

Takes the *_standings.csv written by tournament.py.

Usage:
    python plot_standings.py results/tournament_..._standings.csv
    python plot_standings.py results/..._standings.csv --highlight FeudalNet
    python plot_standings.py results/..._standings.csv --out fig/standings.pdf
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

# Points awarded per result — the microRTS competition convention
PTS_WIN, PTS_DRAW, PTS_LOSS = 1.0, 0.5, 0.0


def load_standings(path):
    """Read tournament.py's *_standings.csv. Returns (rows, context)."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError(f"{path} contains no rows")
    if "participant" not in raw[0]:
        raise ValueError(
            f"{path} does not look like a standings file "
            f"(no 'participant' column). Columns: {list(raw[0])}\n"
            f"Pass the *_standings.csv written by tournament.py.")

    rows, context = [], {}
    for r in raw:
        try:
            rows.append({
                "name":   r["participant"],
                "played": int(r["played"]),
                "wins":   int(r["wins"]),
                "draws":  int(r["draws"]),
                "losses": int(r["losses"]),
                "points": float(r["points"]),
            })
        except (KeyError, ValueError) as e:
            raise ValueError(f"Malformed row for {r.get('participant')}: {e}")
        for k in ("map", "seed"):
            if r.get(k):
                context[k] = r[k]

    # The points column must agree with W/D/L — catch corrupt files early
    bad = [(r["name"], r["points"],
            r["wins"] * PTS_WIN + r["draws"] * PTS_DRAW)
           for r in rows
           if abs(r["points"] - (r["wins"] * PTS_WIN
                                 + r["draws"] * PTS_DRAW)) > 1e-6]
    if bad:
        print("  WARNING: points disagree with W/D/L for: "
              + ", ".join(f"{b[0]} ({b[1]:g} vs {b[2]:g})" for b in bad))

    return rows, context


def plot(rows, context, args):
    rows = sorted(rows, key=lambda r: (-r["points"], r["name"]))
    names  = [r["name"] for r in rows]
    points = np.array([r["points"] for r in rows])
    n      = len(rows)

    # An even record scores half the games played. With a complete round robin
    # everyone plays the same number, so this is a single reference line.
    played = [r["played"] for r in rows]
    even   = float(np.median(played)) / 2.0
    uniform_games = len(set(played)) == 1

    # Diverging colours centred on that even-record mark
    hi = max(points.max(), even * 2) or 1.0
    norm = TwoSlopeNorm(vmin=0.0, vcenter=even, vmax=hi)
    cmap = matplotlib.colormaps["RdBu"]
    colors = [cmap(norm(p)) for p in points]

    fig, ax = plt.subplots(figsize=(args.width, max(3.0, 0.42 * n + 1.8)),
                           dpi=args.dpi)

    pos = np.arange(n)[::-1]                     # best at the top
    ax.barh(pos, points, height=0.78, color=colors,
            edgecolor="#222222", linewidth=0.9, zorder=3)

    # Ring the highlighted participant so the agent is unmistakable
    for i, nm in enumerate(names):
        if nm == args.highlight:
            ax.add_patch(Rectangle((0, pos[i] - 0.39), points[i], 0.78,
                                   fill=False, edgecolor=args.highlight_color,
                                   linewidth=2.6, zorder=5))

    # Point totals just past the end of each bar
    span = hi
    for i in range(n):
        ax.text(points[i] + 0.012 * span, pos[i], f"{points[i]:g}",
                ha="left", va="center", fontsize=args.font_size + 1,
                fontweight="bold", zorder=4)

    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=args.font_size, fontweight="bold",
                       fontfamily="monospace")
    # The axis extends past the longest bar to fit the point totals, so pin
    # the xlabel to the centre of the data range rather than the axes centre.
    x_pad = 1.10
    ax.set_xlabel("Total Points", fontsize=args.font_size + 3,
                  fontweight="bold")
    ax.xaxis.set_label_coords(0.5 / x_pad, -0.085)
    ax.set_xlim(0, span * x_pad)
    ax.set_ylim(-0.7, n - 0.3)

    ax.grid(axis="x", color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=args.font_size + 2)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    # The 50% win-rate reference
    if args.show_even_line and even > 0:
        ax.axvline(even, color="#999999", linewidth=1.6, linestyle="--",
                   zorder=2)
        ax.text(even + 0.008 * span, -0.55, "50% WR", ha="left", va="bottom",
                fontsize=args.font_size, style="italic", color="#888888",
                zorder=4)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Header: games played, map, and the scoring convention
    total_games = int(sum(played) / 2)            # each game involves two
    bits = [f"{total_games} Games"]
    map_name = args.map_name or (
        os.path.splitext(os.path.basename(context["map"]))[0]
        if context.get("map") else None)
    if map_name:
        bits.append(f"{map_name} Map")
    bits.append(f"Win = +{PTS_WIN:g}  |  Draw = +{PTS_DRAW:g}  "
                f"|  Loss = +{PTS_LOSS:g}")
    header = "   •   ".join(bits)
    if not uniform_games:
        header += "\n(games played vary by participant)"
    # Centred on the data range for the same reason as the xlabel: the axes
    # extend past the longest bar to fit the point totals.
    ax.set_title(header, fontsize=args.font_size + 3, style="italic",
                 fontweight="bold", color="#555555", pad=14,
                 x=0.5 / x_pad)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Plot tournament standings as a ranked points bar chart")
    p.add_argument("csv", help="*_standings.csv from tournament.py")
    p.add_argument("--out", default=None,
                   help="Output image (.png/.pdf/.svg). Default: next to the CSV")
    p.add_argument("--highlight", default="FeudalNet",
                   help="Participant to ring with a highlight border")
    p.add_argument("--highlight-color", default="#f5b800")
    p.add_argument("--map-name", default=None,
                   help="Override the map name shown in the header")
    p.add_argument("--no-even-line", dest="show_even_line",
                   action="store_false",
                   help="Hide the 50%% win-rate reference line")
    p.add_argument("--width", type=float, default=11.0)
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    rows, context = load_standings(args.csv)
    print(f"Loaded {len(rows)} participants from {args.csv}")

    fig = plot(rows, context, args)
    out = args.out or (os.path.splitext(args.csv)[0] + "_bars.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
