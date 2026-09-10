"""
plot_tournament.py

Render a round-robin cross-table (from tournament.py) as a heatmap.

Each cell is the row player's result against the column player, coloured on a
diverging scale centred on an even record. Saturated cells (a clean sweep in
either direction) are left unannotated so the contested matchups stand out.
The diagonal is grey with a dash, and unplayed pairings are blank.

Accepts either matrix written by tournament.py; the metric is inferred from
the filename:
    <base>_matrix.csv          points   (win 1.0, draw 0.5, loss 0.0)
    <base>_matrix_winrate.csv  win rate %

Usage:
    python plot_tournament.py results/tournament_..._matrix.csv
    python plot_tournament.py results/..._matrix.csv --out fig/xtab.pdf
    python plot_tournament.py results/..._matrix.csv --highlight coacAI
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle


def load_matrix(path):
    """
    Read a tournament matrix CSV.

    Returns (labels, values, kind) where labels are participants in the file's
    order (tournament.py already sorts them by final standing), values is an
    (n, n) array with NaN for the diagonal and for unplayed pairings, and kind
    is 'points' or 'winrate' (inferred from the filename).
    """
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [h.strip() for h in rows[0]]
    if header[0].lower() != "vs":
        raise ValueError(
            f"{path} does not look like a tournament matrix "
            f"(expected first cell 'vs', got {header[0]!r}).\n"
            f"Pass the *_matrix.csv or *_matrix_winrate.csv from tournament.py.")

    kind = "winrate" if "winrate" in os.path.basename(path).lower() else "points"

    has_total = header[-1].strip().upper() == "TOTAL"
    labels = [h.strip() for h in (header[1:-1] if has_total else header[1:])]
    n = len(labels)

    values = np.full((n, n), np.nan)
    for r in rows[1:]:
        name = r[0].strip()
        if name not in labels:
            continue
        i = labels.index(name)
        cells = r[1:-1] if has_total else r[1:]
        for j, cell in enumerate(cells[:n]):
            cell = cell.strip()
            if cell in ("", "-"):
                continue                       # diagonal or unplayed
            try:
                values[i, j] = float(cell)
            except ValueError:
                pass

    return labels, values, kind


def scale_for(values, kind):
    """
    Work out the colour scale and the 'clean sweep' bounds.

    For win rate the scale is fixed at 0-100 with an even record at 50.
    For points the maximum is the games played per pairing, which is recovered
    from the data: the two cells of a pairing always sum to the games played
    between those two participants.
    """
    if kind == "winrate":
        return 0.0, 50.0, 100.0, "Win-Rate (%)"

    n = values.shape[0]
    pair_sums = [values[i, j] + values[j, i]
                 for i in range(n) for j in range(i + 1, n)
                 if np.isfinite(values[i, j]) and np.isfinite(values[j, i])]
    if pair_sums:
        games = float(max(pair_sums))
    else:
        games = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    games = max(games, 1e-9)
    return 0.0, games / 2.0, games, "Points"


def lightened_cmap(name, amount):
    """
    Return `name` with every colour blended toward white by `amount`
    (0 = unchanged, 1 = pure white). Keeps the hue ordering but takes the
    edge off the saturated extremes so annotations sit comfortably on top.
    """
    base = matplotlib.colormaps[name]
    xs = np.linspace(0, 1, 256)
    cols = base(xs)
    cols[:, :3] = cols[:, :3] * (1.0 - amount) + amount      # blend to white
    return LinearSegmentedColormap.from_list(f"{name}_light", cols)


def plot(labels, values, kind, args):
    n = len(labels)

    vmin, vmid, vmax, cbar_label = scale_for(values, kind)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    cmap = lightened_cmap("RdBu", args.lighten)
    cmap.set_bad("#ffffff")                    # unplayed pairings stay white

    cell = args.cell_size
    fig, ax = plt.subplots(figsize=(cell * n + 2.6, cell * n + 1.4),
                           dpi=args.dpi)

    # Cells are drawn as individual rounded tiles rather than a solid image,
    # so each sits in its own patch of white with a gap around it.
    side = 1.0 - args.cell_gap
    off  = side / 2.0
    for i in range(n):
        for j in range(n):
            if i == j:
                face = "#e2e2e2"               # never plays itself
            elif np.isfinite(values[i, j]):
                face = cmap(norm(values[i, j]))
            else:
                continue                       # unplayed: leave blank
            ax.add_patch(Rectangle((j - off, i - off), side, side,
                                   facecolor=face, edgecolor="none",
                                   zorder=3))
        ax.text(i, i, "—", ha="center", va="center", zorder=4,
                fontsize=args.font_size + 1, color="#888888")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)                 # origin at top, like imshow
    ax.set_aspect("equal")

    # Annotate only the contested cells. A clean sweep in either direction
    # (0 or the maximum) is unambiguous from the saturated colour alone, so
    # those are left unlabelled and the interesting matchups stand out.
    # Black text with a white outline stays legible on every colour in the map.
    outline = [path_effects.withStroke(linewidth=2.6, foreground="white")]
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            if i == j or not np.isfinite(v):
                continue
            if not args.annotate_all and (v <= vmin or v >= vmax):
                continue
            ax.text(j, i, f"{v:g}", ha="center", va="center",
                    fontsize=args.font_size, fontweight="bold", zorder=5,
                    color="black", path_effects=outline)

    # Ring the highlighted participant's row and column
    if args.highlight in labels:
        k = labels.index(args.highlight)
        ax.add_patch(Rectangle((-.5, k - .5), n, 1, fill=False, zorder=6,
                               edgecolor=args.highlight_color, linewidth=2.8))
        ax.add_patch(Rectangle((k - .5, -.5), 1, n, fill=False, zorder=6,
                               edgecolor=args.highlight_color, linewidth=2.8))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=args.font_size + 1, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=args.font_size + 1, fontweight="bold")
    ax.tick_params(length=0)

    for s in ax.spines.values():
        s.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label(cbar_label, fontsize=args.font_size + 3,
                   fontweight="bold", rotation=270, labelpad=22)
    cbar.ax.tick_params(labelsize=args.font_size + 2)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cbar.outline.set_visible(False)

    if args.title:
        ax.set_title(args.title, fontsize=args.font_size + 4, style="italic",
                     fontweight="bold", color="#555555", pad=14)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Plot a tournament win-rate cross-table as a heatmap")
    p.add_argument("csv",
                   help="*_matrix.csv (points) or *_matrix_winrate.csv "
                        "from tournament.py")
    p.add_argument("--out", default=None,
                   help="Output image (.png/.pdf/.svg). Default: next to the CSV")
    p.add_argument("--title", default=None,
                   help="Optional header line (omitted by default)")
    p.add_argument("--highlight", default="FeudalNet",
                   help="Participant whose row and column are ringed")
    p.add_argument("--highlight-color", default="#f5b800")
    p.add_argument("--annotate-all", action="store_true",
                   help="Also label clean-sweep cells (0 and the maximum), "
                        "which are hidden by default since the saturated "
                        "colour already says it")
    p.add_argument("--cell-gap", type=float, default=0.06,
                   help="White gap around each cell, as a fraction of the "
                        "cell (0 = tiles touch; default 0.06)")
    p.add_argument("--lighten", type=float, default=0.10,
                   help="Blend the colours toward white by this fraction "
                        "(0 = full-strength RdBu; default 0.10)")
    p.add_argument("--cell-size", type=float, default=0.60,
                   help="Inches per cell (raise for very large tournaments)")
    p.add_argument("--font-size", type=int, default=10)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    labels, values, kind = load_matrix(args.csv)
    played = int(np.isfinite(values).sum())
    _, _, vmax, cbar_label = scale_for(values, kind)
    print(f"Loaded {len(labels)} participants from {args.csv}")
    print(f"  metric: {cbar_label}  (scale 0 to {vmax:g})")
    print(f"  {played} of {len(labels) * (len(labels) - 1)} "
          f"directed pairings have data")

    fig = plot(labels, values, kind, args)
    out = args.out or (os.path.splitext(args.csv)[0] + ".png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
