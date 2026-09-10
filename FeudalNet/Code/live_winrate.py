"""
live_winrate.py

A live-updating win-rate bar chart for evaluation runs.

Opens an interactive matplotlib window that redraws as games finish, so a long
sweep across many opponents can be watched rather than waited on. Styled to
match plot_evaluation.py, so the live view and the final figure look the same.

Designed to be harmless when it cannot work: if there is no display, or no
interactive backend, or matplotlib is missing entirely, the chart disables
itself with one warning and every method becomes a no-op. Evaluation is never
interrupted because a window could not be opened.

Usage:
    chart = LiveWinrateChart(["coacAI", "mayari"], games_per_opponent=50)
    ...
    chart.set_repeat(r)                # once per repeat, when using --repeats
    chart.record("coacAI", "WIN")      # after each game
    ...
    chart.finish(save_path="live.png") # at the end

With repeats, each opponent's bar gains an error bar and a "±" label: the
sample standard deviation of the per-repeat win rates, computed over
COMPLETED repeats only (a half-played repeat would only add noise). Since
evaluation is sequential, "completed" means any (opponent, repeat) bucket
other than the one currently receiving games; finish() finalises the last
one, so the final chart matches the win_pct_std written to the summary CSV.
"""

import os
import time

# Backends worth trying, in order of preference. TkAgg is the most commonly
# present on Linux desktops; Qt variants cover the rest.
_BACKENDS = ["TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg"]

PTS = {"WIN": 1, "DRAW": 0, "LOSS": 0}    # only wins count toward win rate


class LiveWinrateChart:
    """
    Tracks per-opponent results and draws them as a horizontal bar chart.

    The chart owns the counting, so callers just report each result as it
    happens: this keeps it correct when games for one opponent arrive in
    several batches (for example when --sides both splits them).
    """

    def __init__(self, opponents, games_per_opponent=None, title=None,
                 min_redraw_interval=0.25, sort=True, quiet=False):
        self.opponents = list(opponents)
        self.target = games_per_opponent
        self.title = title
        self.min_interval = min_redraw_interval
        self.sort = sort
        # counts[opponent][repeat] -> {"WIN": w, "LOSS": l, "DRAW": d}
        self.counts = {o: {} for o in self.opponents}
        self._repeat = 1
        self._current = None       # (opponent, repeat) now receiving games

        self.enabled = False
        self._last_draw = 0.0
        self._fig = None
        self._ax = None
        self._plt = None
        self._mpl = None

        self._try_open(quiet)

    # -- setup ------------------------------------------------------------

    def _try_open(self, quiet):
        """Open a window, or disable the chart and explain why."""
        if os.name != "nt" and not os.environ.get("DISPLAY") \
                and not os.environ.get("WAYLAND_DISPLAY"):
            if not quiet:
                print("Live chart: no DISPLAY/WAYLAND_DISPLAY — disabled. "
                      "(Headless? Drop --live-chart, or use xvfb-run.)")
            return

        try:
            import matplotlib
        except ImportError:
            if not quiet:
                print("Live chart: matplotlib not installed — disabled.")
            return

        current = matplotlib.get_backend()
        order = ([current] if current.lower() not in ("agg", "template")
                 else []) + _BACKENDS
        for backend in order:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                self._fig, self._ax = plt.subplots(
                    figsize=(9, max(3.0, 0.42 * len(self.opponents) + 1.8)))
                self._plt = plt
                self._mpl = matplotlib
                plt.ion()
                self._fig.canvas.manager.set_window_title(
                    self.title or "Evaluation win rate (live)")
                self._fig.show()
                self.enabled = True
                if not quiet:
                    print(f"Live chart: window open ({backend} backend)")
                return
            except Exception:                                # noqa: BLE001
                self._fig = self._ax = self._plt = None
                continue

        if not quiet:
            print(f"Live chart: no interactive backend available "
                  f"(tried {', '.join(order)}) — disabled.")

    # -- data -------------------------------------------------------------

    def set_repeat(self, repeat):
        """Announce which repeat subsequent record() calls belong to."""
        self._repeat = int(repeat)

    def record(self, opponent, result, redraw=True):
        """Log one finished game. `result` is 'WIN', 'LOSS' or 'DRAW'."""
        if not self.enabled:
            return
        if opponent not in self.counts:
            # An opponent the chart was not told about: add it rather than
            # silently dropping the result
            self.counts[opponent] = {}
            self.opponents.append(opponent)
        bucket = self.counts[opponent].setdefault(
            self._repeat, {"WIN": 0, "LOSS": 0, "DRAW": 0})
        if result in bucket:
            bucket[result] += 1
        self._current = (opponent, self._repeat)
        if redraw:
            self.draw()

    def _std(self, opponent):
        """Sample std of per-repeat win rates over COMPLETED repeats."""
        rates = []
        for r, b in self.counts[opponent].items():
            if (opponent, r) == self._current:
                continue                        # still receiving games
            m = b["WIN"] + b["LOSS"] + b["DRAW"]
            if m:
                rates.append(100.0 * b["WIN"] / m)
        if len(rates) < 2:
            return None
        mean = sum(rates) / len(rates)
        return (sum((x - mean) ** 2 for x in rates)
                / (len(rates) - 1)) ** 0.5

    def _rows(self):
        rows = []
        for o in self.opponents:
            w = sum(b["WIN"]  for b in self.counts[o].values())
            l = sum(b["LOSS"] for b in self.counts[o].values())
            d = sum(b["DRAW"] for b in self.counts[o].values())
            n = w + l + d
            rows.append({
                "name": o, "n": n, "wins": w,
                "pct": (100.0 * w / n) if n else 0.0,
                "std": self._std(o),
                "started": n > 0,
            })
        if self.sort:
            # Finished/in-progress opponents ranked by win rate; untouched
            # ones stay at the bottom in their original order
            rows.sort(key=lambda r: (not r["started"], -r["pct"], r["name"]))
        return rows

    # -- drawing ----------------------------------------------------------

    def draw(self, force=False):
        """Redraw, throttled so frequent results do not stall the eval loop."""
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_draw) < self.min_interval:
            return
        self._last_draw = now

        try:
            import numpy as np
            from matplotlib.colors import TwoSlopeNorm

            rows = self._rows()
            n = len(rows)
            ax = self._ax
            ax.clear()

            norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
            cmap = self._mpl.colormaps["RdBu"]
            pos = np.arange(n)[::-1]

            for i, r in enumerate(rows):
                if not r["started"]:
                    # Pending opponent: a faint placeholder so the axis does
                    # not jump around as the sweep progresses
                    ax.barh(pos[i], 100, height=0.7, color="#f4f4f4",
                            edgecolor="#e0e0e0", linewidth=0.8, zorder=2)
                    ax.text(1.5, pos[i], "pending", ha="left", va="center",
                            fontsize=9, color="#aaaaaa", style="italic",
                            zorder=4)
                    continue

                ax.barh(pos[i], r["pct"], height=0.7,
                        color=cmap(norm(r["pct"])), edgecolor="#222222",
                        linewidth=0.8, zorder=3)
                label = f"{r['pct']:.0f}%"
                if r["std"] is not None:
                    # Error bar at the bar tip, clipped to the valid 0-100
                    # range so it never collides with the labels
                    lo = min(r["std"], r["pct"])
                    hi = min(r["std"], 100.0 - r["pct"])
                    ax.errorbar(r["pct"], pos[i], xerr=[[lo], [hi]],
                                fmt="none", ecolor="#444444", elinewidth=1.4,
                                capsize=3.5, capthick=1.4, zorder=5)
                    label += f" ±{r['std']:.0f}"
                ax.text(104.0, pos[i], label, ha="left",
                        va="center", fontsize=10, fontweight="bold", zorder=4)
                # Progress for the opponent currently being evaluated
                prog = (f"{r['n']}/{self.target}" if self.target
                        else f"{r['n']}")
                inside = r["pct"] >= 25
                ax.text(r["pct"] - 1.5 if inside else r["pct"] + 1.5, pos[i],
                        prog, ha="right" if inside else "left", va="center",
                        fontsize=8,
                        color="white" if inside else "#666666", zorder=4)

            ax.set_yticks(pos)
            ax.set_yticklabels([r["name"] for r in rows], fontsize=10,
                               fontweight="bold", fontfamily="monospace")
            ax.set_xlim(0, 118)
            ax.set_ylim(-0.7, n - 0.3)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xlabel("Win Rate (%)", fontsize=12, fontweight="bold")
            ax.xaxis.set_label_coords(50.0 / 118.0, -0.085)
            ax.grid(axis="x", color="#dddddd", linewidth=0.9, linestyle="--",
                    zorder=0)
            ax.set_axisbelow(True)
            ax.axvline(50, color="#999999", linewidth=1.4, linestyle="--",
                       zorder=1)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)

            done = sum(r["n"] for r in rows)
            total_wins = sum(r["wins"] for r in rows)
            overall = 100.0 * total_wins / done if done else 0.0
            expected = (self.target * len(rows)) if self.target else None
            header = (f"{done}/{expected} games" if expected
                      else f"{done} games")
            header += f"   •   {total_wins} wins overall ({overall:.0f}%)"
            ax.set_title(header, fontsize=11, style="italic",
                         fontweight="bold", color="#555555", pad=10)

            self._fig.tight_layout()
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

        except Exception as e:                               # noqa: BLE001
            # A closed window or a backend hiccup must not stop evaluation
            print(f"Live chart: disabled after a draw error ({e})")
            self.enabled = False

    # -- teardown ---------------------------------------------------------

    def finish(self, save_path=None, keep_open=False):
        """Final redraw, optional save, then close unless asked to keep it."""
        if not self.enabled:
            return None
        self._current = None       # the last repeat is now complete too
        self.draw(force=True)
        written = None
        try:
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                self._fig.savefig(save_path, dpi=200, bbox_inches="tight")
                written = save_path
                print(f"Live chart saved → {save_path}")
            if keep_open:
                print("Live chart: close the window to exit.")
                self._plt.ioff()
                self._plt.show()
            else:
                self._plt.close(self._fig)
        except Exception as e:                               # noqa: BLE001
            print(f"Live chart: error during teardown ({e})")
        self.enabled = False
        return written


# ---------------------------------------------------------------------------
# Self-test: simulate a sweep without needing gym-microrts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    opponents = ["coacAI", "mayari", "workerRushAI", "lightRushAI",
                 "randomBiasedAI", "izanagi", "droplet", "naiveMCTSAI"]
    rates = dict(zip(opponents, [0.88, 1.0, 0.98, 0.96, 1.0, 0.72, 0.68, 0.36]))
    games = 20

    chart = LiveWinrateChart(opponents, games_per_opponent=games,
                             title="Live chart self-test")
    if not chart.enabled:
        print("Nothing to demonstrate without a display.")
    else:
        for o in opponents:
            for rep_i in (1, 2, 3):
                chart.set_repeat(rep_i)
                for _ in range(games):
                    r = "WIN" if random.random() < rates[o] else "LOSS"
                    chart.record(o, r)
                    time.sleep(0.01)
        chart.finish(save_path="live_selftest.png", keep_open=True)
