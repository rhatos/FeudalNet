"""
tournament.py

Round-robin tournament between the trained FeudalNet agent and the scripted
microRTS bots. Every participant plays every other participant.

Scoring (microRTS competition convention):
    win = 1.0, draw = 0.5, loss = 0.0

Outputs three CSVs, all updated incrementally after every pairing so a crash
never loses completed results:
    <base>_matches.csv    one row per match
    <base>_standings.csv  league table, sorted by points
    <base>_matrix.csv     cross-table of POINTS (row player vs column player)
    <base>_matrix_winrate.csv  cross-table of WIN RATE % (row vs column)

IMPORTANT — run this first to verify the bot-vs-bot API on your install:
    python tournament.py --check-api

Usage:
    # Agent + the fast bots, 2 games per pairing (one from each side)
    python tournament.py --checkpoint models/xxx.pt

    # Explicit participant list
    python tournament.py --checkpoint models/xxx.pt \
        --bots coacAI mayari workerRushAI lightRushAI randomBiasedAI \
        --games-per-pairing 4

    # Headless (cluster)
    xvfb-run -a python tournament.py --checkpoint models/xxx.pt --no-render
"""

import os
import csv
import time
import inspect
import argparse
import datetime
import itertools
import numpy as np
import torch

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:                                    # pragma: no cover
    _HAS_TQDM = False

    class tqdm:                                        # minimal stand-in
        """No-op fallback so the script still runs without tqdm installed."""
        def __init__(self, *a, **kw):
            self.total = kw.get("total", 0)
            self.n = 0
        def update(self, k=1):   self.n += k
        def set_description(self, *a, **kw): pass
        def set_postfix_str(self, *a, **kw): pass
        def close(self):         pass
        def __enter__(self):     return self
        def __exit__(self, *a):  return False
        @staticmethod
        def write(msg): print(msg, flush=True)

from feudalnet import FeudalNetwork

REWARD_FUNCTIONS = [
    "WinLoss", "ResourceGather", "ProduceWorker",
    "ProduceBuilding", "Attack", "ProduceCombatUnit",
]

# Bots that run at ~100 steps/s rather than ~6000+ (from test_opponent_speed.py)
SLOW_BOTS = {"naiveMCTSAI", "mixedBot", "izanagi", "tiamat",
             "droplet", "guidedRojoA3N", "mayari"}

FAST_BOTS = ["randomAI", "randomBiasedAI", "passiveAI",
             "workerRushAI", "lightRushAI", "coacAI", "rojo"]

# Every non-PO bot in gym-microrts 0.6.0. The partial-observability baselines
# (POWorkerRush / POLightRush / PORangedRush / POHeavyRush) are deliberately
# excluded: they are designed for partial_obs=True and would play here as
# handicapped rush bots, which is not a meaningful comparison.
ALL_BOTS = FAST_BOTS + ["mayari", "mixedBot", "izanagi", "tiamat",
                        "droplet", "guidedRojoA3N", "naiveMCTSAI"]

# Rough per-game seconds, for the runtime estimate only
_SEC_FAST, _SEC_SLOW, _SEC_SLOW_PAIR = 2.0, 20.0, 40.0

AGENT_NAME = "FeudalNet"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="microRTS round-robin tournament")
parser.add_argument('--checkpoint', type=str, default=None,
                    help='FeudalNet checkpoint. Omit to run a bots-only tournament.')
parser.add_argument('--bots', nargs='+', default=['fast'],
                    help="Bot participants. Accepts explicit names, or the "
                         "keywords 'all' (every non-PO bot, 14), 'fast' "
                         "(default, the 7 quick bots) or 'slow'. Keywords and "
                         "names may be mixed, e.g. --bots fast mayari")
parser.add_argument('--games-per-pairing', type=int, default=2,
                    help='Games per pairing. Even numbers let each participant '
                         'play both map positions equally.')
parser.add_argument('--map-h',      type=int, default=16)
parser.add_argument('--map-w',      type=int, default=16)
parser.add_argument('--map-variant', type=str, default='')
parser.add_argument('--map-path',   type=str, default=None)
parser.add_argument('--max-steps',  type=int, default=2000,
                    help='Step cap per match (draw if reached)')
parser.add_argument('--in-channels', type=int, default=27)
parser.add_argument('--no-render',  action='store_true',
                    help='Disable rendering (much faster; needed when headless '
                         'without a virtual framebuffer)')
parser.add_argument('--game-speed', type=int, default=0,
                    help='Render FPS cap. 0 = as fast as possible.')
parser.add_argument('--csv-out',    type=str, default=None)
parser.add_argument('--cuda',       type=bool, default=True)
parser.add_argument('--seed',       type=int, default=42)
parser.add_argument('--extra-jar', nargs='+', default=[],
                    help='Third-party bot JAR(s) to load. Registers each bot '
                         'so it can be used by name. Optionally pin the class: '
                         "--extra-jar UTS_Imass.jar:uts.imass.Imass:UTS_Imass")
parser.add_argument('--probe-bot-env', action='store_true',
                    help='Play ONE short bot-vs-bot match to validate the '
                         'MicroRTSBotVecEnv API, then exit')
parser.add_argument('--check-api',  action='store_true',
                    help='Report the vec_env classes and signatures available '
                         'in your gym-microrts install, then exit')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Third-party bot JARs — must be registered BEFORE any env starts the JVM
# ---------------------------------------------------------------------------
if args.extra_jar:
    import custom_bots
    for spec in args.extra_jar:
        parts = spec.split(":")
        jar   = parts[0]
        cls   = parts[1] if len(parts) > 1 and parts[1] else None
        alias = parts[2] if len(parts) > 2 and parts[2] else None
        custom_bots.register(jar, class_name=cls, name=alias)

# ---------------------------------------------------------------------------
# --check-api : introspect the installed gym-microrts
# ---------------------------------------------------------------------------

if args.check_api:
    from gym_microrts.envs import vec_env as _ve
    from gym_microrts import microrts_ai
    print("Classes in gym_microrts.envs.vec_env:")
    for name in dir(_ve):
        obj = getattr(_ve, name)
        if inspect.isclass(obj) and "VecEnv" in name:
            print(f"\n=== {name} ===")
            try:
                sig = inspect.signature(obj.__init__)
                for p in list(sig.parameters.values())[1:]:
                    print(f"    {p.name:<22} default={p.default!r}")
            except (TypeError, ValueError) as e:
                print(f"    <signature unavailable: {e}>")
            for m in ("step", "reset", "render", "getattr_depth_check"):
                if hasattr(obj, m):
                    try:
                        print(f"  .{m}{inspect.signature(getattr(obj, m))}")
                    except (TypeError, ValueError):
                        print(f"  .{m}(?)")
    print(f"\nBot-vs-bot support: "
          f"{'MicroRTSBotVecEnv FOUND' if hasattr(_ve, 'MicroRTSBotVecEnv') else 'NOT FOUND'}")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

from gym_microrts.envs import vec_env as VE
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

HAS_BOT_ENV = hasattr(VE, "MicroRTSBotVecEnv")
if HAS_BOT_ENV:
    MicroRTSBotVecEnv = VE.MicroRTSBotVecEnv

# --- Expand the 'all' / 'fast' / 'slow' keywords ---
_expanded, _seen = [], set()
for entry in args.bots:
    group = {"all": ALL_BOTS,
             "fast": FAST_BOTS,
             "slow": [b for b in ALL_BOTS if b in SLOW_BOTS]}.get(entry.lower())
    for name in (group if group else [entry]):
        if name not in _seen:
            _seen.add(name)
            _expanded.append(name)
args.bots = _expanded

# --- Validate bots exist ---
missing = [b for b in args.bots if not hasattr(microrts_ai, b)]
if missing:
    raise ValueError(f"Bots not in your install: {missing}. "
                     f"Run evaluate_agent.py --list-ais to see valid names.")

# --- Load agent (optional) ---
agent_model = None
if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    if "hidden_dim" not in saved and "hidden_dim_manager" in saved:
        saved["hidden_dim"] = saved["hidden_dim_manager"]
    # Architecture is a property of the checkpoint
    for key in ["map_h", "map_w", "in_channels"]:
        if key in saved:
            setattr(args, key, saved[key])
    ARCH = {k: saved.get(k, d) for k, d in [
        ("hidden_dim", 128), ("time_horizon", 25), ("enemy_dim", 64),
        ("enemy_layers", 2), ("worker_heads", 4), ("worker_layers", 4),
        ("T_W", 100), ("manager_heads", 4), ("manager_layers", 6), ("T_M", 160),
    ]}
    print(f"Agent: {args.checkpoint}")
    print(f"  d={ARCH['hidden_dim']} c={ARCH['time_horizon']} "
          f"T_W={ARCH['T_W']} T_M={ARCH['T_M']} map={args.map_h}x{args.map_w}")

map_path = args.map_path or (f"maps/{args.map_h}x{args.map_w}/"
                             f"basesWorkers{args.map_h}x{args.map_w}"
                             f"{args.map_variant}.xml")

participants = ([AGENT_NAME] if args.checkpoint else []) + list(args.bots)

# Order pairings so those involving fewer slow bots run first. Combined with
# incremental CSV writing this means aborting early still leaves a usable
# table rather than only the alphabetically-first results.
def _n_slow_in(pair):
    return sum(1 for p in pair if p in SLOW_BOTS)

pairings = sorted(itertools.combinations(participants, 2), key=_n_slow_in)
n_slow   = sum(1 for p in participants if p in SLOW_BOTS)

def _estimate_seconds():
    total = 0.0
    for pair in pairings:
        k = _n_slow_in(pair)
        per = _SEC_FAST if k == 0 else (_SEC_SLOW if k == 1 else _SEC_SLOW_PAIR)
        total += per * args.games_per_pairing
    return total

print(f"\nMap: {map_path}")
print(f"Participants ({len(participants)}): {', '.join(participants)}")
print(f"Pairings: {len(pairings)} x {args.games_per_pairing} games "
      f"= {len(pairings) * args.games_per_pairing} matches")
print(f"Render: {'OFF' if args.no_render else 'ON'}")
_est = _estimate_seconds()
_h, _m = divmod(int(_est) // 60, 60)
print(f"Rough estimate: {_h}h {_m:02d}m of compute"
      + (" (rendering will make this substantially slower)"
         if not args.no_render else ""))
if n_slow:
    print(f"NOTE: {n_slow} of {len(participants)} participants are slow "
          f"(~100 steps/s): {', '.join(p for p in participants if p in SLOW_BOTS)}")
    print("      Pairings are ordered fast-first, so results accumulate early "
          "and aborting mid-run still leaves a usable table.")
if not HAS_BOT_ENV and len(args.bots) > 1:
    print("\nWARNING: MicroRTSBotVecEnv not found in your install — bot-vs-bot "
          "matches cannot be played. Only the agent's matches will run. "
          "Run --check-api and share the output so this can be adapted.")
print("=" * 70)

# ---------------------------------------------------------------------------
# Match runners
# ---------------------------------------------------------------------------

_env_cache = {}   # envs are never closed: close() shuts down the whole JVM

# Writing with plain print() while a tqdm bar is live garbles the display,
# so all in-loop output goes through tqdm.write().
_pbar = None

def _write(msg):
    if _pbar is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)

def _delay():
    if not args.no_render and args.game_speed > 0:
        time.sleep(1.0 / args.game_speed)


def play_agent_vs_bot(bot_name, n_games):
    """Agent is player 1. Yields (result, steps, raw_rewards) per game."""
    key = ("agent", bot_name)
    if key not in _env_cache:
        env = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0, num_bot_envs=1,
            max_steps=args.max_steps, render_theme=2, partial_obs=False,
            ai2s=[getattr(microrts_ai, bot_name)], map_paths=[map_path],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        model = FeudalNetwork(
            num_workers=1, h=args.map_h, w=args.map_w,
            in_channels=args.in_channels,
            d=ARCH["hidden_dim"], n_cells=args.map_h * args.map_w,
            action_space=env.action_plane_space.nvec.tolist(),
            time_horizon=ARCH["time_horizon"], dilation=10, eps=0.0,
            device=device,
            enemy_dim=ARCH["enemy_dim"], enemy_layers=ARCH["enemy_layers"],
            worker_heads=ARCH["worker_heads"], worker_layers=ARCH["worker_layers"],
            T_W=ARCH["T_W"], manager_heads=ARCH["manager_heads"],
            manager_layers=ARCH["manager_layers"], T_M=ARCH["T_M"],
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        _env_cache[key] = (env, model)
    env, model = _env_cache[key]

    for _ in range(n_games):
        obs = env.reset()
        goals, states, masks = model.init_obj()
        model.repackage_hidden()
        steps, done_flag = 0, False
        while not done_flag:
            if not args.no_render:
                env.render()
                _delay()
            with torch.no_grad():
                logits, goals, states, _, _ = model(
                    torch.FloatTensor(obs).to(device), goals, states, masks[-1],
                    action_mask=torch.tensor(env.get_action_mask(),
                                             dtype=torch.bool, device=device),
                )
                actions, _, _ = FeudalNetwork.sample(logits)
            obs, _, done, infos = env.step(
                np.array(actions.cpu().numpy(), dtype=np.int32))
            masks.pop(0)
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(device))
            steps += 1
            if done[0]:
                done_flag = True
                raw = np.asarray(infos[0].get("raw_rewards", np.zeros(6),),
                                 dtype=float)
                wl  = float(raw[0])
                yield ("WIN" if wl > 0 else "LOSS" if wl < 0 else "DRAW",
                       steps, raw)


# MicroRTSBotVecEnv's API differs subtly from MicroRTSGridModeVecEnv and
# varies between gym-microrts versions:
#   - map_paths defaults to a bare STRING here but a LIST in the grid env
#   - step(ac) still takes an action even though both players are bots
# Rather than hardcode a guess, both are discovered once and cached.
_BOT_MAP_FMT  = None   # "list" or "str"
_BOT_STEP_FMT = None   # index into _STEP_CANDIDATES


def _make_bot_env(bot_a, bot_b):
    """Construct a bot-vs-bot env, discovering the map_paths format once."""
    global _BOT_MAP_FMT
    kwargs = dict(
        ai1s=[getattr(microrts_ai, bot_a)],
        ai2s=[getattr(microrts_ai, bot_b)],
        max_steps=args.max_steps, render_theme=2, partial_obs=False,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    fmts = ([_BOT_MAP_FMT] if _BOT_MAP_FMT
            else ["list", "str"])   # list first: matches the grid env
    last = None
    for fmt in fmts:
        try:
            env = MicroRTSBotVecEnv(
                map_paths=([map_path] if fmt == "list" else map_path), **kwargs)
            if _BOT_MAP_FMT is None:
                _BOT_MAP_FMT = fmt
                _write(f"    [api] MicroRTSBotVecEnv map_paths format: {fmt}")
            return env
        except Exception as e:            # noqa: BLE001 - probing on purpose
            last = e
    raise RuntimeError(
        f"Could not construct MicroRTSBotVecEnv with either map_paths format. "
        f"Last error: {type(last).__name__}: {last}")


def _bot_step(env):
    """Step a bot-vs-bot env, discovering the accepted action format once."""
    global _BOT_STEP_FMT
    # Both players are bots, so the action is ignored — but the shape must
    # survive whatever unpacking the wrapper does before discarding it.
    candidates = [
        ("[[]]",              lambda: env.step([[]])),
        ("[[], []]",          lambda: env.step([[], []])),
        ("np.zeros((1,0))",   lambda: env.step(np.zeros((1, 0), dtype=np.int32))),
        ("[None]",            lambda: env.step([None])),
        ("None",              lambda: env.step(None)),
    ]
    if _BOT_STEP_FMT is not None:
        return candidates[_BOT_STEP_FMT][1]()
    last = None
    for i, (label, fn) in enumerate(candidates):
        try:
            out = fn()
            _BOT_STEP_FMT = i
            _write(f"    [api] MicroRTSBotVecEnv step() action format: {label}")
            return out
        except Exception as e:            # noqa: BLE001 - probing on purpose
            last = e
    raise RuntimeError(
        f"No accepted action format for MicroRTSBotVecEnv.step(). "
        f"Last error: {type(last).__name__}: {last}")


def play_bot_vs_bot(bot_a, bot_b, n_games):
    """
    bot_a is player 1. Yields (result_for_a, steps, raw_rewards) per game.
    Requires MicroRTSBotVecEnv.
    """
    if not HAS_BOT_ENV or n_games <= 0:
        return
    key = ("bot", bot_a, bot_b)
    if key not in _env_cache:
        _env_cache[key] = _make_bot_env(bot_a, bot_b)
    env = _env_cache[key]

    for _ in range(n_games):
        env.reset()
        steps, done_flag = 0, False
        while not done_flag:
            if not args.no_render:
                env.render()
                _delay()
            _, _, done, infos = _bot_step(env)
            steps += 1
            if done[0]:
                done_flag = True
                raw = np.asarray(infos[0].get("raw_rewards", np.zeros(6)),
                                 dtype=float)
                wl  = float(raw[0])
                yield ("WIN" if wl > 0 else "LOSS" if wl < 0 else "DRAW",
                       steps, raw)


# ---------------------------------------------------------------------------
# CSV setup (incremental)
# ---------------------------------------------------------------------------

if args.csv_out:
    base = args.csv_out
else:
    tag   = (os.path.splitext(os.path.basename(args.checkpoint))[0]
             if args.checkpoint else "botsonly")
    mname = os.path.splitext(os.path.basename(map_path))[0]
    base  = os.path.join("results", "tournament_"
                         f"{tag}_{mname}_"
                         f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

matches_path   = f"{base}_matches.csv"
standings_path = f"{base}_standings.csv"
matrix_path    = f"{base}_matrix.csv"
matrix_wr_path = f"{base}_matrix_winrate.csv"

MATCH_FIELDS = (["match", "player1", "player2", "result_p1", "winner",
                 "steps", "points_p1", "points_p2"]
                + [f"reward_{fn}" for fn in REWARD_FUNCTIONS]
                + ["map", "seed"])
with open(matches_path, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=MATCH_FIELDS).writeheader()

points  = {p: 0.0 for p in participants}
record  = {p: {"W": 0, "D": 0, "L": 0} for p in participants}
played  = {p: 0 for p in participants}
# head-to-head points, for the cross-table
h2h     = {a: {b: [] for b in participants} for a in participants}
matches = []


def flush_standings():
    rows = []
    for p in sorted(participants, key=lambda x: (-points[x], x)):
        n = played[p]
        rows.append({
            "participant": p,
            "played":  n,
            "wins":    record[p]["W"],
            "draws":   record[p]["D"],
            "losses":  record[p]["L"],
            "points":  round(points[p], 1),
            "points_pct": round(100 * points[p] / n, 2) if n else "",
            "win_pct": round(100 * record[p]["W"] / n, 2) if n else "",
        })
    with open(standings_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["map", "seed"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "map": map_path, "seed": args.seed})

    order = [r["participant"] for r in rows]

    # Cross-table 1: cell = POINTS row-player earned against column-player
    # (win 1.0, draw 0.5, loss 0.0 — so this credits draws)
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vs"] + order + ["TOTAL"])
        for a in order:
            cells = []
            for b in order:
                vals = h2h[a][b]
                cells.append(round(sum(vals), 1) if vals else ("-" if a == b else ""))
            w.writerow([a] + cells + [round(points[a], 1)])

    # Cross-table 2: cell = WIN RATE % of row-player against column-player.
    # h2h stores per-game points, so a 1.0 entry is a win and 0.5 a draw.
    # Unlike the points table this gives draws NO credit, which is why the
    # two tables can rank participants differently.
    with open(matrix_wr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vs"] + order + ["TOTAL"])
        for a in order:
            cells = []
            for b in order:
                vals = h2h[a][b]
                if not vals:
                    cells.append("-" if a == b else "")
                else:
                    n_win = sum(1 for v in vals if v == 1.0)
                    cells.append(round(100.0 * n_win / len(vals), 1))
            overall = (round(100.0 * record[a]["W"] / played[a], 1)
                       if played[a] else "")
            w.writerow([a] + cells + [overall])


def record_match(p1, p2, result_p1, steps, raw):
    pts1 = 1.0 if result_p1 == "WIN" else 0.5 if result_p1 == "DRAW" else 0.0
    pts2 = 1.0 - pts1
    points[p1] += pts1
    points[p2] += pts2
    played[p1] += 1
    played[p2] += 1
    record[p1]["W" if pts1 == 1 else "D" if pts1 == 0.5 else "L"] += 1
    record[p2]["W" if pts2 == 1 else "D" if pts2 == 0.5 else "L"] += 1
    h2h[p1][p2].append(pts1)
    h2h[p2][p1].append(pts2)

    rec = {
        "match": len(matches) + 1, "player1": p1, "player2": p2,
        "result_p1": result_p1,
        "winner": p1 if pts1 == 1 else p2 if pts2 == 1 else "DRAW",
        "steps": steps, "points_p1": pts1, "points_p2": pts2,
        "map": map_path, "seed": args.seed,
    }
    for j, fn in enumerate(REWARD_FUNCTIONS):
        rec[f"reward_{fn}"] = round(float(raw[j]), 3) if j < len(raw) else 0.0
    matches.append(rec)
    with open(matches_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MATCH_FIELDS).writerow(rec)


# ---------------------------------------------------------------------------
# Run the tournament
# ---------------------------------------------------------------------------

# --- Optional API probe: one short bot-vs-bot match ---
if args.probe_bot_env:
    if not HAS_BOT_ENV:
        raise SystemExit("MicroRTSBotVecEnv not available in this install.")
    print("Probing bot-vs-bot API with one randomBiasedAI vs workerRushAI match...")
    res = list(play_bot_vs_bot("randomBiasedAI", "workerRushAI", 1))
    print(f"Result: {res[0][0]} in {res[0][1]} steps  "
          f"(raw_rewards[0]={res[0][2][0]})")
    print("Bot-vs-bot works. Run the full tournament without --probe-bot-env.")
    raise SystemExit(0)

print(f"\nCSV output (updated after every pairing):")
print(f"  matches   -> {matches_path}")
print(f"  standings -> {standings_path}")
print(f"  matrix    -> {matrix_path}")
print(f"  win-rate  -> {matrix_wr_path}\n")

t0 = time.time()
total_matches = len(pairings) * args.games_per_pairing
if not _HAS_TQDM:
    print("(tqdm not installed — progress bar disabled; "
          "pip install tqdm to enable it)")

_pbar = tqdm(total=total_matches, unit="match", dynamic_ncols=True,
             smoothing=0.05,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")

def _leader_str():
    top = sorted(participants, key=lambda x: (-points[x], x))[:2]
    return "  ".join(f"{p}:{points[p]:.1f}" for p in top)

with _pbar:
    for i, (a, b) in enumerate(pairings, 1):
        # Split games between positions so neither side gets a fixed advantage
        n_first  = args.games_per_pairing // 2 + args.games_per_pairing % 2
        n_second = args.games_per_pairing - n_first

        _pbar.set_description(f"[{i}/{len(pairings)}] {a} vs {b}")
        _pbar.set_postfix_str(f"lead {_leader_str()}")

        try:
            if a == AGENT_NAME:
                # Agent is always player 1 (env wrapper limitation)
                for res, steps, raw in play_agent_vs_bot(
                        b, args.games_per_pairing):
                    record_match(a, b, res, steps, raw)
                    _pbar.update(1)
            elif b == AGENT_NAME:
                for res, steps, raw in play_agent_vs_bot(
                        a, args.games_per_pairing):
                    record_match(b, a, res, steps, raw)
                    _pbar.update(1)
            else:
                for res, steps, raw in play_bot_vs_bot(a, b, n_first):
                    record_match(a, b, res, steps, raw)
                    _pbar.update(1)
                for res, steps, raw in play_bot_vs_bot(b, a, n_second):
                    record_match(b, a, res, steps, raw)
                    _pbar.update(1)
        except Exception as e:                     # noqa: BLE001
            _write(f"    SKIPPED {a} vs {b} ({type(e).__name__}: {e})")
            # Keep the bar's total honest about games that never ran
            _pbar.total = max(_pbar.n, _pbar.total - args.games_per_pairing)
            continue

        flush_standings()
        _write(f"  [{i}/{len(pairings)}] {a} vs {b}  ->  "
               f"{a}: {points[a]:.1f} pts   {b}: {points[b]:.1f} pts")

_pbar = None

# ---------------------------------------------------------------------------
# Final standings
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL STANDINGS")
print("=" * 70)
print(f"{'#':>2} {'Participant':<18} {'P':>4} {'W':>4} {'D':>4} {'L':>4} "
      f"{'Pts':>6} {'Pts%':>7}")
print("-" * 70)
for rank, p in enumerate(sorted(participants, key=lambda x: (-points[x], x)), 1):
    n = played[p]
    print(f"{rank:>2} {p:<18} {n:>4} {record[p]['W']:>4} {record[p]['D']:>4} "
          f"{record[p]['L']:>4} {points[p]:>6.1f} "
          f"{100 * points[p] / n if n else 0:>6.1f}%")
print("=" * 70)
print(f"{len(matches)} matches in {time.time() - t0:.0f}s")
print(f"  {matches_path}")
print(f"  {standings_path}")
print(f"  {matrix_path}")
print(f"  {matrix_wr_path}")
