"""
evaluate_agent.py

Evaluate a trained transformer-agent checkpoint (WeightedAgent /
MixedEmbeddedAgent) against any gym-microrts AI, with incremental CSV
tracking and console reporting. --parallel steps P games at once in one
vectorised env, for every opponent.

Only the agent-specific parts differ: the model is built from
transformer_agent (base or embedded variant) and actions come from
agent.get_action() after reshaping observations with the matching
feature map.

Updated for gym-microrts 0.6.0 (matching our install): the env takes
map_paths=[...] / partial_obs / max_steps, and step() accepts a plain
numpy action array of shape (num_envs, h*w, 7) — the env builds the Java
valid-action arrays internally, so the old manual JArray filtering is gone.

Usage:
    # Single opponent
    python evaluate_agent.py --checkpoint models/transformer.pt --opponent coacAI

    # All opponents, 5 games each
    python evaluate_agent.py --checkpoint models/transformer.pt --opponent all --num-games 5

    # Embedded variant
    python evaluate_agent.py --checkpoint models/embedded.pt --agent-type embedded --opponent mayari

    # Fast, many games, no window
    python evaluate_agent.py --checkpoint models/transformer.pt --num-games 50 --no-render
"""

import os
import sys
import csv
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F

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

from transformer_agent.mixed_embedded_agent import (
    MixedEmbeddedAgent, reshape_observation_mixed_embedded)
from transformer_agent.weighted_agent import (
    WeightedAgent, reshape_observation_extended)


def reshape_observation_extended_fast(x, device):
    """
    Vectorised drop-in for transformer_agent.weighted_agent.
    reshape_observation_extended (the base / non-embedded agent's feature
    map). Identical outputs, no per-env loop. Each entity row is the one-hot
    of its grid position concatenated with its feature vector; ordering is
    player-1, player-2, resource cells, each by ascending position.
    """
    player_1, player_2, resource = 11, 12, 14
    N, H, W, C = x.shape
    HW = H * W
    x_r = x.reshape(N, HW, C)

    p1  = x_r[:, :, player_1] != 0
    p2  = x_r[:, :, player_2] != 0
    res = x_r[:, :, resource] != 0

    group = torch.full((N, HW), 3, dtype=torch.long, device=x_r.device)
    group[res] = 2
    group[p2]  = 1
    group[p1]  = 0
    pos   = torch.arange(HW, device=x_r.device).expand(N, HW)
    order = torch.argsort(group * HW + pos, dim=1)

    c1 = p1.sum(1); c2 = p2.sum(1); c3 = res.sum(1)
    total = c1 + c2 + c3
    idx   = pos
    valid = idx < total[:, None]
    vf    = valid[:, :, None].to(x_r.dtype)

    onehot   = F.one_hot(order, HW).to(x_r.dtype) * vf
    gathered = torch.gather(x_r, 1, order[:, :, None].expand(N, HW, C)) * vf
    x_reshaped = torch.cat((onehot, gathered), dim=-1).to(device)

    entity_mask       = (~valid).to(device)
    player_unit_mask  = (~(idx < c1[:, None])).to(device)
    enemy_unit_mask   = (~((idx >= c1[:, None]) & (idx < (c1 + c2)[:, None]))).to(device)
    neutral_unit_mask = (~((idx >= (c1 + c2)[:, None]) & valid)).to(device)
    entity_unit_counts = total.to(device)
    player_unit_positions = x_r[:, :, player_1].to(device)
    return (x_reshaped, entity_mask, entity_unit_counts, player_unit_positions,
            player_unit_mask, enemy_unit_mask, neutral_unit_mask)


def reshape_observation_mixed_embedded_fast(x, device):
    """
    Vectorised drop-in for transformer_agent.mixed_embedded_agent.
    reshape_observation_mixed_embedded. Identical outputs, no per-env loop.

    Entities are ordered player-1 cells, then player-2 cells, then resource
    cells, each in ascending grid position - reproduced with a single argsort
    over a group*HW+position key, which is unique per cell so no tie-breaking
    is needed.
    """
    player_1, player_2, resource = 11, 12, 14
    N, H, W, C = x.shape
    HW = H * W
    x_r = x.reshape(N, HW, C)

    p1  = x_r[:, :, player_1] != 0
    p2  = x_r[:, :, player_2] != 0
    res = x_r[:, :, resource] != 0

    group = torch.full((N, HW), 3, dtype=torch.long, device=x_r.device)
    group[res] = 2
    group[p2]  = 1
    group[p1]  = 0
    pos   = torch.arange(HW, device=x_r.device).expand(N, HW)
    order = torch.argsort(group * HW + pos, dim=1)          # [N, HW]

    c1 = p1.sum(1); c2 = p2.sum(1); c3 = res.sum(1)
    total = c1 + c2 + c3
    idx   = pos                                             # [N, HW] row index
    valid = idx < total[:, None]

    gathered = torch.gather(x_r, 1, order[:, :, None].expand(N, HW, C))
    out = torch.zeros(N, HW, C + 1, dtype=torch.int16, device=device)
    out[:, :, 0]  = (order * valid).to(torch.int16).to(device)
    out[:, :, 1:] = (gathered * valid[:, :, None]).to(torch.int16).to(device)

    entity_mask       = (~valid).to(device)
    player_unit_mask  = (~(idx < c1[:, None])).to(device)
    enemy_unit_mask   = (~((idx >= c1[:, None]) & (idx < (c1 + c2)[:, None]))).to(device)
    neutral_unit_mask = (~((idx >= (c1 + c2)[:, None]) & valid)).to(device)
    entity_unit_counts = total.to(device)
    player_unit_positions = x_r[:, :, player_1].to(device)
    return (out, entity_mask, entity_unit_counts, player_unit_positions,
            player_unit_mask, enemy_unit_mask, neutral_unit_mask)


# Names of the six raw reward components, in the order microRTS returns them
REWARD_FUNCTIONS = [
    "WinLoss", "ResourceGather", "ProduceWorker",
    "ProduceBuilding", "Attack", "ProduceCombatUnit",
]

# ---------------------------------------------------------------------------
# All available AIs in gym-microrts 0.6.0
# ---------------------------------------------------------------------------

ALL_AIS = [
    # Random / baseline
    "randomAI",
    "randomBiasedAI",
    "passiveAI",

    # Rush strategies
    "workerRushAI",
    "lightRushAI",

    # Search-based baseline
    "naiveMCTSAI",

    # Competition winners / strong scripted bots
    "coacAI",      # IEEE-CoG 2020 competition winner
    "mayari",      # IEEE-CoG 2021 competition winner (strongest available)
    "mixedBot",
    "rojo",
    "izanagi",
    "tiamat",
    "droplet",
    "guidedRojoA3N",
]

# Bots measured at ~100 steps/s rather than ~6000+ (see test_opponent_speed.py).
# With --opponent all these are evaluated LAST so results accumulate quickly
# and aborting early still leaves a useful table.
SLOW_AIS = {"naiveMCTSAI", "mixedBot", "izanagi", "tiamat",
            "droplet", "guidedRojoA3N", "mayari"}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Evaluate a transformer-agent checkpoint')

parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to the .pt checkpoint file '
                         '(required unless --list-ais)')
parser.add_argument('--opponent',   type=str, default='coacAI',
                    help='Opponent name, or "all". Names from --extra-jar are also accepted.')
parser.add_argument('--num-games',  type=int, default=10,
                    help='Number of games per opponent PER REPEAT')
parser.add_argument('--game-speed', type=int, default=12,
                    help='Render speed in FPS — lower = slower, higher = faster')
parser.add_argument('--map-h',      type=int, default=8)
parser.add_argument('--map-w',      type=int, default=8)
parser.add_argument('--cuda',       type=bool, default=True)
parser.add_argument('--seed',       type=int, default=42)
parser.add_argument('--map-variant', type=str, default='',
                    help="Map variant suffix, e.g. 'A' for basesWorkers16x16A. "
                         "Empty = the base map. Use to test generalisation to "
                         "maps unseen during training.")
parser.add_argument('--map-path',   type=str, default=None,
                    help='Full map path override, e.g. '
                         'maps/16x16/basesWorkers16x16noResources.xml')
parser.add_argument('--sides',       type=str, default='p1',
                    choices=['p1', 'p2', 'both'],
                    help="Which side the agent plays. 'p1' = default start "
                         "position; 'p2' = the opposite side (via a map with "
                         "player ownership swapped); 'both' = split games "
                         "evenly and report each side separately. Use 'both' "
                         "to remove starting-position bias.")
parser.add_argument('--csv-out',     type=str, default=None,
                    help='Base path for CSV output. Two files are written: '
                         '<base>_games.csv (one row per game) and '
                         '<base>_summary.csv (one row per opponent). '
                         'Default: results/eval_<checkpoint>_<map>_<timestamp>')
parser.add_argument('--no-csv',     action='store_true',
                    help='Disable CSV output')
parser.add_argument('--repeats',    type=int, default=1,
                    help='Independent repetitions of the whole evaluation '
                         'per opponent (e.g. --num-games 250 --repeats 4 '
                         'plays 4 separate 250-game runs). Each repeat is '
                         're-seeded with seed+repeat for reproducibility, '
                         'tagged in the games CSV, and the summary gains '
                         'win_pct_std — the sample standard deviation of '
                         'the per-repeat win rates — plus the raw rates in '
                         'win_pct_reps.')
parser.add_argument('--live-chart', action='store_true',
                    help='Open a live-updating win-rate bar chart that '
                         'redraws as games finish. Needs a desktop session; '
                         'silently disables itself when headless.')
parser.add_argument('--live-chart-out', type=str, default=None,
                    help='Save the final live chart here (default: alongside '
                         'the CSVs as <base>_livechart.png)')
parser.add_argument('--live-chart-keep-open', action='store_true',
                    help='Leave the chart window open when evaluation ends '
                         '(blocks until you close it)')
parser.add_argument('--parallel',   type=int, default=1,
                    help='Number of games to run simultaneously (as parallel '
                         'bot envs in one vectorised env, sharing one batched '
                         'forward pass). Games are quota-assigned per env '
                         'slot so the result is an unbiased sample — naively '
                         'taking the first N completions would over-count '
                         'short games. Applies to every opponent.')
parser.add_argument('--profile',    action='store_true',
                    help='Print a per-step timing split (feature map / agent '
                         'forward / envs.step) every 200 steps, to locate '
                         'the bottleneck when --parallel does not scale.')
parser.add_argument('--no-render',  action='store_true',
                    help='Disable rendering. Much faster, and required when '
                         'headless without a virtual framebuffer.')
parser.add_argument('--extra-jar', nargs='+', default=[],
                    help='Third-party bot JAR(s) to load. Registers each bot '
                         'so it can be used by name. Optionally pin the class: '
                         "--extra-jar UTS_Imass.jar:uts.imass.Imass:UTS_Imass")
parser.add_argument('--list-ais',   action='store_true',
                    help='Print the AIs actually available in the installed '
                         'gym_microrts.microrts_ai module, then exit')

# --- Agent-specific arguments ---
parser.add_argument('--agent-type', type=str, default='base',
                    choices=['base', 'embedded'],
                    help="Which transformer agent variant the checkpoint is: "
                         "'base' = WeightedAgent, 'embedded' = "
                         "MixedEmbeddedAgent")
parser.add_argument('--transformer-layers', type=int, default=5,
                    help='the number of layers to use in the transformer '
                         'encoder')
parser.add_argument('--attention-heads', type=int, default=7,
                    help='determines number of heads to use for multi-headed '
                         'attention in transformer')
parser.add_argument('--padding', type=int, default=0,
                    help='how much to pad inputs to the transformer encoder '
                         'by (can use this to balance out heads)')
parser.add_argument('--feed-forward-neurons', type=int, default=512,
                    help='the number of feed-forward neurons in a '
                         'transformer layer')
parser.add_argument('--embed-size', type=int, default=64,
                    help='when using embeddings, determines how large an '
                         'embedding generated from an observation feature '
                         'should be (embedded variant only)')

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
# --list-ais: report what the installed gym-microrts actually provides
# ---------------------------------------------------------------------------
if args.list_ais:
    from gym_microrts import microrts_ai
    available = sorted(
        n for n in dir(microrts_ai)
        if not n.startswith("_") and callable(getattr(microrts_ai, n))
    )
    print(f"AIs available in installed gym_microrts.microrts_ai "
          f"({len(available)}):")
    for name in available:
        known = " " if name in ALL_AIS else " (not in this script's list)"
        print(f"  {name}{known}")
    missing = [a for a in ALL_AIS if not hasattr(microrts_ai, a)]
    if missing:
        print(f"\nListed here but NOT in your install: {', '.join(missing)}")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(args.seed)
np.random.seed(args.seed)

cuda_ok = torch.cuda.is_available() and args.cuda
device  = torch.device("cuda" if cuda_ok else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

if args.checkpoint is None:
    parser.error("--checkpoint is required (unless using --list-ais)")

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

print(f"Loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

# This agent's checkpoints are plain state_dicts (no saved args), so the
# architecture comes from the CLI flags above rather than the checkpoint.
# Our wrapped {"model": ..., "args": ...} format is also tolerated.
state_dict = (ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt
              else ckpt)

# ---------------------------------------------------------------------------
# Infer the architecture from the checkpoint. These checkpoints are bare
# state_dicts, but the
# tensor shapes encode everything except the head count:
#   network.layers.<N>.*            -> number of transformer layers
#   network.layers.0.linear1.weight -> (dim_feedforward, padded_size)
#   map_embedder.weight             -> present only for the embedded variant;
#                                      shape (map_size, embed_size)
# padded_size then gives the padding once the input size is known:
#   base:     input = map_h*map_w + 27
#   embedded: input = embed_size  + 27
# The CLI flags remain as fallbacks if inference fails, and the head count
# always comes from the CLI (it is not recoverable from shapes).
# ---------------------------------------------------------------------------
try:
    _layer_ids = {int(k.split(".")[2]) for k in state_dict
                  if k.startswith("network.layers.")}
    _ff_w = state_dict["network.layers.0.linear1.weight"]
    _n_layers    = max(_layer_ids) + 1
    _ff          = _ff_w.shape[0]
    _padded_size = _ff_w.shape[1]

    if "map_embedder.weight" in state_dict:
        _emb_w = state_dict["map_embedder.weight"]
        inferred_type = "embedded"
        _embed_size   = _emb_w.shape[1]
        _ckpt_mapsize = _emb_w.shape[0]          # max_embed_value == map_size
        if _ckpt_mapsize != args.map_h * args.map_w:
            raise SystemExit(
                f"Checkpoint was trained on a {_ckpt_mapsize}-cell map "
                f"(likely {int(_ckpt_mapsize ** 0.5)}x"
                f"{int(_ckpt_mapsize ** 0.5)}), but --map-h/--map-w give "
                f"{args.map_h}x{args.map_w} = {args.map_h * args.map_w} "
                f"cells. Pass the map size it was trained on.")
        _input_size = _embed_size + 27
    else:
        inferred_type = "base"
        _embed_size   = args.embed_size
        _input_size   = args.map_h * args.map_w + 27

    _padding = _padded_size - _input_size
    if _padding < 0:
        raise SystemExit(
            f"Checkpoint transformer width is {_padded_size}, smaller than "
            f"the input size {_input_size} implied by a "
            f"{args.map_h}x{args.map_w} map — this checkpoint was trained "
            f"on a smaller map. Pass the correct --map-h/--map-w.")

    if inferred_type != args.agent_type:
        print(f"Note: checkpoint is the '{inferred_type}' variant — "
              f"overriding --agent-type {args.agent_type}.")
        args.agent_type = inferred_type
    for _name, _cli, _inf in [("transformer-layers", args.transformer_layers, _n_layers),
                              ("feed-forward-neurons", args.feed_forward_neurons, _ff),
                              ("padding", args.padding, _padding),
                              ("embed-size", args.embed_size, _embed_size)]:
        if _cli != _inf:
            print(f"Note: --{_name} inferred from checkpoint: {_inf} "
                  f"(CLI gave {_cli}).")
    args.transformer_layers   = _n_layers
    args.feed_forward_neurons = _ff
    args.padding              = _padding
    args.embed_size           = _embed_size
except SystemExit:
    raise
except Exception as _e:                                      # noqa: BLE001
    print(f"WARNING: could not infer the architecture from the checkpoint "
          f"({type(_e).__name__}: {_e}) — using the CLI flags as given.")
    _padded_size = (args.embed_size if args.agent_type == "embedded"
                    else args.map_h * args.map_w) + 27 + args.padding

# The transformer requires the head count to divide the (padded) width.
# Checked here so a bad combination fails BEFORE the progress bar starts.
if _padded_size % args.attention_heads != 0:
    _divs = [h for h in range(1, _padded_size + 1) if _padded_size % h == 0]
    raise SystemExit(
        f"--attention-heads {args.attention_heads} does not divide the "
        f"transformer width {_padded_size}. Valid head counts for this "
        f"checkpoint/map: {_divs}")

print(f"Agent: {args.agent_type} | layers={args.transformer_layers}, "
      f"heads={args.attention_heads}, ff={args.feed_forward_neurons}, "
      f"padding={args.padding}"
      + (f", embed={args.embed_size}" if args.agent_type == "embedded" else "")
      + f" | map={args.map_h}x{args.map_w}")

# Resolve the map path — explicit override, else base name + optional variant
if args.map_path:
    map_path = args.map_path
else:
    map_path = (f"maps/{args.map_h}x{args.map_w}/"
                f"basesWorkers{args.map_h}x{args.map_w}{args.map_variant}.xml")
print(f"Map: {map_path}")

# Step delay controls render speed
step_delay = (1.0 / args.game_speed) if args.game_speed > 0 else 0.0

# Opponents to evaluate — either one or all
if args.opponent == "all":
    # Fast bots first: with incremental CSV writing, aborting a long sweep
    # then still leaves most opponents recorded.
    opponents_to_eval = sorted(ALL_AIS, key=lambda a: (a in SLOW_AIS,
                                                       ALL_AIS.index(a)))
else:
    opponents_to_eval = [args.opponent]

# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------
# Writing with plain print() while a tqdm bar is live garbles the display, so
# all in-loop output goes through tqdm.write() when a bar is active.

_pbar = None
_live_chart = None       # set in the main section when --live-chart is given

def _close_pbar_on_error(exc_type, exc, tb):
    """Close a live progress bar before the traceback prints — otherwise
    tqdm's __del__ runs at interpreter shutdown and buries the real error
    under its own TypeError."""
    global _pbar
    if _pbar is not None:
        try:
            _pbar.close()
        except Exception:                                    # noqa: BLE001
            pass
        _pbar = None
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = _close_pbar_on_error

def _write(msg):
    if _pbar is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Side swapping
# ---------------------------------------------------------------------------
# MicroRTSGridModeVecEnv has no ai1s parameter — the agent is always player 1.
# To play the other side we rewrite the map so player 0 and player 1 own each
# other's starting units. The agent stays player 1 but now begins from the
# opposite position. Neutral units (player="-1", i.e. resources) are untouched.

def make_swapped_map(rel_map_path):
    """
    Build a side-swapped copy of a map and return a path gym-microrts accepts.

    gym-microrts resolves map_paths internally against its own package
    directory (the Java client builds a URL from the string, so an ABSOLUTE
    path produces "MalformedURLException: spec is null"). The swapped map is
    therefore written next to the built-in maps and referenced by the same
    kind of relative path, e.g. maps/16x16/basesWorkers16x16_sideswap.xml
    """
    import xml.etree.ElementTree as ET
    import gym_microrts

    microrts_root = os.path.join(os.path.dirname(gym_microrts.__file__),
                                 "microrts")
    src = os.path.join(microrts_root, rel_map_path)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Source map not found: {src}\n"
            f"(resolved from --map-path/--map-variant against {microrts_root})")

    # Same directory as the source so the relative path stays valid
    rel_dir  = os.path.dirname(rel_map_path)
    stem     = os.path.splitext(os.path.basename(rel_map_path))[0]
    rel_dst  = os.path.join(rel_dir, f"{stem}_sideswap.xml")
    abs_dst  = os.path.join(microrts_root, rel_dst)

    tree = ET.parse(src)
    root = tree.getroot()
    n = 0
    for el in root.iter():
        p = el.get("player")
        if p == "0":
            el.set("player", "1"); n += 1
        elif p == "1":
            el.set("player", "0"); n += 1
        # player="-1" (neutral resources) deliberately untouched
    if n == 0:
        raise RuntimeError(
            f"No player-owned units found in {src} — cannot swap sides.")

    try:
        tree.write(abs_dst, encoding="UTF-8", xml_declaration=True)
    except OSError as e:
        raise OSError(
            f"Could not write the swapped map to {abs_dst}: {e}\n"
            f"gym-microrts can only load maps from inside its own package "
            f"directory, so that location must be writable. Either fix the "
            f"permissions or run with --sides p1."
        ) from e

    _write(f"  Side-swapped map: {rel_dst}  ({n} unit ownerships inverted)")
    return rel_dst


# ---------------------------------------------------------------------------
# Evaluation function — runs N games against a single opponent
# ---------------------------------------------------------------------------

def _obs_to_tensor(raw_obs, device):
    """
    Convert the env's observation to a float tensor on `device` without the
    element-wise fallback that torch.Tensor(...) takes for anything that is
    not already a contiguous float ndarray. np.asarray handles lists of
    per-env arrays and JPype arrays in C; from_numpy then shares memory and
    the cast/transfer is a single kernel.
    """
    if not isinstance(raw_obs, np.ndarray):
        raw_obs = np.asarray(raw_obs)
    if raw_obs.dtype == object:
        raw_obs = np.stack([np.asarray(o) for o in raw_obs])
    arr = np.ascontiguousarray(raw_obs)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32,
                                    non_blocking=True)


class _FastMaskEnv:
    """
    Thin wrapper handed to agent.get_action() in place of the raw env.

    MixedEmbeddedAgent.get_action fetches invalid-action masks with
    np.array(envs.vec_client.getMasks(0)), which converts a nested JPype
    JArray of shape (P, h*w, 79) element by element in Python. At P=1 that is
    tolerable; at P=250 it is ~5M Java->Python conversions per step, and the
    JVM sits idle while Python does them - which is why --parallel showed no
    CPU scaling. Routing the fetch through envs.get_action_mask() uses the
    vec env's buffer-backed path (the same one the FeudalNet evaluator uses),
    and also performs get_action_mask()'s source_unit_mask side effect that
    step() relies on.
    """
    def __init__(self, envs):
        self._envs = envs
        self.action_space = envs.action_space
        self.vec_client = self          # get_action calls envs.vec_client.getMasks(0)

    def getMasks(self, _player):
        return np.asarray(self._envs.get_action_mask())

    def __getattr__(self, name):        # anything else falls through to the env
        return getattr(self._envs, name)


def evaluate_opponent(opponent_name, num_games, device, args,
                      step_delay, eval_map_path=None, side='p1'):
    """
    Run num_games games against opponent_name.
    Returns (wins, losses, draws, lengths, game_records).
    """
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from gym_microrts import microrts_ai

    # Build AI class map dynamically from microrts_ai module
    ai_cls = getattr(microrts_ai, opponent_name, None)
    if ai_cls is None:
        print(f"  WARNING: {opponent_name} not found in microrts_ai — skipping")
        return 0, 0, 0, {"win": [], "loss": [], "draw": []}, []

    if opponent_name in ("mayari", "naiveMCTSAI", "guidedRojoA3N",
                         "izanagi", "droplet", "tiamat", "mixedBot"):
        _write(f"  NOTE: {opponent_name} is compute-heavy (~100 steps/s) — "
              f"use a modest --num-games.")

    # Parallel games: P bot envs stepped together, one batched forward pass.
    P = max(1, min(args.parallel, num_games))

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=P,
        max_steps=2000,
        render_theme=2,
        partial_obs=False,
        ai2s=[ai_cls] * P,
        map_paths=[(eval_map_path or map_path)] * P,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )

    mapsize = args.map_h * args.map_w

    # Rebuild the agent per opponent — its constructor takes the env, and the
    # action space may differ between maps/opponents.
    if args.agent_type == 'base':
        agent = WeightedAgent(
            mapsize, envs, device, args.transformer_layers,
            args.feed_forward_neurons, args.attention_heads,
            args.padding).to(device)
        feature_map = reshape_observation_extended_fast
    else:  # embedded
        agent = MixedEmbeddedAgent(
            mapsize, envs, device, args.transformer_layers,
            args.feed_forward_neurons, args.attention_heads,
            args.padding, args.embed_size).to(device)
        feature_map = reshape_observation_mixed_embedded_fast

    agent.load_state_dict(state_dict)
    agent.eval()

    wins = losses = draws = 0
    len_win, len_loss, len_draw = [], [], []
    game_records = []

    def _record(env_result, ep_steps, raw_rewards, game_no):
        """Book one finished game: tallies, length lists, CSV record, prints."""
        nonlocal wins, losses, draws
        if env_result > 0:
            result = "WIN";  wins   += 1; len_win.append(ep_steps)
        elif env_result < 0:
            result = "LOSS"; losses += 1; len_loss.append(ep_steps)
        else:
            result = "DRAW"; draws  += 1; len_draw.append(ep_steps)

        rec = {
            "opponent": opponent_name,
            "game":     game_no,
            "side":     side,
            "result":   result,
            "steps":    ep_steps,
        }
        for j, fn in enumerate(REWARD_FUNCTIONS):
            rec[f"reward_{fn}"] = (float(raw_rewards[j])
                                   if j < len(raw_rewards) else 0.0)
        game_records.append(rec)

        win_pct = 100 * wins / game_no
        _write(
            f"  Game {game_no:>3}/{num_games} | {result:>5} | "
            f"steps={ep_steps:>4} | "
            f"W={wins} L={losses} D={draws} | {win_pct:.0f}%"
        )
        if _live_chart is not None:
            _live_chart.record(opponent_name, result)
        if _pbar is not None:
            _pbar.update(1)
            _pbar.set_postfix_str(
                f"{opponent_name} W={wins} L={losses} D={draws} "
                f"({win_pct:.0f}%)")

    # Fast mask path for the agent (see _FastMaskEnv) - the raw env is still
    # used for reset()/step()/render().
    mask_env = _FastMaskEnv(envs)

    if P == 1:
        # ------------------- Sequential: one game at a time -------------------
        for game in range(num_games):
            (next_obs, next_entity_mask, next_entity_count, next_unit_position,
             next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask) = \
                feature_map(_obs_to_tensor(envs.reset(), device), device)

            ep_steps  = 0
            game_done = False

            while not game_done:
                if not args.no_render:
                    envs.render()
                    if step_delay:
                        time.sleep(step_delay)

                with torch.no_grad():
                    action, _, _, _ = agent.get_action(next_obs,
                                                       next_entity_mask,
                                                       next_entity_count,
                                                       next_unit_position,
                                                       next_unit_mask,
                                                       envs=mask_env)

                # gym-microrts 0.6.0: step() takes the per-cell action array of
                # shape (num_envs, h*w, 7) directly — the env applies the source
                # unit mask and builds the Java arrays internally, so the manual
                # JArray valid-action filtering from the old API is unnecessary.
                actions_np = np.array(action.cpu().numpy(), dtype=np.int32)
                raw_obs, reward, done, infos = envs.step(actions_np)

                (next_obs, next_entity_mask, next_entity_count,
                 next_unit_position, next_unit_mask, next_enemy_unit_mask,
                 next_neutral_unit_mask) = \
                    feature_map(_obs_to_tensor(raw_obs, device), device)
                ep_steps += 1

                if done[0]:
                    game_done = True
                    raw_rewards = infos[0].get("raw_rewards", np.zeros(6))
                    _record(float(raw_rewards[0]), ep_steps, raw_rewards,
                            game + 1)

    else:
        # ---------------- Parallel: P games stepped as one batch ----------------
        # Each env slot owns a fixed QUOTA of games and only its first
        # `quota[i]` completions are counted. This matters statistically:
        # counting the first num_games completions across slots would
        # over-sample short games (quick wins finish first), biasing the win
        # rate. With per-slot quotas every counted game is an independent
        # playthrough regardless of its length. Slots that exhaust their
        # quota keep simulating until the batch finishes (a vec env cannot
        # shrink); those results are discarded.
        _prof = [0.0, 0.0, 0.0, 0, 0.0]  # agent, envs.step, obs->tensor, n, feature_map
        base_q, rem = divmod(num_games, P)
        quota     = [base_q + (1 if i < rem else 0) for i in range(P)]
        completed = [0] * P
        finished  = 0

        (next_obs, next_entity_mask, next_entity_count, next_unit_position,
         next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask) = \
            feature_map(_obs_to_tensor(envs.reset(), device), device)
        ep_steps = np.zeros(P, dtype=int)

        while finished < num_games:
            if not args.no_render:
                envs.render()          # renders env 0
                if step_delay:
                    time.sleep(step_delay)

            _t0 = time.perf_counter()
            with torch.no_grad():
                action, _, _, _ = agent.get_action(next_obs,
                                                   next_entity_mask,
                                                   next_entity_count,
                                                   next_unit_position,
                                                   next_unit_mask,
                                                   envs=mask_env)

            actions_np = np.array(action.cpu().numpy(), dtype=np.int32)
            _t1 = time.perf_counter()
            raw_obs, reward, done, infos = envs.step(actions_np)
            _t2 = time.perf_counter()

            if args.profile and _prof[3] == 0:
                _write(f"  [profile] raw_obs type={type(raw_obs).__name__}"
                       + (f" dtype={raw_obs.dtype} shape={raw_obs.shape}"
                          f" contiguous={raw_obs.flags['C_CONTIGUOUS']}"
                          if isinstance(raw_obs, np.ndarray) else
                          f" (not a numpy array: len={len(raw_obs)}, "
                          f"elem={type(raw_obs[0]).__name__})"))
            obs_t = _obs_to_tensor(raw_obs, device)
            _t3 = time.perf_counter()
            (next_obs, next_entity_mask, next_entity_count,
             next_unit_position, next_unit_mask, next_enemy_unit_mask,
             next_neutral_unit_mask) = feature_map(obs_t, device)
            _t4 = time.perf_counter()
            ep_steps += 1

            if args.profile:
                _prof[0] += _t1 - _t0; _prof[1] += _t2 - _t1
                _prof[2] += _t3 - _t2; _prof[4] += _t4 - _t3; _prof[3] += 1
                if _prof[3] % 200 == 0:
                    tot = _prof[0] + _prof[1] + _prof[2] + _prof[4]
                    _write(f"  [profile] P={P} | per step {1000*tot/200:.1f} ms | "
                           f"agent {100*_prof[0]/tot:.0f}% | "
                           f"envs.step (Java) {100*_prof[1]/tot:.0f}% | "
                           f"obs->tensor {100*_prof[2]/tot:.0f}% | "
                           f"feature map {100*_prof[4]/tot:.0f}% | "
                           f"{200*P/tot:.0f} env-steps/s")
                    _prof[:] = [0.0, 0.0, 0.0, 0, 0.0]

            for i in range(P):
                if not done[i]:
                    continue
                # gym-microrts auto-resets a finished env: raw_obs[i] is
                # already the next episode's first observation.
                if completed[i] < quota[i]:
                    completed[i] += 1
                    finished     += 1
                    raw_rewards = infos[i].get("raw_rewards", np.zeros(6))
                    _record(float(raw_rewards[0]), int(ep_steps[i]),
                            raw_rewards, finished)
                ep_steps[i] = 0

    # NOTE: deliberately NOT calling envs.close().
    # MicroRTSGridModeVecEnv.close() calls jpype.shutdownJVM(), which tears
    # down the JVM for the ENTIRE process — and JPype cannot restart a JVM.
    # With --opponent all that made the second opponent fail with
    # "OSError: JVM cannot be restarted". Creating additional envs inside a
    # live JVM is fine, so we leave each one open and let process exit clean
    # up. Each env is a single bot env, so the leak is negligible.
    return (wins, losses, draws,
            {"win": len_win, "loss": len_loss, "draw": len_draw},
            game_records)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

print(f"\nEvaluating: {args.checkpoint}")
print(f"Games per opponent: {args.num_games}"
      + (f" x {args.repeats} repeats" if args.repeats > 1 else ""))
print(f"Map: {map_path}")
print(f"Render: " + ("OFF" if args.no_render else f"ON ({args.game_speed} FPS)"))
if args.parallel > 1:
    print(f"Parallel games per opponent: {args.parallel}")
print("=" * 60)

# --- Side plan: how many games from each starting position ---
_swap_map = None
if args.sides in ('p2', 'both'):
    _swap_map = make_swapped_map(map_path)

if args.sides == 'p1':
    side_plan = [('p1', args.num_games, None)]
elif args.sides == 'p2':
    side_plan = [('p2', args.num_games, _swap_map)]
else:  # both — split evenly, remainder to p1
    n_p2 = args.num_games // 2
    n_p1 = args.num_games - n_p2
    side_plan = [('p1', n_p1, None),
                 ('p2', n_p2, _swap_map)]
print(f"Agent side(s): "
      + ", ".join(f"{s}={n} games" for s, n, _ in side_plan))

# Summary table — collects results across all opponents
summary = []
all_game_records = []

# ---------------------------------------------------------------------------
# Incremental CSV setup — files are written/updated after EACH opponent so a
# crash partway through a long sweep preserves everything already completed.
# ---------------------------------------------------------------------------

csv_enabled  = not args.no_csv
games_path   = summary_path = None
csv_context  = {}

if csv_enabled:
    if args.csv_out:
        csv_base = args.csv_out
    else:
        _ckpt  = os.path.splitext(os.path.basename(args.checkpoint))[0]
        _map   = os.path.splitext(os.path.basename(map_path))[0]
        _stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_base = os.path.join("results", f"eval_{_ckpt}_{_map}_{_stamp}")

    os.makedirs(os.path.dirname(csv_base) or ".", exist_ok=True)
    games_path   = f"{csv_base}_games.csv"
    summary_path = f"{csv_base}_summary.csv"

    csv_context = {
        "checkpoint": args.checkpoint,
        "map":        map_path,
        "sides":      args.sides,
        "seed":       args.seed,
        "num_games":  args.num_games,
        "repeats":    args.repeats,
        "parallel":   args.parallel,
    }

    GAME_FIELDS = (["opponent", "game", "repeat", "side", "result", "steps"]
                   + [f"reward_{fn}" for fn in REWARD_FUNCTIONS]
                   + list(csv_context.keys()))

    # Write the per-game header once; rows are appended per opponent
    with open(games_path, "w", newline="") as _f:
        csv.DictWriter(_f, fieldnames=GAME_FIELDS).writeheader()

    def append_games_csv(records):
        """Append one opponent's games and flush to disk immediately."""
        if not records:
            return
        with open(games_path, "a", newline="") as _f:
            w = csv.DictWriter(_f, fieldnames=GAME_FIELDS)
            for rec in records:
                w.writerow({**rec, **csv_context})

    def write_summary_csv(summary_rows, game_records):
        """
        Rewrite the vertical summary from scratch. Each opponent is a column,
        so adding one changes the layout — rewriting the whole (small) file is
        simpler and safer than trying to patch columns in place.
        """
        if not summary_rows:
            return
        rows = [{k: v for k, v in r.items()
                 if k not in ("len_all", "len_win")} for r in summary_rows]

        if len(rows) > 1:
            tw = sum(r["wins"] for r in rows)
            tl = sum(r["losses"] for r in rows)
            td = sum(r["draws"] for r in rows)
            tg = tw + tl + td
            a  = np.array([g["steps"] for g in game_records], dtype=float)
            rows.append({
                "opponent": "OVERALL", "games": tg,
                "wins": tw, "losses": tl, "draws": td,
                "win_pct":  round(100 * tw / max(tg, 1), 2),
                # Per-repeat OVERALL win rates: pool every opponent's games
                # of repeat r, then take the std across repeats.
                **(lambda reps: {
                    "win_pct_std": (round(float(np.std(reps, ddof=1)), 2)
                                    if len(reps) > 1 else ""),
                    "win_pct_reps": ";".join(f"{x:g}" for x in reps),
                })([
                    round(100 * sum(g["result"] == "WIN" for g in grp)
                          / max(len(grp), 1), 2)
                    for r_id in sorted({g.get("repeat", 1)
                                        for g in game_records})
                    for grp in [[g for g in game_records
                                 if g.get("repeat", 1) == r_id]]
                ]),
                "loss_pct": round(100 * tl / max(tg, 1), 2),
                "draw_pct": round(100 * td / max(tg, 1), 2),
                "capped":   sum(r["capped"] for r in rows),
                "steps_all_mean":   round(float(a.mean()), 2) if len(a) else "",
                "steps_all_std":    (round(float(a.std(ddof=1)), 2)
                                     if len(a) > 1 else 0.0),
                "steps_all_median": round(float(np.median(a)), 2) if len(a) else "",
                "steps_all_min":    round(float(a.min()), 2) if len(a) else "",
                "steps_all_max":    round(float(a.max()), 2) if len(a) else "",
            })

        metric_order = (
            ["games", "wins", "losses", "draws",
             "win_pct", "win_pct_std", "win_pct_reps",
             "loss_pct", "draw_pct", "capped"]
            + [f"steps_{lbl}_{st}"
               for lbl in ["all", "win", "loss", "draw"]
               for st in ["mean", "std", "median", "min", "max"]]
            + [f"{sl}_{m}" for sl in ("p1", "p2")
               for m in ("wins", "losses", "draws", "win_pct")]
            + [f"mean_reward_{fn}" for fn in REWARD_FUNCTIONS]
        )
        extras, seen = [], set()
        for r in rows:
            for k in r:
                if k not in metric_order and k != "opponent" and k not in seen:
                    seen.add(k)
                    extras.append(k)
        metric_order += extras

        opponents = [r["opponent"] for r in rows]
        by_opp    = {r["opponent"]: r for r in rows}

        # Wide form for programmatic use: pd.read_csv(path, index_col=0).T
        with open(summary_path, "w", newline="") as _f:
            w = csv.writer(_f)
            w.writerow(["metric"] + opponents)
            for k, v in csv_context.items():
                w.writerow([k] + [v] * len(opponents))
            w.writerow([])
            for m in metric_order:
                w.writerow([m] + [by_opp[o].get(m, "") for o in opponents])

    print(f"\nCSV output (updated after each opponent):")
    print(f"  per-game -> {games_path}")
    print(f"  summary  -> {summary_path}")

# --- Live chart (opt-in) ---
# Created before the loop so every opponent has a row from the start and the
# axis does not reshuffle as the sweep progresses.
if args.live_chart:
    from live_winrate import LiveWinrateChart
    _live_chart = LiveWinrateChart(
        opponents_to_eval,
        games_per_opponent=args.num_games * args.repeats,
        title=f"{os.path.basename(args.checkpoint)} — "
              f"{os.path.basename(map_path)}",
    )
    if not _live_chart.enabled:
        _live_chart = None

total_games = len(opponents_to_eval) * args.num_games * args.repeats
if not _HAS_TQDM:
    print("(tqdm not installed — progress bar disabled; "
          "pip install tqdm to enable it)")

_pbar = tqdm(total=total_games, unit="game", dynamic_ncols=True,
             smoothing=0.05,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")

for _opp_i, opponent in enumerate(opponents_to_eval, 1):
    _pbar.set_description(f"[{_opp_i}/{len(opponents_to_eval)}] {opponent}")
    _write(f"\nOpponent: {opponent}")
    _write("-" * 40)

    wins = losses = draws = 0
    lengths      = {"win": [], "loss": [], "draw": []}
    game_records = []
    side_tally   = {}
    rep_win_pcts = []          # one win rate per repeat, for the std

    for rep_i in range(1, args.repeats + 1):
        if _live_chart is not None:
            _live_chart.set_repeat(rep_i)
        if args.repeats > 1:
            _write(f"  == repeat {rep_i}/{args.repeats} ==")
            # Reproducible per repeat: the r-th repeat of any run uses the
            # same agent RNG stream. (Bot randomness lives in the JVM and is
            # time-seeded there — repeats differ regardless.)
            torch.manual_seed(args.seed + rep_i - 1)
            np.random.seed(args.seed + rep_i - 1)

        r_wins = r_games = 0

        for side_label, n_games, side_map in side_plan:
            if n_games <= 0:
                continue
            if len(side_plan) > 1:
                _write(f"  -- agent as {side_label} ({n_games} games) --")
            s_wins, s_losses, s_draws, s_lengths, s_recs = \
                evaluate_opponent(
                    opponent_name=opponent,
                    num_games=n_games,
                    device=device,
                    args=args,
                    step_delay=step_delay,
                    eval_map_path=side_map,
                    side=side_label,
                )
            wins   += s_wins
            losses += s_losses
            draws  += s_draws
            r_wins  += s_wins
            r_games += s_wins + s_losses + s_draws
            for k in lengths:
                lengths[k].extend(s_lengths[k])
            for rec in s_recs:
                rec["repeat"] = rep_i
            game_records.extend(s_recs)
            t = side_tally.setdefault(side_label, {"wins": 0, "losses": 0,
                                                   "draws": 0})
            t["wins"]   += s_wins
            t["losses"] += s_losses
            t["draws"]  += s_draws
            tn = t["wins"] + t["losses"] + t["draws"]
            t["win_pct"] = round(100 * t["wins"] / max(tn, 1), 2)

        rep_win_pcts.append(round(100 * r_wins / max(r_games, 1), 2))

    if args.repeats > 1:
        _rp = np.array(rep_win_pcts, dtype=float)
        _write(f"\n  Win rate over {args.repeats} repeats of "
               f"{args.num_games} games: "
               f"{_rp.mean():.1f}% +/- {_rp.std(ddof=1):.1f}%  "
               f"(per repeat: {', '.join(f'{x:.0f}%' for x in _rp)})")

    total   = wins + losses + draws
    win_pct = 100 * wins   / max(total, 1)
    los_pct = 100 * losses / max(total, 1)
    drw_pct = 100 * draws  / max(total, 1)

    # --- Episode-length statistics (report these in write-ups) ---
    MAX_EP_STEPS = 2000   # must match max_steps in the env above
    all_len = lengths["win"] + lengths["loss"] + lengths["draw"]

    def stats(v):
        if not v:
            return None
        a = np.array(v, dtype=float)
        return {"n": len(a), "mean": a.mean(), "std": a.std(ddof=1) if len(a) > 1 else 0.0,
                "median": np.median(a), "min": a.min(), "max": a.max()}

    if len(side_tally) > 1:
        _write("\n  By starting side:")
        for sl, t in side_tally.items():
            _write(f"    {sl}: W={t['wins']} L={t['losses']} D={t['draws']} "
                   f"({t['win_pct']:.1f}%)")

    _write("\n  Episode length (steps):")
    for label in ["win", "loss", "draw"]:
        s = stats(lengths[label])
        if s:
            _write(f"    {label:<5} n={s['n']:<4} "
                  f"mean={s['mean']:7.1f} +/- {s['std']:6.1f}  "
                  f"median={s['median']:6.1f}  "
                  f"range=[{s['min']:.0f}, {s['max']:.0f}]")
    s_all = stats(all_len)
    if s_all:
        _write(f"    {'ALL':<5} n={s_all['n']:<4} "
              f"mean={s_all['mean']:7.1f} +/- {s_all['std']:6.1f}  "
              f"median={s_all['median']:6.1f}  "
              f"range=[{s_all['min']:.0f}, {s_all['max']:.0f}]")
    n_capped = sum(1 for x in all_len if x >= MAX_EP_STEPS)
    _write(f"    games reaching the {MAX_EP_STEPS}-step cap: {n_capped}"
          + ("  <-- lengths are right-censored; mean understates true value"
             if n_capped else "  (none — lengths are uncensored)"))

    all_game_records.extend(game_records)

    row = {
        "opponent":   opponent,
        "games":      total,
        "wins":       wins,
        "losses":     losses,
        "draws":      draws,
        "win_pct":    round(win_pct, 2),
        "win_pct_std": (round(float(np.std(rep_win_pcts, ddof=1)), 2)
                        if len(rep_win_pcts) > 1 else ""),
        "win_pct_reps": ";".join(f"{x:g}" for x in rep_win_pcts),
        "loss_pct":   round(los_pct, 2),
        "draw_pct":   round(drw_pct, 2),
        "capped":     n_capped,
    }
    for label in ["all", "win", "loss", "draw"]:
        s = s_all if label == "all" else stats(lengths[label])
        for stat in ["mean", "std", "median", "min", "max"]:
            row[f"steps_{label}_{stat}"] = (round(float(s[stat]), 2)
                                            if s else "")
    # Per-side breakdown — exposes starting-position bias
    for sl in ("p1", "p2"):
        t = side_tally.get(sl)
        row[f"{sl}_wins"]    = t["wins"]    if t else ""
        row[f"{sl}_losses"]  = t["losses"]  if t else ""
        row[f"{sl}_draws"]   = t["draws"]   if t else ""
        row[f"{sl}_win_pct"] = t["win_pct"] if t else ""

    # Mean reward components across this opponent's games
    for fn in REWARD_FUNCTIONS:
        vals = [g[f"reward_{fn}"] for g in game_records]
        row[f"mean_reward_{fn}"] = (round(float(np.mean(vals)), 3)
                                    if vals else "")

    summary.append({**row, "len_all": s_all,
                    "len_win": stats(lengths["win"])})

    # --- Flush this opponent's results to disk before starting the next ---
    if csv_enabled:
        append_games_csv(game_records)
        write_summary_csv(summary, all_game_records)
        _write(f"  CSV updated ({len(all_game_records)} games written so far)")

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

if _pbar is not None:
    _pbar.close()
    _pbar = None

if _live_chart is not None:
    _chart_path = args.live_chart_out
    if _chart_path is None and csv_enabled:
        _chart_path = f"{csv_base}_livechart.png"
    _live_chart.finish(save_path=_chart_path,
                       keep_open=args.live_chart_keep_open)

print("\n" + "=" * 60)
print(f"{'Opponent':<22} {'W':>4} {'L':>4} {'D':>4} {'Win%':>6} "
      f"{'steps mean+/-sd':>18} {'cap':>4}")
print("-" * 60)

total_w = total_l = total_d = 0
for r in summary:
    L = r["len_all"]
    len_str = (f"{L['mean']:.0f} +/- {L['std']:.0f}" if L else "n/a")
    print(
        f"{r['opponent']:<22} {r['wins']:>4} {r['losses']:>4} "
        f"{r['draws']:>4} {r['win_pct']:>5.1f}% {len_str:>18} "
        f"{r['capped']:>4}"
    )
    total_w += r['wins']
    total_l += r['losses']
    total_d += r['draws']

if len(summary) > 1:
    total   = total_w + total_l + total_d
    overall = 100 * total_w / max(total, 1)
    print("-" * 60)
    print(f"{'OVERALL':<22} {total_w:>4} {total_l:>4} {total_d:>4} {overall:>5.1f}%")

print("=" * 60)

if csv_enabled:
    print(f"\nResults written:")
    print(f"  {games_path}    ({len(all_game_records)} rows, one per match)")
    print(f"  {summary_path}  ({len(summary)} opponent column(s) + OVERALL)")
