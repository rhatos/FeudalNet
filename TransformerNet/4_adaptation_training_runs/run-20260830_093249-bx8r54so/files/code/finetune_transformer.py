"""
finetune_transformer.py

Continue training a transformer-agent checkpoint (WeightedAgent /
MixedEmbeddedAgent) against ONE specified opponent — the transformer
counterpart of finetune_opponent.py, with the same structure: architecture
from the checkpoint, resume/export support, a single opponent for both
training and evaluation, run-config CSV, and self-describing output
checkpoints.

The OPTIMISATION is the transformer's own regimen, transplanted from its
original training script (train_embedded_agent.py): PPO with GAE, shuffled
minibatch epochs, clipped policy and value losses, and invalid-action-masked
categoricals — updated for gym-microrts 0.6.0 (map_paths / partial_obs /
numpy step; the old manual JArray action filtering is gone).

Architecture resolution:
  * fine-tuning checkpoints written by THIS script are self-describing
    ({"model", "args", ...}) — everything is restored from them
  * ORIGINAL transformer checkpoints are bare state_dicts, so the
    architecture is inferred from tensor shapes (layers, feed-forward width,
    padding, embedded-vs-base, embed size, and — for embedded — the map).
    Only --attention-heads (and --map-h/--map-w for the base variant) cannot
    be inferred and fall back to the CLI.

Usage:
    python finetune_transformer.py \\
        --checkpoint models/transformer.pt \\
        --opponent izanagi

    # Shorter run, gentler settings, custom output
    python finetune_transformer.py \\
        --checkpoint models/transformer.pt \\
        --opponent mayari --max-steps 20000000 \\
        --lr 3e-5 --ent-coef 0.02 \\
        --out-checkpoint models/transformer_vs_mayari.pt

    # Resume after a crash: the SAME command plus --resume
    python finetune_transformer.py \\
        --checkpoint models/transformer.pt \\
        --opponent izanagi --resume
"""

import os
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import wandb

from transformer_agent.mixed_embedded_agent import (
    MixedEmbeddedAgent, reshape_observation_mixed_embedded)
from transformer_agent.weighted_agent import (
    WeightedAgent, reshape_observation_extended)
from run_config import write_run_config


def str2bool(v):
    """Proper bool parsing — argparse type=bool treats any non-empty string as True."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


REWARD_FUNCTIONS = [
    "WinLossRewardFunction",
    "ResourceGatherRewardFunction",
    "ProduceWorkerRewardFunction",
    "ProduceBuildingRewardFunction",
    "AttackRewardFunction",
    "ProduceCombatUnitRewardFunction",
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Fine-tune a transformer-agent checkpoint against a single '
                'opponent')

# --- Required: what to fine-tune, and against whom ---
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Checkpoint to fine-tune. Architecture is restored '
                         'from it (self-describing fine-tune checkpoints) or '
                         'inferred from its tensor shapes (bare originals).')
parser.add_argument('--opponent',   type=str, required=True,
                    help='The single opponent used for BOTH training and '
                         'evaluation (e.g. izanagi, mayari, coacAI)')

# --- Fine-tuning schedule ---
parser.add_argument('--max-steps',    type=int,   default=int(1e8),
                    help='Env steps to fine-tune for (default 100M). The new '
                         'run counts from 0 regardless of the source step.')
parser.add_argument('--lr',           type=float, default=2.5e-4,
                    help='Learning rate. Default matches original training; '
                         'for a gentler fine-tune consider e.g. 5e-5')
parser.add_argument('--ent-coef',     type=float, default=0.01,
                    help='Entropy coefficient (original training default). '
                         'Raising it (e.g. 0.02) re-opens exploration '
                         'against the new opponent')
parser.add_argument('--anneal-lr',    type=str2bool, default=True,
                    help='Linear LR decay across the update budget, exactly '
                         'as in original training (their default). Set false '
                         'for a flat LR.')
parser.add_argument('--cosine',       type=str2bool, default=False,
                    help='Cosine-anneal the LR instead (overrides '
                         '--anneal-lr)')
parser.add_argument('--reset-optimizer', type=str2bool, default=False,
                    help='Start Adam fresh instead of loading the moments '
                         'from the checkpoint (bare originals carry none)')

# --- Crash recovery ---
parser.add_argument('--resume', action='store_true',
                    help='Continue an interrupted fine-tuning run: re-use the '
                         'output checkpoint, its step count and its wandb run. '
                         'Safe to pass even if nothing was saved yet - it then '
                         'starts from the beginning. Use the SAME command as '
                         'the original run plus this flag.')
parser.add_argument('--resume-from', type=str, default=None,
                    help='Resume from a specific fine-tuning checkpoint '
                         'instead of the derived --out-checkpoint path')
parser.add_argument('--export-only', action='store_true',
                    help='Do not train: write a checkpoint from the latest '
                         'saved state and exit. Starts no JVM and no wandb '
                         'run, so it is instant.')

# --- PPO (the transformer\'s own regimen; defaults from its training) ---
parser.add_argument('--num-workers',   type=int,   default=24,
                    help='Bot envs, all playing --opponent')
parser.add_argument('--num-steps',     type=int,   default=256,
                    help='Rollout length per env per update')
parser.add_argument('--n-minibatch',   type=int,   default=4)
parser.add_argument('--update-epochs', type=int,   default=4)
parser.add_argument('--gamma',         type=float, default=0.99)
parser.add_argument('--gae',           type=str2bool, default=True)
parser.add_argument('--gae-lambda',    type=float, default=0.95)
parser.add_argument('--clip-coef',     type=float, default=0.1)
parser.add_argument('--clip-vloss',    type=str2bool, default=True)
parser.add_argument('--norm-adv',      type=str2bool, default=True)
parser.add_argument('--vf-coef',       type=float, default=0.5)
parser.add_argument('--max-grad-norm', type=float, default=0.5)
parser.add_argument('--cuda',          type=str2bool, default=True)
parser.add_argument('--seed',          type=int,   default=1,
                    help='0 means time-based, as in original training')
parser.add_argument('--reward-weights', type=float, nargs=6, default=None,
                    help='Override the 6 shaped-reward weights. Default: the '
                         'original training weights 20 1 1 0.2 1 4')
parser.add_argument('--sparse-rewards', type=str2bool, default=False,
                    help='Win/loss reward only (1 0 0 0 0 0)')

# --- Architecture fallbacks (used only when not restorable/inferable) ---
parser.add_argument('--agent-type', type=str, default='base',
                    choices=['base', 'embedded'])
parser.add_argument('--attention-heads', type=int, default=7,
                    help='NOT inferable from checkpoint shapes — must match '
                         'the value the checkpoint was trained with')
parser.add_argument('--map-h', type=int, default=8,
                    help='Only used for BASE-variant bare checkpoints '
                         '(embedded ones carry the map in their shapes)')
parser.add_argument('--map-w', type=int, default=8)
parser.add_argument('--embed-size', type=int, default=64,
                    help='Only used for EMBEDDED-variant bare checkpoints '
                         'whose embed size cannot be read (normally it can)')

# --- Evaluation (same single opponent) ---
parser.add_argument('--winrate-window', type=int, default=5,
                    help='Per-env rolling window for the live win-rate '
                         'readout: each env keeps its last N game outcomes, '
                         'and the console/wandb metric is the average of the '
                         'per-env win rates across all envs (draws count as '
                         'non-wins). Purely observational — no effect on '
                         'learning.')
parser.add_argument('--eval-every',   type=int,   default=int(1e7),
                    help='Evaluate every N env steps (0 disables)')
parser.add_argument('--eval-games',   type=int,   default=100)
parser.add_argument('--eval-max-steps', type=int, default=2000,
                    help='Episode cap for the in-training evaluator. Must '
                         'match the final evaluation protocol (2000 ticks, '
                         'undecided games are draws and count as losses) '
                         'or the early-stopping winrate is measured under '
                         'different game conditions than the reported one.')
parser.add_argument('--eval-winrate', type=float, default=1.0,
                    help='Stop early once this win rate is reached')
parser.add_argument('--eval-workers', type=int,   default=8)
parser.add_argument('--eval-record',  type=str2bool, default=True)
parser.add_argument('--eval-video-length', type=int, default=2000)

# --- Output ---
parser.add_argument('--wandb-project', type=str, default='fun-microrts')
parser.add_argument('--wandb-entity',  type=str, default=None)
parser.add_argument('--run-name',      type=str, default=None,
                    help='Defaults to "<checkpoint filename>-<opponent>"')
parser.add_argument('--save-dir',      type=str, default='models')
parser.add_argument('--out-checkpoint', type=str, default=None,
                    help='Where to write the fine-tuned model. Defaults to '
                         '<save-dir>/<checkpoint filename>-<opponent>.pt')
parser.add_argument('--run-id-file',   type=str, default='run_id_finetune.txt')
parser.add_argument('--config-csv', type=str, default=None,
                    help='Where to write the run parameter CSV. Default: '
                         '<save-dir>/configs/<run name>_<run id>_config.csv')
parser.add_argument('--no-config-csv', dest='config_csv_enabled',
                    action='store_false',
                    help='Skip writing the run parameter CSV')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Locate the file to continue from (resume support)
# ---------------------------------------------------------------------------

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

_ckpt_stem_early = os.path.splitext(os.path.basename(args.checkpoint))[0]
if args.out_checkpoint is None:
    args.out_checkpoint = os.path.join(
        args.save_dir, f"{_ckpt_stem_early}-{args.opponent}.pt")


def _latest_finetune_state(out_path, save_dir):
    """
    Find the most advanced fine-tuning state on disk.

    Considers the rolling output file and any numbered snapshots
    (<stem>_step=N.pt), returning whichever has the highest step. Snapshot
    steps come from the filename, so only one file is actually loaded.
    """
    import glob, re
    stem = os.path.splitext(os.path.basename(out_path))[0]
    best, best_step = None, -1

    for path in glob.glob(os.path.join(save_dir, f"{stem}_step=*.pt")):
        m = re.search(r"_step=(\d+)\.pt$", path)
        if m and int(m.group(1)) > best_step:
            best, best_step = path, int(m.group(1))

    if os.path.exists(out_path):
        try:
            s = int(torch.load(out_path, map_location="cpu",
                               weights_only=False).get("step", 0))
        except Exception:                                   # noqa: BLE001
            s = 0
        if s >= best_step:
            best, best_step = out_path, s

    return best, best_step


args.resume_path = None
if args.resume_from:
    if not os.path.exists(args.resume_from):
        raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")
    args.resume_path = args.resume_from
elif args.resume or args.export_only:
    _cand, _cand_step = _latest_finetune_state(args.out_checkpoint,
                                               args.save_dir)
    if _cand:
        args.resume_path = _cand
        print(f"Latest fine-tuning state: {_cand} (step {_cand_step:,})")
    elif args.export_only:
        print(f"No fine-tuning checkpoint found for {args.out_checkpoint} — "
              f"exporting from the source checkpoint instead.")
    else:
        print(f"--resume given but no fine-tuning checkpoint at "
              f"{args.out_checkpoint} — starting the fine-tune from the "
              f"beginning.")

# ---------------------------------------------------------------------------
# Restore / infer the architecture
# ---------------------------------------------------------------------------
# Fine-tuning checkpoints from this script carry an args dict. Bare original
# checkpoints do not, so their architecture is read from the tensor shapes —
# the same inference evaluate_agent.py uses:
#   network.layers.<N>.*            -> number of transformer layers
#   network.layers.0.linear1.weight -> (dim_feedforward, padded_size)
#   map_embedder.weight             -> embedded variant; (map_size, embed)

ARCH_KEYS = ["agent_type", "transformer_layers", "feed_forward_neurons",
             "padding", "embed_size", "attention_heads", "map_h", "map_w"]

_arch_src = args.resume_path or args.checkpoint
_ckpt  = torch.load(_arch_src, map_location="cpu", weights_only=False)
_state = _ckpt["model"] if isinstance(_ckpt, dict) and "model" in _ckpt \
    else _ckpt
_saved = dict(_ckpt.get("args", {})) if isinstance(_ckpt, dict) else {}

if all(k in _saved for k in ARCH_KEYS):
    print(f"Architecture restored from {_arch_src}:")
    for _k in ARCH_KEYS:
        setattr(args, _k, _saved[_k])
        print(f"    {_k:<22} = {_saved[_k]}")
    args.transformer_layers = int(args.transformer_layers)
    args.padding = int(args.padding)
else:
    _layer_ids = {int(k.split(".")[2]) for k in _state
                  if k.startswith("network.layers.")}
    _ff_w = _state["network.layers.0.linear1.weight"]
    args.transformer_layers   = max(_layer_ids) + 1
    args.feed_forward_neurons = _ff_w.shape[0]
    _padded_size              = _ff_w.shape[1]

    if "map_embedder.weight" in _state:
        args.agent_type = "embedded"
        _emb_w = _state["map_embedder.weight"]
        args.embed_size = _emb_w.shape[1]
        _cells = _emb_w.shape[0]
        _side  = int(round(_cells ** 0.5))
        if _side * _side == _cells and _cells != args.map_h * args.map_w:
            args.map_h = args.map_w = _side
        args.padding = _padded_size - (args.embed_size + 27)
    else:
        args.agent_type = "base"
        args.padding = _padded_size - (args.map_h * args.map_w + 27)

    if args.padding < 0:
        raise SystemExit(
            f"Checkpoint transformer width {_padded_size} is smaller than "
            f"the input implied by a {args.map_h}x{args.map_w} map — pass "
            f"the map size it was trained on via --map-h/--map-w.")
    _cli_keys = {"attention_heads"}
    if args.agent_type == "base":
        _cli_keys |= {"map_h", "map_w", "embed_size"}
    print(f"Architecture inferred from {_arch_src} shapes:")
    for _k in ARCH_KEYS:
        src = "CLI" if _k in _cli_keys else "shapes"
        print(f"    {_k:<22} = {getattr(args, _k)}   ({src})")

_padded = ((args.embed_size if args.agent_type == "embedded"
            else args.map_h * args.map_w) + 27 + args.padding)
if _padded % args.attention_heads != 0:
    _divs = [h for h in range(1, _padded + 1) if _padded % h == 0]
    raise SystemExit(
        f"--attention-heads {args.attention_heads} does not divide the "
        f"transformer width {_padded}. Valid head counts: {_divs}")

if args.run_name is None:
    args.run_name = f"{_ckpt_stem_early}-{args.opponent}"
if os.path.abspath(args.out_checkpoint) == os.path.abspath(args.checkpoint):
    raise ValueError(
        "--out-checkpoint would overwrite the source checkpoint. "
        "Choose a different path.")
if args.resume_path and os.path.abspath(args.resume_path) != \
        os.path.abspath(args.out_checkpoint):
    print(f"Note: resuming from {args.resume_path} but writing to "
          f"{args.out_checkpoint}")

# Both training and evaluation use the one opponent
args.eval_opponent = args.opponent

if args.reward_weights is not None:
    REWARD_WEIGHT = np.array(args.reward_weights)
elif args.sparse_rewards:
    REWARD_WEIGHT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
else:
    REWARD_WEIGHT = np.array([20.0, 1.0, 1.0, 0.2, 1.0, 4.0])   # as trained

MAP_PATH = (f"maps/{args.map_h}x{args.map_w}/"
            f"basesWorkers{args.map_h}x{args.map_w}.xml")


# ---------------------------------------------------------------------------
# --export-only : write a checkpoint from the latest saved state, then exit
# ---------------------------------------------------------------------------

def export_checkpoint(args):
    """
    Materialise --out-checkpoint from whatever state was last saved, without
    training anything. No environments and no wandb run are created.
    """
    src  = args.resume_path or args.checkpoint
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    is_finetune = isinstance(ckpt, dict) and "model" in ckpt

    model_state = ckpt["model"] if is_finetune else ckpt
    step        = int(ckpt.get("step", 0)) if is_finetune else 0

    out_args = dict(ckpt.get("args", {})) if is_finetune else {}
    for k in ARCH_KEYS:
        out_args[k] = getattr(args, k)
    out_args["opponent"] = args.opponent

    payload = {
        "model": model_state,
        "step":  step,
        "args":  out_args,
        "source_checkpoint": (ckpt.get("source_checkpoint", args.checkpoint)
                              if is_finetune else args.checkpoint),
        "source_step":       int(ckpt.get("source_step", 0)) if is_finetune
                             else 0,
        "finetune_opponent": (ckpt.get("finetune_opponent", args.opponent)
                              if is_finetune else args.opponent),
        "exported_from":     src,
    }
    if is_finetune and "optim" in ckpt:
        payload["optim"] = ckpt["optim"]
    if is_finetune and ckpt.get("wandb_run_id"):
        payload["wandb_run_id"] = ckpt["wandb_run_id"]

    os.makedirs(os.path.dirname(args.out_checkpoint) or ".", exist_ok=True)
    torch.save(payload, args.out_checkpoint)

    print("\n" + "=" * 62)
    print("Exported checkpoint (no training performed)")
    print(f"  from   : {src}")
    print(f"  step   : {step:,}")
    print(f"  output : {args.out_checkpoint}")
    print(f"  optim  : {'included' if 'optim' in payload else 'not present'}")
    print(f"  wandb  : {payload.get('wandb_run_id', 'not recorded')}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_envs(args, n, max_steps):
    """n bot envs, every one playing the single specified opponent."""
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from gym_microrts import microrts_ai

    ai_cls = getattr(microrts_ai, args.opponent, None)
    if ai_cls is None:
        available = sorted(x for x in dir(microrts_ai)
                           if not x.startswith("_")
                           and callable(getattr(microrts_ai, x)))
        raise ValueError(
            f"Unknown opponent {args.opponent!r}.\n"
            f"Available: {', '.join(available)}")

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=n,
        max_steps=max_steps,
        render_theme=2,
        partial_obs=False,
        ai2s=[ai_cls] * n,
        map_paths=[MAP_PATH] * n,
        reward_weight=REWARD_WEIGHT,
    )
    return envs


def build_agent(args, envs, n_envs, device):
    """Construct the right variant and its observation pipeline."""
    mapsize = args.map_h * args.map_w
    if args.agent_type == "base":
        agent = WeightedAgent(
            mapsize, envs, device, args.transformer_layers,
            args.feed_forward_neurons, args.attention_heads,
            args.padding).to(device)
        feature_map = reshape_observation_extended
        obs_dtype, obs_feat = torch.float32, mapsize + 27
    else:
        agent = MixedEmbeddedAgent(
            mapsize, envs, device, args.transformer_layers,
            args.feed_forward_neurons, args.attention_heads,
            args.padding, args.embed_size).to(device)
        feature_map = reshape_observation_mixed_embedded
        obs_dtype, obs_feat = torch.int16, 27 + 1
    return agent, feature_map, obs_dtype, obs_feat


# ---------------------------------------------------------------------------
# Episode tracker (identical role to finetune_opponent.py's)
# ---------------------------------------------------------------------------

class EpisodeTracker:
    def __init__(self, B, opponent, winrate_window=5):
        from collections import deque
        self.B         = B
        self.opponent  = opponent
        self.ep_reward = np.zeros(B)
        self.ep_components = np.zeros((B, len(REWARD_FUNCTIONS)))
        self.ep_length = np.zeros(B, dtype=int)
        self.finished  = []
        # Rolling outcome window per env: 1.0 for a win, 0.0 otherwise
        # (draws count against, consistent with the Evaluator)
        self.recent = [deque(maxlen=winrate_window) for _ in range(B)]

    def update(self, rewards, reward_components, dones):
        self.ep_reward     += rewards
        self.ep_components += reward_components
        self.ep_length     += 1
        for i in range(self.B):
            if dones[i]:
                entry = {
                    "episode/total_reward": float(self.ep_reward[i]),
                    "episode/length":       int(self.ep_length[i]),
                    "episode/opponent":     self.opponent,
                }
                for j, fn in enumerate(REWARD_FUNCTIONS):
                    entry[f"charts/episode_reward/{fn}"] = \
                        float(self.ep_components[i, j])
                entry["charts/episode_reward/WinLossRewardFunction/"
                      f"{self.opponent}"] = float(self.ep_components[i, 0])
                self.recent[i].append(
                    1.0 if self.ep_components[i, 0] > 0 else 0.0)
                self.finished.append(entry)
                self.ep_reward[i]     = 0.0
                self.ep_components[i] = 0.0
                self.ep_length[i]     = 0

    def flush(self):
        out, self.finished = self.finished, []
        return out

    def rolling_winrate(self):
        """
        Average of the per-env win rates over each env's last N games.

        Returns (winrate or None, envs_reporting, games_in_window). Envs
        that have not finished a game yet are excluded; with equal, full
        windows this equals the pooled win rate over the last N*B games.
        """
        per_env = [sum(d) / len(d) for d in self.recent if len(d)]
        if not per_env:
            return None, 0, 0
        games = sum(len(d) for d in self.recent)
        return float(np.mean(per_env)), len(per_env), games


# ---------------------------------------------------------------------------
# In-training evaluator — periodic evaluation with early stopping
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates the current policy against the fine-tuning opponent.

    The eval env and eval model are created lazily on first use and reused
    (the JVM cannot be restarted, but extra envs inside a live JVM are fine).
    Weights are copied from the training model before each evaluation.
    """

    def __init__(self, args, device):
        self.args   = args
        self.device = device
        self.envs   = None
        self.model  = None
        self.feature_map = None

    def _lazy_init(self):
        n = self.args.eval_workers
        # Must match the game conditions used in the final evaluation and in
        # training (max_steps=2000): a game undecided at the cap is a draw
        # and counts against the winrate.
        eval_max_steps = self.args.eval_max_steps
        self.envs = make_envs(self.args, n, eval_max_steps)

        if self.args.eval_record:
            # Wrap ONCE and keep the wrapper alive for the whole run: a
            # per-eval wrapper is garbage-collected after evaluate() returns,
            # sb3's VecVideoRecorder.__del__ closes the wrapped env, and
            # gym-microrts env.close() shuts down the ENTIRE JVM — killing
            # the training envs too. reset() starts a new recording, so each
            # evaluation still gets its own video.
            from stable_baselines3.common.vec_env import VecVideoRecorder
            self.envs = VecVideoRecorder(
                self.envs, "videos/eval",
                record_video_trigger=lambda x: x == 0,
                video_length=self.args.eval_video_length,
                name_prefix="eval",
            )

        self.model, self.feature_map, _, _ = build_agent(
            self.args, self.envs, n, self.device)

    def evaluate(self, train_model, step=0):
        """
        Plays exactly eval_games against the eval opponent with current
        weights, distributed across eval_workers parallel envs as fixed
        per-env quotas (so the worker count sets parallelism, never the
        sample size, and slow games and full-length draws are always
        included). Returns winrate in [0, 1]. Games are capped at
        eval_max_steps (default 2000, matching the evaluation protocol);
        games undecided at the cap are draws, and draws and losses both
        count against the winrate.
        """
        if self.envs is None:
            self._lazy_init()

        self.model.load_state_dict(train_model.state_dict())
        self.model.eval()

        envs  = self.envs
        n     = self.args.eval_workers
        # Play exactly eval_games, pre-assigned as fixed per-env quotas
        # (as even as possible). Each env contributes its first quota[i]
        # completed games, however long they take, and the loop runs until
        # every quota is filled. Racing all envs for a shared quota (the
        # previous behaviour) is length-biased: fast games (mostly wins)
        # fill the quota before slow games and full-cap draws can finish.
        # Fixed quotas keep the sample size at eval_games regardless of
        # worker count, with workers providing parallelism only.
        total = self.args.eval_games
        remaining = [total // n + (1 if i < total % n else 0)
                     for i in range(n)]
        wins  = games = 0
        feats = self.feature_map(
            torch.Tensor(envs.reset()).to(self.device), self.device)
        (nobs, nent_mask, nent_count, nunit_pos, nunit_mask, _, _) = feats

        with torch.no_grad():
            while games < total:
                action, _, _, _ = self.model.get_action(
                    nobs, nent_mask, nent_count, nunit_pos, nunit_mask,
                    envs=envs)
                actions_np = np.array(action.cpu().numpy(), dtype=np.int32)
                raw_obs, _, done, infos = envs.step(actions_np)
                (nobs, nent_mask, nent_count, nunit_pos, nunit_mask,
                 _, _) = self.feature_map(
                    torch.Tensor(raw_obs).to(self.device), self.device)

                for i in range(len(done)):
                    if done[i] and remaining[i] > 0:
                        remaining[i] -= 1
                        games += 1
                        raw = infos[i].get("raw_rewards", np.zeros(6))
                        if float(raw[0]) > 0:
                            wins += 1

        if self.args.eval_record:
            envs.close_video_recorder()
            import glob
            mp4s = sorted(glob.glob(os.path.join("videos", "eval", "*.mp4")),
                          key=os.path.getmtime)
            if mp4s:
                wandb.log({"global_step": step,
                           "eval/video": wandb.Video(mp4s[-1], format="mp4")})

        return wins / max(games, 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment(args):
    os.makedirs(args.save_dir, exist_ok=True)

    cuda_ok = torch.cuda.is_available() and args.cuda
    device  = torch.device("cuda" if cuda_ok else "cpu")
    args.device = device

    if not args.seed:                    # as in original training: 0 -> time
        args.seed = int(time.time())
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # --- Load the file we are continuing from, if resuming ---
    resume_ckpt = resume_run_id = None
    if args.resume_path:
        resume_ckpt = torch.load(args.resume_path, map_location=device,
                                 weights_only=False)
        resume_run_id = resume_ckpt.get("wandb_run_id")

    # --- wandb: fresh fine-tune = NEW run; resume rejoins the stored one ---
    if resume_run_id:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         id=resume_run_id, resume="allow")
        print(f"Rejoined wandb run: {run.name} ({run.id})")
    else:
        if args.resume_path:
            print("  note: the resume checkpoint has no wandb run id — "
                  "starting a new run")
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name,
            config={**vars(args),
                    "source_checkpoint": args.checkpoint,
                    "finetune_opponent": args.opponent},
            save_code=True,
        )
        print(f"New wandb run: {run.name} ({run.id})")

    if args.run_id_file:
        with open(args.run_id_file, "w") as f:
            f.write(run.id)
        print(f"Run id {run.id} written to {args.run_id_file}")

    if args.config_csv_enabled:
        write_run_config(args, run=run, path=args.config_csv)

    # --- Environment ---
    train_max_steps = 2000        # as in original training (all maps)
    envs = make_envs(args, args.num_workers, train_max_steps)
    print(f"Envs: {args.num_workers} x {args.opponent} on {MAP_PATH}")

    num_envs       = args.num_workers
    batch_size     = int(num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.n_minibatch)
    mapsize        = args.map_h * args.map_w

    # --- Model & optimizer ---
    agent, feature_map, obs_dtype, obs_feat = build_agent(
        args, envs, num_envs, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    wandb.run.summary["model/n_params"] = n_params
    print(f"Model: {args.agent_type} | parameters: {n_params:,}")

    # --- Load weights: the resume file if continuing, else the source ---
    if resume_ckpt is not None:
        src_state = resume_ckpt["model"]
        print(f"Resuming from {args.resume_path}")
    else:
        _src = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        src_state = (_src["model"] if isinstance(_src, dict)
                     and "model" in _src else _src)
    agent.load_state_dict(src_state)
    # Explicitly train mode. (train_embedded_agent.py's crash-resume path
    # calls agent.eval() after loading and never switches back, silently
    # disabling the transformer's dropout for resumed training — a quirk we
    # deliberately do NOT replicate; fresh runs there train with dropout.)
    agent.train()

    if args.reset_optimizer:
        print("  optimizer: fresh (--reset-optimizer)")
    else:
        _optim_src = resume_ckpt if resume_ckpt is not None else (
            _src if isinstance(_src, dict) else {})
        try:
            optimizer.load_state_dict(_optim_src["optim"])
            print("  optimizer: Adam moments restored")
        except (KeyError, TypeError, ValueError) as e:
            print(f"  optimizer: none to restore ({type(e).__name__}) — "
                  f"starting fresh")

    if resume_ckpt is not None:
        step        = int(resume_ckpt.get("step", 0))
        source_step = int(resume_ckpt.get("source_step", 0))
        print(f"Fine-tuning vs {args.opponent}: resuming at step {step:,} "
              f"of {args.max_steps:,}")
        if step >= args.max_steps:
            print("  the step budget is already met — raise --max-steps to "
                  "continue further")
    else:
        step        = 0
        source_step = int(_src.get("step", 0)) if isinstance(_src, dict) \
            else 0
        print(f"Loaded {args.checkpoint}"
              + (f" (trained to step {source_step:,})" if source_step else ""))
        print(f"Fine-tuning vs {args.opponent} for {args.max_steps:,} steps, "
              f"counting from 0")

    ckpt_stem = os.path.splitext(os.path.basename(args.out_checkpoint))[0]

    # --- Rollout storage (as in the original training script) ---
    S, N = args.num_steps, num_envs
    obs            = torch.zeros((S, N, mapsize, obs_feat),
                                 dtype=obs_dtype).to(device)
    entity_masks   = torch.ones((S, N, mapsize),  dtype=torch.bool).to(device)
    entity_counts  = torch.zeros((S, N),          dtype=torch.int64).to(device)
    unit_positions = torch.zeros((S, N, mapsize)).to(device)
    unit_masks     = torch.ones((S, N, mapsize),  dtype=torch.bool).to(device)
    enemy_unit_masks   = torch.ones((S, N, mapsize),
                                    dtype=torch.bool).to(device)
    neutral_unit_masks = torch.ones((S, N, mapsize),
                                    dtype=torch.bool).to(device)
    action_space_shape   = (mapsize, 7)
    invalid_action_shape = (mapsize,
                            int(envs.action_plane_space.nvec.sum()) + 1)
    actions  = torch.zeros((S, N) + action_space_shape,
                           dtype=torch.int16).to(device)
    logprobs = torch.zeros((S, N)).to(device)
    rewards  = torch.zeros((S, N)).to(device)
    dones    = torch.zeros((S, N)).to(device)
    values   = torch.zeros((S, N)).to(device)
    invalid_action_masks = torch.zeros((S, N) + invalid_action_shape,
                                       dtype=torch.bool).to(device)

    (next_obs, next_entity_mask, next_entity_count, next_unit_position,
     next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask) = \
        feature_map(torch.Tensor(envs.reset()).to(device), device)
    next_done = torch.zeros(num_envs).to(device)

    ep_tracker = EpisodeTracker(B=num_envs, opponent=args.opponent,
                                winrate_window=args.winrate_window)
    evaluator  = Evaluator(args, device)
    if args.eval_every > 0:
        next_eval_at = ((step // args.eval_every) + 1) * args.eval_every
    else:
        next_eval_at = float("inf")

    save_steps = [s for s in range(0, int(args.max_steps),
                                   max(int(args.max_steps) // 10, 1))
                  if s > step]

    print(f"Fine-tuning | {num_envs} envs vs {args.opponent} | "
          f"{args.max_steps:,} steps | map {args.map_h}x{args.map_w} | "
          f"batch {batch_size} / minibatch {minibatch_size} | "
          f"wandb: {run.name}")

    # Update-indexed loop, exactly as in original training: precisely
    # max_steps // batch_size full updates (floor), with the LR anneal
    # driven by the update index so a resumed run continues the schedule
    # from where it stopped.
    num_updates = args.max_steps // batch_size
    if num_updates == 0:
        raise SystemExit(
            f"--max-steps {args.max_steps:,} is smaller than one batch "
            f"({batch_size:,} = num_workers x num_steps) — nothing to run.")
    starting_update = step // batch_size + 1

    train_start    = time.time()
    steps_at_start = step

    for update in range(starting_update, num_updates + 1):

        # --- LR schedule: linear anneal by default (original training's
        # exact formula); cosine available as a fine-tuning alternative ---
        lr_now = args.lr
        if args.cosine:
            lr_min = args.lr / 4
            lr_now = lr_min + 0.5 * (args.lr - lr_min) * (
                1 + math.cos(math.pi * (update - 1.0) / num_updates))
        elif args.anneal_lr:
            frac   = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.lr
        optimizer.param_groups[0]["lr"] = lr_now

        # =========================== Rollout ===========================
        for t in range(args.num_steps):
            obs[t]                = next_obs
            entity_masks[t]       = next_entity_mask
            entity_counts[t]      = next_entity_count
            unit_positions[t]     = next_unit_position
            unit_masks[t]         = next_unit_mask
            enemy_unit_masks[t]   = next_enemy_unit_mask
            neutral_unit_masks[t] = next_neutral_unit_mask
            dones[t]              = next_done

            with torch.no_grad():
                values[t] = agent.get_value(
                    obs[t], entity_masks[t], entity_counts[t],
                    unit_masks[t], enemy_unit_masks[t],
                    neutral_unit_masks[t]).flatten()
                action, logproba, _, invalid_action_masks[t] = \
                    agent.get_action(obs[t], entity_masks[t],
                                     entity_counts[t], unit_positions[t],
                                     unit_masks[t], envs=envs)
            actions[t]  = action
            logprobs[t] = logproba

            # gym-microrts 0.6.0: step() takes the (num_envs, h*w, 7) array
            # directly; the JArray filtering of the old API is internal now.
            actions_np = np.array(action.cpu().numpy(), dtype=np.int32)
            raw_obs, rs, ds, infos = envs.step(actions_np)

            (next_obs, next_entity_mask, next_entity_count,
             next_unit_position, next_unit_mask, next_enemy_unit_mask,
             next_neutral_unit_mask) = feature_map(
                torch.Tensor(raw_obs).to(device), device)
            rewards[t] = torch.Tensor(rs).to(device)
            next_done  = torch.Tensor(ds.astype(np.float32)).to(device)

            reward_components = np.array([
                info.get("raw_rewards", np.zeros(len(REWARD_FUNCTIONS)))
                for info in infos])
            ep_tracker.update(np.asarray(rs), reward_components,
                              np.asarray(ds))
            step += num_envs

        completed = ep_tracker.flush()
        for ep_log in completed:
            wandb.log({"global_step": step, **ep_log})

        # ==================== GAE / returns (as trained) ====================
        with torch.no_grad():
            last_value = agent.get_value(
                next_obs, next_entity_mask, next_entity_count,
                next_unit_mask, next_enemy_unit_mask,
                next_neutral_unit_mask).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues      = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues      = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues \
                        * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma \
                        * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return     = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return     = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal \
                        * next_return
                advantages = returns - values

        # =============== Flatten + PPO epochs (as trained) ===============
        b_obs            = obs.reshape((-1,) + obs.shape[2:])
        b_entity_masks   = entity_masks.reshape(-1, mapsize)
        b_entity_counts  = entity_counts.flatten()
        b_unit_positions = unit_positions.reshape(-1, mapsize)
        b_unit_masks     = unit_masks.reshape(-1, mapsize)
        b_enemy_unit_masks   = enemy_unit_masks.reshape(-1, mapsize)
        b_neutral_unit_masks = neutral_unit_masks.reshape(-1, mapsize)
        b_logprobs   = logprobs.reshape(-1)
        b_actions    = actions.reshape((-1,) + action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape(
            (-1,) + invalid_action_shape)

        inds = np.arange(batch_size)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start:start + minibatch_size]
                mb_advantages = b_advantages[mb]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) \
                        / (mb_advantages.std() + 1e-8)

                _, newlogproba, entropy, _ = agent.get_action(
                    b_obs[mb], b_entity_masks[mb], b_entity_counts[mb],
                    b_unit_positions[mb], b_unit_masks[mb],
                    b_actions[mb], b_invalid_action_masks[mb], envs)

                ratio     = (newlogproba - b_logprobs[mb]).exp()
                approx_kl = (b_logprobs[mb] - newlogproba).mean()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss      = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                new_values = agent.get_value(
                    b_obs[mb], b_entity_masks[mb], b_entity_counts[mb],
                    b_unit_masks[mb], b_enemy_unit_masks[mb],
                    b_neutral_unit_masks[mb]).view(-1)
                if args.clip_vloss:
                    v_unclipped = (new_values - b_returns[mb]) ** 2
                    v_clipped   = b_values[mb] + torch.clamp(
                        new_values - b_values[mb],
                        -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(
                        v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss \
                    + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         args.max_grad_norm)
                optimizer.step()

        # =========================== Logging ===========================
        wr, wr_envs, wr_games = ep_tracker.rolling_winrate()
        wandb.log({
            "global_step":           step,
            **({"charts/rolling_winrate": wr,
                "charts/rolling_winrate_games": wr_games} if wr is not None
               else {}),
            "charts/learning_rate":  lr_now,
            "losses/value_loss":     v_loss.item(),
            "losses/policy_loss":    pg_loss.item(),
            "losses/entropy":        entropy_loss.item(),
            "losses/approx_kl":      approx_kl.item(),
        })

        if completed:
            mean_ep  = np.mean([e["episode/total_reward"]
                                for e in completed])
            mean_win = np.mean([
                e["charts/episode_reward/WinLossRewardFunction"]
                for e in completed])
            ep_str = f"ep_ret={mean_ep:.2f}  win={mean_win:.2f}"
        else:
            ep_str = "ep_ret=n/a  win=n/a"
        W = args.winrate_window
        if wr is not None:
            ep_str += (f"  wr{W}={100 * wr:.0f}%"
                       + (f" ({wr_envs}/{num_envs} envs)"
                          if wr_envs < num_envs else ""))
        else:
            ep_str += f"  wr{W}=n/a"

        elapsed = time.time() - train_start
        sps     = (step - steps_at_start) / max(elapsed, 1e-6)
        eta_sec = (args.max_steps - step) / max(sps, 1)
        eta_h, rem   = divmod(int(eta_sec), 3600)
        eta_m, eta_s = divmod(rem, 60)
        eta_str = (f"{eta_h}h {eta_m:02d}m" if eta_h
                   else f"{eta_m}m {eta_s:02d}s")
        wandb.log({"global_step": step, "charts/steps_per_sec": sps})

        print(f"step={step:>10,} ({100 * step / args.max_steps:4.1f}%) | "
              f"{ep_str} | pg={pg_loss.item():.4f} v={v_loss.item():.4f} "
              f"kl={approx_kl.item():.4f} | {sps:,.0f} sps | ETA {eta_str}")

        # ========================= Checkpoint =========================
        # `args` carries the resolved architecture, so the fine-tuned file
        # is self-describing: evaluate_agent.py and this script both load it
        # without extra flags.
        ckpt_data = {
            "model": agent.state_dict(),
            "optim": optimizer.state_dict(),
            "step":  step,
            "args":  vars(args),
            "source_checkpoint": args.checkpoint,
            "source_step":       source_step,
            "finetune_opponent": args.opponent,
            "wandb_run_id":      run.id,
        }
        torch.save(ckpt_data, args.out_checkpoint)   # rolling save

        if save_steps and step > save_steps[0]:
            numbered = os.path.join(args.save_dir,
                                    f"{ckpt_stem}_step={step}.pt")
            torch.save(ckpt_data, numbered)
            wandb.save(numbered)
            print(f"Checkpoint saved → {numbered}")
            save_steps.pop(0)

        # ============= Periodic evaluation & early stopping =============
        if args.eval_every > 0 and step >= next_eval_at:
            next_eval_at += args.eval_every
            print(f"[eval] step={step:,} | playing {args.eval_games} games "
                  f"vs {args.eval_opponent}...")
            winrate = evaluator.evaluate(agent, step)
            wandb.log({"global_step": step, "eval/winrate": winrate})
            print(f"[eval] step={step:,} | winrate={100 * winrate:.1f}% "
                  f"(target {100 * args.eval_winrate:.0f}%)")
            agent.train()
            if winrate >= args.eval_winrate:
                torch.save(ckpt_data, args.out_checkpoint)
                wandb.save(args.out_checkpoint)
                print(f"Early stopping: winrate {100 * winrate:.1f}% >= "
                      f"{100 * args.eval_winrate:.0f}%")
                break

    # --- Final checkpoint ---
    final = {
        "model": agent.state_dict(),
        "optim": optimizer.state_dict(),
        "step":  step,
        "args":  vars(args),
        "source_checkpoint": args.checkpoint,
        "source_step":       source_step,
        "finetune_opponent": args.opponent,
        "wandb_run_id":      run.id,
    }
    torch.save(final, args.out_checkpoint)
    wandb.save(args.out_checkpoint)

    if hasattr(envs, "close"):
        envs.close()      # end of process: the JVM may shut down now
    wandb.finish()
    print("\n" + "=" * 62)
    print(f"Fine-tuning complete: {step:,} steps vs {args.opponent}")
    print(f"  source : {args.checkpoint}"
          + (f" (step {source_step:,})" if source_step else ""))
    print(f"  output : {args.out_checkpoint}")
    print(f"  wandb  : {run.name}")
    print("=" * 62)


if __name__ == "__main__":
    if args.export_only:
        export_checkpoint(args)      # no envs, no JVM, no wandb run
    else:
        experiment(args)
