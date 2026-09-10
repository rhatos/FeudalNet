"""
finetune_opponent.py

Continue training an existing checkpoint against ONE specified opponent.

Everything that defines the network — hidden dim, transformer sizes, time
horizon, map, enemy encoder — is restored from the checkpoint, so the loaded
weights always fit the model that is built. Only the knobs that are genuinely
choices at fine-tuning time (learning rate, entropy, opponent, budget) come
from the command line.

Differences from main-16x16.py:
  * --checkpoint is required; all architecture args come from it
  * a SINGLE --opponent fills every training env, and is also the evaluation
    opponent
  * a NEW wandb run is created with its own step counter starting at 0, named
    "<checkpoint filename>-<opponent>"
  * the fine-tuned weights are written to a new checkpoint file; the source
    checkpoint is never overwritten

Usage:
    python finetune_opponent.py \
        --checkpoint models/tqlgxayp_latest.pt \
        --opponent izanagi

    # Shorter run, gentler settings, custom output
    python finetune_opponent.py \
        --checkpoint models/tqlgxayp_step=300000000.pt \
        --opponent mayari --max-steps 20000000 \
        --lr 3e-5 --entropy-coef 0.02 \
        --out-checkpoint models/vs_mayari.pt

    # Resume after a crash: the SAME command plus --resume. Picks up the
    # fine-tuning checkpoint, its step count, and the same wandb run.
    # Safe in a slurm chain: if nothing was saved yet it starts from scratch.
    python finetune_opponent.py \\
        --checkpoint models/tqlgxayp_latest.pt \\
        --opponent izanagi --resume
"""

import os
import argparse
import numpy as np
import torch
import wandb
import math
import time

from feudalnet import FeudalNetwork, feudal_loss, ppg_aux_loss, ACTION_SPACE
from storage import Storage
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
    description='Fine-tune a FeudalNet checkpoint against a single opponent')

# --- Required: what to fine-tune, and against whom ---
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Checkpoint to fine-tune. All architecture '
                         'hyperparameters are restored from it.')
parser.add_argument('--opponent',   type=str, required=True,
                    help='The single opponent used for BOTH training and '
                         'evaluation (e.g. izanagi, mayari, coacAI)')

# --- Fine-tuning schedule ---
parser.add_argument('--max-steps',    type=int,   default=int(1e8),
                    help='Env steps to fine-tune for (default 100M). The new '
                         'run counts from 0 regardless of the source step.')
parser.add_argument('--lr',           type=float, default=5e-5,
                    help='Flat fine-tuning LR — low, to avoid destroying the '
                         'loaded policy')
parser.add_argument('--entropy-coef', type=float, default=0.02,
                    help='Slightly above training default, to re-open '
                         'exploration against the new opponent')
parser.add_argument('--cosine',       type=str2bool, default=False,
                    help='Anneal the LR over the fine-tuning window '
                         '(default off: flat LR)')
parser.add_argument('--reset-optimizer', type=str2bool, default=False,
                    help='Start Adam fresh instead of loading the moments '
                         'from the checkpoint')

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
                         'saved state and exit. Useful when a run died before '
                         'its final save, or to turn a numbered snapshot into '
                         'the named output file. Starts no JVM and no wandb '
                         'run, so it is instant.')

# --- Rollout / optimisation (safe to change; not baked into the weights) ---
parser.add_argument('--num-workers',  type=int,   default=24)
parser.add_argument('--num-steps',    type=int,   default=50)
parser.add_argument('--cuda',         type=str2bool, default=True)
parser.add_argument('--grad-clip',    type=float, default=0.5)
parser.add_argument('--seed',         type=int,   default=0)
parser.add_argument('--gamma-w',      type=float, default=0.99)
parser.add_argument('--gamma-m',      type=float, default=0.99)
parser.add_argument('--alpha',        type=float, default=0.5)
parser.add_argument('--gae-lambda',   type=float, default=0.95)
parser.add_argument('--value-coef',   type=float, default=1.0)
parser.add_argument('--n-pi',         type=int,   default=8)
parser.add_argument('--e-aux',        type=int,   default=2)

# --- Evaluation (same single opponent) ---
parser.add_argument('--eval-every',    type=int,   default=int(1e7),
                    help='Evaluate every N env steps (0 disables)')
parser.add_argument('--eval-games',    type=int,   default=100)
parser.add_argument('--eval-winrate',  type=float, default=1.0,
                    help='Stop early once this win rate is reached')
parser.add_argument('--eval-workers',  type=int,   default=8)
parser.add_argument('--eval-record',   type=str2bool, default=True)
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
# Restore every architecture hyperparameter from the checkpoint
# ---------------------------------------------------------------------------
# The weights only fit a model built with the SAME shapes, so these are not
# command-line choices — reading them from the checkpoint makes a shape
# mismatch impossible.

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

# Where to read architecture from. When resuming or exporting, the
# fine-tuning checkpoint is authoritative: it was written by this script and
# already carries both the architecture and the progress made so far.
_arch_src = args.checkpoint
args.resume_path = None

# The output name is needed before the resume search, so derive it here
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


if args.resume_from:
    if not os.path.exists(args.resume_from):
        raise FileNotFoundError(
            f"--resume-from not found: {args.resume_from}")
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

if args.resume_path:
    _arch_src = args.resume_path

_ckpt  = torch.load(_arch_src, map_location="cpu", weights_only=False)
_saved = dict(_ckpt.get("args", {}))
if "hidden_dim" not in _saved and "hidden_dim_manager" in _saved:
    _saved["hidden_dim"] = _saved["hidden_dim_manager"]      # older runs

ARCH_KEYS = [
    "hidden_dim", "time_horizon", "eps", "dilation",
    "map_h", "map_w", "in_channels",
    "enemy_dim", "enemy_layers",
    "worker_heads", "worker_layers", "T_W",
    "manager_heads", "manager_layers", "T_M",
]
ARCH_DEFAULTS = {
    "hidden_dim": 128, "time_horizon": 25, "eps": 1e-4, "dilation": 10,
    "map_h": 16, "map_w": 16, "in_channels": 27,
    "enemy_dim": 64, "enemy_layers": 2,
    "worker_heads": 4, "worker_layers": 4, "T_W": 100,
    "manager_heads": 4, "manager_layers": 6, "T_M": 160,
}
print(f"Architecture restored from {_arch_src}:")
for _k in ARCH_KEYS:
    _v = _saved.get(_k, ARCH_DEFAULTS[_k])
    setattr(args, _k, _v)
    print(f"    {_k:<16} = {_v}"
          + ("" if _k in _saved else "   (not in checkpoint - default)"))

_ckpt_name = _ckpt_stem_early
if args.run_name is None:
    args.run_name = f"{_ckpt_name}-{args.opponent}"
if os.path.abspath(args.out_checkpoint) == os.path.abspath(args.checkpoint):
    raise ValueError(
        "--out-checkpoint would overwrite the source checkpoint. "
        "Choose a different path.")

# Resuming writes back into the fine-tuning checkpoint by design
if args.resume_path and os.path.abspath(args.resume_path) != \
        os.path.abspath(args.out_checkpoint):
    print(f"Note: resuming from {args.resume_path} but writing to "
          f"{args.out_checkpoint}")

# Both training and evaluation use the one opponent
args.opponents     = [args.opponent]
args.eval_opponent = args.opponent



# ---------------------------------------------------------------------------
# --export-only : write a checkpoint from the latest saved state, then exit
# ---------------------------------------------------------------------------

def export_checkpoint(args):
    """
    Materialise --out-checkpoint from whatever state was last saved, without
    training anything.

    Reads the newest available state (the fine-tuning checkpoint if one exists,
    otherwise the source), normalises it to the same self-describing layout
    this script writes during training, and saves it. No environments and no
    wandb run are created, so this is instant and works on a login node.
    """
    src = args.resume_path or args.checkpoint
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise KeyError(
            f"{src} has no 'model' weights (keys: {list(ckpt)[:8]}) — "
            f"nothing to export.")

    step = int(ckpt.get("step", 0))
    is_finetune = args.resume_path is not None

    # Preserve the checkpoint's own args (which carry the architecture) and
    # overlay the resolved values, so the export stays loadable with no flags.
    out_args = dict(ckpt.get("args", {}))
    for k in ARCH_KEYS:
        out_args[k] = getattr(args, k)
    out_args["opponent"] = args.opponent

    payload = {
        "model": ckpt["model"],
        "step":  step,
        "args":  out_args,
        "source_checkpoint": ckpt.get("source_checkpoint", args.checkpoint),
        "source_step":       int(ckpt.get("source_step", 0 if is_finetune
                                          else step)),
        "finetune_opponent": ckpt.get("finetune_opponent", args.opponent),
        "exported_from":     src,
    }
    # Carry the optimizer and wandb id through when present, so the exported
    # file can still be resumed from later
    if "optim" in ckpt:
        payload["optim"] = ckpt["optim"]
    if ckpt.get("wandb_run_id"):
        payload["wandb_run_id"] = ckpt["wandb_run_id"]

    os.makedirs(os.path.dirname(args.out_checkpoint) or ".", exist_ok=True)
    torch.save(payload, args.out_checkpoint)

    print("\n" + "=" * 62)
    print("Exported checkpoint (no training performed)")
    print(f"  from   : {src}")
    print(f"  step   : {step:,}"
          + ("  (fine-tuning steps vs "
             f"{payload['finetune_opponent']})" if is_finetune
             else "  (source training steps)"))
    print(f"  output : {args.out_checkpoint}")
    print(f"  optim  : {'included' if 'optim' in payload else 'not present'}")
    print(f"  wandb  : {payload.get('wandb_run_id', 'not recorded')}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_envs(args):
    """Every environment plays the one specified opponent."""
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from gym_microrts import microrts_ai

    # Resolved dynamically so ANY bot in the installed gym-microrts can be
    # used, not just a hardcoded handful.
    ai_cls = getattr(microrts_ai, args.opponent, None)
    if ai_cls is None:
        available = sorted(n for n in dir(microrts_ai)
                           if not n.startswith("_") and callable(
                               getattr(microrts_ai, n)))
        raise ValueError(
            f"Unknown opponent {args.opponent!r}.\n"
            f"Available: {', '.join(available)}")

    env_opps = [args.opponent] * args.num_workers
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=args.num_workers,
        max_steps=4000,
        render_theme=2,
        partial_obs=False,
        ai2s=[ai_cls] * args.num_workers,
        map_paths=[f"maps/{args.map_h}x{args.map_w}/"
                   f"basesWorkers{args.map_h}x{args.map_w}.xml"],
        reward_weight=np.array([20.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    return envs, env_opps


def obs_to_tensor(obs, device):
    return torch.FloatTensor(obs).to(device)


# ---------------------------------------------------------------------------
# Episode tracker
# ---------------------------------------------------------------------------

class EpisodeTracker:
    def __init__(self, B, env_opponents):
        self.B             = B
        self.env_opponents = env_opponents
        self.ep_reward     = np.zeros(B)
        self.ep_components = np.zeros((B, len(REWARD_FUNCTIONS)))
        self.ep_length     = np.zeros(B, dtype=int)
        self.finished      = []

    def update(self, rewards, reward_components, dones):
        self.ep_reward     += rewards
        self.ep_components += reward_components
        self.ep_length     += 1

        for i in range(self.B):
            if dones[i]:
                opp   = self.env_opponents[i]
                entry = {
                    "episode/total_reward": float(self.ep_reward[i]),
                    "episode/length":       int(self.ep_length[i]),
                    "episode/opponent":     opp,
                }
                for j, fn in enumerate(REWARD_FUNCTIONS):
                    entry[f"charts/episode_reward/{fn}"] = float(self.ep_components[i, j])
                win_v = float(self.ep_components[i, 0])
                entry[f"charts/episode_reward/WinLossRewardFunction/{opp}"] = win_v

                self.finished.append(entry)
                self.ep_reward[i]     = 0.0
                self.ep_components[i] = 0.0
                self.ep_length[i]     = 0

    def flush(self):
        out, self.finished = self.finished, []
        return out


# ---------------------------------------------------------------------------
# In-training evaluator — periodic evaluation with early stopping
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates the current policy against a fixed opponent during training.

    The eval environment and eval model are created lazily on first use and
    reused for every subsequent evaluation (the JVM cannot be restarted, but
    creating additional environments within the same JVM is fine). The eval
    model is a separate instance with eps=0 (no random goals); weights are
    copied from the training model before each evaluation, so the training
    model's hidden state is never disturbed.
    """

    def __init__(self, args, action_space, device):
        self.args         = args
        self.action_space = action_space
        self.device       = device
        self.envs         = None
        self.model        = None

    def _lazy_init(self):
        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
        from gym_microrts import microrts_ai

        ai_cls = getattr(microrts_ai, self.args.eval_opponent, None)
        if ai_cls is None:
            raise ValueError(f"Unknown eval opponent: {self.args.eval_opponent}")

        n = self.args.eval_workers
        eval_max_steps = 4000 if self.args.map_h >= 16 else 2000
        self.envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=n,
            max_steps=eval_max_steps,
            render_theme=2,
            partial_obs=False,
            ai2s=[ai_cls] * n,
            map_paths=[f"maps/{self.args.map_h}x{self.args.map_w}/"
                       f"basesWorkers{self.args.map_h}x{self.args.map_w}.xml"],
            reward_weight=np.array([20.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )

        if self.args.eval_record:
            # Wrap ONCE and keep the wrapper alive for the whole run.
            # A per-eval wrapper gets garbage-collected after evaluate()
            # returns; sb3's VecVideoRecorder.__del__ closes the wrapped env,
            # and gym-microrts env.close() shuts down the ENTIRE JVM —
            # killing the training envs too (JVMNotRunning).
            # sb3 1.x reset() starts a new recording, so each evaluation
            # (which begins with reset) still gets its own video.
            from stable_baselines3.common.vec_env import VecVideoRecorder
            self.envs = VecVideoRecorder(
                self.envs, "videos/eval",
                record_video_trigger=lambda x: x == 0,
                video_length=self.args.eval_video_length,
                name_prefix="eval",
            )

        self.model = FeudalNetwork(
            num_workers=n,
            h=self.args.map_h, w=self.args.map_w,
            in_channels=self.args.in_channels,
            d=self.args.hidden_dim,
            n_cells=self.args.map_h * self.args.map_w,
            action_space=self.action_space,
            time_horizon=self.args.time_horizon,
            dilation=self.args.dilation,
            eps=0.0,   # no random goals during eval
            device=self.device,
            enemy_dim=self.args.enemy_dim,
            enemy_layers=self.args.enemy_layers,
            worker_heads=self.args.worker_heads,
            worker_layers=self.args.worker_layers,
            T_W=self.args.T_W,
            manager_heads=self.args.manager_heads,
            manager_layers=self.args.manager_layers,
            T_M=self.args.T_M,
        )

    def evaluate(self, train_model, step=0):
        """
        Plays eval_games against the eval opponent with current weights.
        Returns winrate in [0, 1]. Draws and losses both count against it.

        If eval_record is set, the evaluation is recorded via VecVideoRecorder
        (same pattern as ppo_gridnet) and the video is uploaded to wandb.
        """
        if self.envs is None:
            self._lazy_init()

        self.model.load_state_dict(train_model.state_dict())
        self.model.eval()

        envs = self.envs
        wins = games = 0
        obs  = envs.reset()   # reset also starts a new video recording
        goals, states, masks = self.model.init_obj()
        self.model.repackage_hidden()

        with torch.no_grad():
            while games < self.args.eval_games:
                obs_t = torch.FloatTensor(obs).to(self.device)
                action_mask = torch.tensor(
                    envs.get_action_mask(), dtype=torch.bool,
                    device=self.device,
                )

                logits_list, goals, states, _, _ = self.model(
                    obs_t, goals, states, masks[-1], action_mask=action_mask
                )
                actions, _, _ = FeudalNetwork.sample(logits_list)

                actions_np = np.array(actions.cpu().numpy(), dtype=np.int32)
                obs, _, done, infos = envs.step(actions_np)

                mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(self.device)
                masks.pop(0)
                masks.append(mask)

                for i in range(len(done)):
                    if done[i]:
                        games += 1
                        raw = infos[i].get("raw_rewards", np.zeros(6))
                        if float(raw[0]) > 0:
                            wins += 1

        # Flush the recording (close_video_recorder only finalises the video
        # file — it does NOT close the env or the JVM) and upload newest mp4
        if self.args.eval_record:
            envs.close_video_recorder()
            import glob
            mp4s = sorted(glob.glob(os.path.join("videos", "eval", "*.mp4")),
                          key=os.path.getmtime)
            if mp4s:
                wandb.log({
                    "global_step": step,
                    "eval/video":  wandb.Video(mp4s[-1], format="mp4"),
                })

        return wins / max(games, 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment(args):
    os.makedirs(args.save_dir, exist_ok=True)

    cuda_ok = torch.cuda.is_available() and args.cuda
    device  = torch.device("cuda" if cuda_ok else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # --- Load the file we are continuing from, if resuming ---
    resume_ckpt = None
    resume_run_id = None
    if args.resume_path:
        resume_ckpt = torch.load(args.resume_path, map_location=device,
                                 weights_only=False)
        resume_run_id = resume_ckpt.get("wandb_run_id")

    # --- wandb ---
    # A fresh fine-tune gets a NEW run named <checkpoint>-<opponent>, with its
    # own step axis starting at 0. Resuming rejoins that same run so the curve
    # continues rather than restarting, using the id stored in the checkpoint.
    if resume_run_id:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=resume_run_id,
            resume="allow",
        )
        print(f"Rejoined wandb run: {run.name} ({run.id})")
    else:
        if args.resume_path:
            print("  note: the resume checkpoint has no wandb run id "
                  "(written before resume support) — starting a new run")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={**vars(args),
                    "source_checkpoint": args.checkpoint,
                    "finetune_opponent": args.opponent},
            save_code=True,
        )
        print(f"New wandb run: {run.name} ({run.id})")

    # Write run id to file so dependent jobs (e.g. slurm resume) can pick it up
    if args.run_id_file:
        with open(args.run_id_file, "w") as f:
            f.write(run.id)
        print(f"Run id {run.id} written to {args.run_id_file}")

    # --- Record every parameter of this run ---
    # Written before training starts, so an interrupted run still leaves a
    # complete record of how it was configured.
    if args.config_csv_enabled:
        write_run_config(args, run=run, path=args.config_csv)

    # --- Environment ---
    envs, env_opponents = make_envs(args)
    x = envs.reset()
    print(f"Opponents: { {i: env_opponents[i] for i in range(args.num_workers)} }")

    # Derive action space from env
    # action_plane_space.nvec gives per-cell action dims without source unit selector
    action_space = envs.action_plane_space.nvec.tolist()
    print(f"Action space: {action_space}  (sum={sum(action_space)})")
    n_cells = args.map_h * args.map_w

    # --- Model ---
    feudalnet = FeudalNetwork(
        num_workers=args.num_workers,
        h=args.map_h, w=args.map_w, in_channels=args.in_channels,
        d=args.hidden_dim,
        n_cells=n_cells,
        action_space=action_space,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        eps=args.eps,
        device=device,
        enemy_dim=args.enemy_dim,
        enemy_layers=args.enemy_layers,
        worker_heads=args.worker_heads,
        worker_layers=args.worker_layers,
        T_W=args.T_W,
        manager_heads=args.manager_heads,
        manager_layers=args.manager_layers,
        T_M=args.T_M,
    )

    optimizer = torch.optim.Adam(
        feudalnet.parameters(), lr=args.lr, eps=1e-5
    )

    n_params = sum(p.numel() for p in feudalnet.parameters() if p.requires_grad)
    wandb.run.summary["model/n_params"] = n_params
    print(f"Model parameters: {n_params:,}")
    print(f"cosine: {args.cosine}")

    # --- Load weights: the resume file if continuing, else the source ---
    if resume_ckpt is not None:
        ckpt = resume_ckpt
        print(f"Resuming from {args.resume_path}")
    else:
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
    # strict=False: older checkpoints carry now-removed keys (e.g. critic_aux)
    missing, unexpected = feudalnet.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  note: {len(missing)} parameter(s) not in the checkpoint "
              f"(randomly initialised): {missing[:4]}"
              + (" ..." if len(missing) > 4 else ""))
    if unexpected:
        print(f"  note: {len(unexpected)} checkpoint key(s) not in the model "
              f"(ignored): {unexpected[:4]}"
              + (" ..." if len(unexpected) > 4 else ""))

    if args.reset_optimizer:
        print("  optimizer: fresh (--reset-optimizer)")
    else:
        try:
            optimizer.load_state_dict(ckpt["optim"])
            print(f"  optimizer: Adam moments restored from "
                  f"{'the resume checkpoint' if resume_ckpt is not None else 'the checkpoint'}")
        except (KeyError, ValueError) as e:
            print(f"  optimizer: could not restore ({e}) — starting fresh")

    if resume_ckpt is not None:
        # Continue the fine-tuning run's own step axis
        step = int(ckpt.get("step", 0))
        source_step = int(ckpt.get("source_step", 0))
        remaining = max(args.max_steps - step, 0)
        print(f"Fine-tuning vs {args.opponent}: resuming at step {step:,} "
              f"of {args.max_steps:,} ({remaining:,} remaining)")
        if remaining == 0:
            print("  the step budget is already met — raise --max-steps to "
                  "continue further")
    else:
        step = 0                      # the new run has its own step axis
        source_step = int(ckpt.get("step", 0))
        print(f"Loaded {args.checkpoint} (trained to step {source_step:,})")
        print(f"Fine-tuning vs {args.opponent} for {args.max_steps:,} steps, "
              f"counting from 0")

    # Stem for numbered snapshots, e.g. models/<stem>_step=20000000.pt
    ckpt_stem = os.path.splitext(os.path.basename(args.out_checkpoint))[0]

    # --- Init state ---
    goals, states, masks = feudalnet.init_obj()
    aux_rollout_buffer = []
    pi_phase_count     = 0
    ep_tracker = EpisodeTracker(B=args.num_workers, env_opponents=env_opponents)

    # --- In-training evaluator & early stopping ---
    evaluator = Evaluator(args, action_space, device)
    # Align to the next clean multiple past the current step, so a resumed
    # run does not immediately re-run an evaluation it already did
    if args.eval_every > 0:
        next_eval_at = ((step // args.eval_every) + 1) * args.eval_every
    else:
        next_eval_at = float('inf')

    # Drop snapshot milestones already passed before the crash
    save_steps = [s for s in torch.arange(
        0, int(args.max_steps), max(int(args.max_steps) // 10, 1)
    ).numpy() if s > step]

    print(f"Fine-tuning | {args.num_workers} envs vs {args.opponent} | "
          f"{args.max_steps:,} steps | map {args.map_h}x{args.map_w} | "
          f"wandb: {run.name}")

    train_start    = time.time()
    rollout_start  = time.time()
    steps_at_start = step   # may be > 0 if resuming

    while step < args.max_steps:

        # Detach hidden states and goals between rollouts
        feudalnet.repackage_hidden()
        goals = [g.detach() for g in goals]

        # Linear LR annealing — decays from args.lr to 0 over max_steps
        frac   = 1.0 - (step / args.max_steps)
        lr_now = args.lr * frac

        if args.cosine:
            lr_min = args.lr / 4
            lr_now = lr_min + 0.5 * (args.lr - lr_min) * (1 + math.cos(math.pi * step / args.max_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        storage = Storage(
            size=args.num_steps,
            keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                  's_goal_cos', 'm', 'ret_w', 'ret_m', 'adv_m', 'adv_w',
                  'u_w'],
        )

        for _ in range(args.num_steps):
            obs_t = obs_to_tensor(x, device)

            action_mask = torch.tensor(
                envs.get_action_mask(), dtype=torch.bool, device=device,
            )

            logits_list, goals, states, value_m, value_w = feudalnet(
                obs_t, goals, states, masks[-1], action_mask=action_mask
            )

            actions, logp, entropy = FeudalNetwork.sample(logits_list)

            # Step env
            actions_np = np.array(actions.cpu().numpy(), dtype=np.int32)
            x, reward, done, infos = envs.step(actions_np)
            #envs.render()  # disabled: conflicts with the eval env's
            # render client in the shared JVM (deadlocks after first eval),
            # and serves no purpose during training anyway

            reward_components = np.array([
                info.get("raw_rewards", np.zeros(len(REWARD_FUNCTIONS)))
                for info in infos
            ])
            ep_tracker.update(reward, reward_components, done)

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r':           torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i':         feudalnet.intrinsic_reward(states, goals, masks),
                'v_w':         value_w,
                'v_m':         value_m,
                'logp':        logp.mean(dim=-1).unsqueeze(-1),   # mean over cells
                'entropy':     entropy.unsqueeze(-1) if entropy.dim() == 0 else entropy.unsqueeze(-1),
                's_goal_cos':  feudalnet.state_goal_cosine(states, goals, masks),
                'm':           mask,
                'u_w':         feudalnet.worker.last_u.cpu(),
            })

            step += args.num_workers

        # Log completed episodes
        for ep_log in ep_tracker.flush():
            wandb.log({"global_step": step, **ep_log})

        # Bootstrap
        with torch.no_grad():
            obs_last = obs_to_tensor(x, device)
            *_, next_v_m, next_v_w = feudalnet(
                obs_last, goals, states, mask, save=False
            )

        optimizer.zero_grad()
        loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
        optimizer.step()

        # PPG: store per-step entries for auxiliary phase
        pi_phase_count += 1
        ret_w_stacked = torch.stack(storage.ret_w)
        for t in range(args.num_steps):
            aux_rollout_buffer.append({
                'u':     storage.u_w[t],
                'ret_w': ret_w_stacked[t].detach().cpu(),
            })

        # PPG: run auxiliary phase every n_pi policy phases
        if pi_phase_count >= args.n_pi:
            torch.cuda.empty_cache()
            feudalnet.repackage_hidden()
            for _ in range(args.e_aux):
                optimizer.zero_grad()
                _, aux_dict = ppg_aux_loss(feudalnet, aux_rollout_buffer, args, device)
                torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
                optimizer.step()
                torch.cuda.empty_cache()
            wandb.log({"global_step": step, **aux_dict})
            aux_rollout_buffer = []
            pi_phase_count     = 0


        # Log update metrics
        wandb.log({"global_step": step, "charts/learning_rate": lr_now, **loss_dict})

        # Console logging with ETA
        completed = ep_tracker.flush()  # peek without clearing
        ep_tracker.finished = completed  # put back
        if completed:
            mean_ep  = np.mean([e["episode/total_reward"] for e in completed])
            mean_win = np.mean([e["charts/episode_reward/WinLossRewardFunction"] for e in completed])
            ep_str   = f"ep_ret={mean_ep:.2f}  win={mean_win:.2f}"
        else:
            ep_str = "ep_ret=n/a  win=n/a"

        # Steps/s and ETA — measured from training start (or resume point)
        elapsed_total   = time.time() - train_start
        steps_done      = step - steps_at_start
        sps             = steps_done / max(elapsed_total, 1e-6)
        steps_remaining = args.max_steps - step
        eta_sec         = steps_remaining / max(sps, 1)

        # Format ETA as Xh Ym or Xm Ys
        eta_h, rem  = divmod(int(eta_sec), 3600)
        eta_m, eta_s = divmod(rem, 60)
        if eta_h > 0:
            eta_str = f"{eta_h}h {eta_m:02d}m"
        else:
            eta_str = f"{eta_m}m {eta_s:02d}s"

        pct = 100 * step / args.max_steps

        wandb.log({"global_step": step, "charts/steps_per_sec": sps})

        print(
            f"step={step:>10,} ({pct:4.1f}%) | {ep_str} | "
            f"loss={loss_dict['loss/total']:.4f} | "
            f"r_int={loss_dict['worker/intrinsic_reward']:.4f} | "
            f"{sps:,.0f} sps | ETA {eta_str}"
        )

        # --- Checkpoint ---
        # `args` carries the restored architecture, so the fine-tuned file is
        # self-describing exactly like the source: evaluate_agent.py and this
        # script can both load it without extra flags.
        ckpt_data = {
            "model": feudalnet.state_dict(),
            "optim": optimizer.state_dict(),
            "step":  step,
            "args":  vars(args),
            "source_checkpoint": args.checkpoint,
            "source_step":       source_step,
            "finetune_opponent": args.opponent,
            # Stored so --resume can rejoin this same wandb run unaided
            "wandb_run_id":      run.id,
        }
        # Rolling save so a crash never loses the fine-tuning progress
        torch.save(ckpt_data, args.out_checkpoint)

        if len(save_steps) > 0 and step > save_steps[0]:
            numbered_path = os.path.join(
                args.save_dir, f"{ckpt_stem}_step={step}.pt")
            torch.save(ckpt_data, numbered_path)
            wandb.save(numbered_path)
            print(f"Checkpoint saved → {numbered_path}")
            save_steps.pop(0)

        # --- Periodic evaluation & early stopping ---
        if args.eval_every > 0 and step >= next_eval_at:
            next_eval_at += args.eval_every
            print(f"[eval] step={step:,} | playing {args.eval_games} games "
                  f"vs {args.eval_opponent}...")
            winrate = evaluator.evaluate(feudalnet, step)
            wandb.log({"global_step": step, "eval/winrate": winrate})
            print(f"[eval] step={step:,} | winrate={100*winrate:.1f}% "
                  f"(target {100*args.eval_winrate:.0f}%)")

            if winrate >= args.eval_winrate:
                torch.save(ckpt_data, args.out_checkpoint)
                wandb.save(args.out_checkpoint)
                print(f"Early stopping: winrate {100*winrate:.1f}% >= "
                      f"{100*args.eval_winrate:.0f}%")
                break

    # --- Final checkpoint ---
    final = {
        "model": feudalnet.state_dict(),
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

    envs.close()
    wandb.finish()
    print("\n" + "=" * 62)
    print(f"Fine-tuning complete: {step:,} steps vs {args.opponent}")
    print(f"  source : {args.checkpoint} (step {source_step:,})")
    print(f"  output : {args.out_checkpoint}")
    print(f"  wandb  : {run.name}")
    print("=" * 62)


if __name__ == "__main__":
    if args.export_only:
        # Pure file operation: no envs, no JVM, no wandb run
        export_checkpoint(args)
    else:
        experiment(args)
