import os
import argparse
import numpy as np
import torch
import wandb
import math
import time
from itertools import cycle

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

parser = argparse.ArgumentParser(description='FeudalNet for gym-microrts')

# Generic
parser.add_argument('--lr',           type=float, default=2.5e-4)
parser.add_argument('--num-workers',  type=int,   default=24)
parser.add_argument('--num-steps',    type=int,   default=50)
parser.add_argument('--max-steps',    type=int,   default=int(1e8))
parser.add_argument('--cuda',         type=str2bool, default=True)
parser.add_argument('--grad-clip',    type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--seed',         type=int,   default=0)

# FeudalNet
parser.add_argument('--time-horizon',       type=int,   default=100)
parser.add_argument('--hidden-dim',         type=int,   default=64,
                    help='Embedding dim d used by perception, Manager and Worker')
parser.add_argument('--gamma-w',   type=float, default=0.99)
parser.add_argument('--gamma-m',   type=float, default=0.999)
parser.add_argument('--alpha',      type=float, default=0.5)
parser.add_argument('--gae-lambda', type=float, default=0.95,
                    help='GAE lambda — blends TD(1) and Monte Carlo returns')
parser.add_argument('--value-coef',  type=float, default=1.0,
                    help='Coefficient on the value losses (policy/entropy use mean-logp scaling, so on bigger maps the value loss dominates; lower this on 16x16)')
parser.add_argument('--n-pi',       type=int,   default=8,
                    help='Policy phases between each PPG auxiliary phase')
parser.add_argument('--e-aux',      type=int,   default=2,
                    help='Epochs over aux buffer per auxiliary phase')
parser.add_argument('--eps',       type=float, default=1e-4)
parser.add_argument('--cosine',    type=str2bool, default=True)

# In-training evaluation / early stopping
parser.add_argument('--eval-every',    type=int,   default=int(1e7),
                    help='Evaluate every N env steps (0 disables evaluation)')
parser.add_argument('--eval-opponent', type=str,   default='coacAI',
                    help='Opponent to evaluate against')
parser.add_argument('--eval-games',    type=int,   default=100,
                    help='Number of games per evaluation')
parser.add_argument('--eval-winrate',  type=float, default=1.0,
                    help='Stop training when eval winrate >= this fraction (1.0 = 100%%)')
parser.add_argument('--eval-workers',  type=int,   default=8,
                    help='Parallel envs used for evaluation')
parser.add_argument('--eval-record',   type=str2bool, default=True,
                    help='Record a video of each evaluation and upload to wandb')
parser.add_argument('--eval-video-length', type=int, default=2000,
                    help='Max frames per evaluation video')


# Enemy encoder
parser.add_argument('--enemy-dim',      type=int, default=64)
parser.add_argument('--enemy-layers',   type=int, default=4)

# Transformer
parser.add_argument('--T-W',            type=int, default=25,
                    help='Worker context window (steps)')
parser.add_argument('--T-M',            type=int, default=100,
                    help='Manager context window (Manager steps)')
parser.add_argument('--worker-heads',   type=int, default=4)
parser.add_argument('--worker-layers',  type=int, default=2)
parser.add_argument('--manager-heads',  type=int, default=4)
parser.add_argument('--manager-layers', type=int, default=3)

# microRTS
parser.add_argument('--map-h',       type=int, default=16)
parser.add_argument('--map-w',       type=int, default=16)
parser.add_argument('--in-channels', type=int, default=27)
parser.add_argument('--opponents', nargs='+', default=[
    "coacAI",         "coacAI",         "coacAI",
    "coacAI",         "coacAI",         "coacAI",
    "coacAI",         "coacAI",         "coacAI",
    "coacAI",         "coacAI",         "coacAI",
    "workerRushAI",   "workerRushAI",   "workerRushAI",
    "workerRushAI",   "lightRushAI",    "lightRushAI",
    "lightRushAI",    "lightRushAI",    "randomBiasedAI",
    "randomBiasedAI", "randomBiasedAI", "randomBiasedAI",
])

# wandb
parser.add_argument('--wandb-project', type=str, default='fun-microrts')
parser.add_argument('--wandb-entity',  type=str, default=None)
parser.add_argument('--run-name',      type=str, default='feudalnet-8x8')

# Resume
parser.add_argument('--resume',   action='store_true')
parser.add_argument('--run-id',   type=str, default=None)
parser.add_argument('--save-dir', type=str, default='models')
parser.add_argument('--run-id-file', type=str, default='run_id.txt',
                    help='File to write the wandb run id to (for slurm dependency jobs)')
parser.add_argument('--config-csv', type=str, default=None,
                    help='Where to write the run parameter CSV. Default: '
                         '<save-dir>/configs/<run name>_<run id>_config.csv')
parser.add_argument('--no-config-csv', dest='config_csv_enabled',
                    action='store_false',
                    help='Skip writing the run parameter CSV')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_envs(args):
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from gym_microrts import microrts_ai

    ai_cls_map = {
        "coacAI":       microrts_ai.coacAI,
        "workerRushAI": microrts_ai.workerRushAI,
        "lightRushAI":  microrts_ai.lightRushAI,
        "randomBiasedAI":     microrts_ai.randomBiasedAI,
    }

    opp_cycle    = cycle(args.opponents)
    env_opps     = [next(opp_cycle) for _ in range(args.num_workers)]
    ai2s         = [ai_cls_map[o] for o in env_opps]

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=args.num_workers,
        max_steps=4000,
        render_theme=2,
        partial_obs=False,
        ai2s=ai2s,
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
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
            dilation=10,
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

    # --- wandb ---
    if args.resume and args.run_id:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.run_id,
            resume="must",
        )
        print(f"Resuming wandb run: {args.run_id}")
    else:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            save_code=True,
        )

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
        dilation=10,
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

    # --- Resume checkpoint ---
    step = 0
    if args.resume and args.run_id:
        ckpt_path = os.path.join(args.save_dir, f"{args.run_id}_latest.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # strict=False: older checkpoints carry now-removed critic_aux keys
        feudalnet.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"Resumed from step {step:,}")

    # --- Init state ---
    goals, states, masks = feudalnet.init_obj()
    aux_rollout_buffer = []
    pi_phase_count     = 0
    ep_tracker = EpisodeTracker(B=args.num_workers, env_opponents=env_opponents)

    # --- In-training evaluator & early stopping ---
    evaluator = Evaluator(args, action_space, device)
    if args.eval_every > 0:
        # Align next eval to the next multiple of eval_every (handles resume)
        next_eval_at = ((step // args.eval_every) + 1) * args.eval_every
    else:
        next_eval_at = float('inf')

    save_steps = list(torch.arange(
        0, int(args.max_steps), int(args.max_steps) // 10
    ).numpy())

    print(f"Training | {args.num_workers} envs | {args.max_steps:,} steps | wandb: {run.name}")

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
            # envs.render()  # disabled: conflicts with the eval env's
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

        # Checkpoint
        # Always save a latest.pt so resume always works
        ckpt_data = {
            "model": feudalnet.state_dict(),
            "optim": optimizer.state_dict(),
            "step":  step,
            "args":  vars(args),
        }
        latest_path = os.path.join(args.save_dir, f"{run.id}_latest.pt")
        torch.save(ckpt_data, latest_path)

        if len(save_steps) > 0 and step > save_steps[0]:
            numbered_path = os.path.join(args.save_dir, f"{run.id}_step={step}.pt")
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
                early_path = os.path.join(
                    args.save_dir, f"{run.id}_earlystop_step={step}.pt"
                )
                torch.save(ckpt_data, early_path)
                wandb.save(early_path)
                print(f"Early stopping: winrate {100*winrate:.1f}% >= "
                      f"{100*args.eval_winrate:.0f}% → {early_path}")
                break

    envs.close()
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    experiment(args)
