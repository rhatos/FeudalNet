import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize
import torch.nn.functional as F
import numpy as np

from transformer import WorkerTransformer, ManagerTransformer
from enemy_encoder import EnemyEncoder

# ---------------------------------------------------------------------------
# CategoricalMasked — correct masked entropy (from base_agent.py)
# Uses -1e8 instead of -inf to avoid NaN in entropy calculation.
# Entropy is computed only over valid (unmasked) actions.
# ---------------------------------------------------------------------------

class CategoricalMasked(torch.distributions.Categorical):
    """
    Categorical distribution with invalid action masking.
    Matches ppo_gridnet.py exactly — uses -1e8 mask_value (not -inf)
    to prevent NaN in entropy. mask_value stored as tensor attribute.
    """
    def __init__(self, logits, masks=None, mask_value=None):
        self.masks = masks
        if masks is not None:
            if mask_value is None:
                mask_value = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            logits = torch.where(masks.bool(), logits, mask_value)
        super().__init__(logits=logits)


# gym-microrts 8x8 per-unit action space
ACTION_SPACE = [6, 4, 4, 4, 4, 7, 49]   # sum = 78


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Perception — replaces Atari CNN with microRTS grid CNN
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    IMPALA-CNN residual block.
    Two conv layers with a residual connection — gives gradients a clean
    path through the network and prevents feature collapse.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual   # residual connection


class Perception(nn.Module):
    """
    IMPALA-CNN encoder for gym-microrts observations.

    Replaces the plain CNN with the IMPALA architecture — deeper residual
    blocks that produce richer, more stable feature representations.
    Residual connections prevent gradient vanishing and feature collapse,
    which is critical for the Manager's state space to be meaningful.

    Architecture (per stage):
        Conv2d → ResBlock → MaxPool (optional)

    Input:  (B, H, W, F)
    Output: (B, d)
    """
    def __init__(self, h, w, in_channels=27, d=256):
        super().__init__()

        # Three stages with increasing channel depth
        # No pooling on small 8x8 maps to preserve spatial resolution
        self.stage1 = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)),
            ResBlock(16),
        )
        self.stage2 = nn.Sequential(
            layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            ResBlock(32),
        )
        self.stage3 = nn.Sequential(
            layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            ResBlock(32),
        )

        # Compute flattened size after stages
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, h, w)
            cnn_out = self._forward_cnn(dummy).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(cnn_out, d)),
            nn.ReLU(),
        )

    def _forward_cnn(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x   # (B, 32, H, W) — spatial features before flatten

    def forward(self, obs):
        # obs: (B, H, W, F) → permute to (B, F, H, W) for Conv2d
        x       = obs.permute(0, 3, 1, 2).float()
        spatial = self._forward_cnn(x)            # (B, 32, H, W)
        z       = self.fc(spatial.flatten(1))     # (B, d)
        return z, spatial


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class Manager(nn.Module):
    """
    Manager using ManagerTransformer.

    Replaces DilatedLSTM with a transformer over a 100-step context window.
    Only updates every c steps — true temporal abstraction.
    Between Manager steps, cached goal and value are reused.

    State:
        tf_state:     (context (B,T_M,d), step_count)
        cached_goal:  (B, d)
        cached_value: (B, 1)
    """
    def __init__(self, c, d, eps, device, d_e=64,
                 n_heads=4, n_layers=3, T_M=100):
        super().__init__()
        self.c      = c
        self.d      = d
        self.eps    = eps
        self.device = device

        # Project CNN embedding into Manager state space
        self.Mspace = nn.Linear(d, d)

        # Fuse with enemy embedding if provided
        self.fusion = layer_init(nn.Linear(d + d_e, d))

        # Transformer replaces DilatedLSTM
        self.Mrnn = ManagerTransformer(d=d, n_heads=n_heads,
                                       n_layers=n_layers, T_M=T_M, c=c)
        self.critic = nn.Linear(d, 1)

    def init_state(self, batch_size, device=None):
        dev = device or torch.device("cpu")
        tf_state     = self.Mrnn.init_state(batch_size, device=dev)
        cached_goal  = torch.zeros(batch_size, self.d, device=dev)
        cached_value = torch.zeros(batch_size, 1,      device=dev)
        return tf_state, cached_goal, cached_value

    def forward(self, z, state, mask, enemy_emb=None):
        tf_state, cached_goal, cached_value = state

        # Reset cached goal/value on episode boundary — otherwise the dead
        # episode's goal keeps biasing the new episode's Worker for up to
        # c steps, and its value pollutes the new episode's targets
        cached_goal  = cached_goal  * mask
        cached_value = cached_value * mask

        s = self.Mspace(z).relu()

        if enemy_emb is not None:
            s = self.fusion(torch.cat([s, enemy_emb], dim=-1)).relu()

        # Apply episode mask to context buffer
        context, step_count = tf_state
        context = context * mask.unsqueeze(-1)

        goal_hat, is_m_step, tf_state = self.Mrnn(s, (context, step_count))

        if is_m_step:
            goal = normalize(goal_hat)
            if self.eps > torch.rand(1)[0]:
                goal = torch.randn_like(goal, requires_grad=False)
            value        = self.critic(goal_hat)
            cached_goal  = goal.detach()
            # NOT detached: the cached prediction is reused for the next c
            # steps, so every per-step value loss in the window contributes
            # gradient to this one prediction. With .detach() the Manager
            # critic only learned on Manager steps (~4% of samples).
            # repackage_hidden() detaches at rollout boundaries.
            cached_value = value
        else:
            goal  = cached_goal
            value = cached_value

        new_state = (tf_state, cached_goal, cached_value)
        return goal, value, s.detach(), new_state

    def state_goal_cosine(self, states, goals, masks):
        t    = self.c
        mask = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)
        cos  = d_cos(states[t + self.c] - states[t], goals[t])
        return mask * cos.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Worker — multi-discrete for microRTS
# ---------------------------------------------------------------------------

class Worker(nn.Module):
    """
    Worker adapted for gym-microrts:
        - Multi-discrete action space: 7 sub-action heads per cell
        - Per-cell actions across H*W = 64 cells
        - LSTM hidden = k * sum_pu (flat U matrix, same as original design)
          BUT we decouple LSTM size from U to avoid memory explosion:
          LSTM hidden = d, then fc_U projects to k * sum_pu * n_cells
    """
    def __init__(self, b, c, d, n_cells, action_space, device,
                 T_W=25, n_heads=4, n_layers=2):
        super().__init__()
        self.b            = b
        self.c            = c
        self.n_cells      = n_cells
        self.action_space = list(action_space)
        self.sum_pu       = sum(action_space)
        self.device       = device

        # WorkerTransformer — context window and depth configurable
        self.Wrnn = WorkerTransformer(d=d, n_heads=n_heads, n_layers=n_layers, T_W=T_W)

        # ConvTranspose actor decoder — spatially aware, matches ppo_gridnet
        # Takes IMPALA-CNN spatial features (B, 32, H, W) and decodes to
        # per-cell logits (B, H, W, sum_pu). Each cell sees its local context.
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, self.sum_pu, kernel_size=3, stride=1, padding=1), std=0.01),
        )

        # Goal projection — maps goal to sum_pu-dim bias broadcast over all cells
        self.phi = nn.Linear(d, self.sum_pu, bias=False)

        # Critic — direct Linear(d, 1) from LSTM hidden state
        self.critic = layer_init(nn.Linear(d, 1), std=1.0)

    def forward(self, z, spatial, goals, hidden, mask, action_mask=None):
        """
        Args:
            z:       (B, d)          global CNN embedding for LSTM
            spatial: (B, 32, H, W)  spatial CNN features for ConvTranspose actor
        """
        # Apply episode mask to context buffer — reset on done
        context = hidden * mask.unsqueeze(-1)
        u, context = self.Wrnn(z, context)

        # Expose hidden for the value-replay aux phase (stored detached)
        self.last_u = u.detach()

        # Accumulated goal projected to action-space bias
        goals_sum = torch.stack(goals).detach().sum(dim=0)   # (B, d)
        goal_bias = self.phi(goals_sum)                       # (B, sum_pu)

        value_est = self.critic(u)

        # ConvTranspose decoder — spatially aware per-cell logits
        # spatial: (B, 32, H, W) → actor → (B, sum_pu, H, W)
        # permute → (B, H, W, sum_pu) → reshape → (B, n_cells, sum_pu)
        B = z.shape[0]
        logits_spatial = self.actor(spatial)                               # (B, sum_pu, H, W)
        logits_flat    = logits_spatial.permute(0, 2, 3, 1).reshape(B, self.n_cells, self.sum_pu)

        # Add goal bias — Manager's goal shifts all cells' distributions uniformly
        logits_flat = logits_flat + goal_bias.unsqueeze(1)

        # Split into 7 heads and apply mask using CategoricalMasked (-1e8)
        logits_splits = logits_flat.split(self.action_space, dim=-1)
        if action_mask is not None:
            if action_mask.dim() == 2:
                action_mask = action_mask.view(B, self.n_cells, self.sum_pu)
            mask_splits = action_mask.split(self.action_space, dim=-1)
        else:
            mask_splits = [None] * len(self.action_space)

        mask_value = torch.tensor(-1e8, dtype=logits_flat.dtype, device=logits_flat.device)
        logits_list = []
        for logits_i, mask_i in zip(logits_splits, mask_splits):
            if mask_i is not None:
                logits_i = torch.where(mask_i.bool(), logits_i, mask_value)
            logits_list.append(logits_i)

        return logits_list, context, value_est

    def intrinsic_reward(self, states, goals, masks):
        t    = self.c
        r_i  = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)

        for i in range(1, self.c + 1):
            r_i_t = d_cos(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i  += mask * r_i_t
            mask  = mask * masks[t - i]

        return r_i.detach() / self.c


# ---------------------------------------------------------------------------
# FeudalNetwork
# ---------------------------------------------------------------------------

class FeudalNetwork(nn.Module):
    def __init__(self, num_workers, h, w, in_channels, d, n_cells,
                 action_space, time_horizon, dilation, eps, device,
                 enemy_dim=64, enemy_layers=2,
                 worker_heads=4, worker_layers=2, T_W=25,
                 manager_heads=4, manager_layers=3, T_M=100):
        super().__init__()
        self.b      = num_workers
        self.c      = time_horizon
        self.d      = d
        self.device = device

        self.percept       = Perception(h=h, w=w, in_channels=in_channels, d=d)
        self.enemy_encoder = EnemyEncoder(d_e=enemy_dim, map_size=h * w, num_layers=enemy_layers)
        self.manager       = Manager(c=time_horizon, d=d, eps=eps, device=device,
                                     d_e=enemy_dim, n_heads=manager_heads,
                                     n_layers=manager_layers, T_M=T_M)
        self.worker        = Worker(b=num_workers, c=time_horizon, d=d,
                                    n_cells=n_cells, action_space=action_space,
                                    device=device, T_W=T_W,
                                    n_heads=worker_heads, n_layers=worker_layers)

        # Manager state: (tf_state, cached_goal, cached_value)
        self.state_m = None

        # Worker context buffer: (B, T_W, d)
        self.context_w = torch.zeros(num_workers, T_W, d).to(device)

        # Enemy encoder hidden state
        self.hidden_e = self.enemy_encoder.init_state(num_workers, device=device)
        self.to(device)

    def forward(self, x, goals, states, mask, action_mask=None, save=True):
        z, spatial = self.percept(x)

        # Enemy encoder — reset LSTM state on episode boundary, otherwise
        # the previous episode's enemy history leaks into the new one
        hidden_e_in = [(h * mask, c_ * mask) for h, c_ in self.hidden_e]
        enemy_emb, hidden_e = self.enemy_encoder(x, hidden_e_in)

        # Manager — uses transformer state
        goal, value_m, s, new_state_m = self.manager(
            z, self.state_m, mask, enemy_emb=enemy_emb
        )

        # Only mutate the history when save=True. The bootstrap forward
        # (save=False) must NOT append: the same observation is forwarded
        # again as the first step of the next rollout, and a duplicate entry
        # permanently shifts the temporal alignment of intrinsic_reward and
        # state_goal_cosine.
        if save:
            if len(goals) > (2 * self.c + 1):
                goals.pop(0)
                states.pop(0)
            goals.append(goal)
            states.append(s)

        # Worker — uses context buffer
        logits_list, new_context_w, value_w = self.worker(
            z, spatial, goals[:self.c + 1], self.context_w, mask,
            action_mask=action_mask
        )

        if save:
            self.state_m   = new_state_m
            self.context_w = new_context_w
            self.hidden_e  = hidden_e

        return logits_list, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)

    def repackage_hidden(self):
        # Detach Manager transformer state
        if self.state_m is not None:
            tf_state, cached_goal, cached_value = self.state_m
            context, step_count = tf_state
            self.state_m = (
                (context.detach(), step_count),
                cached_goal.detach(),
                cached_value.detach(),
            )
        # Detach Worker context buffer
        self.context_w = self.context_w.detach()
        # Detach enemy encoder state
        self.hidden_e = [(h.detach(), c.detach()) for h, c in self.hidden_e]

    def init_obj(self):
        # Initialise Manager transformer state
        self.state_m = self.manager.init_state(self.b, device=self.device)
        template = torch.zeros(self.b, self.d).to(self.device)
        goals  = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        states = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        masks  = [torch.ones(self.b, 1).to(self.device) for _ in range(2 * self.c + 1)]
        return goals, states, masks

    def evaluate(self, logits_list, actions):
        """
        Compute log probs and entropy for stored actions.
        Used in PPO's update epochs to compute the probability ratio.

        Args:
            logits_list: 7 tensors of (B, n_cells, n_i) — already masked
            actions:     (B, n_cells, 7) stored actions from rollout

        Returns:
            log_probs: (B,)   sum of log probs across cells and heads
            entropy:   scalar mean entropy
        """
        log_probs_list = []
        entropy_list   = []
        for i, logits_i in enumerate(logits_list):
            dist       = CategoricalMasked(logits=logits_i)
            action_i   = actions[:, :, i]   # (B, n_cells)
            log_probs_list.append(dist.log_prob(action_i))
            entropy_list.append(dist.entropy())

        log_probs = torch.stack(log_probs_list, dim=-1).mean(dim=-1)  # (B, n_cells)
        entropy   = torch.stack(entropy_list,   dim=-1).mean()
        return log_probs, entropy

    @staticmethod
    def sample(logits_list):
        # logits already have -1e8 for invalid actions (applied in Worker.forward)
        # CategoricalMasked computes correct entropy excluding masked actions
        actions_list   = []
        log_probs_list = []

        for logits_i in logits_list:
            # No mask needed here — -1e8 values already suppress invalid actions
            dist       = CategoricalMasked(logits=logits_i)
            action_i   = dist.sample()
            log_prob_i = dist.log_prob(action_i)
            actions_list.append(action_i)
            log_probs_list.append(log_prob_i)

        actions   = torch.stack(actions_list,   dim=-1)
        log_probs = torch.stack(log_probs_list, dim=-1).mean(dim=-1)
        entropy   = torch.stack([
            CategoricalMasked(logits=l).entropy()
            for l in logits_list
        ], dim=-1).mean()

        return actions, log_probs, entropy


# ---------------------------------------------------------------------------
# Loss (adapted from feudal_loss in original)
# ---------------------------------------------------------------------------

def feudal_loss(storage, next_v_m, next_v_w, args):
    """
    Feudal loss with GAE.
    No return normalisation — destabilises sparse rewards.
    Advantage normalisation for stable gradient scale.
    Writes ret_w/ret_m back to storage for PPG auxiliary phase.
    """
    storage.placeholder()

    # Bootstrap values are targets, not predictions — never let gradient
    # flow into them (the Manager's cached-value path can arrive attached)
    next_v_m = next_v_m.detach()
    next_v_w = next_v_w.detach()

    rewards_intrinsic, value_m, value_w, logps, entropy, \
        state_goal_cosines, masks, rewards = storage.stack(
            ['r_i', 'v_m', 'v_w', 'logp', 'entropy',
             's_goal_cos', 'm', 'r'])

    # GAE for Worker
    adv_w = torch.zeros_like(value_w)
    gae_w = torch.zeros_like(next_v_w)
    for i in reversed(range(args.num_steps)):
        next_val = value_w[i + 1] if i < args.num_steps - 1 else next_v_w
        delta_w  = rewards[i] + args.gamma_w * next_val * masks[i] - value_w[i]
        gae_w    = delta_w + args.gamma_w * args.gae_lambda * masks[i] * gae_w
        adv_w[i] = gae_w
    ret_w = adv_w + value_w

    # GAE for Manager
    adv_m = torch.zeros_like(value_m)
    gae_m = torch.zeros_like(next_v_m)
    for i in reversed(range(args.num_steps)):
        next_val = value_m[i + 1] if i < args.num_steps - 1 else next_v_m
        delta_m  = rewards[i] + args.gamma_m * next_val * masks[i] - value_m[i]
        gae_m    = delta_m + args.gamma_m * args.gae_lambda * masks[i] * gae_m
        adv_m[i] = gae_m
    ret_m = adv_m + value_m

    # Write returns back to storage for PPG auxiliary phase
    for i in range(args.num_steps):
        storage.ret_w[i] = ret_w[i]
        storage.ret_m[i] = ret_m[i]

    # Worker advantage includes intrinsic reward
    advantage_w = adv_w + args.alpha * rewards_intrinsic
    advantage_m = adv_m

    # Normalise advantages — stable gradient scale
    advantage_w = (advantage_w - advantage_w.mean()) / (advantage_w.std() + 1e-8)
    advantage_m = (advantage_m - advantage_m.mean()) / (advantage_m.std() + 1e-8)

    loss_worker  = (logps * advantage_w.detach()).mean()
    loss_manager = (state_goal_cosines * advantage_m.detach()).mean()

    value_w_loss = 0.5 * (ret_w - value_w).pow(2).mean()
    value_m_loss = 0.5 * (ret_m - value_m).pow(2).mean()

    entropy = entropy.mean()
    value_coef = getattr(args, 'value_coef', 1.0)
    loss = (-loss_worker - loss_manager
            + value_coef * (value_w_loss + value_m_loss)
            - args.entropy_coef * entropy)

    return loss, {
        'loss/total':              loss.item(),
        'loss/worker':             loss_worker.item(),
        'loss/manager':            loss_manager.item(),
        'loss/value_worker':       value_w_loss.item(),
        'loss/value_manager':      value_m_loss.item(),
        'worker/entropy':          entropy.item(),
        'worker/advantage':        advantage_w.mean().item(),
        'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
        'manager/cosines':         state_goal_cosines.mean().item(),
        'manager/advantage':       advantage_m.mean().item(),
    }


# ---------------------------------------------------------------------------
# Value Replay Auxiliary Phase
# ---------------------------------------------------------------------------

def ppg_aux_loss(feudalnet, aux_buffer, args, device):
    """
    Value replay: retrains the REAL worker critic on stored
    (hidden, GAE return) pairs from the last n_pi rollouts.

    The hidden u was computed and detached at collection time, so features
    are consistent with the returns and no policy gradient is involved —
    the critic head simply gets more passes over the same expensive samples.
    One batched op per call; orders of magnitude faster than recomputing
    perception per entry.
    """
    if len(aux_buffer) == 0:
        dummy = torch.tensor(0.0, device=device)
        return dummy, {'ppg/aux_value_worker': 0.0, 'ppg/aux_n': 0}

    u_all   = torch.cat([e['u']     for e in aux_buffer], dim=0).to(device)
    ret_all = torch.cat([e['ret_w'] for e in aux_buffer], dim=0).to(device)

    # NaN guard — drop corrupt rows
    valid = ~(ret_all.isnan().any(dim=-1) | u_all.isnan().any(dim=-1))
    if valid.sum() == 0:
        dummy = torch.tensor(0.0, device=device)
        return dummy, {'ppg/aux_value_worker': 0.0, 'ppg/aux_n': 0}
    u_all, ret_all = u_all[valid], ret_all[valid]

    v_pred = feudalnet.worker.critic(u_all)
    loss   = 0.5 * (v_pred - ret_all.detach()).pow(2).mean()
    loss.backward()

    return loss.detach(), {
        'ppg/aux_value_worker': loss.item(),
        'ppg/aux_n':            int(valid.sum().item()),
    }
