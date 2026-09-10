"""
transformer.py

Transformer building blocks for FeudalNet.

Uses torch.nn.functional.scaled_dot_product_attention which automatically
uses Flash Attention on CUDA with PyTorch 2.0+.

Modules:
    FullSelfAttention      — full (bidirectional) attention, every token sees every other
    TransformerBlock       — pre-norm attention + feedforward with residuals
    WorkerTransformer      — rolling 25-step context window, full attention
    ManagerTransformer     — rolling 100-step context window, full attention,
                             only updates every c steps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Full Self Attention
# ---------------------------------------------------------------------------

class FullSelfAttention(nn.Module):
    """
    Multi-head full (bidirectional) self-attention.

    Every token attends to every other token in the context window.
    Used for both Worker and Manager — both reason over a fixed historical
    buffer where full context visibility is appropriate.

    Flash Attention is used automatically via scaled_dot_product_attention
    when running on CUDA with PyTorch >= 2.0.

    Args:
        d:       embedding dimension
        n_heads: number of attention heads (must divide d)
    """

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"

        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.d       = d

        self.qkv  = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        """
        Args:
            x: (B, T, d)

        Returns:
            out: (B, T, d)
        """
        B, T, _ = x.shape

        q, k, v = self.qkv(x).split(self.d, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Flash Attention — no causal mask, full attention over context window
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm residuals.

    Pre-norm (LayerNorm before each sublayer) is more stable than post-norm,
    especially early in training when gradients can be large.

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = FullSelfAttention(d, n_heads)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            layer_init(nn.Linear(d, 4 * d)),
            nn.GELU(),
            layer_init(nn.Linear(4 * d, d)),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Worker Transformer
# ---------------------------------------------------------------------------

class WorkerTransformer(nn.Module):
    """
    Transformer for the Worker with a rolling 25-step context window.

    Each step:
        1. Append new input z to context buffer, drop oldest
        2. Add positional embeddings
        3. Run full attention over all 25 steps
        4. Return last token output as h_t (equivalent to LSTM hidden state)

    Full attention — every step in the window can attend to every other step.
    The last token output represents the current timestep's contextualised
    representation, incorporating information from the last 25 steps.

    State:
        context: (B, T_W, d) — rolling buffer of past inputs
    """

    def __init__(self, d, n_heads=4, n_layers=2, T_W=25):
        super().__init__()
        self.d   = d
        self.T_W = T_W

        # Learned positional embeddings — one per context slot
        # Position 0 = oldest, position T_W-1 = most recent
        self.pos_emb = nn.Embedding(T_W, d)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d=d, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d)

    def init_state(self, batch_size, device=None):
        """
        Context buffer initialised to zeros.
        Shape: (B, T_W, d)
        """
        dev = device or torch.device("cpu")
        return torch.zeros(batch_size, self.T_W, self.d, device=dev)

    def forward(self, z, context):
        """
        Args:
            z:       (B, d)       new input for this step
            context: (B, T_W, d)  rolling context buffer

        Returns:
            h_t:     (B, d)       current output — replaces LSTM hidden state
            context: (B, T_W, d)  updated context buffer
        """
        # Append new input, drop oldest (shift left)
        context = torch.cat([context[:, 1:, :], z.unsqueeze(1)], dim=1)  # (B, T_W, d)

        # Add positional embeddings
        positions = torch.arange(self.T_W, device=z.device)
        x = context + self.pos_emb(positions).unsqueeze(0)               # (B, T_W, d)

        # Full attention over all T_W steps
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)

        # Last token = current timestep's output
        h_t = x[:, -1, :]   # (B, d)

        return h_t, context


# ---------------------------------------------------------------------------
# Manager Transformer
# ---------------------------------------------------------------------------

class ManagerTransformer(nn.Module):
    """
    Transformer for the Manager with a rolling 100-step context window.

    Unlike the Worker which runs every step, the Manager only updates
    every c steps — true temporal abstraction. Between Manager steps
    the cached goal and value are reused.

    On a Manager step:
        1. Append new Manager state to context buffer, drop oldest
        2. Add positional embeddings
        3. Run full attention over all 100 Manager-level steps
        4. Return last token as goal_hat

    On a non-Manager step:
        Returns None for goal_hat — caller uses cached goal

    State:
        context:    (B, T_M, d) — rolling buffer of past Manager states
        step_count: int          — tracks which env steps are Manager steps
    """

    def __init__(self, d, n_heads=4, n_layers=3, T_M=100, c=10):
        super().__init__()
        self.d   = d
        self.T_M = T_M
        self.c   = c

        # Learned positional embeddings
        self.pos_emb = nn.Embedding(T_M, d)

        # Deeper than Worker — Manager runs less often so can afford more layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d=d, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d)

    def init_state(self, batch_size, device=None):
        """
        Returns:
            context:    (B, T_M, d)  Manager context buffer
            step_count: int           tracks Manager update steps
        """
        dev = device or torch.device("cpu")
        return torch.zeros(batch_size, self.T_M, self.d, device=dev), 0

    def forward(self, s, state):
        """
        Args:
            s:     (B, d)                    Manager latent state for this step
            state: (context, step_count)

        Returns:
            goal_hat:      (B, d) or None    None on non-Manager steps
            is_manager_step: bool
            state:         updated (context, step_count)
        """
        context, step_count = state
        is_manager_step     = (step_count % self.c == 0)

        if is_manager_step:
            # Append new state, drop oldest
            context = torch.cat([context[:, 1:, :], s.unsqueeze(1)], dim=1)

            # Add positional embeddings
            positions = torch.arange(self.T_M, device=s.device)
            x = context + self.pos_emb(positions).unsqueeze(0)

            # Full attention over all T_M past Manager states
            for block in self.blocks:
                x = block(x)

            x = self.ln_out(x)

            # Last token = current Manager output
            goal_hat = x[:, -1, :]   # (B, d)
        else:
            goal_hat = None

        step_count = step_count + 1
        return goal_hat, is_manager_step, (context, step_count)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    B, d = 4, 256
    T_W, T_M, c = 25, 100, 10

    print("Testing WorkerTransformer...")
    worker = WorkerTransformer(d=d, n_heads=4, n_layers=2, T_W=T_W)
    ctx    = worker.init_state(B)
    z      = torch.rand(B, d)

    t0 = time.time()
    for _ in range(T_W):
        h_t, ctx = worker(z, ctx)
    elapsed = (time.time() - t0) * 1000

    assert h_t.shape == (B, d)
    assert ctx.shape == (B, T_W, d)
    assert not h_t.isnan().any()
    print(f"  h_t:     {h_t.shape}")
    print(f"  context: {ctx.shape}")
    print(f"  time ({T_W} steps): {elapsed:.1f}ms")
    print("  WorkerTransformer OK ✓")

    print("\nTesting ManagerTransformer...")
    manager     = ManagerTransformer(d=d, n_heads=4, n_layers=3, T_M=T_M, c=c)
    state       = manager.init_state(B)
    s           = torch.rand(B, d)
    n_m_steps   = 0
    last_goal   = None

    for step in range(c * 3):
        goal_hat, is_m, state = manager(s, state)
        if is_m:
            n_m_steps += 1
            last_goal  = goal_hat
            assert goal_hat.shape == (B, d)
        else:
            assert goal_hat is None

    print(f"  Manager steps in {c*3} env steps: {n_m_steps} (expected 3)")
    assert n_m_steps == 3
    assert not last_goal.isnan().any()
    print(f"  goal_hat: {last_goal.shape}")
    print(f"  context:  {state[0].shape}")
    print("  ManagerTransformer OK ✓")

    n_w = sum(p.numel() for p in worker.parameters())
    n_m = sum(p.numel() for p in manager.parameters())
    print(f"\nWorker  parameters: {n_w:,}")
    print(f"Manager parameters: {n_m:,}")
    print("\ntransformer.py OK.")
