"""
enemy_encoder.py

Encodes enemy unit positions and types as a simple 8x8 grid,
then feeds it through an LSTM to track enemy behaviour over time.

Grid encoding:
    Each cell gets a single integer unit type ID:
        0 = empty (no enemy unit)
        1 = resource
        2 = base
        3 = barracks
        4 = worker
        5 = light
        6 = heavy
        7 = ranged

This gives a (B, 64) input per step — small, clean, and directly
interpretable. The LSTM accumulates this over time to build a summary
of what the enemy has been doing.
"""

import torch
import torch.nn as nn
import numpy as np

# Owner plane for player 2 (enemy)
ENEMY_OWNER_CHANNEL = 12

# Unit type planes 13-20 in order — index+1 gives the type ID
# plane 13 = no unit type (0), plane 14 = resource (1), etc.
# We use argmax over planes 13-20 to get the type ID for each cell
UNIT_TYPE_START = 13
UNIT_TYPE_END   = 21   # exclusive
N_UNIT_TYPES    = UNIT_TYPE_END - UNIT_TYPE_START   # 8


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class EnemyEncoder(nn.Module):
    """
    LSTM encoder for enemy behaviour using a simple grid representation.

    Each step the 8x8 map is encoded as a flat vector of unit type IDs
    (one integer per cell, 0 if no enemy unit). This is projected into
    an embedding and passed through an LSTMCell to accumulate history.

    Args:
        d_e:      output embedding size (default 64)
        map_size: H*W (64 for 8x8 map)

    State:
        (h, c_x): LSTM hidden and cell state (B, d_e) each

    Forward:
        obs:   (B, H, W, 27)
        state: (h, c_x)

    Returns:
        enemy_emb: (B, d_e)  latent vector for the Manager
        state:     updated (h, c_x)
    """

    def __init__(self, d_e=64, map_size=64, num_layers=1):
        super().__init__()
        self.d_e        = d_e
        self.map_size   = map_size
        self.num_layers = num_layers

        # Embedding table: one vector per unit type (0-7)
        # Maps each cell's type ID to a learned d_e-dim vector
        self.type_embedding = nn.Embedding(N_UNIT_TYPES, d_e)

        # Project the flattened grid embedding to d_e
        # Input: all 64 cell embeddings concatenated (B, 64 * d_e)
        # This preserves position — worker at cell 0 != worker at cell 63
        self.input_proj = nn.Sequential(
            layer_init(nn.Linear(map_size * d_e, d_e)),
            nn.ReLU(),
        )

        # Stack of LSTMCells — each layer takes the hidden state of the
        # previous layer as input, building increasingly abstract representations
        # Layer 0: reacts to immediate changes in enemy units
        # Layer 1+: builds higher-level patterns over time
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(d_e, d_e) for _ in range(num_layers)
        ])

        # Output projection
        self.out_proj = nn.Sequential(
            layer_init(nn.Linear(d_e, d_e)),
            nn.ReLU(),
        )

    def init_state(self, batch_size, device=None):
        """
        Returns a list of (h, c) tuples — one per LSTM layer.
        Each h and c has shape (B, d_e).
        """
        dev = device or torch.device("cpu")
        return [
            (torch.zeros(batch_size, self.d_e, device=dev),
             torch.zeros(batch_size, self.d_e, device=dev))
            for _ in range(self.num_layers)
        ]

    def extract_enemy_grid(self, obs):
        """
        Build a flat grid of enemy unit type IDs.

        For each cell:
            - If no enemy unit: type_id = 0
            - If enemy unit present: type_id = argmax over unit type planes + 1

        Args:
            obs: (B, H, W, 27)

        Returns:
            grid: (B, H*W)  long tensor of unit type IDs (0-7)
        """
        B, H, W, C = obs.shape
        flat = obs.view(B, H * W, C).float()           # (B, 64, 27)

        # Which cells have an enemy unit
        enemy_mask = flat[:, :, ENEMY_OWNER_CHANNEL] > 0   # (B, 64) bool

        # Get unit type for each cell — argmax over the 8 type planes
        # argmax gives 0-7, then we add 1 so 0 is reserved for "empty"
        unit_type_planes = flat[:, :, UNIT_TYPE_START:UNIT_TYPE_END]  # (B, 64, 8)
        type_ids = unit_type_planes.argmax(dim=-1) + 1                # (B, 64) values 1-8

        # Zero out cells that don't have an enemy unit
        grid = (type_ids * enemy_mask.long())                          # (B, 64) values 0-8

        # Clamp to valid embedding range [0, N_UNIT_TYPES-1]
        grid = grid.clamp(0, N_UNIT_TYPES - 1).long()

        return grid   # (B, 64)

    def forward(self, obs, state):
        """
        Args:
            obs:   (B, H, W, 27)
            state: list of (h, c) tuples, one per LSTM layer

        Returns:
            enemy_emb: (B, d_e)
            state:     updated list of (h, c) tuples
        """
        # 1. Build enemy grid — (B, 64) integer type IDs
        grid = self.extract_enemy_grid(obs)

        # 2. Embed each cell's type ID — (B, 64, d_e)
        cell_embeddings = self.type_embedding(grid)

        # 3. Flatten to (B, 64 * d_e) — preserves which unit is in which cell
        grid_emb = cell_embeddings.view(cell_embeddings.size(0), -1)  # (B, 64*d_e)

        # 4. Project down to d_e
        x = self.input_proj(grid_emb)    # (B, d_e)

        # 5. Pass through each LSTM layer in sequence
        # Each layer receives the previous layer's hidden state as input
        new_state = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = state[i]
            h, c = lstm_cell(x, (h, c))
            new_state.append((h, c))
            x = h   # next layer takes this layer's hidden state as input

        # 6. Project final layer's hidden state to enemy embedding
        enemy_emb = self.out_proj(x)    # (B, d_e)

        return enemy_emb, new_state


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, c = 4, 8, 8, 10
    d_e        = 64

    encoder = EnemyEncoder(d_e=d_e, map_size=H * W, num_layers=2)
    state   = encoder.init_state(B)
    obs     = torch.zeros(B, H, W, 27)

    # Enemy worker at (2,3): owner=player2, unit type=worker (plane 17)
    obs[:, 2, 3, ENEMY_OWNER_CHANNEL] = 1
    obs[:, 2, 3, 17]                  = 1
    # Enemy heavy at (4,5)
    obs[:, 4, 5, ENEMY_OWNER_CHANNEL] = 1
    obs[:, 4, 5, 19]                  = 1   # heavy

    for step in range(c):
        enemy_emb, state = encoder(obs, state)

    assert enemy_emb.shape == (B, d_e)
    assert not enemy_emb.isnan().any()
    print(f"enemy_emb shape: {enemy_emb.shape}")
    print(f"EnemyEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("EnemyEncoder OK.")
