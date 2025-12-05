import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.utils.data import Dataset, DataLoader


# ---------------- Env helper ----------------

def make_breakout_env(seed: int = 0):
    """
    Create an ALE/Breakout-v5 environment with standard Atari preprocessing:
    grayscale observations, 84x84 resolution, frame_skip=4, and 4-frame stacking.
    """
    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
        frameskip=1,                 # frame_skip is handled by AtariPreprocessing
        repeat_action_probability=0,
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,             # still 0–255; later divided by 255 in the encoder
    )

    env = FrameStackObservation(
        env,
        stack_size=4,
        padding_type="reset",
    )

    env.reset(seed=seed)
    return env


# ------------- 1) CNN state encoder -------------

class CNNStateEncoder(nn.Module):
    """
    Encode stacked Atari frames.

    Input:  (B*K, 4, 84, 84)
    Output: (B*K, d_model)
    """
    def __init__(self, in_channels: int = 4, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0          # (B*K, 4, 84, 84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ------------- 2) Causal Transformer -------------

class CausalTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)


def build_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular mask for causal attention.

    True  = block future position,
    False = allow attention.
    """
    return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()


# ------------- 3) Decision Transformer (discrete) -------------

class DecisionTransformerDiscrete(nn.Module):
    """
    Decision Transformer for discrete action spaces.

    Each time step is represented by three tokens:
        [rtg_t, state_t, action_{t-1}]

    The model predicts action_t at the action token positions.
    """
    def __init__(
        self,
        num_actions: int,
        d_model: int = 256,
        max_timestep: int = 4096,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        rtg_scale: float = 1000.0,
        pixel_inputs: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.d_model = d_model
        self.pixel_inputs = pixel_inputs
        self.rtg_scale = rtg_scale

        if pixel_inputs:
            self.state_encoder = CNNStateEncoder(in_channels=in_channels, d_model=d_model)
        else:
            # use a linear projection for vector observations
            self.state_proj = nn.Linear(in_channels, d_model)

        # +1 for the START token used at a_{-1}
        self.embed_action = nn.Embedding(num_actions + 1, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_time = nn.Embedding(max_timestep, d_model)
        # token type: 0 = rtg, 1 = state, 2 = action
        self.embed_tokentype = nn.Embedding(3, d_model)

        self.transformer = CausalTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )
        self.action_head = nn.Linear(d_model, num_actions)
        self.layer_norm = nn.LayerNorm(d_model)

    def _interleave(
        self,
        rtg_emb: torch.Tensor,
        state_emb: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        B, K, D = rtg_emb.shape
        seq = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, K, 3, D)
        seq = seq.view(B, 3 * K, D)
        return seq

    def forward(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            rtg:       (B, K, 1)
            states:    (B, K, 4, 84, 84)
            actions:   (B, K), previous actions a_{t-1}, with -1 at t=0 (START)
            timesteps: (B, K)
            attention_mask: (B, 3K) bool, True for valid tokens

        Returns:
            logits: (B, K, num_actions)
        """
        B, K = actions.shape
        device = actions.device

        t_embed = self.embed_time(timesteps)           # (B, K, D)

        # 1) RTG tokens
        rtg_in = rtg / self.rtg_scale
        rtg_emb = (
            self.embed_rtg(rtg_in)
            + t_embed
            + self.embed_tokentype(torch.zeros_like(actions))  # token type 0
        )

        # 2) State tokens
        if self.pixel_inputs:
            s = states.view(B * K, *states.shape[2:])  # (B*K, 4, 84, 84)
            s_emb = self.state_encoder(s).view(B, K, self.d_model)
        else:
            s_emb = self.state_proj(states)
        s_emb = (
            s_emb
            + t_embed
            + self.embed_tokentype(torch.ones_like(actions))   # token type 1
        )

        # 3) Action tokens
        a_ids = actions.clone()
        a_ids = torch.where(
            a_ids < 0,
            torch.full_like(a_ids, self.num_actions),
            a_ids,
        )
        a_emb = (
            self.embed_action(a_ids)
            + t_embed
            + self.embed_tokentype(torch.full_like(actions, 2))  # token type 2
        )

        x = self._interleave(rtg_emb, s_emb, a_emb)    # (B, 3K, D)
        x = self.layer_norm(x)

        L = x.size(1)
        causal_mask = build_causal_mask(L, device)

        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: True = keep; key_padding_mask: True = pad
            key_padding_mask = ~attention_mask

        out = self.transformer(
            x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )

        # action token positions: 2, 5, 8, ... = 3*i + 2
        idx = torch.arange(K, device=device) * 3 + 2
        idx = idx.unsqueeze(0).expand(B, -1)  # (B, K)
        a_hidden = out.gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(B, K, self.d_model),
        )

        logits = self.action_head(a_hidden)            # (B, K, num_actions)
        return logits


# ------------- 4) Loss / RTG utils -------------

def step_mask_to_token_mask(step_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand per-step mask (B, K) to per-token mask (B, 3K).
    """
    B, K = step_mask.shape
    return step_mask.unsqueeze(-1).expand(B, K, 3).reshape(B, 3 * K)


def compute_rtg(rewards, gamma: float = 1.0) -> torch.Tensor:
    """
    Compute return-to-go.

    rewards: (T,)
    returns: (T, 1) where rtg[t] = Σ_{t'≥t} gamma^{t'-t} * r[t']
    """
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    T = rewards.shape[0]
    rtg = torch.zeros(T, dtype=torch.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma * running
        rtg[t] = running
    return rtg.unsqueeze(-1)


def compute_dt_loss(model: nn.Module, batch: dict, ignore_index: int = -100):
    """
    Cross-entropy loss over action tokens, ignoring padded steps.
    """
    rtg        = batch["rtg"]          # (B, K, 1)
    states     = batch["states"]       # (B, K, 4, 84, 84)
    actions_in = batch["actions_in"]   # (B, K)
    targets    = batch["actions_target"]
    timesteps  = batch["timesteps"]    # (B, K)
    step_mask  = batch["step_mask"]    # (B, K)

    token_mask = step_mask_to_token_mask(step_mask)
    logits = model(rtg, states, actions_in, timesteps, attention_mask=token_mask)

    B, K, A = logits.shape
    logits_flat = logits.reshape(B * K, A)
    targets_flat = targets.reshape(B * K)

    targets_masked = targets_flat.masked_fill(~step_mask.view(-1), ignore_index)
    loss = F.cross_entropy(logits_flat, targets_masked, ignore_index=ignore_index)
    return loss, logits


# ------------- 5) Offline dataset for DT -------------

def collect_random_trajectories(
    env,
    num_episodes: int = 20,
    max_steps: int = 400,
):
    """
    Collect a small offline dataset in Breakout using a random policy.

    Returns:
        trajectories: list of dicts with keys:
            "states":  (T, 4, 84, 84) uint8
            "actions": (T,) int64
            "rewards": (T,) float32
    """
    trajectories = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = np.array(obs)   # (4, 84, 84)

        states = []
        actions = []
        rewards = []

        for t in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = np.array(next_obs)

            if terminated or truncated:
                break

        if len(actions) == 0:
            continue

        traj = {
            "states":  np.stack(states, axis=0).astype(np.uint8),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
        }
        trajectories.append(traj)
        print(f"Collected episode {ep + 1}/{num_episodes}, length={len(actions)}")

    return trajectories


class DTTrajectoryDataset(Dataset):
    """
    Dataset that samples short windows of length K from a list of trajectories.
    """
    def __init__(self, trajectories, K: int, gamma: float = 1.0):
        self.trajectories = trajectories
        self.K = K
        self.gamma = gamma
        self.lengths = [len(traj["actions"]) for traj in trajectories]

    def __len__(self):
        # Rough estimate; we randomly choose trajectories in __getitem__.
        return sum(self.lengths)

    def __getitem__(self, idx):
        traj = self.trajectories[np.random.randint(len(self.trajectories))]
        T = len(traj["actions"])

        start = np.random.randint(0, T)
        end = min(T, start + self.K)
        L = end - start

        states  = traj["states"][start:end]      # (L, 4, 84, 84)
        actions = traj["actions"][start:end]
        rewards = traj["rewards"][start:end]

        rtg_full = compute_rtg(rewards, gamma=self.gamma)  # (L, 1)

        K = self.K
        states_padded = np.zeros((K, 4, 84, 84), dtype=np.uint8)
        actions_in    = -np.ones((K,), dtype=np.int64)   # a_{t-1}
        actions_tgt   = -np.ones((K,), dtype=np.int64)
        rtg_padded    = np.zeros((K, 1), dtype=np.float32)
        timesteps     = np.zeros((K,), dtype=np.int64)
        step_mask     = np.zeros((K,), dtype=bool)

        states_padded[:L] = states
        actions_tgt[:L]   = actions
        rtg_padded[:L]    = rtg_full.numpy()
        timesteps[:L]     = np.arange(start, start + L)
        step_mask[:L]     = True

        if L > 0:
            actions_in[1:L] = actions[:L - 1]

        batch = {
            "states":         torch.from_numpy(states_padded),
            "actions_in":     torch.from_numpy(actions_in),
            "actions_target": torch.from_numpy(actions_tgt),
            "rtg":            torch.from_numpy(rtg_padded),
            "timesteps":      torch.from_numpy(timesteps),
            "step_mask":      torch.from_numpy(step_mask),
        }
        return batch


# ------------- 6) Evaluation -------------

@torch.no_grad()
def evaluate_dt_discrete(env, model, K: int = 20, target_return: float = 50.0):
    """
    Run a single online rollout in Breakout using the Decision Transformer.
    Returns the episode return.
    """
    model.eval()
    device = next(model.parameters()).device

    obs, info = env.reset()
    obs = np.array(obs)          # (4, 84, 84)
    episode_return = 0.0
    done = False
    t = 0

    states_seq = []
    actions_seq = []
    rtg_seq = []
    t_seq = []
    running_rtg = target_return

    while not done:
        states_seq.append(obs)
        rtg_seq.append([running_rtg])
        t_seq.append(t)
        if len(actions_seq) == 0:
            actions_seq.append(-1)
        else:
            actions_seq.append(actions_seq[-1])

        s_np = np.array(states_seq[-K:], dtype=np.uint8)
        a_np = np.array(actions_seq[-K:], dtype=np.int64)
        rtg_np = np.array(rtg_seq[-K:], dtype=np.float32)
        t_np = np.array(t_seq[-K:], dtype=np.int64)

        L = s_np.shape[0]
        states = np.zeros((K, 4, 84, 84), dtype=np.uint8)
        actions_in = -np.ones((K,), dtype=np.int64)
        rtg = np.zeros((K, 1), dtype=np.float32)
        timesteps = np.zeros((K,), dtype=np.int64)

        states[:L] = s_np
        rtg[:L] = rtg_np
        timesteps[:L] = t_np
        if L > 0:
            actions_in[1:L] = a_np[:L - 1]
            actions_in[0] = -1

        states_t = torch.from_numpy(states).unsqueeze(0).to(device)
        rtg_t = torch.from_numpy(rtg).unsqueeze(0).to(device)
        a_in_t = torch.from_numpy(actions_in).unsqueeze(0).to(device)
        ts_t = torch.from_numpy(timesteps).unsqueeze(0).to(device)

        logits = model(rtg_t, states_t, a_in_t, ts_t)     # (1, K, A)
        action = int(torch.argmax(logits[0, L - 1]).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        running_rtg -= reward

        obs = np.array(next_obs)
        actions_seq[-1] = action
        t += 1

    return episode_return


# ------------- 7) Main train & plot -------------

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    train_env = make_breakout_env(seed=0)
    num_actions = train_env.action_space.n
    print(
        "Breakout actions:", num_actions,
        train_env.unwrapped.get_action_meanings(),
    )

    # 1) Collect offline data
    trajectories = collect_random_trajectories(
        train_env,
        num_episodes=20,
        max_steps=400,
    )
    print("Total trajectories:", len(trajectories))

    K = 20
    d_model = 256
    rtg_scale = 100.0

    model = DecisionTransformerDiscrete(
        num_actions=num_actions,
        d_model=d_model,
        rtg_scale=rtg_scale,
    ).to(device)

    dataset = DTTrajectoryDataset(trajectories, K=K, gamma=1.0)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # 2) Training loop
    model.train()
    max_updates = 1000
    step = 0
    start_time = time.time()

    while step < max_updates:
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            loss, _ = compute_dt_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Update {step}/{max_updates} | loss={loss.item():.4f}")
            step += 1
            if step >= max_updates:
                break

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 3) Evaluation and plot
    eval_env = make_breakout_env(seed=123)
    num_eval_eps = 50
    returns = []
    for ep in range(num_eval_eps):
        R = evaluate_dt_discrete(eval_env, model, K=K, target_return=50.0)
        returns.append(R)
        print(f"[DT Breakout] Eval episode {ep + 1}/{num_eval_eps}: return={R}")

    plt.figure(figsize=(6, 5))
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Decision Transformer on Breakout")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
