import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # allow MPS -> CPU fallback for unsupported ops

import random
import time

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.utils.data import Dataset, DataLoader


# ================================================================
# 0. Utilities
# ================================================================
def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================
# 1. Breakout environment (84x84, 4 stacked frames)
# ================================================================
def make_breakout_env():
    """Create an ALE/Breakout-v5 environment with standard preprocessing."""
    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0,
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,  # output: (84, 84)
        scale_obs=False,
    )

    env = FrameStackObservation(
        env,
        stack_size=4,
        padding_type="reset",
    )

    obs, _ = env.reset()
    print("Env obs shape after wrappers:", np.array(obs).shape)
    print("Num actions:", env.action_space.n)
    print("Action meanings:", env.unwrapped.get_action_meanings())
    return env


def obs_to_chw(obs):
    """
    Convert observation to (4, 84, 84) uint8 in CHW format.

    Supports both (4, 84, 84) and (84, 84, 4) input shapes.
    """
    obs = np.array(obs)
    if obs.ndim == 3:
        if obs.shape[0] == 4:
            # (4, 84, 84)
            return obs
        elif obs.shape[-1] == 4:
            # (84, 84, 4)
            return np.transpose(obs, (2, 0, 1))

    raise ValueError(f"Unexpected obs shape: {obs.shape}")


# ================================================================
# 2. CNN encoder + causal Transformer + Decision Transformer (discrete)
# ================================================================
class CNNStateEncoder(nn.Module):
    """Encode stacked Atari frames: (B*K, 4, 84, 84) -> (B*K, d_model)."""

    def __init__(self, in_channels=4, d_model=256):
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

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CausalTransformer(nn.Module):
    """Standard TransformerEncoder used in a causal manner."""

    def __init__(self, d_model=256, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x:                (B, L, d)
        attn_mask:        (L, L) with True = block, False = allow
        key_padding_mask: (B, L) with True = pad, False = keep
        """
        # On MPS, src_key_padding_mask can trigger nested tensor ops that
        # are not implemented, so we omit it there.
        if x.device.type == "mps":
            return self.encoder(x, mask=attn_mask)
        else:
            return self.encoder(
                x,
                mask=attn_mask,
                src_key_padding_mask=key_padding_mask,
            )


def build_causal_mask(L, device):
    """Upper-triangular mask: True blocks attention to future positions."""
    return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()


class DecisionTransformerDiscrete(nn.Module):
    """
    Decision Transformer for discrete action spaces.

    Per time step we use three tokens:
        [RTG_t, state_t, action_{t-1}]
    and predict action_t at each action token.
    """

    def __init__(
        self,
        num_actions,
        d_model=256,
        max_timestep=4096,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        rtg_scale=1000.0,
        in_channels=4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.d_model = d_model
        self.rtg_scale = rtg_scale

        self.state_encoder = CNNStateEncoder(in_channels=in_channels, d_model=d_model)

        # Action embedding (+1 slot for START token that encodes a_{-1})
        self.embed_action = nn.Embedding(num_actions + 1, d_model)
        # RTG, timestep, and token-type embeddings
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_time = nn.Embedding(max_timestep, d_model)
        self.embed_tokentype = nn.Embedding(3, d_model)  # 0=rtg, 1=state, 2=action

        self.transformer = CausalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.action_head = nn.Linear(d_model, num_actions)
        self.layer_norm = nn.LayerNorm(d_model)

    def _interleave(self, rtg_emb, state_emb, action_emb):
        """Interleave as [rtg, state, action] for each time step."""
        B, K, D = rtg_emb.shape
        seq = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, K, 3, D)
        seq = seq.view(B, K * 3, D)
        return seq

    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        """
        Args:
            rtg:       (B, K, 1)
            states:    (B, K, 4, 84, 84)
            actions:   (B, K)    previous actions a_{t-1}, with -1 at t=0
            timesteps: (B, K)
        Returns:
            logits: (B, K, num_actions)
        """
        B, K = actions.shape
        device = actions.device

        timesteps = timesteps.long().clamp(0, self.embed_time.num_embeddings - 1)
        t_emb = self.embed_time(timesteps)  # (B, K, D)

        # RTG tokens
        rtg = rtg.to(torch.float32)
        rtg_in = rtg / self.rtg_scale
        rtg_emb = (
            self.embed_rtg(rtg_in)
            + t_emb
            + self.embed_tokentype(torch.zeros_like(actions))  # token type 0
        )

        # State tokens
        s = states.to(torch.float32).view(B * K, *states.shape[2:])
        s_emb = self.state_encoder(s).view(B, K, self.d_model)
        s_emb = (
            s_emb
            + t_emb
            + self.embed_tokentype(torch.ones_like(actions))  # token type 1
        )

        # Action tokens for a_{t-1}
        a_ids = actions.clone()
        a_ids = torch.where(a_ids < 0, torch.full_like(a_ids, self.num_actions), a_ids)
        a_emb = (
            self.embed_action(a_ids)
            + t_emb
            + self.embed_tokentype(torch.full_like(actions, 2))  # token type 2
        )

        x = self._interleave(rtg_emb, s_emb, a_emb)  # (B, 3K, D)
        x = self.layer_norm(x)

        L = x.size(1)
        causal_mask = build_causal_mask(L, device)

        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: True = keep -> key_padding_mask: True = pad
            key_padding_mask = ~attention_mask

        out = self.transformer(
            x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )

        # Extract hidden state at action token positions: 2,5,8,... => 3*i + 2
        idx = torch.arange(K, device=device) * 3 + 2
        idx = idx.unsqueeze(0).expand(B, -1)
        a_hidden = out.gather(1, idx.unsqueeze(-1).expand(B, K, self.d_model))

        logits = self.action_head(a_hidden)  # (B, K, num_actions)
        return logits


# ================================================================
# 3. Loss, RTG, and action sampling
# ================================================================
def step_mask_to_token_mask(step_mask):
    """Expand per-step mask (B, K) to token mask (B, 3K)."""
    B, K = step_mask.shape
    return step_mask.unsqueeze(-1).expand(B, K, 3).reshape(B, 3 * K)


def compute_dt_loss(model, batch, ignore_index=-100):
    """Cross-entropy loss over action tokens, masking padded steps."""
    device = next(model.parameters()).device

    rtg = batch["rtg"].to(device)
    states = batch["states"].to(device)
    actions_in = batch["actions_in"].to(device)
    targets = batch["actions_target"].to(device)
    timesteps = batch["timesteps"].to(device)
    step_mask = batch["step_mask"].to(device)

    token_mask = step_mask_to_token_mask(step_mask)

    logits = model(rtg, states, actions_in, timesteps, attention_mask=token_mask)

    B, K, A = logits.shape
    logits_flat = logits.view(B * K, A)
    targets_flat = targets.view(B * K)

    targets_flat_masked = targets_flat.masked_fill(~step_mask.view(-1), ignore_index)

    loss = F.cross_entropy(logits_flat, targets_flat_masked, ignore_index=ignore_index)
    return loss, logits


def compute_rtg(rewards, gamma=1.0):
    """Compute return-to-go (optionally discounted)."""
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    T = rewards.shape[0]
    rtg = torch.zeros(T, dtype=torch.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma * running
        rtg[t] = running
    return rtg.unsqueeze(-1)


@torch.no_grad()
def evaluate_dt_discrete(env, model, device, K=20, target_return=2.0):
    """Run one evaluation episode in Breakout using the Decision Transformer."""
    model.eval()
    obs, _ = env.reset()
    obs = obs_to_chw(obs)

    states_seq = []
    actions_seq = []
    rtg_seq = []
    t_seq = []

    running_rtg = target_return
    episode_return = 0.0
    done = False
    t = 0

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
            actions_in[1:L] = a_np[: L - 1]
            actions_in[0] = -1

        states_t = torch.from_numpy(states).unsqueeze(0).to(device)
        rtg_t = torch.from_numpy(rtg).unsqueeze(0).to(device)
        a_in_t = torch.from_numpy(actions_in).unsqueeze(0).to(device)
        ts_t = torch.from_numpy(timesteps).unsqueeze(0).to(device)

        logits = model(rtg_t, states_t, a_in_t, ts_t)
        action = int(torch.argmax(logits[0, L - 1]).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward
        running_rtg -= reward

        obs = obs_to_chw(next_obs)
        actions_seq[-1] = action
        t += 1

    return episode_return


# ================================================================
# 4. Trajectories and dataset (collected from a random policy)
# ================================================================
def collect_random_trajectories(env, num_episodes=20, max_steps=500):
    """Collect offline trajectories with a random policy."""
    trajectories = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = obs_to_chw(obs)

        states = []
        actions = []
        rewards = []

        done = False
        t = 0
        while not done and t < max_steps:
            states.append(obs)
            action = env.action_space.sample()
            actions.append(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

            obs = obs_to_chw(next_obs)
            t += 1

        traj = {
            "states": np.array(states, dtype=np.uint8),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
        }
        trajectories.append(traj)
        print(f"Collected episode {ep+1}/{num_episodes}, length={len(actions)}")

    print("Total trajectories:", len(trajectories))
    return trajectories


class DTTrajectoryDataset(Dataset):
    """
    Dataset of short subsequences sampled from a list of trajectories.

    Each item is a window of at most K steps, padded to length K.
    """

    def __init__(self, trajectories, K=20, gamma=1.0):
        self.trajectories = trajectories
        self.K = K
        self.gamma = gamma

    def __len__(self):
        # Use a large nominal length; sampling chooses a random trajectory each time.
        return 100000

    def __getitem__(self, idx):
        traj = random.choice(self.trajectories)
        T = len(traj["actions"])

        start = np.random.randint(0, T)
        end = min(T, start + self.K)
        length = end - start

        states = traj["states"][start:end]
        actions = traj["actions"][start:end]
        rewards = traj["rewards"][start:end]

        rtg_full = compute_rtg(rewards, gamma=self.gamma)

        K = self.K
        states_padded = np.zeros((K, 4, 84, 84), dtype=np.uint8)
        actions_in = -np.ones((K,), dtype=np.int64)
        actions_tgt = -np.ones((K,), dtype=np.int64)
        rtg_padded = np.zeros((K, 1), dtype=np.float32)
        timesteps = np.zeros((K,), dtype=np.int64)
        step_mask = np.zeros((K,), dtype=bool)

        states_padded[:length] = states
        actions_tgt[:length] = actions
        rtg_padded[:length] = rtg_full.numpy()
        timesteps[:length] = np.arange(start, start + length)
        step_mask[:length] = True

        if length > 0:
            actions_in[1:length] = actions[: length - 1]
            actions_in[0] = -1

        batch = {
            "states": torch.from_numpy(states_padded),
            "actions_in": torch.from_numpy(actions_in),
            "actions_target": torch.from_numpy(actions_tgt),
            "rtg": torch.from_numpy(rtg_padded),
            "timesteps": torch.from_numpy(timesteps),
            "step_mask": torch.from_numpy(step_mask),
        }
        return batch


# ================================================================
# 5. Training, evaluation, and plots
# ================================================================
def main():
    set_seed(0)

    env = make_breakout_env()
    num_actions = env.action_space.n

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Offline data collection
    trajectories = collect_random_trajectories(env, num_episodes=20, max_steps=500)

    K = 20
    dataset = DTTrajectoryDataset(trajectories, K=K, gamma=1.0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # Model
    d_model = 256
    rtg_scale = 1000.0
    model = DecisionTransformerDiscrete(
        num_actions=num_actions,
        d_model=d_model,
        rtg_scale=rtg_scale,
        in_channels=4,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Training loop with periodic evaluation
    num_updates = 1000
    eval_interval = 100
    eval_episodes = 5

    train_losses = []
    eval_points = []
    eval_returns = []

    start_time = time.time()
    data_iter = iter(loader)

    for update in range(num_updates):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss, _ = compute_dt_loss(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if update % 50 == 0:
            print(f"Update {update}/{num_updates} | loss={loss.item():.4f}")

        if (update + 1) % eval_interval == 0:
            returns_this_ckpt = []
            for ep in range(eval_episodes):
                R = evaluate_dt_discrete(env, model, device, K=K, target_return=2.0)
                returns_this_ckpt.append(R)
                print(
                    f"[DT Breakout] Eval episode {ep+1}/{eval_episodes} "
                    f"after {update+1} updates: return={R:.1f}"
                )
            avg_R = float(np.mean(returns_this_ckpt))
            eval_points.append(update + 1)
            eval_returns.append(avg_R)
            print(
                f"[DT Breakout] >>> After {update+1} updates: "
                f"avg return over {eval_episodes} eps = {avg_R:.2f}"
            )

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # Plot 1: training loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel("Update")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("Decision Transformer Training Loss on Breakout")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: evaluation return vs training updates
    plt.figure(figsize=(6, 4))
    plt.plot(eval_points, eval_returns, marker="o")
    plt.xlabel("Training updates")
    plt.ylabel("Average return")
    plt.title("Decision Transformer on Breakout\nEval return vs training updates")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
