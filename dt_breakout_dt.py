import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Mac/MPS workaround
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Env ---

def make_breakout_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False
    )
    env = FrameStackObservation(env, stack_size=4, padding_type="reset")
    return env


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    # Handle (H, W, C) -> (C, H, W) if needed
    obs = np.array(obs)
    if obs.ndim == 3 and obs.shape[-1] == 4:
        return np.transpose(obs, (2, 0, 1))
    return obs

# --- Models ---

class CNNStateEncoder(nn.Module):
    def __init__(self, in_channels=4, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, d_model)

    def forward(self, x):
        x = x.float() / 255.0 # normalize on the fly
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



class CausalTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=dropout, batch_first=True
            ),
            num_layers=n_layers
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # MPS doesn't support nested tensor ops for src_key_padding_mask yet
        if x.device.type == "mps":
            return self.encoder(x, mask=attn_mask)
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)


class DecisionTransformerDiscrete(nn.Module):
    def __init__(self, num_actions, d_model=256, max_timestep=4096, 
                 n_layers=4, n_heads=4, dropout=0.1, rtg_scale=1000.0):
        super().__init__()
        self.num_actions = num_actions
        self.d_model = d_model
        self.rtg_scale = rtg_scale

        self.state_encoder = CNNStateEncoder(in_channels=4, d_model=d_model)
        
        # +1 for start token (-1)
        self.embed_action = nn.Embedding(num_actions + 1, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_time = nn.Embedding(max_timestep, d_model)
        self.embed_tokentype = nn.Embedding(3, d_model)

        self.transformer = CausalTransformer(d_model, n_layers, n_heads, dropout)
        self.action_head = nn.Linear(d_model, num_actions)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, rtg, states, actions, timesteps, attention_mask=None):
        B, K = actions.shape
        device = actions.device

        # Embeddings
        t_emb = self.embed_time(timesteps.long().clamp(0, self.embed_time.num_embeddings - 1))
        
        # 1. RTG
        rtg_emb = self.embed_rtg(rtg.float() / self.rtg_scale) + t_emb + \
                  self.embed_tokentype(torch.zeros_like(actions))

        # 2. State
        s_flat = states.float().view(B * K, *states.shape[2:])
        s_emb = self.state_encoder(s_flat).view(B, K, self.d_model)
        s_emb = s_emb + t_emb + self.embed_tokentype(torch.ones_like(actions))

        # 3. Action (Handle start token -1)
        a_ids = actions.clone()
        a_ids[a_ids < 0] = self.num_actions
        a_emb = self.embed_action(a_ids) + t_emb + self.embed_tokentype(torch.full_like(actions, 2))

        # Stack: (B, K, 3, D) -> (B, 3*K, D)
        x = torch.stack([rtg_emb, s_emb, a_emb], dim=2).view(B, K * 3, self.d_model)
        x = self.layer_norm(x)

        # Causal Mask
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=device), diagonal=1).bool()
        
        kp_mask = None
        if attention_mask is not None:
            kp_mask = ~attention_mask # True = pad

        out = self.transformer(x, attn_mask=mask, key_padding_mask=kp_mask)
        idx = torch.arange(K, device=device) * 3 + 1 # Use state embedding to predict action
        idx = idx.unsqueeze(0).unsqueeze(-1).expand(B, K, self.d_model)
        
        s_preds = out.gather(1, idx)
        return self.action_head(s_preds)




# --- Utils & Training ---

def compute_dt_loss(model, batch):
    device = next(model.parameters()).device
    
    # Expand step mask (B, K) -> token mask (B, 3K)
    step_mask = batch["step_mask"].to(device)
    B, K = step_mask.shape
    token_mask = step_mask.unsqueeze(-1).expand(B, K, 3).reshape(B, 3 * K)

    logits = model(
        rtg=batch["rtg"].to(device),
        states=batch["states"].to(device),
        actions=batch["actions_in"].to(device),
        timesteps=batch["timesteps"].to(device),
        attention_mask=token_mask
    )

    targets = batch["actions_target"].to(device).view(-1)
    logits = logits.view(-1, logits.size(-1))
    
    # Mask out padding
    targets = targets.masked_fill(~step_mask.view(-1), -100)
    return F.cross_entropy(logits, targets, ignore_index=-100), logits


def compute_rtg(rewards, gamma=1.0):
    rewards = torch.tensor(rewards, dtype=torch.float32) if not torch.is_tensor(rewards) else rewards
    rtg = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg.unsqueeze(-1)



# --- Data ---

def collect_dummy_data(env, episodes=20, max_steps=500):
    # Just for testing pipeline, uses random policy
    data = []
    print(f"Collecting {episodes} dummy trajectories...")
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = obs_to_chw(obs)
        s, a, r = [], [], []
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, done, trunc, _ = env.step(action)
            
            s.append(obs)
            a.append(action)
            r.append(reward)
            obs = obs_to_chw(next_obs)
            if done or trunc: break
            
        data.append({
            "states": np.array(s, dtype=np.uint8),
            "actions": np.array(a, dtype=np.int64),
            "rewards": np.array(r, dtype=np.float32),
        })
    return data


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, context_len=20, gamma=1.0):
        self.data = trajectories
        self.K = context_len
        self.gamma = gamma

    def __len__(self):
        return 50000 # Hack for infinite-ish sampling

    def __getitem__(self, idx):
        traj = random.choice(self.data)
        T = len(traj["actions"])
        
        # Random window
        si = random.randint(0, T - 1)
        ei = min(si + self.K, T)
        real_len = ei - si

        # Slice
        s = traj["states"][si:ei]
        a = traj["actions"][si:ei]
        r = traj["rewards"][si:ei]
        rtg = compute_rtg(r, self.gamma)

        # Padding
        s_pad = np.zeros((self.K, 4, 84, 84), dtype=np.uint8)
        a_in = -np.ones(self.K, dtype=np.int64)
        a_tgt = -np.ones(self.K, dtype=np.int64)
        rtg_pad = np.zeros((self.K, 1), dtype=np.float32)
        t_pad = np.zeros(self.K, dtype=np.int64)
        mask = np.zeros(self.K, dtype=bool)

        s_pad[:real_len] = s
        a_tgt[:real_len] = a
        rtg_pad[:real_len] = rtg
        t_pad[:real_len] = np.arange(si, ei)
        mask[:real_len] = True

        # Input action is shifted: a_in[t] = a[t-1]
        if real_len > 0:
            a_in[0] = -1 # Start token
            if real_len > 1:
                a_in[1:real_len] = a[:real_len-1]

        return {
            "states": torch.from_numpy(s_pad),
            "actions_in": torch.from_numpy(a_in),
            "actions_target": torch.from_numpy(a_tgt),
            "rtg": torch.from_numpy(rtg_pad),
            "timesteps": torch.from_numpy(t_pad),
            "step_mask": torch.from_numpy(mask)
        }


@torch.no_grad()
def eval_rollout(env, model, device, context_len=20, target_rtg=2.0):
    model.eval()
    obs, _ = env.reset()
    obs = obs_to_chw(obs)
    
    s_hist, a_hist, rtg_hist, t_hist = [], [-1], [target_rtg], [0]
    total_reward = 0
    
    done = False

    while not done:
        s_hist.append(obs)
        
        # Prepare context window
        s_in = np.array(s_hist[-context_len:], dtype=np.uint8)
        a_in = np.array(a_hist[-context_len:], dtype=np.int64)
        rtg_in = np.array(rtg_hist[-context_len:], dtype=np.float32).reshape(-1, 1)
        t_in = np.array(t_hist[-context_len:], dtype=np.int64)
        
        # Pad if needed (left padding usually for inference, but simple slicing works if model handles pos emb correctly)
        # Here we just use what we have, model expects [B, K, ...]
        cur_len = len(s_in)
        pad_len = context_len - cur_len
        
        if pad_len > 0:
            # Right pad for simplicity with existing model logic
            s_final = np.pad(s_in, ((0, pad_len), (0,0), (0,0), (0,0)))
            a_final = np.pad(a_in, (0, pad_len), constant_values=-1)
            rtg_final = np.pad(rtg_in, ((0, pad_len), (0,0)))
            t_final = np.pad(t_in, (0, pad_len))
        else:
            s_final, a_final, rtg_final, t_final = s_in, a_in, rtg_in, t_in

        # Forward
        logits = model(
            rtg=torch.from_numpy(rtg_final).unsqueeze(0).to(device),
            states=torch.from_numpy(s_final).unsqueeze(0).to(device),
            actions=torch.from_numpy(a_final).unsqueeze(0).to(device),
            timesteps=torch.from_numpy(t_final).unsqueeze(0).to(device)
        )
        
        # Pred at last valid step
        action = int(torch.argmax(logits[0, cur_len-1]).item())
        
        obs, r, term, trunc, _ = env.step(action)
        obs = obs_to_chw(obs)
        done = term or trunc
        total_reward += r
        
        a_hist[-1] = action # replace dummy/prev with chosen
        a_hist.append(-1)   # next step dummy

        rtg_hist.append(rtg_hist[-1] - r)
        t_hist.append(t_hist[-1] + 1)
        
    return total_reward



def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Params
    BATCH_SIZE = 32
    CONTEXT_LEN = 20
    UPDATES = 1000
    LR = 1e-4

    env = make_breakout_env()
    
    # 1. Collect Data
    data = collect_dummy_data(env, episodes=20, max_steps=500)
    dataset = TrajectoryDataset(data, context_len=CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Init Model
    model = DecisionTransformerDiscrete(
        num_actions=env.action_space.n,
        d_model=256,
        rtg_scale=100.0 # Adjusted scale
    ).to(device)
    
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 3. Train
    print("Starting training...")
    losses = []
    
    iter_loader = iter(loader)
    for step in range(UPDATES):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
            
        loss, _ = compute_dt_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        if step > 0 and step % 500 == 0:
            ret = eval_rollout(env, model, device, CONTEXT_LEN)
            print(f"Step {step} | Eval Return: {ret}")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    main()