import time
import numpy as np
import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.utils.data import Dataset, DataLoader

# ---------------- 1. Environment Helper ----------------

def make_breakout_env(seed: int = 0):
    """
    Create an ALE/Breakout-v5 environment with standard Atari preprocessing.
    """
    if "ALE/Breakout-v5" not in gym.registry:
        gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
        frameskip=1,                 # Handled by AtariPreprocessing
        repeat_action_probability=0,
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,             # Keep as uint8 (0-255) to save memory
    )

    env = FrameStackObservation(
        env,
        stack_size=4,
        padding_type="reset",
    )

    env.reset(seed=seed)
    return env


# ---------------- 2. Model Architecture ----------------

class CNNStateEncoder(nn.Module):
    """
    Encodes 4-stacked Atari frames into a vector.
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
        # Normalization happens here on the fly
        x = x.float() / 255.0  
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DecisionTransformerDiscrete(nn.Module):
    """
    Decision Transformer for Discrete Actions (Atari).
    Sequence: [Return, State, Action]
    """
    def __init__(
        self,
        num_actions: int,
        d_model: int = 256,
        max_timestep: int = 4096,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        rtg_scale: float = 100.0,
        in_channels: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.d_model = d_model
        self.rtg_scale = rtg_scale

        # 1. Embeddings
        self.state_encoder = CNNStateEncoder(in_channels=in_channels, d_model=d_model)
        
        # Action embedding: num_actions + 1 (for the START token -1)
        self.embed_action = nn.Embedding(num_actions + 1, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_time = nn.Embedding(max_timestep, d_model)
        
        # Token type embeddings (RTG=0, State=1, Action=2)
        self.embed_tokentype = nn.Embedding(3, d_model)

        # 2. Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True, # Pre-norm is generally more stable
            ),
            num_layers=n_layers,
        )

        # 3. Prediction Head
        self.action_head = nn.Linear(d_model, num_actions)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        B, K = actions.shape
        device = actions.device

        # --- Embeddings ---
        t_embed = self.embed_time(timesteps)

        # A. RTG
        rtg_emb = self.embed_rtg(rtg / self.rtg_scale) + t_embed + self.embed_tokentype(torch.zeros_like(actions))

        # B. State (Combine Batch and Time dims for CNN)
        s_flat = states.view(B * K, *states.shape[2:]) 
        s_emb = self.state_encoder(s_flat).view(B, K, self.d_model)
        s_emb = s_emb + t_embed + self.embed_tokentype(torch.ones_like(actions))

        # C. Action (Handle -1 for start token)
        a_ids = actions.clone()
        a_ids[a_ids < 0] = self.num_actions # Map -1 to the last embedding index
        a_emb = self.embed_action(a_ids) + t_embed + self.embed_tokentype(torch.full_like(actions, 2))

        # --- Interleave [R, s, a] ---
        # Stack: (B, K, 3, D) -> View: (B, 3*K, D)
        x = torch.stack([rtg_emb, s_emb, a_emb], dim=2).view(B, 3 * K, self.d_model)
        x = self.layer_norm(x)

        # --- Causal Masking ---
        # We need a mask size of 3*K
        L = 3 * K
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        # Handle padding mask (if provided)
        key_padding_mask = None
        if attention_mask is not None:
            # [CRITICAL FIX] Expand mask from (B, K) to (B, 3*K)
            # attention_mask: True = Valid, False = Pad
            # 1. Expand last dim: (B, K) -> (B, K, 3)
            # 2. Reshape: (B, K, 3) -> (B, 3*K)
            expanded_mask = attention_mask.unsqueeze(-1).expand(B, K, 3).reshape(B, 3 * K)
            
            # PyTorch transformer expects: True = Pad (Ignore), False = Keep
            # So we invert our mask (where we used True for Valid)
            key_padding_mask = ~expanded_mask

        # --- Transformer Pass ---
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # --- Prediction ---
        # We predict action given state. State tokens are at indices 1, 4, 7...
        idx = torch.arange(K, device=device) * 3 + 1
        idx = idx.unsqueeze(0).unsqueeze(-1).expand(B, K, self.d_model)
        
        s_hidden = out.gather(dim=1, index=idx)
        logits = self.action_head(s_hidden) # (B, K, num_actions)
        
        return logits


# ---------------- 3. Expert Data Collection (RAM Hacking) ----------------

def get_expert_action(env):
    """
    Reads Atari RAM to locate the paddle and ball, returning the optimal action.
    """
    # Breakout RAM: 72 = Paddle X, 99 = Ball X
    try:
        ram = env.unwrapped.ale.getRAM()
        paddle_x = ram[72]
        ball_x = ram[99]
    except AttributeError:
        # Fallback if ALE interface isn't accessible directly (rare)
        return env.action_space.sample()

    # Action 2 = Right, 3 = Left, 1 = Fire
    if paddle_x < ball_x - 2:
        return 2 
    elif paddle_x > ball_x + 2:
        return 3
    else:
        # If aligned, or ball not started, FIRE to start or wait
        return 1 

def collect_expert_trajectories(env, num_episodes=50, max_steps=10000):
    trajectories = []
    print(f"Collecting {num_episodes} Expert Episodes...")
    
    total_frames = 0
    scores = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = np.array(obs)
        
        states, actions, rewards = [], [], []
        done = False
        ep_return = 0
        
        for _ in range(max_steps):
            # 95% Expert, 5% Random noise (to improve robustness)
            if np.random.rand() < 0.95:
                action = get_expert_action(env)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            obs = np.array(next_obs)
            ep_return += reward
            
            if done:
                break
        
        # Only save valid games
        if len(actions) > 10:
            trajectories.append({
                "states": np.stack(states).astype(np.uint8),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
            })
            total_frames += len(actions)
            scores.append(ep_return)
            print(f"  Ep {ep+1}: Return {ep_return:.0f}, Length {len(actions)}")

    print(f"Collected {len(trajectories)} trajectories. Avg Score: {np.mean(scores):.2f}")
    return trajectories, np.mean(scores)


# ---------------- 4. Dataset & Utils ----------------

def compute_rtg(rewards, gamma=1.0):
    T = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running_sum = 0
    for t in reversed(range(T)):
        running_sum = rewards[t] + gamma * running_sum
        rtg[t] = running_sum
    return rtg

class DTTrajectoryDataset(Dataset):
    def __init__(self, trajectories, K=20, gamma=1.0):
        self.trajectories = trajectories
        self.K = K
        self.gamma = gamma
        self.indices = []
        
        # Index all possible start positions
        for i, traj in enumerate(trajectories):
            T = len(traj["actions"])
            # We can start anywhere from 0 to T-1
            for t in range(T):
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]
        T = len(traj["actions"])
        
        end_t = min(start_t + self.K, T)
        real_len = end_t - start_t
        
        # Extract slices
        s = traj["states"][start_t:end_t]
        a = traj["actions"][start_t:end_t]
        r = traj["rewards"][start_t:end_t]
        
        # Compute RTG for this segment (simplified)
        # Ideally we take the full episode RTG and slice it
        full_rtg = compute_rtg(traj["rewards"][start_t:], gamma=self.gamma)
        rtg_slice = full_rtg[:real_len]

        # Prepare padded buffers
        s_pad = np.zeros((self.K, 4, 84, 84), dtype=np.uint8)
        a_pad = -np.ones(self.K, dtype=np.int64) # -1 is padding
        rtg_pad = np.zeros((self.K, 1), dtype=np.float32)
        t_pad = np.zeros(self.K, dtype=np.int64)
        mask_pad = np.zeros(self.K, dtype=bool) # False means padding

        # Fill data
        s_pad[:real_len] = s
        # For actions input, we shift: a_input[t] = a[t-1]
        # At start_t=0, the first input is dummy -1
        a_pad[0] = -1
        if real_len > 1:
            a_pad[1:real_len] = a[:real_len-1]
            
        rtg_pad[:real_len, 0] = rtg_slice
        t_pad[:real_len] = np.arange(start_t, end_t)
        mask_pad[:real_len] = True
        
        # Targets are the actual actions
        a_target = -100 * np.ones(self.K, dtype=np.int64) # -100 ignored by CE loss
        a_target[:real_len] = a

        return {
            "states": torch.from_numpy(s_pad),
            "actions": torch.from_numpy(a_pad), # Input to model
            "rtg": torch.from_numpy(rtg_pad),
            "timesteps": torch.from_numpy(t_pad),
            "mask": torch.from_numpy(mask_pad),
            "targets": torch.from_numpy(a_target) # Ground truth
        }


# ---------------- 5. Evaluation Loop ----------------

@torch.no_grad()
def evaluate_agent(env, model, device, rtg_target, K=20):
    model.eval()
    obs, _ = env.reset()
    obs = np.array(obs) # (4, 84, 84)
    
    # Context buffers
    states = collections.deque(maxlen=K)
    actions = collections.deque(maxlen=K)
    rtgs = collections.deque(maxlen=K)
    timesteps = collections.deque(maxlen=K)
    
    # Initial state
    states.append(obs)
    actions.append(-1) # Start token
    rtgs.append(rtg_target)
    timesteps.append(0)
    
    ep_return = 0
    done = False
    cur_step = 0
    
    while not done:
        # Prepare batch
        # Convert deques to tensors
        # Pad if length < K
        cur_len = len(states)
        
        s_tensor = torch.zeros((1, K, 4, 84, 84), dtype=torch.uint8, device=device)
        a_tensor = -torch.ones((1, K), dtype=torch.long, device=device)
        r_tensor = torch.zeros((1, K, 1), dtype=torch.float32, device=device)
        t_tensor = torch.zeros((1, K), dtype=torch.long, device=device)
        mask = torch.zeros((1, K), dtype=torch.bool, device=device)
        
        # Fill from right (standard for transformers usually) or left? 
        # DT usually fills normally 0..L.
        s_np = np.array(states)
        s_tensor[0, :cur_len] = torch.from_numpy(s_np)
        
        a_np = np.array(actions)
        a_tensor[0, :cur_len] = torch.from_numpy(a_np)
        
        r_np = np.array(rtgs)
        r_tensor[0, :cur_len, 0] = torch.from_numpy(r_np)
        
        t_np = np.array(timesteps)
        t_tensor[0, :cur_len] = torch.from_numpy(t_np)
        
        mask[0, :cur_len] = True
        
        # Predict
        logits = model(r_tensor, s_tensor, a_tensor, t_tensor, attention_mask=mask)
        
        # Get action from the last valid position
        last_logits = logits[0, cur_len-1]
        action = torch.argmax(last_logits).item()
        
        # Step env
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward
        
        # Update context
        cur_step += 1
        states.append(np.array(next_obs))
        actions.append(action)
        # Decrement RTG
        current_rtg = rtgs[-1] - reward
        rtgs.append(current_rtg)
        timesteps.append(cur_step)
        
        if cur_step > 2000: break # Safety break
        
    return ep_return


# ---------------- 6. MAIN ----------------

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Environment & Expert Data
    env = make_breakout_env()
    expert_trajs, expert_mean = collect_expert_trajectories(env, num_episodes=20)
    
    # 2. Dataset
    K = 20
    dataset = DTTrajectoryDataset(expert_trajs, K=K)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 3. Model
    # Note: rtg_scale helps normalize the RTG inputs. 
    # Expert score is high, so we scale it down.
    model = DecisionTransformerDiscrete(
        num_actions=env.action_space.n,
        rtg_scale=10.0, 
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 4. Training Loop
    updates = 0
    max_updates = 2000
    
    print("Starting Training...")
    model.train()
    start_time = time.time()
    
    losses = []
    
    while updates < max_updates:
        for batch in loader:
            optimizer.zero_grad()
            
            rtg = batch['rtg'].to(device)
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            
            # Step mask to token mask expansion is handled internally or implicitly by padding mask
            # Here we just pass the padding mask (True = valid, False = pad)
            # The model expects True = pad for src_key_padding_mask if inverted, 
            # let's check logic inside model: key_padding_mask = ~attention_mask
            # So passing standard mask (True=Valid) works.
            
            logits = model(rtg, states, actions, timesteps, attention_mask=mask)
            
            # Calculate Loss (Ignore padding -100)
            # Logits: (B, K, A) -> (B*K, A)
            logits = logits.reshape(-1, env.action_space.n)
            targets = targets.reshape(-1)
            
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            updates += 1
            
            if updates % 100 == 0:
                print(f"Step {updates}: Loss {loss.item():.4f}")
            
            if updates >= max_updates:
                break

    print(f"Training finished in {time.time()-start_time:.1f}s")
    
    # 5. Final Evaluation
    print("\nRunning Evaluation...")
    eval_scores = []
    for i in range(5):
        # Prompt with slightly higher than expert mean to encourage best play
        target = expert_mean * 1.1 
        score = evaluate_agent(env, model, device, rtg_target=target, K=K)
        eval_scores.append(score)
        print(f"Eval Ep {i+1}: Score {score:.0f} (Target {target:.0f})")
        
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Updates")
    plt.show()

if __name__ == "__main__":
    main()