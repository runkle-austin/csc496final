import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import matplotlib.pyplot as plt

try:
    import d4rl
except ImportError:
    pass 



class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=20, max_ep_len=4096, dropout=0.1):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.embed_t = nn.Embedding(max_ep_len, hidden_size)
        self.embed_s = nn.Linear(state_dim, hidden_size)
        self.embed_a = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=4 * hidden_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(), # Continuous action space usually bounded [-1, 1]
        )

    def forward(self, states, actions, rtg, timesteps):
        # Input shapes: (B, T, dim)
        B, T, _ = states.shape

        if T > self.max_length:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            rtg = rtg[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            T = self.max_length

        # Embeddings
        time_emb = self.embed_t(timesteps)
        s_emb = self.embed_s(states) + time_emb
        a_emb = self.embed_a(actions) + time_emb
        rtg_emb = self.embed_rtg(rtg) + time_emb

        # Interleave: [R_t, s_t, a_t]
        # (B, T, 3, H) -> (B, 3*T, H)
        stacked = (
            torch.stack((rtg_emb, s_emb, a_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(B, 3 * T, self.hidden_size)
        )
        stacked = self.embed_ln(stacked)

        # Causal mask
        mask = torch.triu(torch.ones(3 * T, 3 * T, device=states.device), diagonal=1).bool()

        x = self.transformer(stacked, mask=mask)

        # Extract state embeddings (middle token in triple)
        # x: (B, 3*T, H) -> (B, T, 3, H)
        x = x.reshape(B, T, 3, self.hidden_size)
        
        # Predict action based on s_t (index 1)
        return self.predict_action(x[:, :, 1, :])



class DebugDataset(Dataset):
    """Placeholder for D4RL dataset when testing pipeline mechanics."""
    def __init__(self, context_len, state_dim, act_dim, size=1000):
        self.context_len = context_len
        self.size = size
        # Pre-gen dummy data
        self.data = []
        print(f"Generating {size} dummy trajectories for debugging...")
        for _ in range(size):
            L = np.random.randint(50, 200)
            self.data.append({
                's': np.random.randn(L, state_dim).astype(np.float32),
                'a': np.random.randn(L, act_dim).astype(np.float32),
                'r': np.random.randn(L, 1).astype(np.float32),
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        traj = self.data[idx]
        T = traj['s'].shape[0]
        
        si = np.random.randint(0, max(1, T - self.context_len))
        ei = min(si + self.context_len, T)
        
        s = traj['s'][si:ei]
        a = traj['a'][si:ei]
        # Simplified RTG for debug
        # axis=0 to prevent flattening (fixes RuntimeError: mat1 and mat2 shapes...)
        rtg = np.cumsum(traj['r'][si:ei][::-1], axis=0)[::-1].copy()
        t = np.arange(si, ei)
        

        # Padding
        if s.shape[0] < self.context_len:
            pad = self.context_len - s.shape[0]
            s = np.pad(s, ((0, pad), (0, 0)))
            a = np.pad(a, ((0, pad), (0, 0)))
            rtg = np.pad(rtg, ((0, pad), (0, 0)))
            t = np.pad(t, (0, pad))
            
        return s, a, rtg, t




def eval_rollout(env, model, device, context_len, target_rtg):
    model.eval()
    state, _ = env.reset()
    
    # Context buffers
    states = torch.from_numpy(state).view(1, 1, -1).float().to(device)
    actions = torch.zeros((1, 1, model.act_dim), device=device)
    rtg = torch.tensor([[[target_rtg]]], device=device).float()
    timesteps = torch.zeros((1, 1), device=device, dtype=torch.long)

    total_rew = 0
    done = False
    
    while not done:
        # Windowing
        states = states[:, -context_len:]
        actions = actions[:, -context_len:]
        rtg = rtg[:, -context_len:]
        timesteps = timesteps[:, -context_len:]

        with torch.no_grad():
            preds = model(states, actions, rtg, timesteps)
            action = preds[0, -1].cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_rew += reward

        # Catting tensors in loop is slow, pre-alloc is better for prod but fine for eval script
        cur_a = torch.from_numpy(action).view(1, 1, -1).to(device)
        cur_s = torch.from_numpy(next_state).view(1, 1, -1).to(device).float()
        cur_r = rtg[0, -1, 0] - reward
        cur_r = cur_r.view(1, 1, 1).to(device)
        cur_t = timesteps[0, -1].view(1, 1) + 1

        actions = torch.cat([actions, cur_a], dim=1)
        states = torch.cat([states, cur_s], dim=1)
        rtg = torch.cat([rtg, cur_r], dim=1)
        timesteps = torch.cat([timesteps, cur_t], dim=1)

    return total_rew



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ctx_len", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rtg", type=float, default=1000.)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    env = gym.make(args.env)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    model = DecisionTransformer(s_dim, a_dim, args.dim, max_length=args.ctx_len).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Use dummy data for now
    ds = DebugDataset(args.ctx_len, s_dim, a_dim)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print(f"Training on {args.env} | s_dim: {s_dim} a_dim: {a_dim}")
    
    # Track all losses for plotting
    all_losses = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        
        for s, a, r, t in loader:
            s, a, r, t = s.to(device), a.to(device), r.to(device), t.to(device)
            
            # Predict action
            preds = model(s, a, r, t)
            loss = loss_fn(preds, a)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            all_losses.append(loss.item()) # Save loss
        
        dt = time.time() - t0
        print(f"Ep {epoch+1} | Loss: {np.mean(losses):.4f} | Time: {dt:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            score = eval_rollout(env, model, device, args.ctx_len, args.rtg)
            print(f"  >> Eval Score: {score:.2f}")

    # Plotting Logic
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses)
    plt.title("Training Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()