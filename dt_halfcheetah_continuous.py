import time

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class DecisionTransformerContinuous(nn.Module):
    """
    Decision Transformer for continuous action spaces (e.g., HalfCheetah).

    Sequence per time step:
        [return-to-go_t, state_t, action_{t-1}]
    The model predicts action_t at each step.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int = 20,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # Time, state, action, and return-to-go embeddings
        self.embed_t = nn.Embedding(max_ep_len, hidden_size)
        self.embed_s = nn.Linear(state_dim, hidden_size)
        self.embed_a = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # Transformer encoder (causal)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=4 * hidden_size,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

        # Action prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() if action_tanh else nn.Identity(),
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Args:
            states:       (B, K, state_dim)
            actions:      (B, K, act_dim)
            returns_to_go:(B, K, 1)
            timesteps:    (B, K)
        Returns:
            action_preds: (B, K, act_dim)
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Input embeddings
        time_embeddings = self.embed_t(timesteps)
        s_embeddings = self.embed_s(states) + time_embeddings
        a_embeddings = self.embed_a(actions) + time_embeddings
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # Interleave as [rtg, state, action] for each step
        stacked_inputs = (
            torch.stack((rtg_embeddings, s_embeddings, a_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Causal attention mask (block attending to future tokens)
        mask = torch.triu(
            torch.ones(3 * seq_length, 3 * seq_length, device=states.device),
            diagonal=1,
        ).bool()

        x = self.transformer(stacked_inputs, mask=mask)

        # Extract representations at state token positions (index 1, 4, 7, ...)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size)
        state_reps = x[:, :, 1, :]

        action_preds = self.predict_action(state_reps)
        return action_preds


def get_batch(batch_size, context_len, state_dim, act_dim, device):
    """
    Dummy batch generator using random data.

    In a full offline RL setup, this would sample subsequences from
    an offline dataset (e.g., D4RL HalfCheetah).
    """
    states = torch.randn(batch_size, context_len, state_dim, device=device)
    actions = torch.randn(batch_size, context_len, act_dim, device=device)
    rtg = torch.randn(batch_size, context_len, 1, device=device)
    timesteps = torch.randint(
        0, 100, (batch_size, context_len), device=device, dtype=torch.long
    )
    return states, actions, rtg, timesteps


def evaluate_dt(
    env,
    model,
    state_dim,
    act_dim,
    device,
    target_return: float = 1000.0,
    max_len: int = 20,
):
    """
    Run a single rollout in the environment using the trained model.
    """
    model.eval()
    state, _ = env.reset()

    states = torch.from_numpy(state).reshape(1, 1, state_dim).float().to(device)
    actions = torch.zeros((1, 1, act_dim), device=device).float()
    rtg = torch.tensor([[[target_return]]], device=device).float()
    timesteps = torch.tensor([[0]], device=device).long()

    episode_return = 0.0
    done = False

    while not done:
        # Truncate context to max_len
        if states.shape[1] > max_len:
            states = states[:, -max_len:, :]
            actions = actions[:, -max_len:, :]
            rtg = rtg[:, -max_len:, :]
            timesteps = timesteps[:, -max_len:]

        with torch.no_grad():
            action_preds = model(states, actions, rtg, timesteps)
            action = action_preds[0, -1].cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        # Append new transition to the context
        cur_action_tensor = (
            torch.from_numpy(action).reshape(1, 1, act_dim).float().to(device)
        )
        next_state_tensor = (
            torch.from_numpy(next_state).reshape(1, 1, state_dim).float().to(device)
        )
        next_rtg = rtg[0, -1, 0] - reward
        next_rtg_tensor = next_rtg.reshape(1, 1, 1).to(device)
        next_timestep_tensor = (timesteps[0, -1] + 1).reshape(1, 1).to(device)

        actions = torch.cat([actions, cur_action_tensor], dim=1)
        states = torch.cat([states, next_state_tensor], dim=1)
        rtg = torch.cat([rtg, next_rtg_tensor], dim=1)
        timesteps = torch.cat([timesteps, next_timestep_tensor], dim=1)

    return episode_return


def main():
    # Environment setup
    env_name = "HalfCheetah-v4"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"Env: {env_name}, state_dim={state_dim}, act_dim={act_dim}")

    # Choose device (prefer MPS on Apple Silicon if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    BATCH_SIZE = 64
    CONTEXT_LEN = 20
    HIDDEN_SIZE = 128
    LR = 1e-4
    STEPS = 1000

    # Model and optimizer
    model = DecisionTransformerContinuous(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=HIDDEN_SIZE,
        max_length=CONTEXT_LEN,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Track training loss for plotting
    loss_history = []

    model.train()
    start_time = time.time()

    for step in range(STEPS):
        states, true_actions, rtg, timesteps = get_batch(
            BATCH_SIZE, CONTEXT_LEN, state_dim, act_dim, device
        )
        action_preds = model(states, true_actions, rtg, timesteps)
        loss = loss_fn(action_preds, true_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # Plot training loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Decision Transformer (HalfCheetah) Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Single evaluation rollout
    print("Running evaluation rollout...")
    score = evaluate_dt(env, model, state_dim, act_dim, device, target_return=500.0)
    print(f"Episode reward: {score}")


if __name__ == "__main__":
    main()
