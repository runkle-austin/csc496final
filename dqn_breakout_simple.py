import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# ----------------- 1. Environment Helper -----------------
def make_breakout_env(seed: int = 0):
    if "ALE/Breakout-v5" not in gym.registry:
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
        grayscale_newaxis=False,
        scale_obs=False, 
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


# convolution neural network to work with atari
class QLearningNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        # Renamed variable to match the call in forward()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Added normalization and float conversion
        x = x.float() / 255.0
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


# TODO - Adapt to be a Q-learning agent <- Neural network
# Removed BaseAgent inheritance for standalone execution
class TD_QLearningAgent:
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount")
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size")
        # Added epsilon for exploration
        self.epsilon = agent_info.get("epsilon", 0.1)

        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")

        # initialize the neural network

        # This line is drawn from PyTorch documentation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent running on device: {self.device}")
        
        self.q_net = QLearningNetwork(self.num_actions).to(self.device)
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=self.step_size, momentum=0.9)
        self.loss_fn = nn.MSELoss()

        # initialize the agent init state and agent to none
        self.state = None
        self.action = None
        self.last_state = None
        self.last_action = None

    def agent_start(self, state):
        # Adjusted tensor shape for CNN input
        tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_net(tensor)
        
        # Added epsilon-greedy strategy
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = torch.argmax(q_values, dim=1).item()
            
        self.last_state = state
        self.last_action = action
        return action

    def agent_step(self, reward, state):
        # get the current and next state as tensor
        cur_state = torch.tensor(self.last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_values = self.q_net(cur_state)
        current_q = q_values[0, self.last_action]

        # Added torch.no_grad for target calculation stability
        with torch.no_grad():
            next_q_values = self.q_net(next_state)
            max_next_q = torch.max(next_q_values)
            target = reward + self.discount * max_next_q

        loss = self.loss_fn(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # e greedy next action
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = torch.argmax(next_q_values, dim=1).item()
            
        self.last_state = state
        self.last_action = action
        return action


    def agent_end(self, reward):
        # for agent_end compute just the last action 
        cur_state = torch.tensor(self.last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        q_values = self.q_net(cur_state)
        current_q = q_values[0, self.last_action]
        
        target = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        loss = self.loss_fn(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def agent_cleanup(self):        
        self.last_state = None
        self.last_action = None


# ----------------- 4. Main Training Loop -----------------
def main():
    env = make_breakout_env()
    
    # get the number of actions
    num_actions = env.action_space.n
    
    # get the actions associated with inputs
    # for breakout
    # 0 = Back
    # 1 = launch
    # 2 = left
    # 3 = right
    meaning = env.unwrapped.get_action_meanings()
    print(f"Action Space: {num_actions} ({meaning})")

    agent = TD_QLearningAgent()
    agent.agent_init({
        "seed": 42,
        "discount": 0.99,
        "step_size": 1e-4, 
        "epsilon": 0.1,
        "num_actions": num_actions,
        "num_states": None 
    })

    num_episodes = 20
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        action = agent.agent_start(obs)
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            # for testing 
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                agent.agent_end(reward)
            else:
                action = agent.agent_step(reward, next_obs)
            
            total_reward += reward
            steps += 1
            
        print(f"Episode {ep+1}: Reward = {total_reward}, Steps = {steps}")

    agent.agent_cleanup()
    env.close()

if __name__ == "__main__":
    main()