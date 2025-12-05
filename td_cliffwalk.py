import numpy as np
import matplotlib.pyplot as plt


class CliffWalkEnvironment:
    """
    Standard 4x12 Cliff Walking gridworld.

    Layout (row, col):
      - Start at (3, 0)
      - Goal at  (3, 11)
      - Cells (3, 1) ... (3, 10) are the cliff:
          stepping on them gives reward -100 and sends the agent back to start
      - Every step gives reward -1
    Actions:
      0 = UP, 1 = LEFT, 2 = DOWN, 3 = RIGHT
    """

    def __init__(self, grid_h: int = 4, grid_w: int = 12):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.start_loc = (grid_h - 1, 0)
        self.goal_loc = (grid_h - 1, grid_w - 1)
        self.cliff = [(grid_h - 1, i) for i in range(1, grid_w - 1)]
        self.agent_loc = self.start_loc

    def state(self, loc):
        """Map (row, col) to a single integer state id."""
        return loc[0] * self.grid_w + loc[1]

    def env_start(self):
        """Reset the environment and return the start state id."""
        self.agent_loc = self.start_loc
        return self.state(self.agent_loc)

    def is_in_bounds(self, x, y):
        return 0 <= x < self.grid_h and 0 <= y < self.grid_w

    def env_step(self, action: int):
        """
        Take one environment step.

        Returns:
            reward (float)
            next_state (int)
            terminal (bool)
        """
        x, y = self.agent_loc

        # 0: UP, 1: LEFT, 2: DOWN, 3: RIGHT
        if action == 0:
            x -= 1
        elif action == 1:
            y -= 1
        elif action == 2:
            x += 1
        elif action == 3:
            y += 1
        else:
            raise ValueError(f"Invalid action {action}")

        # stay in place if hitting a wall
        if not self.is_in_bounds(x, y):
            x, y = self.agent_loc

        self.agent_loc = (x, y)

        reward = -1.0
        terminal = False

        # falling off the cliff
        if self.agent_loc in self.cliff:
            reward = -100.0
            self.agent_loc = self.start_loc

        # reaching the goal
        if self.agent_loc == self.goal_loc:
            terminal = True

        return reward, self.state(self.agent_loc), terminal


class QLearningAgent:
    """
    Tabular Q-learning agent for the Cliff Walking environment.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: [state, action]
        self.Q = np.zeros((num_states, num_actions), dtype=np.float32)

        self.last_state = None
        self.last_action = None

    def _epsilon_greedy(self, state: int) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        values = self.Q[state]
        return int(np.argmax(values))

    def start(self, state: int) -> int:
        """Agent start; choose initial action."""
        action = self._epsilon_greedy(state)
        self.last_state = state
        self.last_action = action
        return action

    def step(self, reward: float, state: int) -> int:
        """
        Q-learning update for a non-terminal transition
        and choose next action.
        """
        a_next = self._epsilon_greedy(state)
        s, a = self.last_state, self.last_action

        td_target = reward + self.gamma * np.max(self.Q[state])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

        self.last_state = state
        self.last_action = a_next
        return a_next

    def end(self, reward: float) -> None:
        """Q-learning update on terminal transition."""
        s, a = self.last_state, self.last_action
        td_target = reward  # no bootstrapping at terminal
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Anneal ε toward epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_cliffwalk_experiment(
    num_episodes: int = 5000,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.999,
    max_steps_per_episode: int = 1000,
    plot_title: str = "Policy Evaluation on Optimal Policy",
):
    """
    Train a Q-learning agent on CliffWalking and plot episode returns.
    """
    env = CliffWalkEnvironment()
    num_states = env.grid_h * env.grid_w
    num_actions = 4

    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    episode_returns = []
    episode_lengths = []
    reached_goal = 0
    fell_off_cliff = 0

    for ep in range(num_episodes):
        state = env.env_start()
        action = agent.start(state)

        done = False
        G = 0.0
        t = 0

        while not done and t < max_steps_per_episode:
            reward, next_state, terminal = env.env_step(action)
            G += reward
            t += 1

            # statistics
            if terminal and env.agent_loc == env.goal_loc:
                reached_goal += 1
            if reward == -100.0:
                fell_off_cliff += 1

            if terminal:
                agent.end(reward)
                done = True
            else:
                action = agent.step(reward, next_state)

        agent.decay_epsilon()

        episode_returns.append(G)
        episode_lengths.append(t)

        if (ep + 1) % 1000 == 0 or ep == 0:
            avg_recent = np.mean(episode_returns[-1000:])
            print(
                f"[CliffWalk Q-learning] Episode {ep + 1}/{num_episodes} | "
                f"Return: {G:.2f} | "
                f"Avg over last {min(1000, len(episode_returns))} eps: {avg_recent:.2f} | "
                f"Steps: {t}"
            )

    print(f"Reached goal episodes: {reached_goal}")
    print(f"Fell off cliff episodes: {fell_off_cliff}")

    # plot learning curve
    plt.figure(figsize=(6, 5))
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(plot_title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return episode_returns, episode_lengths, agent


if __name__ == "__main__":
    run_cliffwalk_experiment()
