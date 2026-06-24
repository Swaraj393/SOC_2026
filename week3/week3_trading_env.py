# week3_trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils.env_checker import check_env


class ToyTradingEnv(gym.Env):
    """
    Simplest possible trading environment.

    Dynamics:
    - Asset price follows a random walk (log-normal returns).
    - Episode length is fixed at `episode_length` steps.

    Observation (2-dimensional):
    [last_price_return, current_position]
    position in {-1.0 (short), 0.0 (flat), 1.0 (long)}

    Actions (discrete):
    0 = Go Long (buy / stay long)
    1 = Hold (keep current position)
    2 = Go Short (sell / stay short)

    Reward:
    current_position * price_return * 100

    This rewards the agent when its position matches the direction of the price move,
    and penalizes it when the position is on the wrong side.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_length=100, initial_price=100.0, daily_vol=0.01):
        super().__init__()
        self.episode_length = episode_length
        self.initial_price = initial_price
        self.daily_vol = daily_vol

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -1.0], dtype=np.float32),
            high=np.array([np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=long, 1=hold, 2=short

        self.price = None
        self.position = None
        self.step_count = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.initial_price
        self.position = 0.0  # start flat
        self.step_count = 0

        obs = np.array([0.0, self.position], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Price dynamics:
        # self.np_random.normal() samples a random number from a normal distribution.
        # The random seed comes from reset(seed=...) through gymnasium's internal RNG,
        # so if a seed is provided, the environment becomes reproducible.
        price_return = float(self.np_random.normal(0.0, self.daily_vol))
        self.price *= (1.0 + price_return)

        # Map action to new position.
        # action = 1 means Hold, so the position stays exactly the same.
        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]

        # Reward uses the current position because this position was held during
        # the price move. Multiplying by price_return gives profit if the agent
        # was aligned with the move, and loss if it was on the wrong side.
        reward = self.position * price_return * 100.0

        # Advance state
        self.position = new_position
        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated = False

        obs = np.array([price_return, self.position], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        print(
            f"Step {self.step_count:3d} | "
            f"Price {self.price:8.2f} | "
            f"Position {self.position:+.0f}"
        )


def run_random_agent(n_episodes=10):
    """Run a random agent and report average total reward."""
    env = ToyTradingEnv()
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)
        print(f"Episode {ep + 1:2d}: total reward = {total_reward:7.2f}")

    print(f"\nMean over {n_episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()
    return total_rewards


if __name__ == "__main__":
    check_env(ToyTradingEnv(), warn=True)
    run_random_agent()