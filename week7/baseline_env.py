import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class HistoricalTradingEnv(gym.Env):
    """
    Baseline historical trading environment.
    Observation: 5 historical returns + current position = 6 values.
    Actions:
        0 = Go Long
        1 = Hold
        2 = Go Short
    Reward:
        position * price_return * reward_scale
        - transaction_cost * |new_position - old_position|
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        episode_length=100,
        history_len=5,
        transaction_cost=0.1,
        reward_scale=100.0,
    ):
        super().__init__()

        returns = np.asarray(returns, dtype=np.float64)
        assert len(returns) > episode_length + 1, "not enough data"

        self.returns = returns
        self.episode_length = episode_length
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale

        obs_dim = history_len + 1

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1e6, dtype=np.float32),
            high=np.full(obs_dim, 1e6, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)

        self.position = 0.0
        self.step_count = 0
        self._start = 0
        self._price_history = deque([0.0] * history_len, maxlen=history_len)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_valid_start = len(self.returns) - self.episode_length - 1
        self._start = int(self.np_random.integers(0, last_valid_start + 1))

        self.position = 0.0
        self.step_count = 0
        self._price_history = deque([0.0] * self.history_len, maxlen=self.history_len)

        return self._get_obs(), {}

    def step(self, action):
        idx = self._start + self.step_count
        price_return = float(self.returns[idx])

        self._price_history.append(price_return)

        action_to_position = {
            0: 1.0,
            1: self.position,
            2: -1.0,
        }
        new_position = action_to_position[int(action)]

        pnl = self.position * price_return * self.reward_scale
        cost = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost

        self.position = new_position
        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        history = np.array(list(self._price_history), dtype=np.float32)
        return np.append(history, self.position).astype(np.float32)

    def render(self):
        print(
            f"Step {self.step_count:3d} | "
            f"Position {self.position:+.0f}"
        )