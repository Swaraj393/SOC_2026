import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class ImprovedTradingEnv(gym.Env):
    """
    Improved historical trading environment.

    Upgrades:
    1) Richer observations:
       - rolling volatility
       - rolling momentum

    2) Five position sizes:
       +1, +0.5, 0, -0.5, -1

    Observation:
        [last 5 returns, rolling_volatility, rolling_momentum, current_position]
        => 8 values total when history_len=5
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        episode_length=100,
        history_len=5,
        feature_window=10,
        transaction_cost=0.1,
        reward_scale=100.0,
    ):
        super().__init__()

        returns = np.asarray(returns, dtype=np.float64)
        assert len(returns) > episode_length + feature_window, "not enough data"

        self.returns = returns
        self.episode_length = episode_length
        self.history_len = history_len
        self.feature_window = feature_window
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale

        obs_dim = history_len + 3

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1e6, dtype=np.float32),
            high=np.full(obs_dim, 1e6, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(5)

        self.position = 0.0
        self.step_count = 0
        self._start = 0
        self._price_history = deque([0.0] * history_len, maxlen=history_len)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_valid_start = len(self.returns) - self.episode_length - 1

        # Start far enough into the series so the rolling feature window is valid.
        self._start = int(
            self.np_random.integers(self.feature_window, last_valid_start + 1)
        )

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
            1: 0.5,
            2: 0.0,
            3: -0.5,
            4: -1.0,
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
        idx = self._start + self.step_count
        window = self.returns[idx - self.feature_window : idx]

        volatility = np.std(window)
        momentum = np.mean(window)

        history = np.array(list(self._price_history), dtype=np.float32)

        obs = np.concatenate(
            [
                history,
                np.array([volatility, momentum, self.position], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def render(self):
        print(
            f"Step {self.step_count:3d} | "
            f"Position {self.position:+.1f}"
        )