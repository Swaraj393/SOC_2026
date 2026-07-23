import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class HistoricalTradingEnv(gym.Env):
    """
    Trading environment that replays real historical returns.

    Each episode picks a random contiguous window of episode_length
    returns from the array passed to the constructor.

    Observation, actions, and reward are identical to TradingEnvV3.
    The only difference is that returns now come from historical data
    instead of being generated from an AR(1) process.

    IMPORTANT:
    The train/test split happens OUTSIDE this class.
    Construct one environment with the training returns and another
    environment with the test returns.
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
        assert len(returns) > episode_length + 1, "Not enough historical data."

        self.returns = returns
        self.episode_length = episode_length
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale

        obs_dim = history_len + 1

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)

        self.position = None
        self.step_count = None
        self._start = None
        self._price_history = None

    # -------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random contiguous window of history for every episode.
        last_valid_start = len(self.returns) - self.episode_length - 1
        self._start = int(self.np_random.integers(0, last_valid_start))

        # Episodes start at random positions so the agent experiences many
        # different parts of the historical dataset instead of memorizing
        # one fixed sequence that always starts from day 0.
        self.position = 0.0
        self.step_count = 0

        self._price_history = deque(
            [0.0] * self.history_len,
            maxlen=self.history_len,
        )

        return self._get_obs(), {}

    # -------------------------------------------------------------

    def step(self, action):

        # Return comes directly from historical data.
        price_return = float(
            self.returns[self._start + self.step_count]
        )

        self._price_history.append(price_return)

        action_to_position = {
            0: 1.0,
            1: self.position,
            2: -1.0,
        }

        new_position = action_to_position[int(action)]

        pnl = self.position * price_return * self.reward_scale

        cost = self.transaction_cost * abs(
            new_position - self.position
        )

        reward = pnl - cost

        self.position = new_position

        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated = False

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {},
        )

    # -------------------------------------------------------------

    def _get_obs(self):
        history = np.array(
            list(self._price_history),
            dtype=np.float32,
        )

        return np.append(
            history,
            self.position,
        ).astype(np.float32)