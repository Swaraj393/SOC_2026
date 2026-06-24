# week4_trading_env_v2.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class TradingEnvV2(gym.Env):
    """
    Extended trading environment with:
      - A window of the last `history_len` price returns in the state.
      - A transaction cost subtracted from the reward when position changes.

    Observation (history_len + 1 dimensional):
        [r_{t-history_len+1}, ..., r_t, current_position]

    Actions (discrete):
        0 = Go Long
        1 = Hold
        2 = Go Short

    Reward:
        position * price_return * 100
            - transaction_cost * |new_position - old_position|
    """

    def __init__(
        self,
        episode_length=100,
        initial_price=100.0,
        daily_vol=0.01,
        history_len=5,
        transaction_cost=0.1,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.initial_price = initial_price
        self.daily_vol = daily_vol
        self.history_len = history_len
        self.transaction_cost = transaction_cost

        obs_dim = history_len + 1  # history returns + current position
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.price = None
        self.position = None
        self.step_count = None
        self._price_history = None

    # -------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.initial_price
        self.position = 0.0
        self.step_count = 0
        self._price_history = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        return self._get_obs(last_return=0.0), {}

    # -------------------------------------------------------------- #
    def step(self, action):
        # Price dynamics
        price_return = float(
            self.np_random.normal(0.0, self.daily_vol)
        )
        self.price *= (1.0 + price_return)
        self._price_history.append(price_return)

        # Map action to new position
        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]

        # Reward: P&L from old position minus transaction cost
        pnl = self.position * price_return * 100.0
        cost = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost

        # The cost depends on how much the position *changed* this step,
        # not on the size of the position itself. So if the agent stays
        # put (Hold, or picks the same direction it was already in),
        # new_position - self.position is 0 and no cost is paid. The
        # cost only kicks in when the agent actually flips or opens a
        # position — e.g. going from -1 to +1 means |2| * 0.1 = 0.2
        # gets deducted. This is what discourages the agent from
        # churning back and forth for no reason.
        self.position = new_position

        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False

        return self._get_obs(price_return), reward, terminated, truncated, {}

    # -------------------------------------------------------------- #
    def _get_obs(self, last_return):
        history = np.array(list(self._price_history), dtype=np.float32)
        return np.append(history, self.position).astype(np.float32)

    # -------------------------------------------------------------- #
    def render(self):
        print(
            f"Step {self.step_count:3d} | "
            f"Price {self.price:8.2f} | "
            f"Position {self.position:+.0f}"
        )


# ================================================================== #
# Sanity checks — run this file directly to verify everything works  #
# ================================================================== #
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    # ---- 1. Gymnasium API check ---- #
    print("Running gymnasium env_checker...")
    check_env(TradingEnvV2(), warn=True)
    print("env_checker passed.\n")

    # ---- 2. Three random-agent episodes ---- #
    env = TradingEnvV2()
    print("Running 3 random-agent episodes on TradingEnvV2:")
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"  Episode {ep + 1}: total reward = {total_reward:.4f} "
              f"(finite={np.isfinite(total_reward)})")
    env.close()

    # ---- 3. Observation shape comparison ---- #
    from week3_trading_env import ToyTradingEnv

    v1_env = ToyTradingEnv()
    v2_env = TradingEnvV2()
    v1_obs, _ = v1_env.reset()
    v2_obs, _ = v2_env.reset()
    print("\nObservation shape comparison:")
    print(f"  ToyTradingEnv (v1) obs shape: {v1_obs.shape}  -> {v1_obs}")
    print(f"  TradingEnvV2  (v2) obs shape: {v2_obs.shape}  -> {v2_obs}")
    v1_env.close()
    v2_env.close()