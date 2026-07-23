import numpy as np
from gymnasium.utils.env_checker import check_env
from improved_env import ImprovedTradingEnv

returns = np.load("data/spy_returns.npy")

env = ImprovedTradingEnv(returns)

check_env(env)

print("Environment check passed!")