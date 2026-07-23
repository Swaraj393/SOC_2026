import numpy as np
from gymnasium.utils.env_checker import check_env

from week6_historical_env import HistoricalTradingEnv

returns = np.load("data/returns.npy")

env = HistoricalTradingEnv(returns)

check_env(env, warn=True)

print("Environment check passed!")