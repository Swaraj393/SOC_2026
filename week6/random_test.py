import numpy as np

from week6_historical_env import HistoricalTradingEnv

returns = np.load("data/returns.npy")

env = HistoricalTradingEnv(returns)

for episode in range(3):

    obs, _ = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        done = terminated or truncated

    print(
        f"Episode {episode+1}: "
        f"Total Reward = {total_reward:.2f}"
    )