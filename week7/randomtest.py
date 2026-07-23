import numpy as np
from improved_env import ImprovedTradingEnv

returns = np.load("data/spy_returns.npy")

env = ImprovedTradingEnv(returns)

for episode in range(3):

    obs, _ = env.reset()

    done = False

    total = 0

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        total += reward

        done = terminated or truncated

    print(f"Episode {episode+1}: {total:.2f}")