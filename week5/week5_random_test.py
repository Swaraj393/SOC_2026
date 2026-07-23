import numpy as np
from week5_trading_env_v3 import TradingEnvV3

env = TradingEnvV3(rho=0.0)

for episode in range(3):

    obs, _ = env.reset()

    total_reward = 0

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        done = terminated or truncated

    print(f"Episode {episode+1}: {total_reward:.2f}")