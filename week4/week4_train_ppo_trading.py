import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from week3_trading_env import ToyTradingEnv


class EpisodeRewardLogger(BaseCallback):
    """Records the total reward of each completed episode."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def train_ppo_on_trading(total_timesteps=100_000):
    env = ToyTradingEnv()
    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        # n_steps is basically how long PPO "plays" before it stops to learn.
        # It runs the env for 512 steps, stores everything in a buffer, and
        # only then does an update. Since one episode here is 100 steps,
        # that's roughly 5 episodes of data per update — feels like a decent
        # tradeoff between waiting too long and updating on too little.
        n_steps=512,
        # batch_size is the mini-batch size used inside that update.
        # PPO takes the 512-step buffer and chops it into chunks of 64 to do
        # gradient descent on. Smaller chunks = noisier gradients but more
        # steps per epoch. 64 is the SB3 default and it works fine here,
        # didn't see a reason to mess with it.
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()
    return model, callback.episode_rewards


def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_rewards(episode_rewards,
                 filename="plots/ppo_trading_rewards.png"):
    os.makedirs("plots", exist_ok=True)
    rewards = np.array(episode_rewards)
    plt.figure(figsize=(9, 4))
    plt.plot(rewards, alpha=0.25, color="steelblue",
             label="Episode reward")
    plt.plot(moving_average(rewards), color="steelblue",
             label="20-ep moving avg")
    plt.axhline(0, color="gray", linestyle="--",
                linewidth=0.8, label="Zero")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on ToyTradingEnv")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")

def random_agent_baseline(n_episodes=100):
    """Run a random agent for n episodes and return mean total reward."""
    env = ToyTradingEnv()
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        totals.append(total)
    env.close()
    return np.mean(totals)


if __name__ == "__main__":
    model, rewards = train_ppo_on_trading(total_timesteps=100_000)
    plot_rewards(rewards)
    np.save("ppo_trading_returns.npy", np.array(rewards))
    print(f"Final 20-episode average: {np.mean(rewards[-20:]):.2f}")
    random_avg = random_agent_baseline(n_episodes=100)
    ppo_avg = np.mean(rewards[-20:])
    print("\n--- Comparison ---")
    print(f"Random agent average (100 eps):  {random_avg:.2f}")
    print(f"PPO final 20-episode average:    {ppo_avg:.2f}")