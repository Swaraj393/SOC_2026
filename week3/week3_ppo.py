# week3_ppo.py

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


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
        return True  # continue training


def train_ppo(env_name="CartPole-v1", total_timesteps=50_000):
    env = gym.make(env_name)
    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,   # Number of environment steps collected before each PPO update
        batch_size=64,  # Mini-batch size used when optimizing the policy/value networks
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"PPO evaluation: {mean_reward:.1f} +/- {std_reward:.1f}")

    env.close()
    return callback.episode_rewards


if __name__ == "__main__":
    ppo_returns = train_ppo()
    np.save("ppo_returns.npy", np.array(ppo_returns))
    print("Saved episode returns to ppo_returns.npy")