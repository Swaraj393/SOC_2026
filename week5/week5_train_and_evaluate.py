import os
import numpy as np
from stable_baselines3 import PPO
from week5.week5_trading_env_v3 import TradingEnvV3
from week4.week4_evaluate import average_cumulative_pnl, plot_pnl


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    env = TradingEnvV3(rho=0.5, transaction_cost=0.1)

    print("Training PPO on the momentum market (rho = 0.5) ...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=100_000)
    print("Training complete.")

    env_eval = TradingEnvV3(rho=0.5, transaction_cost=0.1)

    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=50, model=model, strategy="trained"
    )
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="random"
    )
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="buy_and_hold"
    )

    print("Final cumulative P&L (mean over 50 episodes):")
    print(f" PPO trained: {pnl_trained[-1]:+8.2f}")
    print(f" Random agent: {pnl_random[-1]:+8.2f}")
    print(f" Buy and hold: {pnl_bnh[-1]:+8.2f}")

    plot_pnl(
        pnl_trained,
        pnl_random,
        pnl_bnh,
        filename="plots/cumulative_pnl_momentum.png",
    )

    print("Saved plot to plots/cumulative_pnl_momentum.png")

    print("\nInspecting learned behavior step by step:\n")

    obs, _ = env_eval.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_eval.step(action)
        print(f"last return {obs[-2]:+.4f} -> position {obs[-1]:+.0f}")
        done = terminated or truncated