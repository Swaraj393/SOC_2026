import numpy as np
from stable_baselines3 import PPO
from week5.week5_trading_env_v3 import TradingEnvV3
from week4.week4_evaluate import average_cumulative_pnl


def train_and_score(rho, transaction_cost, total_timesteps=50_000):
    """Train PPO, return the final mean cumulative P&L over 50 episodes."""
    env = TradingEnvV3(rho=rho, transaction_cost=transaction_cost)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps)

    env_eval = TradingEnvV3(rho=rho, transaction_cost=transaction_cost)
    pnl = average_cumulative_pnl(
        env_eval,
        n_episodes=50,
        model=model,
        strategy="trained",
    )
    return pnl[-1]


if __name__ == "__main__":
    print("--- Sweep 1: signal strength (transaction_cost = 0.1) ---")
    for rho in [0.0, 0.25, 0.5]:
        score = train_and_score(rho, transaction_cost=0.1)
        print(f"rho = {rho:.2f} | final mean cumulative P&L = {score:+8.2f}")

    print("--- Sweep 2: transaction cost (rho = 0.25) ---")
    for cost in [0.0, 0.1, 0.3]:
        score = train_and_score(0.25, transaction_cost=cost)
        print(f"cost = {cost:.2f} | final mean cumulative P&L = {score:+8.2f}")