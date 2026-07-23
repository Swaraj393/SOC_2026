import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


import numpy as np
from stable_baselines3 import PPO

from week6.week6_historical_env import HistoricalTradingEnv
from week4.week4_evaluate import average_cumulative_pnl, plot_pnl


def chronological_split(returns, train_fraction=0.8):
    """
    First 80% of days -> training
    Last 20% of days -> testing
    NEVER shuffle before splitting.
    """

    # Never shuffle time-series data before splitting because doing so
    # would mix future market information into the training data.
    # This introduces lookahead bias, allowing the agent to indirectly
    # learn from future events and producing unrealistically optimistic
    # evaluation results that would not occur in real trading.

    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]


if __name__ == "__main__":

    # Load the historical log returns
    returns = np.load(os.path.join(DATA_DIR, "returns.npy"))

    # Split chronologically into training and testing periods
    train_returns, test_returns = chronological_split(returns)

    print(f"Train: {len(train_returns)} days | Test: {len(test_returns)} days")

    # -------------------- Training --------------------

    train_env = HistoricalTradingEnv(train_returns)

    print("Training PPO on the training years ...")

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=100_000)

    print("Training complete.")

    # -------------------- Evaluation --------------------

    # Evaluate on the training data (seen during training)
    train_eval = HistoricalTradingEnv(train_returns)

    # Evaluate on the unseen test data
    test_eval = HistoricalTradingEnv(test_returns)

    pnl_train = average_cumulative_pnl(
        train_eval,
        n_episodes=50,
        model=model,
        strategy="trained",
    )

    pnl_test = average_cumulative_pnl(
        test_eval,
        n_episodes=50,
        model=model,
        strategy="trained",
    )

    pnl_random = average_cumulative_pnl(
        test_eval,
        n_episodes=50,
        model=None,
        strategy="random",
    )

    pnl_bnh = average_cumulative_pnl(
        test_eval,
        n_episodes=50,
        model=None,
        strategy="buy_and_hold",
    )

    print("\nFinal mean cumulative P&L over 50 episodes:")

    print(f" PPO on TRAIN data: {pnl_train[-1]:+8.2f}")
    print(f" PPO on TEST data:  {pnl_test[-1]:+8.2f}")
    print(f" Random (test):     {pnl_random[-1]:+8.2f}")
    print(f" Buy & hold (test): {pnl_bnh[-1]:+8.2f}")

    plot_pnl(
    pnl_test,
    pnl_random,
    pnl_bnh,
    filename=os.path.join(PLOTS_DIR, "test_cumulative_pnl.png"),
)