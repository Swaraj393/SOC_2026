import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from week4_trading_env_v2 import TradingEnvV2


def run_episode(env, model=None):
    """Run one episode. model=None means random agent."""
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            # deterministic=True makes PPO pick the highest-probability action
            # from its policy instead of sampling. During training you want
            # sampling (deterministic=False) so the agent keeps exploring,
            # but at evaluation time we want to see what the agent *actually*
            # thinks is best — no randomness, no exploration noise. Otherwise
            # two eval runs of the same model would give different actions
            # and the comparison wouldn't really be apples-to-apples.
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


def buy_and_hold_episode(env):
    """Always stay long (action=0). Returns per-step rewards."""
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


def average_cumulative_pnl(env, n_episodes=50,
                           model=None, strategy="random"):
    """Return the mean cumulative P&L curve over n_episodes."""
    all_cumulative = []
    for _ in range(n_episodes):
        if strategy == "buy_and_hold":
            step_rewards = buy_and_hold_episode(env)
        else:
            step_rewards = run_episode(env, model=model)
        # np.cumsum turns the per-step rewards [r0, r1, r2, ...] into a
        # running total [r0, r0+r1, r0+r1+r2, ...]. That running total *is*
        # the cumulative P&L curve — at step t it tells us how much money
        # the agent has made (or lost) from the start of the episode up to
        # that point. Plotting it shows whether profits are building up
        # steadily or just bouncing around zero.
        all_cumulative.append(np.cumsum(step_rewards))
    min_len = min(len(c) for c in all_cumulative)
    stacked = np.array([c[:min_len] for c in all_cumulative])
    return stacked.mean(axis=0)


def plot_pnl(pnl_trained, pnl_random, pnl_bnh,
             filename="plots/cumulative_pnl.png",
             title="Agent comparison on TradingEnvV2"):
    os.makedirs("plots", exist_ok=True)
    steps = np.arange(len(pnl_trained))
    plt.figure(figsize=(9, 4))
    plt.plot(steps, pnl_trained, color="steelblue",
             label="PPO (trained)")
    plt.plot(steps, pnl_random, color="darkorange",
             label="Random agent")
    plt.plot(steps, pnl_bnh, color="green",
             label="Buy and hold")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Cumulative P&L (mean over 50 episodes)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")


def evaluate_setup(transaction_cost, plot_filename, plot_title):
    """Train a fresh PPO on TradingEnvV2 with the given cost and evaluate."""
    env = TradingEnvV2(transaction_cost=transaction_cost)
    print(f"\nTraining PPO (transaction_cost={transaction_cost}) ...")
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99,
    )
    model.learn(total_timesteps=100_000)
    print("Training complete.")

    env_eval = TradingEnvV2(transaction_cost=transaction_cost)
    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=50, model=model, strategy="trained")
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="buy_and_hold")

    print(f"Final cumulative P&L (mean over 50 episodes) "
          f"[cost={transaction_cost}]:")
    print(f"  PPO trained : {pnl_trained[-1]:.2f}")
    print(f"  Random agent: {pnl_random[-1]:.2f}")
    print(f"  Buy and hold: {pnl_bnh[-1]:.2f}")

    plot_pnl(pnl_trained, pnl_random, pnl_bnh,
             filename=plot_filename, title=plot_title)

    return pnl_trained[-1], pnl_random[-1], pnl_bnh[-1]


if __name__ == "__main__":
    # --- Default run: with transaction cost = 0.1 ---
    res_with_cost = evaluate_setup(
        transaction_cost=0.1,
        plot_filename="plots/cumulative_pnl.png",
        plot_title="Agent comparison on TradingEnvV2 (cost=0.1)",
    )

    # --- Second run: transaction cost = 0.0 (for the Task 3.3 question) ---
    res_no_cost = evaluate_setup(
        transaction_cost=0.0,
        plot_filename="plots/cumulative_pnl_nocost.png",
        plot_title="Agent comparison on TradingEnvV2 (cost=0.0)",
    )

    print("\n=== Summary ===")
    print(f"With cost=0.1  -> PPO: {res_with_cost[0]:.2f}, "
          f"Random: {res_with_cost[1]:.2f}, BnH: {res_with_cost[2]:.2f}")
    print(f"With cost=0.0  -> PPO: {res_no_cost[0]:.2f}, "
          f"Random: {res_no_cost[1]:.2f}, BnH: {res_no_cost[2]:.2f}")