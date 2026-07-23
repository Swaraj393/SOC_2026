import numpy as np
from week5.week5_trading_env_v3 import TradingEnvV3


def collect_returns(env, n_steps=10_000):
    """
    Step through the env with Hold actions and record each return.

    We use Hold actions because the environment's price dynamics do not depend
    on the agent's action. We want to measure the market signal itself, not a
    strategy effect. Any action sequence would produce the same return process;
    Hold just keeps the position unchanged and makes the check simple.
    """
    returns = []
    obs, _ = env.reset(seed=0)

    for _ in range(n_steps):
        obs, reward, terminated, truncated, _ = env.step(1)  # Hold
        returns.append(obs[-2])  # newest return in the history window

        if terminated or truncated:
            obs, _ = env.reset()

    return np.array(returns)


def lag1_autocorrelation(returns):
    """
    Correlation between each return and the one right after it.

    np.corrcoef(returns[:-1], returns[1:]) compares the return series shifted
    by one step against itself, so it measures how strongly r_t and r_{t+1}
    move together.
    """
    return np.corrcoef(returns[:-1], returns[1:])[0, 1]


def random_agent_mean_reward(env, n_episodes=20):
    totals = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False

        while not done:
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            total += reward
            done = terminated or truncated

        totals.append(total)

    return float(np.mean(totals))


if __name__ == "__main__":
    for rho in [0.0, 0.5, -0.5]:
        env = TradingEnvV3(rho=rho, transaction_cost=0.0)
        r = collect_returns(env)
        ac = lag1_autocorrelation(r)
        mean_rand = random_agent_mean_reward(TradingEnvV3(rho=rho))

        print(
            f"rho = {rho:+.2f} | measured autocorr = {ac:+.3f} | "
            f"random agent mean reward = {mean_rand:+8.2f}"
        )