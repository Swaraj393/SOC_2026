import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_prices(ticker="SPY", start="2015-01-01", end="2024-12-31", filename="data/prices.csv"):
    os.makedirs("data", exist_ok=True)
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    close = data["Close"].squeeze().dropna()  # a pandas Series

    # auto_adjust=True folds in dividends and stock splits, so the price series
    # reflects the real total-return-like price history instead of showing fake
    # jumps caused by corporate actions.
    close.to_csv(filename)
    print(f"Saved {len(close)} rows to {filename}")
    return close


def load_prices(filename="data/prices.csv"):
    """Fallback loader: works with any CSV whose first column is a date
    and second column is a closing price."""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df.iloc[:, 0].dropna()


def compute_log_returns(close):
    prices = close.to_numpy(dtype=float)

    # np.diff(np.log(prices)) computes day-to-day log returns.
    # Log returns are used because they are approximately equal to percentage
    # changes for small daily moves, and they add cleanly over time.
    return np.diff(np.log(prices))


def describe_returns(returns):
    lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    print(f"Number of days: {len(returns)}")
    print(f"Mean daily return: {returns.mean():+.5f}")
    print(f"Std of daily returns: {returns.std():.5f}")
    print(f"Lag-1 autocorrelation: {lag1:+.4f}")


def plot_price_and_returns(close, returns, filename="plots/price_and_returns.png"):
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    ax1.plot(close.to_numpy(dtype=float), color="steelblue")
    ax1.set_ylabel("Close price")
    ax2.plot(returns, color="darkorange", linewidth=0.5)
    ax2.set_ylabel("Daily log return")
    ax2.set_xlabel("Trading day")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    close = download_prices("SPY")
    returns = compute_log_returns(close)
    describe_returns(returns)
    plot_price_and_returns(close, returns)
    np.save("data/returns.npy", returns)