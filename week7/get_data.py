import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def download_market(ticker, name):
    """
    Download adjusted closing prices and compute log returns.
    """

    print(f"\nDownloading {ticker}...")

    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2024-12-31",
        auto_adjust=True,
        progress=False,
    )

    close = data["Close"].squeeze().dropna()

    close.to_csv(f"data/{name}_prices.csv")

    prices = close.to_numpy(dtype=float)

    returns = np.diff(np.log(prices))

    np.save(f"data/{name}_returns.npy", returns)

    lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]

    print(f"\n{name.upper()} Statistics")
    print("-" * 30)
    print(f"Days                : {len(returns)}")
    print(f"Mean Return         : {returns.mean():+.6f}")
    print(f"Std Return          : {returns.std():.6f}")
    print(f"Lag-1 Autocorr      : {lag1:+.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(prices)
    ax1.set_title(f"{ticker} Adjusted Close")

    ax2.plot(returns, linewidth=0.5)
    ax2.set_title("Daily Log Returns")

    plt.tight_layout()

    plt.savefig(f"plots/{name}_price_returns.png")

    plt.close()

    return {
        "ticker": ticker,
        "days": len(returns),
        "mean": returns.mean(),
        "std": returns.std(),
        "lag1": lag1,
    }


if __name__ == "__main__":

    spy = download_market("SPY", "spy")

    nifty = download_market("^NSEI", "nifty")

    print("\nSummary")
    print("=" * 60)

    for market in [spy, nifty]:
        print(
            f"{market['ticker']:8s}"
            f" Mean={market['mean']:+.6f}"
            f" Std={market['std']:.6f}"
            f" Lag1={market['lag1']:+.4f}"
        )