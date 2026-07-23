import numpy as np


def sharpe_ratio(step_rewards, periods_per_year=252):
    """
    Annualised Sharpe ratio for a sequence of per-step rewards.
    """
    r = np.asarray(step_rewards, dtype=float)
    if r.size == 0 or np.isclose(r.std(), 0.0):
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(step_rewards):
    """
    Largest peak-to-trough decline of the cumulative P&L curve.
    """
    r = np.asarray(step_rewards, dtype=float)
    equity = np.cumsum(r)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def episode_summary(step_rewards):
    """
    Returns a dict with final P&L, Sharpe, max drawdown, and the curve.
    """
    r = np.asarray(step_rewards, dtype=float)
    return {
        "final_pnl": float(r.sum()),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "curve": np.cumsum(r),
    }


def mean_curve(curves):
    """
    Mean of multiple cumulative P&L curves after aligning lengths.
    """
    if len(curves) == 0:
        return np.array([], dtype=float)

    min_len = min(len(c) for c in curves)
    stacked = np.array([np.asarray(c, dtype=float)[:min_len] for c in curves])
    return stacked.mean(axis=0)