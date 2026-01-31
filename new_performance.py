from __future__ import annotations

from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd


def equity_curve_from_exits(trades_df: pd.DataFrame, *, initial_capital: float) -> pd.Series:
    """Build cumulative PnL series at EXIT timestamps.

    The returned Series is indexed by exit timestamps and contains
    cumulative PnL measured as ``capital_after - initial_capital`` at
    each exit event.  If the trades DataFrame is empty or missing
    required columns the result is an empty Series.
    """
    if trades_df is None or trades_df.empty:
        return pd.Series(dtype="float64")
    if "type" not in trades_df.columns or "ts" not in trades_df.columns:
        return pd.Series(dtype="float64")
    exits = trades_df[trades_df["type"] == "EXIT"].copy()
    if exits.empty:
        return pd.Series(dtype="float64")
    if "capital_after" not in exits.columns:
        raise ValueError("EXIT trades must contain 'capital_after'")
    exits = exits.sort_values("ts")
    equity_vals = exits["capital_after"].astype(float) - float(initial_capital)
    return pd.Series(equity_vals.values, index=exits["ts"], name="equity_exit")


def build_equity_full(
    *,
    trades_df: pd.DataFrame,
    quote_ts: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Build full equity curve aligned to quote timestamps (forward-filled)."""
    if quote_ts is None or len(quote_ts) == 0:
        return pd.Series(dtype="float64")
    idx = pd.Index(quote_ts)
    if trades_df is None or trades_df.empty:
        return pd.Series(0.0, index=idx, name="equity_full")
    if "ts" not in trades_df.columns or "capital_after" not in trades_df.columns:
        raise ValueError("trades_df must contain 'ts' and 'capital_after' columns")
    tdf = trades_df[["ts", "capital_after"]].copy()
    tdf = tdf.sort_values("ts").drop_duplicates("ts", keep="last")
    cap_series = pd.Series(tdf["capital_after"].astype(float).values, index=tdf["ts"])
    cap_full = cap_series.reindex(idx, method="ffill")
    cap_full = cap_full.fillna(float(initial_capital))
    equity_full = cap_full - float(initial_capital)
    equity_full.name = "equity_full"
    return equity_full


def performance_metrics(
    *,
    equity_full: pd.Series,
    turnover: float,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> Dict[str, float]:
    """Compute a dictionary of performance metrics from an equity curve."""
    if equity_full is None or equity_full.empty:
        return {
            "cum_pnl": 0.0,
            "cum_return_simple": 0.0,
            "sharpe_daily": 0.0,
            "sharpe_annualized": 0.0,
            "max_drawdown": 0.0,
            "turnover_ratio": 0.0,
        }
    eq = equity_full.astype(float)
    cum_pnl = float(eq.iloc[-1])
    cum_return_simple = float(cum_pnl / float(initial_capital))
    rets = eq.diff().fillna(0.0)
    if periods_per_year is None:
        diffs = eq.index.to_series().diff().dropna()
        if not diffs.empty:
            median_sec = diffs.dt.total_seconds().median()
            if median_sec and median_sec > 0:
                sec_per_year = 365.25 * 24 * 3600
                periods_per_year = int(round(sec_per_year / median_sec))
            else:
                periods_per_year = 1
        else:
            periods_per_year = 1
    rf_per_period = float(risk_free_rate) / float(periods_per_year)
    std = float(rets.std(ddof=0))
    mean = float(rets.mean())
    sharpe_daily = 0.0 if std == 0 else (mean / std)
    sharpe_ann = 0.0 if std == 0 else ((mean - rf_per_period) / std) * np.sqrt(periods_per_year)
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd = float(dd.min())
    turnover_ratio = float(turnover) / float(initial_capital)
    return {
        "cum_pnl": cum_pnl,
        "cum_return_simple": cum_return_simple,
        "sharpe_daily": float(sharpe_daily),
        "sharpe_annualized": float(sharpe_ann),
        "max_drawdown": max_dd,
        "turnover_ratio": turnover_ratio,
    }


def performance_summary(
    equity: pd.Series,
    *,
    turnover: float,
    initial_capital: float,
    periods_per_year: int = 365 * 24 * 60 * 60,
) -> Dict[str, float]:
    """Compute a simplified performance summary."""
    if equity.empty:
        return {}
    returns = equity.diff().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = 0.0
    if std_ret > 0:
        sharpe = mean_ret / std_ret * np.sqrt(periods_per_year)
    turnover_pct = turnover / initial_capital * 100.0
    return {
        "final_capital": equity.iloc[-1],
        "total_pnl": equity.iloc[-1] - initial_capital,
        "turnover_pct": turnover_pct,
        "sharpe": sharpe,
        "max_drawdown": (equity - equity.cummax()).min(),
    }


def performance_metrics_from_exits(
    *,
    equity_exit: pd.Series,
    turnover: float,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    trades_per_year: int = 365 * 24 * 60,
) -> Dict[str, float]:
    """Compute performance metrics from exit-only equity curve."""
    if equity_exit is None or equity_exit.empty:
        return {
            "cum_pnl": 0.0,
            "cum_return_simple": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover_ratio": 0.0,
        }
    eq = equity_exit.astype(float)
    rets = eq.diff().dropna()
    mean = rets.mean()
    std = rets.std(ddof=0)
    sharpe = 0.0
    if std > 0:
        sharpe = mean / std * np.sqrt(trades_per_year)
    running_max = eq.cummax()
    dd = eq - running_max
    return {
        "cum_pnl": float(eq.iloc[-1]),
        "cum_return_simple": float(eq.iloc[-1] / initial_capital),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "turnover_ratio": float(turnover / initial_capital),
    }