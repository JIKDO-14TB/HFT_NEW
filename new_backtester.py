from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Types
# ------------------------------------------------------------------------------
Action = Optional[str]  # "LONG" | "SHORT" | None


@dataclass
class BacktestConfig:
    """Configuration parameters for the backtester.

    Attributes
    ----------
    initial_capital : float
        Starting capital in the quote currency.
    maker_fee_bp : float
        Maker fee in basis points applied to limit orders.
    taker_fee_bp : float
        Taker fee in basis points applied to market orders.
    latency_ms : int
        Execution latency applied to every order in milliseconds.
    decision_ms : int
        Minimum time gap between successive decisions in milliseconds.
    max_holding_seconds : int
        Maximum holding period for a position.  Positions are
        automatically closed if exceeded.
    limit_ratio : float
        Fraction of capital allocated to the primary (target) order.
    target_bp : float
        Price offset in basis points for the primary limit order.
    partial_bp : float
        Price offset in basis points for the secondary order.
    stop_bp : float
        Stop loss threshold in basis points.  Positions are closed
        immediately when loss exceeds this magnitude.
    partial_as_limit : bool
        If ``True``, submit the secondary slice as a limit order
        (maker), otherwise submit it as a market order (taker).
    """

    initial_capital: float = 100.0
    maker_fee_bp: float = 5.0
    taker_fee_bp: float = 2.0
    latency_ms: int = 10
    decision_ms: int = 10
    max_holding_seconds: int = 240
    limit_ratio: float = 0.7
    target_bp: float = 3.0
    partial_bp: float = 1.0
    stop_bp: float = 10.0
    partial_as_limit: bool = False


class SimpleBacktester:
    """Execute a simple long/short strategy on tick data.

    Parameters
    ----------
    quotes : pandas.DataFrame
        A DataFrame with columns ``ts`` (datetime64[ns] or datetime64[ns, tz]) and
        ``price`` representing the mid price.  Additional columns are ignored.
    config : BacktestConfig
        Configuration controlling fees, latency and order sizes.

    Notes
    -----
    Internally the backtester converts the ``ts`` column into a NumPy
    array of nanoseconds to accelerate comparisons.  Price and other
    numeric fields are also converted to NumPy arrays.  No copying of
    the DataFrame is done beyond the minimal conversion in order to
    reduce memory consumption.
    """

    def __init__(self, quotes: pd.DataFrame, *, config: BacktestConfig) -> None:
        if quotes is None or quotes.empty:
            raise ValueError("quotes empty")

        # Ensure we don't mutate the original DataFrame.  We only keep
        # the columns needed for execution.  Converting timestamps and
        # prices to NumPy upfront eliminates overhead inside the loop.
        df = quotes.copy()

        # Normalise timestamps to naive datetime64[ns] and sort.
        ts = pd.to_datetime(df["ts"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df["ts"] = ts.astype("datetime64[ns]")
        df = df.sort_values("ts").reset_index(drop=True)

        self.cfg = config
        self.quotes = df
        # Convert to numpy for speed
        self.ts_ns: np.ndarray = df["ts"].values.astype("datetime64[ns]").astype("int64")
        # Cast to float64 to avoid precision loss when computing returns
        self.price: np.ndarray = df.get("price").astype("float64").values

        # Latencies and durations in nanoseconds
        self.latency_ns = int(config.latency_ms) * 1_000_000
        self.decision_ns = int(config.decision_ms) * 1_000_000
        self.max_hold_ns = int(config.max_holding_seconds) * 1_000_000_000

    # ------------------------------------------------------------------
    # Fee helper
    # ------------------------------------------------------------------
    @staticmethod
    def _fee(notional: float, bp: float) -> float:
        """Calculate fee given a notional and a fee in basis points."""
        return notional * bp / 10_000.0

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(
        self,
        signal_func: Callable[[Dict[str, Any], Dict[str, Any]], Action],
        *,
        start_capital: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, float, float]:
        """Execute the backtest.

        Parameters
        ----------
        signal_func : callable
            Function invoked at each decision tick with arguments
            ``signal_func(info, state) -> Action``.  ``info`` contains
            at least the current index ``i``.  ``state`` contains
            mutable strategy state (position, entry price etc.).  The
            function must return ``"LONG"``, ``"SHORT"`` or ``None``.
        start_capital : float, optional
            If provided, use this value instead of ``config.initial_capital``
            as the starting capital.  Useful when chaining multi-day
            backtests.

        Returns
        -------
        (trades, final_capital, turnover)
            ``trades`` is a DataFrame detailing each ENTER and EXIT
            event with columns ``ts``, ``type``, ``side`` (for entries),
            ``capital_before`` and ``capital_after``.  ``final_capital``
            is the ending capital after all trades and fees.  ``turnover``
            is the total notional traded.
        """
        cfg = self.cfg
        n = len(self.ts_ns)

        # Initialise strategy state.  Using a plain dict rather than a
        # dataclass avoids attribute lookups in the hot loop.  See
        # analysis notes on Python overhead【761670738180802†L382-L404】.
        state: Dict[str, Any] = {
            "position": None,  # type: Optional[str]
            "entry_price": 0.0,
            "entry_ts": 0,  # nanoseconds
            "capital": cfg.initial_capital if start_capital is None else float(start_capital),
            "turnover": 0.0,
            "last_decision_ts": -1,  # nanoseconds of last decision
            "trades": [],  # list of trade dictionaries
        }

        exec_i = 0  # pointer to execution index (after latency)

        # Main loop over decision timestamps
        for i in range(n):
            decision_ts = self.ts_ns[i]

            # Apply latency: find first quote index with ts >= decision_ts + latency
            target_ts = decision_ts + self.latency_ns
            while exec_i < n and self.ts_ns[exec_i] < target_ts:
                exec_i += 1
            if exec_i >= n:
                break

            mid = self.price[exec_i]

            # Stop loss: close position if return falls below threshold
            if state["position"] is not None:
                side = state["position"]
                entry = state["entry_price"]
                # Compute return in bp for current side
                ret_bp = (
                    (mid - entry) / entry * 10_000
                    if side == "LONG"
                    else (entry - mid) / entry * 10_000
                )
                if ret_bp <= -cfg.stop_bp:
                    self._exit(exec_i, state)
                    continue

            # Time stop: exit if holding duration exceeds max_hold_ns
            if state["position"] is not None and (self.ts_ns[exec_i] - state["entry_ts"] >= self.max_hold_ns):
                self._exit(exec_i, state)
                continue

            # Entry: enforce decision gating
            if state["position"] is None:
                #if decision_ts - state["last_decision_ts"] < self.decision_ns:
                #    continue
                action = signal_func({"i": i}, state)
                state["last_decision_ts"] = decision_ts
                if action in ("LONG", "SHORT"):
                    self._enter(exec_i, action, state)

        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(state["trades"])
        return trades_df, state["capital"], state["turnover"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enter(self, i: int, side: str, state: Dict[str, Any]) -> None:
        """Handle entering a new position.

        Parameters
        ----------
        i : int
            Index into the quotes arrays at which the order is filled.
        side : {"LONG", "SHORT"}
            Direction of the trade.
        state : dict
            Mutable state of the strategy.  This method updates the
            capital, position and trade history in place.
        """
        cfg = self.cfg
        cap_before = state["capital"]
        d = 1 if side == "LONG" else -1
        mid = self.price[i]

        # Notional sizes
        primary_notional = cap_before * cfg.limit_ratio
        secondary_notional = cap_before - primary_notional

        # Price levels (basis points offset)
        # For a LONG we want to buy below or above the current mid: the
        # existing convention in the reference code places a limit order
        # below the mid and a market/limit order above.  For a SHORT it
        # inverts these.  Users may adjust offsets in the config.
        primary_price = mid * (1 - d * cfg.target_bp / 10_000.0)
        secondary_price = mid * (1 + d * cfg.partial_bp / 10_000.0)

        # Fees: maker on primary; maker or taker on secondary depending on flag
        fee_primary = self._fee(primary_notional, cfg.maker_fee_bp)
        if cfg.partial_as_limit:
            fee_secondary = self._fee(secondary_notional, cfg.maker_fee_bp)
        else:
            fee_secondary = self._fee(secondary_notional, cfg.taker_fee_bp)

        total_fee = fee_primary + fee_secondary

        # Effective entry price: weighted average of the two slices
        # Note: this assumes both orders are fully filled at their
        # specified prices.  In a live environment one would need to
        # handle partial fills separately.
        weighted_price = (primary_notional * primary_price + secondary_notional * secondary_price) / cap_before

        # Update capital after paying entry fees
        cap_after = cap_before - total_fee

        # Update state
        state["capital"] = cap_after
        state["position"] = side
        state["entry_price"] = weighted_price
        state["entry_ts"] = self.ts_ns[i]
        state["turnover"] += cap_before

        # Record trade
        state["trades"].append(
            {
                "ts": self.quotes["ts"].iloc[i],
                "type": "ENTER",
                "side": side,
                "capital_before": cap_before,
                "capital_after": cap_after,
            }
        )

    def _exit(self, i: int, state: Dict[str, Any]) -> None:
        """Handle exiting the current position.

        Parameters
        ----------
        i : int
            Index into the quotes arrays at which the exit occurs.
        state : dict
            Mutable strategy state.  This method updates capital,
            position and trade history in place.
        """
        cfg = self.cfg
        cap_before = state["capital"]
        d = 1 if state["position"] == "LONG" else -1
        mid = self.price[i]

        # Calculate return relative to entry price
        ret = d * (mid - state["entry_price"]) / state["entry_price"]
        pnl_before_fee = cap_before * ret

        # Exit uses taker fee on the entire position
        fee = self._fee(cap_before, cfg.taker_fee_bp)

        cap_after = cap_before + pnl_before_fee - fee

        state["capital"] = cap_after
        state["turnover"] += cap_before
        state["position"] = None

        # Record trade
        state["trades"].append(
            {
                "ts": self.quotes["ts"].iloc[i],
                "type": "EXIT",
                "capital_before": cap_before,
                "capital_after": cap_after,
            }
        )