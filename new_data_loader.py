"""Data loading utilities for the backtesting engine.

This module defines a :class:`DataLoader` capable of reading Binance‐style
parquet files for aggregated trades (`aggTrades`), best bid/ask quotes
(`bookTicker`) and minute candlesticks (`klines_1m`).  It provides
helpers to iterate over date ranges and resample quote data to a
coarser clock frequency, as well as convenience routines for computing
tick spreads.

The loader intentionally performs no strategy specific computations;
it simply returns clean pandas DataFrames ready for analysis.  You can
extend or subclass :class:`DataLoader` to add caching, remote data
fetching or other behaviours without touching the rest of your
backtesting code.

Example usage:

.. code-block:: python

   from new_backtest_engine import DataLoader

   loader = DataLoader(root="D:/fpa_data")
   # Load 1 January through 3 March 2024 for BTCUSDT at 1 second clock
   quotes = loader.load_bookticker("BTCUSDT", "2024-01-01", "2024-03-03", freq="1s")
   trades = loader.load_aggtrades("BTCUSDT", "2024-01-01", "2024-03-03")
   klines = loader.load_klines("BTCUSDT", "2024-01-01", "2024-03-03")

"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class DataLoader:
    """Load historical market data from a local directory structure.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory that contains the ``um_daily/aggTrades``,
        ``um_daily/bookTicker`` and ``klines_1m`` subdirectories.  See
        :meth:`_path_for` for the expected file layout.
    tz : str, optional
        Timezone name to convert timestamps into.  Defaults to
        ``"UTC"``.  All timestamps returned by the loader are tz-aware
        pandas ``Timestamp`` objects.
    """

    def __init__(self, root: str | Path, tz: str = "UTC") -> None:
        self.root = Path(root)
        self.tz = tz

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _path_for(self, dtype: str, symbol: str, ymd: str) -> Path:
        """Construct the absolute path to a daily parquet file.

        The loader assumes the following directory layout under ``root``::

           um_daily/aggTrades/<symbol>/<YYYY-MM-DD>.parquet
           um_daily/bookTicker/<symbol>/<YYYY-MM-DD>.parquet
           klines_1m/<symbol>/<SYMBOL>_<YYYY-MM-DD>_1m.parquet

        Parameters
        ----------
        dtype : {"aggTrades", "bookTicker", "klines_1m"}
            Data type to locate.
        symbol : str
            Trading pair (e.g. ``"BTCUSDT"``).
        ymd : str
            Date in ``YYYY-MM-DD`` format.

        Returns
        -------
        pathlib.Path
            Absolute file path.  The caller should verify existence.
        """
        if dtype == "aggTrades":
            return self.root / "um_daily" / "aggTrades" / symbol / f"{ymd}.parquet"
        if dtype == "bookTicker":
            return self.root / "um_daily" / "bookTicker" / symbol / f"{ymd}.parquet"
        if dtype == "klines_1m":
            return self.root / "klines_1m" / symbol / f"{symbol}_{ymd}_1m.parquet"
        raise ValueError(f"Unknown dtype '{dtype}'")

    def _date_range(self, start_date: str, end_date: str) -> Sequence[str]:
        """Return a list of date strings between ``start_date`` and ``end_date`` inclusive.

        The inputs must be parsable by :func:`pd.to_datetime`.  The
        output strings are formatted as ``YYYY-MM-DD``.
        """
        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()
        if start > end:
            raise ValueError("start_date must be <= end_date")
        dates = pd.date_range(start, end, freq="D")
        return [d.strftime("%Y-%m-%d") for d in dates]

    def _ms_to_ts(self, ms: pd.Series) -> pd.Series:
        """Convert a series of millisecond timestamps to tz-aware timestamps.

        Any NaN values are coerced to 0 before conversion.  The result
        uses the loader's timezone.  Values are converted to int64 to
        avoid pandas issues with object dtype.
        """
        s = pd.to_numeric(ms, errors="coerce").fillna(0).astype("int64")
        ts = pd.to_datetime(s, unit="ms", utc=True)
        return ts.dt.tz_convert(self.tz)

    # ------------------------------------------------------------------
    # Public API: bookTicker
    # ------------------------------------------------------------------
    def load_bookticker(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        freq: Optional[str] = None,
        drop_partial_last_bin: bool = False,
    ) -> pd.DataFrame:
        """Load best bid/ask quotes for a date range.

        When ``freq`` is ``None`` the raw tick‐by‐tick data is
        returned.  Otherwise, quotes are aggregated into bins of width
        ``freq`` (e.g. ``"100ms"``, ``"1s"``, ``"1min"``).  For each
        bin, the minimum and maximum bid/ask prices, the last bid/ask
        price and quantity, the number of ticks and the mid price are
        computed.  Empty bins are dropped.

        The returned DataFrame always contains a ``ts`` column with
        timezone aware timestamps.  Aggregated DataFrames also include
        ``best_bid_price``, ``best_ask_price`` and ``price`` for ease
        of access.

        Parameters
        ----------
        symbol : str
            Trading pair to load.
        start_date, end_date : str
            Date range (inclusive).  Strings must be interpretable by
            :func:`pandas.to_datetime`.
        freq : str or None, optional
            Resampling frequency.  If ``None`` (the default) the raw
            tick data is returned.  Otherwise the string is passed
            directly to :func:`pandas.Grouper`.
        drop_partial_last_bin : bool, optional
            Whether to drop the last bin when it may contain only a
            handful of ticks (which can distort indicator values).  This
            mirrors the behaviour of the earlier backtester.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by timestamp with bid/ask columns and
            optional aggregated statistics.
        """
        dfs: list[pd.DataFrame] = []
        for ymd in self._date_range(start_date, end_date):
            fp = self._path_for("bookTicker", symbol, ymd)
            if not fp.exists():
                # silently skip missing days to simplify experiments
                continue
            df = pd.read_parquet(fp)
            # Coerce numeric types and drop invalid quotes
            for col in ["update_id", "transaction_time"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            for col in ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df = df[(df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)]
            df = df[df["best_bid_price"] <= df["best_ask_price"]]
            # timestamp column
            df["ts"] = self._ms_to_ts(df["transaction_time"])
            df["symbol"] = symbol
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        df_all = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)

        if freq is None:
            df_all["price"] = 0.5 * (df_all["best_bid_price"] + df_all["best_ask_price"])
            return df_all

        # aggregated quotes
        dfi = df_all.set_index("ts", drop=False)
        g = dfi.groupby(pd.Grouper(key="ts", freq=freq))
        agg = g.agg(
            best_bid_price_min=("best_bid_price", "min"),
            best_bid_price_max=("best_bid_price", "max"),
            best_ask_price_min=("best_ask_price", "min"),
            best_ask_price_max=("best_ask_price", "max"),
            best_bid_price_last=("best_bid_price", "last"),
            best_ask_price_last=("best_ask_price", "last"),
            best_bid_qty=("best_bid_qty", "last"),
            best_ask_qty=("best_ask_qty", "last"),
            n_ticks=("update_id", "count"),
            transaction_time_last=("transaction_time", "last"),
            update_id_last=("update_id", "last"),
        )
        # Drop bins with no ticks
        agg = agg.dropna(subset=["best_bid_price_last", "best_ask_price_last"]).reset_index()
        if drop_partial_last_bin and len(agg) > 1:
            agg = agg.iloc[:-1].copy()
        # Mid price and convenience columns
        agg["price"] = 0.5 * (agg["best_bid_price_last"] + agg["best_ask_price_last"])
        agg["best_bid_price"] = agg["best_bid_price_last"]
        agg["best_ask_price"] = agg["best_ask_price_last"]
        agg["symbol"] = symbol
        return agg.sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API: klines
    # ------------------------------------------------------------------
    def load_klines(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load minute candlesticks for a date range."""
        dfs: list[pd.DataFrame] = []
        for ymd in self._date_range(start_date, end_date):
            fp = self._path_for("klines_1m", symbol, ymd)
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            required = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"[klines_1m] missing columns: {missing} in {fp}")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            for col in ["open_time_ms", "close_time_ms"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            df["open_ts"] = self._ms_to_ts(df["open_time_ms"])
            df["close_ts"] = self._ms_to_ts(df["close_time_ms"])
            df["symbol"] = symbol
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        klines = pd.concat(dfs, ignore_index=True).sort_values("open_ts").reset_index(drop=True)
        return klines

    # ------------------------------------------------------------------
    # Public API: aggTrades
    # ------------------------------------------------------------------
    def load_aggtrades(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load aggregated trade data for a date range."""
        dfs: list[pd.DataFrame] = []
        for ymd in self._date_range(start_date, end_date):
            fp = self._path_for("aggTrades", symbol, ymd)
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            required = ["agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"[aggTrades] missing columns: {missing} in {fp}")
            for col in ["agg_trade_id", "transact_time", "first_trade_id", "last_trade_id"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            for col in ["price", "quantity"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df["ts"] = self._ms_to_ts(df["transact_time"])
            if df["is_buyer_maker"].dtype == "object":
                df["is_buyer_maker"] = (
                    df["is_buyer_maker"].astype(str).str.upper().isin(["TRUE", "1", "T", "Y"])
                )
            df["symbol"] = symbol
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        trades = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
        return trades

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def compute_spread(self, quotes: pd.DataFrame) -> pd.Series:
        """Compute the instantaneous bid/ask spread."""
        if quotes.empty:
            return pd.Series(dtype="float64")
        if "best_bid_price" not in quotes.columns or "best_ask_price" not in quotes.columns:
            raise ValueError("DataFrame must contain 'best_bid_price' and 'best_ask_price' columns")
        spread = quotes["best_ask_price"] - quotes["best_bid_price"]
        return spread

    def load_bookticker_one_day(
        self,
        symbol: str,
        ymd: str,
        *,
        freq: Optional[str] = "1s",
        drop_partial_last_bin: bool = True,
    ) -> pd.DataFrame:
        """Load a single day of bookTicker data and optionally resample.

        This helper avoids reading multiple days of data into memory when
        backtesting on a daily basis.  Only a subset of columns are
        retained and numeric types are downcast to reduce memory.
        """
        fp = self._path_for("bookTicker", symbol, ymd)
        if not fp.exists():
            return pd.DataFrame()
        df = pd.read_parquet(fp)
        # Keep only the required columns
        keep_cols = [
            "transaction_time",
            "best_bid_price",
            "best_ask_price",
            "best_bid_qty",
            "best_ask_qty",
        ]
        df = df[keep_cols]
        df["transaction_time"] = pd.to_numeric(df["transaction_time"], errors="coerce").astype("int64")
        for c in keep_cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df["ts"] = self._ms_to_ts(df["transaction_time"]).dt.tz_convert(None)
        if freq is None:
            df["price"] = 0.5 * (df["best_bid_price"] + df["best_ask_price"])
            return df.sort_values("ts").reset_index(drop=True)
        # resample
        dfi = df.set_index("ts", drop=False)
        g = dfi.groupby(pd.Grouper(key="ts", freq=freq))
        agg = g.agg(
            best_bid_price=("best_bid_price", "last"),
            best_ask_price=("best_ask_price", "last"),
            best_bid_qty=("best_bid_qty", "last"),
            best_ask_qty=("best_ask_qty", "last"),
        ).dropna().reset_index()
        agg["price"] = 0.5 * (agg["best_bid_price"] + agg["best_ask_price"])
        return agg.sort_values("ts").reset_index(drop=True)