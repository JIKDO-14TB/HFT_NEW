import numpy as np
import pandas as pd
from new_data_loader import DataLoader

def check_tick_spread(
    *,
    root: str,
    symbol: str,
    ymd: str,
):
    loader = DataLoader(root, tz="UTC")

    book = loader.load_bookticker_one_day(
        symbol=symbol,
        ymd=ymd,
        freq=None
    )

    if book is None or book.empty:
        print("No bookTicker data")
        return None

    # =========================
    # 시간 컬럼 자동 선택
    # =========================
    if "ts_ms" in book.columns:
        book["time"] = pd.to_datetime(book["ts_ms"], unit="ms", utc=True)
    elif "event_time" in book.columns:
        book["time"] = pd.to_datetime(book["event_time"], unit="ms", utc=True)
    elif "transaction_time" in book.columns:
        book["time"] = pd.to_datetime(book["transaction_time"], unit="ms", utc=True)
    else:
        raise ValueError("No usable timestamp column found")

    # 시간 정렬
    book = book.sort_values("time").reset_index(drop=True)

    # =========================
    # Spread 계산
    # =========================
    book["mid"] = (book["best_bid_price"] + book["best_ask_price"]) / 2
    book["spread"] = book["best_ask_price"] - book["best_bid_price"]
    book["spread_bp"] = book["spread"] / book["mid"] * 1e4

    print(f"\n=== {symbol} | {ymd} ===")
    print(book[[
        "time",
        "best_bid_price",
        "best_ask_price",
        "spread",
        "spread_bp"
    ]].head(10))

    print("\nSpread bp summary:")
    print(book["spread_bp"].describe())

    print("\nUnique spread (price units):")
    print(np.sort(book["spread"].unique())[:10])
    print(f"... total unique spreads: {book['spread'].nunique()}")

    print("\nUnique spread (bp):")
    print(np.round(np.sort(book["spread_bp"].unique())[:10], 4))
    print(f"... total unique spreads: {book['spread_bp'].nunique()}")
    print("\nSpread summary:")
    print(f"Min spread (price) : {book['spread'].min()}")
    print(f"Max spread (price) : {book['spread'].max()}")
    print(f"Mean spread (price): {book['spread'].mean()}")

    print(f"Min spread (bp) : {book['spread_bp'].min():.4f}")
    print(f"Max spread (bp) : {book['spread_bp'].max():.4f}")
    print(f"Mean spread (bp): {book['spread_bp'].mean():.4f}")
    return book