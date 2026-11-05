"""Market data for the trading arena: FX from the lab's Nautilus catalog, crypto from ccxt.

  catalog_bars("EURUSD.DERIV")         15-minute OHLC from ~/till-data/catalog (read only), with the median
                                       bid/ask spread per bar from the catalog's quote ticks
  ccxt_bars("coinbase", "BTC/USD")      15-minute OHLC through ccxt, cached as CSV under data/ (not committed)

Nautilus stores prices as 16-byte little-endian fixed-point integers scaled by 1e16 (high-precision mode).
Every frame has columns time (UTC), open, high, low, close, spread (in price units).
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CATALOG = os.path.expanduser(os.environ.get("TILL_CATALOG", "~/till-data/catalog/data"))
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SCALE = 10**16


def _fixed(col):
    return np.array([int.from_bytes(b, "little", signed=True) / SCALE for b in col.to_pylist()])


def catalog_bars(instrument):
    files = sorted(glob.glob(os.path.join(CATALOG, "bar", f"{instrument}-15-MINUTE-LAST-EXTERNAL", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no 15-minute bars for {instrument} under {CATALOG}")
    t = pq.read_table(files[-1])
    df = pd.DataFrame({k: _fixed(t[k]) for k in ("open", "high", "low", "close")})
    df["time"] = pd.to_datetime(t["ts_event"].to_numpy(), unit="ns", utc=True)
    qfiles = sorted(glob.glob(os.path.join(CATALOG, "quote_tick", instrument, "*.parquet")))
    if qfiles:
        q = pq.read_table(qfiles[-1])
        qt = pd.DataFrame(
            {
                "spread": _fixed(q["ask_price"]) - _fixed(q["bid_price"]),
                "time": pd.to_datetime(q["ts_event"].to_numpy(), unit="ns", utc=True),
            }
        )
        # Each bar's quote is stamped a nanosecond before the bar, and bars are not on the quarter hour.
        df = pd.merge_asof(
            df.sort_values("time"),
            qt.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("1min"),
        )
    df["spread"] = df.get("spread", pd.Series(np.nan, index=df.index)).ffill().bfill()
    return df.sort_values("time").reset_index(drop=True)[["time", "open", "high", "low", "close", "spread"]]


def ccxt_bars(exchange, symbol, days=90, spread_frac=0.0001):
    """15-minute bars, paged through ccxt and cached. Crypto spread is not in OHLCV: a fixed fraction of price."""
    import ccxt

    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{exchange}_{symbol.replace('/', '')}_15m_{days}d.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        df = pd.read_csv(path, parse_dates=["time"])
    else:
        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        since = ex.milliseconds() - days * 86400 * 1000
        rows = []
        while since < ex.milliseconds() - 15 * 60 * 1000:
            batch = ex.fetch_ohlcv(symbol, "15m", since=since, limit=300)
            if not batch:
                break
            rows += batch
            since = batch[-1][0] + 15 * 60 * 1000
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["time", "open", "high", "low", "close"]]
        df.to_csv(path, index=False)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["spread"] = df["close"] * spread_frac
    return df.sort_values("time").reset_index(drop=True)
