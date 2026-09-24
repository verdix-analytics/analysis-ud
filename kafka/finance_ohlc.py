from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Import symbol loader from asset_symbols module (repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_symbols import get_stock_symbols, get_crypto_symbols, get_symbol_metadata, get_forex_symbols, get_commodity_symbols

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1h"
ONE_YEAR_PERIOD = "1y"
ONE_DAY_INTERVAL = "1d"
FIVE_MINUTE_INTERVAL = "5m"
ONE_MINUTE_FALLBACK_PERIOD = "1d"

# Load FIXED_SYMBOLS from assets.json
FIXED_SYMBOLS = get_stock_symbols() + get_crypto_symbols() + get_forex_symbols() + get_commodity_symbols()


OUTPUT_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "name",
    "asset_class",
    "exchange",
]

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _extract_symbol_frame(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    if isinstance(raw_df.columns, pd.MultiIndex):
        # New yfinance (≥0.2.x): MultiIndex is (PriceType, Ticker) — extract by ticker at level 1.
        # Old yfinance: MultiIndex is (Ticker, PriceType) — extract by ticker at level 0.
        try:
            symbol_df = raw_df.xs(symbol, level=1, axis=1).copy()
        except (KeyError, TypeError):
            try:
                symbol_df = raw_df[symbol].copy()
            except (KeyError, TypeError):
                return pd.DataFrame()
    else:
        # Single-symbol downloads can be returned in flat column format.
        symbol_df = raw_df.copy()

    if symbol_df.empty:
        return pd.DataFrame()

    symbol_df = symbol_df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open", "high", "low", "close"]
    if not all(column in symbol_df.columns for column in required):
        return pd.DataFrame()

    keep = [column for column in ["open", "high", "low", "close", "volume"] if column in symbol_df.columns]
    symbol_df = symbol_df[keep].dropna(subset=["open", "high", "low", "close"], how="any")

    if symbol_df.empty:
        return pd.DataFrame()

    symbol_df = symbol_df.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
    symbol_df["timestamp"] = pd.to_datetime(symbol_df["timestamp"], utc=True, errors="coerce")
    symbol_df = symbol_df.dropna(subset=["timestamp"])
    symbol_df["symbol"] = symbol
    symbol_df["volume"] = pd.to_numeric(symbol_df.get("volume"), errors="coerce")

    # Lookup metadata for the symbol (search across asset classes)
    metadata = get_symbol_metadata(symbol)
    symbol_df["name"] = metadata.get("name", symbol)
    symbol_df["asset_class"] = metadata.get("asset_class", "unknown")
    symbol_df["exchange"] = metadata.get("exchange", "UNKNOWN")

    return symbol_df[OUTPUT_COLUMNS]


def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    if not symbols:
        return FIXED_SYMBOLS

    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(normalized))


def fetch_ohlc_1h_5d(symbols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Fetch 1h OHLC for the past 5d for the fixed symbol set.

    Symbols: AAPL, MSFT, ORCL, ADBE, AMZN.
    Returns a normalized long DataFrame with one row per symbol/timestamp and
    required keys: symbol, timestamp, open, high, low, close, volume,
    name, asset_class, exchange.
    """
    return _fetch_ohlc(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL, symbols=symbols)


def fetch_ohlc_1d_1y(symbols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Fetch 1d OHLC for the past 1y for the fixed symbol set.

    Symbols: AAPL, MSFT, ORCL, ADBE, AMZN.
    Returns the same normalized schema used by fetch_ohlc_1h_5d().
    """
    return _fetch_ohlc(period=ONE_YEAR_PERIOD, interval=ONE_DAY_INTERVAL, symbols=symbols)


def fetch_ohlc_1m_1h(symbols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Fetch 1m OHLC for the full current trading day for the fixed symbol set.

    Symbols: AAPL, MSFT, ORCL, ADBE, AMZN.
    Returns the same normalized schema used by fetch_ohlc_1h_5d().
    """
    return _fetch_ohlc(period=ONE_MINUTE_FALLBACK_PERIOD, interval=FIVE_MINUTE_INTERVAL, symbols=symbols)


def _fetch_ohlc(period: str, interval: str, symbols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    symbol_list = _normalize_symbols(symbols)
    logger.debug("Downloading OHLC data period=%s interval=%s symbols=%s", period, interval, symbol_list)

    # Download each symbol individually to prevent yfinance MultiIndex cross-ticker contamination
    frames = []
    for symbol in symbol_list:
        raw_df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        frame = _extract_symbol_frame(raw_df, symbol)
        if not frame.empty:
            frames.append(frame)
        else:
            logger.warning("No OHLC data returned for symbol=%s period=%s interval=%s", symbol, period, interval)

    if not frames:
        logger.warning("No OHLC data returned for any symbol period=%s interval=%s", period, interval)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Drop candles whose close deviates >50% from that symbol's batch median.
    # This kills cross-ticker yfinance contamination (e.g. $99 rows in an AAPL ~$260 batch)
    # without any DB dependency — it works purely on the freshly fetched data.
    result = _filter_price_outliers(result)

    # Strip pre/post-market candles for hourly data (keep 13:30–20:00 UTC = 9:30 AM–4:00 PM ET)
    if interval == "1h":
        ts = pd.to_datetime(result["timestamp"], utc=True)
        market_open = pd.Timestamp("00:00", tz="UTC").time().replace(hour=13, minute=30)
        market_close = pd.Timestamp("00:00", tz="UTC").time().replace(hour=20, minute=0)
        result = result[ts.dt.time.between(market_open, market_close)].reset_index(drop=True)

    logger.info(
        "Fetched %s OHLC rows for %s symbol(s) at interval=%s period=%s",
        len(result),
        result["symbol"].nunique(),
        interval,
        period,
    )
    return result[OUTPUT_COLUMNS]


def _filter_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop candles where close deviates more than 50% from the symbol's median close.

    Catches cross-ticker yfinance contamination (e.g. $99 rows for AAPL at ~$260)
    without needing hardcoded per-symbol price ranges.
    """
    if df.empty:
        return df

    clean_frames = []
    for symbol, group in df.groupby("symbol"):
        median_close = group["close"].median()
        if median_close <= 0:
            continue
        lower = median_close * 0.5
        upper = median_close * 2.0
        clean = group[(group["close"] >= lower) & (group["close"] <= upper)]
        dropped = len(group) - len(clean)
        if dropped:
            logger.warning(
                "Dropped %d outlier candle(s) for %s (median=%.2f bounds=[%.2f, %.2f])",
                dropped,
                symbol,
                median_close,
                lower,
                upper,
            )
        clean_frames.append(clean)

    if not clean_frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(clean_frames, ignore_index=True)


def to_payload_records(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert normalized DataFrame to JSON-ready records with stable keys."""
    if df.empty:
        return []

    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        timestamp = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        volume = None if pd.isna(row["volume"]) else int(row["volume"])
        records.append(
            {
                "symbol": row["symbol"],
                "timestamp": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": volume,
                "name": row["name"],
                "asset_class": row["asset_class"],
                "exchange": row["exchange"],
            }
        )
    return records


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Finance 1h OHLC data for the last 5 days for AAPL, MSFT, ORCL, ADBE, AMZN."
    )
    parser.add_argument("--head", type=int, default=10, help="Rows to print from the result")
    return parser


def main() -> None:
    _configure_logging()
    parser = _build_cli_parser()
    args = parser.parse_args()

    logger.info("Running OHLC fetch preview")
    df = fetch_ohlc_1h_5d()
    if df.empty:
        logger.warning("No OHLC data returned.")
        return

    records = to_payload_records(df)
    preview = records[: args.head]
    print(json.dumps(preview, indent=2))
    print(f"\nFetched {len(records)} rows for {df['symbol'].nunique()} symbol(s).")
    logger.info("Previewed %s rows for %s symbol(s)", len(records), df["symbol"].nunique())


if __name__ == "__main__":
    main()


