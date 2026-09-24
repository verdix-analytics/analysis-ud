import asyncio
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import cast
import numpy as np
import pandas as pd
from stock_analysis.context_scores.candlestick_score import candlestick_score
from stock_analysis.context_scores.chart_context_score import context_score
from stock_analysis.context_scores.harmonic_score import harmonic_score
import yfinance as yf
import logging
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_symbols import get_stock_symbols, get_crypto_symbols, get_symbol_metadata, get_forex_symbols, get_commodity_symbols
from kafka.prodHelper import CADENCE, is_due, mark_run

from stock_analysis.engine import (
    harmonic_patterns_exec,
    chart_patterns_exec,
    candlestick_patterns_exec,
)

load_dotenv()

logger = logging.getLogger(__name__)

TOPIC_HARMONIC = "harmonic"
TOPIC_CHART = "chart"
TOPIC_CANDLESTICK = "candlestick"

SYMBOLS = get_stock_symbols() + get_crypto_symbols() + get_forex_symbols() + get_commodity_symbols()
POLL_SECONDS = int(os.getenv("TD_POLL_SECONDS", "60"))
NUM_PARTITIONS = int(os.getenv("KAFKA_NUM_PARTITIONS", "24"))
REPLICATION_FACTOR = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))

MAX_WORKERS = int(os.getenv("PRODUCER_MAX_WORKERS", str(NUM_PARTITIONS)))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

CATEGORY_TOPIC_MAP = {
    "harmonic": TOPIC_HARMONIC,
    "chart": TOPIC_CHART,
    "candlestick": TOPIC_CANDLESTICK,
}

producer = Producer({
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", KAFKA_BOOTSTRAP)
})

admin_client = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})

def ensure_topics():
    """
    Create or validate Kafka topics for pattern analysis.

    Partitioning Strategy:
    - Uses hash partitioning with symbol ID (stock ticker) as the key
    - Guarantees ALL messages for the same instrument go to the same partition
    - Maintains message ordering per instrument across all consumers
    - 24 partitions enables parallel consumption across multiple workers
      while keeping related data together for efficient processing
    """
    existing = admin_client.list_topics(timeout=10).topics
    to_create = []

    for topic in [TOPIC_HARMONIC, TOPIC_CHART, TOPIC_CANDLESTICK]:
        if topic not in existing:
            to_create.append(NewTopic(
                topic,
                num_partitions=NUM_PARTITIONS,
                replication_factor=REPLICATION_FACTOR,
            ))
            print(f"[admin] Topic '{topic}' not found — will create "
                  f"with {NUM_PARTITIONS} partitions.")
        else:
            actual = len(existing[topic].partitions)
            if actual < NUM_PARTITIONS:
                print(f"[admin] WARNING: topic '{topic}' has {actual} partition(s), "
                      f"expected {NUM_PARTITIONS}. Consider increasing manually via "
                      f"kafka-topics.sh --alter.")
            else:
                print(f"[admin] Topic '{topic}' OK ({actual} partition(s)).")

    if to_create:
        results = admin_client.create_topics(to_create)
        for topic, future in results.items():
            try:
                future.result()
                print(f"[admin] Created topic '{topic}'.")
            except KafkaException as e:
                print(f"[admin] Failed to create topic '{topic}': {e}")


# ── Kafka Producer ────────────────────────────────────────────────────────────
producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "linger.ms": 100,
    "batch.num.messages": 200,
    "compression.type": "gzip",
    "queue.buffering.max.messages": 100_000,
    "queue.buffering.max.kbytes": 256_000,
    "retries": 5,
    "retry.backoff.ms": 300,
})


# ── data fetching ─────────────────────────────────────────────────────────────
_yf_download_lock = threading.Lock()
def download_yf_df(symbol: str, period: str, interval: str, max_retries: int = 10) -> pd.DataFrame:
    """Download OHLC for a single symbol, compatible with both old and new yfinance MultiIndex formats.

    We handle both and always produce a flat DataFrame with columns Open/High/Low/Close
    plus a Datetime column. Run the download api until the maximum retries are completed
    or the dataframe has more than one record
    """
    raw_df = pd.DataFrame()
    while True:
        with _yf_download_lock:
            raw_df = yf.download(
                tickers=symbol,
                period = period,
                interval = interval,
                auto_adjust = False,
                progress = False,
            )
        if raw_df.empty:
            max_retries -= 1
        if not raw_df.empty or max_retries <= 0:
            break
        
    if raw_df.empty:
        return raw_df
    if isinstance(raw_df.columns, pd.MultiIndex):
        try:
            flat = raw_df.xs(symbol, level=1, axis=1).copy()
        except (KeyError, TypeError):
            try:
                flat = raw_df[symbol].copy()
            except (KeyError, TypeError):
                logger.error(f"Cannot extract data for {symbol} - unknown MultiIndex indices: {raw_df.columns.tolist()}")
                return pd.DataFrame()
    else:
        flat = raw_df.copy()
        
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in flat.columns for c in required):
        logger.error(f"Missing OHLC columns for {symbol}. Got: {flat.columns.tolist()}")
        return pd.DataFrame()

    flat = flat[required].dropna()
    if flat.empty:
        return pd.DataFrame()

    flat = flat.reset_index()
    # The index column is 'Datetime' for intraday or 'Date' for daily
    time_col = flat.columns[0]
    flat = flat.rename(columns={time_col: "Datetime"})
    flat["Ticker"] = symbol
    logger.info(f"[{symbol}] Fetched {len(flat)} rows (period={period} interval={interval})")
    return flat

def fetch_chart_data(symbol: str) -> pd.DataFrame:
    """1 month of data, 1h candles — for chart patterns."""
    ohlc_df = download_yf_df(symbol, period="1mo", interval="1h")
    if ohlc_df.empty:
        return pd.DataFrame()
    df_ticker: pd.DataFrame = cast(pd.DataFrame, ohlc_df[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True))
    # Insert integer row-index as first column so iloc[:, 0]=int and iloc[:, 1]=Datetime
    df_ticker.insert(0, "row_idx", range(len(df_ticker)))
    return df_ticker
    
def fetch_harmonic_data(symbol: str) -> pd.DataFrame:
    """1 year of data, 1d candles — for harmonic patterns."""
    ohlc_data = download_yf_df(symbol, period='1y', interval='1d')
    if ohlc_data.empty:
        return pd.DataFrame()
    df_ticker: pd.DataFrame = cast(pd.DataFrame, ohlc_data[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True))
    df_ticker.columns = ["Datetime", "open", "high", "low", "close", "volume"]
    return df_ticker

def fetch_candlestick_data(symbol: str) -> pd.DataFrame:
    """1 week of data, 1m candles — for candlestick patterns."""
    ohlc_data = download_yf_df(symbol, period="1d", interval="5m")
    if ohlc_data.empty:
        return pd.DataFrame()
    df_ticker: pd.DataFrame = cast(pd.DataFrame, ohlc_data[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True))
    df_ticker.insert(0, "row_idx", range(len(df_ticker)))
    return df_ticker


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(i) for i in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(obj)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

# making data fetching async
async def fetch_chart_data_async(symbol: str) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_chart_data, symbol)

async def fetch_harmonic_data_async(symbol: str) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_harmonic_data, symbol)

async def fetch_candlestick_data_async(symbol: str) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_candlestick_data, symbol)

def download_batch(symbols: list[str], period: str, interval: str, max_retries: int = 3) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    result: dict[str, pd.DataFrame] = {}
    pending = list(symbols)

    for attempt in range(1, max_retries + 1):
        if not pending:
            break

        logger.info(f"[batch] Attempt {attempt}/{max_retries} — downloading {len(pending)} symbols "
                    f"(period={period}, interval={interval})...")

        try:
            raw_df = yf.download(
                tickers=pending,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
            )
        except Exception as e:
            logger.error(f"[batch] yf.download failed on attempt {attempt}: {e}")
            continue

        still_missing = []
        for symbol in pending:
            try:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    flat = raw_df[symbol].copy()
                else:
                    # Single symbol returned — not a MultiIndex
                    flat = raw_df.copy()

                required = ["Open", "High", "Low", "Close", "Volume"]
                if not all(c in flat.columns for c in required):
                    logger.warning(f"[{symbol}] Missing columns on attempt {attempt}: {flat.columns.tolist()}")
                    still_missing.append(symbol)
                    continue

                flat = flat[required].dropna()
                if flat.empty:
                    logger.warning(f"[{symbol}] Empty after dropna on attempt {attempt}.")
                    still_missing.append(symbol)
                    continue

                flat = flat.reset_index()
                flat = flat.rename(columns={flat.columns[0]: "Datetime"})
                flat["Ticker"] = symbol
                result[symbol] = flat

            except (KeyError, TypeError) as e:
                logger.warning(f"[{symbol}] Could not extract on attempt {attempt}: {e}")
                still_missing.append(symbol)

        if still_missing:
            logger.warning(f"[batch] {len(still_missing)} symbols still missing after attempt {attempt}: {still_missing}")

        pending = still_missing

    if pending:
        logger.error(f"[batch] Gave up on {len(pending)} symbols after {max_retries} attempts: {pending}")
    else:
        logger.info(f"[batch] All {len(symbols)} symbols fetched successfully.")

    return result
# ── publishing ────────────────────────────────────────────────────────────────

def publish_pattern(topic, stock_ticker, pattern_type, context_score, output):
    message = {
        'stock_ticker': stock_ticker,
        'pattern_type': pattern_type,
        'context_score': convert_to_serializable(context_score),
        'result': convert_to_serializable(output),
        'detected_at': convert_to_serializable(datetime.now(timezone.utc)),
    }
    producer.produce(
        topic,
        key=stock_ticker,
        value=json.dumps(message),
    )
    producer.flush(10)


# ── per-category analysis + publish ──────────────────────────────────────────

async def run_and_publish(symbol: str, df:pd.DataFrame, exec_fn, pattern_type: str, score_fn=None):
    """Fetch data, run pattern analysis, publish results to Kafka."""

    if df.empty:
        print(f"[{symbol}] No data returned for {pattern_type}, skipping.")
        return

    logger.info(f"[{symbol}] {len(df)} candles — running {pattern_type} analysis...")
    try:
        results = await asyncio.to_thread(exec_fn, df)
    except Exception as e:
        print(f"[{symbol}] {pattern_type} analysis error: {e}")
        return

    if not results:
        logger.warning(f"[{symbol}] No {pattern_type} patterns detected.")
        return

    topic = CATEGORY_TOPIC_MAP[pattern_type]

    cxt_score = None
    if score_fn is not None:
        try:
            cxt_score = await asyncio.to_thread(score_fn, results)
        except Exception as e:
            logger.error(f"{pattern_type} Scoring Failed - Reason: {e}")

    # Build full-df position -> ISO timestamp lookup
    index_to_time = {}
    total_rows = len(df)
    if pattern_type == "chart":
        try:
            # df has two "Datetime" cols: col[0]=integer row index, col[1]=actual datetime
            dt_series = df.iloc[:, 1]
            for i, ts in enumerate(dt_series):
                try:
                    index_to_time[i] = pd.Timestamp(ts).isoformat()
                except Exception:
                    index_to_time[i] = str(ts)
        except Exception as e:
            logger.warning(f"Could not build index_to_time lookup: {e}")
    
    for result in results:
        signal = result.get("signal", {})
        confidence = int(round(result.get("confidence", 0)))

        # Enrich chart pivots with timestamps
        if pattern_type == "chart" and index_to_time:
            # window_count=1 → last 60 rows, window_count=2 → rows [-120:-60], etc.
            chart_window_size = 60
            window_count = result.get("window_count") or 1
            window_start_idx = total_rows - chart_window_size * window_count

            pivots = signal.get("pivots") or []
            for pivot in pivots:
                if isinstance(pivot, dict) and "index" in pivot and "time" not in pivot:
                    try:
                        full_idx = window_start_idx + int(pivot["index"])
                        pivot["time"] = index_to_time.get(full_idx)
                    except Exception:
                        pass
                    
        if pattern_type == "candlestick" and index_to_time:
            wc = result.get("window_count")
            if wc is not None:
                try:
                    candlestick_window_size = 10
                    last_candle_idx = total_rows - candlestick_window_size * (int(wc) - 1) - 1
                    candle_ts = index_to_time.get(last_candle_idx)
                    if candle_ts:
                        result["detected_at"] = candle_ts
                except Exception:
                    pass
                
        publish_pattern(
            topic=topic,
            stock_ticker=symbol,
            pattern_type=pattern_type,
            context_score=cxt_score,
            output=result
        )
        logger.info(
            f"[{symbol}] Published -> [{topic}] {result.get('pattern')} "
            f"(conf={confidence}, direction={signal.get('direction', 'n/a')})"
            f"(context_score={cxt_score['score'] if cxt_score else 0})"
        )
        
# ── main polling loop ─────────────────────────────────────────────────────────
async def run_cycle():
    due = [k for k in CADENCE if is_due(k)]
    if not due:
        logger.info("Nothing to run")
        return due

    # ── Batch fetch all symbols per pattern type ──────────────────────────────
    chart_dfs: dict[str, pd.DataFrame] = {}
    harmonic_dfs: dict[str, pd.DataFrame] = {}
    candlestick_dfs: dict[str, pd.DataFrame] = {}

    if "chart" in due:
        logger.info("[batch] Fetching chart data for all symbols...")
        chart_dfs = await asyncio.to_thread(
            download_batch, SYMBOLS, "1mo", "1h"
        )
        logger.info(f"[batch] chart: got data for {len(chart_dfs)}/{len(SYMBOLS)} symbols")

    if "harmonic" in due:
        logger.info("[batch] Fetching harmonic data for all symbols...")
        harmonic_dfs = await asyncio.to_thread(
            download_batch, SYMBOLS, "1y", "1d"
        )
        logger.info(f"[batch] harmonic: got data for {len(harmonic_dfs)}/{len(SYMBOLS)} symbols")

    if "candlestick" in due:
        logger.info("[batch] Fetching candlestick data for all symbols...")
        candlestick_dfs = await asyncio.to_thread(
            download_batch, SYMBOLS, "1d", "5m"
        )
        logger.info(f"[batch] candlestick: got data for {len(candlestick_dfs)}/{len(SYMBOLS)} symbols")

    # ── Fan out analysis concurrently (no more fetching inside) ───────────────
    CONCURRENCY = 20
    sem = asyncio.Semaphore(CONCURRENCY)

    async def analyse_symbol(symbol: str):
        async with sem:
            tasks = []

            if "chart" in due and symbol in chart_dfs:
                df = chart_dfs[symbol].copy()
                df.insert(0, "row_idx", range(len(df)))
                tasks.append(run_and_publish(symbol, df, chart_patterns_exec, "chart", context_score))

            if "harmonic" in due and symbol in harmonic_dfs:
                df = harmonic_dfs[symbol].copy()
                df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
                df.columns = ["Datetime", "open", "high", "low", "close", "volume"]
                tasks.append(run_and_publish(symbol, df, harmonic_patterns_exec, "harmonic", harmonic_score))

            if "candlestick" in due and symbol in candlestick_dfs:
                df = candlestick_dfs[symbol].copy()
                df.insert(0, "row_idx", range(len(df)))
                tasks.append(run_and_publish(symbol, df, candlestick_patterns_exec, "candlestick", candlestick_score))

            if tasks:
                await asyncio.gather(*tasks)

    await asyncio.gather(*[analyse_symbol(s) for s in SYMBOLS])
    return due

async def analyse_symbol_on_demand(symbol: str) -> None:
    """
    Entry point for user-triggered analysis of any symbol.
    Always fetches fresh data directly (not from batch cache).
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return

    tasks = []

    df_raw = await fetch_chart_data_async(symbol)
    tasks.append(run_and_publish(symbol, df_raw, chart_patterns_exec, "chart", context_score))

    df_raw = await fetch_harmonic_data_async(symbol)
    tasks.append(run_and_publish(symbol, df_raw, harmonic_patterns_exec, "harmonic", harmonic_score))

    df_raw = await fetch_candlestick_data_async(symbol)
    tasks.append(run_and_publish(symbol, df_raw, candlestick_patterns_exec, "candlestick", candlestick_score))

    if tasks:
        await asyncio.gather(*tasks)
    else:
        logger.warning(f"[{symbol}] No data fetched for on-demand analysis.")
        
def trigger_analysis_for_symbol(symbol: str) -> None:
    """Sync entry point for on-demand user-triggered analysis."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.ensure_future(analyse_symbol_on_demand(symbol))
    else:
        asyncio.run(analyse_symbol_on_demand(symbol))


logger.info(f"Starting poll loop: symbols={SYMBOLS}, interval={POLL_SECONDS}s")

if __name__ == "__main__":
    ensure_topics()

    logger.info(
        f"\n[producer] Starting poll loop\n"
        f"  symbols   = {len(SYMBOLS)} tickers\n"
        f"  interval  = {POLL_SECONDS}s\n"
        f"  partitions= {NUM_PARTITIONS} per topic\n"
        f"  workers   = {MAX_WORKERS} parallel threads\n"
    )
    
    while True:
        cycle_start = time.time()
        print(f"\n[producer] === New cycle at {datetime.now(timezone.utc).isoformat()} ===")   
        
        due = asyncio.run(run_cycle())
                                       
        producer.flush(30)

        elapsed = time.time() - cycle_start
        sleep_time = max(0, POLL_SECONDS - elapsed)
        print(
            f"[producer] Cycle complete in {elapsed:.1f}s. "
            f"Sleeping {sleep_time:.1f}s...\n"
        )
        time.sleep(sleep_time)
        for pattern_type in due:
            mark_run(pattern_type)
