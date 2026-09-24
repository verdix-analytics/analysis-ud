import json
import logging
import os
import threading
import time
from typing import Any

from confluent_kafka import Consumer
from dotenv import load_dotenv

from kafka.finance_ohlc_producer import (
    publish_records_all_timeframes_for_symbol,
    publish_records_1m,
    publish_records_1h,
    publish_records_1y,
)
from kafka.producer2 import trigger_analysis_for_symbol

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRIGGER_TOPIC = os.getenv("KAFKA_TRIGGER_TOPIC", "run-external-stock-analysis")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
GROUP_ID = os.getenv("KAFKA_TRIGGER_GROUP_ID", "external-stock-analysis-runner")

ANALYSIS_INTERVAL_SECONDS = int(os.getenv("EXTERNAL_ANALYSIS_INTERVAL_SECONDS", "30"))
POLL_TICK_SECONDS = float(os.getenv("EXTERNAL_SCHEDULER_TICK_SECONDS", "1"))

RUN_IMMEDIATELY_ON_TRIGGER = True  # keep simple

# -------------------------
# Utils
# -------------------------
def _decode_message(raw_value: bytes | None) -> dict[str, Any] | None:
    if not raw_value:
        return None

    try:
        payload = json.loads(raw_value.decode("utf-8"))
    except Exception:
        return None

    ticker = payload.get("ticker")
    if not ticker:
        return None

    return payload


def build_consumer() -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,  # simplify
        }
    )
    consumer.subscribe([TRIGGER_TOPIC])
    return consumer


# -------------------------
# Core Logic
# -------------------------
def _publish_initial_full(ticker: str):
    """Publish all timeframes (1m, 1h, 1y) for a new ticker."""
    try:
        logger.info("Publishing initial OHLC data for %s", ticker)
        publish_counts = publish_records_all_timeframes_for_symbol(ticker)
        logger.info(
            "Published OHLC for '%s' (initial): 1m=%s, 1h=%s, 1y=%s",
            ticker,
            publish_counts.get("1m"),
            publish_counts.get("1h"),
            publish_counts.get("1y"),
        )
    except Exception as e:
        logger.exception("Error publishing initial OHLC for %s: %s", ticker, e)


def _run_initial_full_analysis(ticker: str):
    """Publish all timeframes then run analysis for a new ticker.

    Publish must complete before analysis starts: both call yf.download for the
    same symbol, and yfinance's internal session is not thread-safe. Running them
    concurrently causes the wrong interval's data to be returned (e.g. 1h data
    published to financial-1y instead of 1d data).
    """
    try:
        logger.info("Running initial full analysis for %s", ticker)

        publish_thread = threading.Thread(
            target=_publish_initial_full,
            args=(ticker,),
            daemon=True,
        )
        publish_thread.start()
        publish_thread.join()  # wait for all yfinance downloads to finish before analysis

        analysis_thread = threading.Thread(
            target=trigger_analysis_for_symbol,
            args=(ticker,),
            daemon=True,
        )
        analysis_thread.start()

        logger.info("Initial OHLC publish done, analysis triggered for %s", ticker)

    except Exception as e:
        logger.exception("Error running initial analysis for %s: %s", ticker, e)


def _publish_1m_only(ticker: str):
    """Publish only the 1m data."""
    try:
        logger.info("Publishing 1m data for %s", ticker)
        count = publish_records_1m(symbols=[ticker])
        logger.info("Published %s 1m records for %s", count, ticker)
    except Exception as e:
        logger.exception("Error publishing 1m for %s: %s", ticker, e)


def _run_1m_only(ticker: str):
    """Run 1m publish then analysis sequentially to avoid yfinance race condition."""
    try:
        logger.info("Running 1m publish and analysis for %s", ticker)

        publish_thread = threading.Thread(
            target=_publish_1m_only,
            args=(ticker,),
            daemon=True,
        )
        analysis_thread = threading.Thread(
            target=trigger_analysis_for_symbol,
            args=(ticker,),
            daemon=True,
        )

        publish_thread.start()
        publish_thread.join()  # wait for yfinance download to finish before analysis
        analysis_thread.start()

        logger.info("1m publish done, analysis triggered for %s", ticker)

    except Exception as e:
        logger.exception("Error running 1m analysis for %s: %s", ticker, e)


def _publish_hourly(ticker: str):
    """Publish 1h and 1y data."""
    try:
        logger.info("Publishing hourly (1h+1y) data for %s", ticker)
        count_1h = publish_records_1h(symbols=[ticker])
        count_1y = publish_records_1y(symbols=[ticker])
        logger.info("Published for %s: 1h=%s, 1y=%s", ticker, count_1h, count_1y)
    except Exception as e:
        logger.exception("Error publishing hourly data for %s: %s", ticker, e)


def _run_hourly(ticker: str):
    """Run hourly (1h+1y) publish then analysis sequentially to avoid yfinance race condition."""
    try:
        logger.info("Running hourly (1h+1y) publish and analysis for %s", ticker)

        publish_thread = threading.Thread(
            target=_publish_hourly,
            args=(ticker,),
            daemon=True,
        )
        analysis_thread = threading.Thread(
            target=trigger_analysis_for_symbol,
            args=(ticker,),
            daemon=True,
        )

        publish_thread.start()
        publish_thread.join()  # wait for yfinance downloads to finish before analysis
        analysis_thread.start()

        logger.info("Hourly publish done, analysis triggered for %s", ticker)

    except Exception as e:
        logger.exception("Error running hourly analysis for %s: %s", ticker, e)


def scheduler_loop(scheduled_tickers: dict[str, dict], lock: threading.Lock):
    """Scheduler loop now supports per-ticker state and separate timeframes.

    scheduled_tickers maps ticker -> {
        'initial_run_at': float,
        'initial_done': bool,
        'last_1m': float,
        'last_1h': float,
        'last_1y': float,
    }
    """
    HOUR = 3600
    MINUTE = 60

    while True:
        now = time.time()

        with lock:
            for ticker, info in list(scheduled_tickers.items()):
                logger.debug(f"[SCHED CHECK] {ticker} -> now={now}, info={info}")

                # Initial full publish for new tickers
                if not info.get("initial_done", False):
                    initial_at = info.get("initial_run_at", 0)
                    if now >= initial_at:
                        logger.info("Initial time reached for %s", ticker)
                        # mark as done and set last run times to now
                        info["initial_done"] = True
                        info["last_1m"] = now
                        info["last_1h"] = now
                        info["last_1y"] = now

                        threading.Thread(
                            target=_run_initial_full_analysis,
                            args=(ticker,),
                            daemon=True,
                        ).start()

                    # skip further checks until initial run is done
                    continue

                # Subsequent periodic runs
                # 1m every minute
                if now >= info.get("last_1m", 0) + MINUTE:
                    logger.info("1m time reached for %s", ticker)
                    info["last_1m"] = now
                    threading.Thread(target=_run_1m_only, args=(ticker,), daemon=True).start()

                # 1h and 1y every hour (run together)
                if now >= info.get("last_1h", 0) + HOUR:
                    logger.info("Hourly time reached for %s", ticker)
                    info["last_1h"] = now
                    info["last_1y"] = now
                    threading.Thread(target=_run_hourly, args=(ticker,), daemon=True).start()

        time.sleep(POLL_TICK_SECONDS)


# -------------------------
# Main
# -------------------------
def main():
    consumer = build_consumer()

    scheduled_tickers: dict[str, dict] = {}
    lock = threading.Lock()

    # start scheduler thread
    threading.Thread(
        target=scheduler_loop,
        args=(scheduled_tickers, lock),
        daemon=True,
    ).start()

    logger.info("Listening for Kafka messages...")

    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            continue

        payload = _decode_message(msg.value())
        if not payload:
            logger.warning(f"Invalid message: {msg.value()}")
            continue

        ticker = payload["ticker"].strip().upper()

        with lock:
            next_run = time.time() if RUN_IMMEDIATELY_ON_TRIGGER else time.time() + ANALYSIS_INTERVAL_SECONDS

            if ticker not in scheduled_tickers:
                # Schedule initial full run for new ticker
                scheduled_tickers[ticker] = {
                    "initial_run_at": next_run,
                    "initial_done": False,
                    "last_1m": 0.0,
                    "last_1h": 0.0,
                    "last_1y": 0.0,
                }
            else:
                info = scheduled_tickers[ticker]
                if not info.get("initial_done", False):
                    # reschedule initial run
                    info["initial_run_at"] = next_run
                else:
                    # for already-initialized tickers, trigger an immediate 1m run
                    # by setting last_1m to the past so scheduler picks it up
                    info["last_1m"] = time.time() - 60 if RUN_IMMEDIATELY_ON_TRIGGER else info.get("last_1m", 0)

        logger.info("Received ticker: %s -> scheduled", ticker)



if __name__ == "__main__":
    main()
