from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer
from dotenv import load_dotenv

from kafka.finance_ohlc import fetch_ohlc_1d_1y, fetch_ohlc_1h_5d, fetch_ohlc_1m_1h, to_payload_records

load_dotenv()

TOPIC_1H = os.getenv("KAFKA_TOPIC_1H", "financial-1h")
TOPIC_1Y = os.getenv("KAFKA_TOPIC_1Y", "financial-1y")
TOPIC_1M = os.getenv("KAFKA_TOPIC_1M", "financial-1m")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _delivery_report(err: Any, msg: Any) -> None:
    if err is not None:
        logger.error("Delivery failed for key=%r topic=%s: %s", msg.key(), msg.topic(), err)
    else:
        logger.debug("Delivered key=%r to %s[%d]@%d", msg.key(), msg.topic(), msg.partition(), msg.offset())


def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if symbols is None:
        return None

    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    return list(dict.fromkeys(normalized)) or None


def _parse_symbols_arg(symbols_arg: str | None) -> list[str] | None:
    if not symbols_arg:
        return None

    return _normalize_symbols([symbol.strip() for symbol in symbols_arg.split(",")])


def publish_records_1h(topic: str = TOPIC_1H, symbols: list[str] | tuple[str, ...] | None = None) -> int:
    """Fetch 1h/5d OHLC rows and publish each payload record to Kafka."""
    df = fetch_ohlc_1h_5d(symbols=_normalize_symbols(symbols))
    records = to_payload_records(df)

    if not records:
        logger.info("No 1h records to publish for topic=%s.", topic)
        return 0

    logger.info("Publishing %s 1h records to topic=%s", len(records), topic)
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    produced_at = datetime.now(timezone.utc).isoformat()

    for record in records:
        key = record["symbol"]
        producer.produce(
            topic,
            key=str(key),
            value=json.dumps({**record, "produced_at": produced_at}),
            callback=_delivery_report,
        )
        producer.poll(0)

    remaining = producer.flush(60)
    if remaining:
        logger.warning("%d 1h message(s) not delivered to topic=%s after flush timeout", remaining, topic)
    return len(records) - remaining


def publish_records_1y(topic: str = TOPIC_1Y, symbols: list[str] | tuple[str, ...] | None = None) -> int:
    """Fetch 1d/1y OHLC rows and publish each payload record to Kafka."""
    df = fetch_ohlc_1d_1y(symbols=_normalize_symbols(symbols))
    records = to_payload_records(df)

    if not records:
        logger.info("No 1y records to publish for topic=%s.", topic)
        return 0

    logger.info("Publishing %s 1y records to topic=%s", len(records), topic)
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    produced_at = datetime.now(timezone.utc).isoformat()

    for record in records:
        key = record["symbol"]
        producer.produce(
            topic,
            key=str(key),
            value=json.dumps({**record, "produced_at": produced_at}),
            callback=_delivery_report,
        )
        producer.poll(0)

    remaining = producer.flush(60)
    if remaining:
        logger.warning("%d 1y message(s) not delivered to topic=%s after flush timeout", remaining, topic)
    return len(records) - remaining


def publish_records_1m(topic: str = TOPIC_1M, symbols: list[str] | tuple[str, ...] | None = None) -> int:
    """Fetch 1m/1h OHLC rows and publish each payload record to Kafka."""
    df = fetch_ohlc_1m_1h(symbols=_normalize_symbols(symbols))
    records = to_payload_records(df)

    if not records:
        logger.info("No 1m records to publish for topic=%s.", topic)
        return 0

    logger.info("Publishing %s 1m records to topic=%s", len(records), topic)
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    produced_at = datetime.now(timezone.utc).isoformat()

    for record in records:
        key = record["symbol"]
        producer.produce(
            topic,
            key=str(key),
            value=json.dumps({**record, "produced_at": produced_at}),
            callback=_delivery_report,
        )
        producer.poll(0)

    remaining = producer.flush(60)
    if remaining:
        logger.warning("%d 1m message(s) not delivered to topic=%s after flush timeout", remaining, topic)
    return len(records) - remaining


def publish_records(topic: str = TOPIC_1H) -> int:
    """Backward-compatible wrapper for the original 1h publisher."""
    return publish_records_1h(topic=topic)


def publish_records_all_timeframes_for_symbol(symbol: str) -> dict[str, int]:
    ticker = (symbol or "").strip().upper()
    if not ticker:
        return {"1m": 0, "1h": 0, "1y": 0}

    symbols = [ticker]
    return {
        "1h": publish_records_1h(symbols=symbols),
        "1m": publish_records_1m(symbols=symbols),
        "1y": publish_records_1y(symbols=symbols),
    }


def run_continuous_publish_loop(
    symbols: list[str] | tuple[str, ...] | None = None,
    interval_seconds: int = 60,
) -> None:
    normalized_symbols = _normalize_symbols(symbols)
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    logger.info("Starting continuous OHLC publish loop: 1m every %s seconds, 1h and 1y every 3600 seconds.", interval_seconds)

    # Track last execution times for 1h and 1y (run every hour)
    # Initialize to negative value to ensure first run publishes all data
    last_1h_1y_execution = -float('inf')
    hour_interval = 3600  # 1 hour in seconds

    try:
        while True:
            cycle_started = time.monotonic()
            try:
                counts = {
                    "1m": publish_records_1m(symbols=normalized_symbols),
                    "1h": 0,
                    "1y": 0,
                }

                # Run 1h and 1y only if an hour has passed since last execution
                # On first run, this condition will always be true
                if cycle_started - last_1h_1y_execution >= hour_interval:
                    counts["1h"] = publish_records_1h(symbols=normalized_symbols)
                    counts["1y"] = publish_records_1y(symbols=normalized_symbols)
                    last_1h_1y_execution = cycle_started
                    logger.info(
                        "Completed continuous publish cycle: 1m=%s, 1h=%s, 1y=%s",
                        counts["1m"],
                        counts["1h"],
                        counts["1y"],
                    )
                else:
                    logger.info(
                        "Completed continuous publish cycle: 1m=%s (1h and 1y skipped, next run in %d seconds)",
                        counts["1m"],
                        int(hour_interval - (cycle_started - last_1h_1y_execution)),
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception("Continuous publish cycle failed: %s", exc)

            elapsed = time.monotonic() - cycle_started
            sleep_for = max(0.0, interval_seconds - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        logger.info("Stopping continuous OHLC publish loop.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Yahoo Finance OHLC records to Kafka.")
    parser.add_argument(
        "--mode",
        choices=["1m", "1h", "1y"],
        default="1h",
        help="One-shot mode only: 1m=interval 1m period 1h, 1h=interval 1h period 5d, 1y=interval 1d period 1y",
    )
    parser.add_argument("--topic", default=None, help="Kafka topic override")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol filter (default: built-in symbol set)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single publish cycle instead of the default continuous loop",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Delay between continuous publish cycles",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and print sample records without publishing")
    parser.add_argument("--head", type=int, default=3, help="Records to print in dry-run mode")
    return parser


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    symbols = _parse_symbols_arg(args.symbols)

    if not args.once:
        if args.dry_run:
            preview = {
                "1m": to_payload_records(fetch_ohlc_1m_1h(symbols=symbols))[: args.head],
                "1h": to_payload_records(fetch_ohlc_1h_5d(symbols=symbols))[: args.head],
                "1y": to_payload_records(fetch_ohlc_1d_1y(symbols=symbols))[: args.head],
            }
            print(json.dumps(preview, indent=2))
            print("\nDry-run complete for continuous mode. Use --once for a single publish cycle.")
            logger.info("Completed continuous-mode dry run for symbols=%s", symbols or "default universe")
            return

        run_continuous_publish_loop(symbols=symbols, interval_seconds=args.interval_seconds)
        return

    if args.mode == "1m":
        df = fetch_ohlc_1m_1h(symbols=symbols)
        topic = args.topic or TOPIC_1M
    elif args.mode == "1y":
        df = fetch_ohlc_1d_1y(symbols=symbols)
        topic = args.topic or TOPIC_1Y
    else:
        df = fetch_ohlc_1h_5d(symbols=symbols)
        topic = args.topic or TOPIC_1H

    records = to_payload_records(df)

    if args.dry_run:
        print(json.dumps(records[: args.head], indent=2))
        print(f"\nPrepared {len(records)} records for topic '{topic}'.")
        logger.info("Prepared %s records for topic=%s in dry-run mode", len(records), topic)
        return

    if not records:
        logger.warning("No records fetched; nothing published for topic=%s.", topic)
        return

    logger.info("Publishing %s records to topic=%s", len(records), topic)

    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    for record in records:
        producer.produce(
            topic,
            key=str(record["symbol"]),
            value=json.dumps(record),
            callback=_delivery_report,
        )
        producer.poll(0)

    producer.flush(15)
    logger.info("Published %s records to topic=%s", len(records), topic)


if __name__ == "__main__":
    main()


