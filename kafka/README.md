# Capstone CS50 - Kafka + Twelve Data

This project streams market data from Twelve Data into Apache Kafka and consumes it locally.

**What it does**
- Producer pulls 1-minute data for the past 24 hours from Twelve Data.
- Aggregates to 1-hour OHLCV.
- Publishes to a single Kafka topic: `financial-prices`.
- Kafka message key is the stock ticker.
- Message value is OHLCV JSON.

## Requirements
- macOS
- Python 3
- Kafka (Homebrew)
- Python packages: `confluent-kafka`, `pandas`, `requests`

## Install

### Kafka
```bash
brew install kafka
brew services start kafka
```

### Python deps
```bash
pip install confluent-kafka pandas requests
```

## Create topic
```bash
/usr/local/opt/kafka/bin/kafka-topics --bootstrap-server localhost:9092 --create \
  --topic financial-prices --partitions 1 --replication-factor 1
```

## Run

### Producer
```bash
TD_SYMBOLS=AAPL,MSFT TD_POLL_SECONDS=60 python Kafka.py
```

### Consumer
```bash
python kafkaconsumer.py
```

## Configuration
- `TD_SYMBOLS`: Comma-separated tickers. Default `AAPL`.
- `TD_POLL_SECONDS`: Poll interval in seconds. Default `60`.

## Files
- `Kafka.py`: Producer that fetches, aggregates, and publishes data.
- `kafkaconsumer.py`: Consumer that prints messages from Kafka.

## Notes
- The topic name is fixed to `financial-prices` in code.
- Twelve Data API key is currently hardcoded in `Kafka.py`.
