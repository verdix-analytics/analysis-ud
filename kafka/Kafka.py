import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

TOPIC = "financial-prices"
SYMBOLS = os.getenv("TD_SYMBOLS", "AAPL").split(",")
POLL_SECONDS = int(os.getenv("TD_POLL_SECONDS", "60"))

producer = Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")})


def fetch_1m_data(symbol, start_time, end_time):
    params = {
        "symbol": symbol,
        "interval": "1min",
        "apikey": API_KEY,
        "start_date": start_time,
        "end_date": end_time,
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "values" not in data:
        print("API error:", data)
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def aggregate_to_1h(df):
    if df.empty:
        return df
    df = df.set_index("datetime")
    ohlcv = df.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    ohlcv = ohlcv.dropna().reset_index()
    return ohlcv


def publish_1h(symbol, df_1h):
    for _, row in df_1h.iterrows():
        message = {
            "symbol": symbol,
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        producer.produce(
            TOPIC,
            key=symbol,
            value=json.dumps(message),
        )
    producer.flush(10)


while True:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    for symbol in SYMBOLS:
        symbol = symbol.strip()
        if not symbol:
            continue
        df_1m = fetch_1m_data(symbol, start_str, end_str)
        df_1h = aggregate_to_1h(df_1m)
        publish_1h(symbol, df_1h)

    time.sleep(POLL_SECONDS)
