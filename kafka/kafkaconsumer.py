from confluent_kafka import Consumer, KafkaException
from dotenv import load_dotenv

import psycopg2
import json
from collections import deque, defaultdict
import pandas as pd

from stock_analysis.config import SLD_WINDOW_CONFIG
from stock_analysis.Enums.category import Category
from stock_analysis.engine import exec_all_patterns, chart_patterns_exec, candlestick_patterns_exec, harmonic_patterns_exec
import logging 
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ['AAPL']

WINDOW_SIZE = max(
    SLD_WINDOW_CONFIG["chart_patterns"]["window_size"],
    SLD_WINDOW_CONFIG["candlestick_patterns"]["window_size"],
    SLD_WINDOW_CONFIG["harmonic_patterns"]["window_size"]
)

consumer = Consumer(
    {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": "debug-consumer",
        "auto.offset.reset": "earliest",
    }
)

consumer.subscribe([os.getenv("KAFKA_TOPIC")])

sld_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

def db_connection():
    return psycopg2.connect(os.getenv("POSTGRES_URI"))

def build_dataframe(records: deque) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def get_analysis_dataframe(ticker: str) -> tuple[pd.DataFrame, str]:
    """
    Load data from kafka and check if enough data is present, if enough data is not present then 
    fetch the extra data from database
    """
    print(sld_window['AAPL'])
    window_df = build_dataframe(sld_window[ticker]) if sld_window[ticker] else pd.DataFrame()
    print(window_df)
    n_window = len(window_df)
    
    if n_window >= WINDOW_SIZE:
        return window_df.tail(WINDOW_SIZE).reset_index(drop=True), "window"
    
    needed = WINDOW_SIZE - n_window
    db_df = load_from_db(ticker, needed if n_window > 0 else WINDOW_SIZE)
    
    if not db_df.empty:
        if n_window == 0:
            return db_df.tail(WINDOW_SIZE).reset_index(drop=True), f"data only from db size: {len(db_df)}"
        
        combined = pd.concat([db_df, window_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="datetime")
        combined = combined.sort_values("datetime").reset_index(drop=True)
        combined = combined.tail(WINDOW_SIZE).reset_index(drop=True)
        return combined, f"len of data from db: {len(db_df)} + len of data from kafka: {n_window}"
    
    if n_window > 0:
        return window_df, f"limited data, size: {n_window/WINDOW_SIZE}"
    return pd.DataFrame(), "no data"

def analyse(ticker: str, df: pd.DataFrame):
    if df.empty:
        logger.info(f"[{ticker}] skipped - no data available")
        return 
    
    results = {
        "all": exec_all_patterns(df),
        "chart": chart_patterns_exec(df),
        "candlestick": candlestick_patterns_exec(df),
        "harmonic": harmonic_patterns_exec(df)
    }
    
    log_results(ticker, results)
 
def log_results(ticker: str, results: dict):
    for category, formations in results.items():
        if not formations:
            continue
 
        if category == "all":
            for sub_category, sub_formations in formations.items():
                for f in sub_formations:
                    logger.info(
                        f"[{ticker}] ALL > {sub_category.upper()} — {f['pattern']} "
                        f"(confidence={f['confidence']:.2f})"
                    )
        else:
            for f in formations:
                logger.info(
                    f"[{ticker}] {category.upper()} — {f['pattern']} "
                    f"(confidence={f['confidence']:.2f})"
                )

def load_from_db(ticker: str, limit: int) -> pd.DataFrame:
    return pd.DataFrame()

def warmup_windows():
    for symbol in SYMBOLS:
        db_df = load_from_db(symbol, WINDOW_SIZE)
        if not db_df.empty:
            for _, row in db_df.iterrows():
                sld_window[symbol].append(row.to_dict())
            logger.info(f"[{symbol}] has data loaded, size of [{symbol}]_df = {len(sld_window[symbol])}")
        else:
            logger.info(f"[{symbol}] has no data in DB")
    logger.info("Warmup complete")
    
def kafka_main():
    warmup_windows()
    logger.info("Waiting for messages...")
    try:
        while True:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            
            ticker = msg.key().decode() if msg.key() else "UNKNOWN"
            value = json.loads(msg.value().decode())
            
            sld_window[ticker].append(value)
            window_size = len(sld_window[ticker])
            logger.info(f"[{ticker}] +1 candle close={value['close']} window size = {window_size}/{WINDOW_SIZE}")
            
            df, source = get_analysis_dataframe(ticker)
            analyse(ticker, df)
    except KafkaException as kexp:
        logger.error(f"Kafka message consumption failed: {kexp}")
    except Exception as e:
        logger.error(f"Failed to consume kafka data due to: {e}")
    finally:
        consumer.close()
        logger.info("Consumer safely closed")

if __name__ == "__main__":
    kafka_main()