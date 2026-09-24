from financial_websocket.PriceAccumulator import PriceAccumulator
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
import os

load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")

symbols = ["AAPL", "MSFT", "BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]
acc = PriceAccumulator(API_KEY, symbols)
acc.start()

print("Waiting for first trades...")
time.sleep(5)

while True:
    try:
        batch = acc.batch_buffer(symbols, timeout=120)
        print(f"\n--- {datetime.now(timezone.utc).isoformat()} ---")
        for symbol, bar in batch.items():
            print(f"  {symbol}: timestamp:{bar['timestamp']} O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']}")
    except TimeoutError as e:
        print(f"Warning: {e}")