import asyncio
import json
import threading
import time
import os
import websockets
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class PriceAccumulator:
    def __init__(self, api_key, symbols):
        self.api_key = api_key
        self.symbols = symbols
        self.url = f"wss://ws.finnhub.io?token={self.api_key}"
        
        self.lock = threading.Lock()
        self.active_bars = {}
        self.completed_bars = {}
        self.last_bars = {}
                
        self.is_running = False

    def _initialize_bar(self, symbol, price, ts):
        return {
            "ticker": symbol,
            "timestamp": int(ts // 60000) * 60000,
            "open": price, "high": price, "low": price, "close": price, "volume": 0
        }

    async def _socket_loop(self):
        """The actual async websocket connection logic."""
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.is_running = True
                    for s in self.symbols:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": s}))
                        await asyncio.sleep(0.1)

                    async for message in ws:
                        data = json.loads(message)
                        if data.get("type") == 'trade':
                            with self.lock: 
                                for trade in data['data']:
                                    s, p, ts, v = trade['s'], trade['p'], trade['t'], trade['v']
                                    minute = int(ts // 60000) * 60000
                                    
                                    if s not in self.active_bars:
                                        self.active_bars[s] = self._initialize_bar(s, p, ts)
                                    
                                    bar = self.active_bars[s]
                                    if minute > bar["timestamp"]:
                                        self.completed_bars[s] = bar.copy()
                                        self.active_bars[s] = self._initialize_bar(s, p, ts)
                                        bar = self.active_bars[s]

                                    bar["high"] = max(bar["high"], p)
                                    bar["low"] = min(bar["low"], p)
                                    bar["close"] = p
                                    bar["volume"] += v
                        else:
                            logger.warning(f"Non-trade message: {data}")
            except Exception as e:
                self.is_running = False
                print(f"Socket Error: {e}. Reconnecting...")
                await asyncio.sleep(5)

    def start(self):
        """Standard helper to launch the async loop in a background thread."""
        def target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._socket_loop())
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def get_latest_batch(self, symbol_list=None):
        """
        The Sync Helper. 
        """
        target = symbol_list or self.symbols
        with self.lock:
            new_bars = {}
            for s in target:
                bar = self.completed_bars.get(s)
                if bar is None:
                    continue
                if self.last_bars.get(s) != bar["timestamp"]:
                    new_bars[s] = bar.copy()
            
            for s, bar in new_bars.items():
                self.last_bars[s] = bar["timestamp"]
            
            return new_bars
    
    def format_bar(self, bar):
        """Strip internal fields and convert timestamp for consumption."""
        return {
            **{k: v for k, v in bar.items() if k != "timestamp"},
            "timestamp": datetime.fromtimestamp(bar["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    
    def batch_buffer(self, symbol_list=None, timeout=120, poll_interval = 60):
        """
        Polls every minute to the socket asking data for 
        all available stock data from the set of stocks provided 
        """
        targets = symbol_list or self.symbols
        timer = time.monotonic() + timeout
        try:
            while time.monotonic() < timer:
                with self.lock:
                    new_bars = {}
                    for s in targets:
                        bar = self.completed_bars.get(s)
                        if bar and self.last_bars.get(s) != bar["timestamp"]:
                            new_bars[s] = self.format_bar(bar)
                    if new_bars:
                        for s in new_bars:
                            self.last_bars[s] = self.completed_bars[s]["timestamp"]
                        return new_bars
                time.sleep(poll_interval)
        except TimeoutError as terr:
            logger.error(f"No data recieved within {timeout}s for: {targets}")