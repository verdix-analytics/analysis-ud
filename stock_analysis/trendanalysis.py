from pydantic import BaseModel
import pandas as pd
import numpy as np
import math

class TrendAnalysis(BaseModel):
    history: dict | None = None
    
    def pre_trend_analysis(self, df: pd.DataFrame) -> dict | None:
        df = df.rename(columns=lambda c: c.strip().title())
        self.history = self.history_from_df(df)
        return self._classify_trend(df)

    def post_trend_analysis(self, df: pd.DataFrame, signal: dict | None = None) -> dict | None:
        df = df.rename(columns=lambda c: c.strip().title())
        self.history = self.history_from_df(df)
        trend = self._classify_trend(df)
        if trend is None:
            return None

        close = df["Close"].astype(float)
        start_close = float(close.iloc[0])
        end_close = float(close.iloc[-1])
        realized_move_pct = float(((end_close - start_close) / max(start_close, 1e-9)) * 100)

        trend["realized_move_pct"] = round(realized_move_pct, 4)
        trend["signal_confirmed"] = bool(signal.get("confirmed")) if signal else False
        trend["signal_confidence"] = round(float(signal.get("confidence", 0)), 2) if signal else 0.0
        return trend

    def compute_strength(self, price_change: float, slope_per_candle: float) -> float:
        def zscore(value: float, hist: list[float]) -> float:
            sig: float = 0
            if len(hist) < 5:
                sig = 1 - math.exp(-abs(value) / 30)
                return round(sig, 4)
            mean = float(np.mean(hist))
            std = float(np.std(hist))
            if std < 1e-9:
                return 0.5
            z = (abs(value)-mean) / std
            sig = 1 / (1+math.exp(-z))
            return round(sig, 4)
        
        price_hist = self.history.get("price_changes", []) if self.history else []
        slope_hist = self.history.get("slopes", []) if self.history else []
        
        price_str = zscore(price_change, price_hist)
        slope_str = zscore(slope_per_candle, slope_hist)
        
        strength = 0.5*price_str + 0.5*slope_str
        return round(strength, 4)
        
    def _classify_trend(self, df: pd.DataFrame) -> dict | None:
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = df["Close"].astype(float).reset_index(drop=True)
        high = df["High"].astype(float).reset_index(drop=True) if "High" in df.columns else close
        low = df["Low"].astype(float).reset_index(drop=True) if "Low" in df.columns else close

        if len(close) < 3:
            return {
                "label": "unknown",
                "strength": 0.0,
                "price_change_pct": 0.0,
                "slope_pct_per_candle": 0.0,
                "candles": len(close),
            }

        x = np.arange(len(close), dtype=float)
        slope, _ = np.polyfit(x, close.to_numpy(dtype=float), 1)
        baseline = max(abs(float(close.mean())), 1e-9)
        slope_pct_per_candle = float((slope / baseline) * 100)
        price_change_pct = float(((close.iloc[-1] - close.iloc[0]) / max(close.iloc[0], 1e-9)) * 100)

        high_diff = high.diff().dropna()
        low_diff = low.diff().dropna()
        higher_high_ratio = float((high_diff > 0).mean()) if not high_diff.empty else 0.0
        higher_low_ratio = float((low_diff > 0).mean()) if not low_diff.empty else 0.0
        lower_high_ratio = float((high_diff < 0).mean()) if not high_diff.empty else 0.0
        lower_low_ratio = float((low_diff < 0).mean()) if not low_diff.empty else 0.0

        directional_strength = self.compute_strength(price_change_pct, slope_pct_per_candle)

        uptrend = (
            price_change_pct >= 0.25
            and slope_pct_per_candle > 0
            and higher_high_ratio >= 0.55
            and higher_low_ratio >= 0.55
        )
        downtrend = (
            price_change_pct <= -0.25
            and slope_pct_per_candle < 0
            and lower_high_ratio >= 0.55
            and lower_low_ratio >= 0.55
        )

        if uptrend:
            label = "uptrend"
        elif downtrend:
            label = "downtrend"
        else:
            label = "sideways"

        return {
            "label": label,
            "strength": round(float(directional_strength), 4),
            "price_change_pct": round(price_change_pct, 4),
            "slope_pct_per_candle": round(slope_pct_per_candle, 4),
            "higher_high_ratio": round(higher_high_ratio, 4),
            "higher_low_ratio": round(higher_low_ratio, 4),
            "lower_high_ratio": round(lower_high_ratio, 4),
            "lower_low_ratio": round(lower_low_ratio, 4),
            "candles": int(len(close)),
        }

    def history_from_df(self, df:pd.DataFrame, window: int = 20) -> dict:
        close = df["Close"].astype(float).reset_index(drop=True)
    
        price_changes = []
        slopes = []
        
        for i in range(window, len(close)):
            chunk = close.iloc[i - window:i]
            pct = float(((chunk.iloc[-1] - chunk.iloc[0]) / max(chunk.iloc[0], 1e-9)) * 100)
            x = np.arange(len(chunk), dtype=float)
            slope, _ = np.polyfit(x, chunk.to_numpy(dtype=float), 1)
            baseline = max(abs(float(chunk.mean())), 1e-9)
            slope_pct = float((slope / baseline) * 100)
            
            price_changes.append(pct)
            slopes.append(slope_pct)
        
        return {"price_changes": price_changes, "slopes": slopes}