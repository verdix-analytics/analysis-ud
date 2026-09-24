from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
)

import pandas as pd


class BullishEngulfingCandlestickPattern(BasePattern):
    enabled = True
    category = Category.CANDLESTICK.value

    def _run(self, df: pd.DataFrame) -> dict:
        if len(df) < 2:
            return {
                "confirmed": False,
                "structure_detected": False,
                "confidence": 0,
            }

        df_feat = add_candlestick_features(df)

        prev = df_feat.iloc[-2]
        curr = df_feat.iloc[-1]

        confidence = 0

        # Strict bullish engulfing body condition
        shape_ok = (
            prev["direction"] == "bearish"
            and curr["direction"] == "bullish"
            and curr["Open"] < prev["Close"]
            and curr["Close"] > prev["Open"]
        )

        if not shape_ok:
            return {
                "confirmed": False,
                "structure_detected": False,
                "confidence": 0,
            }

        pivot = {
            "price": float(curr["Close"]),
            "time": curr["Datetime"]
        }
        
        prev_body = prev["body"]
        curr_body = curr["body"]

        prev_range = prev["range"]
        curr_range = curr["range"]

        prev_body_ratio = safe_ratio(prev_body, prev_range)
        curr_body_ratio = safe_ratio(curr_body, curr_range)

        body_ratio = safe_ratio(curr_body, prev_body)
        range_ratio = safe_ratio(curr_range, prev_range)

        if prev_body_ratio < 0.25 or curr_body_ratio < 0.45 or body_ratio < 1.15:
            return {
                "confirmed": False,
                "structure_detected": True,
                "confidence": 0,
            }

        scores = {}

        # 1. Body engulf strength (max 30)
        scores["body_engulf"] = min(30, (body_ratio - 1.15) / (2.5 - 1.15) * 30)

        # 2. Current body quality (max 20)
        scores["body_quality"] = min(20, curr_body_ratio * 20)

        # 3. Lower shadow penalty — reward small lower wick on bullish candle (max 15)
        curr_lower_shadow = curr["Open"] - curr["Low"]
        lower_shadow_ratio = safe_ratio(curr_lower_shadow, curr_range)
        scores["lower_shadow"] = max(0, (0.15 - lower_shadow_ratio) / 0.15 * 15)

        # 4. Prior downtrend strength (max 20)
        lookback = df.iloc[:-2]
        trend_move = safe_ratio(
            lookback.iloc[0]["Open"] - prev["Close"],
            lookback.iloc[0]["Open"]
        )
        scores["prior_trend"] = min(20, max(0, trend_move * 200))

        # 5. Volume confirmation (max 15)
        if "Volume" in df.columns:
            avg_vol = df.iloc[:-1]["Volume"].mean()
            vol_ratio = safe_ratio(curr["Volume"], avg_vol)
            scores["volume"] = min(15, max(0, (vol_ratio - 1) * 15))
        else:
            scores["volume"] = scores["body_quality"] / 20 * 15

        confidence = sum(scores.values())

        structure_detected = True
        confirmed = structure_detected and confidence > 50

        return {
            "confirmed": bool(confirmed),
            "structure_detected": structure_detected,
            "confidence": round(float(confidence), 2),
            "direction": "bullish",
            "pivots": [pivot]
        }
