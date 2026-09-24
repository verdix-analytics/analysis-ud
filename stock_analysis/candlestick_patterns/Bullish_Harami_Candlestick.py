from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
)

import pandas as pd


class BullishHaramiCandlestickPattern(BasePattern):
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

        prev_body_high = max(prev["Open"], prev["Close"])
        prev_body_low = min(prev["Open"], prev["Close"])
        curr_body_high = max(curr["Open"], curr["Close"])
        curr_body_low = min(curr["Open"], curr["Close"])

        prev_body_ratio = safe_ratio(prev["body"], prev["range"])
        curr_body_ratio = safe_ratio(curr["body"], curr["range"])
        size_ratio = safe_ratio(curr["body"], max(prev["body"], 1e-9))

        shape_ok = (
            prev["direction"] == "bearish"
            and curr["direction"] == "bullish"
            and prev_body_ratio >= 0.5
            and curr_body_ratio >= 0.15
            and curr_body_high < prev_body_high
            and curr_body_low > prev_body_low
            and size_ratio <= 0.65
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
        
        scores = {}

        # 1. Previous candle bearish body strength (max 35)
        scores["prev_body_strength"] = min(35, (prev_body_ratio - 0.5) / (1.0 - 0.5) * 35)

        # 2. Current candle size relative to previous (max 35)
        # Lower size_ratio = better containment; scale from 0.65 down to 0.1
        scores["size_containment"] = min(35, (0.65 - size_ratio) / (0.65 - 0.1) * 35)

        # 3. Current body centered inside previous body (max 20)
        body_midpoint = (prev_body_high + prev_body_low) / 2
        centeredness = safe_ratio(
            min(curr_body_high - body_midpoint, body_midpoint - curr_body_low),
            max(prev["body"] / 2, 1e-9),
        )

        scores["centeredness"] = min(20, centeredness / 0.5 * 20)

        # 4. Current body compactness (max 10)
        # Reward small curr_body_ratio; best at <= 0.15, fades to 0 at 0.45
        scores["body_compactness"] = max(0, min(10, (0.45 - curr_body_ratio) / (0.45 - 0.15) * 10))

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