from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
)

import pandas as pd


class EveningStarCandlestickPattern(BasePattern):
    enabled = True
    category = Category.CANDLESTICK.value

    def _run(self, df: pd.DataFrame) -> dict:
        # Need at least 3 candles
        if len(df) < 3:
            return {
                "confirmed": False,
                "structure_detected": False,
                "confidence": 0,
            }

        df_feat = add_candlestick_features(df)

        c1 = df_feat.iloc[-3]   # first candle
        c2 = df_feat.iloc[-2]   # middle small candle
        c3 = df_feat.iloc[-1]   # last bearish candle

        confidence = 0
        
        points = [c1, c2, c3]

        # 1. Basic shape conditions
        first_bullish = c1["direction"] == "bullish"
        c1_body_ratio = safe_ratio(c1["body"], c1["range"])
        c2_body_ratio = safe_ratio(c2["body"], c2["range"])
        c3_body_ratio = safe_ratio(c3["body"], c3["range"])
        middle_small = c2_body_ratio <= 0.3
        last_bearish = c3["direction"] == "bearish"

        midpoint_c1 = (c1["Open"] + c1["Close"]) / 2
        recovery_ok = c3["Close"] <= midpoint_c1
        middle_smaller_than_neighbors = (
            c2["body"] < c1["body"] and c2["body"] < c3["body"]
        )
        strong_edges = c1_body_ratio >= 0.45 and c3_body_ratio >= 0.45

        shape_ok = (
            first_bullish
            and middle_small
            and last_bearish
            and recovery_ok
            and middle_smaller_than_neighbors
            and strong_edges
        )

        if not shape_ok:
            return {
                "confirmed": False,
                "structure_detected": False,
                "confidence": 0,
            }

        pivots = []
        for point in points:
            pivots.append({
                "price": float(point["Close"]),
                "time": point["Datetime"]
            })
        # 2. Confidence calculation
        scores = {}

        # A. First candle bullish body strength (max 25)
        # Scales from 0.45 (min threshold) to 1.0 (full body)
        scores["c1_body_strength"] = min(25, (c1_body_ratio - 0.45) / (1.0 - 0.45) * 25)

        # B. Middle candle smallness (max 25)
        # Best at body_ratio=0, fades to 0 at threshold 0.3
        scores["c2_smallness"] = min(25, (0.3 - c2_body_ratio) / 0.3 * 25)

        # C. Third candle bearish body strength (max 25)
        scores["c3_body_strength"] = min(25, (c3_body_ratio - 0.45) / (1.0 - 0.45) * 25)

        # D. Reversal depth into first candle body (max 25)
        c1_body = abs(c1["Close"] - c1["Open"])
        reversal_depth = c1["Close"] - c3["Close"]
        reversal_ratio = safe_ratio(reversal_depth, c1_body)
        scores["reversal_depth"] = min(25, max(0, (reversal_ratio - 0.5) / (1.0 - 0.5) * 25))

        confidence = sum(scores.values())

        structure_detected = True
        confirmed = structure_detected and confidence > 50

        return {
            "confirmed": bool(confirmed),
            "structure_detected": structure_detected,
            "confidence": round(float(confidence), 2),
            "direction": "bearish",
            "pivots": pivots
        }