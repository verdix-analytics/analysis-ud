from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
)

import pandas as pd


class MorningStarCandlestickPattern(BasePattern):
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
        c3 = df_feat.iloc[-1]   # last bullish candle

        points = [c1, c2, c3]
        
        c1_body_ratio = safe_ratio(c1["body"], c1["range"])
        c2_body_ratio = safe_ratio(c2["body"], c2["range"])
        c3_body_ratio = safe_ratio(c3["body"], c3["range"])

        first_bearish = c1["direction"] == "bearish"
        last_bullish = c3["direction"] == "bullish"

        middle_small = c2_body_ratio <= 0.25
        middle_smaller_than_neighbors = (
            c2["body"] < c1["body"] and c2["body"] < c3["body"]
        )

        strong_edges = c1_body_ratio >= 0.45 and c3_body_ratio >= 0.45

        # third candle closes back into first candle's body
        midpoint_c1 = (c1["Open"] + c1["Close"]) / 2
        recovery_ok = c3["Close"] >= midpoint_c1

        # middle candle should sit in the lower zone relative to c1 close / c3 open
        middle_anchor_ok = (
            max(c2["Open"], c2["Close"]) <= max(c1["Open"], c1["Close"])
            and min(c2["Open"], c2["Close"]) <= max(c1["Open"], c1["Close"])
        )

        shape_ok = (
            first_bearish
            and middle_small
            and last_bullish
            and middle_smaller_than_neighbors
            and strong_edges
            and recovery_ok
            and middle_anchor_ok
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
            
        scores = {}
        
        # A. First candle bearish body strength (max 25)
        scores["c1_strength"] = max(0, min(25, (c1_body_ratio - 0.45) / (0.85 - 0.45) * 25))

        # B. Middle candle smallness (max 25)
        scores["c2_smallness"] = max(0, min(25, (0.25 - c2_body_ratio) / 0.25 * 25))


        # C. Third candle bullish body strength (max 25)
        scores["c3_strength"] = max(0, min(25, (c3_body_ratio - 0.45) / (0.85 - 0.45) * 25))


        # D. Recovery strength into first candle body (max 25)
        c1_open = c1["Open"]
        c1_close = c1["Close"]
        c1_body = abs(c1_open - c1_close)

        # for bearish c1, recovery measured upward from c1 close
        recovery_depth = c3["Close"] - c1_close
        recovery_ratio = safe_ratio(recovery_depth, c1_body)
        scores["recovery"] = max(0, min(15, (recovery_ratio - 0.5) / (1.0 - 0.5) * 15))
        
        gap_depth = c1["Close"] - max(c2["Open"], c2["Close"])
        gap_ratio = safe_ratio(gap_depth, c1["body"])
        scores["gap_down"] = max(0, min(10, gap_ratio / 0.3 * 10))

        confidence = sum(scores.values())
        structure_detected = True
        confirmed = structure_detected and confidence > 50

        return {
            "confirmed": bool(confirmed),
            "structure_detected": structure_detected,
            "confidence": round(float(confidence), 2),
            "direction": "bullish",
            "pivots": pivots
        }
