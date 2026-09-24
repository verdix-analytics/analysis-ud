from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
    get_last_candle,
    body_top_ratio,
)

import pandas as pd


class InvertedHammerCandlestickPattern(BasePattern):
    enabled = True
    category = Category.CANDLESTICK.value

    def _run(self, df: pd.DataFrame) -> dict:
        df_feat = add_candlestick_features(df)
        c = get_last_candle(df_feat)

        body = c["body"]
        range_ = c["range"]
        upper_wick = c["upper_wick"]
        lower_wick = c["lower_wick"]

        body_ratio = safe_ratio(body, range_)
        upper_body_ratio = safe_ratio(upper_wick, max(body, 1e-9))
        lower_body_ratio = safe_ratio(lower_wick, max(body, 1e-9))
        body_top_pos = body_top_ratio(c)

        shape_ok = (
            body_ratio <= 0.35
            and upper_body_ratio >= 2.2
            and lower_body_ratio <= 0.4
            and body_top_pos <= 0.45
        )

        if not shape_ok:
            return {
                "confirmed": False,
                "structure_detected": False,
                "confidence": 0,
            }

        pivot = {
            "price": float(c["Close"]),
            "time": c["Datetime"]
        }
        
        scores = {}

        # --- 1. Long upper wick (max 45) ---
        scores["upper_wick"] = max(0, min(45, (upper_body_ratio - 2.2)/(5 - 2.2) * 45))

        # --- 2. Very small lower wick (max 20) ---
        scores["lower_wick"] = max(0, min(20, (0.4 - lower_body_ratio)/0.4 * 20))

        # --- 3. Small body relative to full range (max 20) ---
        scores["body_smallness"] = max(0, min(20, (0.35 - body_ratio)/0.35 * 20))

        # --- 4. Body should sit near the lower end of the candle (max 15) ---
        scores["body_position"] = max(0, min(15, (0.45 - body_top_pos) / 0.45 * 15))

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
