from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
    get_last_candle,
    body_bottom_ratio,
)

import pandas as pd


class HammerCandlestickPattern(BasePattern):
    enabled = True
    category = Category.CANDLESTICK.value

    def _run(self, df: pd.DataFrame) -> dict:
        # --- Add features ---
        df_feat = add_candlestick_features(df)
        c = get_last_candle(df_feat)

        # --- Basic metrics ---
        body = c["body"]
        range_ = c["range"]
        upper_wick = c["upper_wick"]
        lower_wick = c["lower_wick"]

        body_ratio = safe_ratio(body, range_)
        lower_body_ratio = safe_ratio(lower_wick, max(body, 1e-9))
        upper_body_ratio = safe_ratio(upper_wick, max(body, 1e-9))
        body_low_pos = body_bottom_ratio(c)

        shape_ok = (
            body_ratio <= 0.4
            and lower_body_ratio >= 2
            and upper_body_ratio <= 1
            and body_low_pos >= 0.7
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

        # 1. Long lower wick strength (max 50)
        # Scales from minimum threshold 2.2 up to 5.0 (very long wick)
        scores["lower_wick"] = min(50, (lower_body_ratio - 2.2) / (5.0 - 2.2) * 50)

        # 2. Small upper wick (max 25)
        # Best at 0, fades to 0 at threshold 0.5
        scores["upper_wick"] = max(0, min(25, (0.5 - upper_body_ratio) / 0.5 * 25))

        # 3. Small body relative to full range (max 15)
        # Best at 0, fades to 0 at threshold 0.35
        scores["body_smallness"] = max(0, min(15, (0.35 - body_ratio) / 0.35 * 15))

        # 4. Body position in upper portion of candle (max 10)
        # Scales from 0.6 (min threshold) to 1.0 (top of candle)
        scores["body_position"] = min(10, (body_low_pos - 0.6) / (1.0 - 0.6) * 10)

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
