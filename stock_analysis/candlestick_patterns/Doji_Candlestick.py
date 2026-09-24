from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
    get_last_candle,
)

import pandas as pd


class DojiCandlestickPattern(BasePattern):
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
        wick_presence_ratio = safe_ratio(upper_wick + lower_wick, range_)

        shape_ok = (
            range_ > 0
            and body_ratio <= 0.1
            and upper_wick > 0
            and lower_wick > 0
            and wick_presence_ratio >= 0.75
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

        # 1. Body smallness — the closer to 0 the better (max 60)
        # Scales from body_ratio=0.1 (score=0) down to 0 (score=60)
        scores["body_smallness"] = max(0, min(60, (0.1 - body_ratio) / 0.1 * 60))

        # 2. Wick presence (max 20)
        # Scales from wick_presence_ratio=0.75 (score=0) up to 1.0 (score=20)
        scores["wick_presence"] = min(20, (wick_presence_ratio - 0.75) / (1.0 - 0.75) * 20)

        # 3. Wick balance — reward symmetry between upper and lower wicks (max 20)
        wick_diff_ratio = safe_ratio(abs(upper_wick - lower_wick), range_)
        scores["wick_balance"] = max(0, min(20, (0.2 - wick_diff_ratio) / 0.3 * 20))

        confidence = sum(scores.values())
        
        # 🎯 Final Decision
        structure_detected = True
        confirmed = structure_detected and confidence > 50

        return {
            "confirmed": bool(confirmed),
            "structure_detected": structure_detected,
            "confidence": round(float(confidence), 2),
            "direction": "neutral",
            "pivots": [pivot]
        }
