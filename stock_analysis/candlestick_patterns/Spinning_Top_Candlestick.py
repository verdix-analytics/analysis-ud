from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
    get_last_candle,
)

import pandas as pd


class SpinningTopCandlestickPattern(BasePattern):
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
        wick_presence_ratio = safe_ratio(upper_wick + lower_wick, range_)
        wick_balance_ratio = safe_ratio(
            min(upper_wick, lower_wick),
            max(upper_wick, lower_wick),
        )

        shape_ok = (
            range_ > 0
            and body_ratio >= 0.05
            and body_ratio <= 0.3
            and upper_wick > 0
            and lower_wick > 0
            and wick_presence_ratio >= 0.55
            and wick_balance_ratio >= 0.35
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
        
        scores["body_smallness"] = max(0, min(30, (0.3 - body_ratio) / (0.3 - 0.05) * 30))
        scores["wick_balance"] = max(0, min(35, (wick_balance_ratio - 0.35) / (1-0.35) * 35))
        scores["wick_presence"] = max(0, min(25, (wick_presence_ratio - 0.55) / (0.95 - 0.55) * 25))
        
        confidence = sum(scores.values())
        structure_detected = True
        confirmed = structure_detected and confidence > 50

        return {
            "confirmed": bool(confirmed),
            "structure_detected": structure_detected,
            "confidence": round(float(confidence), 2),
            "direction": "neutral",
            "pivots": [pivot]
        }