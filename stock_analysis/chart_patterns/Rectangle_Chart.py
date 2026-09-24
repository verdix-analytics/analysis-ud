from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import detect_pivots, safe_ratio
from stock_analysis.confidence_models.chart_patterns.rectangle_model import (
    load_model,
    learned_confidence,
    rectangle_score,
)

import pandas as pd
import numpy as np


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class RectangleChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = None

    def __init__(self):
        try:
            self._model = load_model()
        except Exception:
            self._model = None

    def _run(self, df: pd.DataFrame) -> dict:
        empty_result = {
            "confirmed": False,
            "confidence": 0,
            "pivots": [],
            "direction": "neutral",
        }

        if len(df) < 12:
            return empty_result

        pivots = detect_pivots(df, lookback=2, min_move_pct=0.005)

        if pivots is None or len(pivots) < 4:
            return empty_result

        highs_df = pivots[pivots["pivot_type"] == "H"].copy()
        lows_df = pivots[pivots["pivot_type"] == "L"].copy()

        highs = highs_df["pivot_price"].tolist()
        lows = lows_df["pivot_price"].tolist()

        if len(highs) < 2 or len(lows) < 2:
            return empty_result

        high_std = float(np.std(highs))
        high_mean = float(np.mean(highs))
        low_std = float(np.std(lows))
        low_mean = float(np.mean(lows))

        high_dev = safe_ratio(high_std, max(abs(high_mean), 1e-9))
        low_dev = safe_ratio(low_std, max(abs(low_mean), 1e-9))

        if high_dev > 0.03:
            return empty_result

        if low_dev > 0.03:
            return empty_result

        upper_bound = max(highs)
        lower_bound = min(lows)
        midline = (upper_bound + lower_bound) / 2

        box_height_pct = safe_ratio(
            upper_bound - lower_bound,
            max(abs(midline), 1e-9),
        )

        if box_height_pct < 0.003 or box_height_pct > 0.25:
            return empty_result

        high_consistency_score = max(
            0.0,
            1.0 - min(high_dev / 0.03, 1.0),
        )

        low_consistency_score = max(
            0.0,
            1.0 - min(low_dev / 0.03, 1.0),
        )

        touch_balance_ratio = safe_ratio(
            min(len(highs), len(lows)),
            max(len(highs), len(lows)),
        )

        touch_richness_score = min((len(highs) + len(lows)) / 8.0, 1.0)

        if 0.01 <= box_height_pct <= 0.10:
            range_reasonable_score = 1.0
        elif 0.006 <= box_height_pct <= 0.15:
            range_reasonable_score = 0.75
        elif 0.003 <= box_height_pct <= 0.20:
            range_reasonable_score = 0.50
        else:
            range_reasonable_score = 0.20

        high_mid = float(np.mean(highs))
        low_mid = float(np.mean(lows))

        center_drift = safe_ratio(
            abs((high_mid + low_mid) / 2 - midline),
            max(abs(midline), 1e-9),
        )

        midline_stability_score = max(
            0.0,
            1.0 - min(center_drift / 0.03, 1.0),
        )

        rule_score = rectangle_score(
            high_consistency_score=high_consistency_score,
            low_consistency_score=low_consistency_score,
            touch_balance_ratio=touch_balance_ratio,
            touch_richness_score=touch_richness_score,
            range_reasonable_score=range_reasonable_score,
            midline_stability_score=midline_stability_score,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    high_consistency_score=high_consistency_score,
                    low_consistency_score=low_consistency_score,
                    touch_balance_ratio=touch_balance_ratio,
                    touch_richness_score=touch_richness_score,
                    range_reasonable_score=range_reasonable_score,
                    midline_stability_score=midline_stability_score,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        last_close = float(df.iloc[-1]["Close"])

        confirmed = confidence >= 50 and (
            last_close > float(upper_bound)
            or last_close < float(lower_bound)
        )

        if last_close > upper_bound:
            direction = "bullish"
        elif last_close < lower_bound:
            direction = "bearish"
        else:
            direction = "neutral"

        best_pivots = []

        for j, (_, row) in enumerate(highs_df.iterrows(), start=1):
            pivot_idx = int(row["candle_index"])

            best_pivots.append({
                "type": "H",
                "price": float(row["pivot_price"]),
                "time": df.iloc[pivot_idx]["Datetime"],
                "label": f"H{j}",
            })

        for j, (_, row) in enumerate(lows_df.iterrows(), start=1):
            pivot_idx = int(row["candle_index"])

            best_pivots.append({
                "type": "L",
                "price": float(row["pivot_price"]),
                "time": df.iloc[pivot_idx]["Datetime"],
                "label": f"L{j}",
            })

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": best_pivots,
            "direction": direction,
        }
