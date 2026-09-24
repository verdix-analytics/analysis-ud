from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.double_bottom_model import (
    load_model,
    learned_confidence,
    double_bottom_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class DoubleBottomChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = "downtrend"

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
            "direction": "bullish",
        }

        if len(df) < 7:
            return empty_result

        pivots = detect_pivots(
            df,
            lookback=2,
            min_move_pct=0.005,
        )

        if pivots is None or len(pivots) < 3:
            return empty_result

        last_pivots = get_last_n_pivots(pivots, 3)
        pivot_info = pivots_to_lists(last_pivots)

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        if types != ["L", "H", "L"]:
            return empty_result

        l1, h1, l2 = prices
        i1, i2, i3 = indices

        avg_low = (l1 + l2) / 2
        bottom_diff_pct = safe_ratio(abs(l1 - l2), avg_low)
        rebound_ratio = safe_ratio(h1 - avg_low, avg_low)

        left_span = i2 - i1
        right_span = i3 - i2
        total_span = i3 - i1

        symmetry_ratio = safe_ratio(
            min(left_span, right_span),
            max(left_span, right_span),
        )

        if bottom_diff_pct > 0.08:
            return empty_result

        if rebound_ratio < 0.005:
            return empty_result

        if left_span < 1 or right_span < 1 or total_span < 3:
            return empty_result

        rule_score = double_bottom_score(
            bottom_diff_pct=bottom_diff_pct,
            rebound_ratio=rebound_ratio,
            left_span=left_span,
            right_span=right_span,
            total_span=total_span,
            symmetry_ratio=symmetry_ratio,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    bottom_diff_pct=bottom_diff_pct,
                    rebound_ratio=rebound_ratio,
                    left_span=left_span,
                    right_span=right_span,
                    total_span=total_span,
                    symmetry_ratio=symmetry_ratio,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        confirmed = (
            confidence >= 50
            and float(df.iloc[-1]["Close"]) > float(h1)
        )

        best_pivots = [
            {
                "type": "L",
                "price": float(l1),
                "time": df.iloc[int(i1)]["Datetime"],
                "label": "L1",
            },
            {
                "type": "H",
                "price": float(h1),
                "time": df.iloc[int(i2)]["Datetime"],
                "label": "H1",
            },
            {
                "type": "L",
                "price": float(l2),
                "time": df.iloc[int(i3)]["Datetime"],
                "label": "L2",
            },
        ]

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": best_pivots,
            "direction": "bullish",
        }