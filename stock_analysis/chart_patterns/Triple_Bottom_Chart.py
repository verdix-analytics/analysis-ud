from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.triple_bottom_model import (
    load_model,
    learned_confidence,
    triple_bottom_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class TripleBottomChartPattern(BasePattern):
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

        if len(df) < 11:
            return empty_result

        pivots = detect_pivots(df, lookback=2, min_move_pct=0.005)

        if pivots is None or len(pivots) < 5:
            return empty_result

        pivot_info = pivots_to_lists(get_last_n_pivots(pivots, 5))

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        if types != ["L", "H", "L", "H", "L"]:
            return empty_result

        l1, h1, l2, h2, l3 = prices
        i1, i2, i3, i4, i5 = indices

        lows = [l1, l2, l3]
        peaks = [h1, h2]

        avg_low = sum(lows) / 3

        bottom_spread_pct = safe_ratio(
            max(lows) - min(lows),
            max(abs(avg_low), 1e-9),
        )

        rebound_scores = [
            safe_ratio(peak - avg_low, max(abs(avg_low), 1e-9))
            for peak in peaks
        ]

        rebound_height = min(rebound_scores)

        span_1 = i2 - i1
        span_2 = i3 - i2
        span_3 = i4 - i3
        span_4 = i5 - i4
        total_span = i5 - i1

        spans_ok = (
            span_1 >= 1
            and span_2 >= 1
            and span_3 >= 1
            and span_4 >= 1
        )

        if not spans_ok:
            return empty_result

        if bottom_spread_pct > 0.08:
            return empty_result

        if rebound_height < 0.005:
            return empty_result

        if total_span < 4:
            return empty_result

        if total_span >= 10:
            span_score = 1.0
        elif total_span >= 8:
            span_score = 0.8
        elif total_span >= 6:
            span_score = 0.6
        elif total_span >= 4:
            span_score = 0.4
        else:
            span_score = 0.0

        left_block = i3 - i1
        right_block = i5 - i3

        symmetry_ratio = safe_ratio(
            min(left_block, right_block),
            max(left_block, right_block),
        )

        peak_balance_ratio = safe_ratio(
            min(h1, h2),
            max(h1, h2),
        )

        rule_score = triple_bottom_score(
            bottom_spread_pct=bottom_spread_pct,
            rebound_height=rebound_height,
            span_score=span_score,
            symmetry_ratio=symmetry_ratio,
            peak_balance_ratio=peak_balance_ratio,
            total_span=total_span,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    bottom_spread_pct=bottom_spread_pct,
                    rebound_height=rebound_height,
                    span_score=span_score,
                    symmetry_ratio=symmetry_ratio,
                    peak_balance_ratio=peak_balance_ratio,
                    total_span=total_span,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        base_resistance = max(h1, h2)

        confirmed = (
            confidence >= 50
            and float(df.iloc[-1]["Close"]) > float(base_resistance)
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
            {
                "type": "H",
                "price": float(h2),
                "time": df.iloc[int(i4)]["Datetime"],
                "label": "H2",
            },
            {
                "type": "L",
                "price": float(l3),
                "time": df.iloc[int(i5)]["Datetime"],
                "label": "L3",
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